#!/usr/bin/env python3
"""
mppi_mould.py — MPPI in the PointWorld world model for the deformable
mould (moulding) task.

Mirrors pwm_efficient/mppi_in_world_model.py's structure:
  per replan, sample K i.i.d. candidate paddle action sequences,
  forward each through the PointWorld model in chunks, score by chamfer
  to a letter-template field on the anchor frame, MPPI-softmax-weight the
  K candidates, commit K_la frames of the weighted action sequence, advance
  the imagined cloth via the WM's prediction for the chosen U, repeat.

Action space (per model-frame): [dx, dy, dz, dyaw] of the paddle pose.
Integrates kinematically (paddle_pose[t+1] = paddle_pose[t] + a[t]); the
paddle particles are reconstructed each frame from `paddle_template + pose`.

Model is loaded in-process (no HTTP) — same path as rollout_mould_viz.py.
Cost target field uses `letter_template.fit_letter_field` directly (vendored
via sys.path from pwm_efficient — letter_template.py + stats/letter_bank_*.npz
have no heavy deps).

Output (under inference_results/<name>/):
  positions/batch{i:03d}.npz   matches pwm_efficient's mppi_in_world_model.py
                               schema so the same replay/scoring works
  field/batch{i:03d}.npz       coverage + contour for the letter target
  metrics.json                 aggregate stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# ─── sys.path: PointWorld + pwm_efficient (letter_template) ──────────────────
# IMPORTANT: PointWorld must be searched BEFORE pwm_efficient. Both have
# top-level `utils` modules; PointWorld's `pointworld/losses.py` does
# `from utils import safe_loss_computation` which only exists in PointWorld's.
# pwm_efficient's `letter_template` doesn't clash with anything in PointWorld,
# so leaving pwm_efficient second is safe for the imports below.
_PW_ROOT  = Path("/data/user_data/mbronars/packages/PointWorld")
_PWMEFF   = Path("/data/user_data/mbronars/packages/pwm_efficient")
sys.path.insert(0, str(_PWMEFF))
sys.path.insert(0, str(_PW_ROOT))

# Pull PointWorld bits through the same path rollout_mould_viz.py uses.
_orig_argv = sys.argv
sys.argv   = [sys.argv[0]]
from arguments import parse_args as pw_parse_args                # noqa: E402
sys.argv = _orig_argv

from pointworld.base import BaseModel                            # noqa: E402
from pointworld.checkpoint_contract import (                     # noqa: E402
    apply_model_contract_to_args,
    read_checkpoint_contract,
)
from dataset_components.transforms import center_shift           # noqa: E402
from dataset_components.robot import gather_features             # noqa: E402

# Letter target (vendored; pure numpy/torch + scipy).
import letter_template as lt                                     # noqa: E402


DOMAIN      = "mould"
CONTEXT_LEN = 1
PRED_FRAMES = 10
MODEL_T     = CONTEXT_LEN + PRED_FRAMES   # 11


# ─────────────────────────────────────────────────────────────────────────────
# CLI / config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MPPIConfig:
    # Search
    population_size:        int   = 128
    chunk_size:             int   = 16        # K-chunks for the model forward
    temperature:            float = 0.3
    lookahead_commit_size:  int   = 5         # K_la frames committed per replan
    total_frames:           int   = 150       # plan length (model frames)
    horizon_mode:           str   = "window"  # window: score the end of one 10-frame window (original); final: chain windows, score the last plan frame (2026-09-26)

    # Action sampling (per model-frame Δ on paddle pose). Defaults below are
    # the EMPIRICAL training-data distribution from
    #   mould_new_6k_data_pointworld/train (1k windows, 10k deltas):
    #     Δx, Δy std ≈ 0.0041, hard-capped at ±0.008
    #     Δz     std ≈ 0.0037, hard-capped at ±0.008
    #     Δyaw   std ≈ 0.105,  hard-capped at ±0.25
    # (computed via Kabsch yaw fit + centroid xyz on the paddle 256-point cloud
    # from the saved PointWorld training shards — what the model actually saw.)
    std_xyz:                float = 0.004
    std_yaw:                float = 0.10
    max_xyz:                float = 0.008
    max_yaw:                float = 0.25

    # Objective: 'letter' (default, fit a target letter field) or 'centroid'
    # (move blob centroid xy to anchor + offset, cloud_centroid task).
    objective:              str   = "letter"

    # Letter target (used iff objective == 'letter')
    letter:                 str   = "C"
    letter_bank_path:       str   = str(_PWMEFF / "stats/letter_bank_highres.npz")
    letter_fit_rot_range:   float = 40.0
    letter_coverage_points: int   = 150
    letter_coverage_weight: float = 1.0
    letter_patch_frac:      float = 0.2
    letter_density_scale:   float = 0.75
    letter_random_rotation: bool  = True
    letter_coverage_frac:   float = 0.75
    letter_chosen_rot_deg:  Optional[float] = None

    # Centroid target (used iff objective == 'centroid'). Anchor is the
    # blob's frame-0 xy centroid; target_xy = anchor_centroid_xy + offset.
    centroid_offset_x:      float = 0.0
    centroid_offset_y:      float = 0.0
    # If set, clip predicted blob xy positions to ±clamp_xy before computing
    # the centroid cost. Prevents WM-hallucinated outliers from gaming the
    # cost (they get pinned to the clamp limit). None = no clipping.
    centroid_clamp_xy:      Optional[float] = None

    # Misc
    seed:                   int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# Demo loader (one full mould trajectory)
# ─────────────────────────────────────────────────────────────────────────────

def load_mould_demo(test_dir: str, demo_idx: int,
                    skip_frames: int = 0, stride: int = 2) -> dict:
    """Return blob + paddle particle trajectories (stride-subsampled) from
    one test-shard entry. obj_ids: 0 = blob, 1 = paddle."""
    import glob, pickle, tarfile
    tar_paths = sorted(glob.glob(f"{test_dir}/*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"no tars in {test_dir}")
    seen = 0
    for tp in tar_paths:
        with tarfile.open(tp) as tar:
            groups: dict[str, list] = {}
            for m in tar.getmembers():
                stem, _, ext = m.name.partition(".")
                groups.setdefault(stem, []).append((ext, m))
            for stem in sorted(groups.keys()):
                if seen == demo_idx:
                    data = {ext: pickle.loads(tar.extractfile(m).read())
                            for ext, m in groups[stem]}
                    pp  = data["particle_pos.pyd"][0]        # (T, N, 3)
                    vp  = data["valid_particles.pyd"][0]     # (T, N)
                    oid = data["obj_ids.pyd"][0]             # (N,)
                    break
                seen += 1
            else:
                continue
            break
    else:
        raise IndexError(f"demo_idx={demo_idx} out of range ({seen} demos)")

    pp = pp[skip_frames::stride]
    vp = vp[skip_frames::stride]
    blob_mask   = (oid == 0)
    paddle_mask = (oid == 1)
    return dict(
        particle_pos    = pp,
        valid_particles = vp,
        obj_ids         = oid,
        blob_mask       = blob_mask,
        paddle_mask     = paddle_mask,
        blob_traj       = pp[:, blob_mask],          # (T, N_blob, 3)
        paddle_traj     = pp[:, paddle_mask],        # (T, N_paddle, 3)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Paddle template + per-pose reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def make_paddle_template(paddle_world_t0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (template_local, pose0). The paddle is treated as a rigid body
    centered at its world centroid; the template is the paddle's local-frame
    point cloud at frame 0 (with yaw=0 reference orientation)."""
    centroid = paddle_world_t0.mean(axis=0).astype(np.float32)        # (3,)
    template = (paddle_world_t0 - centroid).astype(np.float32)        # (N_paddle, 3)
    pose0    = np.array([centroid[0], centroid[1], centroid[2], 0.0], dtype=np.float32)
    return template, pose0


def paddle_particles_from_pose(template: np.ndarray,
                                pose:    np.ndarray) -> np.ndarray:
    """pose = (x, y, z, yaw). Returns (N_paddle, 3) = R_yaw @ template + xyz."""
    yaw = float(pose[3])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return (template @ R.T + pose[:3][None, :]).astype(np.float32)


def paddle_particles_from_poses_K(template: np.ndarray,
                                   poses_K: np.ndarray) -> np.ndarray:
    """poses_K: (K, T, 4) [x, y, z, yaw] → (K, T, N_paddle, 3)."""
    K, T, _ = poses_K.shape
    yaw = poses_K[..., 3]                                              # (K, T)
    c, s = np.cos(yaw), np.sin(yaw)
    # Rotated template per (K, T): R[k, t] @ template
    # Vectorise via einsum on (K, T, 3, 3) × (N, 3).
    R = np.zeros((K, T, 3, 3), dtype=np.float32)
    R[..., 0, 0] =  c; R[..., 0, 1] = -s
    R[..., 1, 0] =  s; R[..., 1, 1] =  c
    R[..., 2, 2] = 1.0
    # (K, T, N, 3) = einsum("ktij,nj->ktni", R, template)
    rotated = np.einsum("ktij,nj->ktni", R, template)
    return (rotated + poses_K[..., None, :3]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-window batch construction
# ─────────────────────────────────────────────────────────────────────────────

def build_window_batch_K(blob_ctx: np.ndarray,            # (N_blob, 3) world frame
                         paddle_K_window: np.ndarray,      # (K, MODEL_T, N_paddle, 3)
                         joint_K_window:  np.ndarray,      # (K, MODEL_T, 1)
                         robot_features:  List[str],
                         scene_features:  List[str],
                         device: str) -> tuple[dict, np.ndarray]:
    """Build one batched model-input dict for K candidates of one window.

    Runs the repo's center_shift + gather_features per-K (cheap numpy ops),
    then stacks. Returns (batch_dict, shift_K) where shift_K is (K, 3).
    """
    K     = paddle_K_window.shape[0]
    N_b   = blob_ctx.shape[-2]

    samples_out = []
    shifts      = []
    for k in range(K):
        robot_win   = paddle_K_window[k].astype(np.float32)            # (T, N_p, 3)
        ctx_k = blob_ctx[k] if blob_ctx.ndim == 3 else blob_ctx          # (K, N_b, 3): per-candidate context (horizon_mode final)
        scene_flows = np.tile(ctx_k[None].astype(np.float32),
                              (MODEL_T, 1, 1))                          # (T, N_b, 3)
        sample = {
            "scene_flows":         scene_flows,
            "robot_flows":         robot_win,
            "robot_normals":       np.zeros_like(robot_win),
            "right_gripper_open":  np.zeros((MODEL_T, 1), dtype=np.float32),
            "left_gripper_open":   np.zeros((MODEL_T, 1), dtype=np.float32),
            # joint_positions: optional, only used if listed in robot_features.
            "joint_positions":     joint_K_window[k].astype(np.float32),
        }
        sample = center_shift(sample)
        shifts.append(-np.asarray(sample["__shift_amount__"], dtype=np.float32))
        sample = gather_features(
            sample,
            robot_features=robot_features,
            scene_features=scene_features,
            random_context_mode="fixed",
            context_horizon=CONTEXT_LEN,
            has_bimanual_robot=False,
            domain=DOMAIN,
        )
        samples_out.append(sample)

    def stack_to_torch(key: str) -> torch.Tensor:
        arr = np.stack([s[key] for s in samples_out], axis=0)
        return torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)

    batch = {
        "scene_flows":    stack_to_torch("scene_flows"),
        "scene_features": stack_to_torch("scene_features"),
        "scene_exists":   torch.ones(K, MODEL_T, N_b, dtype=torch.bool, device=device),
        "robot_flows":    stack_to_torch("robot_flows"),
        "robot_features": stack_to_torch("robot_features"),
        "robot_exists":   torch.ones(K, MODEL_T, paddle_K_window.shape[2],
                                     dtype=torch.bool, device=device),
        "__domain__":     [DOMAIN] * K,
    }
    return batch, np.stack(shifts, axis=0).astype(np.float32)


@torch.no_grad()
def model_forward_K(model, blob_ctx: np.ndarray, paddle_K_window: np.ndarray,
                    joint_K_window: np.ndarray,
                    robot_features: List[str], scene_features: List[str],
                    chunk_size: int, device: str) -> np.ndarray:
    """K candidates → (K, MODEL_T, N_blob, 3) predicted blob in world frame.

    Splits K into `chunk_size`-sized batches so peak GPU memory stays bounded.
    """
    K = paddle_K_window.shape[0]
    out_chunks = []
    for cs in range(0, K, chunk_size):
        ce = min(cs + chunk_size, K)
        sub_paddle = paddle_K_window[cs:ce]
        sub_joint  = joint_K_window[cs:ce]
        batch, shifts = build_window_batch_K(
            blob_ctx[cs:ce] if blob_ctx.ndim == 3 else blob_ctx, sub_paddle, sub_joint,
            robot_features, scene_features, device,
        )
        out = model(batch, training=False)
        pred = out["scene_flows"].float().cpu().numpy()                # (chunk, T, N_blob, 3)
        pred_world = pred + shifts[:, None, None, :]
        out_chunks.append(pred_world)
    return np.concatenate(out_chunks, axis=0).astype(np.float32)        # (K, T, N_blob, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Letter field + chamfer
# ─────────────────────────────────────────────────────────────────────────────

def build_letter_field_from_blob(blob_xy_t0: np.ndarray,
                                  cfg: MPPIConfig,
                                  device: torch.device,
                                  rng: np.random.Generator) -> lt.LetterField:
    """Fit a letter target field to the blob's anchor-frame XY footprint.

    Equivalent to pwm_efficient/guidance_objectives.build_letter_field but
    calls lt directly so we don't need pwm_efficient's TrajectoryBatch."""
    anchor_xy = torch.as_tensor(blob_xy_t0[None], dtype=torch.float32, device=device)   # (1, N, 2)
    block_mask = torch.ones((1, anchor_xy.shape[1]), dtype=torch.bool, device=device)
    anchor_z   = torch.zeros(1, dtype=torch.float32, device=device)                     # not used here

    images, letters = lt.load_letter_bank(cfg.letter_bank_path)
    variants = lt.get_letter_variants(images, letters, cfg.letter)

    if cfg.letter_coverage_frac > 0.0:
        n_block = int(blob_xy_t0.shape[0])
        coverage_points = max(1, int(round(cfg.letter_coverage_frac * n_block)))
    else:
        coverage_points = cfg.letter_coverage_points

    if cfg.letter_chosen_rot_deg is not None:
        variants = lt.pre_rotate_variants(variants, float(cfg.letter_chosen_rot_deg))
        params = lt.FitParams(
            rot_range_deg=0.0, n_rots=1, coverage_points=coverage_points,
            patch_frac=cfg.letter_patch_frac,
            letter_density_scale=cfg.letter_density_scale,
        )
    elif cfg.letter_random_rotation:
        # Sample from np.random's GLOBAL state (matching
        # pwm_efficient/guidance_objectives.build_letter_field). The caller
        # seeds the global state per-demo so that two runs (this planner +
        # pwm_efficient's mppi_in_world_model.py) with the same seed produce
        # identical letter rotations, and therefore identical target fields.
        rot_deg = float(np.random.uniform(-cfg.letter_fit_rot_range, cfg.letter_fit_rot_range))
        variants = lt.pre_rotate_variants(variants, rot_deg)
        params = lt.FitParams(
            rot_range_deg=0.0, n_rots=1, coverage_points=coverage_points,
            patch_frac=cfg.letter_patch_frac,
            letter_density_scale=cfg.letter_density_scale,
        )
    else:
        params = lt.FitParams(
            rot_range_deg=cfg.letter_fit_rot_range,
            coverage_points=coverage_points,
            patch_frac=cfg.letter_patch_frac,
            letter_density_scale=cfg.letter_density_scale,
        )
    return lt.fit_letter_field(anchor_xy, block_mask, anchor_z, variants, params)


def chamfer_batched_np(block_K: np.ndarray, cov_xy: np.ndarray,
                       coverage_weight: float = 1.0) -> np.ndarray:
    """(K, N_blob, 2) × (n_cov, 2) → (K,) symmetric chamfer.

    Same definition as letter_template_loss: mean of cloud→template min
    distances + coverage_weight × template→cloud min distances.
    """
    # (K, N, 1, 2) - (1, 1, n_cov, 2) → (K, N, n_cov, 2)
    diff = block_K[:, :, None, :] - cov_xy[None, None, :, :]
    d2 = (diff ** 2).sum(axis=-1)                                       # (K, N, n_cov)
    cloud_to_tmpl = d2.min(axis=2).mean(axis=1)
    tmpl_to_cloud = d2.min(axis=1).mean(axis=1)
    return (cloud_to_tmpl + coverage_weight * tmpl_to_cloud).astype(np.float32)


def centroid_dist_batched_np(block_K: np.ndarray, target_xy: np.ndarray,
                              clamp_xy: float | None = None) -> np.ndarray:
    """(K, N_blob, 2) → (K,) per-candidate || mean_xy - target_xy ||.

    Used as the cost for objective='centroid'. block_K should already be the
    final-frame blob xy (use ``pred_blob_K[:, H_active, :, :2]``).

    If ``clamp_xy`` is set, particle xy positions are clipped to
    [-clamp_xy, +clamp_xy] (per axis) before the mean is taken. This
    prevents the cost from being gamed by WM-hallucinated particles that
    have flown far outside the physical scene — those get pinned to the
    clamp limit, capping their pull on the centroid.
    """
    if clamp_xy is not None:
        block_K = np.clip(block_K, -clamp_xy, clamp_xy)
    cent = block_K.mean(axis=1)                                   # (K, 2)
    return np.linalg.norm(cent - target_xy[None, :], axis=-1).astype(np.float32)


def mppi_softmax_weights(costs: np.ndarray, temperature: float) -> np.ndarray:
    """Scale-invariant softmax: normalise by cost-std then softmax with `T`."""
    adv = costs - costs.min()
    s = float(adv.std())
    adv = adv / (s + 1e-9)
    w = np.exp(-adv / max(temperature, 1e-6))
    z = float(w.sum())
    if z <= 0 or not np.isfinite(z):
        return np.full_like(costs, 1.0 / costs.shape[0])
    return (w / z).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MPPI rollout
# ─────────────────────────────────────────────────────────────────────────────

def mppi_rollout(model, demo: dict, robot_features: List[str],
                 scene_features: List[str], cfg: MPPIConfig,
                 device: str) -> dict:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    blob_traj   = demo["blob_traj"]                        # (T_demo, N_blob, 3)
    paddle_traj = demo["paddle_traj"]                      # (T_demo, N_paddle, 3)
    T_demo      = blob_traj.shape[0]
    N_blob      = blob_traj.shape[1]
    N_paddle    = paddle_traj.shape[1]

    # Anchor state at frame 0.
    blob_world   = blob_traj[0].astype(np.float32).copy()  # (N_blob, 3)
    paddle_tpl, paddle_pose = make_paddle_template(paddle_traj[0])

    # ── Target: either letter coverage field or a centroid target xy ─────────
    if cfg.objective == "letter":
        letter_field = build_letter_field_from_blob(
            blob_world[:, :2], cfg, torch.device(device), rng,
        )
        cov_xy = letter_field.coverage[0].detach().cpu().numpy().astype(np.float32)  # (n_cov, 2)
        contour_world = letter_field.contour[0].detach().cpu().numpy().astype(np.float32)  # (P, 3)
        target_xy = None
        anchor_centroid_xy = None
    elif cfg.objective == "centroid":
        # cloud_centroid task: anchor at frame-0 blob xy centroid; target is
        # anchor + commanded offset.
        anchor_centroid_xy = blob_world[:, :2].mean(axis=0).astype(np.float32)
        target_xy = (anchor_centroid_xy
                     + np.array([cfg.centroid_offset_x, cfg.centroid_offset_y],
                                dtype=np.float32))
        cov_xy = None
        contour_world = None
    else:
        raise ValueError(f"Unknown objective {cfg.objective!r} "
                         f"(expected 'letter' or 'centroid')")

    # ── Plan trajectories (world frame).
    plan_blob   = np.zeros((cfg.total_frames + 1, N_blob,   3), dtype=np.float32)
    plan_paddle = np.zeros((cfg.total_frames + 1, N_paddle, 3), dtype=np.float32)
    plan_actions = np.zeros((cfg.total_frames, 4), dtype=np.float32)
    plan_blob[0]   = blob_world
    plan_paddle[0] = paddle_traj[0]

    K     = cfg.population_size
    K_la  = cfg.lookahead_commit_size

    # Nominal trajectory (the previous replan's plan, shifted forward). Starts
    # at all zeros (no warm start). After each replan we shift its committed
    # frames out so nominal[0] is the next action we *would* execute.
    nominal = np.zeros((PRED_FRAMES, 4), dtype=np.float32)
    std  = np.array([cfg.std_xyz, cfg.std_xyz, cfg.std_xyz, cfg.std_yaw], dtype=np.float32)
    amax = np.array([cfg.max_xyz, cfg.max_xyz, cfg.max_xyz, cfg.max_yaw], dtype=np.float32)

    log = {"replan_step": [], "best_cost": [], "mean_cost": [],
           "weights_entropy": [], "replan_time_sec": [],
           "n_committed_this_replan": []}

    committed = 0
    replan_idx = 0
    while committed < cfg.total_frames:
        t0 = time.time()
        # Per-replan horizon: planning extends through PRED_FRAMES, but never
        # beyond the episode budget.
        H_active = min(PRED_FRAMES, cfg.total_frames - committed)
        if cfg.horizon_mode == "final":
            # Chain 10-frame windows per candidate out to the plan's last frame; score there; commit the argmin candidate.
            H_rem = cfg.total_frames - committed; W = int(np.ceil(H_rem / PRED_FRAMES)); L = W * PRED_FRAMES
            if replan_idx == 0:
                nominal_f = np.zeros((cfg.total_frames + PRED_FRAMES, 4), dtype=np.float32)
            noise = rng.normal(0.0, std, size=(K, L, 4)).astype(np.float32)
            V = nominal_f[None, :L] + noise
            np.clip(V, -amax, amax, out=V)
            blob_k = np.repeat(blob_world[None].astype(np.float32), K, 0)
            pose_k = np.repeat(np.asarray(paddle_pose, np.float32)[None], K, 0)
            all_blob = [blob_k]; all_pose = [pose_k]
            for w in range(W):
                cum = np.cumsum(V[:, w * PRED_FRAMES:(w + 1) * PRED_FRAMES], axis=1)
                poses_w = np.zeros((K, MODEL_T, 4), dtype=np.float32); poses_w[:, 0] = pose_k; poses_w[:, 1:] = pose_k[:, None] + cum
                pred_w = model_forward_K(model, blob_k, paddle_particles_from_poses_K(paddle_tpl, poses_w), poses_w[..., 2:3],
                                         robot_features, scene_features, chunk_size=cfg.chunk_size, device=device)
                for f in range(1, MODEL_T):
                    all_blob.append(pred_w[:, f]); all_pose.append(poses_w[:, f])
                blob_k = pred_w[:, -1]; pose_k = poses_w[:, -1]
            all_blob = np.stack(all_blob, 1); all_pose = np.stack(all_pose, 1)     # (K, 1 + L, ...)
            fin_xy = all_blob[:, H_rem, :, :2]
            costs = (chamfer_batched_np(fin_xy, cov_xy, coverage_weight=cfg.letter_coverage_weight) if cfg.objective == "letter"
                     else centroid_dist_batched_np(fin_xy, target_xy, clamp_xy=cfg.centroid_clamp_xy))
            weights = mppi_softmax_weights(costs, cfg.temperature)              # logged only
            best_k = int(np.argmin(costs)); n_commit = min(K_la, H_rem)
            for i in range(n_commit):
                plan_blob[committed + 1 + i] = all_blob[best_k, 1 + i]
                plan_paddle[committed + 1 + i] = paddle_particles_from_pose(paddle_tpl, all_pose[best_k, 1 + i])
                plan_actions[committed + i] = V[best_k, i]
            blob_world = plan_blob[committed + n_commit].copy(); paddle_pose = all_pose[best_k, n_commit].astype(np.float32).copy()
            committed += n_commit
            nominal_f = np.zeros_like(nominal_f); nominal_f[:L - n_commit] = V[best_k, n_commit:L]
            log["replan_step"].append(replan_idx); log["best_cost"].append(float(costs.min())); log["mean_cost"].append(float(costs.mean()))
            log["weights_entropy"].append(float(-(weights * np.log(weights + 1e-12)).sum())); log["replan_time_sec"].append(time.time() - t0)
            log["n_committed_this_replan"].append(int(n_commit))
            print(f"  replan {replan_idx:3d}  committed {committed:3d}/{cfg.total_frames}  windows {W}  best={costs.min():.4e}  "
                  f"mean={costs.mean():.4e}  plan {log['replan_time_sec'][-1]:.2f}s")
            replan_idx += 1
            continue

        # K i.i.d. candidate action sequences, Gaussian around nominal, clamped.
        noise = rng.normal(0.0, std, size=(K, PRED_FRAMES, 4)).astype(np.float32)
        V_full = nominal[None] + noise                                  # (K, PRED_FRAMES, 4)
        np.clip(V_full, -amax, amax, out=V_full)

        # Integrate to absolute paddle poses for each candidate, MODEL_T frames.
        # Pose at frame 0 of the window = current paddle_pose; frames 1..MODEL_T-1
        # add cumsum(V_full).
        cum = np.cumsum(V_full, axis=1)                                  # (K, PRED_FRAMES, 4)
        poses_K = np.zeros((K, MODEL_T, 4), dtype=np.float32)
        poses_K[:, 0]  = paddle_pose
        poses_K[:, 1:] = paddle_pose[None, None] + cum                   # (K, PRED_FRAMES, 4)

        # Paddle particles per (K, MODEL_T, N_paddle, 3).
        paddle_K_window = paddle_particles_from_poses_K(paddle_tpl, poses_K)

        # joint_positions: use paddle z as a 1-DoF placeholder; the contract in
        # the data converter sets it to zeros, but feeding z is harmless if
        # robot_features doesn't include it, and meaningful otherwise.
        joint_K_window = poses_K[..., 2:3]

        # Forward through model in chunks: (K, MODEL_T, N_blob, 3) predicted.
        pred_blob_K = model_forward_K(
            model, blob_world, paddle_K_window, joint_K_window,
            robot_features, scene_features,
            chunk_size=cfg.chunk_size, device=device,
        )

        # Score at the FINAL predicted frame of the window.
        final_blob_xy = pred_blob_K[:, H_active, :, :2]                  # (K, N_blob, 2)
        if cfg.objective == "letter":
            costs = chamfer_batched_np(final_blob_xy, cov_xy,
                                        coverage_weight=cfg.letter_coverage_weight)
        else:   # centroid
            costs = centroid_dist_batched_np(final_blob_xy, target_xy,
                                              clamp_xy=cfg.centroid_clamp_xy)

        # MPPI softmax + weighted-average action sequence.
        weights = mppi_softmax_weights(costs, cfg.temperature)
        U = (weights[:, None, None] * V_full).sum(axis=0)                # (PRED_FRAMES, 4)

        # Pick the WM's prediction for the WEIGHTED-AVERAGE action. The
        # weighted average isn't one of the K candidates we already forwarded,
        # so we'd need an extra forward to be self-consistent. Instead, use
        # the best_k candidate's predictions (greedy MPPI) — keeps the imagined
        # state consistent with one of the actually evaluated trajectories.
        best_k = int(np.argmin(costs))
        pred_for_step = pred_blob_K[best_k]                              # (MODEL_T, N_blob, 3)
        poses_for_step = poses_K[best_k]                                 # (MODEL_T, 4)
        actions_for_step = V_full[best_k]                                # (PRED_FRAMES, 4)

        # Commit K_la frames of the chosen candidate.
        n_commit = min(K_la, H_active)
        for i in range(n_commit):
            plan_blob[committed + 1 + i]   = pred_for_step[1 + i]
            plan_paddle[committed + 1 + i] = paddle_particles_from_pose(
                paddle_tpl, poses_for_step[1 + i],
            )
            plan_actions[committed + i] = actions_for_step[i]

        # Advance imagined state to the committed-end frame.
        blob_world  = plan_blob[committed + n_commit].copy()
        paddle_pose = poses_for_step[n_commit].astype(np.float32).copy()
        committed += n_commit

        # Shift the nominal forward by n_commit; pad the tail with zeros.
        nominal = np.roll(actions_for_step, -n_commit, axis=0)
        nominal[-n_commit:] = 0.0

        log["replan_step"].append(replan_idx)
        log["best_cost"].append(float(costs.min()))
        log["mean_cost"].append(float(costs.mean()))
        log["weights_entropy"].append(float(-(weights * np.log(weights + 1e-12)).sum()))
        log["replan_time_sec"].append(time.time() - t0)
        log["n_committed_this_replan"].append(int(n_commit))
        print(f"  replan {replan_idx:3d}  committed {committed:3d}/{cfg.total_frames}  "
              f"best={costs.min():.4e}  mean={costs.mean():.4e}  "
              f"plan {log['replan_time_sec'][-1]:.2f}s")
        replan_idx += 1

    # Final-frame metric on the imagined planned blob.
    final_xy = plan_blob[-1, :, :2]
    if cfg.objective == "letter":
        final_chamfer = float(chamfer_batched_np(
            final_xy[None], cov_xy, coverage_weight=cfg.letter_coverage_weight,
        )[0])
        final_centroid_dist = float("nan")
        cloud_shift = float("nan")
    else:   # centroid
        final_centroid_xy = final_xy.mean(axis=0)
        final_centroid_dist = float(np.linalg.norm(final_centroid_xy - target_xy))
        cloud_shift         = float(np.linalg.norm(final_centroid_xy - anchor_centroid_xy))
        final_chamfer = float("nan")

    return dict(
        plan_blob       = plan_blob,
        plan_paddle     = plan_paddle,
        plan_actions    = plan_actions,
        gt_particle_pos = demo["particle_pos"][:cfg.total_frames + 1],
        obj_ids         = demo["obj_ids"],
        blob_mask       = demo["blob_mask"],
        paddle_mask     = demo["paddle_mask"],
        contour_world   = contour_world,
        cov_xy          = cov_xy,
        letter_field    = letter_field if cfg.objective == "letter" else None,
        final_chamfer_interior = final_chamfer,
        final_centroid_dist     = final_centroid_dist,
        cloud_shift             = cloud_shift,
        anchor_centroid_xy      = anchor_centroid_xy,
        target_xy               = target_xy,
        objective               = cfg.objective,
        replan_log      = log,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output: npz schema matches pwm_efficient/mppi_in_world_model.py
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(result: dict, output_dir: Path, batch_idx: int,
                 cfg: MPPIConfig) -> None:
    N_total = result["obj_ids"].shape[0]
    N_blob   = int(result["blob_mask"].sum())
    N_paddle = int(result["paddle_mask"].sum())
    T_plan   = result["plan_blob"].shape[0]
    blob_idx   = np.where(result["blob_mask"])[0]
    paddle_idx = np.where(result["paddle_mask"])[0]

    # Stitch blob + paddle predictions into a single (T, N_total, 3) array
    # following the same obj_id ordering as the GT.
    pred = np.zeros((T_plan, N_total, 3), dtype=np.float32)
    pred[:, blob_idx]   = result["plan_blob"]
    pred[:, paddle_idx] = result["plan_paddle"]

    # Targets: contour at the final frame for the letter task; for centroid
    # task we leave it all-NaN (replay/scoring uses target_xy from a side
    # channel, not the targets array).
    targets = np.full((1, T_plan, N_total, 3), np.nan, dtype=np.float32)
    if result.get("contour_world") is not None:
        n_wire = min(result["contour_world"].shape[0], N_total)
        targets[0, -1, :n_wire] = result["contour_world"][:n_wire]

    pos_dir = output_dir / "positions"
    pos_dir.mkdir(parents=True, exist_ok=True)

    extra = {}
    if result.get("objective") == "centroid":
        extra["centroid_target_xy"]  = result["target_xy"].astype(np.float32)
        extra["centroid_anchor_xy"]  = result["anchor_centroid_xy"].astype(np.float32)
        extra["final_centroid_dist"] = np.float32(result["final_centroid_dist"])
        extra["cloud_shift"]         = np.float32(result["cloud_shift"])

    np.savez(
        pos_dir / f"batch{batch_idx:03d}.npz",
        predicted=pred[None, None].astype(np.float32),      # (S=1, B=1, T, N, 3)
        gt=result["gt_particle_pos"][None].astype(np.float32),
        targets=targets.astype(np.float32),
        object_ids=result["obj_ids"][None],
        action_mask=(result["obj_ids"][None] != 0),
        valid_mask=np.ones((1, T_plan, N_total), dtype=bool),
        context_len=np.int32(CONTEXT_LEN),
        plan_actions=result["plan_actions"],
        **extra,
    )
    print(f"  saved: {pos_dir / f'batch{batch_idx:03d}.npz'}")

    # Letter-mode also writes a coverage/contour field file for downstream
    # viz; centroid mode skips it (no field to viz).
    lf = result.get("letter_field")
    if lf is not None:
        field_dir = output_dir / "field"
        field_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            field_dir / f"batch{batch_idx:03d}.npz",
            coverage=lf.coverage.detach().cpu().numpy(),
            contour =lf.contour.detach().cpu().numpy(),
            extent  =lf.extent.detach().cpu().numpy(),
            plane_z =lf.plane_z.detach().cpu().numpy(),
        )
        print(f"  saved: {field_dir / f'batch{batch_idx:03d}.npz'}")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (mirrors rollout_mould_viz.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_args(checkpoint: str, data_dir: str, device: str):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = pw_parse_args(skip_command_line=True)
    args.domains            = [DOMAIN]
    args.data_dirs          = [data_dir]
    args.norm_stats_path    = str(Path(data_dir) / "norm_stats")
    args.device             = device
    args.distributed        = False
    args.deterministic_train = False
    args.deterministic_data  = False
    ckpt_args = ckpt["args"]
    args.robot_features = list(ckpt_args.robot_features)
    args.scene_features = list(ckpt_args.scene_features)
    context = f"mould rollout checkpoint '{checkpoint}'"
    model_contract, _ = read_checkpoint_contract(ckpt, context=context)
    apply_model_contract_to_args(args, model_contract, context=context,
                                 explicit_cli_dests=set())
    return args, ckpt


def load_model(args, ckpt, data_info_dict: dict, device: str) -> BaseModel:
    model = BaseModel(args, data_info_dict, rank=0)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model = model.to(device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",
                   default="/data/group_data/katefgroup/VLA/3DPCWM/mould_new_6k_data_pointworld",
                   help="PointWorld mould dataset (used only for norm_stats / model contract).")
    p.add_argument("--test_dir",
                   default="/data/group_data/katefgroup/VLA/3DPCWM/deformable_mould_10k/test",
                   help="Raw mould WDS test shards (the demo loader reads these).")
    p.add_argument("--output_name", required=True,
                   help="<inference_results>/<output_name>/ is created here.")
    p.add_argument("--inference_root", default="/data/user_data/mbronars/packages/PointWorld/inference_results")

    p.add_argument("--demo_idx", type=int, nargs="+", default=[0],
                   help="Test-demo indices to plan for. One npz per index.")
    p.add_argument("--seed", type=int, default=0)

    # MPPI
    p.add_argument("--population_size",      type=int,   default=128)
    p.add_argument("--chunk_size",           type=int,   default=16)
    p.add_argument("--temperature",          type=float, default=0.3)
    p.add_argument("--lookahead_commit_size",type=int,   default=5)
    p.add_argument("--total_frames",         type=int,   default=150)
    p.add_argument("--horizon_mode", choices=["window", "final"], default="window",
                   help="window: score the end of one 10-frame window (original); final: chain windows per candidate and score the plan's last frame")

    # Action sampling
    p.add_argument("--std_xyz",   type=float, default=0.004,
                   help="Empirical training-data std on paddle Δxyz/step.")
    p.add_argument("--std_yaw",   type=float, default=0.10,
                   help="Empirical training-data std on paddle Δyaw/step.")
    p.add_argument("--max_xyz",   type=float, default=0.008,
                   help="Empirical hard cap on paddle Δxyz/step (sim PD ceiling).")
    p.add_argument("--max_yaw",   type=float, default=0.25,
                   help="Empirical hard cap on paddle Δyaw/step.")

    # Objective
    p.add_argument("--objective", choices=["letter", "centroid"], default="letter",
                   help="MPPI cost target: 'letter' fits a target letter field "
                        "(default — alphabet sweep), 'centroid' minimises "
                        "||final_blob_centroid_xy - target_xy|| (cloud_centroid task).")

    # Letter (used iff --objective letter)
    p.add_argument("--letter", default="C")
    p.add_argument("--letter_bank_path",
                   default=str(_PWMEFF / "stats/letter_bank_highres.npz"))
    p.add_argument("--letter_fit_rot_range",  type=float, default=40.0)
    p.add_argument("--letter_coverage_points",type=int,   default=150)
    p.add_argument("--letter_coverage_weight",type=float, default=1.0)
    p.add_argument("--letter_patch_frac",     type=float, default=0.2)
    p.add_argument("--letter_density_scale",  type=float, default=0.75)
    p.add_argument("--no_letter_random_rotation", action="store_true")
    p.add_argument("--letter_coverage_frac",  type=float, default=0.75)
    p.add_argument("--letter_chosen_rot_deg", type=float, default=None)

    # Centroid (used iff --objective centroid)
    p.add_argument("--centroid_offset_x", type=float, default=0.0,
                   help="Target offset x (m) from frame-0 blob centroid.")
    p.add_argument("--centroid_offset_y", type=float, default=0.0,
                   help="Target offset y (m) from frame-0 blob centroid.")
    p.add_argument("--centroid_clamp_xy", type=float, default=None,
                   help="If set, clip predicted blob xy to ±this (m) before "
                        "computing the centroid cost. Caps the influence of "
                        "WM-hallucinated outlier particles. e.g. 0.15 to "
                        "clamp at 1.5× the ±10 cm scene bound.")

    return p.parse_args()


def main() -> int:
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MPPIConfig(
        population_size       = args.population_size,
        chunk_size            = args.chunk_size,
        temperature           = args.temperature,
        lookahead_commit_size = args.lookahead_commit_size,
        total_frames          = args.total_frames,
        horizon_mode          = args.horizon_mode,
        std_xyz=args.std_xyz, std_yaw=args.std_yaw,
        max_xyz=args.max_xyz, max_yaw=args.max_yaw,
        objective              = args.objective,
        letter                 = args.letter,
        letter_bank_path       = args.letter_bank_path,
        letter_fit_rot_range   = args.letter_fit_rot_range,
        letter_coverage_points = args.letter_coverage_points,
        letter_coverage_weight = args.letter_coverage_weight,
        letter_patch_frac      = args.letter_patch_frac,
        letter_density_scale   = args.letter_density_scale,
        letter_random_rotation = not args.no_letter_random_rotation,
        letter_coverage_frac   = args.letter_coverage_frac,
        letter_chosen_rot_deg  = args.letter_chosen_rot_deg,
        centroid_offset_x      = args.centroid_offset_x,
        centroid_offset_y      = args.centroid_offset_y,
        centroid_clamp_xy      = args.centroid_clamp_xy,
        seed                   = args.seed,
    )

    print(f"\n{'='*60}\nMPPI-in-WM (PointWorld / mould)\n{'='*60}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"objective : {cfg.objective}")
    print(f"K={cfg.population_size}  chunk={cfg.chunk_size}  "
          f"K_la={cfg.lookahead_commit_size}  total={cfg.total_frames}")
    print(f"action std: xyz={cfg.std_xyz} yaw={cfg.std_yaw}   "
          f"clamp: ±xyz={cfg.max_xyz} ±yaw={cfg.max_yaw}")
    if cfg.objective == "letter":
        print(f"letter: {cfg.letter!r}  random_rot={cfg.letter_random_rotation}  "
              f"density_scale={cfg.letter_density_scale}")
    else:
        print(f"centroid target offset: ({cfg.centroid_offset_x:+.4f}, "
              f"{cfg.centroid_offset_y:+.4f}) m")
    print(f"{'='*60}\n")

    # Load model in-process.
    eval_args, ckpt = build_eval_args(args.checkpoint, args.data_dir, device)
    robot_features = list(eval_args.robot_features)
    scene_features = list(eval_args.scene_features)
    print(f"robot_features: {robot_features}")
    print(f"scene_features: {scene_features}\n")

    # Peek a dummy window to discover feature dims (BaseModel needs them).
    dummy_demo = load_mould_demo(args.test_dir, demo_idx=args.demo_idx[0])
    paddle_tpl_dummy, paddle_pose_dummy = make_paddle_template(dummy_demo["paddle_traj"][0])
    dummy_paddle = paddle_particles_from_pose(paddle_tpl_dummy, paddle_pose_dummy)
    dummy_paddle_K = np.broadcast_to(
        dummy_paddle[None, None], (1, MODEL_T, dummy_paddle.shape[0], 3),
    ).copy()
    dummy_joint_K = np.zeros((1, MODEL_T, 1), dtype=np.float32)
    peek, _ = build_window_batch_K(
        dummy_demo["blob_traj"][0], dummy_paddle_K, dummy_joint_K,
        robot_features, scene_features, device,
    )
    data_info_dict = {
        "robot_features_dim": peek["robot_features"].shape[-1],
        "scene_features_dim": peek["scene_features"].shape[-1],
    }
    print(f"robot_features_dim={data_info_dict['robot_features_dim']}  "
          f"scene_features_dim={data_info_dict['scene_features_dim']}\n")
    model = load_model(eval_args, ckpt, data_info_dict, device)

    output_dir = Path(args.inference_root) / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for batch_idx, di in enumerate(args.demo_idx):
        print(f"\n=== Demo {di} (batch_idx {batch_idx}) ===")
        # Mirror pwm_efficient/mppi_in_world_model.py's per-batch seeding so
        # the letter rotation (drawn from np.random's global state inside
        # build_letter_field) matches between the two planners.
        batch_seed = (args.seed * 10_000 + di) & 0xFFFFFFFF
        cfg.seed = batch_seed
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        demo = load_mould_demo(args.test_dir, demo_idx=di)
        t0 = time.time()
        result = mppi_rollout(model, demo, robot_features, scene_features, cfg, device)
        wall = time.time() - t0
        save_outputs(result, output_dir, batch_idx, cfg)

        m = dict(
            demo_idx              = di,
            batch_idx             = batch_idx,
            objective             = cfg.objective,
            final_chamfer_interior= result["final_chamfer_interior"],
            final_centroid_dist   = result.get("final_centroid_dist", float("nan")),
            cloud_shift           = result.get("cloud_shift", float("nan")),
            best_cost             = float(np.min(result["replan_log"]["best_cost"])),
            mean_cost_avg         = float(np.mean(result["replan_log"]["mean_cost"])),
            n_replans             = int(len(result["replan_log"]["replan_step"])),
            generation_time       = float(wall),
            per_replan_plan_sec_mean = float(np.mean(result["replan_log"]["replan_time_sec"])),
        )
        all_metrics.append(m)
        if cfg.objective == "letter":
            print(f"  final_chamfer_interior: {m['final_chamfer_interior']:.4e}  "
                  f"wall {wall:.1f}s  ({m['n_replans']} replans)")
        else:
            print(f"  final_centroid_dist (imagined): {m['final_centroid_dist']:.4e} m   "
                  f"cloud_shift: {m['cloud_shift']:.4e} m   "
                  f"wall {wall:.1f}s  ({m['n_replans']} replans)")

    if all_metrics:
        agg = {k: float(np.nanmean([m[k] for m in all_metrics]))
               for k in all_metrics[0]
               if k not in ("demo_idx", "batch_idx", "objective")}
        (output_dir / "metrics.json").write_text(json.dumps(
            dict(per_batch=all_metrics, aggregate=agg, config=cfg.__dict__),
            indent=2,
        ))
        print(f"\n=== Aggregate ===")
        if cfg.objective == "letter":
            print(f"  final_chamfer_interior: {agg['final_chamfer_interior']:.4e}")
        else:
            print(f"  final_centroid_dist (imagined): {agg['final_centroid_dist']:.4e} m")
            print(f"  cloud_shift:                    {agg['cloud_shift']:.4e} m")
        print(f"  mean wall time:                 {agg['generation_time']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
