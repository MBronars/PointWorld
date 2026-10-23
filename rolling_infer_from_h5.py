#!/usr/bin/env python3
"""
rolling_infer_pointworld.py — Autoregressive rollout for PointWorld from HDF5.

Replaces rolling_infer_from_h5.py for use with PointWorld (PTV3-based) models.
Extracts a long trajectory from an HDF5 pushing demo, then performs rolling
autoregressive inference: each window feeds predicted scene positions back as
context while replaying GT gripper actions.

The PointWorld model processes windows of T=11 frames (1 context + 10 prediction).
At each rolling step we:
  1. Center the scene + robot points (matching training's center_shift)
  2. Build scene_features at t=0: [positions(3), normals(3), dist2robot(T)]
  3. Build robot_features for all T:  [positions(3), normals(3), velocity(3), accel(3)]
  4. Run the model forward  →  predicted scene positions (B,T,Ns,3)
  5. Uncenter predictions, commit `stride` frames, slide forward

Usage:
    python rolling_infer_pointworld.py \\
        --h5 demos/PushBoundary/all_demos.h5 \\
        --checkpoint runs/push_run/model-last.pt \\
        --data-dir data/push_pointworld \\
        --total-frames 100 \\
        --stride 10 \\
        --out-dir rollout_results/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import trange

# ── Project path setup ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Point-cloud helpers (same ones used by h5_to_pointworld_wds.py)
from vis_push_pcd_floating import (
    sample_templates,
    apply_pose,
    ACTOR_POS,
    ACTOR_QUAT,
    FLOAT_QPOS,
    FLOAT_ROOT_POS,
)

# PointWorld model + infra
from pointworld.base import BaseModel
from pointworld.checkpoint_contract import (
    apply_model_contract_to_args,
    read_checkpoint_contract,
    train_domains_from_data_contract,
)
from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf
import transform_utils

# ═══════════════════════════════════════════════════════════════════════════
# Constants  (must match the PointWorld model configuration)
# ═══════════════════════════════════════════════════════════════════════════
DOMAIN = "pushing"
MODEL_T = 11          # CONTEXT_HORIZON(1) + PRED_HORIZON(10)
CONTEXT_FRAMES = 1
PRED_FRAMES = 10
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

from scipy.spatial.transform import Rotation  # for Kabsch yaw estimation

# Contact-metric thresholds (copied from rolling_infer_from_h5.py)
_BLOCK_HALF_SIZE     = 0.025    # cube half-side (m)
_PENETRATION_EPS     = -0.002   # 2 mm slack
_BLOCK_MOVE_PER_STEP = 0.002    # 2 mm/step → "block moving"


def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64).ravel()
    return float(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])


def _max_pen_depth_per_frame(
    gripper_pts_seq: np.ndarray,   # (T, N_hand, 3)
    block_pos_seq:   np.ndarray,   # (T, 3)
    block_quat_seq:  np.ndarray,   # (T, 4)  wxyz
    half_size:       float = _BLOCK_HALF_SIZE,
) -> np.ndarray:
    """Positive = deepest particle this far inside box; negative = outside."""
    T = len(gripper_pts_seq)
    depths = np.zeros(T, dtype=np.float32)
    for t in range(T):
        yaw = _yaw_from_quat_wxyz(block_quat_seq[t])
        cy, sy = np.cos(yaw), np.sin(yaw)
        d = gripper_pts_seq[t] - block_pos_seq[t]
        x_loc =  cy * d[:, 0] + sy * d[:, 1]
        y_loc = -sy * d[:, 0] + cy * d[:, 1]
        z_loc = d[:, 2]
        per = np.minimum(
            half_size - np.abs(x_loc),
            np.minimum(half_size - np.abs(y_loc), half_size - np.abs(z_loc)),
        )
        depths[t] = per.max()
    return depths


def _estimate_block_pose_kabsch(
    block_pts_seq: np.ndarray,   # (T, N_block, 3)
    template:      np.ndarray,   # (N_block, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid → position, yaw-only Kabsch in XY → quaternion wxyz."""
    T = block_pts_seq.shape[0]
    positions = np.zeros((T, 3), dtype=np.float32)
    quats     = np.zeros((T, 4), dtype=np.float32)
    tpl_c = template - template.mean(axis=0, keepdims=True)

    for t in range(T):
        pts = block_pts_seq[t]
        ctd = pts.mean(axis=0)
        positions[t] = ctd
        Pc = pts - ctd
        H  = tpl_c[:, :2].T @ Pc[:, :2]
        U, _, Vt = np.linalg.svd(H)
        if np.linalg.det(Vt.T @ U.T) < 0:
            Vt[-1] *= -1
        R2 = Vt.T @ U.T
        yaw = float(np.arctan2(R2[1, 0], R2[0, 0]))
        quats[t] = np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)], dtype=np.float32)
    return positions, quats


def compute_contact_metrics(
    pred_block:     np.ndarray,   # (T, N_block, 3)
    gt_robot:       np.ndarray,   # (T, N_hand, 3)
    block_template: np.ndarray,   # (N_block, 3)
    half_size:      float = _BLOCK_HALF_SIZE,
    pen_eps:        float = _PENETRATION_EPS,
    move_eps:       float = _BLOCK_MOVE_PER_STEP,
) -> dict:
    """Compute penetration and phantom-motion metrics for pointworld rollouts.

    Unlike the legacy script, scene and robot are already separate here — block
    positions come from the model, gripper positions are the replayed GT.
    """
    T = pred_block.shape[0]
    block_pos, block_quat = _estimate_block_pose_kabsch(pred_block, block_template)
    depth = _max_pen_depth_per_frame(gt_robot, block_pos, block_quat, half_size)

    pen_mask = depth > -pen_eps

    block_disp = np.zeros(T, dtype=np.float32)
    if T > 1:
        block_disp[1:] = np.linalg.norm(np.diff(block_pos[:, :2], axis=0), axis=1)
    not_in_contact = depth < (pen_eps - 0.001)
    phantom_mask   = (block_disp > move_eps) & not_in_contact

    def _safe_mean(a, m): return float(a[m].mean()) if m.any() else 0.0
    def _safe_max(a, m):  return float(a[m].max())  if m.any() else 0.0

    return {
        "pen_depth_per_frame":    depth,
        "pen_mask_per_frame":     pen_mask,
        "n_pen_frames":           int(pen_mask.sum()),
        "mean_pen_depth":         _safe_mean(depth, pen_mask),
        "max_pen_depth":          _safe_max(depth, pen_mask),
        "phantom_mask_per_frame": phantom_mask,
        "n_phantom_frames":       int(phantom_mask.sum()),
        "mean_phantom_dist":      _safe_mean(-depth, phantom_mask),
        "max_phantom_dist":       _safe_max(-depth, phantom_mask),
        "block_pos_est":          block_pos,
        "block_quat_est":         block_quat,
    }


# ── Camera presets & Palette ──────────────────────────────────────────────────
_VIEWS = [
    (55,  -55,  "Front-left"),
    (90,  -90,  "Top-down"),
    (25,  -135, "Back-right"),
]

_GT_BLOCK_C   = "steelblue"
_GT_GRIP_C    = "tomato"
_PRED_BLOCK_C = "deepskyblue"

_GT_ALPHA   = 0.55
_PRED_ALPHA = 0.55

def create_trajectory_comparison_gif(
    gt_block: np.ndarray,          # (T, Ns, 3)
    pred_block: np.ndarray,        # (T, Ns, 3)
    gt_robot: np.ndarray,          # (T, Nr, 3)
    save_path: str,
    fps: int = 8,
    views: list = None,
) -> None:
    """
    Save a multi-view GIF comparing ground-truth and predicted particle
    trajectories. Adapted for separate block and robot arrays.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import imageio
    except ImportError:
        print("  GIF skipped — matplotlib / imageio not installed.")
        return

    views = views or _VIEWS
    T = gt_block.shape[0]
    Ns = gt_block.shape[1]
    Nr = gt_robot.shape[1]

    # ── Axis limits (shared across all frames & views) ────────────────────────
    all_pts = np.concatenate([
        gt_block.reshape(-1, 3), 
        pred_block.reshape(-1, 3), 
        gt_robot.reshape(-1, 3)
    ], axis=0)
    
    mid  = (all_pts.max(0) + all_pts.min(0)) / 2
    span = (all_pts.max(0) - all_pts.min(0)).max() / 2 * 1.25
    x_lim = (mid[0] - span, mid[0] + span)
    y_lim = (mid[1] - span, mid[1] + span)
    z_lim = (mid[2] - span, mid[2] + span)

    # ── Per-frame MSE for title ───────────────────────────────────────────────
    frame_mse = np.mean((gt_block - pred_block) ** 2, axis=(1, 2))  # (T,)

    # ── Render ────────────────────────────────────────────────────────────────
    n_views = len(views)
    frames = []

    print(f"Rendering GIF across {T} frames...")
    
    for t in range(T):
        gt_b_t = gt_block[t]
        pr_b_t = pred_block[t]
        gt_r_t = gt_robot[t]

        suptitle = (
            f"Frame {t + 1}/{T}  │  "
            f"MSE: {frame_mse[t]:.6f}  │  "
            f"Block pts: {Ns}  Grip pts: {Nr}"
        )

        fig, axes = plt.subplots(
            1, n_views,
            figsize=(5.5 * n_views, 4.8),
            subplot_kw={"projection": "3d"},
        )
        if n_views == 1:
            axes = [axes]

        fig.suptitle(suptitle, fontsize=9)

        for col, (elev, azim, vlabel) in enumerate(views):
            ax = axes[col]
            ax.view_init(elev=elev, azim=azim)

            # GT block
            ax.scatter(gt_b_t[:, 0], gt_b_t[:, 1], gt_b_t[:, 2],
                       c=_GT_BLOCK_C, s=1.5, alpha=_GT_ALPHA,
                       label="GT block", depthshade=False)

            # Robot (Only GT exists in this script)
            ax.scatter(gt_r_t[:, 0], gt_r_t[:, 1], gt_r_t[:, 2],
                       c=_GT_GRIP_C, s=3, alpha=_GT_ALPHA,
                       label="Robot", depthshade=False)

            # Pred block
            ax.scatter(pr_b_t[:, 0], pr_b_t[:, 1], pr_b_t[:, 2],
                       c=_PRED_BLOCK_C, s=1.5, alpha=_PRED_ALPHA,
                       label="Pred block", depthshade=False)

            ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
            ax.set_xlabel("X", fontsize=6)
            ax.set_ylabel("Y", fontsize=6)
            ax.set_zlabel("Z", fontsize=6)
            ax.tick_params(labelsize=5)
            ax.set_title(vlabel, fontsize=8, pad=2)

            if col == n_views - 1:
                ax.legend(loc="upper right", fontsize=6,
                          markerscale=2.5, framealpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close(fig)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {save_path}  ({T} frames, {n_views} views)")


# ═══════════════════════════════════════════════════════════════════════════
# H5 helpers  (same as h5_to_pointworld_wds.py / rolling_infer_from_h5.py)
# ═══════════════════════════════════════════════════════════════════════════

def _discover_keys(f: h5py.File):
    """Auto-discover articulation and actor key names from the first trajectory."""
    first = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[0]
    s = f[first]
    arts = list(s["env_states/articulations"].keys())
    acts = list(s["env_states/actors"].keys())
    art = next(
        (k for c in ["floating_gripper", "robot", "panda_stick", "gripper"]
         for k in arts if c in k.lower()),
        arts[0],
    )
    blk = next(
        (k for c in ["push_block", "block", "cube"]
         for k in acts if c in k.lower()),
        acts[0],
    )
    return art, blk


def _build_tcp(art_states: np.ndarray) -> np.ndarray:
    """Build TCP pose [pos(3), quat_wxyz(4)] from articulation states.

    The pushing gripper is a floating body: root_pos + prismatic joints give
    the TCP position; orientation is always identity.
    """
    rp = art_states[:, FLOAT_ROOT_POS].copy()
    jxy = art_states[:, FLOAT_QPOS]
    rp[:, 0] += jxy[:, 0]
    rp[:, 1] += jxy[:, 1]
    q = np.tile(IDENTITY_QUAT_WXYZ, (len(art_states), 1))
    return np.concatenate([rp, q], axis=1).astype(np.float32)  # (T, 7)


# ═══════════════════════════════════════════════════════════════════════════
# Extract a LONG trajectory from HDF5
# ═══════════════════════════════════════════════════════════════════════════

def extract_long_trajectory(
    h5_path: str,
    total_frames: int,
    n_block: int,
    shape: str = "cube",
    subsample: int = 1,
    traj_idx: int | None = None,
    start_frame: int = 5,
    seed: int | None = None,
    compute_normals: bool = True,
    random_start: bool = False,
) -> dict[str, np.ndarray]:
    """Extract block point-clouds, normals, and gripper poses for many frames.

    Returns
    -------
    dict with keys:
        block_world    : (T, n_block, 3)  world-space block surface points
        block_normals  : (T, n_block, 3)  world-space block normals
        gripper_poses  : (T, 7)           TCP pose [pos(3), quat_wxyz(4)]
        joint_positions: (T, 2)           prismatic joint positions [x, y]
    """
    if seed is None:
        seed = int(time.time())
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        art_key, blk_key = _discover_keys(f)
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        if traj_idx is None:
            traj_idx = rng.integers(0, len(traj_keys))
        tk = traj_keys[traj_idx]
        print(f"Trajectory: {tk}  art='{art_key}'  block='{blk_key}'")

        bst = f[tk][f"env_states/actors/{blk_key}"][:]
        ast = f[tk][f"env_states/articulations/{art_key}"][:]

    tcp = _build_tcp(ast)
    # Skip first frame (reset lag) + subsample
    bst, tcp, ast = bst[1::subsample], tcp[1::subsample], ast[1::subsample]
    T_avail = bst.shape[0]

    # Optionally pick a random slice of length `total_frames` from this trajectory.
    if random_start:
        max_start = max(0, T_avail - total_frames)
        start_frame = int(rng.integers(0, max_start + 1))
        print(f"Random start frame: {start_frame}  (T_avail={T_avail}, requested={total_frames})")

    end = min(start_frame + total_frames, T_avail)
    actual = end - start_frame
    if actual < total_frames:
        print(f"WARNING: requested {total_frames} frames but only {actual} available")
        total_frames = actual

    # Sample a fixed block template (consistent geometry across all frames)
    tpls = sample_templates(n_block, n_block, rng)
    block_tpl = tpls["T"] if shape == "T" else tpls["cube"]  # (n_block, 3) local

    # Build world-space block point clouds for each frame
    block_world = np.zeros((total_frames, n_block, 3), dtype=np.float32)
    block_quats = np.zeros((total_frames, 4), dtype=np.float32)

    for i in range(total_frames):
        t = start_frame + i
        bs = bst[t]
        block_world[i] = apply_pose(block_tpl, bs[ACTOR_POS], bs[ACTOR_QUAT])
        block_quats[i] = bs[ACTOR_QUAT]

    # Analytical local normals for the block template, rotated to world frame.
    # Skipped entirely when scene normals are disabled.
    block_normals = None
    if compute_normals:
        block_normals_local = _block_normals_local(block_tpl, shape)  # (n_block, 3)
        block_normals = _rotate_normals_world(block_normals_local, block_quats)

    gripper_poses = tcp[start_frame : start_frame + total_frames]
    joint_positions = ast[start_frame : start_frame + total_frames, FLOAT_QPOS].astype(np.float32)

    print(f"Extracted {total_frames} frames, n_block={n_block}")
    return {
        "block_world": block_world,
        "block_normals": block_normals,
        "gripper_poses": gripper_poses,
        "joint_positions": joint_positions,
        "block_template": block_tpl.astype(np.float32),
    }


# ── Geometry helpers (from h5_to_pointworld_wds.py) ────────────────────────

def _block_normals_local(pts: np.ndarray, shape: str) -> np.ndarray:
    if shape == "cube":
        return _cube_normals_local(pts)
    return _pca_normals(pts)


def _cube_normals_local(pts: np.ndarray) -> np.ndarray:
    abs_pts = np.abs(pts)
    face_axis = np.argmax(abs_pts, axis=1)
    normals = np.zeros_like(pts, dtype=np.float32)
    rows = np.arange(len(pts))
    normals[rows, face_axis] = np.sign(pts[rows, face_axis]).astype(np.float32)
    zero_mask = normals[rows, face_axis] == 0.0
    normals[zero_mask, face_axis[zero_mask]] = 1.0
    return normals


def _pca_normals(pts: np.ndarray, k: int = 10) -> np.ndarray:
    N = len(pts)
    k = min(k, N - 1)
    sq_dist = ((pts[:, None] - pts[None]) ** 2).sum(-1)
    knn_idx = np.argpartition(sq_dist, k + 1, axis=1)[:, : k + 1]
    centroid = pts.mean(0)
    normals = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        nbrs = pts[knn_idx[i]]
        c = nbrs.mean(0)
        cov = (nbrs - c).T @ (nbrs - c) / len(nbrs)
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0].astype(np.float32)
        if np.dot(n, pts[i] - centroid) < 0:
            n = -n
        normals[i] = n
    return normals


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _rotate_normals_world(
    local_normals: np.ndarray,   # (N, 3)
    quats_wxyz: np.ndarray,      # (T, 4)
) -> np.ndarray:                 # (T, N, 3)
    T, N = len(quats_wxyz), len(local_normals)
    out = np.empty((T, N, 3), dtype=np.float32)
    for t in range(T):
        R = _quat_wxyz_to_rotmat(quats_wxyz[t])
        out[t] = (R @ local_normals.T).T
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Robot flow generation via RobotSampler  (matches training pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_robot_flows(
    robot_sampler: TorchRobotSampler,
    gripper_poses: np.ndarray,   # (T_total, 7)
    n_hand: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate robot surface point-clouds for every frame.

    Uses the URDF robot sampler (same as training) then transforms points
    to world frame using the GT gripper poses.

    Returns
    -------
    robot_flows  : (T_total, n_hand, 3)  world-space robot points
    robot_normals: (T_total, n_hand, 3)  world-space robot normals
    """
    T = gripper_poses.shape[0]
    device = robot_sampler.device
    dtype = robot_sampler.dtype

    robot_sampler.presample(n_hand, seed=seed)

    # Compute local-frame points (joints = 0 for floating gripper)
    zeros = torch.zeros(T, device=device, dtype=dtype)
    joint_dict = {"joint_x": zeros, "joint_y": zeros}
    robot_flows, _colors, robot_normals = robot_sampler.compute_points(joint_dict)
    robot_flows = robot_flows.cpu().numpy()       # (T, n_hand, 3) local
    robot_normals = robot_normals.cpu().numpy()   # (T, n_hand, 3) local

    # Transform to world frame using gripper poses
    T_g = transform_utils.convert_pose_quat2mat(gripper_poses)  # (T, 4, 4)
    ones = np.ones(robot_flows.shape[:-1] + (1,), dtype=robot_flows.dtype)
    pts_h = np.concatenate([robot_flows, ones], axis=-1)  # (T, N, 4)
    robot_flows = np.einsum("tij,tkj->tki", T_g, pts_h)[..., :3]
    R_g = T_g[:, :3, :3]
    robot_normals = np.einsum("tij,tkj->tki", R_g, robot_normals)

    return robot_flows, robot_normals


# ═══════════════════════════════════════════════════════════════════════════
# Feature computation  (mirrors gather_features + transforms from training)
# ═══════════════════════════════════════════════════════════════════════════

def compute_flow_derivatives(flows: np.ndarray):
    """Central-diff velocity and acceleration.  flows: (T, N, 3)."""
    T = flows.shape[0]
    vel = np.zeros_like(flows)
    acc = np.zeros_like(flows)
    if T >= 2:
        vel[0] = flows[1] - flows[0]
        for t in range(1, T - 1):
            vel[t] = (flows[t + 1] - flows[t - 1]) / 2.0
        vel[-1] = flows[-1] - flows[-2]
    if T >= 3:
        acc[0] = vel[1] - vel[0]
        for t in range(1, T - 1):
            acc[t] = (vel[t + 1] - vel[t - 1]) / 2.0
        acc[-1] = vel[-1] - vel[-2]
    return vel, acc


def compute_scene_features(
    scene_pos_t0: np.ndarray,             # (Ns, 3)  centred
    scene_normals_t0: np.ndarray | None,  # (Ns, 3)  or None when disabled
    robot_flows_window: np.ndarray,       # (T_win, Nr, 3) centred
    use_scene_normals: bool = True,
) -> np.ndarray:
    """Scene features at t=0: [flows(3), (normals(3),) dist2robot(T_win)].

    Returns shape (1, Ns, 3 + (3 if use_scene_normals else 0) + T_win).
    """
    T_win = robot_flows_window.shape[0]
    Ns = scene_pos_t0.shape[0]

    dist2robot = np.zeros((T_win, Ns), dtype=np.float32)
    for t in range(T_win):
        tree = cKDTree(robot_flows_window[t])
        dist2robot[t], _ = tree.query(scene_pos_t0)

    feat_parts = [scene_pos_t0]                  # (Ns, 3)
    if use_scene_normals:
        feat_parts.append(scene_normals_t0)      # (Ns, 3)
    feat_parts.append(dist2robot.T)              # (Ns, T_win)

    scene_feats = np.concatenate(feat_parts, axis=-1)
    return scene_feats[None]  # (1, Ns, 3 + (3) + T_win)


def compute_robot_features(
    robot_flows: np.ndarray,    # (T_win, Nr, 3) centred
    robot_normals: np.ndarray,  # (T_win, Nr, 3)
) -> np.ndarray:
    """Robot features: [flows(3), normals(3), velocity(3), acceleration(3)].

    Returns shape (T_win, Nr, 12).
    """
    vel, acc = compute_flow_derivatives(robot_flows)
    return np.concatenate([robot_flows, robot_normals, vel, acc], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Batch construction (single window → dict the model expects)
# ═══════════════════════════════════════════════════════════════════════════

def build_window_batch(
    scene_pos_t0: np.ndarray,             # (Ns, 3)  world frame
    scene_normals_t0: np.ndarray | None,  # (Ns, 3)  world frame, or None
    robot_flows: np.ndarray,              # (MODEL_T, Nr, 3)  world frame
    robot_normals: np.ndarray,            # (MODEL_T, Nr, 3)  world frame
    domain: str,
    device: str | torch.device,
    use_scene_normals: bool = True,
) -> tuple[dict, np.ndarray]:
    """Build a single-sample batch dict for one inference window.

    Applies center_shift (mean of scene+robot at t=0) to match training.

    Returns
    -------
    batch : dict[str, Tensor]   ready for model.forward()
    shift : (3,) ndarray        world-space shift (add back to uncenter outputs)
    """
    Ns = scene_pos_t0.shape[0]
    Nr = robot_flows.shape[1]

    # ── center shift: mean of combined scene + robot at t=0 ───────────────
    combined = np.concatenate([scene_pos_t0, robot_flows[0]], axis=0)
    shift = combined.mean(axis=0).astype(np.float32)

    scene_pos_c = scene_pos_t0 - shift                 # (Ns, 3) centred
    robot_flows_c = robot_flows.copy() - shift          # (T, Nr, 3) centred

    # ── scene_flows tensor: model only reads [:, 0] ──────────────────────
    scene_flows = np.zeros((MODEL_T, Ns, 3), dtype=np.float32)
    scene_flows[0] = scene_pos_c

    # ── features ─────────────────────────────────────────────────────────
    scene_feats = compute_scene_features(
        scene_pos_c, scene_normals_t0, robot_flows_c, use_scene_normals
    )
    robot_feats = compute_robot_features(robot_flows_c, robot_normals)

    # ── assemble batch dict ──────────────────────────────────────────────
    batch = {
        "scene_flows": torch.from_numpy(scene_flows[None]).float().to(device),       # (1,T,Ns,3)
        "scene_features": torch.from_numpy(scene_feats[None]).float().to(device),    # (1,1,Ns,Ds)
        "scene_exists": torch.ones(1, MODEL_T, Ns, dtype=torch.bool, device=device),
        "robot_flows": torch.from_numpy(robot_flows_c[None]).float().to(device),     # (1,T,Nr,3)
        "robot_features": torch.from_numpy(robot_feats[None]).float().to(device),    # (1,T,Nr,12)
        "robot_exists": torch.ones(1, MODEL_T, Nr, dtype=torch.bool, device=device),
        "__domain__": [domain],
    }
    return batch, shift


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

# Dimension look-up for building data_info_dict from feature name lists.
_FEAT_DIM = {
    "robot_flows": 3,
    "robot_normals": 3,
    "robot_colors": 3,
    "robot_velocity": 3,
    "robot_acceleration": 3,
    "gripper_open": 1,
    "scene_flows": 3,
    "scene_normals": 3,
    "scene_colors": 3,
    # dist2robot has dimension T (added at runtime)
}


def _robot_feat_dim(feat_names: list[str]) -> int:
    return sum(_FEAT_DIM[f] for f in feat_names)


def _scene_feat_dim(feat_names: list[str], T: int = MODEL_T) -> int:
    d = 0
    for f in feat_names:
        if f == "dist2robot":
            d += T
        else:
            d += _FEAT_DIM[f]
    return d


def load_pointworld_model(
    checkpoint_path: str,
    data_dir: str,
    domain: str = DOMAIN,
    device: str = "cuda",
    use_scene_normals: bool = True,
) -> BaseModel:
    """Load a PointWorld BaseModel from a training checkpoint.

    Parameters
    ----------
    checkpoint_path : path to ``model-last.pt``
    data_dir        : dataset root (must contain ``norm_stats/norm_stats.json``)
    domain          : domain string registered in norm_stats
    device          : torch device
    """
    # ── load checkpoint & extract contract ────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    context = f"rolling inference checkpoint '{checkpoint_path}'"
    model_contract, data_contract = read_checkpoint_contract(ckpt, context=context)
    train_domains = train_domains_from_data_contract(data_contract, context=context)

    # ── build a minimal args namespace ────────────────────────────────────
    args = argparse.Namespace(
        device=device,
        distributed=False,
        domains=list(train_domains),
        data_dirs=[data_dir] * len(train_domains),
    )
    # Apply model contract (fills predictor_dim, ptv3_size, etc.)
    apply_model_contract_to_args(args, model_contract, context=context, explicit_cli_dests=set())

    # Ensure required fields have defaults if the contract didn't set them
    defaults = {
        "predictor_dim": 128,
        "ptv3_size": "small",
        "ptv3_patch_size": 32,
        "grid_size": 0.01,
        "dynamics_head_init_scale": 0.01,
        "robot_features": [
            "robot_flows", "robot_normals", "robot_velocity", "robot_acceleration",
        ],
        "scene_features": [
            "scene_flows", "scene_normals", "dist2robot",
        ],
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Honour --no-scene-normals: drop scene_normals from the scene feature list
    if not use_scene_normals and "scene_normals" in args.scene_features:
        args.scene_features = [f for f in args.scene_features if f != "scene_normals"]
        print("scene_normals disabled — excluded from scene_features")

    # ── build data_info_dict ─────────────────────────────────────────────
    rf_dim = _robot_feat_dim(args.robot_features)
    sf_dim = _scene_feat_dim(args.scene_features, MODEL_T)
    data_info_dict = {
        "robot_features_dim": rf_dim,
        "scene_features_dim": sf_dim,
    }
    print(f"robot_features_dim={rf_dim}  scene_features_dim={sf_dim}")
    print(f"train_domains={train_domains}  predictor_dim={args.predictor_dim}")

    # ── instantiate model ────────────────────────────────────────────────
    model = BaseModel(args, data_info_dict, rank=0)

    # ── load weights ─────────────────────────────────────────────────────
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        model.load_state_dict(state, strict=False)
        print("Loaded model weights from checkpoint")
    else:
        print("WARNING: could not locate state_dict in checkpoint — model uses random weights")

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Rolling inference loop
# ═══════════════════════════════════════════════════════════════════════════

def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@torch.no_grad()
def rolling_inference(
    model: BaseModel,
    gt_block: np.ndarray,                 # (T_total, Ns, 3)
    gt_block_normals: np.ndarray | None,  # (T_total, Ns, 3) or None
    gt_robot: np.ndarray,                 # (T_total, Nr, 3)
    gt_robot_normals: np.ndarray,         # (T_total, Nr, 3)
    stride: int = PRED_FRAMES,
    domain: str = DOMAIN,
    device: str = "cuda",
    use_scene_normals: bool = True,
) -> dict:
    """Autoregressive rolling inference.

    Parameters
    ----------
    gt_block / gt_block_normals : ground-truth scene point clouds + normals
    gt_robot / gt_robot_normals : ground-truth robot point clouds + normals
    stride : number of predicted frames to commit per window (≤ PRED_FRAMES)

    Returns
    -------
    dict  with gt_positions, predicted_positions, per_frame_mse, etc.
    """
    assert stride <= PRED_FRAMES, f"stride ({stride}) must be ≤ PRED_FRAMES ({PRED_FRAMES})"
    T_total = gt_block.shape[0]
    Ns = gt_block.shape[1]
    Nr = gt_robot.shape[1]
    print(f"\nRolling inference: T_total={T_total}, Ns={Ns}, Nr={Nr}, stride={stride}")

    # ── output buffer ────────────────────────────────────────────────────
    pred_block = np.zeros_like(gt_block)   # (T_total, Ns, 3)
    pred_block[0] = gt_block[0]            # first frame is GT context

    # Current context (world frame)
    ctx_scene = gt_block[0].copy()
    # TODO: track predicted normals instead of using GT
    ctx_normals = gt_block_normals[0].copy() if use_scene_normals else None

    next_commit = 1  # next global frame index to fill
    window_num = 0

    pbar = trange(0, T_total, stride, desc="Rolling")
    for _ in pbar:
        if next_commit >= T_total:
            break

        # ── assemble MODEL_T robot frames for this window ────────────────
        # Frame 0 = context robot (one frame before first prediction)
        ctx_robot_idx = next_commit - 1
        # Frames 1..PRED_FRAMES = prediction robot
        pred_robot_indices = list(range(next_commit, min(next_commit + PRED_FRAMES, T_total)))
        F = len(pred_robot_indices)  # actual prediction frames available

        robot_indices = [ctx_robot_idx] + pred_robot_indices
        # Pad to MODEL_T by repeating last frame
        while len(robot_indices) < MODEL_T:
            robot_indices.append(robot_indices[-1])
        robot_indices = robot_indices[:MODEL_T]

        robot_win = gt_robot[robot_indices]                # (MODEL_T, Nr, 3)
        robot_norm_win = gt_robot_normals[robot_indices]   # (MODEL_T, Nr, 3)

        # ── build batch ──────────────────────────────────────────────────
        batch, shift = build_window_batch(
            ctx_scene, ctx_normals,
            robot_win, robot_norm_win,
            domain, device,
            use_scene_normals=use_scene_normals,
        )

        # ── model forward ────────────────────────────────────────────────
        # DIAG: dump exact model inputs for side-by-side with pointworld_server.py
        import os
        _diag_dir = os.environ.get("PW_DIAG_DUMP_DIR")
        if _diag_dir:
            os.makedirs(_diag_dir, exist_ok=True)
            _dump = {
                k: v.detach().cpu().numpy()
                for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            np.savez(os.path.join(_diag_dir, f"REF_window{window_num:02d}.npz"),
                     shift_world=shift, **_dump)
            print(f"  [DIAG REF] window={window_num} "
                  f"scene_flows={tuple(batch['scene_flows'].shape)} "
                  f"scene_features={tuple(batch['scene_features'].shape)} "
                  f"robot_flows={tuple(batch['robot_flows'].shape)} "
                  f"robot_features={tuple(batch['robot_features'].shape)} "
                  f"shift={shift.tolist()}")
        outputs = model(batch, training=False)

        # scene_flows: (1, MODEL_T, Ns, 3) in centred frame
        pred_centred = outputs["scene_flows"][0].cpu().numpy()   # (MODEL_T, Ns, 3)
        pred_world = pred_centred + shift                        # uncenter

        # ── commit frames ────────────────────────────────────────────────
        commit_F = min(stride, F, T_total - next_commit)
        for i in range(commit_F):
            pred_block[next_commit + i] = pred_world[1 + i]     # index 0 = context echo

        # ── update context for next window ───────────────────────────────
        ctx_scene = pred_world[commit_F].copy()   # last committed prediction
        # TODO: derive normals from predicted positions (e.g. PCA or predicted rotation)
        #       For now, fall back to GT normals at the corresponding frame.
        if use_scene_normals:
            gt_idx = min(next_commit + commit_F - 1, T_total - 1)
            ctx_normals = gt_block_normals[gt_idx].copy()

        # ── logging ──────────────────────────────────────────────────────
        gt_compare = gt_block[next_commit : next_commit + commit_F]
        pr_compare = pred_block[next_commit : next_commit + commit_F]
        win_mse = float(((gt_compare - pr_compare) ** 2).mean())
        pbar.set_postfix(win=window_num, done=next_commit + commit_F, mse=f"{win_mse:.6f}")

        next_commit += commit_F
        window_num += 1

    # ── aggregate metrics ────────────────────────────────────────────────
    T_pred = next_commit
    gt_trim = gt_block[:T_pred]
    pr_trim = pred_block[:T_pred]

    diff = gt_trim - pr_trim                                          # (T_pred, Ns, 3)
    per_particle_dist = np.linalg.norm(diff, axis=-1)                 # (T_pred, Ns)
    per_frame_dist    = per_particle_dist.mean(axis=1)                # (T_pred,)
    per_frame_mse     = (diff ** 2).mean(axis=(1, 2))                 # (T_pred,)

    overall   = float(per_frame_dist[CONTEXT_FRAMES:].mean())   # exclude context
    final     = float(per_frame_dist[-1])
    sum_dist  = float(per_frame_dist[CONTEXT_FRAMES:].sum())    # Σ over predicted frames
    sum_mse   = float(per_frame_mse [CONTEXT_FRAMES:].sum())

    print(f"\n{'=' * 60}")
    print(f"Rolling inference complete: {window_num} windows, {T_pred} frames predicted")
    print(f"  Mean L2 dist (post-ctx):  {overall:.6f}")
    print(f"  Σ-Dist over {T_pred - CONTEXT_FRAMES} frames: {sum_dist:.6f}")
    print(f"  Σ-MSE  over {T_pred - CONTEXT_FRAMES} frames: {sum_mse:.6e}")
    print(f"  Final-frame L2 dist:      {final:.6f}")
    print(f"{'=' * 60}")

    return {
        "gt_positions": gt_trim,                # (T_pred, Ns, 3)
        "predicted_positions": pr_trim,         # (T_pred, Ns, 3)
        "per_frame_dist": per_frame_dist,       # (T_pred,)
        "per_frame_mse":  per_frame_mse,        # (T_pred,)
        "sum_dist_post_context": sum_dist,
        "sum_mse_post_context":  sum_mse,
        "gt_robot": gt_robot[:T_pred],          # (T_pred, Nr, 3)
        "num_windows": window_num,
        "total_frames": T_pred,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def save_dist_plot(
    mean_dist: np.ndarray,
    std_dist: np.ndarray,
    save_path: str,
    context_len: int,
    n_samples: int,
):
    """Per-frame mean ± std L2 distance plot aggregated over samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(mean_dist)
    frames = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, mean_dist, color="black", linewidth=1.5, label="Mean dist")
    ax.fill_between(
        frames, mean_dist - std_dist, mean_dist + std_dist,
        color="black", alpha=0.2, label="±1 std",
    )
    ax.axvline(context_len, color="gray", linestyle="--", alpha=0.5, label="Context end")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean L2 distance")
    ax.set_title(f"Per-frame L2 distance — {n_samples} sample(s)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, T - 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Distance plot → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI + main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser(
        description="Rolling autoregressive inference for PointWorld from HDF5",
    )
    # ── I/O ──
    pa.add_argument("--h5", required=True, help="Input HDF5 demo file")
    pa.add_argument("--checkpoint", required=True, help="PointWorld model checkpoint (.pt)")
    pa.add_argument("--data-dir", required=True,
                    help="Dataset root (contains norm_stats/norm_stats.json)")
    pa.add_argument("--out-dir", default="rollout_results_pw")

    # ── Trajectory selection ──
    pa.add_argument("--traj-idx", type=int, default=None,
                    help="Fixed trajectory index (overrides --num-samples).")
    pa.add_argument("--num-samples", type=int, default=1,
                    help="Number of randomly sampled trajectories to evaluate (default: 1).")
    pa.add_argument("--start-frame", type=int, default=5,
                    help="Fixed start frame within the trajectory "
                         "(ignored if --random-start is set).")
    pa.add_argument("--random-start", action="store_true",
                    help="Pick a uniform-random start frame per sampled trajectory "
                         "so the 100-frame slice is randomized.")
    pa.add_argument("--total-frames", type=int, default=100,
                    help="Total frames to roll out (including context).")
    pa.add_argument("--start-eval-frame", type=int, default=CONTEXT_FRAMES,
                    help=f"Frame index from which to start counting toward the "
                         f"aggregated Σ-Dist / Σ-MSE metrics (inclusive). "
                         f"Default {CONTEXT_FRAMES} = skip just the context frame. "
                         f"Set higher (e.g. 10) to ignore the model's near-future "
                         f"easy predictions. Per-frame curves are still saved "
                         f"in full regardless of this setting.")

    # ── Point cloud ──
    pa.add_argument("--n-block", type=int, default=128)
    pa.add_argument("--n-hand", type=int, default=128)
    pa.add_argument("--shape", default="cube", choices=["cube", "T"])
    pa.add_argument("--subsample", type=int, default=1)
    pa.add_argument("--no-scene-normals", action="store_true",
                    help="Drop scene_normals from scene_features and skip all "
                         "related computation (use with models trained without it)")

    # ── Rolling parameters ──
    pa.add_argument("--stride", type=int, default=PRED_FRAMES,
                    help=f"Frames to commit per window (≤{PRED_FRAMES}, default: {PRED_FRAMES})")

    # ── Model ──
    pa.add_argument("--domain", default=DOMAIN)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--seed", type=int, default=None)
    
    # ── Output ──
    pa.add_argument("--save-gifs", action="store_true", help="Render multi-view overlay GIF")
    pa.add_argument("--gif-fps", type=int, default=8)

    return pa.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.start_eval_frame < 0:
        raise ValueError(f"--start-eval-frame must be >= 0, got {args.start_eval_frame}")
    if args.start_eval_frame >= args.total_frames:
        raise ValueError(
            f"--start-eval-frame ({args.start_eval_frame}) must be < "
            f"--total-frames ({args.total_frames}); otherwise no frames "
            f"would contribute to the aggregated metrics."
        )

    sample_rng = np.random.default_rng(args.seed)

    use_scene_normals = not args.no_scene_normals

    # ── Load model ───────────────────────────────────────────────────────
    model = load_pointworld_model(
        args.checkpoint, args.data_dir, domain=args.domain, device=device,
        use_scene_normals=use_scene_normals,
    )

    # ── URDF-based robot sampler (shared across samples) ─────────────────
    urdf_path = resolve_robot_urdf(args.domain)
    robot_sampler = TorchRobotSampler(urdf_path=urdf_path, gripper_only=True, device="cpu")

    # ── Build sample list: (traj_idx, per_sample_seed) pairs ─────────────
    # Pre-screen trajectories by length: a trajectory contributes a
    # `total_frames`-frame rollout only if it has enough samples after the
    # reset-lag skip + temporal subsample.
    #   - With --random-start, we need T_avail >= total_frames.
    #   - Without, we need T_avail >= start_frame + total_frames.
    # The same `bst[1::subsample]` slice is applied inside extract_long_trajectory,
    # so its length is len(range(1, n_raw, subsample)).
    def _t_avail_after_subsample(n_raw: int, subsample: int) -> int:
        if n_raw <= 1:
            return 0
        return (n_raw - 2 + subsample) // subsample  # == len(range(1, n_raw, subsample))

    required = args.total_frames if args.random_start else (args.start_frame + args.total_frames)

    with h5py.File(args.h5, "r") as f:
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        n_trajs = len(traj_keys)
        # Use the first trajectory to discover the actor/articulation key names,
        # then read the per-trajectory frame count cheaply via `.shape[0]`.
        art_key, blk_key = _discover_keys(f)
        usable_indices: list[int] = []
        for idx, tk in enumerate(traj_keys):
            n_raw = f[tk][f"env_states/actors/{blk_key}"].shape[0]
            T_avail = _t_avail_after_subsample(n_raw, args.subsample)
            if T_avail >= required:
                usable_indices.append(idx)

    n_skipped = n_trajs - len(usable_indices)
    print(f"H5 contains {n_trajs} trajectories; "
          f"{len(usable_indices)} have >= {required} usable frames "
          f"(after skip + subsample={args.subsample}). "
          f"Skipping {n_skipped} too-short trajectory(ies).")

    if args.traj_idx is not None:
        # Fixed trajectory: trust the user even if it's short. The downstream
        # code will warn + truncate if it ends up too short.
        sample_list = [(args.traj_idx, int(sample_rng.integers(0, 2**31)))]
    else:
        if len(usable_indices) == 0:
            raise RuntimeError(
                f"No trajectories in {args.h5} have >= {required} usable frames "
                f"(start_frame={args.start_frame}, total_frames={args.total_frames}, "
                f"random_start={args.random_start}, subsample={args.subsample}). "
                f"Lower --total-frames or --start-frame, or remove --random-start."
            )
        take = min(args.num_samples, len(usable_indices))
        if take < args.num_samples:
            print(f"WARNING: only {len(usable_indices)} usable trajectories — "
                  f"requested {args.num_samples}, using {take}.")
        traj_indices = sample_rng.choice(
            usable_indices, size=take, replace=False,
        ).tolist()
        sample_seeds = sample_rng.integers(0, 2**31, size=take).tolist()
        sample_list = list(zip(traj_indices, sample_seeds))

    all_dist_curves: list[np.ndarray] = []
    all_mse_curves:  list[np.ndarray] = []
    sample_records: list[dict] = []

    for sample_i, (traj_idx, sample_seed) in enumerate(sample_list):
        print(f"\n{'=' * 60}")
        print(f"Sample {sample_i + 1}/{len(sample_list)}  traj_idx={traj_idx}  seed={sample_seed}")
        print(f"{'=' * 60}")

        # ── Extract trajectory ──────────────────────────────────────────
        raw = extract_long_trajectory(
            h5_path=args.h5,
            total_frames=args.total_frames,
            n_block=args.n_block,
            shape=args.shape,
            subsample=args.subsample,
            traj_idx=traj_idx,
            start_frame=args.start_frame,
            seed=sample_seed,
            compute_normals=use_scene_normals,
            random_start=args.random_start,
        )
        block_world    = raw["block_world"]
        block_normals  = raw["block_normals"]
        gripper_poses  = raw["gripper_poses"]
        block_template = raw["block_template"]

        # ── Generate robot point clouds ─────────────────────────────────
        robot_flows, robot_normals = generate_all_robot_flows(
            robot_sampler, gripper_poses, args.n_hand, seed=42,
        )

        # ── Rolling inference ───────────────────────────────────────────
        results = rolling_inference(
            model=model,
            gt_block=block_world,
            gt_block_normals=block_normals,
            gt_robot=robot_flows,
            gt_robot_normals=robot_normals,
            stride=args.stride,
            domain=args.domain,
            device=device,
            use_scene_normals=use_scene_normals,
        )
        dist_curve = _to_np(results["per_frame_dist"])
        pred_block_np = _to_np(results["predicted_positions"])
        gt_robot_np   = _to_np(results["gt_robot"])

        # ── Contact metrics ─────────────────────────────────────────────
        contact = compute_contact_metrics(
            pred_block     = pred_block_np,
            gt_robot       = gt_robot_np,
            block_template = block_template,
        )
        T_eval = pred_block_np.shape[0]
        print(f"  Penetration : {contact['n_pen_frames']}/{T_eval} frames "
              f"(mean {contact['mean_pen_depth']*100:.2f} cm, "
              f"max {contact['max_pen_depth']*100:.2f} cm)")
        print(f"  Phantom move: {contact['n_phantom_frames']}/{T_eval} frames "
              f"(mean {contact['mean_phantom_dist']*100:.2f} cm, "
              f"max {contact['max_phantom_dist']*100:.2f} cm)")

        all_dist_curves.append(dist_curve)
        mse_curve = _to_np(results["per_frame_mse"])
        all_mse_curves.append(mse_curve)

        # Per-sample aggregated metrics over the eval slice (frames
        # [start_eval_frame:]). Per-frame curves are also kept in full above.
        eval_slice = slice(args.start_eval_frame, None)
        sample_records.append({
            "sample_index":           sample_i,
            "traj_idx":               int(traj_idx),
            "sample_seed":            int(sample_seed),
            "total_frames":           int(results["total_frames"]),
            "start_eval_frame":       int(args.start_eval_frame),
            "mean_dist_eval":         float(np.nanmean(dist_curve[eval_slice])),
            "mean_mse_eval":          float(np.nanmean(mse_curve [eval_slice])),
            "sum_dist_eval":          float(np.nansum (dist_curve[eval_slice])),
            "sum_mse_eval":           float(np.nansum (mse_curve [eval_slice])),
            "final_frame_dist":       float(dist_curve[-1]),
            "final_frame_mse":        float(mse_curve [-1]),
            "per_frame_dist":         dist_curve.tolist(),
            "per_frame_mse":          mse_curve.tolist(),
            "n_pen_frames":           contact["n_pen_frames"],
            "mean_pen_depth":         contact["mean_pen_depth"],
            "max_pen_depth":          contact["max_pen_depth"],
            "n_phantom_frames":       contact["n_phantom_frames"],
            "mean_phantom_dist":      contact["mean_phantom_dist"],
            "max_phantom_dist":       contact["max_phantom_dist"],
            "pen_depth_per_frame":    contact["pen_depth_per_frame"].tolist(),
            "pen_mask_per_frame":     contact["pen_mask_per_frame"].astype(int).tolist(),
            "phantom_mask_per_frame": contact["phantom_mask_per_frame"].astype(int).tolist(),
        })

        # ── Per-sample GIF ──────────────────────────────────────────────
        if args.save_gifs:
            gif_path = str(out / f"rollout_comparison_s{sample_i:03d}.gif")
            create_trajectory_comparison_gif(
                gt_block   = _to_np(results["gt_positions"]),
                pred_block = pred_block_np,
                gt_robot   = gt_robot_np,
                save_path  = gif_path,
                fps        = args.gif_fps,
            )

    # ── Aggregate across samples (NaN-pad shorter curves) ───────────────
    # Per-frame curves are kept in full; the eval slice only affects the
    # headline Σ-Dist / Σ-MSE numbers below.
    max_T = max(d.shape[0] for d in all_dist_curves)
    n_samples = len(all_dist_curves)
    padded_dist = np.full((n_samples, max_T), np.nan, dtype=np.float64)
    padded_mse  = np.full((n_samples, max_T), np.nan, dtype=np.float64)
    for i, (d, m) in enumerate(zip(all_dist_curves, all_mse_curves)):
        padded_dist[i, : d.shape[0]] = d
        padded_mse [i, : m.shape[0]] = m
    mean_dist_per_frame = np.nanmean(padded_dist, axis=0)
    std_dist_per_frame  = np.nanstd (padded_dist, axis=0)
    mean_mse_per_frame  = np.nanmean(padded_mse,  axis=0)
    std_mse_per_frame   = np.nanstd (padded_mse,  axis=0)

    # ── Per-sample arrays over the eval slice ───────────────────────────
    sum_dist_arr        = np.array([s["sum_dist_eval"]    for s in sample_records], dtype=np.float64)
    sum_mse_arr         = np.array([s["sum_mse_eval"]     for s in sample_records], dtype=np.float64)
    mean_dist_eval_arr  = np.array([s["mean_dist_eval"]   for s in sample_records], dtype=np.float64)
    mean_mse_eval_arr   = np.array([s["mean_mse_eval"]    for s in sample_records], dtype=np.float64)
    final_frame_dist_arr= np.array([s["final_frame_dist"] for s in sample_records], dtype=np.float64)
    final_frame_mse_arr = np.array([s["final_frame_mse"]  for s in sample_records], dtype=np.float64)
    n_pen_arr           = np.array([s["n_pen_frames"]     for s in sample_records], dtype=np.float64)
    n_phantom_arr       = np.array([s["n_phantom_frames"] for s in sample_records], dtype=np.float64)
    max_pen_arr         = np.array([s["max_pen_depth"]    for s in sample_records], dtype=np.float64)
    max_phantom_arr     = np.array([s["max_phantom_dist"] for s in sample_records], dtype=np.float64)
    ddof = 1 if n_samples > 1 else 0
    eval_n_frames = int(max(0, max_T - args.start_eval_frame))

    def _mean_std(arr: np.ndarray) -> tuple[float, float]:
        return float(arr.mean()), float(arr.std(ddof=ddof))

    sum_dist_m,         sum_dist_s         = _mean_std(sum_dist_arr)
    sum_mse_m,          sum_mse_s          = _mean_std(sum_mse_arr)
    mean_dist_m,        mean_dist_s        = _mean_std(mean_dist_eval_arr)
    mean_mse_m,         mean_mse_s         = _mean_std(mean_mse_eval_arr)
    final_dist_m,       final_dist_s       = _mean_std(final_frame_dist_arr)
    final_mse_m,        final_mse_s        = _mean_std(final_frame_mse_arr)
    n_pen_m,            n_pen_s            = _mean_std(n_pen_arr)
    n_phantom_m,        n_phantom_s        = _mean_std(n_phantom_arr)
    max_pen_m,          max_pen_s          = _mean_std(max_pen_arr)
    max_phantom_m,      max_phantom_s      = _mean_std(max_phantom_arr)

    agg_stats = {
        "seed":                       args.seed,
        "num_samples":                n_samples,
        "context_len":                CONTEXT_FRAMES,
        "total_frames_requested":     args.total_frames,
        "aggregated_T":               int(max_T),
        "start_eval_frame":           int(args.start_eval_frame),
        "eval_n_frames":              eval_n_frames,
        "random_start":               bool(args.random_start),
        # Headline: per-sample sum over frames in [start_eval_frame:], then
        # mean ± std across the N sampled trajectories.
        "sum_dist_eval_mean":         sum_dist_m,
        "sum_dist_eval_std":          sum_dist_s,
        "sum_mse_eval_mean":          sum_mse_m,
        "sum_mse_eval_std":           sum_mse_s,
        # Same eval slice but reported as per-frame averages.
        "mean_dist_eval_mean":        mean_dist_m,
        "mean_dist_eval_std":         mean_dist_s,
        "mean_mse_eval_mean":         mean_mse_m,
        "mean_mse_eval_std":          mean_mse_s,
        # Final-frame error per sample → mean ± std across samples.
        "final_frame_dist_mean":      final_dist_m,
        "final_frame_dist_std":       final_dist_s,
        "final_frame_mse_mean":       final_mse_m,
        "final_frame_mse_std":        final_mse_s,
        # Full per-frame aggregated curves (length max_T). These IGNORE
        # start_eval_frame — slice them however you want offline.
        "mean_dist_per_frame":        mean_dist_per_frame.tolist(),
        "std_dist_per_frame":         std_dist_per_frame.tolist(),
        "mean_mse_per_frame":         mean_mse_per_frame.tolist(),
        "std_mse_per_frame":          std_mse_per_frame.tolist(),
        # Contact metrics (mean ± std across samples).
        "n_pen_frames_mean":          n_pen_m,
        "n_pen_frames_std":           n_pen_s,
        "n_phantom_frames_mean":      n_phantom_m,
        "n_phantom_frames_std":       n_phantom_s,
        "max_pen_depth_mean":         max_pen_m,
        "max_pen_depth_std":          max_pen_s,
        "max_phantom_dist_mean":      max_phantom_m,
        "max_phantom_dist_std":       max_phantom_s,
        "samples":                    sample_records,
    }

    print(f"\nAggregated stats over {n_samples} sample(s)"
          f"  (full horizon: {max_T} frames, eval slice: frames "
          f"{args.start_eval_frame}..{max_T - 1} = {eval_n_frames} frames, "
          f"random_start={args.random_start}):")
    print(f"  Σ-Dist over slice  (mean ± std) : {sum_dist_m:.6f} ± {sum_dist_s:.6f}")
    print(f"  Σ-MSE  over slice  (mean ± std) : {sum_mse_m:.6e} ± {sum_mse_s:.6e}")
    print(f"  Mean per-frame Dist (mean ± std): {mean_dist_m:.6f} ± {mean_dist_s:.6f}")
    print(f"  Mean per-frame MSE  (mean ± std): {mean_mse_m:.6e} ± {mean_mse_s:.6e}")
    print(f"  Final-frame Dist    (mean ± std): {final_dist_m:.6f} ± {final_dist_s:.6f}")
    print(f"  Final-frame MSE     (mean ± std): {final_mse_m:.6e} ± {final_mse_s:.6e}")
    print(f"  n_pen_frames        (mean ± std): {n_pen_m:.2f} ± {n_pen_s:.2f}")
    print(f"  n_phantom_frames    (mean ± std): {n_phantom_m:.2f} ± {n_phantom_s:.2f}")
    print(f"  max_pen_depth       (mean ± std): {max_pen_m:.4f} ± {max_pen_s:.4f} m")
    print(f"  max_phantom_dist    (mean ± std): {max_phantom_m:.4f} ± {max_phantom_s:.4f} m")

    # ── Per-frame breakdown over the eval slice (mean ± std across samples) ─
    print(f"\nPer-frame breakdown (frames {args.start_eval_frame}..{max_T - 1}):")
    print(f"  {'frame':>5}   {'Dist mean ± std':<27}    {'MSE mean ± std':<27}")
    print(f"  {'-' * 5}   {'-' * 27}    {'-' * 27}")
    for t in range(args.start_eval_frame, max_T):
        d_m, d_s = mean_dist_per_frame[t], std_dist_per_frame[t]
        m_m, m_s = mean_mse_per_frame [t], std_mse_per_frame [t]
        d_str = f"{d_m:.6f} ± {d_s:.6f}"
        m_str = f"{m_m:.4e} ± {m_s:.4e}"
        print(f"  {t:>5}   {d_str:<27}    {m_str:<27}")

    import json
    json_path = out / "agg_stats.json"
    with open(json_path, "w") as jf:
        json.dump(agg_stats, jf, indent=2)
    print(f"\n  Aggregated stats → {json_path}")

    save_dist_plot(
        mean_dist_per_frame, std_dist_per_frame,
        str(out / "dist_error_per_frame.png"),
        context_len=args.start_eval_frame, n_samples=n_samples,
    )

    print(f"\nAll outputs → {out}/")
    print("Done.")

if __name__ == "__main__":
    main()