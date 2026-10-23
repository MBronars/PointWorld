#!/usr/bin/env python3
"""
imagined_mppi.py

Receding-horizon MPPI inside the PointWorld world model. The world model is
the "real world" for the whole plan: it advances the imagined block point
cloud between replans, the gripper integrates kinematically. No simulator
is touched. The output is a saved action sequence the caller can replay in
SAPIEN (or any other env) later.

What's intentionally NOT here:
  * No SAPIEN / mani_skill / gymnasium / sapien imports.
  * No scene_normals — the WM was trained without them for this entry point.
  * No simulator playback. We save the actions; replay is the caller's job.

Action clamp: each per-step delta-xy is clamped to L2-norm ≤ --action_max_norm
(default 0.015 m, matching FaceApproachPushPolicy.MAX_STEP in the scripted
data-collection policy). Direction is preserved; only the magnitude is capped.

Initial state must be provided explicitly (this file has no env to sample
from). Either:
  --block_pos X Y Z --block_quat W X Y Z --gripper_xy X Y --target_xy X Y
or
  --state_npz path/to/state.npz   (with keys block_pos, block_quat,
                                   gripper_xy, target_xy)

Output (under --output_dir):
  actions.npz       Executed action sequence + imagined trajectory + metadata
  plan_log.json     Per-replan diagnostics (costs, plan times, H_active)

Usage:
    python imagined_mppi.py \
        --block_pos -0.135 0.0 0.026 --block_quat 1 0 0 0 \
        --gripper_xy -0.085 0.0 --target_xy -0.185 0.0 \
        --server_url http://127.0.0.1:5123 \
        --output_dir out/plan_001/ \
        --total_steps 80 --population_size 64 \
        --dynamic_horizon --horizon_base 10 --window_cap 3
"""

from __future__ import annotations

import argparse
import os
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import fpsample
import numpy as np
import requests


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# PointWorld training-time contract (must match server / checkpoint).
MODEL_T        = 11    # 1 context frame + 10 predicted frames
CONTEXT_FRAMES = 1
PRED_FRAMES    = 10

# Push env geometry — duplicated from envs/push_boundary.py so this file
# has no sapien-tainted imports. Bump in lockstep if you change the env.
BOUNDARY_CENTER_X = -0.135
BOUNDARY_CENTER_Y =  0.00
GRIPPER_Z_FIXED   =  0.085
CUBE_HALF         =  0.025

GRIPPER_RADIUS = 0.01
GRIPPER_LENGTH = 0.12

# Scripted policy speed cap (FaceApproachPushPolicy.MAX_STEP in
# Planning_wrapper/scripts/collect_demos_single.py). Per-step gripper delta
# magnitude. We clamp MPPI candidates to this so their speed distribution
# stays in the training distribution.
DEFAULT_ACTION_MAX_NORM = 0.015

_IDENTITY_QUAT_WXYZ     = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_DEFAULT_TARGET_REACHED = 0.015
_DEFAULT_SERVER_URL     = "http://127.0.0.1:5123"


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy helpers (copies of the versions in
# Planning_wrapper/scripts/live_worldmodel_v2.py — kept inline so this file
# has zero sim-side dependencies).
# ─────────────────────────────────────────────────────────────────────────────

def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).ravel()
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
        [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x)],
        [    2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def apply_pose(local_pts: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    R = _quat_wxyz_to_rot(quat_wxyz)
    return (local_pts.astype(np.float64) @ R.T
            + np.asarray(pos, np.float64)).astype(np.float32)


# NOTE: these MUST match the templates the PointWorld model was trained on
# (and that playback_pointworld_client.py + vis_push_pcd_floating.py use).
# Copied verbatim from Planning_wrapper/scripts/vis_push_pcd_floating.py so
# the inference-time geometry is byte-for-byte identical:
#   * 20x oversample, then FPS-decimate to N (uniform spatial coverage)
#   * cylinder INCLUDES end caps (not just the lateral surface)
# Earlier ad-hoc samplers (uniform-on-face cube + lateral-only cylinder, no
# FPS) produced a slightly out-of-distribution gripper geometry; the model
# returned mushy/sunburst predictions when fed those templates.

def _sample_box(offset: np.ndarray, half: np.ndarray,
                n: int, rng: np.random.Generator) -> np.ndarray:
    """N points uniformly on the surface of an axis-aligned box."""
    n_initial = n * 20
    hx, hy, hz = half
    areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
    counts = np.round(areas / areas.sum() * n_initial).astype(int)
    counts[-1] += n_initial - counts.sum()
    pts = []
    if counts[0] > 0:
        signs = rng.choice([-1.0, 1.0], counts[0])
        pts.append(np.stack([signs*hx,
                             rng.uniform(-hy, hy, counts[0]),
                             rng.uniform(-hz, hz, counts[0])], 1))
    if counts[1] > 0:
        signs = rng.choice([-1.0, 1.0], counts[1])
        pts.append(np.stack([rng.uniform(-hx, hx, counts[1]),
                             signs*hy,
                             rng.uniform(-hz, hz, counts[1])], 1))
    if counts[2] > 0:
        signs = rng.choice([-1.0, 1.0], counts[2])
        pts.append(np.stack([rng.uniform(-hx, hx, counts[2]),
                             rng.uniform(-hy, hy, counts[2]),
                             signs*hz], 1))
    pts_all = np.concatenate(pts).astype(np.float32) + offset.astype(np.float32)
    return pts_all[fpsample.fps_sampling(pts_all, n)]


def _sample_cylinder(radius: float, length: float,
                     n: int, rng: np.random.Generator) -> np.ndarray:
    """N points on the LATERAL surface and BOTH END CAPS of a Z-axis cylinder
    centred at the origin (z ∈ [-length/2, +length/2])."""
    n_initial = n * 20
    lat_a  = 2 * np.pi * radius * length
    cap_a  = 2 * np.pi * radius ** 2
    total  = lat_a + cap_a
    n_lat  = max(1, int(n_initial * lat_a / total)) if total > 0 else 0
    n_cap  = n_initial - n_lat
    n_top, n_bot = n_cap // 2, n_cap - n_cap // 2

    pts = []
    if n_lat > 0:
        th = rng.uniform(0, 2*np.pi, n_lat)
        z  = rng.uniform(-length/2, length/2, n_lat)
        pts.append(np.stack([radius*np.cos(th), radius*np.sin(th), z], 1))
    for nz, z0 in [(n_top, length/2), (n_bot, -length/2)]:
        if nz > 0:
            r  = radius * np.sqrt(rng.uniform(0, 1, nz))
            th = rng.uniform(0, 2*np.pi, nz)
            pts.append(np.stack([r*np.cos(th), r*np.sin(th), np.full(nz, z0)], 1))

    pts_all = np.concatenate(pts).astype(np.float32)
    return pts_all[fpsample.fps_sampling(pts_all, n)]


def _kabsch_yaw_2d(template_xy: np.ndarray, observed_xy: np.ndarray) -> float:
    """2D Kabsch — best yaw (rotation about Z) that maps `template_xy` to
    `observed_xy` in least-squares sense. Both inputs (N, 2); centroid is
    removed internally so the caller can pass world-frame xy directly.

    Used by the scripted proposer to recover the imagined block's current
    yaw from the WM-predicted point cloud (the WM rotates the cloud during
    pushes but we never tracked the rotation as a scalar — Kabsch reads it
    back from the points). Point ORDER must match between template and
    observed; the pointworld server's identity grid-sample patch preserves
    order, so this is exact.
    """
    A = np.asarray(template_xy, dtype=np.float64)
    B = np.asarray(observed_xy, dtype=np.float64)
    if A.shape != B.shape or A.shape[-1] != 2:
        raise ValueError(f"Kabsch needs matched (N, 2); got {A.shape} vs {B.shape}")
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    H = A.T @ B                                                     # (2, 2)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return float(np.arctan2(R[1, 0], R[0, 0]))


def clamp_action_norm(actions: np.ndarray, max_norm: float) -> np.ndarray:
    """Clamp per-action L2 norm to `max_norm`. Direction preserved.

    `actions` shape (..., 2). Each leaf 2-vector is independently scaled.
    Matches FaceApproachPushPolicy's speed cap: actions are produced as
    (unit_dir * MAX_STEP), so norm ≤ MAX_STEP everywhere along the demo
    distribution.
    """
    norms = np.linalg.norm(actions, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return (actions * scale).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ScriptedProposer — K independent FaceApproachPushPolicy rollouts as MPPI
# proposal source. Same idea as live_worldmodel_v2.ScriptedProposer; here we
# don't have a K-parallel sim, so each rollout uses block_xy held fixed at
# the current imagined value (policy's push-phase action only depends on
# yaw, so this is exact in push and approximate during approach when the
# block is already close to stationary anyway).
#
# Requires `FaceApproachPushPolicy` from
# Planning_wrapper/scripts/collect_demos_single.py, which transitively
# imports sapien. So this is only available when imagined_mppi.py is run
# in the `planning_wrapper` conda env. Lazy-imported so the rest of the
# file still loads under `pointworld-env`.
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedProposer:
    """K instances of FaceApproachPushPolicy, rolled out for H_active steps
    each with block_xy held fixed. Returns a (K, H_active, 2) action stack."""

    def __init__(self, K: int, base_seed: int,
                 hx: float = CUBE_HALF, hy: float = CUBE_HALF):
        try:
            import sys
            sys.path.insert(0, "/data/user_data/mbronars/packages/Planning_wrapper")
            sys.path.insert(0, "/data/user_data/mbronars/packages/Planning_wrapper/scripts")
            from collect_demos_single import FaceApproachPushPolicy
        except ImportError as e:
            raise SystemExit(
                "ScriptedProposer needs `FaceApproachPushPolicy` from "
                "Planning_wrapper/scripts/collect_demos_single.py (transitively "
                "imports sapien). Run imagined_mppi.py with the "
                "`planning_wrapper` conda env, e.g.\n\n  "
                "/data/user_data/mbronars/.miniconda3/envs/planning_wrapper/"
                "bin/python imagined_mppi.py --proposal_source scripted ...\n\n"
                f"original ImportError: {e}"
            )
        self.K = int(K)
        self.base_seed = int(base_seed)
        self.hx = float(hx)
        self.hy = float(hy)
        self.policies = [FaceApproachPushPolicy() for _ in range(self.K)]
        self._reset_count = 0

    def reset_all(self, gripper_xy: np.ndarray, block_xy: np.ndarray,
                  block_yaw: float) -> None:
        """Reset all K policies — each draws a uniform random face."""
        ss          = np.random.SeedSequence((self.base_seed, self._reset_count))
        child_seeds = ss.spawn(self.K)
        gxy = np.asarray(gripper_xy, dtype=np.float64).ravel()[:2]
        bxy = np.asarray(block_xy,   dtype=np.float64).ravel()[:2]
        yaw = float(block_yaw)
        for k, pol in enumerate(self.policies):
            pol.__init__()                              # reset internal FSM
            rng = np.random.default_rng(child_seeds[k])
            pol.reset(gxy, bxy, yaw, self.hx, self.hy, rng,
                       initial_face=None, obstacles=None)
        self._reset_count += 1

    def rollout(self, gripper_xy: np.ndarray, block_xy: np.ndarray,
                block_yaw: float, H: int) -> np.ndarray:
        """K rollouts × H steps. Block is held fixed at `block_xy`/`block_yaw`
        throughout (the policy's push-phase action ignores block_xy; approach
        phase sees a static obstacle to navigate around). Gripper xy
        integrates kinematically per K."""
        V = np.zeros((self.K, H, 2), dtype=np.float32)
        gxy_K = np.broadcast_to(
            np.asarray(gripper_xy, np.float64).ravel()[:2][None, :],
            (self.K, 2),
        ).copy()
        bxy = np.asarray(block_xy, np.float64).ravel()[:2]
        yaw = float(block_yaw)
        for t in range(H):
            for k in range(self.K):
                a = self.policies[k].act(gxy_K[k], bxy,
                                          block_yaw=yaw, obstacle_xys=None)
                V[k, t]   = a
                gxy_K[k]  = gxy_K[k] + a
        return V


# ─────────────────────────────────────────────────────────────────────────────
# WorldModelClient — thin HTTP wrapper over pointworld_server.py /predict.
# Same wire format as live_worldmodel_v2.py:WorldModelClient minus the
# scene_normals path.
# ─────────────────────────────────────────────────────────────────────────────

class WorldModelClient:
    _endpoint    = "/predict"
    _health_path = "/health"
    response_key = "scene_flows"

    def __init__(self, base_url: str = _DEFAULT_SERVER_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._session = requests.Session()

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}{self._health_path}", timeout=5.0)
            return r.ok and r.json().get("status") == "ok"
        except (requests.RequestException, ValueError):
            return False

    def wait_until_ready(self, max_wait_sec: float = 60.0, poll: float = 1.0) -> None:
        t0 = time.time()
        while time.time() - t0 < max_wait_sec:
            if self.health():
                return
            time.sleep(poll)
        raise RuntimeError(
            f"World-model server at {self.base_url} not ready within "
            f"{max_wait_sec:.0f}s. Did you launch pointworld_server.py?"
        )

    @staticmethod
    def _serialize_request(arrays: Dict[str, np.ndarray]) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        return buf.getvalue()

    @staticmethod
    def _parse_response(raw: bytes) -> Dict[str, np.ndarray]:
        with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    def predict_batch(self, windows: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """POST K windows in ONE request; returns (K, PRED_FRAMES, N, 3)."""
        if not windows:
            raise ValueError("predict_batch: empty window list")
        keys = list(windows[0].keys())
        batched = {k: np.stack([w[k] for w in windows], axis=0) for k in keys}
        body = self._serialize_request(batched)

        try:
            r = self._session.post(
                f"{self.base_url}{self._endpoint}",
                data=body,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"POST {self._endpoint} failed: {e}") from e
        if not r.ok:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            raise RuntimeError(f"Server error ({r.status_code}): {detail}")

        out = self._parse_response(r.content)
        if self.response_key not in out:
            raise RuntimeError(
                f"Response missing '{self.response_key}'. Got keys: {list(out)}"
            )
        pred = np.asarray(out[self.response_key], dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[None]
        if pred.ndim != 4:
            raise RuntimeError(f"Expected (K, T, N, 3) prediction, got {pred.shape}.")
        if pred.shape[0] != len(windows):
            raise RuntimeError(
                f"Server returned {pred.shape[0]} predictions for "
                f"{len(windows)} windows."
            )
        if pred.shape[1] < MODEL_T:
            raise RuntimeError(
                f"Prediction has {pred.shape[1]} frames, expected ≥ {MODEL_T}."
            )
        return pred[:, CONTEXT_FRAMES:CONTEXT_FRAMES + PRED_FRAMES]


# ─────────────────────────────────────────────────────────────────────────────
# Cost + MPPI config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CostConfig:
    w_goal:           float = 1.0
    sparse:           bool  = False
    sparse_epsilon:   float = _DEFAULT_TARGET_REACHED
    sparse_miss_cost: float = 1.0
    w_reach:          float = 0.0
    w_ctrl:           float = 0.0


def make_cost_fn(cfg: CostConfig) -> Callable:
    def cost_fn(pred_block_world: np.ndarray, gripper_traj_xy: np.ndarray,
                actions: np.ndarray, target_xy: np.ndarray) -> float:
        centroids_xy = pred_block_world.mean(axis=1)[:, :2]
        final_dist = float(np.linalg.norm(centroids_xy[-1] - target_xy))
        if cfg.sparse:
            cost = 0.0 if final_dist < cfg.sparse_epsilon else cfg.sparse_miss_cost
        else:
            cost = cfg.w_goal * final_dist
        if cfg.w_reach > 0.0:
            reach = float(np.linalg.norm(gripper_traj_xy - centroids_xy, axis=1).mean())
            cost += cfg.w_reach * reach
        if cfg.w_ctrl > 0.0:
            cost += cfg.w_ctrl * float((actions ** 2).sum())
        return cost
    return cost_fn


@dataclass
class PlannerConfig:
    population_size: int   = 64
    horizon:         int   = PRED_FRAMES
    replan_every:    int   = 5
    noise_sigma:     float = 0.01     # tuned for action_max_norm = 0.015
    temperature:     float = 0.3
    noise_beta:      float = 0.0
    keep_nominal:    bool  = True
    action_max_norm: float = DEFAULT_ACTION_MAX_NORM
    seed:            int   = 0
    select:          str   = "weighted"   # weighted = MPPI softmax average (original); argmin = execute the best candidate


# ─────────────────────────────────────────────────────────────────────────────
# ImaginedMPPIPlanner
# ─────────────────────────────────────────────────────────────────────────────

class ImaginedMPPIPlanner:
    """MPPI whose forward dynamics — and whose receding-horizon "real world" —
    is the PointWorld server. No sim is ever stepped. Per-step delta-xy
    actions are L2-norm-clamped to cfg.action_max_norm (default 0.015 m,
    matching the scripted data-collection policy's MAX_STEP).

    Imagined state advanced between replans:
        _imagined_block_pts   (N, 3) world-frame predicted block point cloud
        _imagined_gripper_xy  (2,)   integrated kinematically from actions

    The initial block_quat is cached (`_init_block_quat`) only so the saved
    actions.npz can be replayed in a sim later — the planner itself never
    reads it after reset(). Cost / replans look at the centroid of the
    point cloud, never an extracted rotation.
    """

    def __init__(self,
                 cfg:      PlannerConfig,
                 cost_cfg: CostConfig,
                 target_xy: np.ndarray,
                 client:   WorldModelClient,
                 total_steps: int,
                 n_block: int = 128,
                 n_hand:  int = 128,
                 dynamic_windows: bool  = False,
                 horizon_base:    int   = PRED_FRAMES,
                 window_cap:      int   = 4,
                 per_frame_dist:  float = 0.015,
                 dump_n_random_pcds: int = 0,
                 scripted_proposer: Optional["ScriptedProposer"] = None):
        self.cfg         = cfg
        self.cost_cfg    = cost_cfg
        self.target_xy   = np.asarray(target_xy, np.float32)[:2].copy()
        self.client      = client
        self.total_steps = int(total_steps)
        self.cost_fn     = make_cost_fn(cost_cfg)

        self.dynamic_windows = bool(dynamic_windows)
        self.horizon_base    = int(horizon_base)
        self.window_cap      = int(window_cap)
        self.per_frame_dist  = float(per_frame_dist)
        self._dump_n_random_pcds = int(dump_n_random_pcds)
        self.scripted_proposer = scripted_proposer
        if self.horizon_base % PRED_FRAMES != 0 or self.horizon_base < PRED_FRAMES:
            raise SystemExit(
                f"--horizon_base must be a positive multiple of PRED_FRAMES "
                f"(={PRED_FRAMES}); got {self.horizon_base}."
            )

        if self.dynamic_windows:
            self.H = self.window_cap * self.horizon_base
            if cfg.horizon != self.H:
                ar_per_step = self.horizon_base // PRED_FRAMES
                print(f"  note: --dynamic_horizon overriding --horizon "
                      f"{cfg.horizon} -> alloc {self.H} (cap={self.window_cap} × "
                      f"horizon_base={self.horizon_base}). Per-replan active "
                      f"prefix = N × {self.horizon_base}; each N step = "
                      f"{ar_per_step} AR model call(s) of {PRED_FRAMES} frames.")
        else:
            self.H = min(cfg.horizon, PRED_FRAMES)
            if cfg.horizon > PRED_FRAMES:
                print(f"  note: horizon clamped {cfg.horizon} → {PRED_FRAMES} "
                      f"(single-window static mode; pass --dynamic_horizon "
                      f"for multi-window AR rollout).")

        if cfg.noise_sigma > cfg.action_max_norm * 2.0:
            print(f"  note: noise_sigma={cfg.noise_sigma:.4f} is much larger "
                  f"than action_max_norm={cfg.action_max_norm:.4f} — most "
                  f"sampled actions will saturate the clamp.")

        self._rng        = np.random.default_rng(cfg.seed)
        # Use the EXACT sampling pipeline the model was trained on
        # (Planning_wrapper/scripts/vis_push_pcd_floating.sample_templates):
        # 20x oversample → FPS decimation. Cylinder includes both end caps.
        self.block_tpl   = _sample_box(
            np.zeros(3), np.full(3, CUBE_HALF), n_block, self._rng)
        self.hand_tpl    = _sample_cylinder(
            GRIPPER_RADIUS, GRIPPER_LENGTH, n_hand, self._rng)

        self.U = np.zeros((self.H, 2), dtype=np.float32)

        # Imagined state (populated in reset()).
        self._imagined_block_pts:   Optional[np.ndarray] = None
        self._imagined_block_pos:   Optional[np.ndarray] = None
        self._imagined_gripper_xy:  Optional[np.ndarray] = None
        self._imagined_step = 0

        # Initial state cached (saved with the action npz for replay).
        # block_quat is only needed for sim replay's set_pose call.
        self._init_block_pos:  Optional[np.ndarray] = None
        self._init_block_quat: Optional[np.ndarray] = None
        self._init_gripper_xy: Optional[np.ndarray] = None

        # Action sequence accumulator (the artifact this script produces).
        self.action_sequence: List[np.ndarray] = []

        # Per-replan predicted block point cloud, one frame per imagined step.
        # Variable N per frame is OK (PTV3 voxel pooling); flat list keeps it
        # shape-stable.
        self._pred_cache_frames: List[np.ndarray] = []
        self._pred_cache_idx:    int              = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self, block_pos: np.ndarray, block_quat: np.ndarray,
              gripper_xy: np.ndarray) -> None:
        bp  = np.asarray(block_pos,  np.float32).ravel()
        bq  = np.asarray(block_quat, np.float32).ravel()
        gxy = np.asarray(gripper_xy, np.float32).ravel()[:2]
        if bp.size != 3 or bq.size != 4 or gxy.size != 2:
            raise ValueError(
                f"reset shapes: block_pos (3,), block_quat (4,), "
                f"gripper_xy (2,); got {bp.shape}/{bq.shape}/{gxy.shape}"
            )

        self._init_block_pos  = bp.copy()
        self._init_block_quat = bq.copy()
        self._init_gripper_xy = gxy.copy()

        # Seed the imagined block point cloud at the spawn pose. After this,
        # the cloud evolves frame-by-frame from WM predictions — quat is
        # never referenced again.
        self._imagined_block_pts  = apply_pose(self.block_tpl, bp, bq).astype(np.float32)
        self._imagined_block_pos  = bp.copy()
        self._imagined_gripper_xy = gxy.copy()
        self._imagined_step       = 0

        self.action_sequence      = []
        self._pred_cache_frames   = []
        self._pred_cache_idx      = 0
        self.U[:]                 = 0.0

    # ── query construction ──────────────────────────────────────────────────

    def _hand_pts_world(self, gripper_xy_batch: np.ndarray) -> np.ndarray:
        K = gripper_xy_batch.shape[0]
        tcp_pos = np.concatenate(
            [gripper_xy_batch, np.full((K, 1), GRIPPER_Z_FIXED, dtype=np.float32)],
            axis=1,
        )
        return (self.hand_tpl[None, :, :].astype(np.float32)
                + tcp_pos[:, None, :])

    def _build_query(self, block_pts_world: np.ndarray,
                     gripper_traj_xy: np.ndarray,
                     shift: np.ndarray) -> Dict[str, np.ndarray]:
        """Assemble the raw npz arrays pointworld_server.py expects for one
        candidate. NO scene_normals."""
        T = MODEL_T
        N = block_pts_world.shape[0]
        block_c = (block_pts_world - shift).astype(np.float32)
        view0_scene_flows = np.broadcast_to(block_c, (T, N, 3)).astype(np.float32)
        visibility = np.ones((T, N), dtype=bool)
        joint_positions = (gripper_traj_xy
                           - np.array([BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y],
                                      dtype=np.float32)).astype(np.float32)
        tcp_pos = np.concatenate(
            [gripper_traj_xy, np.full((T, 1), GRIPPER_Z_FIXED, dtype=np.float32)],
            axis=1)
        right_gripper_pose = np.concatenate(
            [tcp_pos - shift, np.tile(_IDENTITY_QUAT_WXYZ, (T, 1))],
            axis=1).astype(np.float32)
        return {
            "view0_scene_flows":      view0_scene_flows,
            "view0_scene_visibility": visibility,
            "joint_positions":        joint_positions,
            "right_gripper_pose":     right_gripper_pose,
            "right_gripper_open":     np.zeros((T, 1), dtype=np.float32),
        }

    # ── noise sampling + horizon sizing ─────────────────────────────────────

    def _sample_noise(self, H_active: int) -> np.ndarray:
        cfg = self.cfg
        eps = self._rng.normal(0.0, cfg.noise_sigma,
                               size=(cfg.population_size, H_active, 2)).astype(np.float32)
        if cfg.noise_beta > 0.0:
            b = cfg.noise_beta
            for h in range(1, H_active):
                eps[:, h] = b * eps[:, h - 1] + (1.0 - b) * eps[:, h]
        return eps

    def _dynamic_N(self, block_xy: np.ndarray, gripper_xy: np.ndarray) -> int:
        travel = float(np.linalg.norm(gripper_xy - block_xy)
                       + np.linalg.norm(block_xy - self.target_xy))
        budget = max(self.horizon_base * self.per_frame_dist, 1e-6)
        N = int(np.ceil(travel / budget))
        return max(1, min(self.window_cap, N))

    # ── AR rollout (shared between K-scoring and the K=1 cache pass) ────────

    def _ar_rollout(self,
                    gripper_xy:     np.ndarray,
                    block_pts_w0:   np.ndarray,
                    V:              np.ndarray,
                    num_ar_windows: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """K-candidate AR rollout.

        Returns (list of (K, PRED_FRAMES, N_w, 3) per window, full_grip
        shape (K, 1 + num_ar_windows*PRED_FRAMES, 2)).
        """
        K        = V.shape[0]
        H_active = V.shape[1]
        H_padded = num_ar_windows * PRED_FRAMES

        # Pad to a full last window with zero deltas (gripper stays put).
        if H_padded > H_active:
            pad   = np.zeros((K, H_padded - H_active, 2), dtype=np.float32)
            V_pad = np.concatenate([V, pad], axis=1)
        else:
            V_pad = V

        cum          = np.cumsum(V_pad, axis=1)
        gripper_xy_K = np.broadcast_to(gripper_xy[None, :], (K, 2)).astype(np.float32)
        full_grip    = np.concatenate(
            [gripper_xy_K[:, None, :], gripper_xy_K[:, None, :] + cum], axis=1
        )

        ar_block_pts_K = np.broadcast_to(
            block_pts_w0[None, :, :], (K, block_pts_w0.shape[0], 3)
        ).astype(np.float32).copy()

        all_preds_world: List[np.ndarray] = []
        for w in range(num_ar_windows):
            win_grip   = full_grip[:, w * PRED_FRAMES : w * PRED_FRAMES + MODEL_T]
            hand_pts_K = self._hand_pts_world(win_grip[:, 0])
            combined   = np.concatenate([ar_block_pts_K, hand_pts_K], axis=1)
            shifts_K   = combined.mean(axis=1).astype(np.float32)

            windows = [self._build_query(ar_block_pts_K[k],
                                         win_grip[k], shifts_K[k])
                       for k in range(K)]
            preds_centred = self.client.predict_batch(windows)
            preds_world   = preds_centred + shifts_K[:, None, None, :]
            all_preds_world.append(preds_world.astype(np.float32))

            ar_block_pts_K = preds_world[:, -1].astype(np.float32)

        return all_preds_world, full_grip

    # ── one MPPI update ─────────────────────────────────────────────────────

    def get_actions(self) -> Tuple[np.ndarray, dict]:
        cfg = self.cfg
        K   = cfg.population_size
        block_pts_w0 = self._imagined_block_pts.astype(np.float32)
        block_pos    = self._imagined_block_pos.astype(np.float32)
        gripper_xy   = self._imagined_gripper_xy.astype(np.float32)

        # H_active cascade: (dynamic mode | static) → self.H → remaining budget.
        if self.dynamic_windows:
            N_active   = self._dynamic_N(block_pos[:2].astype(np.float32),
                                         gripper_xy.astype(np.float32))
            H_uncapped = N_active * self.horizon_base
        else:
            N_active   = 1
            H_uncapped = self.H

        remaining = max(1, self.total_steps - self._imagined_step)
        H_active  = min(self.H, H_uncapped, remaining)
        num_ar_windows = (H_active + PRED_FRAMES - 1) // PRED_FRAMES

        if self.dynamic_windows:
            print(f"  [dynamic windows] gripper→block="
                  f"{np.linalg.norm(gripper_xy - block_pos[:2]):.3f} m  "
                  f"block→target={np.linalg.norm(block_pos[:2] - self.target_xy):.3f} m  "
                  f"→ N={N_active} × base={self.horizon_base} = "
                  f"H_uncapped={H_uncapped}")
        if H_active < H_uncapped:
            print(f"  [horizon clip] step={self._imagined_step}/{self.total_steps} "
                  f"remaining={remaining}  H_uncapped={H_uncapped} → H_active={H_active} "
                  f"({num_ar_windows} AR window(s))")
        elif num_ar_windows > 1:
            print(f"  H_active={H_active}  →  {num_ar_windows} AR window(s) × {PRED_FRAMES}")

        # K candidate delta-xy sequences, clamped per-step to action_max_norm.
        if self.scripted_proposer is not None:
            # Estimate current block yaw by Kabsch-aligning the canonical
            # cube template to the imagined point cloud (the WM rotates the
            # cloud during pushes but we never track yaw as a scalar).
            yaw = _kabsch_yaw_2d(self.block_tpl[:, :2],
                                 self._imagined_block_pts[:, :2])
            self.scripted_proposer.reset_all(gripper_xy, block_pos[:2], yaw)
            V_raw = self.scripted_proposer.rollout(
                gripper_xy, block_pos[:2], yaw, H_active)
            V = clamp_action_norm(V_raw, cfg.action_max_norm)
            # No keep_nominal under scripted: candidates are structured face
            # pushes, not perturbations of the previous nominal.
        else:
            noise = self._sample_noise(H_active)
            V_raw = self.U[None, :H_active] + noise            # (K, H_active, 2)
            V     = clamp_action_norm(V_raw, cfg.action_max_norm)
            if cfg.keep_nominal:
                V[0] = clamp_action_norm(self.U[:H_active], cfg.action_max_norm)

        # K-candidate AR rollout.
        t0 = time.time()
        all_preds_world, full_grip = self._ar_rollout(
            gripper_xy, block_pts_w0, V, num_ar_windows,
        )
        plan_time = time.time() - t0

        centroids_per_w = [w.mean(axis=2).astype(np.float32) for w in all_preds_world]
        centroids_KH    = np.concatenate(centroids_per_w, axis=1)[:, :H_active]
        proxy           = centroids_KH[:, :, None, :]

        costs      = np.zeros(K, dtype=np.float32)
        pred_final = np.zeros((K, 2), dtype=np.float32)
        for k in range(K):
            gripper_pred_xy = full_grip[k, 1:1 + H_active]
            costs[k]        = self.cost_fn(proxy[k], gripper_pred_xy,
                                           V[k], self.target_xy)
            pred_final[k]   = centroids_KH[k, -1, :2]

        adv     = costs - costs.min()
        adv     = adv / (adv.std() + 1e-9)
        weights = np.exp(-adv / max(cfg.temperature, 1e-6))
        weights = weights / (weights.sum() + 1e-9)
        U_new   = (weights[:, None, None] * V).sum(axis=0)
        if cfg.select == "argmin":                                  # best-candidate execution (2026-09-25 option; default unchanged)
            U_new = V[int(np.argmin(costs))].copy()

        # Clamp the chosen update too — weighted average of in-disk vectors
        # is itself in the disk, but be defensive.
        self.U[:] = 0.0
        self.U[:H_active] = clamp_action_norm(U_new, cfg.action_max_norm)

        # K=1 AR chain for the chosen U_new → flat per-frame cache for step().
        if cfg.select == "argmin" and os.environ.get("PW_REUSE_WINNER"):   # 2026-10-09: the winner's frames are already in the K rollout; skip the K=1 re-rollout
            _kb = int(np.argmin(costs)); u_preds_world = [w[_kb:_kb + 1] for w in all_preds_world]
        else:
          u_preds_world, _ = self._ar_rollout(
            gripper_xy, block_pts_w0,
            self.U[None, :H_active],
            num_ar_windows,
          )
        cache_frames: List[np.ndarray] = []
        for w_preds in u_preds_world:
            for f in range(w_preds.shape[1]):
                cache_frames.append(w_preds[0, f].astype(np.float32))
        self._pred_cache_frames = cache_frames[:H_active]
        self._pred_cache_idx    = 0

        best_k = int(np.argmin(costs))

        # Chosen-U cached centroid trajectory (what step() will follow).
        chosen_centroid_xy = np.stack(
            [f.mean(axis=0)[:2] for f in self._pred_cache_frames], axis=0
        )                                                          # (H_active, 2)

        # Select winner + n_dump_random other candidates for full-pcd dump.
        n_random = int(self._dump_n_random_pcds)
        if n_random > 0:
            pool = [k for k in range(K) if k != best_k]
            picks = self._rng.choice(pool, size=min(n_random, len(pool)),
                                      replace=False)
            selected_k = np.concatenate([[best_k], picks]).astype(np.int64)
        else:
            selected_k = np.array([best_k], dtype=np.int64)
        # Selected candidates' per-window predicted block point clouds.
        # all_preds_world[w] has shape (K, PRED_FRAMES, N_w, 3).
        selected_pcds_per_window = [all_preds_world[w][selected_k]
                                    for w in range(num_ar_windows)]
        npts_per_window = np.asarray(
            [all_preds_world[w].shape[2] for w in range(num_ar_windows)],
            dtype=np.int32,
        )

        info = {
            "costs":           costs,
            "weights":         weights,
            "pred_final_xy":   pred_final,
            "best_k":          best_k,
            "best_cost":       float(costs[best_k]),
            "mean_cost":       float(costs.mean()),
            "plan_time_sec":   float(plan_time),
            "H_active":        int(H_active),
            "n_ar_windows":    int(num_ar_windows),
            # Intermediate-prediction artifacts (saved per replan if dump dir set).
            "candidate_centroids_xy": centroids_KH[..., :2].astype(np.float32),  # (K, H_active, 2)
            "candidate_gripper_xy":   full_grip[:, 1:1 + H_active, :].astype(np.float32),  # (K, H_active, 2)
            "chosen_centroid_xy":     chosen_centroid_xy.astype(np.float32),  # (H_active, 2)
            "chosen_actions":         self.U[:H_active].copy(),
            "block_xy_at_replan":     block_pos[:2].astype(np.float32),
            "gripper_xy_at_replan":   gripper_xy.astype(np.float32),
            # Full predicted block point clouds for the selected candidates
            # (winner + n_dump_random). Each entry is (n_selected, PRED_FRAMES, N_w, 3);
            # N_w varies per window due to PTV3 voxel pooling.
            "selected_pcds_per_window": selected_pcds_per_window,
            "selected_k_indices":       selected_k,
            "selected_costs":           costs[selected_k].astype(np.float32),
            "selected_gripper_xy":      full_grip[selected_k, 1:1 + H_active, :].astype(np.float32),
            "npts_per_window":          npts_per_window,
        }
        return self.U[:H_active].copy(), info

    # ── stepping the imagined world ─────────────────────────────────────────

    def step(self, action: np.ndarray) -> None:
        if self._pred_cache_idx >= len(self._pred_cache_frames):
            raise RuntimeError(
                "step() called without a fresh pred cache (call get_actions first, "
                "and keep k_exec <= H_active)."
            )
        a = clamp_action_norm(
            np.asarray(action, np.float32).ravel()[:2], self.cfg.action_max_norm)
        self._imagined_gripper_xy = (self._imagined_gripper_xy + a).astype(np.float32)
        self._imagined_block_pts  = self._pred_cache_frames[self._pred_cache_idx]
        self._imagined_block_pos  = self._imagined_block_pts.mean(axis=0).astype(np.float32)
        self._pred_cache_idx += 1
        self._imagined_step  += 1
        self.action_sequence.append(a.copy())

    def advance(self, k_executed: int) -> None:
        k = min(k_executed, self.H)
        if k < self.H:
            self.U[:-k] = self.U[k:]
            self.U[-k:] = 0.0
        else:
            self.U[:] = 0.0
        self._pred_cache_frames = []
        self._pred_cache_idx    = 0


# ─────────────────────────────────────────────────────────────────────────────
# MPC driver
# ─────────────────────────────────────────────────────────────────────────────

def _dump_replan_npz(save_path: Path,
                     step: int,
                     info: dict,
                     target_xy: np.ndarray) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        step                    = np.int32(step),
        costs                   = info["costs"].astype(np.float32),
        weights                 = info["weights"].astype(np.float32),
        candidate_centroids_xy  = info["candidate_centroids_xy"],
        candidate_gripper_xy    = info["candidate_gripper_xy"],
        chosen_centroid_xy      = info["chosen_centroid_xy"],
        chosen_actions          = info["chosen_actions"],
        block_xy_at_replan      = info["block_xy_at_replan"],
        gripper_xy_at_replan    = info["gripper_xy_at_replan"],
        target_xy               = np.asarray(target_xy, np.float32),
        plan_time_sec           = np.float32(info["plan_time_sec"]),
        best_cost               = np.float32(info["best_cost"]),
        mean_cost               = np.float32(info["mean_cost"]),
        H_active                = np.int32(info["H_active"]),
        n_ar_windows            = np.int32(info["n_ar_windows"]),
    )
    if "selected_pcds_per_window" in info and info["selected_pcds_per_window"]:
        payload["selected_k_indices"] = info["selected_k_indices"].astype(np.int64)
        payload["selected_costs"]     = info["selected_costs"]
        payload["selected_gripper_xy"] = info["selected_gripper_xy"]
        payload["npts_per_window"]    = info["npts_per_window"]
        # N_w varies per window → save each window as its own key (n_selected, PRED_FRAMES, N_w, 3).
        for w, arr in enumerate(info["selected_pcds_per_window"]):
            payload[f"selected_pcds_w{w}"] = arr.astype(np.float32)
    np.savez(save_path, **payload)


def imagined_run_loop(planner: ImaginedMPPIPlanner,
                      target_xy: np.ndarray,
                      total_steps: int,
                      target_reached_threshold: float,
                      verbose: bool = True,
                      replan_dump_dir: Optional[Path] = None) -> dict:
    block_pos  = planner._imagined_block_pos.astype(np.float32)
    gripper_xy = planner._imagined_gripper_xy.astype(np.float32)
    target     = np.asarray(target_xy, np.float32)[:2].copy()
    replan_every = planner.cfg.replan_every

    log = dict(
        block_xy   = [block_pos[:2].copy()],
        gripper_xy = [gripper_xy.copy()],
        block_pts  = [planner._imagined_block_pts.astype(np.float32).copy()],   # imagined 3-D block cloud per executed frame (2026-09-25)
        actions    = [],
        replan_steps = [], plan_times = [], best_costs = [], mean_costs = [],
        H_active   = [], n_ar_windows = [],
    )

    step, n_replans = 0, 0
    while step < total_steps:
        dist = float(np.linalg.norm(block_pos[:2] - target))
        if dist < target_reached_threshold:
            if verbose:
                print(f"[step {step}] Target reached — block→target {dist*100:.2f} cm")
            break

        if verbose:
            print(f"\n[step {step}] Replan #{n_replans}  block→target: {dist*100:.2f} cm")
        try:
            U, info = planner.get_actions()
        except Exception as e:
            print(f"[error] get_actions failed at step {step}: {e}")
            break

        log['replan_steps'].append(step)
        log['plan_times'].append(info['plan_time_sec'])
        log['best_costs'].append(info['best_cost'])
        log['mean_costs'].append(info['mean_cost'])
        log['H_active'].append(info['H_active'])
        log['n_ar_windows'].append(info['n_ar_windows'])
        if replan_dump_dir is not None:
            _dump_replan_npz(replan_dump_dir / f"replan_{n_replans:03d}.npz",
                             step, info, target)
        if verbose:
            print(f"  plan time: {info['plan_time_sec']:.3f}s  "
                  f"best: {info['best_cost']:.4f}  mean: {info['mean_cost']:.4f}  "
                  f"H_active: {info['H_active']} ({info['n_ar_windows']} window(s))")

        k_exec = min(replan_every, total_steps - step, len(U))
        if verbose:
            print(f"  executing {k_exec} actions")

        reached = False
        k_actually_executed = 0
        for k in range(k_exec):
            action = np.asarray(U[k], np.float32)
            planner.step(action)
            block_pos  = planner._imagined_block_pos.astype(np.float32)
            gripper_xy = planner._imagined_gripper_xy.astype(np.float32)
            log['block_xy'].append(block_pos[:2].copy())
            log['gripper_xy'].append(gripper_xy.copy())
            log['block_pts'].append(planner._imagined_block_pts.astype(np.float32).copy())
            log['actions'].append(action.copy())
            step += 1
            k_actually_executed = k + 1
            if float(np.linalg.norm(block_pos[:2] - target)) < target_reached_threshold:
                if verbose:
                    print(f"  [step {step}] Target reached mid-execution")
                reached = True
                break

        planner.advance(k_actually_executed)
        n_replans += 1
        if reached:
            break

    final_dist = float(np.linalg.norm(block_pos[:2] - target))
    log['target_xy']  = target
    log['final_dist'] = final_dist
    log['success']    = bool(final_dist < target_reached_threshold)
    log['n_replans']  = n_replans
    log['n_steps']    = step

    print(f"\n{'='*60}")
    print(f"  Imagined MPC complete")
    print(f"  Final block→target distance: {final_dist*100:.2f} cm")
    print(f"  Replans                    : {n_replans}")
    print(f"  Actions saved              : {step}")
    print(f"  Imagined success           : "
          f"{'✓' if final_dist < target_reached_threshold else '✗'}")
    if log['plan_times']:
        print(f"  Mean plan time             : {np.mean(log['plan_times']):.3f} s")
    print(f"{'='*60}\n")
    return log


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_actions_npz(planner: ImaginedMPPIPlanner, log: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        # The artifact: the open-loop action sequence to replay in the sim.
        actions = (np.asarray(log['actions'], np.float32)
                   if log['actions'] else np.zeros((0, 2), np.float32)),
        # Initial state needed by the replayer to set the spawn.
        init_block_pos  = planner._init_block_pos.astype(np.float32),
        init_block_quat = planner._init_block_quat.astype(np.float32),
        init_gripper_xy = planner._init_gripper_xy.astype(np.float32),
        target_xy       = np.asarray(log['target_xy'], np.float32),
        # Imagined trajectory (for visualisation / sanity-check vs sim replay).
        imagined_block_xy   = np.asarray(log['block_xy'],   np.float32),
        imagined_gripper_xy = np.asarray(log['gripper_xy'], np.float32),
        imagined_block_pts  = np.stack(log['block_pts']).astype(np.float32),   # [T+1, N_block, 3] world frame, frame 0 = spawn
        # Per-replan diagnostics.
        replan_steps  = np.asarray(log['replan_steps'],  np.int32),
        plan_times    = np.asarray(log['plan_times'],    np.float32),
        best_costs    = np.asarray(log['best_costs'],    np.float32),
        mean_costs    = np.asarray(log['mean_costs'],    np.float32),
        H_active      = np.asarray(log['H_active'],      np.int32),
        n_ar_windows  = np.asarray(log['n_ar_windows'],  np.int32),
        # Summary.
        final_dist      = np.float32(log['final_dist']),
        success         = np.int8(log['success']),
        n_replans       = np.int32(log['n_replans']),
        n_steps         = np.int32(log['n_steps']),
        wall_time_sec   = np.float32(log.get('wall_time_sec', 0.0)),
        action_max_norm = np.float32(planner.cfg.action_max_norm),
    )
    print(f"  Saved actions + trajectory: {save_path}")


def save_plan_log_json(log: dict, planner: ImaginedMPPIPlanner, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_steps":   int(log['n_steps']),
        "n_replans": int(log['n_replans']),
        "final_dist_m": float(log['final_dist']),
        "success":   bool(log['success']),
        "target_xy": [float(log['target_xy'][0]), float(log['target_xy'][1])],
        "init_block_pos":  [float(x) for x in planner._init_block_pos],
        "init_block_quat": [float(x) for x in planner._init_block_quat],
        "init_gripper_xy": [float(x) for x in planner._init_gripper_xy],
        "action_max_norm": float(planner.cfg.action_max_norm),
        "per_replan": [
            {
                "step":         int(log['replan_steps'][i]),
                "plan_time_s":  float(log['plan_times'][i]),
                "best_cost":    float(log['best_costs'][i]),
                "mean_cost":    float(log['mean_costs'][i]),
                "H_active":     int(log['H_active'][i]),
                "n_ar_windows": int(log['n_ar_windows'][i]),
            }
            for i in range(len(log['replan_steps']))
        ],
    }
    save_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved plan log:            {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Initial-state loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_state_from_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        out = {k: npz[k] for k in ("block_pos", "block_quat",
                                    "gripper_xy", "target_xy")}
    return out


def resolve_initial_state(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve (block_pos, block_quat, gripper_xy, target_xy) from CLI args
    and/or --state_npz. CLI args override the npz file when both are given."""
    state: Dict[str, np.ndarray] = {}
    if args.state_npz:
        state.update(load_state_from_npz(Path(args.state_npz)))

    if args.block_pos is not None:
        state["block_pos"]  = np.asarray(args.block_pos,  np.float32)
    if args.block_quat is not None:
        state["block_quat"] = np.asarray(args.block_quat, np.float32)
    if args.gripper_xy is not None:
        state["gripper_xy"] = np.asarray(args.gripper_xy, np.float32)
    if args.target_xy is not None:
        state["target_xy"]  = np.asarray(args.target_xy,  np.float32)

    missing = [k for k in ("block_pos", "block_quat",
                            "gripper_xy", "target_xy") if k not in state]
    if missing:
        raise SystemExit(
            f"Missing initial-state values: {missing}. Provide them via CLI "
            f"flags (--block_pos / --block_quat / --gripper_xy / --target_xy) "
            f"or pass --state_npz path/to/state.npz."
        )
    return (np.asarray(state["block_pos"],  np.float32).ravel(),
            np.asarray(state["block_quat"], np.float32).ravel(),
            np.asarray(state["gripper_xy"], np.float32).ravel()[:2],
            np.asarray(state["target_xy"],  np.float32).ravel()[:2])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Receding-horizon MPPI inside the PointWorld world model. "
                    "Saves only the action sequence; no sim is touched.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Initial state (explicit — no sim sampling).
    s = p.add_argument_group("initial state (CLI overrides --state_npz)")
    s.add_argument("--state_npz",  type=str,   default=None,
                   help="Path to npz with keys block_pos, block_quat, "
                        "gripper_xy, target_xy.")
    s.add_argument("--block_pos",  type=float, nargs=3, default=None)
    s.add_argument("--block_quat", type=float, nargs=4, default=None,
                   help="wxyz")
    s.add_argument("--gripper_xy", type=float, nargs=2, default=None)
    s.add_argument("--target_xy",  type=float, nargs=2, default=None)
    s.add_argument("--target_threshold", type=float, default=_DEFAULT_TARGET_REACHED)

    # MPC loop
    p.add_argument("--total_steps",  type=int, default=80,
                   help="Max actions to plan. H_active per replan is "
                        "clipped to total_steps - step.")
    p.add_argument("--replan_every", type=int, default=5)

    # MPPI
    m = p.add_argument_group("MPPI")
    m.add_argument("--population_size", type=int, default=64)
    m.add_argument("--horizon",         type=int, default=PRED_FRAMES,
                   help=f"Static-mode planning horizon (clamped to ≤ "
                        f"PRED_FRAMES={PRED_FRAMES}). Use --dynamic_horizon "
                        f"for multi-window AR.")
    m.add_argument("--noise_sigma",     type=float, default=0.01,
                   help="Gaussian noise std. Default tuned for "
                        "action_max_norm=0.015; raise too far and most "
                        "candidates saturate the L2 clamp.")
    m.add_argument("--temperature",     type=float, default=0.3)
    m.add_argument("--select",          type=str, default="weighted", choices=("weighted", "argmin"),
                   help="weighted: execute the MPPI softmax average (original); argmin: execute the lowest-cost candidate")
    m.add_argument("--noise_beta",      type=float, default=0.0)
    m.add_argument("--no_keep_nominal", action="store_true")
    m.add_argument("--action_max_norm", type=float, default=DEFAULT_ACTION_MAX_NORM,
                   help="Per-step delta-xy L2-norm cap. Matches the scripted "
                        "policy's MAX_STEP — keeps proposal speeds inside "
                        "the training distribution.")
    m.add_argument("--proposal_source", type=str, default="noise",
                   choices=["noise", "scripted"],
                   help="Where MPPI candidates come from. 'noise' = Gaussian "
                        "around the previous nominal (default). 'scripted' = "
                        "K independent FaceApproachPushPolicy rollouts (the "
                        "scripted demo policy from collect_demos_single.py); "
                        "requires the `planning_wrapper` conda env because "
                        "the policy class transitively imports sapien.")

    # Multi-window AR
    d = p.add_argument_group("multi-window AR")
    d.add_argument("--dynamic_horizon", action="store_true",
                   help="Per-replan, size rollout as N × horizon_base frames "
                        "with N from gripper→block→target travel distance "
                        "(clamped to --window_cap).")
    d.add_argument("--horizon_base", type=int, default=PRED_FRAMES,
                   help=f"Must be a positive multiple of PRED_FRAMES (={PRED_FRAMES}).")
    d.add_argument("--window_cap", type=int, default=4)
    d.add_argument("--per_frame_dist", type=float, default=0.015,
                   help="Estimated per-frame gripper travel (m).")

    # Cost
    c = p.add_argument_group("cost (minimised)")
    c.add_argument("--w_goal",         type=float, default=1.0)
    c.add_argument("--sparse",         action="store_true")
    c.add_argument("--sparse_epsilon", type=float, default=_DEFAULT_TARGET_REACHED)
    c.add_argument("--w_reach",        type=float, default=0.0)
    c.add_argument("--w_ctrl",         type=float, default=0.0)

    # Server
    w = p.add_argument_group("world-model server")
    w.add_argument("--server_url",     type=str,   default=_DEFAULT_SERVER_URL)
    w.add_argument("--server_timeout", type=float, default=120.0)
    w.add_argument("--wait_for_server", type=float, default=0.0)
    w.add_argument("--response_key",   type=str,   default="scene_flows")
    w.add_argument("--n_block", type=int, default=128)
    w.add_argument("--n_hand",  type=int, default=128)

    # Output
    p.add_argument("--output_dir", type=str, default="./mpc_plan")
    p.add_argument("--dump_replans", action="store_true",
                   help="Save per-replan npz under <output_dir>/replans/ "
                        "with K-candidate centroid trajectories, costs, "
                        "weights, and the chosen U's predicted trajectory.")
    p.add_argument("--dump_n_random_pcds", type=int, default=0,
                   help="If >0, per replan dump the full predicted block "
                        "point clouds for the WINNER plus N random other "
                        "candidates (under <output_dir>/replans/). One key "
                        "per AR window since N_w varies per window.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed = (args.seed if args.seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0]))
    np.random.seed(seed)

    block_pos, block_quat, gripper_xy, target_xy = resolve_initial_state(args)

    cfg = PlannerConfig(
        population_size = args.population_size,
        horizon         = args.horizon,
        replan_every    = args.replan_every,
        noise_sigma     = args.noise_sigma,
        temperature     = args.temperature,
        noise_beta      = args.noise_beta,
        keep_nominal    = not args.no_keep_nominal,
        action_max_norm = args.action_max_norm,
        seed            = seed,
        select          = args.select,
    )
    cost_cfg = CostConfig(
        w_goal         = args.w_goal,
        sparse         = args.sparse,
        sparse_epsilon = args.sparse_epsilon,
        w_reach        = args.w_reach,
        w_ctrl         = args.w_ctrl,
    )

    client = WorldModelClient(args.server_url, timeout=args.server_timeout)
    client.response_key = args.response_key
    if args.wait_for_server > 0:
        print(f"Waiting up to {args.wait_for_server:.0f}s for server "
              f"at {args.server_url} …")
        client.wait_until_ready(max_wait_sec=args.wait_for_server)
    if not client.health():
        raise SystemExit(
            f"World-model server at {args.server_url} is not reachable. "
            f"Launch pointworld_server.py first."
        )

    proposer = None
    if args.proposal_source == "scripted":
        proposer = ScriptedProposer(K=cfg.population_size, base_seed=seed)
        print(f"  proposal source : scripted "
              f"(K={cfg.population_size} FaceApproachPushPolicy rollouts)")
    else:
        print(f"  proposal source : noise (Gaussian sigma={cfg.noise_sigma})")

    planner = ImaginedMPPIPlanner(
        cfg, cost_cfg, target_xy, client,
        total_steps     = args.total_steps,
        n_block         = args.n_block,
        n_hand          = args.n_hand,
        dynamic_windows = args.dynamic_horizon,
        horizon_base    = args.horizon_base,
        window_cap      = args.window_cap,
        per_frame_dist  = args.per_frame_dist,
        dump_n_random_pcds = args.dump_n_random_pcds,
        scripted_proposer  = proposer,
    )

    print(f"Imagined MPPI (no sim)")
    print(f"  population (K)   : {cfg.population_size}")
    print(f"  horizon (H)      : {planner.H}  replan every: {cfg.replan_every}  "
          f"total {args.total_steps} steps")
    print(f"  action L2 cap    : {cfg.action_max_norm:.4f} m  "
          f"(scripted policy MAX_STEP={DEFAULT_ACTION_MAX_NORM:.4f})")
    print(f"  noise_sigma      : {cfg.noise_sigma}   temperature: {cfg.temperature}")
    print(f"  cost             : "
          + (f"sparse hit (eps={cost_cfg.sparse_epsilon:.3f} m)" if cost_cfg.sparse
             else f"terminal distance-to-goal (w={cost_cfg.w_goal})")
          + (f"  + reach (w={cost_cfg.w_reach})" if cost_cfg.w_reach > 0 else "")
          + (f"  + ctrl  (w={cost_cfg.w_ctrl})" if cost_cfg.w_ctrl > 0 else ""))
    print(f"  seed             : {seed}"
          f"{'  (random — pass --seed %d to reproduce)' % seed if args.seed is None else ''}")
    print(f"  spawn block pos  : {block_pos}")
    print(f"  spawn block quat : {block_quat}")
    print(f"  spawn gripper xy : {gripper_xy}")
    print(f"  target xy        : {target_xy}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    planner.reset(block_pos, block_quat, gripper_xy)

    replan_dump_dir = (output_dir / "replans") if args.dump_replans else None

    t0 = time.time()
    log = imagined_run_loop(
        planner                  = planner,
        target_xy                = target_xy,
        total_steps              = args.total_steps,
        target_reached_threshold = args.target_threshold,
        verbose                  = not args.quiet,
        replan_dump_dir          = replan_dump_dir,
    )
    wall = time.time() - t0

    log['wall_time_sec'] = wall

    plan_times = np.asarray(log['plan_times'], np.float64)
    print(f"\n[timing]")
    print(f"  total wall                   : {wall:.2f} s")
    print(f"  planning (sum of replans)    : {plan_times.sum():.2f} s "
          f"({100*plan_times.sum()/max(wall,1e-9):.1f}% of wall)")
    if plan_times.size:
        print(f"  per-replan plan time         : "
              f"mean={plan_times.mean():.3f}s  std={plan_times.std():.3f}s  "
              f"min={plan_times.min():.3f}s  max={plan_times.max():.3f}s")
        print(f"  per-imagined-step amortized  : "
              f"{wall / max(log['n_steps'], 1):.3f} s/step")

    save_actions_npz(planner, log, output_dir / "actions.npz")
    save_plan_log_json(log, planner, output_dir / "plan_log.json")

    print(f"\nAll outputs saved under: {output_dir}")
    print(f"Replay the saved actions in the sim with whatever harness you "
          f"prefer — the file {output_dir/'actions.npz'} contains the action "
          f"sequence and the initial state needed to reset the env.")


if __name__ == "__main__":
    main()
