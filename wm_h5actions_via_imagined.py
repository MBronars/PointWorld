#!/usr/bin/env python3
"""
wm_h5actions_via_imagined.py

Isolation test: take an h5 demo's initial state AND its recorded delta-xy
actions, build the WM request using EXACTLY the same code path
imagined_mppi.py uses (its `_build_query` + `_hand_pts_world` + the
imagined-state templates), call the server in autoregressive windows,
and render the predicted point cloud.

If predictions here look as clean as wm_predict_gif.py, the issue is in
imagined_mppi.py's MPPI / scripted-proposer rollout. If predictions
here look mushy, the bug is in imagined_mppi's query construction.

Requires the `planning_wrapper` conda env (h5py + sapien-tainted helpers).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

# Drop PointWorld/ from sys.path so its utils.py doesn't shadow
# Planning_wrapper/utils/ when we import the playback client below.
_PW_ROOT  = "/data/user_data/mbronars/packages/Planning_wrapper"
_HERE_DIR = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if Path(p).resolve() != Path(_HERE_DIR).resolve()]
sys.path.insert(0, _PW_ROOT)
sys.path.insert(0, f"{_PW_ROOT}/scripts")
sys.path.insert(0, _HERE_DIR)            # re-add for imagined_mppi import
os.chdir(_PW_ROOT)

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pull *imagined_mppi*'s own helpers — same templates, same _build_query
# semantics — so this script is a faithful "playback through imagined
# inference" rather than the wm_predict_gif h5 path.
from imagined_mppi import (
    _sample_box, _sample_cylinder, apply_pose,
    MODEL_T, PRED_FRAMES,
    BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y,
    GRIPPER_Z_FIXED, CUBE_HALF,
    GRIPPER_RADIUS, GRIPPER_LENGTH,
    _IDENTITY_QUAT_WXYZ,
)

# Use the same HTTP client wire format imagined_mppi uses.
from imagined_mppi import WorldModelClient

# h5-side helpers (for extracting initial state + actions).
from vis_push_pcd_floating import (
    ACTOR_POS, ACTOR_QUAT, FLOAT_QPOS, FLOAT_ROOT_POS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Imagined-mppi-style query construction (copied to keep this script standalone)
# ─────────────────────────────────────────────────────────────────────────────

def _hand_pts_world(hand_tpl: np.ndarray,
                    gripper_xy_batch: np.ndarray) -> np.ndarray:
    """(K, 2) gripper xy → (K, N_hand, 3) world-frame hand cylinder points,
    using identity TCP rotation + GRIPPER_Z_FIXED. Matches
    ImaginedMPPIPlanner._hand_pts_world."""
    K = gripper_xy_batch.shape[0]
    tcp_pos = np.concatenate(
        [gripper_xy_batch, np.full((K, 1), GRIPPER_Z_FIXED, dtype=np.float32)],
        axis=1,
    )
    return (hand_tpl[None, :, :].astype(np.float32)
            + tcp_pos[:, None, :])


def _build_query(block_pts_world: np.ndarray,
                 gripper_traj_xy: np.ndarray,
                 shift: np.ndarray) -> dict:
    """Mirror ImaginedMPPIPlanner._build_query verbatim (without normals)."""
    T = MODEL_T
    N = block_pts_world.shape[0]
    block_c = (block_pts_world - shift).astype(np.float32)
    view0_scene_flows = np.broadcast_to(block_c, (T, N, 3)).astype(np.float32)
    visibility        = np.ones((T, N), dtype=bool)
    joint_positions   = (gripper_traj_xy
                         - np.array([BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y],
                                    dtype=np.float32)).astype(np.float32)
    tcp_pos = np.concatenate(
        [gripper_traj_xy, np.full((T, 1), GRIPPER_Z_FIXED, dtype=np.float32)],
        axis=1,
    )
    right_gripper_pose = np.concatenate(
        [tcp_pos - shift, np.tile(_IDENTITY_QUAT_WXYZ, (T, 1))],
        axis=1,
    ).astype(np.float32)
    return {
        "view0_scene_flows":      view0_scene_flows,
        "view0_scene_visibility": visibility,
        "joint_positions":        joint_positions,
        "right_gripper_pose":     right_gripper_pose,
        "right_gripper_open":     np.zeros((T, 1), dtype=np.float32),
    }


def _candidate_gripper_traj(gripper_xy: np.ndarray,
                            deltas: np.ndarray) -> np.ndarray:
    """(H, 2) deltas → (MODEL_T, 2) absolute gripper xy; pad if H < MODEL_T - 1."""
    future = gripper_xy[None, :] + np.cumsum(deltas, axis=0)
    traj   = np.concatenate([gripper_xy[None, :], future], axis=0)
    if traj.shape[0] < MODEL_T:
        pad  = np.repeat(traj[-1:], MODEL_T - traj.shape[0], axis=0)
        traj = np.concatenate([traj, pad], axis=0)
    return traj[:MODEL_T].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# h5 loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_h5_state_and_actions(h5_path: str, traj_key: str,
                              start: int, total_steps: int) -> dict:
    """Read frame `start` block + TCP state and the next `total_steps`
    delta-xy actions. Mirrors extract_clip's lag-skip (the `[1::]` slice)."""
    with h5py.File(h5_path, "r") as f:
        traj = f[traj_key]
        art_keys   = list(traj["env_states/articulations"].keys())
        actor_keys = list(traj["env_states/actors"].keys())
        art_key = next(
            (k for c in ["floating_gripper", "robot", "panda_stick", "gripper"]
             for k in art_keys if c in k.lower()),
            art_keys[0],
        )
        block_actor_key = next(
            (k for c in ["push_block", "block", "cube"]
             for k in actor_keys if c in k.lower()),
            actor_keys[0],
        )
        block_states = traj[f"env_states/actors/{block_actor_key}"][:]
        art_states   = traj[f"env_states/articulations/{art_key}"][:]

    block_states = block_states[1:]
    art_states   = art_states[1:]
    need = total_steps + 1                                # 1 init + N actions
    if start + need > block_states.shape[0]:
        raise SystemExit(
            f"requested {need} frames from start={start} but only "
            f"{block_states.shape[0]} available"
        )
    bs = block_states[start : start + need]
    art = art_states[start : start + need]

    root = art[:, FLOAT_ROOT_POS].copy()
    jxy  = art[:, FLOAT_QPOS]
    root[:, 0] += jxy[:, 0]
    root[:, 1] += jxy[:, 1]
    tcp_xy = root[:, :2].astype(np.float32)               # (need, 2)

    # Delta-xy actions from successive TCP positions (= what the demo actually did).
    actions = np.diff(tcp_xy, axis=0).astype(np.float32)  # (total_steps, 2)

    return dict(
        initial_block_pos  = bs[0, ACTOR_POS].astype(np.float32),
        initial_block_quat = bs[0, ACTOR_QUAT].astype(np.float32),
        initial_gripper_xy = tcp_xy[0].astype(np.float32),
        actions            = actions,
        gt_tcp_xy          = tcp_xy,
        gt_block_pos       = bs[:, ACTOR_POS].astype(np.float32),
        gt_block_quat      = bs[:, ACTOR_QUAT].astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# AR rollout using imagined-style queries
# ─────────────────────────────────────────────────────────────────────────────

def ar_rollout_with_imagined_queries(demo: dict,
                                     n_ar_windows: int,
                                     n_block: int, n_hand: int,
                                     seed: int,
                                     client: WorldModelClient) -> dict:
    rng = np.random.default_rng(seed)
    block_tpl = _sample_box(np.zeros(3), np.full(3, CUBE_HALF), n_block, rng)
    hand_tpl  = _sample_cylinder(GRIPPER_RADIUS, GRIPPER_LENGTH, n_hand, rng)

    # Initial state (world frame).
    block_pts_w = apply_pose(block_tpl,
                              demo["initial_block_pos"],
                              demo["initial_block_quat"]).astype(np.float32)
    gripper_xy  = demo["initial_gripper_xy"].astype(np.float32).copy()

    pred_block_chunks = []
    hand_chunks       = []

    for w in range(n_ar_windows):
        # H5 actions for this window (PRED_FRAMES per window).
        s = w * PRED_FRAMES
        deltas = demo["actions"][s : s + PRED_FRAMES]      # (PRED_FRAMES, 2)
        if deltas.shape[0] < PRED_FRAMES:                  # pad with zeros at the tail
            pad = np.zeros((PRED_FRAMES - deltas.shape[0], 2), dtype=np.float32)
            deltas = np.concatenate([deltas, pad], axis=0)

        # Gripper trajectory across MODEL_T = 1 + PRED_FRAMES.
        grip_traj = _candidate_gripper_traj(gripper_xy, deltas)

        # Centring shift (mean of block + hand_at_frame_0). Imagined style.
        hand_t0 = _hand_pts_world(hand_tpl, grip_traj[0:1])[0]
        combined = np.concatenate([block_pts_w, hand_t0], axis=0)
        shift = combined.mean(axis=0).astype(np.float32)

        # Imagined-style query.
        q = _build_query(block_pts_w, grip_traj, shift)

        # Call WM. predict_batch slices [1:1+PRED_FRAMES] internally.
        pred_centred = client.predict_batch([q])[0]        # (PRED_FRAMES, N_w, 3)
        pred_world   = pred_centred + shift                # uncenter

        pred_block_chunks.append(pred_world.astype(np.float32))

        # Build hand for frames 1..PRED_FRAMES of this window (for viz).
        hand_window = _hand_pts_world(hand_tpl, grip_traj[1:1 + PRED_FRAMES])
        hand_chunks.append(hand_window.astype(np.float32))

        # Roll forward.
        block_pts_w = pred_world[-1].astype(np.float32)
        gripper_xy  = grip_traj[-1].astype(np.float32)

    initial_hand = _hand_pts_world(hand_tpl, demo["initial_gripper_xy"][None, :])[0]
    block_full = np.concatenate(
        [apply_pose(block_tpl, demo["initial_block_pos"],
                    demo["initial_block_quat"]).astype(np.float32)[None]] +
        [c for c in pred_block_chunks], axis=0,
    )
    hand_full = np.concatenate([initial_hand[None]] + hand_chunks, axis=0)
    return dict(block_full=block_full, hand_full=hand_full)


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────

def render_gif(block_frames: np.ndarray, hand_frames: np.ndarray,
               init_block_pos: np.ndarray, init_gripper_xy: np.ndarray,
               save_path: Path, title_prefix: str,
               fps: int = 6, dpi: int = 100) -> None:
    T = block_frames.shape[0]
    all_xy = np.concatenate(
        [block_frames.reshape(-1, 3)[:, :2],
         hand_frames.reshape(-1, 3)[:, :2],
         init_block_pos[None, :2], init_gripper_xy[None, :]],
        axis=0,
    )
    pad = 0.015
    xmin, ymin = all_xy.min(axis=0) - pad
    xmax, ymax = all_xy.max(axis=0) + pad

    gif_frames = []
    for t in range(T):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        for tt in range(t):
            pts = block_frames[tt]
            ax.scatter(pts[:, 0], pts[:, 1], s=2, color="lightgray",
                       alpha=0.10, edgecolors="none")
        ax.scatter(block_frames[t, :, 0], block_frames[t, :, 1], s=10,
                   color="C0", alpha=0.85, edgecolors="none", label="predicted block")
        ax.scatter(hand_frames[t, :, 0], hand_frames[t, :, 1], s=4,
                   color="orange", alpha=0.85, edgecolors="none", label="gripper")
        ax.scatter(*init_block_pos[:2], color="black", marker="s", s=110,
                   edgecolor="white", zorder=5, label="block t=0")
        ax.scatter(*init_gripper_xy[:2], color="black", marker="^", s=90,
                   edgecolor="white", zorder=5, label="gripper t=0")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title(f"{title_prefix}  frame {t}/{T-1}",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        gif_frames.append(imageio.imread(buf))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_path), gif_frames, fps=fps, loop=0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--traj", default="traj_0")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n_ar_windows", type=int, default=5)
    p.add_argument("--n_block", type=int, default=128)
    p.add_argument("--n_hand", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--server", default="http://localhost:5123")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--fps", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    total_steps = args.n_ar_windows * PRED_FRAMES

    demo = load_h5_state_and_actions(args.h5, args.traj, args.start, total_steps)
    print(f"loaded {args.traj} start={args.start}: "
          f"actions shape={demo['actions'].shape}, "
          f"action norms mean={np.linalg.norm(demo['actions'], axis=1).mean():.4f}")

    client = WorldModelClient(args.server, timeout=args.timeout)
    client.response_key = "scene_flows"
    if not client.health():
        raise SystemExit(f"server {args.server} is unreachable")

    result = ar_rollout_with_imagined_queries(
        demo, args.n_ar_windows, args.n_block, args.n_hand,
        args.seed, client,
    )
    title = (f"{Path(args.h5).stem} {args.traj} start={args.start}  "
             f"(h5-init + h5-actions, imagined-style query)")
    render_gif(result["block_full"], result["hand_full"],
               demo["initial_block_pos"], demo["initial_gripper_xy"],
               Path(args.output), title_prefix=title, fps=args.fps)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
