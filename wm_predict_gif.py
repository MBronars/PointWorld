#!/usr/bin/env python3
"""
wm_predict_gif.py

Pull a clip from an HDF5 demo, run the PointWorld world model on the
gripper trajectory, and render the predicted block point cloud + gripper
as a gif. Supports autoregressive chaining (--n_ar_windows) so you can
extend predictions past the WM's PRED_FRAMES=10 horizon — each window's
last predicted frame becomes the next window's context.

Requires the `planning_wrapper` conda env (transitively pulls in sapien
via the imported h5 / template helpers).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

# PointWorld/utils.py shadows Planning_wrapper/utils/ because Python
# auto-prepends the script's directory to sys.path. Drop it before
# importing so `from utils.output_conversions import ...` (inside the
# Planning_wrapper client) resolves to the namespace package.
_PW_ROOT  = "/data/user_data/mbronars/packages/Planning_wrapper"
_HERE_DIR = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if Path(p).resolve() != Path(_HERE_DIR).resolve()]
sys.path.insert(0, _PW_ROOT)
sys.path.insert(0, f"{_PW_ROOT}/scripts")
os.chdir(_PW_ROOT)

import h5py
from playback_pointworld_client import call_server
from vis_push_pcd_floating import (
    sample_templates, apply_pose,
    ACTOR_POS, ACTOR_QUAT, FLOAT_QPOS, FLOAT_ROOT_POS,
)
from h5_to_pointworld_wds import IDENTITY_QUAT_WXYZ

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRED_FRAMES = 10
MODEL_T     = 11   # 1 context + 10 predicted


# ─────────────────────────────────────────────────────────────────────────────
# Demo loader + AR shard builder
# ─────────────────────────────────────────────────────────────────────────────

def extract_demo_data(h5_path: str, traj_key: str, start: int, total_len: int,
                       n_block: int, n_hand: int, shape: str,
                       seed: int, subsample: int = 1) -> dict:
    """Load (tcp_poses, joint_xy, initial_block_pts, GT block trajectory)
    for a `total_len`-frame slice of the demo. Templates are sampled once."""
    rng = np.random.default_rng(seed)
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

    block_states = block_states[1::subsample]
    art_states   = art_states[1::subsample]
    Tsub = block_states.shape[0]
    if start + total_len > Tsub:
        raise SystemExit(
            f"requested [{start}:{start+total_len}] exceeds available "
            f"trajectory length {Tsub} after lag-skip / subsample"
        )
    bs  = block_states[start : start + total_len]
    art = art_states  [start : start + total_len]

    root_pos        = art[:, FLOAT_ROOT_POS].copy()
    joint_xy        = art[:, FLOAT_QPOS].astype(np.float32)
    root_pos[:, 0] += joint_xy[:, 0]
    root_pos[:, 1] += joint_xy[:, 1]
    tcp_quat        = np.tile(IDENTITY_QUAT_WXYZ, (total_len, 1))
    tcp_poses       = np.concatenate([root_pos, tcp_quat], axis=1).astype(np.float32)

    templates = sample_templates(n_block, n_hand, rng)
    block_tpl = templates["T"] if shape == "T" else templates["cube"]
    hand_tpl  = templates["hand"]

    initial_block_pts = apply_pose(block_tpl, bs[0, ACTOR_POS],
                                    bs[0, ACTOR_QUAT]).astype(np.float32)
    return dict(
        tcp_poses         = tcp_poses,
        joint_xy          = joint_xy,
        hand_template     = hand_tpl.astype(np.float32),
        block_template    = block_tpl.astype(np.float32),
        initial_block_pts = initial_block_pts,
        initial_block_pos = bs[0, ACTOR_POS].astype(np.float32),
        initial_gripper_xy = tcp_poses[0, :2].astype(np.float32),
        gt_block_pos      = bs[:, ACTOR_POS].astype(np.float32),
    )


def build_window_shard(block_pts_world: np.ndarray,
                       hand_tpl: np.ndarray,
                       tcp_poses_window: np.ndarray,
                       joint_xy_window: np.ndarray) -> tuple:
    """Build one MODEL_T=11 frame shard for the server.

    block_pts_world  : (N_block, 3) world-frame at the window's start frame
                       (broadcast across MODEL_T as the model's input cloud)
    hand_tpl         : (N_hand, 3) raw template (cylinder)
    tcp_poses_window : (MODEL_T, 7) gripper poses for the window
    joint_xy_window  : (MODEL_T, 2) prismatic joint positions

    Returns (shard_dict, shift_vec_3)."""
    T, N = MODEL_T, block_pts_world.shape[0]
    hand_world_t0 = apply_pose(hand_tpl, tcp_poses_window[0, :3],
                                tcp_poses_window[0, 3:7]).astype(np.float32)
    shift = np.concatenate([block_pts_world, hand_world_t0], axis=0
                            ).mean(0).astype(np.float32)

    block_c = (block_pts_world - shift).astype(np.float32)
    view0_scene_flows = np.broadcast_to(block_c[None], (T, N, 3)).astype(np.float32).copy()
    visibility        = np.ones((T, N), dtype=bool)

    right_gripper_pose = tcp_poses_window.copy().astype(np.float32)
    right_gripper_pose[:, :3] -= shift
    right_gripper_open = np.zeros((T, 1), dtype=np.float32)

    shard = dict(
        view0_scene_flows      = view0_scene_flows,
        view0_scene_visibility = visibility,
        joint_positions        = joint_xy_window.astype(np.float32),
        right_gripper_pose     = right_gripper_pose,
        right_gripper_open     = right_gripper_open,
    )
    return shard, shift


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive rollout + render
# ─────────────────────────────────────────────────────────────────────────────

def ar_rollout(demo: dict, n_ar_windows: int, server_url: str,
               timeout: float = 120.0) -> dict:
    """Chain n_ar_windows model calls, each producing PRED_FRAMES=10 frames.
    The last predicted frame of window w becomes the input cloud for window w+1.

    Returns dict with:
        block_full (1 + n_ar_windows*PRED_FRAMES, N_block, 3)
        hand_full  (1 + n_ar_windows*PRED_FRAMES, N_hand,  3)
        npts_per_window (n_ar_windows,)
    """
    current_block_pts = demo["initial_block_pts"]
    pred_block_chunks = []   # each (PRED_FRAMES, N_w, 3) world frame
    hand_chunks       = []   # each (PRED_FRAMES, N_hand, 3) world frame
    npts_per_window   = []

    for w in range(n_ar_windows):
        s = w * PRED_FRAMES
        tcp_w   = demo["tcp_poses"][s : s + MODEL_T]
        joint_w = demo["joint_xy" ][s : s + MODEL_T]

        shard, shift = build_window_shard(current_block_pts,
                                          demo["hand_template"],
                                          tcp_w, joint_w)
        server_outputs = call_server(server_url, shard, timeout=timeout)
        sf = server_outputs["scene_flows"]                              # (1, T, N, 3) or (T, N, 3)
        if sf.ndim == 4:
            sf = sf[0]
        # Drop the context frame [0], keep predicted [1:1+PRED_FRAMES].
        pred_centered = sf[1 : 1 + PRED_FRAMES]
        pred_world    = pred_centered + shift                            # (PRED_FRAMES, N_w, 3)
        pred_block_chunks.append(pred_world.astype(np.float32))
        npts_per_window.append(int(pred_world.shape[1]))

        # Hand world per frame in this window (frames 1..PRED_FRAMES).
        hand_window = np.stack([
            apply_pose(demo["hand_template"], tcp_w[t, :3], tcp_w[t, 3:7])
            for t in range(1, MODEL_T)
        ], axis=0).astype(np.float32)
        hand_chunks.append(hand_window)

        current_block_pts = pred_world[-1].astype(np.float32)
        print(f"  window {w+1}/{n_ar_windows}: N_w={pred_world.shape[1]}  "
              f"block_centroid → {pred_world[-1].mean(0)[:2]}")

    # Prepend the initial state (frame 0) before stacking predictions.
    initial_hand = apply_pose(demo["hand_template"],
                               demo["tcp_poses"][0, :3],
                               demo["tcp_poses"][0, 3:7]).astype(np.float32)
    block_full = np.concatenate(
        [demo["initial_block_pts"][None]] +
        [chunk for chunk in pred_block_chunks],
        axis=0,
    )
    hand_full = np.concatenate(
        [initial_hand[None]] + [chunk for chunk in hand_chunks],
        axis=0,
    )
    return dict(
        block_full      = block_full,
        hand_full       = hand_full,
        npts_per_window = np.asarray(npts_per_window, np.int32),
    )


def render_gif(block_pcds_world: list,
               hand_pcds_world: np.ndarray,
               init_block_pos: np.ndarray,
               init_gripper_xy: np.ndarray,
               save_path: Path,
               title_prefix: str = "",
               npts_per_window: np.ndarray = None,
               fps: int = 6, dpi: int = 100) -> None:
    """block_pcds_world can be a numpy array (T, N, 3) when N is constant
    across all frames, or a list of (N_t, 3) arrays when N varies per AR
    window (PTV3 voxel pooling can change N when grid_size is enabled).
    """
    if isinstance(block_pcds_world, np.ndarray):
        block_frames = [block_pcds_world[t] for t in range(block_pcds_world.shape[0])]
    else:
        block_frames = list(block_pcds_world)
    T = len(block_frames)
    hand_pcds_world = np.asarray(hand_pcds_world, np.float32)

    all_xy = np.concatenate(
        [p[:, :2] for p in block_frames] +
        [hand_pcds_world.reshape(-1, 3)[:, :2],
         init_block_pos[None, :2], init_gripper_xy[None, :]],
        axis=0,
    )
    pad = 0.015
    xmin, ymin = all_xy.min(axis=0) - pad
    xmax, ymax = all_xy.max(axis=0) + pad

    frames_rgb = []
    for t in range(T):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        # Faint trail of earlier predicted block frames.
        for tt in range(t):
            pts = block_frames[tt]
            ax.scatter(pts[:, 0], pts[:, 1], s=2, color="lightgray",
                       alpha=0.10, edgecolors="none")
        bpts = block_frames[t]
        ax.scatter(bpts[:, 0], bpts[:, 1], s=10, color="C0",
                   alpha=0.85, edgecolors="none", label="predicted block")
        hpts = hand_pcds_world[t]
        ax.scatter(hpts[:, 0], hpts[:, 1], s=4, color="orange",
                   alpha=0.85, edgecolors="none", label="gripper")
        ax.scatter(*init_block_pos[:2], color="black", marker="s", s=110,
                   edgecolor="white", zorder=5, label="block t=0")
        ax.scatter(*init_gripper_xy[:2], color="black", marker="^", s=90,
                   edgecolor="white", zorder=5, label="gripper t=0")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        # Which AR window does frame t belong to? Frame 0 is the context;
        # frames 1..PRED_FRAMES are window 1, etc.
        w_idx = 0 if t == 0 else (t - 1) // PRED_FRAMES + 1
        n_in_w = bpts.shape[0]
        ax.set_title(f"{title_prefix}  frame {t}/{T-1}  "
                     f"(AR win {w_idx}/{(T-1)//PRED_FRAMES})  N_w={n_in_w}",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frames_rgb.append(imageio.imread(buf))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_path), frames_rgb, fps=fps, loop=0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--traj", default="traj_0")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n_ar_windows", type=int, default=1,
                   help="Number of autoregressive windows. Each produces "
                        "PRED_FRAMES=10 predicted frames. Output gif has "
                        "1 + N*PRED_FRAMES frames total (context + preds).")
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--n_block", type=int, default=128)
    p.add_argument("--n_hand", type=int, default=128)
    p.add_argument("--shape", default="cube", choices=["cube", "T"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--server", default="http://localhost:5123")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--no_scene_normals", action="store_true",
                   help="(default behaviour — this script never sends normals)")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--fps", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    total_len = 1 + args.n_ar_windows * PRED_FRAMES   # frames needed from h5

    demo = extract_demo_data(
        h5_path   = args.h5,
        traj_key  = args.traj,
        start     = args.start,
        total_len = total_len,
        n_block   = args.n_block,
        n_hand    = args.n_hand,
        shape     = args.shape,
        seed      = args.seed,
        subsample = args.subsample,
    )

    print(f"AR rollout: {args.n_ar_windows} window(s) × {PRED_FRAMES} frames "
          f"= {args.n_ar_windows * PRED_FRAMES} predicted frames "
          f"(+ 1 context = {total_len} total)")
    result = ar_rollout(demo, args.n_ar_windows, args.server, args.timeout)
    print(f"point counts per window: {result['npts_per_window'].tolist()}")

    title = f"{Path(args.h5).stem} {args.traj} start={args.start}"
    render_gif(
        block_pcds_world = result["block_full"],
        hand_pcds_world  = result["hand_full"],
        init_block_pos   = demo["initial_block_pos"],
        init_gripper_xy  = demo["initial_gripper_xy"],
        save_path        = Path(args.output),
        title_prefix     = title,
        npts_per_window  = result["npts_per_window"],
        fps              = args.fps,
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
