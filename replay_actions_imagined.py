#!/usr/bin/env python3
"""
replay_actions_imagined.py

Render a SAPIEN video of the WORLD-MODEL IMAGINED trajectory, with NO
dynamics — both the block AND the gripper are pose-set every frame from
the planner's `imagined_block_xy` / `imagined_gripper_xy` arrays in
actions.npz. Output: <plan_dir>/replay_imagined/rollout.mp4 (and .gif).

Use case: paper figure that shows what the model *thinks* it commanded vs
what actually happens in sim (`replay_actions.py --use_set_pose`). For a
side-by-side visualization. The two videos share initial state but the
imagined one ignores physics entirely.

Block orientation is held at the initial quat (the planner doesn't predict
a block rotation; if rotation matters we can fit one from the predicted
point cloud later).

Run in the `planning_wrapper` conda env.

Usage:
    python replay_actions_imagined.py /path/to/plan_dir
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_PW_ROOT = Path("/data/user_data/mbronars/packages/Planning_wrapper")
sys.path.insert(0, str(_PW_ROOT))
sys.path.insert(0, str(_PW_ROOT / "scripts"))
os.chdir(str(_PW_ROOT))

import gymnasium as gym
import sapien  # noqa: F401
import envs    # noqa: F401

from live_worldmodel_v2 import (
    _unwrap, _to_np, _setup_episode, _read_env0,
)

try:
    import imageio.v2 as imageio
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False


def _quat_wxyz_to_sapien(q: np.ndarray):
    """Sapien expects (w, x, y, z) too — pass straight through."""
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _z_rotation_quat(theta_rad: float) -> np.ndarray:
    """wxyz quat for a rotation of `theta_rad` about the +z world axis."""
    c, s = np.cos(theta_rad / 2.0), np.sin(theta_rad / 2.0)
    return np.array([c, 0.0, 0.0, s], dtype=np.float32)


def _fit_2d_rotation(P_ref_xy: np.ndarray, P_cur_xy: np.ndarray) -> float:
    """Procrustes 2D rotation that maps P_ref to P_cur (both (N, 2)).

    Returns the rotation angle θ in radians (rotation about z, world frame).
    Both clouds are first centered at their own centroids; the optimal
    rotation is then recovered via SVD of H = P_ref_c.T @ P_cur_c.
    Sign-flips the rotation if det < 0 (rare; would indicate reflection).
    """
    ref_c = P_ref_xy - P_ref_xy.mean(axis=0)
    cur_c = P_cur_xy - P_cur_xy.mean(axis=0)
    H = ref_c.T @ cur_c                                   # (2, 2)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T                                        # (2, 2)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _compute_imagined_block_quats(
    positions_npz: Path,
    init_block_quat: np.ndarray,
) -> np.ndarray | None:
    """For each frame in positions.npz, fit a 2D rotation of the BLOCK
    particles relative to the anchor (ctx_len-1) frame, then compose with
    `init_block_quat` to get the per-frame block quaternion.

    Returns (T_imagined, 4) wxyz array, or None if positions.npz unusable.
    The number of frames matches imagined_block_xy in actions.npz.
    """
    if not positions_npz.is_file():
        return None
    with np.load(positions_npz, allow_pickle=False) as d:
        pred_centered = d["predicted_centered"]           # (T, N, 3)
        offset_world  = d["offset_world"]                 # (3,)
        object_ids    = d["object_ids"]                   # (N,)
        ctx_len       = int(d["context_len"])
    block_mask = object_ids == 0
    if not np.any(block_mask):
        return None
    pred_world = pred_centered + offset_world[None, None, :]
    block_traj = pred_world[:, block_mask, :2]            # (T, N_block, 2)

    ref = block_traj[ctx_len - 1]                         # anchor frame
    quats = np.zeros((block_traj.shape[0], 4), dtype=np.float32)
    for t in range(block_traj.shape[0]):
        theta = _fit_2d_rotation(ref, block_traj[t])
        qz = _z_rotation_quat(theta)
        # Apply world-frame z-rotation on top of the initial block quat:
        # q_t = q_z(world) * q_init
        quats[t] = _quat_mul_wxyz(qz, init_block_quat)
    return quats


def _set_block_pose(base, xy: np.ndarray, z: float, quat_wxyz: np.ndarray):
    """Teleport the block to (x, y, z) with the given quat. No dynamics."""
    from sapien import Pose
    base.block.set_pose(Pose(
        p=[float(xy[0]), float(xy[1]), float(z)],
        q=_quat_wxyz_to_sapien(quat_wxyz),
    ))
    try:
        base.block.set_velocity([0.0, 0.0, 0.0])
        base.block.set_angular_velocity([0.0, 0.0, 0.0])
    except Exception:
        pass


def _set_gripper_qpos(base, xy: np.ndarray):
    """Teleport the gripper to absolute (x, y). Joint coords are relative to
    boundary centre (see envs.push_boundary)."""
    from envs.push_boundary import BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y
    robot = base.agent.robot
    n_dof = robot.dof if hasattr(robot, "dof") else len(robot.get_qpos())
    if hasattr(n_dof, "item"):
        n_dof = int(n_dof.item())
    qpos = torch.zeros(n_dof, dtype=torch.float32)
    qpos[0] = float(xy[0]) - BOUNDARY_CENTER_X
    qpos[1] = float(xy[1]) - BOUNDARY_CENTER_Y
    robot.set_qpos(qpos.unsqueeze(0).expand(base.num_envs, -1))
    robot.set_qvel(torch.zeros_like(qpos).unsqueeze(0).expand(base.num_envs, -1))


def replay_imagined(plan_dir: Path, video_fps: int = 20, write_gif: bool = True):
    actions_npz = plan_dir / "actions.npz"
    if not actions_npz.is_file():
        raise SystemExit(f"missing {actions_npz}")

    with np.load(actions_npz, allow_pickle=False) as d:
        init_block_pos      = d["init_block_pos"].astype(np.float32)      # (3,)
        init_block_quat     = d["init_block_quat"].astype(np.float32)     # (4,) wxyz
        init_gripper_xy     = d["init_gripper_xy"].astype(np.float32)     # (2,)
        target_xy           = d["target_xy"].astype(np.float32)           # (2,)
        imagined_block_xy   = d["imagined_block_xy"].astype(np.float32)   # (T+ctx, 2)
        imagined_gripper_xy = d["imagined_gripper_xy"].astype(np.float32) # (T+ctx, 2)
        imagined_final      = float(d["final_dist"])

    block_z = float(init_block_pos[2])

    # Per-frame block rotation from Procrustes on the predicted particle cloud.
    # Falls back to a constant initial quat if positions.npz is missing.
    positions_npz = plan_dir / "positions.npz"
    imagined_block_quat = _compute_imagined_block_quats(positions_npz, init_block_quat)
    if imagined_block_quat is None:
        print(f"  [warn] {positions_npz} not found — using fixed init_block_quat")
        imagined_block_quat = np.tile(init_block_quat[None, :],
                                      (imagined_block_xy.shape[0], 1))
    elif imagined_block_quat.shape[0] != imagined_block_xy.shape[0]:
        # Length mismatch shouldn't happen, but be defensive.
        print(f"  [warn] positions/actions frame count mismatch "
              f"({imagined_block_quat.shape[0]} vs {imagined_block_xy.shape[0]}); "
              f"truncating to min.")
        n = min(imagined_block_quat.shape[0], imagined_block_xy.shape[0])
        imagined_block_quat = imagined_block_quat[:n]
        imagined_block_xy   = imagined_block_xy[:n]
        imagined_gripper_xy = imagined_gripper_xy[:n]

    env = gym.make(
        "PushBoundary",
        obs_mode="state_dict",
        control_mode="floating_vel",
        render_mode="rgb_array",
        sim_backend="cpu",
        num_envs=1,
        robot_uids="floating_gripper",
        target_xy=(float(target_xy[0]), float(target_xy[1])),
    )
    base = _unwrap(env)
    env.reset()
    if hasattr(base, "set_target_xy"):
        base.set_target_xy(target_xy)

    # Set initial state (same as a normal episode setup)
    _setup_episode(base, init_block_pos, init_block_quat, init_gripper_xy)

    frames: List[np.ndarray] = []
    if _IMAGEIO_AVAILABLE:
        frames.append(_to_np(env.render()[0]).copy())

    t0 = time.time()
    # Drive both block and gripper from the planner's imagined trajectory.
    # NO env.step — pure teleportation per frame. We DO need to advance the
    # render scene so that the camera/light updates pick up new poses; calling
    # env.step(zeros) once does that without disturbing dynamics (set_qpos
    # was just applied; PD has same target). To completely skip dynamics we
    # rely on the block's set_pose + zeroed velocity to keep it pinned.
    for t in range(imagined_block_xy.shape[0]):
        _set_block_pose(base, imagined_block_xy[t], block_z, imagined_block_quat[t])
        _set_gripper_qpos(base, imagined_gripper_xy[t])
        # Single step with action=0 to refresh the render; block.set_pose was
        # just applied so any tiny dynamics step pulls it back to the pinned
        # pose (velocity zeroed, gripper PD target equals current).
        env.step(np.zeros(2, dtype=np.float32))
        # Re-pin block in case the step nudged it (it shouldn't, since
        # velocity is zeroed and there's no contact w/ static gripper drive).
        _set_block_pose(base, imagined_block_xy[t], block_z, imagined_block_quat[t])
        if _IMAGEIO_AVAILABLE:
            frames.append(_to_np(env.render()[0]).copy())
    elapsed = time.time() - t0

    replay_dir = plan_dir / "replay_imagined"
    replay_dir.mkdir(parents=True, exist_ok=True)

    if _IMAGEIO_AVAILABLE and frames:
        mp4_path = replay_dir / "rollout.mp4"
        imageio.mimsave(str(mp4_path), frames, fps=video_fps)
        print(f"  wrote {mp4_path}  ({len(frames)} frames @ {video_fps} fps)")
        if write_gif:
            gif_path = replay_dir / "rollout.gif"
            imageio.mimsave(str(gif_path), frames, fps=video_fps, loop=0)
            print(f"  wrote {gif_path}")

    # Also write the trajectory for any downstream analysis.
    np.savez(
        replay_dir / "imagined_trajectory.npz",
        block_xy=imagined_block_xy,
        block_quat=imagined_block_quat,   # (T, 4) per-frame quats from Procrustes
        gripper_xy=imagined_gripper_xy,
        target_xy=target_xy,
        imagined_final_dist=imagined_final,
        block_z=block_z,
        init_block_quat=init_block_quat,
    )
    print(f"  replay_imagined done in {elapsed:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan_dir", type=Path)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--no_gif", action="store_true")
    args = p.parse_args()
    replay_imagined(args.plan_dir, video_fps=args.video_fps, write_gif=not args.no_gif)


if __name__ == "__main__":
    main()
