#!/usr/bin/env python3
"""
replay_actions.py

Open-loop replay of a saved imagined_mppi.py plan in SAPIEN. Loads
`actions.npz` from a plan dir, resets PushBoundary to the saved spawn,
steps through the actions, captures a video + sim trajectory, and prints
the sim final block→target distance alongside the WM-imagined final
distance from the plan.

Run in the `planning_wrapper` conda env (this file transitively imports
sapien via Planning_wrapper/envs).

Usage:
    python replay_actions.py /data/user_data/mbronars/packages/PointWorld/runs/seed0_first3/seed0_ep000

    # Batch — globs subdirs that contain actions.npz:
    python replay_actions.py /data/user_data/mbronars/packages/PointWorld/runs/seed0_first3 --glob 'seed0_ep*'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_PW_ROOT = Path("/data/user_data/mbronars/packages/Planning_wrapper")
sys.path.insert(0, str(_PW_ROOT))
sys.path.insert(0, str(_PW_ROOT / "scripts"))
# The floating_gripper URDF is referenced by a path RELATIVE to cwd by
# FloatingGripper.urdf_path; chdir up front so mani_skill resolves it.
os.chdir(str(_PW_ROOT))

import gymnasium as gym
import sapien  # noqa: F401  — register sapien backend for mani_skill
import envs  # noqa: F401  — registers PushBoundary + FloatingGripper

from live_worldmodel_v2 import (
    _unwrap, _to_np, _setup_episode, _read_env0,
    _DEFAULT_TARGET_REACHED,
)

try:
    import imageio.v2 as imageio
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False


def _set_pose_step(env, base, grip_xy_abs: np.ndarray,
                   action: np.ndarray) -> np.ndarray:
    """Teleport the gripper to its absolute next-step pose, then advance one
    control step with zero action.

    `grip_xy_abs` is the running absolute (x, y) of the gripper before this
    step. `action` is the (Δx, Δy) for this step. We update grip_xy_abs by
    the delta, set_qpos to the absolute joint coords (relative to the boundary
    centre, as joint_x/joint_y are defined), then step with action=0 so the
    PD controller's `_target_qpos += 0` keeps the target at the just-set
    pose — no fight, no drift.

    Replaces the old incremental-qpos + manual scene.step pattern that left
    the PD controller's drive target stale and dragged the gripper back.
    Matches Planning_wrapper/scripts/playback_floating._set_gripper_xy +
    env.step(zeros). Returns the updated absolute (x, y)."""
    from envs.push_boundary import BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y

    grip_xy_abs = (np.asarray(grip_xy_abs, dtype=np.float32)
                   + np.asarray(action, dtype=np.float32))[:2]

    robot = base.agent.robot
    n_dof = robot.dof if hasattr(robot, "dof") else len(robot.get_qpos())
    if hasattr(n_dof, "item"):
        n_dof = int(n_dof.item())
    qpos = torch.zeros(n_dof, dtype=torch.float32)
    qpos[0] = float(grip_xy_abs[0]) - BOUNDARY_CENTER_X
    qpos[1] = float(grip_xy_abs[1]) - BOUNDARY_CENTER_Y
    robot.set_qpos(qpos.unsqueeze(0).expand(base.num_envs, -1))
    robot.set_qvel(torch.zeros_like(qpos).unsqueeze(0).expand(base.num_envs, -1))

    env.step(np.zeros(2, dtype=np.float32))
    return grip_xy_abs


def replay_one(plan_dir: Path,
               threshold: float = _DEFAULT_TARGET_REACHED,
               want_video: bool = True,
               video_fps: int = 20,
               use_set_pose: bool = False,
               block_physics: dict | None = None) -> dict:
    actions_npz = plan_dir / "actions.npz"
    if not actions_npz.is_file():
        raise SystemExit(f"missing {actions_npz}")

    with np.load(actions_npz, allow_pickle=False) as d:
        actions     = d["actions"].astype(np.float32)              # (T, 2)
        block_pos   = d["init_block_pos"].astype(np.float32)       # (3,)
        block_quat  = d["init_block_quat"].astype(np.float32)      # (4,) wxyz
        gripper_xy  = d["init_gripper_xy"].astype(np.float32)      # (2,)
        target_xy   = d["target_xy"].astype(np.float32)            # (2,)
        imagined_block_xy   = d["imagined_block_xy"].astype(np.float32)
        imagined_gripper_xy = d["imagined_gripper_xy"].astype(np.float32)
        imagined_final = float(d["final_dist"])

    env = gym.make(
        "PushBoundary",
        obs_mode     = "state_dict",
        control_mode = "floating_vel",
        render_mode  = "rgb_array" if want_video else None,
        sim_backend  = "cpu",
        num_envs     = 1,
        robot_uids   = "floating_gripper",
        target_xy    = (float(target_xy[0]), float(target_xy[1])),
    )
    base = _unwrap(env)
    env.reset()
    if hasattr(base, "set_target_xy"):
        base.set_target_xy(target_xy)
    _setup_episode(base, block_pos, block_quat, gripper_xy)
    if block_physics is not None:
        # The env's own cube is mu 1.5/1.2 at density 200; a dataset collected
        # with other block physics must have them re-applied here or the replay
        # runs different dynamics than the plan was trained on.
        from envs.push_boundary import apply_block_physics
        apply_block_physics(base, block_physics)

    frames: List[np.ndarray] = []
    if want_video and _IMAGEIO_AVAILABLE:
        frames.append(_to_np(env.render()[0]).copy())

    bp0, bq0, gxy0 = _read_env0(base)
    sim_log = dict(
        block_pos  = [bp0.copy()],
        block_quat = [bq0.copy()],
        gripper_xy = [gxy0.copy()],
    )

    t0 = time.time()
    # Running absolute gripper xy for the set_pose path (deltas accumulate).
    grip_xy_abs = np.asarray(gripper_xy, dtype=np.float32).copy()
    for action in actions:
        if use_set_pose:
            grip_xy_abs = _set_pose_step(env, base, grip_xy_abs, action)
        else:
            env.step(np.asarray(action, np.float32))
        if want_video and _IMAGEIO_AVAILABLE:
            frames.append(_to_np(env.render()[0]).copy())
        bp, bq, gxy = _read_env0(base)
        sim_log['block_pos'].append(bp.copy())
        sim_log['block_quat'].append(bq.copy())
        sim_log['gripper_xy'].append(gxy.copy())
    replay_wall = time.time() - t0

    block_pos_arr  = np.asarray(sim_log['block_pos'],  np.float32)
    block_quat_arr = np.asarray(sim_log['block_quat'], np.float32)
    gripper_xy_arr = np.asarray(sim_log['gripper_xy'], np.float32)
    sim_final = float(np.linalg.norm(block_pos_arr[-1, :2] - target_xy))

    replay_subdir = "replay_set_pose" if use_set_pose else "replay"
    replay_dir = plan_dir / replay_subdir
    replay_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        replay_dir / "sim_trajectory.npz",
        block_pos  = block_pos_arr,
        block_quat = block_quat_arr,
        gripper_xy = gripper_xy_arr,
        actions    = actions,
        target_xy  = target_xy,
        sim_final_dist     = np.float32(sim_final),
        imagined_final_dist = np.float32(imagined_final),
        sim_success         = np.int8(sim_final < threshold),
        replay_wall_sec     = np.float32(replay_wall),
    )
    if want_video and _IMAGEIO_AVAILABLE and frames:
        imageio.mimsave(str(replay_dir / "rollout.mp4"), frames, fps=video_fps)

    if _VIZ_AVAILABLE:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.plot(imagined_block_xy[:, 0],   imagined_block_xy[:, 1],
                "--", color="C0", lw=2, label="block (imagined / WM)")
        ax.plot(block_pos_arr[:, 0],       block_pos_arr[:, 1],
                "-",  color="C0", lw=2.5, label="block (sim replay)")
        ax.plot(imagined_gripper_xy[:, 0], imagined_gripper_xy[:, 1],
                "--", color="C1", lw=1, alpha=0.7, label="gripper (imagined)")
        ax.plot(gripper_xy_arr[:, 0],      gripper_xy_arr[:, 1],
                "-",  color="C1", lw=1.2, alpha=0.7, label="gripper (sim)")
        ax.scatter(block_pos[0], block_pos[1], color="green", marker="o", s=110,
                   edgecolor="black", zorder=5, label="start")
        ax.scatter(imagined_block_xy[-1, 0], imagined_block_xy[-1, 1],
                   color="C0", marker="D", s=80, edgecolor="black",
                   zorder=5, label="imagined end")
        ax.scatter(block_pos_arr[-1, 0], block_pos_arr[-1, 1],
                   color="black", marker="s", s=110, edgecolor="white",
                   zorder=5, label="sim end")
        ax.scatter(target_xy[0], target_xy[1], color="red", marker="*", s=300,
                   edgecolor="black", zorder=6, label="target")
        circ = plt.Circle(target_xy, threshold, color="red", fill=False, ls=":",
                          alpha=0.5)
        ax.add_artist(circ)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title(f"{plan_dir.name}\n"
                     f"imagined: {imagined_final*100:.2f} cm  |  "
                     f"sim: {sim_final*100:.2f} cm  "
                     f"({'✓' if sim_final < threshold else '✗'})")
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        fig.savefig(replay_dir / "imagined_vs_sim.png", dpi=130)
        plt.close(fig)

    env.close()

    return dict(
        plan_dir       = str(plan_dir),
        imagined_final = imagined_final,
        sim_final      = sim_final,
        sim_success    = sim_final < threshold,
        replay_wall    = replay_wall,
        n_actions      = int(actions.shape[0]),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=str,
                   help="Either a single plan dir (containing actions.npz), "
                        "or a parent dir with --glob to discover plan dirs.")
    p.add_argument("--glob", type=str, default=None,
                   help="If set, treat `root` as a parent and replay every "
                        "matching subdir that has actions.npz.")
    p.add_argument("--threshold", type=float, default=_DEFAULT_TARGET_REACHED)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--block_physics", type=str, default=None,
                   help="JSON file (or inline JSON) of block physics to apply to "
                        "the cube before stepping: static_friction, "
                        "dynamic_friction, restitution, mass, com_x, com_y, "
                        "ang_damp. Use the values the dataset was collected with "
                        "(recorded per episode in the demo json). Omitted = the "
                        "env's own defaults.")
    p.add_argument("--use_set_pose", action="store_true",
                   help="Teleport the gripper to qpos+action each step "
                        "(bypass PD controller). Smoke-test mode — outputs go "
                        "under <plan>/replay_set_pose/ to keep them separate.")
    return p.parse_args()


def main():
    args = parse_args()
    _bp = None
    if args.block_physics:
        _bp = (json.loads(args.block_physics) if args.block_physics.lstrip().startswith("{")
               else json.loads(Path(args.block_physics).read_text()))
        print(f"  block physics applied: {_bp}")
    root = Path(args.root)
    if args.glob is not None:
        plan_dirs = sorted([p for p in root.glob(args.glob)
                            if (p / "actions.npz").is_file()])
    else:
        plan_dirs = [root]

    if not plan_dirs:
        raise SystemExit(f"No plan dirs found under {root} (glob={args.glob}).")

    print(f"Replaying {len(plan_dirs)} plan(s)…\n")
    results = []
    for plan in plan_dirs:
        print(f"[{plan.name}]")
        try:
            r = replay_one(plan,
                           threshold=args.threshold,
                           want_video=not args.no_video,
                           video_fps=args.video_fps,
                           use_set_pose=args.use_set_pose,
                           block_physics=_bp)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        results.append(r)
        delta = r['sim_final'] - r['imagined_final']
        print(f"  imagined final: {r['imagined_final']*100:6.2f} cm")
        print(f"  sim final     : {r['sim_final']*100:6.2f} cm  "
              f"{'✓' if r['sim_success'] else '✗'}  (Δ {delta*100:+.2f} cm)")
        print(f"  sim wall      : {r['replay_wall']:.2f} s  "
              f"({r['n_actions']} actions)\n")

    if len(results) > 1:
        print(f"{'='*68}")
        print(f"  Summary across {len(results)} replay(s)")
        print(f"{'='*68}")
        print(f"  {'episode':<24} {'imagined':>10} {'sim':>10} {'Δ':>9}")
        for r in results:
            print(f"  {Path(r['plan_dir']).name:<24} "
                  f"{r['imagined_final']*100:>9.2f}cm "
                  f"{r['sim_final']*100:>9.2f}cm "
                  f"{(r['sim_final']-r['imagined_final'])*100:>+8.2f}cm")
        ims = np.array([r['imagined_final'] for r in results])
        sms = np.array([r['sim_final']      for r in results])
        succ = np.array([r['sim_success']   for r in results], dtype=bool)
        print(f"  -")
        print(f"  imagined mean ± std: {ims.mean()*100:.2f} ± {ims.std()*100:.2f} cm")
        print(f"  sim      mean ± std: {sms.mean()*100:.2f} ± {sms.std()*100:.2f} cm")
        print(f"  sim success rate   : {succ.mean()*100:.0f}%  ({succ.sum()}/{len(results)})")
        # save summary json
        out = {"results": results,
               "imagined_mean": float(ims.mean()),
               "imagined_std":  float(ims.std()),
               "sim_mean":      float(sms.mean()),
               "sim_std":       float(sms.std()),
               "sim_success_rate": float(succ.mean())}
        (root / "replay_summary.json").write_text(json.dumps(out, indent=2))
        print(f"  Summary JSON       : {root / 'replay_summary.json'}")


if __name__ == "__main__":
    main()
