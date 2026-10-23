"""Claim-and-go worker for the SAPIEN-as-dynamics MPPI baseline on
block pushing seed_42 scenarios.

For each scenario (loaded from
`PointWorld/scenarios/{difficulty}/seed_{S}/ep_{NN}.npz`):
  1. Invoke `Planning_wrapper/scripts/live_worldmodel_v2.py --planner sim`
     with explicit --initial_block_pos / --initial_block_quat /
     --initial_gripper_xy / --target_xy from the scenario file.
  2. The script writes trajectory.npz with sim block trajectory + final_dist.
  3. Save to inference_results/<SWEEP_NAME>/seed{S}_ep{NN}/.

Env knobs:
  SWEEP_NAME              defaults to block_pushing_sim_mppi_seed42_<DIFF>
  SCENARIO_DIFFICULTY     "random" or "hard"
  SEEDS                   comma-sep, defaults to "42"
  POPULATION_SIZE         MPPI K, default 256
  HORIZON                 default 20 (horizon per replan)
  REPLAN_EVERY            default 5 (matches DIFF/MPPI baselines)
  TOTAL_STEPS             default 60 (block pushing T=60)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
POINTWORLD = HERE.parent
PLANNING_WRAPPER = Path("/data/user_data/mbronars/packages/Planning_wrapper")
PLANNER_SCRIPT = PLANNING_WRAPPER / "scripts" / "live_worldmodel_v2.py"

SCENARIO_DIFFICULTY = os.environ.get("SCENARIO_DIFFICULTY", "random")
SWEEP_NAME = os.environ.get(
    "SWEEP_NAME", f"block_pushing_sim_mppi_seed42_{SCENARIO_DIFFICULTY}"
)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42").split(",")]
EPISODES_PER_SEED = int(os.environ.get("EPISODES_PER_SEED", "50"))
POPULATION_SIZE = int(os.environ.get("POPULATION_SIZE", "256"))
HORIZON = int(os.environ.get("HORIZON", "20"))
REPLAN_EVERY = int(os.environ.get("REPLAN_EVERY", "5"))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "60"))

SCENARIO_ROOT = POINTWORLD / "scenarios" / SCENARIO_DIFFICULTY
SWEEP_OUTPUT_ROOT = Path("/data/group_data/katefgroup/datasets/mbronars")
LOCK_STALE_S = 6 * 3600

PYTHON_BIN = "/data/user_data/mbronars/.miniconda3/envs/planning_wrapper/bin/python"


def ensure_symlink() -> None:
    sweep_dir = SWEEP_OUTPUT_ROOT / SWEEP_NAME
    sweep_dir.mkdir(parents=True, exist_ok=True)
    # PointWorld doesn't have inference_results; just leave the sweep dir.


def list_tasks() -> list[tuple[int, int]]:
    return [(s, e) for s in SEEDS for e in range(EPISODES_PER_SEED)]


def task_dir(seed: int, ep: int) -> Path:
    return SWEEP_OUTPUT_ROOT / SWEEP_NAME / f"seed{seed}_ep{ep:02d}"


def try_claim(out_dir: Path) -> tuple[bool, str]:
    lock = out_dir / ".lock"
    try:
        lock.mkdir(exist_ok=False)
        (lock / "owner.txt").write_text(
            f"host={socket.gethostname()}  pid={os.getpid()}  "
            f"start={time.time():.0f}\n"
        )
        return True, "claimed"
    except FileExistsError:
        pass
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return try_claim(out_dir)
    if age > LOCK_STALE_S:
        try:
            (lock / "owner.txt").write_text(
                f"host={socket.gethostname()}  pid={os.getpid()}  "
                f"start={time.time():.0f}  (took over stale lock age={age:.0f}s)\n"
            )
            return True, f"stale (age={age/3600:.1f}h)"
        except Exception:
            return False, "stale-takeover-failed"
    return False, f"locked (age={age/60:.0f}m)"


def release_lock(out_dir: Path) -> None:
    lock = out_dir / ".lock"
    try:
        (lock / "owner.txt").unlink(missing_ok=True)
        lock.rmdir()
    except (FileNotFoundError, OSError):
        pass


def run_plan(seed: int, ep: int, out_dir: Path) -> int:
    scenario_path = SCENARIO_ROOT / f"seed_{seed}" / f"ep_{ep:03d}.npz"
    if not scenario_path.exists():
        print(f"  [seed={seed} ep={ep:02d}] MISSING {scenario_path}", flush=True)
        return 2
    with np.load(scenario_path) as d:
        block_pos  = d["block_pos"].astype(np.float32)
        block_quat = d["block_quat"].astype(np.float32)
        gripper_xy = d["gripper_xy"].astype(np.float32)
        target_xy  = d["target_xy"].astype(np.float32)

    cmd = [
        PYTHON_BIN, str(PLANNER_SCRIPT),
        "--planner", "sim",
        "--initial_block_pos",  f"{block_pos[0]:.6f}",  f"{block_pos[1]:.6f}",  f"{block_pos[2]:.6f}",
        "--initial_block_quat", f"{block_quat[0]:.6f}", f"{block_quat[1]:.6f}",
                                f"{block_quat[2]:.6f}", f"{block_quat[3]:.6f}",
        "--initial_gripper_xy", f"{gripper_xy[0]:.6f}", f"{gripper_xy[1]:.6f}",
        "--target_xy",          f"{target_xy[0]:.6f}",  f"{target_xy[1]:.6f}",
        "--population_size",    str(POPULATION_SIZE),
        "--horizon",            str(HORIZON),
        "--replan_every",       str(REPLAN_EVERY),
        "--total_steps",        str(TOTAL_STEPS),
        "--seed",               str(seed * 10_000 + ep),    # deterministic per scenario
        "--output_dir",         str(out_dir),
        "--no_video",
        "--no_plan_dump",
        "--quiet",
    ]
    t0 = time.time()
    print(f"  [seed={seed} ep={ep:02d}] sim_mppi K={POPULATION_SIZE} H={HORIZON} R={REPLAN_EVERY}", flush=True)
    rc = subprocess.run(cmd, cwd=str(PLANNING_WRAPPER)).returncode
    elapsed = time.time() - t0
    (out_dir / "wallclock_sec.txt").write_text(f"{elapsed:.2f}\n")
    print(f"  [seed={seed} ep={ep:02d}] rc={rc} elapsed={elapsed:.0f}s", flush=True)
    return rc


def run_one(seed: int, ep: int) -> int:
    out_dir = task_dir(seed, ep)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "trajectory.npz").exists():
        return 0
    return run_plan(seed, ep, out_dir)


def main() -> int:
    ensure_symlink()
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    tasks = list_tasks()
    print(f"[worker {worker_id}] sim_mppi starting "
          f"({len(tasks)} tasks, sweep={SWEEP_NAME}, "
          f"diff={SCENARIO_DIFFICULTY}, K={POPULATION_SIZE})",
          flush=True)
    n_done = 0; n_skip = 0; n_fail = 0
    for seed, ep in tasks:
        out_dir = task_dir(seed, ep)
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "trajectory.npz").exists():
            n_skip += 1
            continue
        claimed, reason = try_claim(out_dir)
        if not claimed:
            print(f"[worker {worker_id}] seed{seed}_ep{ep:02d}: skip ({reason})", flush=True)
            n_skip += 1
            continue
        t_start = time.time()
        rc = run_one(seed, ep)
        elapsed = time.time() - t_start
        if rc == 0:
            release_lock(out_dir)
            n_done += 1
        elif elapsed < 60.0:
            release_lock(out_dir)
            n_fail += 1
            print(f"[worker {worker_id}] seed{seed}_ep{ep:02d}: FAST-FAIL rc={rc} aborting",
                  flush=True)
            return 0
        else:
            n_fail += 1
            print(f"[worker {worker_id}] seed{seed}_ep{ep:02d}: FAILED rc={rc} (leaving lock)",
                  flush=True)
    print(f"[worker {worker_id}] exit: done={n_done} skip={n_skip} fail={n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
