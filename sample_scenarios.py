#!/usr/bin/env python3
"""
sample_scenarios.py

Sample N starting scenarios per seed using the SAME sampling distribution
that the live planner / scripted demo collection uses (the boundary +
spawn constants in Planning_wrapper/envs/push_boundary.py via
live_worldmodel_v2.sample_initial_config).

Must be run in the `planning_wrapper` conda env (it transitively imports
envs.push_boundary → sapien). The OUTPUT npz files are plain numpy with
the four keys imagined_mppi.py's --state_npz loader expects, so they can
be consumed from any env afterwards.

Output layout:
    <output_root>/seed_{S}/ep_{NN}.npz   per-episode, four keys each
    <output_root>/seed_{S}/all.npz       batched (N, ...) of the same keys

Usage:
    python sample_scenarios.py \
        --seeds 0 42 \
        --n_per_seed 50 \
        --difficulty random \
        --output_root /data/user_data/mbronars/packages/PointWorld/scenarios
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PW_ROOT = Path("/data/user_data/mbronars/packages/Planning_wrapper")
sys.path.insert(0, str(_PW_ROOT))
sys.path.insert(0, str(_PW_ROOT / "scripts"))

from live_worldmodel_v2 import sample_initial_config, DIFFICULTY_CHOICES


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42])
    p.add_argument("--n_per_seed", type=int, default=50)
    p.add_argument("--difficulty", type=str, default="random",
                   choices=DIFFICULTY_CHOICES)
    p.add_argument("--output_root", type=str,
                   default="/data/user_data/mbronars/packages/PointWorld/scenarios")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_root) / args.difficulty
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Sampling {args.n_per_seed} scenarios per seed × {len(args.seeds)} "
          f"seed(s) at difficulty='{args.difficulty}'")
    print(f"  output root: {out_root}")

    for seed in args.seeds:
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        batched = {
            "block_pos":  [],
            "block_quat": [],
            "gripper_xy": [],
            "target_xy":  [],
        }
        for ep in range(args.n_per_seed):
            cfg = sample_initial_config(args.difficulty, rng)
            ep_path = seed_dir / f"ep_{ep:03d}.npz"
            np.savez(
                ep_path,
                block_pos  = cfg["block_pos"].astype(np.float32),
                block_quat = cfg["block_quat"].astype(np.float32),
                gripper_xy = cfg["gripper_xy"].astype(np.float32),
                target_xy  = cfg["target_xy"].astype(np.float32),
            )
            for k in batched:
                batched[k].append(cfg[k])

        all_path = seed_dir / "all.npz"
        np.savez(
            all_path,
            block_pos  = np.stack(batched["block_pos"]).astype(np.float32),
            block_quat = np.stack(batched["block_quat"]).astype(np.float32),
            gripper_xy = np.stack(batched["gripper_xy"]).astype(np.float32),
            target_xy  = np.stack(batched["target_xy"]).astype(np.float32),
        )
        print(f"  seed {seed:>3d}: wrote {args.n_per_seed} ep files + all.npz "
              f"→ {seed_dir}")

        # Quick spread sanity print.
        bp = np.stack(batched["block_pos"])
        gx = np.stack(batched["gripper_xy"])
        tx = np.stack(batched["target_xy"])
        gb = np.linalg.norm(gx - bp[:, :2], axis=1)
        bt = np.linalg.norm(bp[:, :2] - tx,  axis=1)
        print(f"           gripper→block (m): mean={gb.mean():.3f}  "
              f"std={gb.std():.3f}  min={gb.min():.3f}  max={gb.max():.3f}")
        print(f"           block→target  (m): mean={bt.mean():.3f}  "
              f"std={bt.std():.3f}  min={bt.min():.3f}  max={bt.max():.3f}")

    print(f"\nDone. Per-episode files: {out_root}/seed_*/ep_*.npz")
    print(f"Load one into imagined_mppi.py with:")
    print(f"  python imagined_mppi.py --state_npz "
          f"{out_root}/seed_0/ep_000.npz --output_dir out/...")


if __name__ == "__main__":
    main()
