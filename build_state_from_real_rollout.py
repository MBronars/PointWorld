"""Extract the initial (rod, cube, target) state from a real-world rollout
npz (final_policy_rollout/rollout_XXXX_*.npz) and save it in the format
imagined_mppi.py's --state_npz expects (block_pos (3,), block_quat (4,),
gripper_xy (2,), target_xy (2,)).

We keep the cube_rot's baked-in -66.85° offset as-is because that offset
was applied to match the inference server's rotated block-template anchor
— since we're feeding the same trained model, the offset is correct.

Usage:
    python build_state_from_real_rollout.py \\
        --rollout final_policy_rollout/rollout_0002_20260807_231623.npz \\
        --out    /tmp/state_trial02.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CUBE_HALF_HEIGHT_M = 0.026    # from README §"Geometry not stored in the files"


def yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
    half = 0.5 * yaw_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan-idx", type=int, default=0,
                    help="Which of the trial's plan entries to pull the target "
                         "from (usually 0 — trials had 1 plan each in this batch).")
    args = ap.parse_args()

    d = np.load(args.rollout, allow_pickle=True)
    rod_xy = np.asarray(d["ee_xy"][0], dtype=np.float32).ravel()
    cube_xy = np.asarray(d["cube_xy"][0], dtype=np.float32).ravel()
    cube_yaw = float(d["cube_rot"][0])   # radians, includes -66.85° offset

    plans = d["plans"]
    if args.plan_idx >= len(plans):
        raise SystemExit(f"plan idx {args.plan_idx} > {len(plans)-1} plans in trial")
    target_xy = np.asarray(plans[args.plan_idx]["target"], dtype=np.float32).ravel()

    block_pos  = np.array([cube_xy[0], cube_xy[1], CUBE_HALF_HEIGHT_M], dtype=np.float32)
    block_quat = yaw_to_quat_wxyz(cube_yaw)
    gripper_xy = rod_xy.astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, block_pos=block_pos, block_quat=block_quat,
             gripper_xy=gripper_xy, target_xy=target_xy)
    print(f"wrote {out}")
    print(f"  block_pos  = {block_pos.tolist()}  (cube xy + half-height z)")
    print(f"  block_quat = {block_quat.tolist()}  (from cube_yaw = {cube_yaw:+.4f} rad, "
          f"offset -66.85° KEPT per README)")
    print(f"  gripper_xy = {gripper_xy.tolist()}  (rod-axis-corrected ee_xy)")
    print(f"  target_xy  = {target_xy.tolist()}  (from plans[{args.plan_idx}]['target'])")


if __name__ == "__main__":
    main()
