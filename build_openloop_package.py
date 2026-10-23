"""Repackage the H=60 K=256 planner outputs into a self-contained folder for
open-loop robot playback.

  final_policy_rollout_pw_h60_k256/
      README.md
      plan_NNNN.npz     (1 per rollout, matched to real rollout_NNNN_*.npz)
      summary.csv
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


ROOT     = Path("/data/user_data/mbronars/packages/PointWorld")
PLANS    = ROOT / "runs/real_state_H60_K256"
ROLLOUTS = ROOT / "final_policy_rollout"
DEST     = ROOT / "final_policy_rollout_pw_h60_k256"
DEST.mkdir(parents=True, exist_ok=True)


def stem_for(trial: str) -> str:
    """Match the source rollout's timestamped stem so pairing is 1:1."""
    matches = sorted(ROLLOUTS.glob(f"rollout_{trial}_*.npz"))
    return matches[0].stem   # e.g. rollout_0001_20260807_231411


def main():
    rows = []
    for pl_dir in sorted(PLANS.glob("trial_*")):
        m = re.search(r"trial_(\d+)", pl_dir.name)
        if not m:
            continue
        trial = m.group(1)
        actions_pt = pl_dir / "actions.npz"
        if not actions_pt.exists():
            print(f"  skip trial {trial} — no actions.npz")
            continue

        a = np.load(actions_pt, allow_pickle=True)
        # Real rollout for cross-reference — pull the plan-time centred fields.
        real = np.load(sorted(ROLLOUTS.glob(f"rollout_{trial}_*.npz"))[0],
                       allow_pickle=True)
        real_stem = stem_for(trial)

        actions          = np.asarray(a["actions"], dtype=np.float32)           # (60, 2)
        init_block_pos   = np.asarray(a["init_block_pos"], dtype=np.float32)    # (3,)
        init_block_quat  = np.asarray(a["init_block_quat"], dtype=np.float32)   # (4,) wxyz
        init_gripper_xy  = np.asarray(a["init_gripper_xy"], dtype=np.float32)   # (2,)
        target_xy        = np.asarray(a["target_xy"], dtype=np.float32)         # (2,)
        imagined_block   = np.asarray(a["imagined_block_xy"], dtype=np.float32) # (61, 2)
        imagined_gripper = np.asarray(a["imagined_gripper_xy"], dtype=np.float32)  # (61, 2)

        out = DEST / f"plan_{trial}.npz"
        np.savez(
            out,
            episode          = int(trial),
            source_rollout   = real_stem,
            # Initial state used at plan time (from the real rollout, first frame)
            init_gripper_xy  = init_gripper_xy,
            init_block_pos   = init_block_pos,
            init_block_quat  = init_block_quat,     # -66.85° cube_rot_offset baked in — see README
            target_xy        = target_xy,
            # Open-loop commanded actions
            actions          = actions,             # (60, 2) per-step delta xy in the panda base frame
            action_hz        = np.int32(20),
            action_max_norm  = np.float32(a["action_max_norm"]),   # 0.006 m/step
            # Model-side reference (NOT the sim; the WM's own prediction)
            imagined_block_xy   = imagined_block,       # (61, 2)
            imagined_gripper_xy = imagined_gripper,     # (61, 2)
            # Planner config
            planner            = np.array("imagined_mppi.py:H60_K256_v1", dtype=object),
            checkpoint         = np.array(
                "pushing_no_normals/model-last.pt "
                "(datasets/mbronars/3DPCWM/old_runs_08_03/PointWorld_train_logs/)",
                dtype=object),
            population_size    = np.int32(256),
            replan_every       = np.int32(5),
            total_steps        = np.int32(60),
            horizon_schedule   = np.asarray(a["H_active"]),   # per-replan planning horizon
            imagined_final_dist_m = np.float32(a["final_dist"]),
        )

        rows.append({
            "episode": int(trial),
            "source_rollout": real_stem,
            "init_gripper_xy": init_gripper_xy.tolist(),
            "init_block_xy": init_block_pos[:2].tolist(),
            "target_xy": target_xy.tolist(),
            "init_cube_to_target_mm": float(np.linalg.norm(init_block_pos[:2] - target_xy) * 1000),
            "imagined_final_dist_mm": float(a["final_dist"]) * 1000,
            "real_edge_dist_mm": float(real["final_edge_dist"]) * 1000,
            "real_success":  bool(real["success"]),
        })
        print(f"  wrote plan_{trial}.npz")

    # summary.csv
    if rows:
        cols = list(rows[0].keys())
        with open(DEST / "summary.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote summary.csv  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
