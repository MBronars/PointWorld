#!/usr/bin/env python3
"""
eval_val_trajectories.py — Roll out + score a PointWorld folding checkpoint on
all 8 held-out validation trajectories (DATASET_SPEC sec. 2).

For every val trajectory:
  - autoregressive rollout (same code path as rollout_folding_viz.py)
  - dumps <out>/<traj_name>/pred.npz with pred_cloth, gt_cloth, gt_left,
    gt_right, cloth_idx
  - records cloth L2 (mm) at frames {10, 50, 100, 500, 1000} (skipping any
    frame past the trajectory's actual length)

`--viz-count K` of the 8 trajs (chosen randomly under `--seed`) additionally get
the pgnd-style rollout.mp4 + rollout_strip.png + frames/. The others are pred
+ metrics only.

Per-traj numbers + an aggregate (mean / median across applicable trajs) are
written to <out>/metrics.json.

Usage:
    python eval_val_trajectories.py \\
        --checkpoint train_logs/folding_dense_grid/model-last.pt \\
        --out        eval_logs/folding_dense_grid_val_eval \\
        --viz-count  5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rollout_folding_viz import (                                   # noqa: E402
    CONTEXT_LEN, DOMAIN, MODEL_T, PRED_FRAMES, SKIP_FRAMES, STRIDE,
    build_eval_args, build_window_batch, grid_select_indices,
    load_model, rollout, visualize,
)

# The 8 held-out val trajectories (DATASET_SPEC sec. 2). Order matters only
# for log readability; viz selection is by random sample regardless.
VAL_RELPATHS = [
    "bimanual/existing_000000.npz",         # pants
    "bimanual/existing_000004.npz",         # long_shirt
    "single_arm/000000_000000_left.npz",    # short_shirt
    "single_arm/000001_000000_right.npz",   # short_shirt
    "ik_lift_random/000000.npz",            # short_shirt
    "ik_lift_random/000001.npz",            # short_shirt
    "ik_push/000000.npz",                   # short_shirt
    "ik_push/000001.npz",                   # short_shirt
]
METRIC_FRAMES = [10, 50, 100, 500, 1000]
DEFAULT_ROOT = "/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1"


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True, help="folding model-last.pt")
    p.add_argument("--out", default="eval_logs/folding_val_eval")
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="Dataset root containing the 4 category dirs")
    p.add_argument("--data-dir", default=None,
                   help="Dataset root with norm_stats/ (default: read from checkpoint)")
    p.add_argument("--num-frames", type=int, default=1000)
    p.add_argument("--commit-stride", type=int, default=PRED_FRAMES)
    p.add_argument("--viz-count", type=int, default=5,
                   help="How many of the 8 trajs to render full viz for (the rest get pred+metrics only)")
    p.add_argument("--viz-stride", type=int, default=10)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--no-grid-subsample", action="store_true")
    p.add_argument("--seed", type=int, default=0,
                   help="Controls which trajs get viz")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    cli = parse_cli()
    device = cli.device if (cli.device == "cuda" and torch.cuda.is_available()) else "cpu"
    out_root = Path(cli.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Pick which trajs get full viz ──────────────────────────────────────
    n_viz = max(0, min(cli.viz_count, len(VAL_RELPATHS)))
    viz_set = set(random.Random(cli.seed).sample(VAL_RELPATHS, k=n_viz))
    print(f"viz subset ({n_viz}/{len(VAL_RELPATHS)}): " + ", ".join(sorted(viz_set)))

    # ── Load checkpoint args once ──────────────────────────────────────────
    ckpt_meta = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    data_dir = cli.data_dir or ckpt_meta["args"].data_dirs[0]
    print(f"checkpoint : {cli.checkpoint}")
    print(f"data_dir   : {data_dir}")
    args, ckpt = build_eval_args(cli.checkpoint, data_dir, device)
    grid_size = float(args.grid_size)
    print(f"grid_size  : {grid_size}\n")

    model = None
    summary: dict = {
        "checkpoint":  cli.checkpoint,
        "grid_size":   grid_size,
        "num_frames":  cli.num_frames,
        "seed":        cli.seed,
        "viz_trajs":   sorted(viz_set),
        "per_traj":    [],
    }

    for rel in VAL_RELPATHS:
        traj_path = Path(cli.root) / rel
        name = rel.replace("/", "_").replace(".npz", "")
        do_viz = rel in viz_set
        traj_out = out_root / name
        traj_out.mkdir(parents=True, exist_ok=True)
        print(f"── {rel}  (viz={do_viz}) ──")

        # Load + subsample (same recipe as rollout_folding_viz.py).
        with np.load(traj_path) as d:
            cloth_full = np.asarray(d["cloth_points"],   dtype=np.float32)[SKIP_FRAMES::STRIDE]
            left_full  = np.asarray(d["left_ee_points"], dtype=np.float32)[SKIP_FRAMES::STRIDE]
            right_full = np.asarray(d["right_ee_points"],dtype=np.float32)[SKIP_FRAMES::STRIDE]
            openness   = np.asarray(d["openness"],       dtype=np.float32)[SKIP_FRAMES::STRIDE]
        robot_full = np.concatenate([left_full, right_full], axis=1)
        T_sub = cloth_full.shape[0]
        num_frames = min(cli.num_frames, T_sub - 1)

        # Fixed grid subsample on frame 0 (per-trajectory).
        if cli.no_grid_subsample:
            cloth_idx = np.arange(cloth_full.shape[1])
            robot_idx = np.arange(robot_full.shape[1])
        else:
            shift0 = np.concatenate([cloth_full[0], robot_full[0]], 0).mean(0)
            cloth_idx = grid_select_indices(cloth_full[0] - shift0, grid_size)
            robot_idx = grid_select_indices(robot_full[0] - shift0, grid_size)
        cloth = cloth_full[:, cloth_idx]
        robot = robot_full[:, robot_idx]
        print(f"  T_sub={T_sub}  cloth {cloth_full.shape[1]}->{len(cloth_idx)}  "
              f"gripper {robot_full.shape[1]}->{len(robot_idx)}")

        # Build the model on first traj (so peek can size the feature dims).
        if model is None:
            peek, _ = build_window_batch(
                cloth[0], robot[:MODEL_T],
                openness[:MODEL_T, 1:2], openness[:MODEL_T, 0:1],
                args.robot_features, args.scene_features, device,
            )
            data_info_dict = {
                "robot_features_dim": peek["robot_features"].shape[-1],
                "scene_features_dim": peek["scene_features"].shape[-1],
            }
            print(f"  robot_features_dim={data_info_dict['robot_features_dim']}  "
                  f"scene_features_dim={data_info_dict['scene_features_dim']}")
            model = load_model(args, ckpt, data_info_dict, device)

        # Rollout.
        pred = rollout(
            model, cloth, robot, openness,
            args.robot_features, args.scene_features,
            num_frames, cli.commit_stride, device,
        )
        T = pred.shape[0]
        gt_cloth = cloth[:T]
        gt_left  = left_full[:T]
        gt_right = right_full[:T]

        # Save arrays (always — viz or no).
        np.savez(traj_out / "pred.npz",
                 pred_cloth=pred, gt_cloth=gt_cloth,
                 gt_left=gt_left, gt_right=gt_right,
                 cloth_idx=cloth_idx, robot_idx=robot_idx,
                 traj_relpath=rel)

        # Per-frame cloth L2 (mm), recorded only where the traj actually reaches.
        err_mm = np.linalg.norm(pred - gt_cloth, axis=-1).mean(-1) * 1000.0
        m = {"traj": rel, "T": int(T), "n_cloth_kept": int(len(cloth_idx))}
        for f in METRIC_FRAMES:
            m[f"frame_{f}_mm"] = float(err_mm[f]) if f < T else None
        summary["per_traj"].append(m)
        recorded = " ".join(
            f"f{f}={m[f'frame_{f}_mm']:.1f}" for f in METRIC_FRAMES if m[f"frame_{f}_mm"] is not None
        )
        print(f"  cloth L2 (mm): {recorded}")

        if do_viz:
            print(f"  rendering viz -> {traj_out}/")
            visualize(pred, gt_cloth, gt_left, gt_right,
                      traj_out, cli.viz_stride, cli.fps)

    # ── Aggregate across applicable trajs ──────────────────────────────────
    agg = {}
    for f in METRIC_FRAMES:
        vals = [t[f"frame_{f}_mm"] for t in summary["per_traj"] if t[f"frame_{f}_mm"] is not None]
        if vals:
            agg[f"frame_{f}"] = {
                "n":      len(vals),
                "mean":   float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min":    float(np.min(vals)),
                "max":    float(np.max(vals)),
            }
    summary["aggregate"] = agg

    (out_root / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print("Aggregate cloth L2 (mm), across val trajectories that reach each horizon:")
    for f in METRIC_FRAMES:
        if f"frame_{f}" in agg:
            a = agg[f"frame_{f}"]
            print(f"  f{f:4d}  n={a['n']}  mean {a['mean']:7.1f}  median {a['median']:7.1f}  "
                  f"min {a['min']:6.1f}  max {a['max']:7.1f}")
    print(f"\nmetrics -> {out_root / 'metrics.json'}")
    print(f"per-traj pred.npz / viz under {out_root}/")


if __name__ == "__main__":
    main()
