#!/usr/bin/env python3
"""
recompute_norm_stats_with_grippers.py — re-estimate norm stats for
folding under ``--include_gripper_in_scene true``.

What it does: stream the production train shards through the data pipeline
in TEST mode (no augmentation, deterministic) with the merge transform
enabled, then Welford-accumulate the two things that actually shift when
gripper joins scene:

  - ``scene_features``     (B, 1, Ns, 14) → per-feature mean/var
  - ``gt_scene_flows_relative`` (B, T, Ns, 3) → per-timestep mean/var

Robot features are unchanged by the merge, so we copy their stats from the
existing norm_stats.json verbatim. Output: a new ``norm_stats.json`` in
``--out-dir`` that can be pointed at via ``--norm_stats_path``.

Usage:
    python recompute_norm_stats_with_grippers.py \\
        --data-dir /data/group_data/.../folding_data_pointworld \\
        --out-dir  /data/user_data/mbronars/folding_grippers_norm_stats \\
        --num-batches 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

_orig_argv = sys.argv
sys.argv = [sys.argv[0]]
from arguments import parse_args as pw_parse_args                  # noqa: E402
sys.argv = _orig_argv

from dataset_components.dataloader import build_dataloader         # noqa: E402
from npz_to_pointworld_wds import WelfordAccumulator               # noqa: E402

T = 11  # CONTEXT_HORIZON(1) + PRED_HORIZON(10)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data-dir", required=True,
                   help="Production dataset root (contains norm_stats/, train/, test/)")
    p.add_argument("--out-dir", required=True,
                   help="Where to write the new norm_stats.json")
    p.add_argument("--num-batches", type=int, default=2000,
                   help="Train batches to stream (default 2000 → ~8k clips)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grid-size", type=float, default=0.001)
    p.add_argument("--max-scene-points", type=int, default=16000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    cli = parse_cli()
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old_stats_path = Path(cli.data_dir) / "norm_stats" / "norm_stats.json"
    if not old_stats_path.is_file():
        raise FileNotFoundError(f"old norm_stats not found: {old_stats_path}")
    old_stats = json.loads(old_stats_path.read_text())

    # Build a full PointWorld args namespace mirroring how the trainer would
    # configure the data path -- but in test mode with the merge transform on.
    args = pw_parse_args(skip_command_line=True)
    args.data_dirs           = [cli.data_dir]
    args.norm_stats_path     = str(old_stats_path.parent)
    args.domains             = ["folding"]
    args.batch_size          = cli.batch_size
    args.num_workers         = cli.num_workers
    args.eval_num_workers    = cli.num_workers
    args.distributed         = False
    args.seed                = cli.seed
    args.grid_size           = cli.grid_size
    args.max_scene_points    = cli.max_scene_points
    args.deterministic_train = False
    args.deterministic_data  = False
    args.include_gripper_in_scene = True
    args.robot_features = [
        "robot_flows", "robot_normals", "gripper_open",
        "robot_velocity", "robot_acceleration",
    ]
    args.scene_features = ["scene_flows", "scene_normals", "dist2robot"]

    print(f"streaming train shards under merge transform ...")
    loader, info = build_dataloader(
        args=args, mode="test", rank=0, world_size=1,
        force_resampled_eval=True, override_splits="train",
    )
    assert loader is not None, "no train loader built"
    print(f"loader ready: {info.get('total_samples', '?')} samples available")

    # 14 = 3 (flows) + 11 (dist2robot). Scene_normals is silently skipped for
    # folding so the per-point scene_features dim stays 14 even with gripper
    # in scene.
    scene_acc = WelfordAccumulator(14)
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(T)]

    n_seen = 0
    for i, batch in enumerate(loader):
        if i >= cli.num_batches:
            break
        # scene_features only stores frame 0 (gather_features semantics).
        sf  = batch["scene_features"].numpy()[:, 0]            # (B, Ns, 14)
        scene_acc.update(sf.reshape(-1, 14))
        gtr = batch["gt_scene_flows_relative"].numpy()         # (B, T, Ns, 3)
        for t in range(T):
            gt_rel_accums[t].update(gtr[:, t].reshape(-1, 3))
        n_seen += sf.shape[0]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1:5d}/{cli.num_batches} batches  ({n_seen} samples, "
                  f"{scene_acc.n} scene-points so far)")

    # Patch the existing norm_stats: replace scene_features + per-timestep
    # gt_scene_flows_relative stats. Robot stats and everything else stay.
    new_stats = old_stats
    ns = new_stats["statistics"]["folding"]
    ns["scene_features"]["mean"]     = scene_acc.mean.tolist()
    ns["scene_features"]["variance"] = scene_acc.variance.tolist()

    pts = new_stats["per_timestep_statistics"]["folding"]["gt_scene_flows_relative"]
    for t in range(T):
        pts[f"timestep_{t}"]["mean"]     = gt_rel_accums[t].mean.tolist()
        pts[f"timestep_{t}"]["variance"] = gt_rel_accums[t].variance.tolist()

    new_path = out_dir / "norm_stats.json"
    new_path.write_text(json.dumps(new_stats, indent=2))
    print(f"\nWrote {new_path}")
    print(f"  samples seen           : {n_seen}")
    print(f"  scene-points accumulated: {scene_acc.n:,}")
    print(f"  new scene_features mean: "
          + ", ".join(f"{v:+.4f}" for v in scene_acc.mean))
    print(f"  old scene_features mean: "
          + ", ".join(f"{v:+.4f}" for v in old_stats['statistics']['folding']['scene_features']['mean']))


if __name__ == "__main__":
    main()
