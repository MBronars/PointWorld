#!/usr/bin/env python3
"""
make_overfit_folding_data.py
Build a tiny PointWorld WebDataset from a SINGLE folding trajectory, for
overfitting / debugging.

Every valid window of the trajectory (temporal stride 2, initial skip 30) is
written as one clip — deterministically, no importance sampling — and the exact
same clip set is written to BOTH train/ and test/ so that train_loss and
val_loss track the same data (val_loss -> ~0 confirms the model can fit it).

It reuses the production converter's clip/stat helpers (npz_to_pointworld_wds.py)
so the shard layout and norm_stats are byte-compatible with train.py.

Usage:
    python make_overfit_folding_data.py \\
        --npz /data/group_data/.../sim1_cloth_diverse_v1/ik_push/000028.npz \\
        --out /data/user_data/mbronars/folding_overfit_ikpush28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# The production folding model's robot feature list (npz_to_pointworld_wds.py's
# DEFAULT_ROBOT_FEATURES additionally carries robot_colors, which the folding
# model does NOT use — so spell the 14-dim list out explicitly here).
FOLDING_ROBOT_FEATURES = (
    "robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration"
)

from npz_to_pointworld_wds import (
    CLIP_LEN,
    ROBOT_FEATURE_DIMS,
    ShardWriter,
    WelfordAccumulator,
    _accumulate,
    _load_traj,
    _make_clip,
    _write_clip,
    write_norm_stats,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--npz", required=True, help="Single trajectory .npz")
    p.add_argument("--out", required=True, help="Output dataset root")
    p.add_argument("--robot-features", default=FOLDING_ROBOT_FEATURES,
                   help="Comma-separated robot feature list (must match train.py)")
    p.add_argument("--clip-len", type=int, default=CLIP_LEN)
    p.add_argument("--stride", type=int, default=2,
                   help="Temporal subsampling of native 60Hz frames")
    p.add_argument("--skip-frames", type=int, default=30,
                   help="Native frames skipped at trajectory start")
    p.add_argument("--n-cloth", type=int, default=0,
                   help="Cloth vertices per clip (0 = keep all)")
    p.add_argument("--n-robot", type=int, default=0,
                   help="Gripper points per clip (0 = keep all 2000)")
    p.add_argument("--samples-per-shard", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    robot_features = [s.strip() for s in args.robot_features.split(",") if s.strip()]
    bad = [f for f in robot_features if f not in ROBOT_FEATURE_DIMS]
    if bad:
        raise ValueError(f"Unknown robot feature(s) {bad}")
    robot_feat_dim = sum(ROBOT_FEATURE_DIMS[f] for f in robot_features)
    scene_feat_dim = 3 + args.clip_len

    # ── Load + temporally subsample the single trajectory ──────────────────
    cloth, left, right, joint_q, openness = _load_traj(args.npz)
    sub = slice(args.skip_frames, None, args.stride)
    cloth_sub = cloth[sub]
    robot_sub = np.concatenate([left[sub], right[sub]], axis=1)   # (T_sub, 2000, 3)
    joint_sub = joint_q[sub]
    open_sub  = openness[sub]
    T_sub = cloth_sub.shape[0]
    n_starts = T_sub - args.clip_len + 1
    if n_starts <= 0:
        raise RuntimeError(f"Trajectory too short: {T_sub} subsampled frames")

    print(f"trajectory      : {args.npz}")
    print(f"native frames   : {cloth.shape[0]}   cloth verts: {cloth.shape[1]}")
    print(f"subsampled (s{args.stride},skip{args.skip_frames}): {T_sub} frames")
    print(f"clips (all windows): {n_starts}")

    # ── Build every window as a clip (deterministic, no resampling) ────────
    rng = np.random.default_rng(args.seed)
    clip_args = SimpleNamespace(
        clip_len=args.clip_len, n_cloth=args.n_cloth, n_robot=args.n_robot,
    )
    clips = [
        _make_clip(cloth_sub, robot_sub, joint_sub, open_sub, s, clip_args, rng)
        for s in range(n_starts)
    ]

    # ── Write the same clip set to train/ and test/ ────────────────────────
    for split in ("train", "test"):
        writer = ShardWriter(out, split, args.samples_per_shard, tag="of_")
        for clip in clips:
            _write_clip(clip, writer)
        writer.close()
        print(f"  wrote {writer.total_samples} clips -> {out / split}")

    # ── norm_stats over the clip set ───────────────────────────────────────
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(args.clip_len)]
    scene_accum = WelfordAccumulator(scene_feat_dim)
    robot_accum = WelfordAccumulator(robot_feat_dim)
    for clip in clips:
        _accumulate(clip, args.clip_len, robot_features,
                    gt_rel_accums, scene_accum, robot_accum)

    (out / "metadata_rank0.json").write_text(json.dumps(
        {"train": {"processed_count": len(clips)},
         "test":  {"processed_count": len(clips)}}, indent=2))
    write_norm_stats(out, gt_rel_accums, scene_accum, robot_accum)

    print(f"\nrobot_feat_dim={robot_feat_dim}  scene_feat_dim={scene_feat_dim}")
    print(f"overfit dataset -> {out}")


if __name__ == "__main__":
    main()
