#!/usr/bin/env python3
"""
npz_to_pointworld_wds_with_grippers.py
SIM1 cloth-folding .npz -> PointWorld WebDataset, **gripper points
concatenated INTO scene** (so they are part of the predicted particle set).

Differences vs ``npz_to_pointworld_wds.py``:

  - ``view0_scene_flows.npy`` is ``(T, N_cloth + N_gripper, 3)`` — the cloth
    vertices first, then the 2000 bimanual gripper points (left then right).
  - ``view0_scene_visibility.npy`` is shaped accordingly (all True).
  - ``robot_flows.npy`` is unchanged: the same 2000 gripper points across T
    frames, kept as conditioning so the model still has a future-action signal.
  - Two new scalars per sample, ``n_cloth.npy`` and ``n_gripper.npy``, so eval
    code can split the predicted scene back into cloth vs gripper subsets.

Everything else (clip length, temporal stride, skip frames, val list,
importance sampling, shard layout, norm-stats schema) is identical to the
production converter; this script imports the helpers from it so they stay in
sync.

Usage:
    python npz_to_pointworld_wds_with_grippers.py \\
        --root /data/group_data/.../sim1_cloth_diverse_v1 \\
        --out  /data/group_data/.../folding_data_pointworld_grippers
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from npz_to_pointworld_wds import (
    CATEGORIES,
    CLIP_LEN,
    DOMAIN,
    ROBOT_FEATURE_DIMS,
    ShardWriter,
    VAL_RELPATHS,
    WelfordAccumulator,
    _load_traj,
    _window_starts_and_weights,
    discover_trajectories,
    nearest_robot_dist,
    npy_bytes,
    robot_feature_blocks,
    write_norm_stats,
)

# The folding model's robot feature list (no robot_colors). MUST match
# train.py --robot_features exactly.
FOLDING_ROBOT_FEATURES = (
    "robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration"
)


# ─────────────────────────────────────────────────────────────────────────────
# Clip-level helpers — same skeleton as the production converter, with the
# scene assembly tweak.
# ─────────────────────────────────────────────────────────────────────────────

def _make_clip(cloth_sub, robot_sub, joint_sub, open_sub, start, args, rng):
    """One centered clip dict; scene = [cloth, gripper] concatenated."""
    end = start + args.clip_len
    cloth_w = cloth_sub[start:end]    # (T, Nc, 3)
    robot_w = robot_sub[start:end]    # (T, 2000, 3)
    joint_w = joint_sub[start:end]
    open_w  = open_sub[start:end]     # (T, 2)  [left, right]

    if args.n_cloth > 0:
        Nc = cloth_w.shape[1]
        cidx = np.sort(rng.choice(Nc, size=args.n_cloth, replace=Nc < args.n_cloth))
        cloth_w = cloth_w[:, cidx]
    if args.n_robot > 0:
        Nr = robot_w.shape[1]
        ridx = np.sort(rng.choice(Nr, size=args.n_robot, replace=Nr < args.n_robot))
        robot_w = robot_w[:, ridx]

    # Center on the mean of scene+robot at t=0. With gripper now also in scene,
    # the "scene+robot" union still has each gripper point once-per-set; the
    # arithmetic mean stays well-defined and matches center_shift semantics.
    scene_w = np.concatenate([cloth_w, robot_w], axis=1)    # (T, Nc+Nr, 3)
    shift = scene_w[0].mean(axis=0).astype(np.float32)
    scene_w = (scene_w - shift).astype(np.float32)
    robot_w = (robot_w - shift).astype(np.float32)

    return {
        "scene":      scene_w,                                   # (T, Nc+Nr, 3)
        "robot":      robot_w,                                   # (T, Nr, 3)
        "n_cloth":    int(cloth_w.shape[1]),
        "n_gripper":  int(robot_w.shape[1]),
        "joint":      joint_w.astype(np.float32),
        "open_left":  open_w[:, 0:1].astype(np.float32),
        "open_right": open_w[:, 1:2].astype(np.float32),
    }


def _write_clip(clip, writer):
    T, Ns, _ = clip["scene"].shape
    writer.write({
        "view0_scene_flows.npy":      npy_bytes(clip["scene"]),
        "view0_scene_visibility.npy": npy_bytes(np.ones((T, Ns), dtype=bool)),
        "robot_flows.npy":            npy_bytes(clip["robot"]),
        "joint_positions.npy":        npy_bytes(clip["joint"]),
        "right_gripper_open.npy":     npy_bytes(clip["open_right"]),
        "left_gripper_open.npy":      npy_bytes(clip["open_left"]),
        # Tags so downstream code can split scene back into cloth vs gripper.
        "n_cloth.npy":   npy_bytes(np.array(clip["n_cloth"],   dtype=np.int32)),
        "n_gripper.npy": npy_bytes(np.array(clip["n_gripper"], dtype=np.int32)),
    })


def _accumulate(clip, clip_len, robot_features, gt_rel_accums, scene_accum, robot_accum):
    scene, robot = clip["scene"], clip["robot"]                  # scene already includes gripper
    gt_rel = scene - scene[0:1]
    for t in range(clip_len):
        gt_rel_accums[t].update(gt_rel[t])
    # scene_features @ t=0: [flows(3), dist2robot(T)]. dist2robot is per scene
    # point; gripper-scene points will have ~0 distance to themselves in
    # robot — the model can use that as a "this is a gripper" signal.
    dist2robot = np.stack(
        [nearest_robot_dist(scene[0], robot[t]) for t in range(clip_len)], axis=0,
    )
    scene_accum.update(np.concatenate([scene[0], dist2robot.T], axis=-1))
    blocks = robot_feature_blocks(robot, clip)
    robot_accum.update(np.concatenate([blocks[n] for n in robot_features], axis=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory -> clips
# ─────────────────────────────────────────────────────────────────────────────

def extract_clips(path, args, rng, is_val):
    cloth, left, right, joint_q, openness = _load_traj(path)
    T = cloth.shape[0]
    sub = slice(args.skip_frames, None, args.stride)
    cloth_sub = cloth[sub]
    robot_sub = np.concatenate([left[sub], right[sub]], axis=1)
    joint_sub = joint_q[sub]
    open_sub  = openness[sub]
    T_sub = cloth_sub.shape[0]
    if T_sub < args.clip_len:
        return []
    starts, probs = _window_starts_and_weights(cloth_sub, args)
    if len(starts) == 0:
        return []
    n_clips = max(1, len(starts) // args.clip_spacing)
    if is_val:
        chosen = starts[:: max(1, len(starts) // n_clips)][:n_clips]
    elif probs is None:
        chosen = rng.choice(starts, size=n_clips, replace=True)
    else:
        chosen = rng.choice(starts, size=n_clips, replace=True, p=probs)
    return [_make_clip(cloth_sub, robot_sub, joint_sub, open_sub, int(s), args, rng)
            for s in chosen]


def _process_chunk(task):
    (args, robot_features, paths, tag, split, is_val,
     scene_feat_dim, robot_feat_dim, seed) = task
    writer = ShardWriter(Path(args.out), split, args.samples_per_shard, tag=tag)
    rng = np.random.default_rng(seed)
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(args.clip_len)]
    scene_accum   = WelfordAccumulator(scene_feat_dim)
    robot_accum   = WelfordAccumulator(robot_feat_dim)
    n_failed = 0
    for path in paths:
        try:
            clips = extract_clips(path, args, rng, is_val=is_val)
        except (PermissionError, FileNotFoundError, OSError, KeyError, ValueError) as exc:
            n_failed += 1
            print(f"  SKIP ({split}) {path}: {type(exc).__name__}: {exc}", flush=True)
            continue
        for clip in clips:
            _write_clip(clip, writer)
            if not is_val:
                _accumulate(clip, args.clip_len, robot_features,
                            gt_rel_accums, scene_accum, robot_accum)
    writer.close()
    return {"split": split, "count": writer.total_samples, "n_failed": n_failed,
            "gt_rel": gt_rel_accums, "scene": scene_accum, "robot": robot_accum}


def process(args):
    root = Path(args.root)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.clip_len != CLIP_LEN:
        print(f"WARNING: --clip-len={args.clip_len} != PointWorld release horizon "
              f"({CLIP_LEN}). Training will fail unless the horizon is changed too.")

    robot_features = [s.strip() for s in args.robot_features.split(",") if s.strip()]
    bad = [f for f in robot_features if f not in ROBOT_FEATURE_DIMS]
    if bad:
        raise ValueError(f"Unknown robot feature(s) {bad}; valid: {sorted(ROBOT_FEATURE_DIMS)}")
    robot_feat_dim = sum(ROBOT_FEATURE_DIMS[f] for f in robot_features)
    scene_feat_dim = 3 + args.clip_len  # [scene_flows(3), dist2robot(T)]

    train_paths, val_paths = discover_trajectories(root)
    if args.max_trajs > 0:
        train_paths = train_paths[: args.max_trajs]

    n_workers = args.num_workers if args.num_workers > 0 else len(os.sched_getaffinity(0))
    n_workers = max(1, min(n_workers, max(1, len(train_paths))))
    n_chunks  = max(1, min(n_workers * 4, max(1, len(train_paths))))
    chunks    = [train_paths[i::n_chunks] for i in range(n_chunks)] if train_paths else []

    print(f"clip_len={args.clip_len}  stride={args.stride}  skip={args.skip_frames}")
    print(f"robot_features={robot_features}")
    print(f"scene_feat_dim={scene_feat_dim}  robot_feat_dim={robot_feat_dim}")
    print(f"workers={n_workers}  train_chunks={n_chunks}", flush=True)

    tasks = [
        (args, robot_features, ch, f"w{i:03d}_", "train", False,
         scene_feat_dim, robot_feat_dim, args.seed + 1 + i)
        for i, ch in enumerate(chunks)
    ]
    if val_paths:
        tasks.append((args, robot_features, val_paths, "v000_", "test", True,
                      scene_feat_dim, robot_feat_dim, args.seed + 100000))

    results = []
    if n_workers == 1:
        for task in tqdm(tasks, desc="chunks"):
            results.append(_process_chunk(task))
    else:
        with Pool(n_workers) as pool:
            for r in tqdm(pool.imap_unordered(_process_chunk, tasks),
                          total=len(tasks), desc="chunks"):
                results.append(r)

    # Merge worker stats
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(args.clip_len)]
    scene_accum   = WelfordAccumulator(scene_feat_dim)
    robot_accum   = WelfordAccumulator(robot_feat_dim)
    train_count = test_count = n_failed = 0
    for r in results:
        n_failed += r["n_failed"]
        if r["split"] == "train":
            train_count += r["count"]
            for t in range(args.clip_len):
                gt_rel_accums[t].merge(r["gt_rel"][t])
            scene_accum.merge(r["scene"])
            robot_accum.merge(r["robot"])
        else:
            test_count += r["count"]

    if n_failed:
        print(f"\nWARNING: {n_failed} trajectories could not be read.")
    if train_count == 0:
        raise RuntimeError("No training clips were written — check data access.")

    metadata = {"train": {"processed_count": train_count},
                "test":  {"processed_count": test_count}}
    (out_path / "metadata_rank0.json").write_text(json.dumps(metadata, indent=2))
    write_norm_stats(out_path, gt_rel_accums, scene_accum, robot_accum)

    print(f"\n  Train clips : {train_count}")
    print(f"  Test  clips : {test_count}")
    print(f"  scene_feat_dim: {scene_feat_dim}   robot_feat_dim: {robot_feat_dim}")
    print(f"  Output       : {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--root", required=True, help="sim1_cloth_diverse_v1 dataset root")
    p.add_argument("--out",  required=True, help="Output dataset root")
    g = p.add_argument_group("clip extraction")
    g.add_argument("--clip-len", type=int, default=CLIP_LEN)
    g.add_argument("--stride", type=int, default=2)
    g.add_argument("--skip-frames", type=int, default=30)
    g.add_argument("--clip-spacing", type=int, default=15)
    g = p.add_argument_group("point clouds")
    g.add_argument("--n-cloth", type=int, default=0, help="0 = keep all cloth verts")
    g.add_argument("--n-robot", type=int, default=0, help="0 = keep all 2000 gripper points")
    g = p.add_argument_group("features")
    g.add_argument("--robot-features", default=FOLDING_ROBOT_FEATURES)
    g = p.add_argument_group("importance sampling")
    g.add_argument("--uniform", action="store_true")
    g.add_argument("--alpha", type=float, default=2.0)
    g.add_argument("--motion-floor-mm", type=float, default=0.1)
    g.add_argument("--motion-score-particles", type=int, default=256)
    g = p.add_argument_group("output")
    g.add_argument("--samples-per-shard", type=int, default=200)
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-trajs", type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    process(parse_args())
