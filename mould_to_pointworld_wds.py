#!/usr/bin/env python3
"""
mould_to_pointworld_wds.py — Convert ``deformable_mould_10k`` WDS shards into
PointWorld WebDataset shards for the new ``mould`` domain.

Input layout (one .tar entry per WDS sample, B demos stacked):
  particle_pos.pyd     (B, T, N_pts, 3)   float32   — gripper first, then blob
  valid_particles.pyd  (B, T, N_pts)      bool
  obj_ids.pyd          (B, N_pts)         int32     — 1=gripper, 0=blob
  ... (material_properties / start_state / end_state / absolute_targets / idx)

Output layout (one PointWorld clip per WDS sample, no batch dim):
  view0_scene_flows.npy        (T, N_blob, 3)    float32   blob particles, centered
  view0_scene_visibility.npy   (T, N_blob)       bool
  robot_flows.npy              (T, N_gripper, 3) float32   gripper points, centered
  joint_positions.npy          (T, 1)            float32   placeholder (zeros)
  right_gripper_open.npy       (T, 1)            float32   placeholder (zeros)

This matches the folding shard layout (folding has gripper points baked into
WDS, no URDF involved). Domain wiring in robot.py routes mould through
``_get_robot_flows_mould`` which just reads ``sample['robot_flows']`` directly.

Stats:
  scene_feat_dim = 3 (scene_flows) + clip_len (dist2robot)
  robot_features default = robot_flows + robot_normals + robot_velocity +
                           robot_acceleration  (= 12-D, no gripper_open)

Usage:
    python mould_to_pointworld_wds.py \\
        --input /data/group_data/katefgroup/VLA/3DPCWM/deformable_mould_10k \\
        --out   /data/group_data/katefgroup/VLA/3DPCWM/mould_data_pointworld
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pickle
import tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import numpy as np
import webdataset as wds
from tqdm import tqdm

# Reuse production helpers — only thing we replace is `DOMAIN` and the
# norm_stats writer (so it lands under "mould", not "folding").
from npz_to_pointworld_wds import (
    CLIP_LEN,
    ROBOT_FEATURE_DIMS,
    ShardWriter,
    WelfordAccumulator,
    finite_diff,
    nearest_robot_dist,
    npy_bytes,
    robot_feature_blocks,
)

DOMAIN = "mould"

# Mould paddle is a 256-point rigid box; blob is 1024 deformable particles.
N_GRIPPER = 256
N_BLOB = 1024

# Pushing-style robot feature list (no gripper_open — the mould paddle is a
# solid box, doesn't open/close). Must match train.py --robot_features.
DEFAULT_ROBOT_FEATURES = (
    "robot_flows,robot_normals,robot_velocity,robot_acceleration"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shard iteration
# ─────────────────────────────────────────────────────────────────────────────

def _iter_demos_from_shard(tar_path: str):
    """Yield (key, particle_pos[T,N,3], valid_particles[T,N], obj_ids[N]) for
    every demo in a mould WDS shard."""
    with tarfile.open(tar_path) as tar:
        # Group tar members by stem (one WDS sample = many .pyd files).
        groups: dict[str, list[tuple[str, tarfile.TarInfo]]] = {}
        for m in tar.getmembers():
            stem, _, ext = m.name.partition(".")
            groups.setdefault(stem, []).append((ext, m))
        for stem in sorted(groups.keys()):
            data: dict[str, np.ndarray] = {}
            for ext, m in groups[stem]:
                data[ext] = pickle.loads(tar.extractfile(m).read())
            pp = data["particle_pos.pyd"]           # (B, T, N, 3)
            vp = data["valid_particles.pyd"]        # (B, T, N) bool
            obj = data["obj_ids.pyd"]               # (B, N) int32
            B = pp.shape[0]
            for b in range(B):
                yield f"{stem}_d{b}", pp[b], vp[b], obj[b]


# ─────────────────────────────────────────────────────────────────────────────
# Clip extraction (per demo)
# ─────────────────────────────────────────────────────────────────────────────

def _make_clip(particle_pos: np.ndarray,
               valid_particles: np.ndarray,
               obj_ids: np.ndarray,
               start: int,
               clip_len: int) -> dict:
    """Build one centered clip dict from a sliced demo array.

    particle_pos: (T_sub, N, 3) already after skip/stride.
    """
    end = start + clip_len
    pp = particle_pos[start:end]            # (T, N, 3)
    vp = valid_particles[start:end]         # (T, N)

    blob_mask = (obj_ids == 0)
    grip_mask = (obj_ids == 1)
    cloth = pp[:, blob_mask].astype(np.float32)         # (T, N_blob, 3)
    robot = pp[:, grip_mask].astype(np.float32)         # (T, N_grip, 3)
    cloth_vis = vp[:, blob_mask].astype(bool)            # (T, N_blob)

    # Center on the mean of scene+robot at t=0 (matches center_shift semantics
    # used by every other PointWorld converter).
    shift = np.concatenate([cloth[0], robot[0]], axis=0).mean(axis=0).astype(np.float32)
    cloth = (cloth - shift).astype(np.float32)
    robot = (robot - shift).astype(np.float32)

    return {
        "cloth": cloth,
        "robot": robot,
        "cloth_vis": cloth_vis,
        # Placeholders so the pipeline's gather_features works the same as for
        # pushing / folding (robot_feature_blocks needs an `open_*` entry to
        # build the gripper_open block — even though we don't ship it as a
        # robot feature).
        "open_right": np.zeros((clip_len, 1), dtype=np.float32),
        "open_left":  np.zeros((clip_len, 1), dtype=np.float32),
        "joint":      np.zeros((clip_len, 1), dtype=np.float32),
    }


def extract_clips_from_demo(particle_pos: np.ndarray,
                            valid_particles: np.ndarray,
                            obj_ids: np.ndarray,
                            args: argparse.Namespace) -> list[dict]:
    """All-windows extraction (deterministic, evenly spaced by clip_spacing)."""
    T_raw = particle_pos.shape[0]
    sub = slice(args.skip_frames, None, args.stride)
    pp = particle_pos[sub]
    vp = valid_particles[sub]
    T_sub = pp.shape[0]
    if T_sub < args.clip_len:
        return []
    n_starts = T_sub - args.clip_len + 1
    starts = np.arange(n_starts, dtype=np.int64)
    n_clips = max(1, n_starts // args.clip_spacing)
    # Evenly spaced through the demo.
    chosen = starts[:: max(1, n_starts // n_clips)][:n_clips]
    return [_make_clip(pp, vp, obj_ids, int(s), args.clip_len) for s in chosen]


# ─────────────────────────────────────────────────────────────────────────────
# Write + stats
# ─────────────────────────────────────────────────────────────────────────────

def _write_clip(clip: dict, writer: ShardWriter) -> None:
    writer.write({
        "view0_scene_flows.npy":      npy_bytes(clip["cloth"]),
        "view0_scene_visibility.npy": npy_bytes(clip["cloth_vis"]),
        "robot_flows.npy":            npy_bytes(clip["robot"]),
        "joint_positions.npy":        npy_bytes(clip["joint"]),
        "right_gripper_open.npy":     npy_bytes(clip["open_right"]),
        "left_gripper_open.npy":      npy_bytes(clip["open_left"]),
    })


def _accumulate(clip: dict,
                clip_len: int,
                robot_features: list,
                gt_rel_accums: list,
                scene_accum: WelfordAccumulator,
                robot_accum: WelfordAccumulator) -> None:
    cloth, robot = clip["cloth"], clip["robot"]
    # gt_scene_flows_relative per-timestep stats
    gt_rel = cloth - cloth[0:1]
    for t in range(clip_len):
        gt_rel_accums[t].update(gt_rel[t])
    # scene_features @ t=0: [flows(3), dist2robot(T)]
    dist2robot = np.stack(
        [nearest_robot_dist(cloth[0], robot[t]) for t in range(clip_len)], axis=0,
    )
    scene_accum.update(np.concatenate([cloth[0], dist2robot.T], axis=-1))
    # robot_features in the order specified
    blocks = robot_feature_blocks(robot, clip)
    robot_accum.update(np.concatenate([blocks[n] for n in robot_features], axis=-1))


def _write_norm_stats(out_path: Path,
                      gt_rel_accums: list,
                      scene_accum: WelfordAccumulator,
                      robot_accum: WelfordAccumulator,
                      clip_len: int) -> None:
    per_ts = {}
    for t in range(clip_len):
        per_ts[f"timestep_{t}"] = {
            "mean":     gt_rel_accums[t].mean.tolist(),
            "variance": gt_rel_accums[t].variance.tolist(),
        }
    payload = {
        "statistics": {
            DOMAIN: {
                "robot_features": {
                    "mean":     robot_accum.mean.tolist(),
                    "variance": robot_accum.variance.tolist(),
                },
                "scene_features": {
                    "mean":     scene_accum.mean.tolist(),
                    "variance": scene_accum.variance.tolist(),
                },
            }
        },
        "per_timestep_statistics": {
            DOMAIN: {"gt_scene_flows_relative": per_ts},
        },
    }
    norm_dir = out_path / "norm_stats"
    norm_dir.mkdir(parents=True, exist_ok=True)
    (norm_dir / "norm_stats.json").write_text(json.dumps(payload, indent=2))
    print(f"Norm stats -> {norm_dir / 'norm_stats.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_chunk(task: tuple) -> dict:
    (args, robot_features, shard_paths, tag, split,
     scene_feat_dim, robot_feat_dim, seed) = task
    writer = ShardWriter(Path(args.out), split, args.samples_per_shard, tag=tag)
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(args.clip_len)]
    scene_accum   = WelfordAccumulator(scene_feat_dim)
    robot_accum   = WelfordAccumulator(robot_feat_dim)
    n_failed = 0
    for path in shard_paths:
        try:
            for key, pp, vp, obj in _iter_demos_from_shard(path):
                # Each mould demo: the original mould converter already
                # respects the 2-frame skip from h5_to_wds_push. So pp is
                # ready to go; --skip-frames here is in addition.
                clips = extract_clips_from_demo(pp, vp, obj, args)
                for clip in clips:
                    _write_clip(clip, writer)
                    if split == "train":
                        _accumulate(clip, args.clip_len, robot_features,
                                    gt_rel_accums, scene_accum, robot_accum)
        except (PermissionError, FileNotFoundError, OSError, KeyError, ValueError) as exc:
            n_failed += 1
            print(f"  SKIP ({split}) {path}: {type(exc).__name__}: {exc}", flush=True)
            continue
    writer.close()
    return {
        "split": split, "count": writer.total_samples, "n_failed": n_failed,
        "gt_rel": gt_rel_accums, "scene": scene_accum, "robot": robot_accum,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def process(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    robot_features = [s.strip() for s in args.robot_features.split(",") if s.strip()]
    bad = [f for f in robot_features if f not in ROBOT_FEATURE_DIMS]
    if bad:
        raise ValueError(f"Unknown robot feature(s) {bad}; "
                         f"valid: {sorted(ROBOT_FEATURE_DIMS)}")
    robot_feat_dim = sum(ROBOT_FEATURE_DIMS[f] for f in robot_features)
    scene_feat_dim = 3 + args.clip_len   # [scene_flows(3), dist2robot(T)]

    train_shards = sorted(glob.glob(f"{args.input}/train/*.tar"))
    test_shards  = sorted(glob.glob(f"{args.input}/test/*.tar"))
    if args.max_train_shards > 0:
        train_shards = train_shards[: args.max_train_shards]
    print(f"Train shards: {len(train_shards)}  Test shards: {len(test_shards)}")

    n_workers = args.num_workers if args.num_workers > 0 else len(os.sched_getaffinity(0))
    n_workers = max(1, min(n_workers, max(1, len(train_shards))))
    n_chunks  = max(1, min(n_workers * 4, max(1, len(train_shards))))
    chunks    = [train_shards[i::n_chunks] for i in range(n_chunks)]

    print(f"clip_len={args.clip_len}  stride={args.stride}  "
          f"skip={args.skip_frames}  clip_spacing={args.clip_spacing}")
    print(f"robot_features={robot_features}")
    print(f"scene_feat_dim={scene_feat_dim}  robot_feat_dim={robot_feat_dim}")
    print(f"workers={n_workers}  train_chunks={n_chunks}", flush=True)

    tasks = [
        (args, robot_features, ch, f"w{i:03d}_", "train",
         scene_feat_dim, robot_feat_dim, args.seed + 1 + i)
        for i, ch in enumerate(chunks) if ch
    ]
    if test_shards:
        tasks.append((args, robot_features, test_shards, "v000_", "test",
                      scene_feat_dim, robot_feat_dim, args.seed + 100000))

    results = []
    if n_workers == 1:
        for t in tqdm(tasks, desc="chunks"):
            results.append(_process_chunk(t))
    else:
        with Pool(n_workers) as pool:
            for r in tqdm(pool.imap_unordered(_process_chunk, tasks),
                          total=len(tasks), desc="chunks"):
                results.append(r)

    # ── Merge train stats ──────────────────────────────────────────────────
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
        print(f"\nWARNING: {n_failed} shards could not be read.")
    if train_count == 0:
        raise RuntimeError("No training clips were written — check data access.")

    (out / "metadata_rank0.json").write_text(json.dumps(
        {"train": {"processed_count": train_count},
         "test":  {"processed_count": test_count}}, indent=2))
    _write_norm_stats(out, gt_rel_accums, scene_accum, robot_accum, args.clip_len)

    print(f"\nTrain clips: {train_count}")
    print(f"Test  clips: {test_count}")
    print(f"scene_feat_dim={scene_feat_dim}   robot_feat_dim={robot_feat_dim}")
    print(f"Output -> {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", required=True,
                   help="Input deformable_mould_10k dataset root (train/, test/)")
    p.add_argument("--out",   required=True,
                   help="Output PointWorld dataset root")
    g = p.add_argument_group("clip extraction")
    g.add_argument("--clip-len", type=int, default=CLIP_LEN)
    g.add_argument("--stride", type=int, default=2,
                   help="Temporal subsample stride on the mould demo's "
                        "298 native frames (default 2 → 149 sub frames).")
    g.add_argument("--skip-frames", type=int, default=0,
                   help="Additional frames skipped at trajectory start "
                        "(the mould converter already drops the first 2).")
    g.add_argument("--clip-spacing", type=int, default=15,
                   help="Avg subsampled-frame spacing between clip starts.")
    g = p.add_argument_group("features")
    g.add_argument("--robot-features", default=DEFAULT_ROBOT_FEATURES,
                   help="Comma-separated robot feature list. MUST match "
                        "train.py --robot_features. Default: "
                        f"{DEFAULT_ROBOT_FEATURES}")
    g = p.add_argument_group("output")
    g.add_argument("--samples-per-shard", type=int, default=200)
    g.add_argument("--num-workers", type=int, default=0,
                   help="Parallel processes (0 = all allocated CPUs).")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-train-shards", type=int, default=-1,
                   help="Cap on number of input train shards (for testing).")
    return p.parse_args()


if __name__ == "__main__":
    process(parse_args())
