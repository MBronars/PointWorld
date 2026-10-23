#!/usr/bin/env python3
"""
pickup_to_pointworld_wds.py — Convert the 4-success Pickup-Pose WDS shards
(robert_pickup_data_{place,convex,pick,upright}_success) into PointWorld
WebDataset shards for the new ``pickup`` domain.

Source layout per .tar (B = 4 demos per WDS sample, T = 40):
  particle_pos.pyd       (B, T, 1183, 3)  float32  — obj-first
  obj_ids.pyd            (B, 1183)        int32    — 0=object, 1=gripper
  valid_particles.pyd    (B, T, 1183)     bool
  ... (link_pos / link_quat / joint_*: ignored for PointWorld folding-style)

Output layout (one PointWorld clip per WDS sample):
  view0_scene_flows.npy        (T, 512,  3)  float32   object particles, centered
  view0_scene_visibility.npy   (T, 512)      bool
  robot_flows.npy              (T, 671,  3)  float32   gripper particles, centered
  joint_positions.npy          (T, 1)        float32   placeholder zeros
  right_gripper_open.npy       (T, 1)        float32   placeholder zeros
  left_gripper_open.npy        (T, 1)        float32   placeholder zeros

Splits:
  - Train/val is shard-level, last 10 % of each root's gpu0/*.tar = val.
  - Blocklist drops 299 specific (shard, key, elem) triples (matches v9 / PhysCtrl).

Reuses Welford / ShardWriter / robot_feature_blocks helpers from
``npz_to_pointworld_wds.py`` for compatibility with the standard PointWorld
training pipeline.

Usage:
    python pickup_to_pointworld_wds.py \\
        --out /data/group_data/katefgroup/VLA/3DPCWM/pickup_data_pointworld
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

import numpy as np
from tqdm import tqdm

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

DOMAIN = "pickup"

DEFAULT_ROOTS = [
    "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_place_success",
    "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_convex_success",
    "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success",
    "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_upright_success",
]
DEFAULT_BLOCKLIST = (
    "/data/user_data/robertre/particle_world_model_pickup/"
    "reports/phantom_blocklists/robert_pickup_data_4success_combined.json"
)
VAL_FRACTION = 0.10

# Pickup paddle/Panda fingers don't open/close at a particle-level granularity
# we model in robot_features; the gripper rectangle is just rigid points.
DEFAULT_ROBOT_FEATURES = (
    "robot_flows,robot_normals,robot_velocity,robot_acceleration"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shard discovery + splits + blocklist
# ─────────────────────────────────────────────────────────────────────────────

def _collect_shards_per_root(roots) -> dict[str, dict[str, list[str]]]:
    """Per-root train/val shard lists, 90/10 by shard tail (matches v9/PhysCtrl)."""
    out = {}
    for root in roots:
        # gpu0/*.tar is the canonical layout per the docs.
        shards = sorted(glob.glob(f"{root}/gpu0/*.tar")
                        + glob.glob(f"{root}/gpu0/**/*.tar", recursive=True))
        shards = [s for s in shards if "/_quarantine/" not in s]
        if not shards:
            raise FileNotFoundError(f"No shards under {root}/gpu0")
        n_val = max(1, int(round(len(shards) * VAL_FRACTION)))
        out[root] = {
            "train": shards[:-n_val],
            "val":   shards[-n_val:],
        }
    return out


def _load_blocklist(path: str | None) -> set:
    """{(shard_abspath, key, elem)} set of trajectories to drop, or empty set
    if the file is missing or unreadable (e.g. permission denied)."""
    if not path:
        return set()
    try:
        with open(path) as f:
            bl = json.load(f).get("blocklist", [])
        return {(e["shard"], e["key"], int(e["elem"])) for e in bl}
    except (FileNotFoundError, PermissionError, OSError) as exc:
        print(f"  blocklist unavailable ({type(exc).__name__}): {exc}\n"
              f"  proceeding WITHOUT blocklist (0.56% extra trajs included)")
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Per-shard iteration → per-demo clips
# ─────────────────────────────────────────────────────────────────────────────

def _iter_demos_from_shard(tar_path: str, blocklist: set):
    """Yield (key, elem, particle_pos[T,N,3], valid_particles[T,N], obj_ids[N])
    for each non-blocked demo in a pickup WDS shard."""
    with tarfile.open(tar_path) as tar:
        groups: dict[str, list[tuple[str, tarfile.TarInfo]]] = {}
        for m in tar.getmembers():
            stem, _, ext = m.name.partition(".")
            groups.setdefault(stem, []).append((ext, m))
        for stem in sorted(groups.keys()):
            try:
                # Read only the three blobs we actually need; ignore link/joint/etc.
                want = {"particle_pos.pyd", "obj_ids.pyd", "valid_particles.pyd"}
                data = {}
                for ext, m in groups[stem]:
                    if ext in want:
                        data[ext] = pickle.loads(tar.extractfile(m).read())
                pp  = data["particle_pos.pyd"]      # (B, T, N, 3)
                vp  = data["valid_particles.pyd"]   # (B, T, N) bool
                obj = data["obj_ids.pyd"]           # (B, N) int32
            except KeyError as e:
                # missing one of the three needed fields → skip
                continue
            B = pp.shape[0]
            for b in range(B):
                if (tar_path, stem, b) in blocklist:
                    continue
                yield stem, b, pp[b], vp[b], obj[b]


def _make_clip(particle_pos: np.ndarray,
               valid_particles: np.ndarray,
               obj_ids: np.ndarray,
               start: int,
               clip_len: int) -> dict:
    """Build one centered clip dict from a sliced demo array."""
    end = start + clip_len
    pp = particle_pos[start:end]
    vp = valid_particles[start:end]

    obj_mask  = (obj_ids == 0)
    grip_mask = (obj_ids == 1)
    cloth = pp[:, obj_mask].astype(np.float32)            # (T, 512, 3) object scene
    robot = pp[:, grip_mask].astype(np.float32)           # (T, 671, 3) gripper
    cloth_vis = vp[:, obj_mask].astype(bool)

    shift = np.concatenate([cloth[0], robot[0]], axis=0).mean(axis=0).astype(np.float32)
    cloth = (cloth - shift).astype(np.float32)
    robot = (robot - shift).astype(np.float32)

    return {
        "cloth":      cloth,
        "robot":      robot,
        "cloth_vis":  cloth_vis,
        "open_right": np.zeros((clip_len, 1), dtype=np.float32),
        "open_left":  np.zeros((clip_len, 1), dtype=np.float32),
        "joint":      np.zeros((clip_len, 1), dtype=np.float32),
    }


def extract_clips_from_demo(particle_pos: np.ndarray,
                            valid_particles: np.ndarray,
                            obj_ids: np.ndarray,
                            args: argparse.Namespace) -> list[dict]:
    T_raw = particle_pos.shape[0]
    sub = slice(args.skip_frames, None, args.stride)
    pp = particle_pos[sub]; vp = valid_particles[sub]
    T_sub = pp.shape[0]
    if T_sub < args.clip_len:
        return []
    n_starts = T_sub - args.clip_len + 1
    starts = np.arange(n_starts, dtype=np.int64)
    n_clips = max(1, n_starts // args.clip_spacing)
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


def _accumulate(clip, clip_len, robot_features, gt_rel_accums, scene_accum, robot_accum):
    cloth, robot = clip["cloth"], clip["robot"]
    gt_rel = cloth - cloth[0:1]
    for t in range(clip_len):
        gt_rel_accums[t].update(gt_rel[t])
    dist2robot = np.stack(
        [nearest_robot_dist(cloth[0], robot[t]) for t in range(clip_len)], axis=0,
    )
    scene_accum.update(np.concatenate([cloth[0], dist2robot.T], axis=-1))
    blocks = robot_feature_blocks(robot, clip)
    robot_accum.update(np.concatenate([blocks[n] for n in robot_features], axis=-1))


def _write_norm_stats(out_path, gt_rel_accums, scene_accum, robot_accum, clip_len):
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
     scene_feat_dim, robot_feat_dim, seed, blocklist) = task
    writer = ShardWriter(Path(args.out), split, args.samples_per_shard, tag=tag)
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(args.clip_len)]
    scene_accum   = WelfordAccumulator(scene_feat_dim)
    robot_accum   = WelfordAccumulator(robot_feat_dim)
    n_failed = n_blocked = 0
    for path in shard_paths:
        try:
            for key, elem, pp, vp, obj in _iter_demos_from_shard(path, blocklist):
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

def process(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    robot_features = [s.strip() for s in args.robot_features.split(",") if s.strip()]
    bad = [f for f in robot_features if f not in ROBOT_FEATURE_DIMS]
    if bad:
        raise ValueError(f"Unknown robot feature(s) {bad}; valid: {sorted(ROBOT_FEATURE_DIMS)}")
    robot_feat_dim = sum(ROBOT_FEATURE_DIMS[f] for f in robot_features)
    scene_feat_dim = 3 + args.clip_len

    roots = args.roots if args.roots else DEFAULT_ROOTS
    per_root = _collect_shards_per_root(roots)
    train_shards = []
    val_shards = []
    for root, splits in per_root.items():
        train_shards.extend(splits["train"])
        val_shards.extend(splits["val"])
        print(f"  {Path(root).name}: train={len(splits['train'])}  val={len(splits['val'])}")
    if args.max_train_shards > 0:
        train_shards = train_shards[: args.max_train_shards]
    print(f"\nTotal train shards: {len(train_shards)}  val shards: {len(val_shards)}")

    blocklist = _load_blocklist(args.blocklist)
    print(f"Blocklist entries: {len(blocklist)}")

    n_workers = args.num_workers if args.num_workers > 0 else len(os.sched_getaffinity(0))
    n_workers = max(1, min(n_workers, max(1, len(train_shards))))
    n_chunks  = max(1, min(n_workers * 4, max(1, len(train_shards))))
    chunks    = [train_shards[i::n_chunks] for i in range(n_chunks)]
    print(f"clip_len={args.clip_len}  stride={args.stride}  skip={args.skip_frames}  "
          f"clip_spacing={args.clip_spacing}")
    print(f"robot_features={robot_features}")
    print(f"scene_feat_dim={scene_feat_dim}  robot_feat_dim={robot_feat_dim}")
    print(f"workers={n_workers}  train_chunks={n_chunks}\n", flush=True)

    tasks = [
        (args, robot_features, ch, f"w{i:03d}_", "train",
         scene_feat_dim, robot_feat_dim, args.seed + 1 + i, blocklist)
        for i, ch in enumerate(chunks) if ch
    ]
    if val_shards:
        # The 'test' split label matches what gather_shard_paths expects.
        tasks.append((args, robot_features, val_shards, "v000_", "test",
                      scene_feat_dim, robot_feat_dim, args.seed + 100000, blocklist))

    results = []
    if n_workers == 1:
        for t in tqdm(tasks, desc="chunks"):
            results.append(_process_chunk(t))
    else:
        with Pool(n_workers) as pool:
            for r in tqdm(pool.imap_unordered(_process_chunk, tasks),
                          total=len(tasks), desc="chunks"):
                results.append(r)

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
    print(f"Output -> {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", required=True)
    p.add_argument("--roots", nargs="+", default=None,
                   help=f"Override input roots (default: {DEFAULT_ROOTS})")
    p.add_argument("--blocklist", default=DEFAULT_BLOCKLIST,
                   help="Phantom-trajectory blocklist JSON (set to '' to skip)")
    g = p.add_argument_group("clip extraction")
    g.add_argument("--clip-len", type=int, default=CLIP_LEN)
    g.add_argument("--stride", type=int, default=1,
                   help="Native stride (pickup T=40 is already at the rate "
                        "v9/PhysCtrl train on — keep stride 1).")
    g.add_argument("--skip-frames", type=int, default=0)
    g.add_argument("--clip-spacing", type=int, default=15,
                   help="Avg start-frame spacing between clips per demo. T=40 "
                        "gives 30 possible starts; spacing 15 → ~2 clips/demo.")
    g = p.add_argument_group("features")
    g.add_argument("--robot-features", default=DEFAULT_ROBOT_FEATURES)
    g = p.add_argument_group("output")
    g.add_argument("--samples-per-shard", type=int, default=200)
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-train-shards", type=int, default=-1,
                   help="Cap on input train shards (for testing).")
    return p.parse_args()


if __name__ == "__main__":
    process(parse_args())
