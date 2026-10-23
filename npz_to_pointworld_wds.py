#!/usr/bin/env python3
"""
npz_to_pointworld_wds.py
Convert SIM1 cloth-folding per-trajectory .npz files -> PointWorld WebDataset
shards + norm_stats.json, for the `folding` domain.

This is the cloth-folding analogue of `h5_to_pointworld_wds.py` (the `pushing`
converter). Key differences vs the pushing converter:

  - Input is one .npz per trajectory (not a single .h5).
  - Scene = cloth vertices. By default every clip keeps the garment's native
    vertex count (4952 pants / 7021 short_shirt / 10580 long_shirt) -- samples
    have variable N, which PointWorld handles via collate padding + the
    scene_exists mask. `--n-cloth N` optionally sub-samples to a fixed count.
  - Robot = the bimanual gripper point clouds, which are ALREADY baked into
    each .npz (`left_ee_points` / `right_ee_points`, 1000 pts each, world
    frame). We concatenate both grippers (2000 pts); `--n-robot N` optionally
    sub-samples. => folding needs NO URDF / RobotSampler at all.
  - Train/val split is the exact 8-trajectory held-out list from the dataset
    spec; those 8 go into the `test/` shards.
  - Window starts use temporal stride 2, an initial SKIP, and (by default)
    cloth-motion importance sampling.

Per-sample tar layout (one clip = one sample; N_cloth/N_robot vary per garment
unless --n-cloth / --n-robot are set):
    view0_scene_flows.npy       (T, N_cloth, 3)  float32  cloth world positions
    view0_scene_visibility.npy  (T, N_cloth)     bool     all True
    robot_flows.npy             (T, N_robot, 3)  float32  gripper world positions
    joint_positions.npy         (T, 19)          float32  ARX5 joint angles
    right_gripper_open.npy      (T, 1)           float32  right jaw openness
    left_gripper_open.npy       (T, 1)           float32  left jaw openness

Outputs:
    {out}/train/train_NNNNN.tar
    {out}/test/test_NNNNN.tar
    {out}/metadata_rank0.json
    {out}/norm_stats/norm_stats.json

Usage:
    python npz_to_pointworld_wds.py \\
        --root /data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1 \\
        --out  /data/group_data/katefgroup/VLA/3DPCWM/folding_data_pointworld
"""
from __future__ import annotations

import argparse
import io
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import webdataset as wds
from scipy.spatial import cKDTree
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN = "folding"

# PointWorld release horizon: 1 context + 10 prediction = 11 frames.
# clip_len MUST equal RELEASE_CONTEXT_HORIZON + RELEASE_PRED_HORIZON.
CLIP_LEN = 11

CATEGORIES = ["bimanual", "single_arm", "ik_lift_random", "ik_push"]

# The 8 held-out validation trajectories (exact list from the dataset spec).
# Relative to --root.
VAL_RELPATHS = {
    "bimanual/existing_000000.npz",
    "bimanual/existing_000004.npz",
    "single_arm/000000_000000_left.npz",
    "single_arm/000001_000000_right.npz",
    "ik_lift_random/000000.npz",
    "ik_lift_random/000001.npz",
    "ik_push/000000.npz",
    "ik_push/000001.npz",
}

# Per-point dimension of each robot feature, in PointWorld's gather_features
# semantics. gripper_open is 2-D because `folding` is bimanual ([right, left]).
ROBOT_FEATURE_DIMS = {
    'robot_flows': 3,
    'robot_colors': 3,
    'robot_normals': 3,
    'gripper_open': 2,
    'robot_velocity': 3,
    'robot_acceleration': 3,
}
# The standard PointWorld bimanual robot feature list (matches `behavior`).
# IMPORTANT: pass the SAME list to train.py via --robot_features, and keep the
# order identical -- norm_stats is a flat positional vector.
DEFAULT_ROBOT_FEATURES = (
    'robot_flows,robot_colors,robot_normals,'
    'gripper_open,robot_velocity,robot_acceleration'
)
# robot_colors is synthesized by gather_features as a constant magenta for ALL
# domains (data is ignored), so we accumulate that same constant here.
ROBOT_COLOR_MAGENTA = np.array([1.0, 0.0, 1.0], dtype=np.float32)
# scene_feat_dim = 3 (scene_flows) + CLIP_LEN (dist2robot)  -- no scene_normals.


# ─────────────────────────────────────────────────────────────────────────────
# Welford online statistics (parallel batch form) -- copied from the pushing
# converter so the two stat files are produced identically.
# ─────────────────────────────────────────────────────────────────────────────

class WelfordAccumulator:
    """Numerically stable online mean + variance, supports batch updates."""

    def __init__(self, dim: int) -> None:
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = x.reshape(-1, self.mean.shape[0]).astype(np.float64)
        batch_n = len(x)
        if batch_n == 0:
            return
        batch_mean = x.mean(0)
        batch_M2 = ((x - batch_mean) ** 2).sum(0)
        combined_n = self.n + batch_n
        delta = batch_mean - self.mean
        self.mean = (self.n * self.mean + batch_n * batch_mean) / combined_n
        self.M2 += batch_M2 + delta ** 2 * (self.n * batch_n / combined_n)
        self.n = combined_n

    def merge(self, other: "WelfordAccumulator") -> None:
        """Fold another accumulator into this one (parallel Welford)."""
        if other.n == 0:
            return
        if self.n == 0:
            self.n, self.mean, self.M2 = other.n, other.mean.copy(), other.M2.copy()
            return
        combined_n = self.n + other.n
        delta = other.mean - self.mean
        self.mean = (self.n * self.mean + other.n * other.mean) / combined_n
        self.M2 += other.M2 + delta ** 2 * (self.n * other.n / combined_n)
        self.n = combined_n

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.mean.shape, dtype=np.float64)
        return self.M2 / (self.n - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def finite_diff(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference velocity and acceleration for (T, N, 3) trajectories.

    Mirrors `compute_flow_derivatives` in dataset_components/transforms.py
    (forward/backward diff at the endpoints) so robot_features norm stats match
    what the training pipeline will actually compute.
    """
    T = pos.shape[0]
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    if T >= 2:
        vel[0] = pos[1] - pos[0]
        if T >= 3:
            vel[1:-1] = (pos[2:] - pos[:-2]) * 0.5
            vel[-1] = pos[-1] - pos[-2]
    if T >= 3:
        acc[0] = vel[1] - vel[0]
        if T >= 4:
            acc[1:-1] = (vel[2:] - vel[:-2]) * 0.5
            acc[-1] = vel[-1] - vel[-2]
    return vel, acc


def nearest_robot_dist(scene_pts: np.ndarray, robot_pts: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance from each scene point to any robot point.

    scene_pts: (NS, 3), robot_pts: (NR, 3) -> (NS,) float32.
    Uses a KD-tree (same as the training pipeline's compute_robot_distances);
    ~20x faster than the brute-force pairwise distance for ~1k points.
    """
    dists, _ = cKDTree(robot_pts).query(scene_pts)
    return dists.astype(np.float32)


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def robot_feature_blocks(robot: np.ndarray, clip: dict) -> dict:
    """Per-point robot feature blocks, matching gather_features semantics.

    robot: (T, NR, 3) gripper points for this clip.
    Returns {name: (T, NR, d)} for every supported robot feature, so norm_stats
    can be assembled for whatever --robot-features list is requested.
    """
    T, NR, _ = robot.shape
    vel, acc = finite_diff(robot)
    # bimanual gripper_open slot order is [right, left] (see _allocate_bimanual_features)
    grip = np.concatenate([clip["open_right"], clip["open_left"]], axis=-1)  # (T, 2)
    return {
        'robot_flows': robot,
        'robot_colors': np.broadcast_to(ROBOT_COLOR_MAGENTA, (T, NR, 3)),
        'robot_normals': np.zeros((T, NR, 3), dtype=np.float32),
        'gripper_open': np.broadcast_to(grip[:, None, :], (T, NR, 2)),
        'robot_velocity': vel,
        'robot_acceleration': acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shard writer
# ─────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    """Writes individual clips to a rolling series of WebDataset .tar shards.

    `tag` makes shard filenames and sample keys unique per worker so multiple
    processes can write into the same split directory concurrently.
    """

    def __init__(self, out_dir: Path, split: str, samples_per_shard: int,
                 tag: str = "") -> None:
        self._split_dir = out_dir / split
        self._split_dir.mkdir(parents=True, exist_ok=True)
        self._split = split
        self._tag = tag
        self._per_shard = samples_per_shard
        self._shard = 0
        self._in_shard = 0
        self._total = 0
        self._writer: Optional[wds.TarWriter] = None
        self._open_shard()

    def _open_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
        path = str(self._split_dir / f"{self._split}_{self._tag}{self._shard:05d}.tar")
        self._writer = wds.TarWriter(path)

    def next_key(self) -> str:
        # split name must appear in the key/filename (gather_shard_paths checks it)
        return f"{self._split}_{self._tag}{self._total:08d}"

    def write(self, fields: dict) -> None:
        self._writer.write({"__key__": self.next_key(), **fields})
        self._total += 1
        self._in_shard += 1
        if self._in_shard >= self._per_shard:
            self._shard += 1
            self._in_shard = 0
            self._open_shard()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def total_samples(self) -> int:
        return self._total


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory -> clips
# ─────────────────────────────────────────────────────────────────────────────

def _load_traj(path: str):
    """Load the arrays we need from one trajectory .npz."""
    with np.load(path) as data:
        cloth = np.asarray(data["cloth_points"], dtype=np.float32)      # (T, Nc, 3)
        left = np.asarray(data["left_ee_points"], dtype=np.float32)     # (T, 1000, 3)
        right = np.asarray(data["right_ee_points"], dtype=np.float32)   # (T, 1000, 3)
        joint_q = np.asarray(data["joint_q"], dtype=np.float32)         # (T, 19)
        openness = np.asarray(data["openness"], dtype=np.float32)       # (T, 2) [L, R]
    return cloth, left, right, joint_q, openness


def _window_starts_and_weights(cloth_sub: np.ndarray, args):
    """Valid clip start indices (subsampled-frame space) + importance weights.

    cloth_sub: (T_sub, Nc, 3) -- already temporally strided + skip-trimmed.
    Returns (starts, probs) where probs is None when uniform sampling is used.
    """
    T_sub = cloth_sub.shape[0]
    n_starts = T_sub - args.clip_len + 1
    if n_starts <= 0:
        return np.empty(0, dtype=np.int64), None
    starts = np.arange(n_starts, dtype=np.int64)

    if args.uniform:
        return starts, None

    # Per-(subsampled)-frame mean per-particle cloth motion, in millimetres.
    # Use a particle subset for speed; motion statistics are robust to it.
    Nc = cloth_sub.shape[1]
    if Nc > args.motion_score_particles:
        pidx = np.linspace(0, Nc - 1, args.motion_score_particles).astype(np.int64)
        cloth_score = cloth_sub[:, pidx]
    else:
        cloth_score = cloth_sub
    per_frame_mm = np.linalg.norm(np.diff(cloth_score, axis=0), axis=-1).mean(axis=1)
    per_frame_mm = per_frame_mm * 1000.0                       # m -> mm
    # window score = mean per-frame motion over the clip_len-1 transitions
    csum = np.concatenate([[0.0], np.cumsum(per_frame_mm)])
    win = args.clip_len - 1
    scores = (csum[win:] + 0.0 - csum[:-win])[:n_starts] / win
    scores = np.maximum(scores, args.motion_floor_mm)
    weights = scores ** args.alpha
    probs = weights / weights.sum()
    return starts, probs


def _make_clip(cloth_sub, robot_sub, joint_sub, open_sub, start, args, rng):
    """Build one centered clip dict from subsampled trajectory arrays."""
    end = start + args.clip_len
    cloth_w = cloth_sub[start:end]    # (T, Nc, 3)
    robot_w = robot_sub[start:end]    # (T, 2000, 3)
    joint_w = joint_sub[start:end]    # (T, 19)
    open_w = open_sub[start:end]      # (T, 2) -> [left, right]

    # Optionally sub-sample cloth / gripper points to a fixed count. Default
    # (0 = keep all) keeps every garment's native vertex count and all 2000
    # gripper points; PointWorld batches the resulting variable-N samples via
    # collate padding + the scene_exists mask. When a positive count is given,
    # the index set is fixed across the clip so per-point temporal
    # correspondence is preserved.
    if args.n_cloth > 0:
        Nc = cloth_w.shape[1]
        cidx = np.sort(rng.choice(Nc, size=args.n_cloth, replace=Nc < args.n_cloth))
        cloth_w = cloth_w[:, cidx]
    if args.n_robot > 0:
        Nr = robot_w.shape[1]
        ridx = np.sort(rng.choice(Nr, size=args.n_robot, replace=Nr < args.n_robot))
        robot_w = robot_w[:, ridx]

    # Center on the mean of scene+robot at t=0 (mirrors center_shift).
    shift = np.concatenate([cloth_w[0], robot_w[0]], axis=0).mean(0).astype(np.float32)
    cloth_w = (cloth_w - shift).astype(np.float32)
    robot_w = (robot_w - shift).astype(np.float32)

    return {
        "cloth": cloth_w,
        "robot": robot_w,
        "joint": joint_w.astype(np.float32),
        # openness is stored as [left, right] in the .npz
        "open_left": open_w[:, 0:1].astype(np.float32),    # (T, 1)
        "open_right": open_w[:, 1:2].astype(np.float32),   # (T, 1)
    }


def extract_clips(path: str, args, rng, is_val: bool) -> list[dict]:
    """Extract clips from one trajectory .npz."""
    cloth, left, right, joint_q, openness = _load_traj(path)
    T = cloth.shape[0]

    # Temporal stride + initial skip.
    sub = slice(args.skip_frames, None, args.stride)
    cloth_sub = cloth[sub]
    robot_sub = np.concatenate([left[sub], right[sub]], axis=1)  # (T_sub, 2000, 3)
    joint_sub = joint_q[sub]
    open_sub = openness[sub]
    T_sub = cloth_sub.shape[0]
    if T_sub < args.clip_len:
        return []

    starts, probs = _window_starts_and_weights(cloth_sub, args)
    if len(starts) == 0:
        return []

    n_clips = max(1, len(starts) // args.clip_spacing)

    if is_val:
        # Deterministic, evenly spaced starts for the val split.
        chosen = starts[:: max(1, len(starts) // n_clips)][:n_clips]
    elif probs is None:
        chosen = rng.choice(starts, size=n_clips, replace=True)
    else:
        chosen = rng.choice(starts, size=n_clips, replace=True, p=probs)

    return [_make_clip(cloth_sub, robot_sub, joint_sub, open_sub, int(s), args, rng)
            for s in chosen]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def discover_trajectories(root: Path) -> tuple[list[str], list[str]]:
    """Return (train_paths, val_paths) across the 4 active categories."""
    all_files = []
    for cat in CATEGORIES:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"Category dir missing: {cat_dir}")
        all_files.extend(sorted(str(p) for p in cat_dir.glob("*.npz")))

    val, train = [], []
    for f in all_files:
        rel = os.path.relpath(f, root)
        (val if rel in VAL_RELPATHS else train).append(f)

    if len(val) != len(VAL_RELPATHS):
        found = {os.path.relpath(f, root) for f in val}
        raise RuntimeError(
            f"Expected {len(VAL_RELPATHS)} val trajs, found {len(val)}. "
            f"Missing: {sorted(VAL_RELPATHS - found)}"
        )
    print(f"Discovered {len(all_files)} trajectories: "
          f"{len(train)} train, {len(val)} val")
    return train, val


def write_norm_stats(out_path: Path, gt_rel_accums, scene_accum, robot_accum) -> None:
    per_ts = {}
    for t in range(len(gt_rel_accums)):
        per_ts[f"timestep_{t}"] = {
            "mean": gt_rel_accums[t].mean.tolist(),
            "variance": gt_rel_accums[t].variance.tolist(),
        }
    norm_stats = {
        "statistics": {
            DOMAIN: {
                "robot_features": {
                    "mean": robot_accum.mean.tolist(),
                    "variance": robot_accum.variance.tolist(),
                },
                "scene_features": {
                    "mean": scene_accum.mean.tolist(),
                    "variance": scene_accum.variance.tolist(),
                },
            }
        },
        "per_timestep_statistics": {
            DOMAIN: {"gt_scene_flows_relative": per_ts},
        },
    }
    norm_dir = out_path / "norm_stats"
    norm_dir.mkdir(exist_ok=True, parents=True)
    (norm_dir / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2))
    print(f"Norm stats -> {norm_dir / 'norm_stats.json'}")


def _write_clip(clip: dict, writer: ShardWriter) -> None:
    T = clip["cloth"].shape[0]
    writer.write({
        "view0_scene_flows.npy": npy_bytes(clip["cloth"]),
        "view0_scene_visibility.npy": npy_bytes(
            np.ones((T, clip["cloth"].shape[1]), dtype=bool)
        ),
        "robot_flows.npy": npy_bytes(clip["robot"]),
        "joint_positions.npy": npy_bytes(clip["joint"]),
        "right_gripper_open.npy": npy_bytes(clip["open_right"]),
        "left_gripper_open.npy": npy_bytes(clip["open_left"]),
    })


def _accumulate(clip: dict, clip_len: int, robot_features: list,
                gt_rel_accums: list, scene_accum, robot_accum) -> None:
    cloth, robot = clip["cloth"], clip["robot"]
    # 1. gt_scene_flows_relative (per-timestep training target)
    gt_rel = cloth - cloth[0:1]
    for t in range(clip_len):
        gt_rel_accums[t].update(gt_rel[t])
    # 2. scene_features @ t=0: [flows_t0(3), dist2robot(T)]
    dist2robot = np.stack(
        [nearest_robot_dist(cloth[0], robot[t]) for t in range(clip_len)], axis=0,
    )
    scene_accum.update(np.concatenate([cloth[0], dist2robot.T], axis=-1))
    # 3. robot_features -- assembled in the exact order of --robot-features.
    blocks = robot_feature_blocks(robot, clip)
    robot_accum.update(np.concatenate([blocks[n] for n in robot_features], axis=-1))


def _process_chunk(task: tuple) -> dict:
    """Worker entry point: convert one chunk of trajectories to its own shards.

    Each worker writes uniquely-tagged shard files into the shared split dir and
    returns its clip count + Welford accumulators (merged by the parent).
    """
    (args, robot_features, paths, tag, split, is_val,
     scene_feat_dim, robot_feat_dim, seed) = task

    writer = ShardWriter(Path(args.out), split, args.samples_per_shard, tag=tag)
    rng = np.random.default_rng(seed)
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(args.clip_len)]
    scene_accum = WelfordAccumulator(scene_feat_dim)
    robot_accum = WelfordAccumulator(robot_feat_dim)

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
    return {
        "split": split, "count": writer.total_samples, "n_failed": n_failed,
        "gt_rel": gt_rel_accums, "scene": scene_accum, "robot": robot_accum,
    }


def process(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.clip_len != CLIP_LEN:
        print(f"WARNING: --clip-len={args.clip_len} != PointWorld release horizon "
              f"({CLIP_LEN}). Training will fail unless RELEASE_CONTEXT_HORIZON + "
              f"RELEASE_PRED_HORIZON is changed to match.")

    # Parse + validate the robot feature list (must match train.py --robot_features).
    robot_features = [s.strip() for s in args.robot_features.split(',') if s.strip()]
    bad = [f for f in robot_features if f not in ROBOT_FEATURE_DIMS]
    if bad:
        raise ValueError(f"Unknown robot feature(s) {bad}; "
                         f"valid: {sorted(ROBOT_FEATURE_DIMS)}")
    robot_feat_dim = sum(ROBOT_FEATURE_DIMS[f] for f in robot_features)
    scene_feat_dim = 3 + args.clip_len

    train_paths, val_paths = discover_trajectories(root)
    if args.max_trajs > 0:
        train_paths = train_paths[: args.max_trajs]

    n_workers = args.num_workers if args.num_workers > 0 else len(os.sched_getaffinity(0))
    n_workers = max(1, min(n_workers, len(train_paths)))
    # More chunks than workers -> finer progress + better load balance. Round-robin
    # so the long bimanual trajectories (sorted first) spread across all chunks.
    n_chunks = max(1, min(n_workers * 4, len(train_paths)))
    chunks = [train_paths[i::n_chunks] for i in range(n_chunks)]

    print(f"clip_len={args.clip_len}  stride={args.stride}  skip={args.skip_frames}")
    print(f"n_cloth={args.n_cloth or 'all'}  n_robot={args.n_robot or 'all'}  "
          f"importance_sampling={not args.uniform}")
    print(f"robot_features={robot_features}")
    print(f"scene_feat_dim={scene_feat_dim}  robot_feat_dim={robot_feat_dim}")
    print(f"workers={n_workers}  train_chunks={n_chunks}", flush=True)

    tasks = [
        (args, robot_features, ch, f"w{i:03d}_", "train", False,
         scene_feat_dim, robot_feat_dim, args.seed + 1 + i)
        for i, ch in enumerate(chunks)
    ]
    # Validation: a single task (only 8 trajectories) -> the test/ shards.
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

    # ── Merge worker results ───────────────────────────────────────────────
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(args.clip_len)]
    scene_accum = WelfordAccumulator(scene_feat_dim)
    robot_accum = WelfordAccumulator(robot_feat_dim)
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
        print(f"\nWARNING: {n_failed} trajectories could not be read "
              f"(see SKIP lines above).")
    if train_count == 0:
        raise RuntimeError("No training clips were written -- check data access.")

    metadata = {
        "train": {"processed_count": train_count},
        "test": {"processed_count": test_count},
    }
    (out_path / "metadata_rank0.json").write_text(json.dumps(metadata, indent=2))
    write_norm_stats(out_path, gt_rel_accums, scene_accum, robot_accum)

    print(f"\n{'-' * 52}")
    print(f"  Train clips : {train_count}")
    print(f"  Test  clips : {test_count}")
    print(f"  scene_feat_dim: {scene_feat_dim}   robot_feat_dim: {robot_feat_dim}")
    print(f"  Output       : {out_path}")
    print(f"{'-' * 52}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIM1 cloth-folding .npz -> PointWorld WebDataset + norm_stats"
    )
    p.add_argument("--root", required=True,
                   help="sim1_cloth_diverse_v1 dataset root")
    p.add_argument("--out", required=True, help="Output root directory")

    g = p.add_argument_group("clip extraction")
    g.add_argument("--clip-len", type=int, default=CLIP_LEN,
                   help=f"Frames per clip (default {CLIP_LEN} = 1 context + 10 pred)")
    g.add_argument("--stride", type=int, default=2,
                   help="Temporal subsampling of native 60Hz frames (default 2 = 30Hz)")
    g.add_argument("--skip-frames", type=int, default=30,
                   help="Native frames to skip at trajectory start (cloth settling)")
    g.add_argument("--clip-spacing", type=int, default=15,
                   help="Avg subsampled-frame spacing between sampled clip starts; "
                        "controls clips-per-trajectory density (default 15)")

    g = p.add_argument_group("point clouds")
    g.add_argument("--n-cloth", type=int, default=0,
                   help="Cloth vertices kept per clip (0 = keep all of the "
                        "garment's native vertices; default 0)")
    g.add_argument("--n-robot", type=int, default=0,
                   help="Gripper points kept per clip, from the 2000 bimanual "
                        "points (0 = keep all 2000; default 0)")

    g = p.add_argument_group("features")
    g.add_argument("--robot-features", type=str, default=DEFAULT_ROBOT_FEATURES,
                   help="Comma-separated robot feature list to compute norm_stats "
                        "for. MUST match train.py --robot_features exactly (same "
                        f"names, same order). Valid: {sorted(ROBOT_FEATURE_DIMS)}. "
                        f"Default: {DEFAULT_ROBOT_FEATURES}")

    g = p.add_argument_group("importance sampling")
    g.add_argument("--uniform", action="store_true",
                   help="Disable cloth-motion importance sampling (uniform starts)")
    g.add_argument("--alpha", type=float, default=2.0,
                   help="Importance weight exponent: weight = score**alpha")
    g.add_argument("--motion-floor-mm", type=float, default=0.1,
                   help="Floor on per-window cloth motion score, in mm")
    g.add_argument("--motion-score-particles", type=int, default=256,
                   help="Particle subset size used for motion scoring")

    g = p.add_argument_group("output")
    g.add_argument("--samples-per-shard", type=int, default=200)
    g.add_argument("--num-workers", type=int, default=0,
                   help="Parallel trajectory-conversion processes "
                        "(0 = all CPUs allocated to the job)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-trajs", type=int, default=-1,
                   help="Cap on training trajectories (for quick test runs)")
    return p.parse_args()


if __name__ == "__main__":
    process(parse_args())
