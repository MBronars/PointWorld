#!/usr/bin/env python3
"""
h5_to_pointworld_wds.py
Convert PushBoundary HDF5 → PointWorld WebDataset shards + norm_stats.json.

Key changes from the old h5_to_wds.py:
  - One clip = one sample (not batched groups of 4).
  - Scene data  = block particles only  (stored in the tar as view0_scene_*).
  - Robot data  = gripper geometry generated at training time from joint_positions
                  via the URDF + RobotSampler; we store only joint_positions here.
  - No camera images / depth / intrinsics / extrinsics.
  - Normals are int8-quantised (×127) as required by decode_data.
  - Centering mirrors pointworld's center_shift: mean of scene+robot at t=0.
  - norm_stats.json is written alongside the shards so training can start
    immediately without a separate statistics pass.

Per-sample tar layout:
    view0_scene_flows.npy       (T, N_block, 3)  float32  block world positions
    view0_scene_normals.npy     (T, N_block, 3)  int8     quantised (÷127 = unit)
                                                          [only if --include-normals]
    view0_scene_visibility.npy  (T, N_block)      bool
    joint_positions.npy         (T, 2)            float32  [x_joint, y_joint]
    right_gripper_pose.npy      (T, 7)            float32  [x,y,z, qx,qy,qz,qw]
    right_gripper_open.npy      (T, 1)            float32  zeros (no gripper)

Outputs:
    {out}/train/train_NNNNN.tar
    {out}/test/test_NNNNN.tar
    {out}/metadata_rank0.json
    {out}/norm_stats/norm_stats.json

Usage:
    python h5_to_pointworld_wds.py \\
        --h5  demos/PushBoundary/all_demos.h5 \\
        --out data/push_pointworld \\
        --shape cube \\
        --clip-len 11 --stride 4 \\
        --n-block 128 --n-hand 128
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import webdataset as wds
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from vis_push_pcd_floating import (          # noqa: E402
    sample_templates,
    apply_pose,
    ACTOR_POS,
    ACTOR_QUAT,
    FLOAT_QPOS,
    FLOAT_ROOT_POS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN = "pushing"
QUANTIZED_NORMALS_SCALE = 127.0
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Feature dimensions — must match args.robot_features / args.scene_features in
# your pointworld training config:
#   robot_features: [robot_flows(3), robot_normals(3), robot_velocity(3), robot_acceleration(3)]
#   scene_features: [scene_flows(3), scene_normals(3), dist2robot(T)]
ROBOT_FEAT_DIM = 12          # 3+3+3+3
# scene_feat_dim = 6 + clip_len  (computed at runtime)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cube_normals_local(pts: np.ndarray) -> np.ndarray:
    """
    Analytical face normals for points sampled on a cube surface.
    Each point belongs to the face whose axis has the largest absolute value.
    pts: (N, 3) in local object frame.  Returns (N, 3) float32.
    """
    abs_pts = np.abs(pts)
    face_axis = np.argmax(abs_pts, axis=1)          # 0=X, 1=Y, 2=Z
    normals = np.zeros_like(pts, dtype=np.float32)
    rows = np.arange(len(pts))
    normals[rows, face_axis] = np.sign(pts[rows, face_axis]).astype(np.float32)
    # Fix any zero-sign points (exactly on an edge)
    zero_mask = normals[rows, face_axis] == 0.0
    normals[zero_mask, face_axis[zero_mask]] = 1.0
    return normals


def cylinder_normals_local(pts: np.ndarray, half_len: float = 0.06) -> np.ndarray:
    """
    Analytical normals for an upright cylinder (axis = Z).
    Cylinder parameters must match the floating_gripper URDF
    (radius 0.01 m, half-length 0.06 m).
    pts: (N, 3) local.  Returns (N, 3) float32.
    """
    normals = np.zeros(pts.shape, dtype=np.float32)
    z = pts[:, 2]
    is_top  = z >=  half_len * 0.9
    is_bot  = z <= -half_len * 0.9
    is_side = ~(is_top | is_bot)

    normals[is_top]  = [0.0, 0.0,  1.0]
    normals[is_bot]  = [0.0, 0.0, -1.0]

    xy = pts[is_side, :2]
    r  = np.linalg.norm(xy, axis=1, keepdims=True).clip(1e-8)
    normals[is_side, :2] = xy / r
    return normals


def pca_normals(pts: np.ndarray, k: int = 10) -> np.ndarray:
    """
    PCA-based outward normal estimation (generic; used for T-shape blocks).
    Complexity O(N·k·3²) — acceptable for N ≤ 256.
    pts: (N, 3) local.  Returns (N, 3) float32.
    """
    N = len(pts)
    k = min(k, N - 1)

    # k-nearest neighbours via pairwise squared distances
    sq_dist = ((pts[:, None] - pts[None]) ** 2).sum(-1)          # (N, N)
    knn_idx = np.argpartition(sq_dist, k + 1, axis=1)[:, :k + 1] # (N, k+1)

    centroid = pts.mean(0)
    normals  = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        nbrs = pts[knn_idx[i]]
        c    = nbrs.mean(0)
        cov  = (nbrs - c).T @ (nbrs - c) / len(nbrs)
        _, vecs = np.linalg.eigh(cov)           # ascending eigenvalue order
        n = vecs[:, 0].astype(np.float32)       # smallest eigenvector = normal
        if np.dot(n, pts[i] - centroid) < 0:
            n = -n
        normals[i] = n
    return normals


def block_normals_local(pts: np.ndarray, shape: str) -> np.ndarray:
    """Dispatch to analytical (cube) or PCA (T-shape) normal estimation."""
    if shape == "cube":
        return cube_normals_local(pts)
    return pca_normals(pts)                    # T-shape: use PCA


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q: (4,) wxyz → (3, 3) rotation matrix, float32."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def rotate_normals_world(
    local_normals: np.ndarray,   # (N, 3)
    quats_wxyz:    np.ndarray,   # (T, 4)
) -> np.ndarray:                 # (T, N, 3)
    """Apply per-timestep rotation to a fixed set of local normals."""
    T, N = len(quats_wxyz), len(local_normals)
    out  = np.empty((T, N, 3), dtype=np.float32)
    for t in range(T):
        R = quat_wxyz_to_rotmat(quats_wxyz[t])
        out[t] = (R @ local_normals.T).T
    return out


def finite_diff(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Central-difference velocity and acceleration for (T, N, 3) trajectories.
    Forward / backward differences are used at the endpoints.
    """
    T = pos.shape[0]
    vel = np.empty_like(pos)
    if T >= 2:
        vel[0]    = pos[1]  - pos[0]
        vel[1:-1] = (pos[2:] - pos[:-2]) * 0.5
        vel[-1]   = pos[-1] - pos[-2]
    else:
        vel[:] = 0.0

    acc = np.empty_like(vel)
    if T >= 2:
        acc[0]    = vel[1]  - vel[0]
        acc[1:-1] = (vel[2:] - vel[:-2]) * 0.5
        acc[-1]   = vel[-1] - vel[-2]
    else:
        acc[:] = 0.0
    return vel, acc


def nearest_robot_dist(scene_pts: np.ndarray, robot_pts: np.ndarray) -> np.ndarray:
    """
    Brute-force nearest-neighbour distance from each scene point to any robot point.
    scene_pts: (NS, 3),  robot_pts: (NR, 3) → (NS,) float32.
    For NR ≤ 256 this is faster than building a KD-tree per call.
    """
    diff = scene_pts[:, None, :] - robot_pts[None, :, :]  # (NS, NR, 3)
    return np.linalg.norm(diff, axis=-1).min(axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def npy_bytes(arr: np.ndarray) -> bytes:
    """Serialise a numpy array to raw .npy format bytes."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def quantize_normals(normals: np.ndarray) -> np.ndarray:
    """(..., 3) float32 unit normals → (..., 3) int8  (÷127 recovers float)."""
    q = np.clip(np.round(normals * QUANTIZED_NORMALS_SCALE), -127, 127)
    return q.astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Welford online statistics (parallel batch form)
# ─────────────────────────────────────────────────────────────────────────────

class WelfordAccumulator:
    """Numerically stable online mean + variance, supports batch updates."""

    def __init__(self, dim: int) -> None:
        self.n    = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2   = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """x: (..., dim) — all leading dims are independent observations."""
        x = x.reshape(-1, self.mean.shape[0]).astype(np.float64)
        batch_n    = len(x)
        if batch_n == 0:
            return
        batch_mean = x.mean(0)
        batch_M2   = ((x - batch_mean) ** 2).sum(0)
        combined_n = self.n + batch_n
        delta      = batch_mean - self.mean
        self.mean  = (self.n * self.mean + batch_n * batch_mean) / combined_n
        self.M2   += batch_M2 + delta ** 2 * (self.n * batch_n / combined_n)
        self.n     = combined_n

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.mean.shape, dtype=np.float64)
        return self.M2 / (self.n - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Shard writer
# ─────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    """Writes individual clips to a rolling series of WebDataset .tar shards."""

    def __init__(self, out_dir: Path, split: str, samples_per_shard: int) -> None:
        self._split_dir = out_dir / split
        self._split_dir.mkdir(parents=True, exist_ok=True)
        self._split       = split
        self._per_shard   = samples_per_shard
        self._shard       = 0
        self._in_shard    = 0
        self._total       = 0
        self._writer: Optional[wds.TarWriter] = None
        self._open_shard()

    def _open_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
        path = str(self._split_dir / f"{self._split}_{self._shard:05d}.tar")
        self._writer = wds.TarWriter(path)

    def write(self, key: str, fields: dict) -> None:
        self._writer.write({"__key__": key, **fields})
        self._total    += 1
        self._in_shard += 1
        if self._in_shard >= self._per_shard:
            self._shard    += 1
            self._in_shard  = 0
            self._open_shard()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def total_samples(self) -> int:
        return self._total


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def extract_clips_from_traj(
    block_states : np.ndarray,
    art_states   : np.ndarray,
    args         : argparse.Namespace,
    master_rng   : np.random.Generator,
    use_T        : bool,
) -> list[dict]:
    """Extract all sliding-window clips from one trajectory's raw states."""
    clip_len = args.clip_len
    T        = clip_len
    n_block  = args.n_block
    n_hand   = args.n_hand

    # Build TCP poses from articulation state
    root_pos        = art_states[:, FLOAT_ROOT_POS].copy()
    joint_xy        = art_states[:, FLOAT_QPOS]
    root_pos[:, 0] += joint_xy[:, 0]
    root_pos[:, 1] += joint_xy[:, 1]
    tcp_quat        = np.tile(IDENTITY_QUAT_WXYZ, (len(art_states), 1))
    tcp_poses       = np.concatenate([root_pos, tcp_quat], axis=1)

    # Temporal subsampling
    sub       = slice(None, None, args.subsample)
    block_sub = block_states[sub]
    tcp_sub   = tcp_poses[sub]
    art_sub   = art_states[sub]
    Tsub      = block_sub.shape[0]
    if Tsub < clip_len:
        return []

    traj_rng = np.random.default_rng(master_rng.integers(2**31))
    phase    = int(traj_rng.integers(0, max(args.stride, 1)))
    starts   = list(range(phase, Tsub - clip_len + 1, args.stride))
    if not starts:
        starts = [0]

    clips: list[dict] = []
    for s in starts:
        clip_rng  = np.random.default_rng(master_rng.integers(2**31))
        templates = sample_templates(n_block, n_hand, clip_rng)
        block_tpl = templates["T"] if use_T else templates["cube"]
        hand_tpl  = templates["hand"]

        blk_norm_loc  = block_normals_local(block_tpl, args.shape)
        hand_norm_loc = cylinder_normals_local(hand_tpl)

        end      = s + clip_len
        bs_clip  = block_sub[s:end]
        tcp_clip = tcp_sub[s:end]
        art_clip = art_sub[s:end]

        block_world      = np.empty((T, n_block, 3), dtype=np.float32)
        hand_world       = np.empty((T, n_hand,  3), dtype=np.float32)
        block_quats_wxyz = np.empty((T, 4),          dtype=np.float32)

        for t in range(T):
            bs  = bs_clip[t]
            tcp = tcp_clip[t]
            block_world[t]      = apply_pose(block_tpl, bs[ACTOR_POS], bs[ACTOR_QUAT])
            hand_world[t]       = apply_pose(hand_tpl,  tcp[:3],       tcp[3:7])
            block_quats_wxyz[t] = bs[ACTOR_QUAT]

        shift = np.concatenate(
            [block_world[0], hand_world[0]], axis=0
        ).mean(0).astype(np.float32)
        block_world -= shift
        hand_world  -= shift

        tcp_centered        = tcp_clip.copy()
        tcp_centered[:, :3] -= shift

        block_normals_world = rotate_normals_world(blk_norm_loc,  block_quats_wxyz)
        hand_normals_world  = rotate_normals_world(hand_norm_loc, tcp_clip[:, 3:7])

        joint_positions    = art_clip[:, FLOAT_QPOS].astype(np.float32)
        right_gripper_pose = tcp_centered.astype(np.float32)
        right_gripper_open = np.zeros((T, 1), dtype=np.float32)

        clips.append({
            "block_world":   block_world,
            "block_normals": block_normals_world,
            "hand_world":    hand_world,
            "hand_normals":  hand_normals_world,
            "joint_pos":     joint_positions,
            "grip_pose":     right_gripper_pose,
            "grip_open":     right_gripper_open,
        })

    return clips

def process(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    master_rng = np.random.default_rng(args.seed)

    use_T    = (args.shape == "T")
    clip_len = args.clip_len
    T        = clip_len
    n_block  = args.n_block
    n_hand   = args.n_hand
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    scene_feat_dim = (6 if args.include_normals else 3) + T
    print(f"clip_len={T}  stride={args.stride}  subsample={args.subsample}")
    print(f"n_block={n_block}  n_hand={n_hand}  shape={args.shape}")
    print(f"min_len={args.min_len}  z_eps={args.z_eps}  rot_eps={args.rot_eps}")
    print(f"include_normals={args.include_normals}")
    print(f"scene_feat_dim={scene_feat_dim}  robot_feat_dim={ROBOT_FEAT_DIM}")

    # ── Welford accumulators (train split only) ────────────────────────────
    gt_rel_accums    = [WelfordAccumulator(3) for _ in range(T)]
    scene_feat_accum = WelfordAccumulator(scene_feat_dim)
    robot_feat_accum = WelfordAccumulator(ROBOT_FEAT_DIM)

    # ── Pass 1: scan + filter trajectories (metadata only) ─────────────────
    valid_trajs: list[str] = []
    skip_reasons: dict[str, int] = {}

    with h5py.File(args.h5, "r") as f:
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        print(f"\nFound {len(traj_keys)} trajectories in {args.h5}")

        sample_traj = f[traj_keys[0]]
        art_keys    = list(sample_traj["env_states/articulations"].keys())
        actor_keys  = list(sample_traj["env_states/actors"].keys())

        _robot_cands = ["floating_gripper", "robot", "panda_stick", "gripper"]
        art_key = next(
            (k for c in _robot_cands for k in art_keys if c in k.lower()),
            art_keys[0],
        )
        _block_cands = ["push_block", "block", "cube"]
        block_actor_key = next(
            (k for c in _block_cands for k in actor_keys if c in k.lower()),
            actor_keys[0],
        )
        print(f"Articulation key : '{art_key}'")
        print(f"Actor key        : '{block_actor_key}'\n")

        for traj_key in tqdm(traj_keys, desc="Pass 1: filtering"):
            try:
                block_ds = f[traj_key][f"env_states/actors/{block_actor_key}"]
                art_ds   = f[traj_key][f"env_states/articulations/{art_key}"]
            except KeyError as exc:
                skip_reasons["missing_key"] = skip_reasons.get("missing_key", 0) + 1
                continue

            # Load only the columns we need for filtering — don't pull full state yet.
            n_raw = block_ds.shape[0]
            if n_raw - 2 < max(args.min_len, clip_len):
                skip_reasons["too_short"] = skip_reasons.get("too_short", 0) + 1
                continue

            block_z    = block_ds[2:, ACTOR_POS][:, 2]         # (T,)
            tcp_z      = art_ds[2:,   FLOAT_ROOT_POS][:, 2]    # (T,)
            block_quat = block_ds[2:, ACTOR_QUAT]              # (T, 4) wxyz

            if float(np.ptp(block_z)) > args.z_eps or float(np.ptp(tcp_z)) > args.z_eps:
                skip_reasons["z_movement"] = skip_reasons.get("z_movement", 0) + 1
                continue

            # qx (col 1) and qy (col 2) nonzero → block tilts around a
            # horizontal axis → corner lifts off the table.
            if float(np.abs(block_quat[:, 1:3]).max()) > args.rot_eps:
                skip_reasons["block_tilt"] = skip_reasons.get("block_tilt", 0) + 1
                continue

            valid_trajs.append(traj_key)

    print(f"\nValid trajectories: {len(valid_trajs)} / {len(traj_keys)}")
    if skip_reasons:
        print("Skip reasons:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"  {reason}: {count}")
    if not valid_trajs:
        raise RuntimeError("No valid trajectories — check filter thresholds.")

    # ── Train / test split at trajectory level ─────────────────────────────
    rng_split = np.random.default_rng(args.seed)
    perm      = rng_split.permutation(len(valid_trajs)).tolist()
    shuffled  = [valid_trajs[i] for i in perm]
    n_test    = max(1, int(len(shuffled) * args.test_frac))
    test_trajs  = shuffled[:n_test]
    train_trajs = shuffled[n_test:]
    print(f"Train trajs: {len(train_trajs)}   Test trajs: {len(test_trajs)}")

    # ── Pass 2: extract clips and stream to tar ────────────────────────────
    train_writer = ShardWriter(out_path, "train", args.samples_per_shard)
    test_writer  = ShardWriter(out_path, "test",  args.samples_per_shard)

    def write_clip(clip: dict, writer: ShardWriter) -> None:
        block_world   = clip["block_world"]
        hand_world    = clip["hand_world"]
        joint_pos     = clip["joint_pos"]
        grip_pose     = clip["grip_pose"]
        grip_open     = clip["grip_open"]
        visibility    = np.ones((T, n_block), dtype=bool)
        key = f"{writer._split}_{writer.total_samples:08d}"
        fields = {
            "view0_scene_flows.npy":      npy_bytes(block_world),
            "view0_scene_visibility.npy": npy_bytes(visibility),
            "joint_positions.npy":        npy_bytes(joint_pos),
            "right_gripper_pose.npy":     npy_bytes(grip_pose),
            "right_gripper_open.npy":     npy_bytes(grip_open),
        }
        if args.include_normals:
            fields["view0_scene_normals.npy"] = npy_bytes(
                quantize_normals(clip["block_normals"])
            )
        writer.write(key, fields)

    def accumulate_stats(clip: dict) -> None:
        block_world   = clip["block_world"]
        hand_world    = clip["hand_world"]
        hand_normals  = clip["hand_normals"]

        # 1. gt_scene_flows_relative (training target)
        gt_rel = block_world - block_world[0:1]
        for t in range(T):
            gt_rel_accums[t].update(gt_rel[t])

        # 2. scene_features at t=0: [flows_t0(3), (normals_t0(3) if included), dist2robot(T)]
        scene_pts_t0 = block_world[0]
        dist2robot = np.empty((T, n_block), dtype=np.float32)
        for t in range(T):
            dist2robot[t] = nearest_robot_dist(scene_pts_t0, hand_world[t])
        scene_feat_parts = [scene_pts_t0]
        if args.include_normals:
            scene_feat_parts.append(clip["block_normals"][0])
        scene_feat_parts.append(dist2robot.T)
        scene_feats = np.concatenate(scene_feat_parts, axis=-1)
        scene_feat_accum.update(scene_feats)

        # 3. robot_features: [flows(3), normals(3), vel(3), acc(3)]
        robot_vel, robot_acc = finite_diff(hand_world)
        robot_feats = np.concatenate(
            [hand_world, hand_normals, robot_vel, robot_acc], axis=-1,
        )
        robot_feat_accum.update(robot_feats)

    with h5py.File(args.h5, "r") as f:
        for traj_key in tqdm(train_trajs, desc="Pass 2: writing train"):
            block_states = f[traj_key][f"env_states/actors/{block_actor_key}"][:][2:]
            art_states   = f[traj_key][f"env_states/articulations/{art_key}"][:][2:]
            for clip in extract_clips_from_traj(
                block_states, art_states, args, master_rng, use_T,
            ):
                write_clip(clip, train_writer)
                accumulate_stats(clip)

        for traj_key in tqdm(test_trajs, desc="Pass 2: writing test"):
            block_states = f[traj_key][f"env_states/actors/{block_actor_key}"][:][2:]
            art_states   = f[traj_key][f"env_states/articulations/{art_key}"][:][2:]
            for clip in extract_clips_from_traj(
                block_states, art_states, args, master_rng, use_T,
            ):
                write_clip(clip, test_writer)

    train_writer.close()
    test_writer.close()

    # ── metadata_rank0.json ────────────────────────────────────────────────
    metadata = {
        "train": {"processed_count": train_writer.total_samples},
        "test":  {"processed_count": test_writer.total_samples},
    }
    meta_path = out_path / "metadata_rank0.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata  → {meta_path}")

    # ── norm_stats/norm_stats.json ─────────────────────────────────────────
    per_ts: dict = {}
    for t in range(T):
        per_ts[f"timestep_{t}"] = {
            "mean":     gt_rel_accums[t].mean.tolist(),
            "variance": gt_rel_accums[t].variance.tolist(),
        }

    norm_stats = {
        "statistics": {
            DOMAIN: {
                "robot_features": {
                    "mean":     robot_feat_accum.mean.tolist(),
                    "variance": robot_feat_accum.variance.tolist(),
                },
                "scene_features": {
                    "mean":     scene_feat_accum.mean.tolist(),
                    "variance": scene_feat_accum.variance.tolist(),
                },
            }
        },
        "per_timestep_statistics": {
            DOMAIN: {"gt_scene_flows_relative": per_ts},
        },
    }

    norm_dir = out_path / "norm_stats"
    norm_dir.mkdir(exist_ok=True)
    norm_path = norm_dir / "norm_stats.json"
    norm_path.write_text(json.dumps(norm_stats, indent=2))
    print(f"Norm stats → {norm_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Train samples : {train_writer.total_samples}")
    print(f"  Test  samples : {test_writer.total_samples}")
    print(f"  Shards (train): {len(list((out_path/'train').glob('*.tar')))}")
    print(f"  Shards (test) : {len(list((out_path/'test').glob('*.tar')))}")
    print(f"  scene_feat_dim: {scene_feat_dim}  (set args.scene_features accordingly)")
    print(f"  robot_feat_dim: {ROBOT_FEAT_DIM}  (set args.robot_features accordingly)")
    print(f"{'─'*50}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PushBoundary HDF5 → PointWorld WebDataset + norm_stats"
    )
    p.add_argument("--h5",    required=True,  help="Input HDF5 file")
    p.add_argument("--out",   required=True,  help="Output root directory")
    p.add_argument("--shape", default="cube", choices=["cube", "T"],
                   help="Block geometry type (default: cube)")

    g = p.add_argument_group("clip extraction")
    g.add_argument("--clip-len",   type=int, default=11,
                   help="Frames per clip T (default: 11 = 1 context + 10 pred)")
    g.add_argument("--stride",     type=int, default=4,
                   help="Sliding-window stride over each trajectory (default: 4)")
    g.add_argument("--subsample",  type=int, default=1,
                   help="Temporal subsampling of raw trajectory before clipping (default: 1)")

    g = p.add_argument_group("point cloud")
    g.add_argument("--n-block", type=int, default=128,
                   help="Block surface points per clip (default: 128)")
    g.add_argument("--n-hand",  type=int, default=128,
                   help="Gripper surface points per clip (default: 128)")

    g = p.add_argument_group("output")
    g.add_argument("--test-frac",         type=float, default=0.1,
                   help="Fraction of clips held out for test (default: 0.1)")
    g.add_argument("--samples-per-shard", type=int,   default=200,
                   help="Clips per .tar shard (default: 200)")
    g.add_argument("--seed",              type=int,   default=0)
    g.add_argument("--include-normals",   action="store_true",
                   help="Emit view0_scene_normals.npy and include normals in "
                        "scene_features norm stats (default: off).")
    
    g = p.add_argument_group("filtering")
    g.add_argument("--min-len", type=int, default=50,
                   help="Minimum trajectory length after dropping lag frame. "
                        "Shorter demos are discarded as likely errors (default: 50).")
    g.add_argument("--z-eps", type=float, default=0.001,
                   help="Max allowed z-range (metres) for block and gripper over "
                        "the trajectory; pushing should be planar (default: 0.001).")
    g.add_argument("--rot-eps", type=float, default=0.001,
                   help="Max allowed |qx| or |qy| of block quaternion; larger "
                        "means the block is tilting off the table (default: 0.001).")

    return p.parse_args()


if __name__ == "__main__":
    process(parse_args())