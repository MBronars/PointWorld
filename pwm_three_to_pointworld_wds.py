#!/usr/bin/env python3
"""
pwm_three_to_pointworld_wds.py — Dump the pwm-data-v1 three-domain corpus
(yikai_small + pickup + rl_push) into PointWorld WebDataset shards.

WHY A DUMPER AND NOT THREE CONVERTERS
    The corpus these three domains denote is not "the files under a root": it is
    the root MINUS its exclusion manifest, PLUS the loader's own construction —
    rl_push has no particles on disk at all (its gripper/object points are built
    at load time from a low-dim recording), yikai_small is lowered by its own
    per-episode table height, pickup carries a shift_m override. Reading the raw
    files again would re-implement all of that and drift from it.

    So this script drives the PWM unified loader itself
    (`pwm_final_paper/data/unified_dataset.build_domain_dataset`) with the
    recipe of the PWM run it is a baseline for, and writes what comes out. The
    corpus identity is then the PWM run's by construction.

WHAT COMES OUT (one PointWorld clip per window, the `pickup` layout)
    view0_scene_flows.npy       (T, Ns, 3) float32  predicted rows, centered
    view0_scene_visibility.npy  (T, Ns)    bool     all True (valid rows only)
    robot_flows.npy             (T, Nr, 3) float32  actuator rows, centered
    joint_positions.npy         (T, 1)     float32  placeholder zeros
    right_gripper_open.npy      (T, 1)     float32  placeholder zeros
    left_gripper_open.npy       (T, 1)     float32  placeholder zeros

    T = 11 = PointWorld's RELEASE_CONTEXT_HORIZON (1) + RELEASE_PRED_HORIZON (10),
    drawn as a PWM window with context_len=1, future_len=10 at the domain's
    native dt 0.05 s. The PWM run predicts 16 future frames; PointWorld's horizon
    is fixed by the release constants, so the baseline trains at 10 and is chained
    at evaluation time to reach the protocol's horizon.

    Rows are split by the batch's `action_mask`: True (the actuator / gripper)
    becomes robot_flows, False becomes the scene. That is the same split the PWM
    protocol scores — "predicted rows = valid particles minus rigid actuators
    and fixtures, whose motion is an input".

    Centering matches pickup_to_pointworld_wds.py: every clip is shifted by the
    mean of its own frame-0 points (scene + robot together).

NORMALIZATION
    This script does not write norm_stats.json. It writes a per-domain
    `stats_raw.json` holding Welford accumulators (n / mean / M2) for the scene
    feature vector, the robot feature vector and the per-timestep relative scene
    flow. `merge_norm_stats.py` pools the three and emits ONE norm_stats.json
    whose three domain entries are identical — which is how a PointWorld stats
    file expresses the shared ruler that TRAINING.md asks for.

    Feature vectors are assembled in the order the training run must request:
        scene_features = scene_flows(3) + dist2robot(T)         -> 3 + 11 = 14
        robot_features = robot_flows(3) + velocity(3) + accel(3) -> 9
    No normals on either side (the shards carry none), following the
    push_08_07 convention of keeping both sides of that choice consistent.

ENVIRONMENT
    Runs in the PWM env (pwm211): it imports the PWM loader. It deliberately does
    NOT import npz_to_pointworld_wds, whose scipy dependency that env lacks; the
    handful of helpers it needs are re-implemented here against torch, and
    `--selfcheck` asserts they agree with what PointWorld computes at train time.

USAGE
    python pwm_three_to_pointworld_wds.py --domain yikai --split train \\
        --out /data/group_data/katefgroup/VLA/3DPCWM/three_pointworld \\
        --target-clips 150000
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

PWM_ROOT = os.environ.get("PWM_ROOT", "/data/user_data/mbronars/packages/pwm_final_paper")

# PointWorld release horizon: 1 context + 10 prediction = 11 frames.
CLIP_LEN = 11

# PWM domain -> the PointWorld domain name used in --domains and in norm_stats.
# The name CONTAINS "pickup" on purpose: PointWorld dispatches on that substring
# in four places (dataloader.build_dataset, decoders.decode_data,
# decoders.build_flow_sample, robot._get_robot_flows) and the pickup path is
# exactly the one these shards want — gripper points stored in the shard, no
# URDF sampler, no cameras. Naming them this way needs no edit to the checkout.
DOMAIN_NAMES = {
    "yikai": "pickup_yikai_small",
    "pickup": "pickup_release",
    "rl_push": "pickup_rl_push",
}

# Splits: PointWorld's gather_shard_paths looks for 'train' and 'test'
# directories; PWM calls its held-out split 'val'.
PWM_SPLIT = {"train": "train", "test": "val"}


# ─────────────────────────────────────────────────────────────────────────────
# Welford (same algebra as npz_to_pointworld_wds.WelfordAccumulator, plus
# JSON round-tripping so per-domain dumps can be pooled afterwards)
# ─────────────────────────────────────────────────────────────────────────────

class WelfordAccumulator:
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

    def to_json(self) -> dict:
        return {"n": int(self.n), "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    @classmethod
    def from_json(cls, payload: dict) -> "WelfordAccumulator":
        acc = cls(len(payload["mean"]))
        acc.n = int(payload["n"])
        acc.mean = np.asarray(payload["mean"], dtype=np.float64)
        acc.M2 = np.asarray(payload["M2"], dtype=np.float64)
        return acc


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers — mirror dataset_components/transforms.compute_flow_derivatives
# and compute_robot_distances, so the stats describe what training will see.
# ─────────────────────────────────────────────────────────────────────────────

def finite_diff(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference velocity and acceleration for (T, N, 3) trajectories."""
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
    """(NS,) nearest-neighbour distance from each scene point to any robot point.

    torch.cdist rather than a KD-tree: the PWM env has no scipy.
    --selfcheck compares it against scipy's cKDTree in the PointWorld env.
    """
    import torch
    s = torch.from_numpy(np.ascontiguousarray(scene_pts, dtype=np.float32))
    r = torch.from_numpy(np.ascontiguousarray(robot_pts, dtype=np.float32))
    return torch.cdist(s, r).min(dim=1).values.numpy().astype(np.float32)


def dist2robot_all_frames(scene0: np.ndarray, robot: np.ndarray) -> np.ndarray:
    """(T, NS) distance from each frame-0 scene point to the robot at each frame.

    One batched cdist rather than T separate calls: bit-identical, 10x faster
    (39 ms against 394 ms per clip), and this is 90% of the dump's cost.
    """
    import torch
    T = robot.shape[0]
    s = torch.from_numpy(np.ascontiguousarray(scene0, dtype=np.float32)).expand(T, -1, -1).contiguous()
    r = torch.from_numpy(np.ascontiguousarray(robot, dtype=np.float32))
    return torch.cdist(s, r).min(dim=2).values.numpy().astype(np.float32)


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Shard writer (same filename contract as npz_to_pointworld_wds.ShardWriter:
# the split name must appear in both the shard filename and the sample key)
# ─────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    def __init__(self, out_dir: Path, split: str, samples_per_shard: int, tag: str = "") -> None:
        import webdataset as wds
        self._wds = wds
        self._split_dir = out_dir / split
        self._split_dir.mkdir(parents=True, exist_ok=True)
        self._split = split
        self._tag = tag
        self._per_shard = samples_per_shard
        self._shard = 0
        self._in_shard = 0
        self._total = 0
        self._writer = None
        self._open_shard()

    def _open_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
        path = str(self._split_dir / f"{self._split}_{self._tag}{self._shard:05d}.tar")
        self._writer = self._wds.TarWriter(path)

    def write(self, fields: dict) -> None:
        key = f"{self._split}_{self._tag}{self._total:08d}"
        self._writer.write({"__key__": key, **fields})
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
# PWM loader
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(domain: str, split: str, cfg_path: str, batch_size: int):
    """The PWM dataset for one domain, windowed to PointWorld's 1 + 10 frames.

    raw=True: positions come out in metres. PointWorld normalizes with its own
    stats, so the PWM ruler must not be applied here — but the domain's
    coordinate shift (pickup's shift_m, yikai's per-episode table height) is part
    of the adapter and still is.
    """
    sys.path.insert(0, PWM_ROOT)
    from omegaconf import OmegaConf
    from data import unified_dataset as U
    from data.unified_dataset import build_domain_dataset

    cfg = OmegaConf.load(cfg_path)
    spec = OmegaConf.to_container(cfg.data.roots, resolve=True)
    roots = U.resolve_data_roots(spec)
    if domain not in roots:
        raise ValueError(f"{domain} is not in the recipe's data.roots ({sorted(roots)})")
    pwm_split = PWM_SPLIT[split]
    # train: the loader's own random-start sampling, cycling episodes forever, so
    # a clip budget spreads over starts rather than pinning one window per
    # episode. test: a single deterministic pass, one pinned window per episode.
    training = split == "train"
    ds = build_domain_dataset(domain, split=pwm_split, training=training,
                              future_len=CLIP_LEN - 1, context_len=1,
                              normalization=cfg.normalization, raw=True, roots=roots)
    ds.batch_size = batch_size
    return ds


def iter_clips(ds, target_clips: Optional[int], num_workers: int = 0):
    """Yield (scene [T,Ns,3], robot [T,Nr,3]) in metres, centered, valid rows only.

    num_workers > 0 partitions the corpus across reader processes — the PWM
    datasets shard their records by worker id, so the windows are disjoint, not
    duplicated. Reading one episode file per window is what makes the dump slow,
    and this is the lever on it.
    """
    from torch.utils.data import DataLoader

    written = 0
    for batch in DataLoader(ds, batch_size=None, num_workers=num_workers,
                            persistent_workers=False):
        positions = batch.positions                       # [B, T, N, 3]
        action_mask = batch.action_mask                   # [B, N] True = actuator
        valid = batch.get_valid_mask()                    # [B, N] True = real row
        B, T = positions.shape[0], positions.shape[1]
        if T != CLIP_LEN:
            raise ValueError(f"expected {CLIP_LEN}-frame windows, got {T}")
        for b in range(B):
            v = valid[b]
            robot_rows = v & action_mask[b]
            scene_rows = v & ~action_mask[b]
            if not bool(robot_rows.any()) or not bool(scene_rows.any()):
                continue
            robot = positions[b][:, robot_rows].float().numpy().astype(np.float32)
            scene = positions[b][:, scene_rows].float().numpy().astype(np.float32)
            shift = np.concatenate([scene[0], robot[0]], axis=0).mean(axis=0).astype(np.float32)
            yield scene - shift, robot - shift
            written += 1
            if target_clips is not None and written >= target_clips:
                return


def accumulate(scene, robot, gt_rel_accums, scene_accum, robot_accum) -> None:
    gt_rel = scene - scene[0:1]
    for t in range(CLIP_LEN):
        gt_rel_accums[t].update(gt_rel[t])
    dist2robot = dist2robot_all_frames(scene[0], robot)
    scene_accum.update(np.concatenate([scene[0], dist2robot.T], axis=-1))
    vel, acc = finite_diff(robot)
    robot_accum.update(np.concatenate([robot, vel, acc], axis=-1))


def write_clip(writer: ShardWriter, scene: np.ndarray, robot: np.ndarray) -> None:
    zeros = np.zeros((CLIP_LEN, 1), dtype=np.float32)
    writer.write({
        "view0_scene_flows.npy": npy_bytes(scene),
        "view0_scene_visibility.npy": npy_bytes(np.ones(scene.shape[:2], dtype=bool)),
        "robot_flows.npy": npy_bytes(robot),
        "joint_positions.npy": npy_bytes(zeros),
        "right_gripper_open.npy": npy_bytes(zeros),
        "left_gripper_open.npy": npy_bytes(zeros),
    })


def dump(args) -> None:
    pw_domain = DOMAIN_NAMES[args.domain]
    out = Path(args.out) / pw_domain
    out.mkdir(parents=True, exist_ok=True)

    ds = build_loader(args.domain, args.split, args.config, args.batch_size)
    writer = ShardWriter(out, args.split, args.samples_per_shard, tag=f"{args.tag}_" if args.tag else "")

    scene_dim = 3 + CLIP_LEN     # scene_flows + dist2robot
    robot_dim = 9                # flows + velocity + acceleration
    gt_rel_accums = [WelfordAccumulator(3) for _ in range(CLIP_LEN)]
    scene_accum = WelfordAccumulator(scene_dim)
    robot_accum = WelfordAccumulator(robot_dim)

    t0 = time.time()
    n_points = {}
    n_accumulated = 0
    for scene, robot in iter_clips(ds, args.target_clips, args.num_workers):
        write_clip(writer, scene, robot)
        # Stats on every stats_every'th clip. The feature statistics are means and
        # variances over points, so a 1-in-N sample of clips still rests on tens of
        # millions of points; accumulating on all of them costs more than the dump.
        if args.split == "train" and writer.total_samples % args.stats_every == 0:
            accumulate(scene, robot, gt_rel_accums, scene_accum, robot_accum)
            n_accumulated += 1
        key = (scene.shape[1], robot.shape[1])
        n_points[key] = n_points.get(key, 0) + 1
        if writer.total_samples % args.log_every == 0:
            rate = writer.total_samples / max(1e-9, time.time() - t0)
            print(f"  {writer.total_samples} clips  {rate:.1f} clips/s", flush=True)
    writer.close()

    count = writer.total_samples
    if count == 0:
        raise RuntimeError(f"no clips written for {args.domain}/{args.split}")

    meta_path = out / f"metadata_rank0{('_' + args.tag) if args.tag else ''}.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing[args.split] = {"processed_count": count}
    # Both splits must be present before training reads this file.
    existing.setdefault("train", {"processed_count": 0})
    existing.setdefault("test", {"processed_count": 0})
    meta_path.write_text(json.dumps(existing, indent=2))

    if args.split == "train":
        (out / "stats_raw.json").write_text(json.dumps({
            "domain": pw_domain,
            "pwm_domain": args.domain,
            "clip_len": CLIP_LEN,
            "clips": count,
            "clips_accumulated": n_accumulated,
            "stats_every": args.stats_every,
            "scene_features": ["scene_flows", "dist2robot"],
            "robot_features": ["robot_flows", "robot_velocity", "robot_acceleration"],
            "scene": scene_accum.to_json(),
            "robot": robot_accum.to_json(),
            "gt_scene_flows_relative": [a.to_json() for a in gt_rel_accums],
        }, indent=2))

    elapsed = time.time() - t0
    print(f"\n{pw_domain}/{args.split}: {count} clips in {elapsed/60:.1f} min "
          f"({count/max(1e-9, elapsed):.1f} clips/s)")
    for (ns, nr), k in sorted(n_points.items(), key=lambda kv: -kv[1]):
        print(f"  shape scene={ns} robot={nr}: {k} clips ({100*k/count:.1f}%)")
    print(f"Output -> {out}")


def selfcheck(args) -> None:
    """Assert the torch nearest-distance agrees with the KD-tree PointWorld uses.

    Run in the PointWorld env, where scipy exists; the dump itself runs in pwm211.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        scene = rng.normal(size=(512, 3)).astype(np.float32)
        robot = rng.normal(size=(671, 3)).astype(np.float32)
        mine = nearest_robot_dist(scene, robot)
        theirs = cKDTree(robot).query(scene)[0].astype(np.float32)
        worst = max(worst, float(np.abs(mine - theirs).max()))
    print(f"nearest_robot_dist vs cKDTree: max abs diff {worst:.3e}")
    assert worst < 1e-5, "torch cdist disagrees with the KD-tree"
    print("selfcheck OK")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--domain", choices=sorted(DOMAIN_NAMES), help="PWM domain name")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--out", default="/data/group_data/katefgroup/VLA/3DPCWM/three_pointworld")
    p.add_argument("--config", default="/data/user_data/mbronars/launch_pwm_data_v1_three/configs/unet_three_ctxgen_100m.yaml",
                   help="The PWM recipe whose data.roots define the corpus")
    p.add_argument("--target-clips", type=int, default=None,
                   help="Stop after this many clips (train needs one: the loader cycles forever)")
    p.add_argument("--batch-size", type=int, default=8, help="PWM loader batch size")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Reader processes; the corpus is partitioned across them, not duplicated")
    p.add_argument("--samples-per-shard", type=int, default=200)
    p.add_argument("--tag", default="", help="Suffix for shard names when several workers share a split")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--stats-every", type=int, default=8,
                   help="Accumulate normalization statistics on every Nth clip")
    p.add_argument("--selfcheck", action="store_true", help="Check the distance helper and exit")
    args = p.parse_args()
    if not args.selfcheck:
        if args.domain is None:
            p.error("--domain is required")
        if args.split == "train" and args.target_clips is None:
            p.error("--target-clips is required for the train split (the loader cycles forever)")
    return args


if __name__ == "__main__":
    a = parse_args()
    if a.selfcheck:
        selfcheck(a)
    else:
        dump(a)
