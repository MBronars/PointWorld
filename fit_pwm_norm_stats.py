#!/usr/bin/env python3
"""
fit_pwm_norm_stats.py — The PointWorld normalization statistics for a pwm://
streaming run, fitted without writing a single shard.

With several domains, TRAINING.md-style shared ruler: pool the domains'
statistics and write the pooled numbers under every domain key (how a PWM
normalization artifact expresses one ruler). --per-domain writes each domain's
own. With one domain the two are the same thing.

Feature vectors are assembled in the order the training run requests them:
    scene_features = scene_flows(3) + dist2robot(T)          -> 3 + T
    robot_features = robot_flows(3) + velocity(3) + accel(3)  -> 9
where T = 1 + --pred-horizon is the clip length the run trains on. No normals.

--gripper-open (pickup only) fits the layout of a run trained with
    --robot_features robot_flows,gripper_open,robot_velocity,robot_acceleration   -> 10
    --scene_features scene_flows,gripper_open,dist2robot                          -> 3 + 2T
The binary label's columns are written with mean 0 and variance 1, so the model reads 0/1 as is.

    # the three-domain run (10-frame horizon)
    python fit_pwm_norm_stats.py --out <dir>/norm_stats --windows 20000
    # dex, 16-frame horizon
    python fit_pwm_norm_stats.py --domains pickup_dex --pred-horizon 16 \\
        --recipe /data/user_data/mbronars/launch_pw_dex/configs/unet_dex_v2_filtered2_100m.yaml \\
        --out <dir>/norm_stats
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pwm_stream import PW_TO_PWM, iter_window_arrays
from pwm_three_to_pointworld_wds import WelfordAccumulator, finite_diff


def dist2robot_all_frames(scene0, robot):
    """(T, NS) distance from each frame-0 scene point to the robot at each frame, by KD-tree -- the same
    nearest-neighbour distance the training pipeline computes (dataset_components/transforms.compute_robot_distances).
    The all-pairs cdist version cost 2.4 s per folding window (5.9k cloth x 34k gripper points)."""
    from scipy.spatial import cKDTree
    return np.stack([cKDTree(robot[t]).query(scene0)[0] for t in range(robot.shape[0])]).astype(np.float32)

DEFAULT_RECIPE = "/data/user_data/mbronars/launch_pwm_data_v1_three/configs/unet_three_ctxgen_100m.yaml"
THREE = ["pickup_yikai_small", "pickup_release", "pickup_rl_push"]


def accumulate(scene, robot, gt_rel_accums, scene_accum, robot_accum, opened=None):
    """The statistics PointWorld's pipeline normalizes by, for a clip of any length T."""
    T = scene.shape[0]
    gt_rel = scene - scene[0:1]
    for t in range(T):
        gt_rel_accums[t].update(gt_rel[t])
    vel, acc = finite_diff(robot)
    dist = dist2robot_all_frames(scene[0], robot).T
    if opened is None:
        scene_accum.update(np.concatenate([scene[0], dist], axis=-1))
        robot_accum.update(np.concatenate([robot, vel, acc], axis=-1))
    else:                                              # gather_features' order with gripper_open second
        o = np.asarray(opened, np.float32).reshape(T)
        scene_accum.update(np.concatenate([scene[0], np.repeat(o[None], scene.shape[1], axis=0), dist], axis=-1))
        robot_accum.update(np.concatenate([robot, np.repeat(o[:, None, None], robot.shape[1], axis=1), vel, acc], axis=-1))


class _PartialStats:
    """One DataLoader worker's share of the windows, reduced to Welford partials every `chunk` windows.
    The PWM datasets split their records by (rank, worker) themselves, so the workers read disjoint episodes."""

    def __init__(self, pwm_domain, recipe, clip_len, gripper_open, chunk=25):
        self.args = (pwm_domain, recipe, clip_len, gripper_open, chunk)

    def __iter__(self):
        import torch
        torch.set_num_threads(1)
        pwm_domain, recipe, clip_len, gripper_open, chunk = self.args
        new = lambda: (WelfordAccumulator(3 + clip_len * (2 if gripper_open else 1)),
                       WelfordAccumulator(10 if gripper_open else 9), [WelfordAccumulator(3) for _ in range(clip_len)])
        sc, rb, gt = new(); n = 0
        for fields in iter_window_arrays(pwm_domain, recipe, "train", batch_size=8, clip_len=clip_len,
                                         gripper_open=gripper_open):
            accumulate(fields[0], fields[1], gt, sc, rb, fields[2] if gripper_open else None)
            n += 1
            if n == chunk:
                yield n, sc, rb, gt
                sc, rb, gt = new(); n = 0


def fit_domain_parallel(pwm_domain, recipe, windows, clip_len, log_every, gripper_open, workers):
    import torch
    from torch.utils.data import DataLoader, IterableDataset
    ds = type("PartialStatsDS", (IterableDataset,), {"__iter__": lambda self: iter(_PartialStats(*self.a))})()
    ds.a = (pwm_domain, recipe, clip_len, gripper_open)
    scene = WelfordAccumulator(3 + clip_len * (2 if gripper_open else 1))
    robot = WelfordAccumulator(10 if gripper_open else 9)
    gt_rel = [WelfordAccumulator(3) for _ in range(clip_len)]
    t0, done, last = time.time(), 0, 0
    for n, sc, rb, gt in DataLoader(ds, batch_size=None, num_workers=workers, persistent_workers=False):
        scene.merge(sc); robot.merge(rb)
        for a, b in zip(gt_rel, gt):
            a.merge(b)
        done += n
        if done - last >= log_every:
            print(f"    {done} windows  {done / (time.time() - t0):.1f} w/s", flush=True); last = done
        if done >= windows:
            break
    print(f"  {pwm_domain}: {done} windows of {clip_len} frames in {(time.time()-t0)/60:.1f} min ({workers} workers)")
    return scene, robot, gt_rel


def fit_domain(pwm_domain, recipe, windows, clip_len, log_every, gripper_open=False, workers=1):
    if workers > 1:
        return fit_domain_parallel(pwm_domain, recipe, windows, clip_len, log_every, gripper_open, workers)
    scene = WelfordAccumulator(3 + clip_len * (2 if gripper_open else 1))
    robot = WelfordAccumulator(10 if gripper_open else 9)
    gt_rel = [WelfordAccumulator(3) for _ in range(clip_len)]
    t0 = time.time()
    for i, fields in enumerate(iter_window_arrays(pwm_domain, recipe, "train", batch_size=8, clip_len=clip_len,
                                                  gripper_open=gripper_open), start=1):
        accumulate(fields[0], fields[1], gt_rel, scene, robot, fields[2] if gripper_open else None)
        if i % log_every == 0:
            print(f"    {i} windows  {i / (time.time() - t0):.1f} w/s", flush=True)
        if i >= windows:
            break
    print(f"  {pwm_domain}: {windows} windows of {clip_len} frames in {(time.time()-t0)/60:.1f} min")
    return scene, robot, gt_rel


def entry(scene, robot, gt_rel, gripper_open_cols=None):
    rm, rv, sm, sv = (np.array(a, dtype=np.float64) for a in (robot.mean, robot.variance, scene.mean, scene.variance))
    if gripper_open_cols is not None:                  # the binary label is read as 0/1, not standardized
        rcols, scols = gripper_open_cols
        rm[rcols], rv[rcols], sm[scols], sv[scols] = 0.0, 1.0, 0.0, 1.0
    stats = {
        "robot_features": {"mean": rm.tolist(), "variance": rv.tolist()},
        "scene_features": {"mean": sm.tolist(), "variance": sv.tolist()},
    }
    per_ts = {f"timestep_{t}": {"mean": a.mean.tolist(), "variance": a.variance.tolist()}
              for t, a in enumerate(gt_rel)}
    return stats, {"gt_scene_flows_relative": per_ts}


def main() -> None:
    import torch
    # The reader runs in this process, not in DataLoader workers: pin torch to one thread. With the
    # default (every core of the node) the small per-body tensor ops oversubscribe the job's CPUs and
    # the dex reader drops from 5.7 to 0.9 windows/s.
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", required=True, help="Folder to write norm_stats.json into")
    p.add_argument("--recipe", default=DEFAULT_RECIPE)
    p.add_argument("--domains", default=",".join(THREE), help="PointWorld domain names (see pwm_stream.PW_TO_PWM)")
    p.add_argument("--pred-horizon", type=int, default=10, help="Predicted frames; the clip is 1 + this")
    p.add_argument("--windows", type=int, default=20000,
                   help="Windows per domain: means and variances over points, so 20k windows rest on "
                        "~10M scene points per domain.")
    p.add_argument("--per-domain", action="store_true", help="Each domain's own statistics, not a shared ruler")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--workers", type=int, default=1,
                   help="DataLoader workers, each reading a disjoint share of the episodes (1 = in-process, as before)")
    p.add_argument("--gripper-open", action="store_true",
                   help="Fit the gripper_open layout (pickup only): robot 10, scene 3 + 2T")
    args = p.parse_args()

    clip_len = 1 + args.pred_horizon
    domains = [d for d in args.domains.split(",") if d]
    if args.gripper_open and (len(domains) != 1 or PW_TO_PWM[domains[0]] != "pickup"):
        p.error("--gripper-open is for a pickup-only fit (--domains pickup_release)")
    open_cols = ([3], list(range(3, 3 + clip_len))) if args.gripper_open else None
    fitted = {}
    for pw_domain in domains:
        print(f"fitting {pw_domain} ({PW_TO_PWM[pw_domain]}), clip {clip_len} frames")
        fitted[pw_domain] = fit_domain(PW_TO_PWM[pw_domain], args.recipe, args.windows, clip_len, args.log_every,
                                       gripper_open=args.gripper_open, workers=args.workers)

    statistics, per_timestep = {}, {}
    if args.per_domain or len(fitted) == 1:
        for pw_domain, acc in fitted.items():
            statistics[pw_domain], per_timestep[pw_domain] = entry(*acc, gripper_open_cols=open_cols)
        ruler = "per-domain" if len(fitted) > 1 else "single domain"
    else:
        pooled_scene, pooled_robot = WelfordAccumulator(3 + clip_len), WelfordAccumulator(9)
        pooled_gt = [WelfordAccumulator(3) for _ in range(clip_len)]
        for scene, robot, gt_rel in fitted.values():
            pooled_scene.merge(scene)
            pooled_robot.merge(robot)
            for t in range(clip_len):
                pooled_gt[t].merge(gt_rel[t])
        shared_stats, shared_per_ts = entry(pooled_scene, pooled_robot, pooled_gt)
        for pw_domain in fitted:
            statistics[pw_domain], per_timestep[pw_domain] = shared_stats, shared_per_ts
        ruler = "shared (pooled over the domains)"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "norm_stats.json").write_text(json.dumps(
        {"statistics": statistics, "per_timestep_statistics": per_timestep,
         "fit": {"recipe": args.recipe, "domains": domains, "clip_len": clip_len, "windows_per_domain": args.windows,
                 "gripper_open": bool(args.gripper_open), "workers": args.workers}},
        indent=2))
    first = statistics[next(iter(statistics))]
    print(f"\nruler: {ruler}; scene feature dim {len(first['scene_features']['variance'])}, "
          f"robot feature dim {len(first['robot_features']['variance'])}, {clip_len} timesteps")
    print(f"Wrote {out / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
