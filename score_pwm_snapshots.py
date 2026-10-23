#!/usr/bin/env python3
"""
score_pwm_snapshots.py — Score PointWorld snapshots of a pwm:// run on the
held-out windows, and dump clouds for a viewer.

For each checkpoint, every held-out window of each domain is predicted once:
PointWorld's native task — frame 0 of the scene plus the gripper's whole
trajectory in, the scene's next 10 frames out. Batches come from the run's own
test pipeline (evaluation.tester.Tester builds the model from the checkpoint's
contract and the loader from the same build_dataloader training uses), so the
windows and transforms are exactly what the run's eval-test sees.

The held-out streams are deterministic single passes (one pinned window per
episode) that cycle; each domain is collected until its pass is complete, so
every checkpoint is scored on the identical set of windows. The script asserts
that, by comparing the ground truth across checkpoints.

Per window, in millimetres, over the scene points and the 10 predicted frames:
    err_mean   mean |pred - gt|
    err_final  mean |pred - gt| at the last frame
    motion     mean |gt_t - gt_0| — the error of predicting no motion at all
and the per-frame curve of err.

Writes <out>/scores.json and <out>/clouds.npz (int16 millimetres: GT scene,
gripper, and each checkpoint's prediction, for the selected windows — the first
N of a fixed-seed random draw per domain, then the worst K by the LAST
checkpoint's err_mean).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HELD_OUT = {"pickup_yikai_small": 334, "pickup_release": 200, "pickup_rl_push": 200}
RECIPE = "/data/user_data/mbronars/launch_pwm_data_v1_three/configs/unet_three_ctxgen_100m.yaml"
STATS = "/data/group_data/katefgroup/runs_pw_mbronars/three_stream/norm_stats"
PWM_OF = {"pickup_yikai_small": "yikai", "pickup_release": "pickup", "pickup_rl_push": "rl_push"}


def build_tester(model_path: str):
    from arguments import parse_args
    from evaluation.tester import Tester
    domains = list(HELD_OUT)
    argv = [
        "score",
        "--domains", ",".join(domains),
        "--data_dirs", ",".join(f"pwm://{PWM_OF[d]}@{RECIPE}" for d in domains),
        "--norm_stats_path", STATS,
        "--robot_features", "robot_flows,robot_velocity,robot_acceleration",
        "--scene_features", "scene_flows,dist2robot",
        "--grid_size", "0.001",
        "--batch_size", "8",
        "--num_workers", "1",
        "--model_path", model_path,
        "--eval_skip_viz", "true",
        "--allow_missing_confidence_mask", "true",
        "--eval_exp_name", "score_pwm_snapshots",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        args = parse_args()
    finally:
        sys.argv = old
    args.model_path, args.eval_exp_name = model_path, None
    return Tester(args)


@torch.no_grad()
def score_checkpoint(model_path: str):
    tester = build_tester(model_path)
    tester.model.eval()
    dl, _ = tester._build_eval_loader("test")
    need = dict(HELD_OUT)
    rows = {d: [] for d in HELD_OUT}
    t0 = time.time()
    for batch in dl:
        doms = [str(d) for d in batch["__domain__"]]
        if all(len(rows[d]) >= need[d] for d in need):
            break
        gpu = {k: (v.to(tester.device, non_blocking=True) if torch.is_tensor(v) else v)
               for k, v in batch.items()}
        out = tester.model(gpu, training=False)
        pred = out["scene_flows"].float().cpu().numpy()          # (B, T, Ns, 3)
        gt = batch["gt_scene_flows"].float().numpy()              # (B, T, Ns, 3)
        exists = batch["scene_exists"].numpy()                    # (B, T, Ns)
        robot = batch["robot_flows"].float().numpy()              # (B, T, Nr, 3)
        r_exists = batch["robot_exists"].numpy()                  # (B, T, Nr)
        for i, d in enumerate(doms):
            if len(rows[d]) >= need[d]:
                continue
            m = exists[i, 0].astype(bool)
            rm = r_exists[i, 0].astype(bool)
            g, p = gt[i][:, m], pred[i][:, m]
            e = np.linalg.norm(p - g, axis=-1) * 1000.0            # (T, Ns) mm
            mot = np.linalg.norm(g - g[0:1], axis=-1) * 1000.0
            rows[d].append({
                "err_curve": e.mean(axis=1)[1:].tolist(),
                "err_mean": float(e[1:].mean()),
                "err_final": float(e[-1].mean()),
                "motion": float(mot[1:].mean()),
                "motion_curve": mot.mean(axis=1)[1:].tolist(),
                "gt": g, "pred": p, "robot": robot[i][:, rm],
            })
    print(f"  scored {sum(len(v) for v in rows.values())} windows in {time.time()-t0:.0f}s "
          f"({', '.join(f'{d}={len(v)}' for d, v in rows.items())})")
    meta = {"sample_count": int(tester.sample_count), "batch_count": int(tester.batch_count)}
    del tester
    torch.cuda.empty_cache()
    return rows, meta


def q16(a: np.ndarray) -> np.ndarray:
    return np.clip(np.round(a * 1000.0), -32767, 32767).astype(np.int16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoints", nargs="+", required=True, help="label=path, in budget order")
    p.add_argument("--out", required=True)
    p.add_argument("--random", type=int, default=8)
    p.add_argument("--worst", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in a.checkpoints:
        label, path = spec.split("=", 1)
        print(f"checkpoint {label}: {path}")
        results[label] = score_checkpoint(path)

    labels = list(results)
    ref = results[labels[0]][0]
    for lab in labels[1:]:
        for d in HELD_OUT:
            for j, (x, y) in enumerate(zip(ref[d], results[lab][0][d])):
                if x["gt"].shape != y["gt"].shape or not np.array_equal(x["gt"], y["gt"]):
                    raise AssertionError(f"{d} window {j} differs between {labels[0]} and {lab}")

    scores = {"checkpoints": {}, "held_out": HELD_OUT, "selection": {}}
    clouds = {}
    last = results[labels[-1]][0]
    for lab in labels:
        rows, meta = results[lab]
        scores["checkpoints"][lab] = {"meta": meta, "domains": {}}
        for d, rs in rows.items():
            scores["checkpoints"][lab]["domains"][d] = {
                "err_mean": [r["err_mean"] for r in rs],
                "err_final": [r["err_final"] for r in rs],
                "motion": [r["motion"] for r in rs],
                "err_curve_mean": np.mean([r["err_curve"] for r in rs], axis=0).tolist(),
                "motion_curve_mean": np.mean([r["motion_curve"] for r in rs], axis=0).tolist(),
            }
    for d in HELD_OUT:
        n = len(last[d])
        rand = np.random.default_rng(a.seed).choice(n, size=min(a.random, n), replace=False).tolist()
        order = np.argsort([-r["err_mean"] for r in last[d]]).tolist()
        worst = [j for j in order if j not in rand][: a.worst]
        scores["selection"][d] = {"random": rand, "worst": worst}
        for j in rand + worst:
            key = f"{d}/{j}"
            clouds[f"{key}/gt"] = q16(ref[d][j]["gt"])
            clouds[f"{key}/robot"] = q16(ref[d][j]["robot"])
            for lab in labels:
                clouds[f"{key}/pred/{lab}"] = q16(results[lab][0][d][j]["pred"])

    (out / "scores.json").write_text(json.dumps(scores))
    np.savez_compressed(out / "clouds.npz", **clouds)
    print(f"wrote {out/'scores.json'} and {out/'clouds.npz'}")
    for lab in labels:
        for d in HELD_OUT:
            s = scores["checkpoints"][lab]["domains"][d]
            print(f"  {lab:>8} {d:<20} err {np.mean(s['err_mean']):6.2f} mm  final {np.mean(s['err_final']):6.2f} mm  "
                  f"no-motion {np.mean(s['motion']):6.2f} mm")


if __name__ == "__main__":
    main()
