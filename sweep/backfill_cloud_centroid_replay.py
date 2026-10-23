#!/usr/bin/env python3
"""Backfill Genesis replay + scoring for cloud_centroid_pw eps that already
have a planner output (positions/batch000.npz) but no DONE marker.

Bug it works around: an earlier worker version looked for
``positions/batch{scenario_batch_idx:03d}.npz`` but ``mppi_mould.py`` always
saves as ``batch000.npz`` (its local enumerate index). So every non-ep_000
scenario silently failed the replay step. The planner output is still there
and reusable.

This script just runs Genesis replay + score on those existing outputs and
writes the DONE marker so subsequent worker passes skip them.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path("/data/group_data/katefgroup/datasets/mbronars")
PWM_EFF = Path("/data/user_data/mbronars/packages/pwm_efficient")
PYTHON_GENESIS = "/data/user_data/mbronars/.miniconda3/envs/genesis/bin/python"
REPLAY_SCRIPT = "/data/user_data/mbronars/packages/Genesis/examples/replay_planner_output.py"

CONFIGS = [
    ("cloud_centroid_pw_seed0_T100",  0),
    ("cloud_centroid_pw_seed0_T150",  0),
    ("cloud_centroid_pw_seed42_T100", 42),
    ("cloud_centroid_pw_seed42_T150", 42),
]

# How many genesis replays to run concurrently. Each pinned to a different GPU
# if multiple are available; sequential CPU fallback if not. Genesis is GPU-
# heavy but each scenario uses one GPU; cluster-side we get one per array
# task. For local backfill, default to whatever GPUs nvidia-smi shows.
N_WORKERS = int(os.environ.get("N_WORKERS", "1"))


def _scenario_offset(seed: int, ep_idx: int) -> tuple[float, float]:
    p = PWM_EFF / "scenarios" / "cloud_centroid" / f"seed_{seed}" / f"ep_{ep_idx:03d}.npz"
    if not p.exists():
        return float("nan"), float("nan")
    d = np.load(p)
    return float(d["target_offset_xy"][0]), float(d["target_offset_xy"][1])


def replay_and_score(ep_dir: Path, seed: int, ep_idx: int) -> str:
    pos = ep_dir / "positions" / "batch000.npz"
    replay_dir = ep_dir / "replay"
    pkl = replay_dir / "replay.pkl"
    done = ep_dir / "DONE"
    score_json = replay_dir / "sim_centroid.json"

    if done.exists():
        return f"{ep_dir.name}: already DONE"

    if not pos.exists():
        return f"{ep_dir.name}: SKIP no batch000.npz"

    if (ep_dir / ".lock").is_dir():
        return f"{ep_dir.name}: SKIP currently locked"

    replay_dir.mkdir(parents=True, exist_ok=True)

    if not pkl.exists():
        rc = subprocess.run(
            [PYTHON_GENESIS, REPLAY_SCRIPT,
             "--input_npz",  str(pos),
             "--record_dir", str(replay_dir)],
            capture_output=True, text=True,
        ).returncode
        if rc != 0:
            return f"{ep_dir.name}: REPLAY FAILED rc={rc}"

    dx, dy = _scenario_offset(seed, ep_idx)
    if np.isnan(dx):
        return f"{ep_dir.name}: SKIP scenario file missing"

    try:
        d = np.load(pos)
        oid = d["object_ids"][0]
        ctx_len = int(d["context_len"])
        pred = d["predicted"][0, 0] if d["predicted"].ndim == 5 else d["predicted"][0]
        block_xy_pred = pred[:, oid == 0, :2]
        anchor_centroid = block_xy_pred[ctx_len - 1].mean(axis=0)
        target = anchor_centroid + np.array([dx, dy], dtype=np.float32)
        with open(pkl, "rb") as f:
            rep = pickle.load(f)
        sim_xy = np.asarray(rep["sim_object_particles"])[:, :, :2]
        sim_init = sim_xy[0].mean(axis=0)
        sim_final = sim_xy[-1].mean(axis=0)
        imag_final = block_xy_pred[-1].mean(axis=0)
        score = {
            "batch_idx":              int(ep_idx),
            "target_offset_xy":       [dx, dy],
            "anchor_centroid_xy":     anchor_centroid.tolist(),
            "target_xy":              target.tolist(),
            "sim_centroid_dist":      float(np.linalg.norm(sim_final - target)),
            "sim_cloud_shift":        float(np.linalg.norm(sim_final - sim_init)),
            "imagined_centroid_dist": float(np.linalg.norm(imag_final - target)),
            "imagined_cloud_shift":   float(np.linalg.norm(imag_final - anchor_centroid)),
        }
        score_json.write_text(json.dumps(score, indent=2))
        done.touch()
        return (f"{ep_dir.name}: OK  sim_dist={score['sim_centroid_dist']*100:.2f}cm  "
                f"sim_shift={score['sim_cloud_shift']*100:.2f}cm")
    except Exception as e:
        return f"{ep_dir.name}: SCORE FAILED: {e!r}"


def main() -> int:
    todo: list[tuple[Path, int, int]] = []
    for tag, seed in CONFIGS:
        cdir = ROOT / tag
        if not cdir.is_dir():
            continue
        for ep_dir in sorted((cdir / f"seed_{seed}").glob("ep_*")):
            ep_idx = int(ep_dir.name.split("_")[1])
            pos = ep_dir / "positions" / "batch000.npz"
            done = ep_dir / "DONE"
            if pos.exists() and not done.exists() and not (ep_dir / ".lock").is_dir():
                todo.append((ep_dir, seed, ep_idx))
    print(f"backfill candidates: {len(todo)}")
    if not todo:
        return 0

    if N_WORKERS <= 1:
        for ep_dir, seed, ep_idx in todo:
            print(replay_and_score(ep_dir, seed, ep_idx), flush=True)
        return 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(replay_and_score, ep_dir, s, i): ep_dir
                for ep_dir, s, i in todo}
        for f in as_completed(futs):
            print(f.result(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
