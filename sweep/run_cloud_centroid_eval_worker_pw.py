"""Claim-and-go worker for the cloud-centroid PointWorld evaluation.

Mirrors ``pwm_efficient/sweep/run_cloud_centroid_eval_worker.py`` but invokes
``mppi_mould.py --objective centroid`` (PointWorld's MPPI-in-WM) instead of
the flow-matching MPPI planner.

For each (seed, ep_idx) scenario in
  pwm_efficient/scenarios/cloud_centroid/seed_{S}/ep_{NN}.npz
this runs ``mppi_mould.py`` once with:
  - ``--objective centroid``
  - ``--centroid_offset_x / -y``  from the scenario
  - ``--demo_idx batch_idx``      (PointWorld iterates the test set by demo
                                   index; scenarios pin batch_idx=ep_idx)
  - ``--total_frames TOTAL_FRAMES`` (env var, defaults to 150)

Then runs Genesis-MPM replay on positions/batch{batch_idx:03d}.npz to compute
the headline ``sim_centroid_dist`` metric.

Lock files use the same mkdir-atomic + stale-takeover pattern as
``run_cloud_centroid_eval_worker.py`` so multiple workers can run in parallel
without double-running a scenario.

Env-var knobs:
  SEEDS              comma-separated, e.g. "0,42"   (default "0,42")
  N_PER_SEED         scenarios per seed             (default 50)
  TOTAL_FRAMES       planner horizon                (default 150)
  SWEEP_NAME         output bucket                  (default ``cloud_centroid_pw_T{TOTAL}``)
  CHECKPOINT         PointWorld ckpt path
  POPULATION_SIZE    MPPI K                         (default 128)
  CHUNK_SIZE         forward chunk size             (default 32)
  RUN_REPLAY         "1" to run Genesis-MPM replay  (default "1")
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


HERE = Path(__file__).parent
POINTWORLD = HERE.parent
PWM_EFF    = Path("/data/user_data/mbronars/packages/pwm_efficient")
sys.path.insert(0, str(POINTWORLD))


# ── Config from env ─────────────────────────────────────────────────────────
SEEDS         = [int(s) for s in os.environ.get("SEEDS", "0,42").split(",")]
N_PER_SEED    = int(os.environ.get("N_PER_SEED", "50"))
TOTAL_FRAMES  = int(os.environ.get("TOTAL_FRAMES", "150"))
SWEEP_NAME    = os.environ.get("SWEEP_NAME", f"cloud_centroid_pw_T{TOTAL_FRAMES}")
CHECKPOINT    = os.environ.get(
    "CHECKPOINT",
    str(POINTWORLD / "train_logs" / "mould_new_6k_1gpu" / "model-eval-cc.pt"),
)
POPULATION_SIZE = int(os.environ.get("POPULATION_SIZE", "128"))
CHUNK_SIZE      = int(os.environ.get("CHUNK_SIZE", "32"))
TEMPERATURE     = float(os.environ.get("TEMPERATURE", "0.3"))
LOOKAHEAD_COMMIT_SIZE = int(os.environ.get("LOOKAHEAD_COMMIT_SIZE", "5"))
PLANNER_SEED    = int(os.environ.get("PLANNER_SEED", "0"))
RUN_REPLAY      = os.environ.get("RUN_REPLAY", "1") == "1"
_clamp = os.environ.get("CENTROID_CLAMP_XY", "").strip()
CENTROID_CLAMP_XY = float(_clamp) if _clamp else None

DATA_DIR        = "/data/group_data/katefgroup/VLA/3DPCWM/mould_new_6k_data_pointworld"
TEST_DIR        = "/data/group_data/katefgroup/VLA/3DPCWM/deformable_mould_new_6k_wds/test"
PYTHON_PW       = "/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python"
PYTHON_GENESIS  = "/data/user_data/mbronars/.miniconda3/envs/genesis/bin/python"
REPLAY_SCRIPT   = "/data/user_data/mbronars/packages/Genesis/examples/replay_planner_output.py"

OUTPUT_ROOT     = Path("/data/group_data/katefgroup/datasets/mbronars")
LOCK_STALE_S    = 6 * 3600


def ensure_symlink() -> None:
    sweep_dir = OUTPUT_ROOT / SWEEP_NAME
    sweep_dir.mkdir(parents=True, exist_ok=True)
    inference_results = POINTWORLD / "inference_results"
    inference_results.mkdir(exist_ok=True)
    link = inference_results / SWEEP_NAME
    if link.is_symlink():
        if link.resolve() == sweep_dir.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink — refusing to clobber.")
    link.symlink_to(sweep_dir, target_is_directory=True)


def build_cli(out_name: str, batch_idx: int, dx: float, dy: float) -> List[str]:
    cmd = [
        PYTHON_PW, str(POINTWORLD / "mppi_mould.py"),
        "--checkpoint",   CHECKPOINT,
        "--data_dir",     DATA_DIR,
        "--test_dir",     TEST_DIR,
        "--output_name",  out_name,
        "--inference_root", str(OUTPUT_ROOT),
        "--demo_idx",     str(batch_idx),
        "--seed",         str(PLANNER_SEED),
        "--objective",    "centroid",
        "--centroid_offset_x", f"{dx:.6f}",
        "--centroid_offset_y", f"{dy:.6f}",
        "--total_frames",       str(TOTAL_FRAMES),
        "--lookahead_commit_size", str(LOOKAHEAD_COMMIT_SIZE),
        "--population_size", str(POPULATION_SIZE),
        "--chunk_size",      str(CHUNK_SIZE),
        "--temperature",     str(TEMPERATURE),
    ]
    if CENTROID_CLAMP_XY is not None:
        cmd += ["--centroid_clamp_xy", f"{CENTROID_CLAMP_XY:.4f}"]
    if os.environ.get("HORIZON_MODE"):                      # final: score the plan's last frame (2026-09-26 option); unset = original
        cmd += ["--horizon_mode", os.environ["HORIZON_MODE"]]
    return cmd


def run_replay_and_score(out_dir: Path, batch_idx: int,
                         dx: float, dy: float) -> int:
    """Genesis-MPM playback → sim_centroid.json (the headline metric)."""
    import pickle
    # mppi_mould.py always saves the per-call result as batch000.npz, because
    # its internal batch_idx is enumerate(args.demo_idx) and we pass exactly
    # one --demo_idx per scenario. The scenario's batch_idx is just the test-
    # set demo index, not the on-disk file name.
    pos = out_dir / "positions" / "batch000.npz"
    replay_dir = out_dir / "replay"
    pkl = replay_dir / "replay.pkl"
    if not pos.exists():
        print(f"  [replay] {pos} missing", flush=True)
        return 1
    replay_dir.mkdir(parents=True, exist_ok=True)
    if not pkl.exists():
        rc = subprocess.run([
            PYTHON_GENESIS, REPLAY_SCRIPT,
            "--input_npz",  str(pos),
            "--record_dir", str(replay_dir),
        ]).returncode
        if rc != 0:
            print(f"  [replay] FAILED rc={rc}", flush=True)
            return rc
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
        sim_xy   = np.asarray(rep["sim_object_particles"])[:, :, :2]
        sim_init = sim_xy[0].mean(axis=0)
        sim_final = sim_xy[-1].mean(axis=0)
        imag_final = block_xy_pred[-1].mean(axis=0)
        score = {
            "batch_idx":              int(batch_idx),
            "target_offset_xy":       [dx, dy],
            "anchor_centroid_xy":     anchor_centroid.tolist(),
            "target_xy":              target.tolist(),
            "sim_centroid_dist":      float(np.linalg.norm(sim_final - target)),
            "sim_cloud_shift":        float(np.linalg.norm(sim_final - sim_init)),
            "imagined_centroid_dist": float(np.linalg.norm(imag_final - target)),
            "imagined_cloud_shift":   float(np.linalg.norm(imag_final - anchor_centroid)),
        }
        (replay_dir / "sim_centroid.json").write_text(json.dumps(score, indent=2))
        print(f"  [score] sim_dist={score['sim_centroid_dist']*100:.2f}cm  "
              f"sim_shift={score['sim_cloud_shift']*100:.2f}cm  "
              f"imag_dist={score['imagined_centroid_dist']*100:.2f}cm", flush=True)
    except Exception as e:
        print(f"  [score] FAILED: {e}", flush=True)
        return 1
    return 0


def try_claim(out_dir: Path) -> tuple[bool, str]:
    lock = out_dir / ".lock"
    try:
        lock.mkdir(exist_ok=False)
        (lock / "owner.txt").write_text(
            f"host={socket.gethostname()}  pid={os.getpid()}  "
            f"start={time.time():.0f}\n"
        )
        return True, "claimed"
    except FileExistsError:
        pass
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return try_claim(out_dir)
    if age > LOCK_STALE_S:
        try:
            (lock / "owner.txt").write_text(
                f"host={socket.gethostname()}  pid={os.getpid()}  "
                f"start={time.time():.0f}  (took over stale lock age={age:.0f}s)\n"
            )
            return True, f"stale (age={age/3600:.1f}h)"
        except Exception:
            return False, "stale-takeover-failed"
    return False, f"locked (age={age/60:.0f}m)"


def release_lock(out_dir: Path) -> None:
    lock = out_dir / ".lock"
    try:
        (lock / "owner.txt").unlink(missing_ok=True)
        lock.rmdir()
    except (FileNotFoundError, OSError):
        pass


def run_one_scenario(seed: int, ep_idx: int) -> int:
    scenario_path = (PWM_EFF / "scenarios" / "cloud_centroid" /
                     f"seed_{seed}" / f"ep_{ep_idx:03d}.npz")
    if not scenario_path.exists():
        print(f"  [seed={seed} ep={ep_idx:02d}] MISSING {scenario_path}", flush=True)
        return 2
    d = np.load(scenario_path)
    batch_idx = int(d["batch_idx"])
    dx, dy    = float(d["target_offset_xy"][0]), float(d["target_offset_xy"][1])

    out_name = f"{SWEEP_NAME}/seed_{seed}/ep_{ep_idx:03d}"
    out_dir  = OUTPUT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    done_marker = out_dir / "DONE"
    if done_marker.exists():
        return 0

    claimed, why = try_claim(out_dir)
    if not claimed:
        print(f"  [seed={seed} ep={ep_idx:02d}] skip — {why}", flush=True)
        return 1

    try:
        (out_dir / "scenario.txt").write_text(
            f"seed={seed}  ep={ep_idx}  batch_idx={batch_idx}  "
            f"target_offset_xy=({dx:.6f}, {dy:.6f})  "
            f"total_frames={TOTAL_FRAMES}\n"
        )
        t0 = time.time()
        # Skip planner if a usable batch000.npz already exists (e.g. left
        # behind by a previous worker pass whose replay step failed). Saves
        # the ~70-140s planner cost; we only re-run replay + score.
        existing_pos = out_dir / "positions" / "batch000.npz"
        if existing_pos.exists():
            print(f"  [seed={seed} ep={ep_idx:02d}] reusing existing "
                  f"{existing_pos.name} — skipping planner", flush=True)
            rc = 0
        else:
            cli = build_cli(out_name, batch_idx, dx, dy)
            print(f"  [seed={seed} ep={ep_idx:02d}] batch_idx={batch_idx}  "
                  f"target=({dx:+.3f},{dy:+.3f})  T={TOTAL_FRAMES}  invoking mppi_mould",
                  flush=True)
            rc = subprocess.run(cli, cwd=str(POINTWORLD)).returncode
        plan_elapsed = time.time() - t0
        if rc != 0:
            (out_dir / "wallclock_sec.txt").write_text(f"{plan_elapsed:.2f}\n")
            print(f"  [seed={seed} ep={ep_idx:02d}] plan rc={rc}", flush=True)
            return rc

        if RUN_REPLAY:
            rc = run_replay_and_score(out_dir, batch_idx, dx, dy)
            total_elapsed = time.time() - t0
        else:
            total_elapsed = plan_elapsed
        (out_dir / "wallclock_sec.txt").write_text(f"{total_elapsed:.2f}\n")
        print(f"  [seed={seed} ep={ep_idx:02d}] rc={rc}  total={total_elapsed:.0f}s "
              f"(plan {plan_elapsed:.0f}s)", flush=True)
        if rc == 0:
            done_marker.touch()
        return rc
    finally:
        release_lock(out_dir)


def main() -> int:
    ensure_symlink()
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    print(f"[worker {worker_id}] cloud_centroid_pw starting "
          f"(sweep={SWEEP_NAME}, seeds={SEEDS}, n_per_seed={N_PER_SEED}, "
          f"T={TOTAL_FRAMES})", flush=True)

    n_done = n_skip = n_fail = 0
    for seed in SEEDS:
        for ep_idx in range(N_PER_SEED):
            rc = run_one_scenario(seed, ep_idx)
            if rc == 0:
                n_done += 1
            elif rc == 1:
                n_skip += 1
            else:
                n_fail += 1

    print(f"[worker {worker_id}] done={n_done}  skipped={n_skip}  failed={n_fail}",
          flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
