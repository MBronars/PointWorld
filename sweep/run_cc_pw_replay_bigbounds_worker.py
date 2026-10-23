"""Claim-and-go worker that re-runs the Genesis playback for previously-
planned cloud_centroid_pw scenarios using a 5× larger MPM domain
(±0.50 m xy, [-0.06, 0.30] z) so MPM boundary clamping can't affect the
sim metrics. Planner is NOT re-run — we just consume the saved
``positions/batch000.npz`` files.

Iterates the 4 seed42 L40S sweeps × 50 eps each (200 episodes total),
claims via ``.lock_bb`` (mkdir-atomic), and writes outputs to
``<scn>/replay_bigbounds/`` so the original ``<scn>/replay/`` is preserved.

Per-episode steps:
  1. Run ``replay_with_big_bounds.py --input_npz <pos.npz>
     --record_dir <scn>/replay_bigbounds``
  2. Score the resulting ``replay.pkl`` against the scenario target →
     ``<scn>/replay_bigbounds/sim_centroid.json``
  3. Touch ``<scn>/DONE_BB``
"""
from __future__ import annotations

import json
import os
import pickle
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
POINTWORLD = HERE.parent
GENESIS_EXAMPLES = Path("/data/user_data/mbronars/packages/Genesis/examples")
sys.path.insert(0, str(POINTWORLD))

PYTHON_GENESIS = "/data/user_data/mbronars/.miniconda3/envs/genesis/bin/python"
REPLAY_SCRIPT = str(GENESIS_EXAMPLES / "replay_with_big_bounds.py")
LOCK_STALE_S = 6 * 3600

OUTPUT_ROOT = Path("/data/group_data/katefgroup/datasets/mbronars")

# Only the T=150 sweeps — paddle reach in T=100 stays under the original
# ±10 cm domain most of the time, while T=150 is the one that needs the
# ±70 cm box.
SWEEPS = [
    ("cloud_centroid_pw_seed42_T150_K128_l40s", 42),
    ("cloud_centroid_pw_seed42_T150_K256_l40s", 42),
]
N_PER_SEED = int(os.environ.get("N_PER_SEED", "50"))


def try_claim(scn_dir: Path) -> tuple[bool, str]:
    lock = scn_dir / ".lock_bb"
    try:
        lock.mkdir(exist_ok=False)
        (lock / "owner.txt").write_text(
            f"host={socket.gethostname()} pid={os.getpid()} start={time.time():.0f}\n"
        )
        return True, "claimed"
    except FileExistsError:
        pass
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return try_claim(scn_dir)
    if age > LOCK_STALE_S:
        try:
            (lock / "owner.txt").write_text(
                f"host={socket.gethostname()} pid={os.getpid()} start={time.time():.0f}  "
                f"(took over stale age={age:.0f}s)\n"
            )
            return True, f"stale-takeover ({age/3600:.1f}h)"
        except Exception:
            return False, "stale-takeover-failed"
    return False, f"locked ({age/60:.0f}m)"


def release_lock(scn_dir: Path) -> None:
    lock = scn_dir / ".lock_bb"
    try:
        (lock / "owner.txt").unlink(missing_ok=True)
        lock.rmdir()
    except (FileNotFoundError, OSError):
        pass


def score_replay(scn_dir: Path, bb_dir: Path) -> int:
    """Compute sim_centroid_dist / sim_cloud_shift from the bigbounds
    replay.pkl + the original planner positions/batch000.npz.
    """
    pos_npz = scn_dir / "positions" / "batch000.npz"
    pkl = bb_dir / "replay.pkl"
    if not pos_npz.exists() or not pkl.exists():
        print(f"  [score] missing inputs: pos={pos_npz.exists()} pkl={pkl.exists()}", flush=True)
        return 1
    try:
        d = np.load(pos_npz)
        oid = d["object_ids"][0]
        ctx_len = int(d["context_len"])
        pred = d["predicted"][0, 0] if d["predicted"].ndim == 5 else d["predicted"][0]
        block_xy_pred = pred[:, oid == 0, :2]
        anchor_centroid = block_xy_pred[ctx_len - 1].mean(axis=0)
        target_xy = np.asarray(d["centroid_target_xy"], dtype=np.float32)
        # target_offset_xy = target - anchor (round-trip safe)
        offset_xy = (target_xy - anchor_centroid).astype(np.float32)

        with open(pkl, "rb") as f:
            rep = pickle.load(f)
        sim_xy = np.asarray(rep["sim_object_particles"])[:, :, :2]
        sim_init = sim_xy[0].mean(axis=0)
        sim_final = sim_xy[-1].mean(axis=0)
        imag_final = block_xy_pred[-1].mean(axis=0)

        score = {
            "target_offset_xy":       offset_xy.tolist(),
            "anchor_centroid_xy":     anchor_centroid.tolist(),
            "target_xy":              target_xy.tolist(),
            "sim_centroid_dist":      float(np.linalg.norm(sim_final - target_xy)),
            "sim_cloud_shift":        float(np.linalg.norm(sim_final - sim_init)),
            "imagined_centroid_dist": float(np.linalg.norm(imag_final - target_xy)),
            "imagined_cloud_shift":   float(np.linalg.norm(imag_final - anchor_centroid)),
            "bounds":                 {"low": [-0.70, -0.70, -0.15],
                                       "high": [ 0.70,  0.70,  0.35]},
        }
        (bb_dir / "sim_centroid.json").write_text(json.dumps(score, indent=2))
        print(f"  [score] sim_dist={score['sim_centroid_dist']*100:.2f}cm  "
              f"sim_shift={score['sim_cloud_shift']*100:.2f}cm", flush=True)
        return 0
    except Exception as e:
        print(f"  [score] FAILED: {e!r}", flush=True)
        return 1


def run_one(scn_dir: Path) -> int:
    pos_npz = scn_dir / "positions" / "batch000.npz"
    if not pos_npz.exists():
        print(f"  skip {scn_dir} — no batch000.npz", flush=True)
        return 2
    done = scn_dir / "DONE_BB"
    if done.exists():
        return 0
    claimed, why = try_claim(scn_dir)
    if not claimed:
        print(f"  skip {scn_dir.parts[-3:]}: {why}", flush=True)
        return 1
    bb_dir = scn_dir / "replay_bigbounds"
    bb_dir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        print(f"  REPLAY {'/'.join(scn_dir.parts[-3:])}  →  {bb_dir}/", flush=True)
        rc = subprocess.run(
            [PYTHON_GENESIS, REPLAY_SCRIPT,
             "--input_npz",  str(pos_npz),
             "--record_dir", str(bb_dir)],
            cwd=str(GENESIS_EXAMPLES),
        ).returncode
        if rc != 0:
            print(f"  [replay] rc={rc}", flush=True)
            return rc
        rc = score_replay(scn_dir, bb_dir)
        if rc != 0:
            return rc
        elapsed = time.time() - t0
        (bb_dir / "wallclock_sec.txt").write_text(f"{elapsed:.2f}\n")
        done.touch()
        print(f"  [done] total={elapsed:.0f}s", flush=True)
        return 0
    finally:
        release_lock(scn_dir)


def main() -> int:
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    print(f"[worker {worker_id}] cc_pw bigbounds replay starting "
          f"(±0.50 m xy, [-0.06, 0.30] z)", flush=True)

    n_done = n_skip = n_fail = 0
    for sweep_name, seed in SWEEPS:
        sweep_dir = OUTPUT_ROOT / sweep_name
        if not sweep_dir.is_dir():
            print(f"  (no sweep dir {sweep_dir})", flush=True)
            continue
        for ep in range(N_PER_SEED):
            scn = sweep_dir / f"seed_{seed}" / f"ep_{ep:03d}"
            if not (scn / "DONE").exists():
                # need the planner output to exist
                continue
            rc = run_one(scn)
            if rc == 0:
                n_done += 1
            elif rc == 1:
                n_skip += 1
            else:
                n_fail += 1
    print(f"[worker {worker_id}] done={n_done} skipped={n_skip} failed={n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
