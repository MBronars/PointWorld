#!/usr/bin/env python3
"""
viz_pcds.py

Render the predicted block point clouds saved by imagined_mppi.py (with
--dump_n_random_pcds > 0). One GIF per replan, animating the predicted
point cloud across the planning horizon with the candidate's gripper
position moving alongside. Winner + N random selections shown as
side-by-side subplots, point count (N_w) annotated per AR window.

Usage:
    python viz_pcds.py /data/user_data/mbronars/packages/PointWorld/runs/<run>/<ep>/
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_replan_pcds(replan_npz: Path):
    """Returns:
        selected_pcds: list of (n_selected, PRED_FRAMES, N_w, 3) per window
        rest: dict of scalars / small arrays needed for plotting.
    """
    with np.load(replan_npz, allow_pickle=False) as d:
        keys = set(d.files)
        if "selected_k_indices" not in keys:
            return None  # this replan wasn't dumped with --dump_n_random_pcds

        sel_k       = d["selected_k_indices"]
        sel_costs   = d["selected_costs"]
        sel_grip    = d["selected_gripper_xy"]      # (n_sel, H_active, 2)
        npts_per_w  = d["npts_per_window"]
        n_ar_windows = int(d["n_ar_windows"])
        H_active    = int(d["H_active"])
        step        = int(d["step"])
        block_now   = d["block_xy_at_replan"]
        gripper_now = d["gripper_xy_at_replan"]
        target_xy   = d["target_xy"]
        plan_time   = float(d["plan_time_sec"])
        mean_cost   = float(d["mean_cost"])
        best_cost   = float(d["best_cost"])
        per_window  = [d[f"selected_pcds_w{w}"] for w in range(n_ar_windows)]

    return dict(
        selected_pcds = per_window,
        selected_k    = sel_k,
        selected_costs = sel_costs,
        selected_gripper_xy = sel_grip,
        npts_per_window = npts_per_w,
        n_ar_windows  = n_ar_windows,
        H_active      = H_active,
        step          = step,
        block_now     = block_now,
        gripper_now   = gripper_now,
        target_xy     = target_xy,
        plan_time     = plan_time,
        mean_cost     = mean_cost,
        best_cost     = best_cost,
    )


def _flat_frames(per_window, sel_idx):
    """Flatten one selected candidate's per-window pcds (n_sel, T, N_w, 3) into
    a per-frame list [(N_w, 3), ...] of length H = sum_w PRED_FRAMES."""
    frames = []
    for w_arr in per_window:                # w_arr: (n_sel, PRED_FRAMES, N_w, 3)
        for f in range(w_arr.shape[1]):
            frames.append(w_arr[sel_idx, f])
    return frames


def _candidate_per_frame(per_window, sel_idx):
    """One selected candidate's per-frame block point clouds, flat list of
    length H_active. Each element is (N_w, 3) — N_w varies per AR window."""
    frames = []
    for w_arr in per_window:                # (n_sel, PRED_FRAMES, N_w, 3)
        for f in range(w_arr.shape[1]):
            frames.append(w_arr[sel_idx, f])
    return frames


def viz_one(replan_npz: Path, save_path: Path,
            fps: int = 6, dpi: int = 100) -> bool:
    d = _load_replan_pcds(replan_npz)
    if d is None:
        return False

    n_sel    = d["selected_k"].shape[0]
    H_active = d["H_active"]

    # Per-candidate per-frame point clouds + per-frame gripper xy.
    cand_frames = [_candidate_per_frame(d["selected_pcds"], col)
                   for col in range(n_sel)]
    grip_paths  = [d["selected_gripper_xy"][col] for col in range(n_sel)]  # (H_active, 2)

    # Shared axis limits across all candidates × frames so the camera doesn't jitter.
    all_xy = np.concatenate(
        [pts[:, :2] for col in cand_frames for pts in col] +
        [g[:, :2] for g in grip_paths] +
        [d["block_now"][None, :], d["gripper_now"][None, :], d["target_xy"][None, :]],
        axis=0,
    )
    pad  = 0.015
    xmin, ymin = all_xy.min(axis=0) - pad
    xmax, ymax = all_xy.max(axis=0) + pad

    npts_per_w = d["npts_per_window"]
    npts_str   = " | ".join(f"w{w}:N={n}" for w, n in enumerate(npts_per_w))

    frames_rgb = []
    for t in range(H_active):
        fig, axes = plt.subplots(1, n_sel, figsize=(5.5 * n_sel + 1, 5.8))
        if n_sel == 1:
            axes = [axes]

        # Which AR window does frame t belong to? (For PRED_FRAMES windows.)
        cum = 0
        w_idx = 0
        for w in range(d["n_ar_windows"]):
            n_in_w = d["selected_pcds"][w].shape[1]   # PRED_FRAMES
            if t < cum + n_in_w:
                w_idx = w
                break
            cum += n_in_w

        for col in range(n_sel):
            ax = axes[col]
            rank_str = "WINNER" if col == 0 else f"random #{col}"
            k_idx    = int(d["selected_k"][col])
            cost     = float(d["selected_costs"][col])

            # All earlier predicted frames very faded (a "trail").
            for tt in range(t):
                pts = cand_frames[col][tt]
                ax.scatter(pts[:, 0], pts[:, 1], s=2,
                           color="lightgray", alpha=0.12, edgecolors="none")
            # Current predicted frame, solid.
            pts = cand_frames[col][t]
            ax.scatter(pts[:, 0], pts[:, 1], s=8, color="C0",
                       alpha=0.85, edgecolors="none", label="predicted block")

            # Gripper path (full faded) + current gripper marker.
            g = grip_paths[col]
            ax.plot([d["gripper_now"][0], *g[:, 0]],
                    [d["gripper_now"][1], *g[:, 1]],
                    "--", color="orange", lw=1.0, alpha=0.35)
            ax.scatter(g[t, 0], g[t, 1], color="orange", marker="^",
                       s=110, edgecolor="black", zorder=6, label="gripper now")

            # Static anchors.
            ax.scatter(*d["block_now"], color="black", marker="s", s=110,
                       edgecolor="white", zorder=5, label="block t=0")
            ax.scatter(*d["target_xy"], color="red",   marker="*", s=280,
                       edgecolor="black", zorder=6, label="target")

            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal"); ax.grid(alpha=0.3)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_title(f"{rank_str}   k={k_idx}   cost={cost:.4f}",
                         fontsize=10)
            if col == n_sel - 1:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

        fig.suptitle(
            f"step {d['step']}  |  frame {t+1}/{H_active}  "
            f"(AR window {w_idx + 1}/{d['n_ar_windows']})  |  "
            f"plan time {d['plan_time']:.2f}s  best={d['best_cost']:.4f}  "
            f"mean={d['mean_cost']:.4f}\n"
            f"point counts per window — {npts_str}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        # Convert figure → RGB array for the gif.
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frames_rgb.append(imageio.imread(buf))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_path), frames_rgb, fps=fps, loop=0)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan_dir", type=str,
                   help="Plan dir containing replans/replan_NNN.npz")
    p.add_argument("--fps", type=int, default=6,
                   help="GIF frame rate.")
    p.add_argument("--dpi", type=int, default=100,
                   help="Figure DPI per gif frame (lower = smaller gif).")
    args = p.parse_args()
    plan_dir = Path(args.plan_dir)
    replan_dir = plan_dir / "replans"
    if not replan_dir.is_dir():
        raise SystemExit(f"no replans/ under {plan_dir}")
    out_dir = plan_dir / "viz" / "pcds"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(replan_dir.glob("replan_*.npz"))
    n_ok, n_skip = 0, 0
    for f in files:
        ok = viz_one(f, out_dir / (f.stem + ".gif"),
                     fps=args.fps, dpi=args.dpi)
        if ok:
            n_ok += 1
        else:
            n_skip += 1

    print(f"  wrote {n_ok} pcd gifs under {out_dir}  ({n_skip} replan(s) "
          f"had no pcd dumps — re-run imagined_mppi.py with --dump_n_random_pcds 2)")

    # Also print a compact per-replan point-count table.
    print(f"\n[point counts per replan]")
    print(f"  {'replan':>6} {'step':>5} {'AR':>3}   N per window")
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            if "npts_per_window" not in d.files:
                continue
            npw = d["npts_per_window"]
            n_ar = int(d["n_ar_windows"])
            step = int(d["step"])
        print(f"  {f.stem.split('_')[1]:>6} {step:>5} {n_ar:>3}   "
              + " ".join(f"w{w}:{n}" for w, n in enumerate(npw)))


if __name__ == "__main__":
    main()
