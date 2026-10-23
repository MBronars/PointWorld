#!/usr/bin/env python3
"""
wm_viz_candidates.py

Render per-candidate point-cloud gifs from a single replan dump produced
by imagined_mppi.py with --dump_n_random_pcds N. One gif per selected
candidate (winner + the random picks).
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


PRED_FRAMES = 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("replan_npz", type=str,
                   help="Path to one replan_NNN.npz from a planner dump dir.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--dpi", type=int, default=100)
    args = p.parse_args()

    with np.load(args.replan_npz, allow_pickle=False) as d:
        if "selected_k_indices" not in d.files:
            raise SystemExit(
                f"{args.replan_npz} has no per-candidate pcd dumps. "
                f"Re-run imagined_mppi.py with --dump_n_random_pcds N."
            )
        sel_k       = d["selected_k_indices"]
        sel_costs   = d["selected_costs"]
        sel_grip    = d["selected_gripper_xy"]                          # (n_sel, H_active, 2)
        n_ar        = int(d["n_ar_windows"])
        H_active    = int(d["H_active"])
        block_now   = d["block_xy_at_replan"]
        gripper_now = d["gripper_xy_at_replan"]
        target_xy   = d["target_xy"]
        step        = int(d["step"])
        per_window  = [d[f"selected_pcds_w{w}"] for w in range(n_ar)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sel = int(sel_k.shape[0])
    print(f"replan @ step {step}: {n_sel} candidates, H_active={H_active}, "
          f"AR windows={n_ar}")

    for c in range(n_sel):
        rank_str = "winner" if c == 0 else f"rand{c:02d}"

        # Frame-by-frame block point clouds for this candidate.
        frames = []
        for w_arr in per_window:                                       # (n_sel, PRED_FRAMES, N_w, 3)
            for f in range(w_arr.shape[1]):
                frames.append(w_arr[c, f])

        # Shared axis limits across candidate's frames.
        all_xy = np.concatenate(
            [pts[:, :2] for pts in frames] + [sel_grip[c]] +
            [block_now[None], gripper_now[None], target_xy[None]],
            axis=0,
        )
        pad = 0.015
        xmin, ymin = all_xy.min(axis=0) - pad
        xmax, ymax = all_xy.max(axis=0) + pad

        gif_frames = []
        for t in range(H_active):
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            for tt in range(t):
                pts = frames[tt]
                ax.scatter(pts[:, 0], pts[:, 1], s=2, color="lightgray",
                           alpha=0.10, edgecolors="none")
            bpts = frames[t]
            ax.scatter(bpts[:, 0], bpts[:, 1], s=10, color="C0",
                       alpha=0.85, edgecolors="none", label="predicted block")
            # Gripper trajectory faded + current marker.
            g = sel_grip[c]
            full_g = np.concatenate([gripper_now[None], g], axis=0)
            ax.plot(full_g[:, 0], full_g[:, 1], "--", color="orange",
                    lw=1.0, alpha=0.4)
            ax.scatter(g[t, 0], g[t, 1], color="orange", marker="^",
                       s=110, edgecolor="black", zorder=6, label="gripper at t")
            ax.scatter(*block_now,   color="black", marker="s", s=110,
                       edgecolor="white", zorder=5, label="block t=0")
            ax.scatter(*gripper_now, color="dimgray", marker="^", s=70,
                       edgecolor="white", zorder=5, label="gripper t=0")
            ax.scatter(*target_xy,   color="red",   marker="*", s=280,
                       edgecolor="black", zorder=6, label="target")
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal"); ax.grid(alpha=0.3)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            w_idx = t // PRED_FRAMES
            ax.set_title(
                f"step {step}  {rank_str}  k={int(sel_k[c])}  "
                f"cost={sel_costs[c]:.4f}\n"
                f"frame {t+1}/{H_active}  (AR win {w_idx+1}/{n_ar})",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=args.dpi)
            plt.close(fig)
            buf.seek(0)
            gif_frames.append(imageio.imread(buf))

        cost_tag = f"{sel_costs[c]:.4f}".replace(".", "p").replace("-", "neg")
        save = out_dir / f"cand{c:02d}_{rank_str}_k{int(sel_k[c]):03d}_cost{cost_tag}.gif"
        imageio.mimsave(str(save), gif_frames, fps=args.fps, loop=0)
        print(f"  {save.name}")


if __name__ == "__main__":
    main()
