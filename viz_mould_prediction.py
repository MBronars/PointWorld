#!/usr/bin/env python3
"""
viz_mould_prediction.py — Render the WM-predicted moulding trajectory
saved by mppi_mould.py as a GIF.

Reads <plan_dir>/positions/batch{NNN}.npz (predicted + obj_ids + targets)
and <plan_dir>/field/batch{NNN}.npz (target letter coverage + contour),
emits side-by-side top-down + 3D scatter per frame, blob in blue and
paddle in orange, with the target letter contour overlaid.

Usage:
    python viz_mould_prediction.py \\
        /data/user_data/mbronars/packages/PointWorld/inference_results/mould_calibrated_demo0
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


def render_gif(npz_path: Path, field_path: Path | None,
               save_path: Path, fps: int = 12, dpi: int = 90,
               skip: int = 1) -> None:
    with np.load(npz_path, allow_pickle=False) as d:
        pred = d["predicted"][0, 0]                            # (T, N, 3)
        obj  = d["object_ids"][0]                              # (N,)
        targets = d["targets"][0]                              # (T, N, 3) NaN except final

    contour = None
    coverage = None
    if field_path is not None and field_path.exists():
        with np.load(field_path, allow_pickle=False) as f:
            contour  = f["contour"][0]                         # (P, 3)
            coverage = f["coverage"][0]                        # (n_cov, 2)

    blob_idx   = np.where(obj == 0)[0]
    paddle_idx = np.where(obj != 0)[0]
    T = pred.shape[0]

    # Camera limits — union of all predicted points + target.
    pts_all = pred.reshape(-1, 3)
    if contour is not None:
        pts_all = np.concatenate([pts_all, contour[~np.isnan(contour).any(-1)]], axis=0)
    pad = 0.03 * (pts_all.max(0) - pts_all.min(0)).max()
    xmin, ymin, zmin = pts_all.min(0) - pad
    xmax, ymax, zmax = pts_all.max(0) + pad

    frames = []
    for t in range(0, T, skip):
        fig = plt.figure(figsize=(11, 5))
        # Top-down
        ax1 = fig.add_subplot(1, 2, 1)
        if coverage is not None:
            ax1.scatter(coverage[:, 0], coverage[:, 1], s=2.0, color="gold",
                        alpha=0.30, label="target coverage", edgecolors="none")
        ax1.scatter(pred[t, blob_idx, 0], pred[t, blob_idx, 1], s=2.5,
                    color="C0", alpha=0.65, label="blob (predicted)")
        ax1.scatter(pred[t, paddle_idx, 0], pred[t, paddle_idx, 1], s=4,
                    color="C1", alpha=0.85, label="paddle")
        ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)
        ax1.set_aspect("equal"); ax1.grid(alpha=0.3)
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
        ax1.set_title(f"top-down  frame {t}/{T-1}")
        if t == 0:
            ax1.legend(loc="upper right", fontsize=7, framealpha=0.85)

        # 3D
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(pred[t, blob_idx, 0],   pred[t, blob_idx, 1],   pred[t, blob_idx, 2],
                    s=2.0, c="C0", alpha=0.55, depthshade=False)
        ax2.scatter(pred[t, paddle_idx, 0], pred[t, paddle_idx, 1], pred[t, paddle_idx, 2],
                    s=3.0, c="C1", alpha=0.85, depthshade=False)
        if contour is not None:
            valid = ~np.isnan(contour).any(-1)
            ax2.plot(contour[valid, 0], contour[valid, 1], contour[valid, 2],
                     "-", color="darkgoldenrod", lw=1.0, alpha=0.85)
        ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax); ax2.set_zlim(zmin, zmax)
        ax2.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        ax2.view_init(elev=25, azim=-60)
        ax2.set_title(f"3D  frame {t}/{T-1}")
        ax2.tick_params(labelsize=6)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_path), frames, fps=fps, loop=0)
    print(f"saved: {save_path}  ({len(frames)} frames @ {fps} fps)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan_dir", type=str,
                   help="MPPI output dir (contains positions/batch{NNN}.npz).")
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--skip", type=int, default=1,
                   help="Render every Nth frame (1 = every frame).")
    args = p.parse_args()

    plan_dir = Path(args.plan_dir)
    npz   = plan_dir / "positions" / f"batch{args.batch_idx:03d}.npz"
    field = plan_dir / "field"     / f"batch{args.batch_idx:03d}.npz"
    out   = plan_dir / "viz"       / f"prediction_batch{args.batch_idx:03d}.gif"
    render_gif(npz, field, out, fps=args.fps, skip=args.skip)


if __name__ == "__main__":
    main()
