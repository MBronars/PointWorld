#!/usr/bin/env python3
"""Render the *final-frame* point cloud (WM-predicted OR Genesis sim) as a
side-by-side top + side view in matplotlib. Unlike the Genesis renderer
this draws every individual particle so dispersed WM predictions show up
honestly (Genesis's MPM renderer extracts an isosurface from a density
grid, which collapses scattered outliers into a smooth ball).

Reads <scenario>/replay/replay.pkl and produces:
  <scenario>/final_render/final_pred_mpl.png
  <scenario>/final_render/final_sim_mpl.png

Usage:
  python render_final_frame_matplotlib.py --scenario <scn> --which both
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Visual constants — paddle drawn from initial geometry centered on the
# tracked pose, blob in blue.
BLOB_COLOR    = "#d99c5b"          # match Genesis MPM orange tint
PADDLE_COLOR  = "#c8581f"          # paddle red/orange
FLOOR_COLOR   = "#a8b3bc"
BG_COLOR      = "#0e1d2b"


def _paddle_pts(pos, quat_wxyz, half=(0.015, 0.004, 0.020)):
    hx, hy, hz = half
    canon = np.array(
        [(sx*hx, sy*hy, sz*hz) for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],
        dtype=np.float32,
    )
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ], dtype=np.float32)
    corners = canon @ R.T + pos[None, :]
    edges = [
        (0,1),(0,2),(1,3),(2,3),  # bottom face
        (4,5),(4,6),(5,7),(6,7),  # top face
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    return corners, edges


def _draw_3d(ax, blob, paddle_corners, paddle_edges, lims):
    ax.scatter(blob[:, 0], blob[:, 1], blob[:, 2],
               s=4, c=BLOB_COLOR, alpha=0.9, edgecolors="none",
               depthshade=False)
    for i, j in paddle_edges:
        ax.plot(*paddle_corners[[i, j]].T, c=PADDLE_COLOR, linewidth=1.5)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect((lims[0][1] - lims[0][0],
                       lims[1][1] - lims[1][0],
                       lims[2][1] - lims[2][0]))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_facecolor(BG_COLOR)


def render_one(blob, pusher_pos, pusher_quat, lims, out_path, title):
    paddle_corners, paddle_edges = _paddle_pts(pusher_pos, pusher_quat)
    fig = plt.figure(figsize=(10, 5), facecolor=BG_COLOR)
    # Top view (look straight down -z)
    ax_top = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_3d(ax_top, blob, paddle_corners, paddle_edges, lims)
    ax_top.view_init(elev=89, azim=-90)
    ax_top.set_title("TOP", color="white", fontsize=10)
    # Side view (look along +y)
    ax_side = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_3d(ax_side, blob, paddle_corners, paddle_edges, lims)
    ax_side.view_init(elev=10, azim=-90)
    ax_side.set_title("SIDE", color="white", fontsize=10)
    fig.suptitle(title, color="white", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--which", choices=("pred", "sim", "both"), default="both")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--xy-margin", type=float, default=0.04,
                    help="margin (m) around the point cloud envelope")
    args = ap.parse_args()

    scn = Path(args.scenario)
    out_dir = Path(args.out_dir) if args.out_dir else scn / "final_render"
    with open(scn / "replay" / "replay.pkl", "rb") as f:
        rep = pickle.load(f)

    pred_blob = rep["pred_object_particles"][-1]
    sim_blob  = rep["sim_object_particles"][-1]
    pred_pos  = rep["pred_pusher_pos"][-1];  pred_quat = rep["pred_pusher_quat"][-1]
    sim_pos   = rep["sim_pusher_pos"][-1];   sim_quat  = rep["sim_pusher_quat"][-1]

    # Shared limits: union of both clouds + paddle positions.
    everything = np.concatenate([
        pred_blob, sim_blob,
        pred_pos[None], sim_pos[None],
    ], axis=0)
    m = args.xy_margin
    lims = (
        (float(everything[:, 0].min()) - m, float(everything[:, 0].max()) + m),
        (float(everything[:, 1].min()) - m, float(everything[:, 1].max()) + m),
        (float(min(0.0, everything[:, 2].min())) - 0.005,
         float(everything[:, 2].max()) + 0.01),
    )

    if args.which in ("pred", "both"):
        render_one(pred_blob, pred_pos, pred_quat, lims,
                   out_dir / "final_pred_mpl.png",
                   f"WM-PREDICTED final frame  (n={pred_blob.shape[0]} particles)")
    if args.which in ("sim", "both"):
        render_one(sim_blob, sim_pos, sim_quat, lims,
                   out_dir / "final_sim_mpl.png",
                   f"GENESIS SIM final frame  (n={sim_blob.shape[0]} particles)")

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
