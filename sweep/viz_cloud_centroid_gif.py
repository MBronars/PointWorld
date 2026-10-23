#!/usr/bin/env python3
"""Render a continuous GIF of the world-model predicted point cloud (and
optionally the sim-side cloud) for a single cloud_centroid scenario.

Default view is top-down so you can see where every point lands frame by
frame. Pass --view 3d for a 3D angle. Pass --which sim to render the actual
sim playback instead of (or alongside) the imagined cloud.

Examples:
  python viz_cloud_centroid_gif.py --scenario .../seed_0/ep_008
  python viz_cloud_centroid_gif.py --scenario .../seed_0/ep_008 --which both
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as mpla
import matplotlib.pyplot as plt
import numpy as np


def load_scenario(scn: Path):
    pos_files = sorted(scn.glob("positions/batch*.npz"))
    if not pos_files:
        raise SystemExit(f"no positions/batch*.npz under {scn}")
    pos = np.load(pos_files[0])
    pkl = scn / "replay" / "replay.pkl"
    if not pkl.exists():
        raise SystemExit(f"no replay.pkl under {scn}")
    with open(pkl, "rb") as f:
        rep = pickle.load(f)

    pred = pos["predicted"][0, 0] if pos["predicted"].ndim == 5 else pos["predicted"][0]
    object_ids = pos["object_ids"][0]
    blob_pred   = pred[:, object_ids == 0, :]      # (T, N_blob, 3)
    paddle_pred = pred[:, object_ids == 1, :]      # (T, N_paddle, 3)

    sim_blob   = np.asarray(rep["sim_object_particles"])      # (S, N_blob, 3)
    sim_pusher_pos  = np.asarray(rep["sim_pusher_pos"])       # (S, 3)
    sim_pusher_quat = np.asarray(rep["sim_pusher_quat"])      # (S, 4)

    anchor_xy = np.asarray(pos["centroid_anchor_xy"])
    target_xy = np.asarray(pos["centroid_target_xy"])
    return {
        "blob_pred":   blob_pred,
        "paddle_pred": paddle_pred,
        "sim_blob":    sim_blob,
        "sim_pusher_pos":  sim_pusher_pos,
        "sim_pusher_quat": sim_pusher_quat,
        "anchor_xy":   anchor_xy,
        "target_xy":   target_xy,
    }


def _paddle_pts(pos, quat_wxyz, half=(0.015, 0.004, 0.020)):
    hx, hy, hz = half
    canon = np.array(
        [(sx*hx, sy*hy, sz*hz)
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] +
        [(0, 0, hz), (0, 0, -hz), (0, hy, 0), (0, -hy, 0),
         (hx, 0, 0), (-hx, 0, 0)],
        dtype=np.float32,
    )
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ], dtype=np.float32)
    return (canon @ R.T + pos[None, :]).astype(np.float32)


def render_gif(scn_dir: Path, out_path: Path, which: str = "imag",
               view: str = "top", fps: int = 20):
    s = load_scenario(scn_dir)

    if which == "sim":
        rows = [("SIM", "sim")]
    elif which == "imag":
        rows = [("IMAGINED (world model)", "imag")]
    else:  # both
        rows = [("IMAGINED (world model)", "imag"), ("SIM (Genesis playback)", "sim")]

    # Frames axis (use the SIM count as the master timeline, but sample IMAG
    # uniformly across its length so both rows line up visually).
    T_imag = s["blob_pred"].shape[0]
    T_sim  = s["sim_blob"].shape[0]
    # Master timeline: pick the cheaper one when only one row is shown, so
    # we don't accidentally upsample the imag side to 2500 sim steps.
    if which == "sim":
        n_frames = T_sim
    elif which == "imag":
        n_frames = T_imag
    else:
        # For "both", cap at the imagined timeline (T_imag ~100-150) — sim
        # has thousands of sub-steps that would make the GIF huge with no
        # visible benefit since the WM rollout is the slower-paced one.
        n_frames = T_imag
    imag_idx = np.linspace(0, T_imag - 1, n_frames).round().astype(int)
    sim_idx  = np.linspace(0, T_sim - 1, n_frames).round().astype(int)

    # Shared limits: union of imag + sim point clouds + markers.
    all_xy = np.concatenate([
        s["blob_pred"].reshape(-1, 3)[:, :2],
        s["paddle_pred"].reshape(-1, 3)[:, :2],
        s["sim_blob"].reshape(-1, 3)[:, :2],
        s["anchor_xy"][None],
        s["target_xy"][None],
    ], axis=0)
    pad = 0.02
    xlim = (float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad)
    ylim = (float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad)
    z_all = np.concatenate([
        s["blob_pred"].reshape(-1, 3)[:, 2],
        s["paddle_pred"].reshape(-1, 3)[:, 2],
        s["sim_blob"].reshape(-1, 3)[:, 2],
        s["sim_pusher_pos"][:, 2],
    ])
    zlim = (float(z_all.min()) - 0.005, float(z_all.max()) + 0.005)

    fig_h = 5 if len(rows) == 1 else 9
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(6, fig_h),
        subplot_kw={"projection": "3d"} if view == "3d" else None,
    )
    if len(rows) == 1:
        axes = [axes]
    suptitle = fig.suptitle("", fontsize=10)

    def draw_frame(i):
        for ax, (label, kind) in zip(axes, rows):
            ax.clear()
            if kind == "imag":
                blob = s["blob_pred"][imag_idx[i]]
                paddle = s["paddle_pred"][imag_idx[i]]
                tag = f"frame {imag_idx[i]+1}/{T_imag}"
            else:
                blob = s["sim_blob"][sim_idx[i]]
                paddle = _paddle_pts(
                    s["sim_pusher_pos"][sim_idx[i]],
                    s["sim_pusher_quat"][sim_idx[i]],
                )
                tag = f"frame {sim_idx[i]+1}/{T_sim}"

            if view == "3d":
                ax.scatter(blob[:, 0], blob[:, 1], blob[:, 2],
                           s=3, c="#1f77b4", alpha=0.7, depthshade=False)
                ax.scatter(paddle[:, 0], paddle[:, 1], paddle[:, 2],
                           s=8, c="#d62728", alpha=0.95, depthshade=False)
                z_floor = zlim[0]
                ax.scatter([s["anchor_xy"][0]], [s["anchor_xy"][1]], [z_floor],
                           s=80, facecolors="none", edgecolors="black",
                           linewidths=1.4)
                ax.scatter([s["target_xy"][0]], [s["target_xy"][1]], [z_floor],
                           s=140, c="#cf2af0", marker="*",
                           edgecolors="black", linewidths=0.5)
                ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
                ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0],
                                   zlim[1]-zlim[0]))
                ax.view_init(elev=30, azim=-60)
                ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            else:
                ax.scatter(blob[:, 0], blob[:, 1],
                           s=5, c="#1f77b4", alpha=0.7)
                ax.scatter(paddle[:, 0], paddle[:, 1],
                           s=10, c="#d62728", alpha=0.95)
                cent = blob.mean(axis=0)
                ax.scatter([cent[0]], [cent[1]], s=80, marker="x",
                           c="#1f77b4", linewidths=2.0)
                ax.scatter([s["anchor_xy"][0]], [s["anchor_xy"][1]],
                           s=120, facecolors="none", edgecolors="black",
                           linewidths=1.4)
                ax.scatter([s["target_xy"][0]], [s["target_xy"][1]],
                           s=180, c="#cf2af0", marker="*",
                           edgecolors="black", linewidths=0.5)
                ax.plot([s["anchor_xy"][0], s["target_xy"][0]],
                        [s["anchor_xy"][1], s["target_xy"][1]],
                        "--", color="#7e57c2", alpha=0.55, linewidth=1.0)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{label}  ({tag})", fontsize=9)

        suptitle.set_text(f"{scn_dir.parts[-3]} / {scn_dir.parts[-2]} / {scn_dir.parts[-1]}")
        return []

    anim = mpla.FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 // fps,
                              blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=mpla.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved {out_path}  ({n_frames} frames @ {fps} fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    help="Path to .../<config>/seed_<S>/ep_<NN>")
    ap.add_argument("--which", choices=["imag", "sim", "both"], default="imag")
    ap.add_argument("--view", choices=["top", "3d"], default="top")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="Output gif path (default: <scenario>/cloud.gif)")
    args = ap.parse_args()
    scn = Path(args.scenario)
    out = Path(args.out) if args.out else scn / f"cloud_{args.which}_{args.view}.gif"
    render_gif(scn, out, which=args.which, view=args.view, fps=args.fps)


if __name__ == "__main__":
    main()
