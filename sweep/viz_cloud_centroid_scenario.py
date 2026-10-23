#!/usr/bin/env python3
"""Per-scenario point-cloud viz for cloud_centroid PointWorld eval.

For each completed scenario (sim_centroid.json present), reads
  positions/batch{NN}.npz  →  imagined trajectory + target/anchor markers
  replay/replay.pkl        →  Genesis-MPM sim trajectory
and renders side-by-side IMAGINED | SIM strips:
  - 5 key frames (f0, fT/4, fT/2, 3fT/4, fT-1)
  - blob in blue (model output / sim playback)
  - paddle in red
  - target xy as a magenta ★, anchor xy as a hollow black ○
  - title shows sim_cloud_shift, sim_centroid_dist, target magnitude

Output: <out_root>/<config>/<seed>/<ep>/{strip.png, top_view.png}

Top-view also drawn for clarity since the centroid task is xy-driven.

Usage:
    # Render every completed scenario across all 4 configs:
    python sweep/viz_cloud_centroid_scenario.py

    # Or one specific scenario:
    python sweep/viz_cloud_centroid_scenario.py \\
        --scenario /data/group_data/.../cloud_centroid_pw_seed0_T100/seed_0/ep_000
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SWEEP_ROOT = Path("/data/group_data/katefgroup/datasets/mbronars")
SWEEP_DIRS = [
    "cloud_centroid_pw_seed0_T100",
    "cloud_centroid_pw_seed0_T150",
    "cloud_centroid_pw_seed42_T100",
    "cloud_centroid_pw_seed42_T150",
]
OUT_ROOT = Path("/data/user_data/mbronars/packages/PointWorld/inference_results"
                "/cloud_centroid_pw_viz")


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario loader
# ─────────────────────────────────────────────────────────────────────────────

def load_scenario(scn: Path) -> Optional[dict]:
    """Return None if scenario isn't completed yet."""
    pos_files = sorted(scn.glob("positions/batch*.npz"))
    pkl       = scn / "replay" / "replay.pkl"
    score_js  = scn / "replay" / "sim_centroid.json"
    if not (pos_files and pkl.exists() and score_js.exists()):
        return None
    pos = np.load(pos_files[0], allow_pickle=True)
    with open(pkl, "rb") as f:
        rep = pickle.load(f)
    score = json.loads(score_js.read_text())

    oid = pos["object_ids"][0]                            # (N,)
    blob_mask   = (oid == 0)
    paddle_mask = (oid == 1)
    pred_full   = pos["predicted"][0, 0]                  # (T, N, 3)
    T_plan      = pred_full.shape[0]

    out = {
        "name":           scn.name,
        "T":              T_plan,
        "pred_blob":      pred_full[:, blob_mask],        # (T, N_blob, 3)
        "pred_paddle":    pred_full[:, paddle_mask],      # (T, N_paddle, 3)
        "sim_blob":       np.asarray(rep["sim_object_particles"]),    # (S, N_blob, 3)
        "sim_pusher_pos": np.asarray(rep["sim_pusher_pos"]),          # (S, 3)
        "sim_pusher_quat":np.asarray(rep["sim_pusher_quat"]),         # (S, 4)
        "anchor_xy":      np.asarray(pos["centroid_anchor_xy"]),
        "target_xy":      np.asarray(pos["centroid_target_xy"]),
        "sim_centroid_dist_m":   float(score["sim_centroid_dist"]),
        "sim_cloud_shift_m":     float(score["sim_cloud_shift"]),
        "imag_centroid_dist_m":  float(score["imagined_centroid_dist"]),
        "target_offset_xy":      np.asarray(score["target_offset_xy"]),
    }
    out["target_mag_cm"] = float(np.linalg.norm(out["target_offset_xy"])) * 100.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def _box_pts_from_pose(pos: np.ndarray, quat_wxyz: np.ndarray,
                        half: tuple = (0.015, 0.004, 0.020),
                        n: int = 64) -> np.ndarray:
    """Sample n points on the paddle box surface for a single pose. Quick &
    cheap — uniform random per-face."""
    hx, hy, hz = half
    canon = np.array(
        [(sx*hx, sy*hy, sz*hz)
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] +
        [(0, 0, hz), (0, 0, -hz), (0, hy, 0), (0, -hy, 0),
         (hx, 0, 0), (-hx, 0, 0)],
        dtype=np.float32,
    )
    # Rotate by quat (wxyz convention).
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ], dtype=np.float32)
    return (canon @ R.T + pos[None, :]).astype(np.float32)


def _draw_3d(ax, blob, paddle_pts, anchor_xy, target_xy, lims, title):
    ax.clear()
    ax.scatter(blob[:, 0], blob[:, 1], blob[:, 2],
               s=2.0, c="#1f77b4", alpha=0.6)
    ax.scatter(paddle_pts[:, 0], paddle_pts[:, 1], paddle_pts[:, 2],
               s=4.0, c="#d62728", alpha=0.85)
    # Anchor (hollow black circle) and target (magenta star) on the table plane.
    z_floor = lims[2][0]
    ax.scatter([anchor_xy[0]], [anchor_xy[1]], [z_floor],
               s=70, facecolors="none", edgecolors="black", linewidths=1.2,
               label="anchor")
    ax.scatter([target_xy[0]], [target_xy[1]], [z_floor],
               s=120, c="#cf2af0", marker="*", edgecolors="black",
               linewidths=0.4, label="target")
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect((lims[0][1] - lims[0][0],
                       lims[1][1] - lims[1][0],
                       lims[2][1] - lims[2][0]))
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=30, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def _draw_top(ax, blob, paddle_xy, anchor_xy, target_xy, lims_xy, title):
    ax.clear()
    ax.scatter(blob[:, 0], blob[:, 1], s=3.5, c="#1f77b4", alpha=0.6)
    ax.scatter(paddle_xy[:, 0], paddle_xy[:, 1], s=5.0, c="#d62728", alpha=0.85)
    cent = blob.mean(axis=0)
    ax.scatter([cent[0]], [cent[1]], s=80, marker="x", c="#1f77b4",
               linewidths=2.0, label="blob centroid")
    ax.scatter([anchor_xy[0]], [anchor_xy[1]],
               s=120, facecolors="none", edgecolors="black", linewidths=1.4,
               label="anchor")
    ax.scatter([target_xy[0]], [target_xy[1]],
               s=180, c="#cf2af0", marker="*",
               edgecolors="black", linewidths=0.5, label="target")
    ax.plot([anchor_xy[0], target_xy[0]], [anchor_xy[1], target_xy[1]],
            "--", color="#7e57c2", alpha=0.6, linewidth=1.0)
    ax.set_xlim(lims_xy[0]); ax.set_ylim(lims_xy[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


# ─────────────────────────────────────────────────────────────────────────────
# Render a single scenario
# ─────────────────────────────────────────────────────────────────────────────

def render_scenario(s: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    T_plan = s["T"]
    # SIM trajectory is at finer time resolution (frame_skip × n_planned_frames).
    # Sample SIM at the same number of frames as planned for an aligned strip.
    n_sim = s["sim_blob"].shape[0]
    sim_indices = np.linspace(0, n_sim - 1, T_plan).astype(int)

    # Key frames (5 columns).
    keys = sorted({0, T_plan // 4, T_plan // 2, 3 * T_plan // 4, T_plan - 1})

    # Shared 3D axis limits across both rows (use union of pred + sim + markers).
    all_xy = np.concatenate([
        s["pred_blob"].reshape(-1, 3)[:, :2],
        s["sim_blob"].reshape(-1, 3)[:, :2],
        s["anchor_xy"][None],
        s["target_xy"][None],
    ], axis=0)
    z_all = np.concatenate([
        s["pred_blob"].reshape(-1, 3)[:, 2],
        s["sim_blob"].reshape(-1, 3)[:, 2],
        s["pred_paddle"].reshape(-1, 3)[:, 2],
        s["sim_pusher_pos"][:, 2],
    ])
    pad = 0.02
    lims = (
        (float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad),
        (float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad),
        (float(z_all.min()) - 0.005,       float(z_all.max()) + 0.005),
    )

    sim_dist_cm   = s["sim_centroid_dist_m"] * 100
    sim_shift_cm  = s["sim_cloud_shift_m"]   * 100
    imag_dist_cm  = s["imag_centroid_dist_m"] * 100

    suptitle = (f"{s['name']}   "
                f"target |Δ|={s['target_mag_cm']:.2f} cm   "
                f"sim_cloud_shift={sim_shift_cm:.2f} cm   "
                f"sim_centroid_dist={sim_dist_cm:.2f} cm   "
                f"imag_centroid_dist={imag_dist_cm:.2f} cm")

    # ── 3D strip ────────────────────────────────────────────────────────────
    n_cols = len(keys)
    fig = plt.figure(figsize=(3.5 * n_cols, 7))
    fig.suptitle(suptitle, fontsize=10)
    for j, f in enumerate(keys):
        # IMAGINED row
        ax_top    = fig.add_subplot(2, n_cols, j + 1, projection="3d")
        # Paddle: we can sample from the pred_paddle particles directly.
        _draw_3d(ax_top,
                 s["pred_blob"][f],
                 s["pred_paddle"][f],
                 s["anchor_xy"], s["target_xy"], lims,
                 f"IMAGINED  f{f}")
        # SIM row
        ax_bot    = fig.add_subplot(2, n_cols, n_cols + j + 1, projection="3d")
        sim_i = sim_indices[f]
        sim_paddle_pts = _box_pts_from_pose(
            s["sim_pusher_pos"][sim_i], s["sim_pusher_quat"][sim_i])
        _draw_3d(ax_bot,
                 s["sim_blob"][sim_i],
                 sim_paddle_pts,
                 s["anchor_xy"], s["target_xy"], lims,
                 f"SIM  f{f}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "strip.png", dpi=110)
    plt.close(fig)

    # ── Top-down view (xy plane) — easier to read for a planning task ───────
    lims_xy = (lims[0], lims[1])
    fig = plt.figure(figsize=(3.5 * n_cols, 6))
    fig.suptitle(suptitle, fontsize=10)
    for j, f in enumerate(keys):
        ax_top = fig.add_subplot(2, n_cols, j + 1)
        _draw_top(ax_top,
                  s["pred_blob"][f],
                  s["pred_paddle"][f][:, :2],
                  s["anchor_xy"], s["target_xy"], lims_xy,
                  f"IMAGINED  f{f}")
        ax_bot = fig.add_subplot(2, n_cols, n_cols + j + 1)
        sim_i = sim_indices[f]
        sim_paddle_pts = _box_pts_from_pose(
            s["sim_pusher_pos"][sim_i], s["sim_pusher_quat"][sim_i])
        _draw_top(ax_bot,
                  s["sim_blob"][sim_i],
                  sim_paddle_pts[:, :2],
                  s["anchor_xy"], s["target_xy"], lims_xy,
                  f"SIM  f{f}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "top_view.png", dpi=110)
    plt.close(fig)

    # ── Per-scenario centroid trajectory (single panel) — most informative
    #    single image. Shows blob centroid path in xy, both pred and sim.
    pred_centroid_xy = s["pred_blob"][:, :, :2].mean(axis=1)   # (T, 2)
    sim_centroid_xy  = s["sim_blob"][:, :, :2].mean(axis=1)    # (S, 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pred_centroid_xy[:, 0], pred_centroid_xy[:, 1],
            "-", color="#4c72b0", alpha=0.9, label="imagined centroid")
    ax.plot(sim_centroid_xy[:, 0],  sim_centroid_xy[:, 1],
            "-", color="#55a868", alpha=0.9, label="sim centroid")
    ax.scatter([s["anchor_xy"][0]], [s["anchor_xy"][1]],
               s=120, facecolors="none", edgecolors="black", linewidths=1.4, label="anchor")
    ax.scatter([s["target_xy"][0]], [s["target_xy"][1]],
               s=200, c="#cf2af0", marker="*", edgecolors="black", linewidths=0.5,
               label="target")
    ax.scatter([pred_centroid_xy[-1, 0]], [pred_centroid_xy[-1, 1]],
               s=80, c="#4c72b0", marker="o", edgecolors="black", linewidths=0.5,
               label="imagined final")
    ax.scatter([sim_centroid_xy[-1, 0]],  [sim_centroid_xy[-1, 1]],
               s=80, c="#55a868", marker="o", edgecolors="black", linewidths=0.5,
               label="sim final")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(suptitle, fontsize=8)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "centroid_traj.png", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None,
                    help="Specific scenario dir (e.g. .../seed_0/ep_000)")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out_root)

    if args.scenario is not None:
        scn = Path(args.scenario)
        s = load_scenario(scn)
        if s is None:
            print(f"  scenario incomplete: {scn}")
            return
        # Try to keep <config>/<seed>/<ep> path.
        rel = scn.parts[-3:]   # e.g. ('cloud_centroid_pw_seed0_T100', 'seed_0', 'ep_000')
        out = out_root / Path(*rel)
        render_scenario(s, out)
        print(f"  rendered -> {out}/")
        return

    n_rendered = n_skipped = 0
    for sweep_dir in SWEEP_DIRS:
        for scn in sorted((SWEEP_ROOT / sweep_dir).glob("seed_*/ep_*")):
            s = load_scenario(scn)
            if s is None:
                n_skipped += 1
                continue
            rel = scn.parts[-3:]
            out = out_root / Path(*rel)
            # Skip if already rendered + scenario hasn't been re-run since.
            if (out / "strip.png").exists():
                continue
            render_scenario(s, out)
            n_rendered += 1
            print(f"  rendered {Path(*rel)}  (sim_shift={s['sim_cloud_shift_m']*100:.2f} cm)")
    print(f"\nrendered {n_rendered} new scenarios; skipped {n_skipped} incomplete.")


if __name__ == "__main__":
    main()
