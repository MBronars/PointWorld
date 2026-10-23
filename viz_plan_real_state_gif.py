"""GIF version of the plan-on-real-state viz: renders per-step frames so the
imagined cube + rod plan (produced by imagined_mppi.py) can be seen unfolding
against the recorded real-world trajectory.

Usage:
    python viz_plan_real_state_gif.py \\
        --rollout   final_policy_rollout/rollout_0002_20260807_231623.npz \\
        --plan_dir  runs/real_state_trial02 \\
        --out       runs/real_state_trial02/plan_vs_real.gif \\
        [--fps 12]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio


BLOCK_HALF_M = 0.025   # 50 mm cube


def _draw_block(ax, center_xy, yaw_rad, color, alpha=0.9, lw=1.5, fill=True):
    ang_deg = np.degrees(yaw_rad)
    ll = (center_xy[0] - BLOCK_HALF_M, center_xy[1] - BLOCK_HALF_M)
    rect = mpatches.Rectangle(ll, 2 * BLOCK_HALF_M, 2 * BLOCK_HALF_M,
                              angle=ang_deg, rotation_point='center',
                              facecolor=color if fill else 'none',
                              edgecolor=color, alpha=alpha, linewidth=lw)
    ax.add_patch(rect)


def _frame(t_imag, imag_grip, imag_block, real_ee, real_cube, cube_yaw0,
           target_xy, lims, title):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_facecolor("#f7f7f8")

    # Full real trajectory as faint dashed context
    ax.plot(real_ee[:, 0],   real_ee[:, 1],   c="#f4a261", ls="--", lw=1.2,
            alpha=0.55, label="real rod path")
    ax.plot(real_cube[:, 0], real_cube[:, 1], c="#88c07f", ls="--", lw=1.2,
            alpha=0.55, label="real cube path")

    # Initial cube outline (translucent green box with yaw)
    _draw_block(ax, real_cube[0], cube_yaw0, "tab:green", alpha=0.20, lw=1.2)

    # Imagined trajectories up to current step
    ax.plot(imag_grip[:t_imag + 1, 0],  imag_grip[:t_imag + 1, 1],
            c="tab:orange", lw=2.4, alpha=0.95, label="imagined rod")
    ax.plot(imag_block[:t_imag + 1, 0], imag_block[:t_imag + 1, 1],
            c="tab:green",  lw=2.4, alpha=0.95, label="imagined cube")

    # Current positions (larger markers)
    ax.scatter([imag_grip[t_imag, 0]],  [imag_grip[t_imag, 1]],  marker="o",
               c="tab:orange", s=110, edgecolors="white", linewidths=0.9, zorder=7)
    ax.scatter([imag_block[t_imag, 0]], [imag_block[t_imag, 1]], marker="s",
               c="tab:green",  s=110, edgecolors="white", linewidths=0.9, zorder=7)

    # Target
    ax.scatter([target_xy[0]], [target_xy[1]], marker="*", c="red", s=380,
               edgecolors="black", linewidths=0.7, alpha=0.85, zorder=8)

    # Frame label
    ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#888", alpha=0.9))

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout",  required=True)
    ap.add_argument("--plan_dir", required=True)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--fps",      type=int, default=12)
    args = ap.parse_args()

    d = np.load(args.rollout, allow_pickle=True)
    real_ee   = np.asarray(d["ee_xy"])
    real_cube = np.asarray(d["cube_xy"])
    cube_yaw0 = float(d["cube_rot"][0])
    target_xy = np.asarray(d["plans"][0]["target"])

    plan = np.load(Path(args.plan_dir) / "actions.npz", allow_pickle=True)
    imag_grip  = np.asarray(plan["imagined_gripper_xy"])   # (T+1, 2)
    imag_block = np.asarray(plan["imagined_block_xy"])     # (T+1, 2)
    T = imag_grip.shape[0]

    imag_final = float(np.linalg.norm(imag_block[-1] - target_xy)) * 1000
    real_final = float(np.linalg.norm(real_cube[-1] - target_xy)) * 1000
    init_dist  = float(np.linalg.norm(real_cube[0]  - target_xy)) * 1000

    # Shared bounds
    stack = np.concatenate([
        real_ee, real_cube, imag_grip, imag_block, target_xy[None],
    ], axis=0)
    pad = 0.02
    lims = [
        (float(stack[:, 0].min()) - pad, float(stack[:, 0].max()) + pad),
        (float(stack[:, 1].min()) - pad, float(stack[:, 1].max()) + pad),
    ]

    frames = []
    for t in range(T):
        title = (
            f"step {t:3d}/{T-1}\n"
            f"init   d = {init_dist:5.1f} mm\n"
            f"imag   d = {float(np.linalg.norm(imag_block[t] - target_xy)) * 1000:5.1f} mm\n"
            f"final  d = {imag_final:5.1f} mm (real {real_final:.1f})"
        )
        frames.append(_frame(t, imag_grip, imag_block, real_ee, real_cube,
                             cube_yaw0, target_xy, lims, title))
    # Small hold on last frame
    frames.extend([frames[-1]] * max(1, args.fps // 2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=args.fps, loop=0)
    # Also transcode to mp4 for smaller file size + smoother playback
    mp4 = out.with_suffix(".mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
             "-movflags", "+faststart", "-pix_fmt", "yuv420p",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-crf", "18", str(mp4)],
            check=True)
        print(f"  -> {out} + {mp4}  ({T} frames @ {args.fps} fps)")
    except Exception as e:
        print(f"  -> {out}  ({T} frames @ {args.fps} fps; ffmpeg failed: {e})")


if __name__ == "__main__":
    main()
