"""Overlay the freshly-planned MPPI trajectory on the recorded real-world
trial so it's visually obvious the plan was produced from the same initial
state. Top-down xy view.

Layers:
    - rod (initial position, orange circle)
    - block (initial position, green filled square with yaw)
    - target (red star)
    - real gripper trajectory     (light-orange dashed line, from ee_xy)
    - real cube trajectory        (light-green dashed line, from cube_xy)
    - imagined gripper trajectory (orange solid, from actions.npz)
    - imagined block trajectory   (green solid, from actions.npz)

Usage:
    python viz_plan_real_state.py \\
        --rollout   final_policy_rollout/rollout_0002_20260807_231623.npz \\
        --plan_dir  /tmp/plan_real_trial02 \\
        --out       /tmp/plan_real_trial02_viz.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


BLOCK_HALF_M = 0.025   # 50 mm cube


def _draw_block(ax, center_xy, yaw_rad, color, alpha=0.9, label=None,
                fill=True, lw=1.5):
    ang_deg = np.degrees(yaw_rad)
    ll = (center_xy[0] - BLOCK_HALF_M, center_xy[1] - BLOCK_HALF_M)
    rect = mpatches.Rectangle(ll, 2 * BLOCK_HALF_M, 2 * BLOCK_HALF_M,
                              angle=ang_deg, rotation_point='center',
                              facecolor=color if fill else 'none',
                              edgecolor=color, alpha=alpha, linewidth=lw,
                              label=label)
    ax.add_patch(rect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout",  required=True)
    ap.add_argument("--plan_dir", required=True)
    ap.add_argument("--out",      required=True)
    args = ap.parse_args()

    # Real-world data
    d = np.load(args.rollout, allow_pickle=True)
    real_ee   = np.asarray(d["ee_xy"])         # (N, 2)
    real_cube = np.asarray(d["cube_xy"])       # (N, 2)
    cube_yaw0 = float(d["cube_rot"][0])
    ee0, cube0 = real_ee[0], real_cube[0]
    target_xy = np.asarray(d["plans"][0]["target"])

    # Plan we produced
    plan_dir = Path(args.plan_dir)
    plan = np.load(plan_dir / "actions.npz", allow_pickle=True)
    print("plan keys:", list(plan.keys()))
    imag_grip = np.asarray(plan["imagined_gripper_xy"])   # (T+1, 2)
    imag_block = np.asarray(plan["imagined_block_xy"])    # (T+1, 2)

    # Bounds
    stack = np.concatenate([
        real_ee, real_cube, imag_grip, imag_block, target_xy[None],
    ], axis=0)
    pad = 0.02
    xlim = (float(stack[:, 0].min()) - pad, float(stack[:, 0].max()) + pad)
    ylim = (float(stack[:, 1].min()) - pad, float(stack[:, 1].max()) + pad)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f7f7f8")

    # Real trajectories (dashed, translucent)
    ax.plot(real_ee[:, 0],   real_ee[:, 1],   c="#f4a261", ls="--", lw=1.4,
            alpha=0.75, label="real rod path")
    ax.plot(real_cube[:, 0], real_cube[:, 1], c="#88c07f", ls="--", lw=1.4,
            alpha=0.75, label="real cube path")

    # Imagined trajectories (solid, saturated)
    ax.plot(imag_grip[:, 0],  imag_grip[:, 1],  c="tab:orange", lw=2.2,
            alpha=0.95, label="imagined rod plan")
    ax.plot(imag_block[:, 0], imag_block[:, 1], c="tab:green", lw=2.2,
            alpha=0.95, label="imagined cube plan")

    # Initial block outline + centroid
    _draw_block(ax, cube0, cube_yaw0, "tab:green", alpha=0.35,
                label="cube (start)", fill=True, lw=1.5)
    ax.scatter([cube0[0]], [cube0[1]], marker="o", c="tab:green", s=40,
               edgecolors="white", linewidths=0.8, zorder=6)

    # Rod initial position
    ax.scatter([ee0[0]], [ee0[1]], marker="o", c="tab:orange", s=90,
               edgecolors="white", linewidths=0.8, zorder=6, label="rod (start)")

    # Target
    ax.scatter([target_xy[0]], [target_xy[1]], marker="*", c="red", s=380,
               edgecolors="black", linewidths=0.7, alpha=0.85, zorder=7,
               label="target")

    # Final positions
    ax.scatter([imag_block[-1, 0]], [imag_block[-1, 1]], marker="s",
               c="tab:green", s=90, edgecolors="black", linewidths=0.7,
               zorder=6, label="imagined cube (end)")
    ax.scatter([real_cube[-1, 0]], [real_cube[-1, 1]], marker="s",
               c="#88c07f", s=90, edgecolors="black", linewidths=0.7,
               alpha=0.65, zorder=6, label="real cube (end)")

    # Numbers on the plot
    imag_final = float(np.linalg.norm(imag_block[-1] - target_xy)) * 1000
    real_final = float(np.linalg.norm(real_cube[-1] - target_xy)) * 1000
    init_dist  = float(np.linalg.norm(cube0 - target_xy)) * 1000
    ax.text(
        0.02, 0.98,
        f"init cube→target : {init_dist:5.1f} mm\n"
        f"imagined final   : {imag_final:5.1f} mm\n"
        f"real final       : {real_final:5.1f} mm\n"
        f"planner          : imagined_mppi.py\n"
        f"checkpoint       : pushing_no_normals/model-last.pt\n"
        f"K=128 pop, T=60 steps, replan every 5",
        transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="#888", alpha=0.9),
    )

    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m, panda base frame)")
    ax.set_ylabel("y (m, panda base frame)")
    ax.set_title("MPPI-in-WM plan on real-world initial state (trial 2)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (init={init_dist:.1f} mm, imagined final={imag_final:.1f} mm, real final={real_final:.1f} mm)")


if __name__ == "__main__":
    main()
