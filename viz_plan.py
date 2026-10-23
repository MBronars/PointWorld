#!/usr/bin/env python3
"""
viz_plan.py

Visualise the outputs of imagined_mppi.py:
  <plan_dir>/actions.npz       — episode-level imagined trajectory + actions
  <plan_dir>/plan_log.json     — per-replan timing + cost
  <plan_dir>/replans/replan_NNN.npz  (when run with --dump_replans)

Emits:
  <plan_dir>/viz/trajectory.png         top-down: imagined block + gripper
  <plan_dir>/viz/cost_and_time.png      per-replan cost + plan-time curves
  <plan_dir>/viz/replans/replan_NNN.png per-replan K-candidate predictions

Usage:
    python viz_plan.py /tmp/imagined_mppi_run/seed0_ep000/
    python viz_plan.py /tmp/imagined_mppi_run/seed0_ep000/ --top_n 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def viz_trajectory(plan_dir: Path, save_path: Path) -> None:
    with np.load(plan_dir / "actions.npz", allow_pickle=False) as d:
        block_xy   = d["imagined_block_xy"]      # (T+1, 2)
        gripper_xy = d["imagined_gripper_xy"]    # (T+1, 2)
        target_xy  = d["target_xy"]
        replan_steps = d["replan_steps"]
        init_block_pos = d["init_block_pos"]
        n_replans  = int(d["n_replans"])
        final_dist = float(d["final_dist"])
        success    = bool(int(d["success"]))
        wall_time  = float(d["wall_time_sec"])
        action_max_norm = float(d["action_max_norm"])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(block_xy[:, 0],   block_xy[:, 1],   "-", color="C0", lw=2.5, label="block centroid")
    ax.plot(gripper_xy[:, 0], gripper_xy[:, 1], "-", color="C1", lw=1.2, alpha=0.85, label="gripper")
    ax.scatter(*init_block_pos[:2], color="green", marker="o", s=110, edgecolor="black",
               zorder=5, label="start block")
    ax.scatter(block_xy[-1, 0], block_xy[-1, 1], color="black", marker="s", s=110,
               edgecolor="white", zorder=5, label="end block")
    ax.scatter(*target_xy, color="red", marker="*", s=320, edgecolor="black",
               zorder=6, label="target")
    for s in replan_steps:
        if s < len(block_xy):
            ax.plot(block_xy[s, 0], block_xy[s, 1], "x", ms=9, color="orange")
    # boundary line for the threshold around the target.
    circ = plt.Circle(target_xy, 0.015, color="red", fill=False, ls=":",
                       alpha=0.5, label="threshold (1.5 cm)")
    ax.add_artist(circ)

    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"{plan_dir.name}  |  imagined dist {final_dist*100:.2f} cm  "
        f"{'✓' if success else '✗'}\n"
        f"{n_replans} replans, wall {wall_time:.1f}s, action L2 cap {action_max_norm:.3f} m"
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {save_path}")


def viz_cost_and_time(plan_dir: Path, save_path: Path) -> None:
    with np.load(plan_dir / "actions.npz", allow_pickle=False) as d:
        replan_steps = d["replan_steps"]
        best_costs   = d["best_costs"]
        mean_costs   = d["mean_costs"]
        plan_times   = d["plan_times"]
        H_active     = d["H_active"]
        n_ar_windows = d["n_ar_windows"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(replan_steps, best_costs, "-o", label="best cost",  color="C0")
    ax.plot(replan_steps, mean_costs, "-s", label="mean cost",  color="C1", alpha=0.7)
    ax.axhline(0.015, color="red", ls=":", alpha=0.5, label="threshold (terminal=0.015)")
    ax.set_xlabel("imagined step")
    ax.set_ylabel("cost (terminal block→target dist, m)")
    ax.set_title("MPPI cost per replan")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(np.arange(len(plan_times)), plan_times, color="C2", alpha=0.8)
    ax.set_xlabel("replan index")
    ax.set_ylabel("plan time (s)")
    ax.set_title(f"Wall time per replan  (mean={plan_times.mean():.2f}s, total={plan_times.sum():.1f}s)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(np.arange(len(H_active)), H_active, color="C3", alpha=0.8,
            label="H_active (frames)")
    ax.bar(np.arange(len(n_ar_windows)), n_ar_windows, color="C4", alpha=0.5,
            label="# AR windows")
    ax.set_xlabel("replan index")
    ax.set_ylabel("count")
    ax.set_title("Horizon size per replan")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {save_path}")


def viz_one_replan(replan_npz: Path, save_path: Path, top_n: int = 5) -> None:
    with np.load(replan_npz, allow_pickle=False) as d:
        cand_block_xy   = d["candidate_centroids_xy"]   # (K, H, 2)
        cand_gripper_xy = d["candidate_gripper_xy"]     # (K, H, 2)
        chosen_xy       = d["chosen_centroid_xy"]       # (H, 2)
        chosen_actions  = d["chosen_actions"]           # (H, 2)
        costs           = d["costs"]                    # (K,)
        weights         = d["weights"]                  # (K,)
        block_now       = d["block_xy_at_replan"]
        gripper_now     = d["gripper_xy_at_replan"]
        target_xy       = d["target_xy"]
        step            = int(d["step"])
        H_active        = int(d["H_active"])
        n_ar_windows    = int(d["n_ar_windows"])
        plan_time       = float(d["plan_time_sec"])

    K = costs.shape[0]
    n = min(top_n, K)
    top_idx  = np.argsort(costs)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Left: predicted block + gripper trajectories.
    ax = axes[0]
    # All K candidates faintly.
    for k in range(K):
        ax.plot(cand_block_xy[k, :, 0],   cand_block_xy[k, :, 1],
                "-", color="gray", alpha=0.10, lw=0.6)
        ax.plot(cand_gripper_xy[k, :, 0], cand_gripper_xy[k, :, 1],
                "--", color="gray", alpha=0.05, lw=0.5)

    cmap = plt.get_cmap("viridis")
    for rank, k in enumerate(top_idx):
        color = cmap(rank / max(n - 1, 1))
        ax.plot(cand_block_xy[k, :, 0], cand_block_xy[k, :, 1],
                "-", color=color, lw=2.0,
                label=f"#{rank+1}  cost={costs[k]:.4f}  w={weights[k]:.3f}")
        ax.scatter(cand_block_xy[k, -1, 0], cand_block_xy[k, -1, 1],
                   color=color, marker="o", s=55, edgecolor="black", zorder=4)
        ax.plot(cand_gripper_xy[k, :, 0], cand_gripper_xy[k, :, 1],
                "--", color=color, lw=1.0, alpha=0.5)

    # Chosen (MPPI-weighted) trajectory.
    ax.plot(chosen_xy[:, 0], chosen_xy[:, 1], "-", color="C3", lw=3.0,
            label="chosen (MPPI weighted)")
    ax.scatter(chosen_xy[-1, 0], chosen_xy[-1, 1], color="C3", marker="D",
               s=110, edgecolor="black", zorder=5)

    # Anchors.
    ax.scatter(*block_now, color="black", marker="s", s=140,
               edgecolor="white", zorder=6, label="block now")
    ax.scatter(*gripper_now, color="black", marker="^", s=110,
               edgecolor="white", zorder=6, label="gripper now")
    ax.scatter(*target_xy, color="red", marker="*", s=300,
               edgecolor="black", zorder=6, label="target")

    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"Replan @ step {step}  |  K={K} candidates, top {n} highlighted\n"
                 f"H_active={H_active}, AR windows={n_ar_windows}, plan time {plan_time:.2f}s")
    ax.legend(loc="best", fontsize=7, framealpha=0.85)

    # Right: cost / weight distribution.
    ax = axes[1]
    order = np.argsort(costs)
    ax.bar(np.arange(K), costs[order], color="C0", alpha=0.65, label="cost (sorted)")
    ax2 = ax.twinx()
    ax2.plot(np.arange(K), weights[order], "-r", lw=1.5, label="weight (sorted)")
    ax.set_xlabel("candidate (sorted by cost)")
    ax.set_ylabel("cost")
    ax2.set_ylabel("weight", color="red")
    ax.set_title("Candidate cost / weight distribution")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def viz_replans(plan_dir: Path, top_n: int) -> None:
    replan_dir = plan_dir / "replans"
    if not replan_dir.is_dir():
        print(f"  no replans/ dir under {plan_dir}; skip per-replan viz "
              f"(re-run imagined_mppi.py with --dump_replans).")
        return
    out_dir = plan_dir / "viz" / "replans"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(replan_dir.glob("replan_*.npz"))
    for f in files:
        out_path = out_dir / (f.stem + ".png")
        viz_one_replan(f, out_path, top_n=top_n)
    print(f"  wrote {len(files)} per-replan plots under {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("plan_dir", type=str)
    p.add_argument("--top_n", type=int, default=5,
                   help="Number of top-cost candidates to highlight per replan.")
    return p.parse_args()


def main():
    args = parse_args()
    plan_dir = Path(args.plan_dir)
    out_dir = plan_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quick textual summary.
    with np.load(plan_dir / "actions.npz", allow_pickle=False) as d:
        n_steps   = int(d["n_steps"])
        n_replans = int(d["n_replans"])
        final     = float(d["final_dist"])
        success   = bool(int(d["success"]))
        wall      = float(d["wall_time_sec"])
        plan_times = d["plan_times"]
    print(f"\n[{plan_dir.name}]")
    print(f"  n_steps={n_steps}  n_replans={n_replans}  final_dist={final*100:.2f} cm  "
          f"{'✓' if success else '✗'}")
    print(f"  wall {wall:.1f}s   per-replan plan time: "
          f"mean={plan_times.mean():.3f}s  std={plan_times.std():.3f}s  "
          f"sum={plan_times.sum():.1f}s")

    viz_trajectory(plan_dir,  out_dir / "trajectory.png")
    viz_cost_and_time(plan_dir, out_dir / "cost_and_time.png")
    viz_replans(plan_dir, top_n=args.top_n)
    print(f"  viz dir: {out_dir}")


if __name__ == "__main__":
    main()
