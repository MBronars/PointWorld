"""Generate flat (horizontal) legend strips for the pile-manipulation and
block-pushing figures. Two small PNGs:

  pile_manip_legend.png     — cloud  + gripper + centroid + target
  block_pushing_legend.png  — block  + gripper + target

Colors / markers / sizes match the upstream renderers exactly so the legend
is consistent with the videos.

Usage:
    python make_legend_image.py --out_dir <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _scatter_handle(color, marker="o", markersize=9,
                     markeredgecolor=None, markeredgewidth=0.0, alpha=1.0):
    return Line2D([0], [0], marker=marker, linestyle="None",
                  markerfacecolor=color, markeredgecolor=markeredgecolor or color,
                  markeredgewidth=markeredgewidth, markersize=markersize, alpha=alpha)


def save_pile_legend(out_path: Path) -> None:
    """Pile-manipulation: cloud + gripper + centroid + target, flat row."""
    handles = [
        _scatter_handle("tab:green",  markersize=8,  alpha=0.85),   # cloud
        _scatter_handle("tab:orange", markersize=9,  alpha=0.9),    # gripper
        _scatter_handle("black", markersize=9, markeredgecolor="white",
                        markeredgewidth=0.8),                         # centroid
        _scatter_handle("red", marker="*", markersize=18,
                        markeredgecolor="black", markeredgewidth=0.6, alpha=0.7),  # target
    ]
    labels = ["pile", "gripper", "centroid", "target"]
    fig, ax = plt.subplots(figsize=(7.2, 0.6))
    ax.axis("off")
    fig.legend(handles, labels,
               loc="center", ncol=len(labels), frameon=False, fontsize=12,
               handletextpad=0.4, columnspacing=1.6)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  -> {out_path}")


def save_block_legend(out_path: Path) -> None:
    """Block-pushing: block + gripper + target (NO centroid), flat row."""
    handles = [
        _scatter_handle("tab:green",  markersize=8,  alpha=0.85),
        _scatter_handle("tab:orange", markersize=9,  alpha=0.9),
        _scatter_handle("red", marker="*", markersize=18,
                        markeredgecolor="black", markeredgewidth=0.6, alpha=0.7),
    ]
    labels = ["block", "gripper", "target"]
    fig, ax = plt.subplots(figsize=(5.6, 0.6))
    ax.axis("off")
    fig.legend(handles, labels,
               loc="center", ncol=len(labels), frameon=False, fontsize=12,
               handletextpad=0.4, columnspacing=1.6)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_pile_legend(out / "pile_manip_legend.png")
    save_block_legend(out / "block_pushing_legend.png")


if __name__ == "__main__":
    main()
