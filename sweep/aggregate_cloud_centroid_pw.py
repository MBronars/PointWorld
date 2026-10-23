#!/usr/bin/env python3
"""Aggregate + visualise the cloud_centroid PointWorld sweep results.

Walks the four sweep dirs
  /data/group_data/katefgroup/datasets/mbronars/cloud_centroid_pw_seed{0,42}_T{100,150}/
reads every ``seed_<S>/ep_<NN>/replay/sim_centroid.json`` it can find, and
emits:

  inference_results/cloud_centroid_pw_summary/
    summary.json        per-config aggregates + per-scenario rows
    summary.md          markdown table for quick paste
    hist_<config>.png   histogram of sim_centroid_dist per config
    compare.png         box + scatter across all 4 configs
    scatter_<config>.png  target magnitude vs sim_centroid_dist

Run any time — re-reads disk fresh, no caching. Safe to call mid-sweep.

Usage:
    python sweep/aggregate_cloud_centroid_pw.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/group_data/katefgroup/datasets/mbronars")
CONFIGS = [
    ("seed0_T100",  "cloud_centroid_pw_seed0_T100",  0,  100),
    ("seed0_T150",  "cloud_centroid_pw_seed0_T150",  0,  150),
    ("seed42_T100", "cloud_centroid_pw_seed42_T100", 42, 100),
    ("seed42_T150", "cloud_centroid_pw_seed42_T150", 42, 150),
]
OUT_DIR = Path("/data/user_data/mbronars/packages/PointWorld/inference_results"
               "/cloud_centroid_pw_summary")


def gather_one(dir_path: Path) -> list[dict]:
    """Read all sim_centroid.json under dir_path/seed_*/ep_*/replay/."""
    rows = []
    if not dir_path.is_dir():
        return rows
    for js in sorted(dir_path.glob("seed_*/ep_*/replay/sim_centroid.json")):
        try:
            d = json.loads(js.read_text())
            ep_name = js.parent.parent.name        # ep_NN
            seed_name = js.parent.parent.parent.name  # seed_S
            d["_episode"] = ep_name
            d["_seed_dir"] = seed_name
            d["_magnitude_cm"] = float(np.linalg.norm(d["target_offset_xy"]) * 100)
            d["_sim_dist_cm"]  = float(d["sim_centroid_dist"]) * 100
            d["_sim_shift_cm"] = float(d["sim_cloud_shift"])   * 100
            d["_imag_dist_cm"] = float(d["imagined_centroid_dist"]) * 100
            rows.append(d)
        except Exception as e:
            print(f"  skip {js}: {e}")
    return rows


def agg_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    sim = np.array([r["_sim_dist_cm"]  for r in rows])
    imag = np.array([r["_imag_dist_cm"] for r in rows])
    shift = np.array([r["_sim_shift_cm"] for r in rows])
    mag = np.array([r["_magnitude_cm"] for r in rows])
    return {
        "n":                       int(len(rows)),
        "sim_dist_cm_mean":        float(sim.mean()),
        "sim_dist_cm_median":      float(np.median(sim)),
        "sim_dist_cm_std":         float(sim.std()),
        "sim_dist_cm_min":         float(sim.min()),
        "sim_dist_cm_max":         float(sim.max()),
        "imag_dist_cm_mean":       float(imag.mean()),
        "imag_dist_cm_median":     float(np.median(imag)),
        "sim_shift_cm_mean":       float(shift.mean()),
        "sim_shift_cm_median":     float(np.median(shift)),
        "target_mag_cm_mean":      float(mag.mean()),
        # Headline-ish: how often does the planner *get close* to target?
        # Define success as sim_centroid_dist < 3 cm (arbitrary threshold).
        "frac_success_3cm":        float((sim < 3.0).mean()),
        "frac_success_5cm":        float((sim < 5.0).mean()),
    }


def write_summary(all_rows: dict[str, list[dict]], all_aggs: dict[str, dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "configs": [
            {
                "tag":          tag,
                "dir":          dir_name,
                "seed":         seed,
                "total_frames": tf,
                "aggregate":    all_aggs[tag],
                "scenarios":    all_rows[tag],
            }
            for tag, dir_name, seed, tf in CONFIGS
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2))

    # Markdown table.
    md = ["# Cloud-Centroid PointWorld eval — summary",
          "",
          f"({sum(len(all_rows[t]) for t,_,_,_ in CONFIGS)}/200 scenarios completed)",
          "",
          "**Headline**: `sim_cloud_shift` — cm the cloud *actually* moved in the "
          "Genesis-MPM playback of the planner's gripper trajectory.",
          "",
          "| config | n | **shift mean ± std (cm)** | shift median (cm) | shift p90 (cm) | target mag mean (cm) | sim_dist mean (cm) | imag_dist mean (cm) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, _, _, _ in CONFIGS:
        a = all_aggs[tag]
        rows = all_rows[tag]
        if a["n"] == 0:
            md.append(f"| `{tag}` | 0 | — | — | — | — | — | — |")
            continue
        shift = np.array([r["_sim_shift_cm"] for r in rows])
        shift_p90 = float(np.percentile(shift, 90))
        md.append(
            f"| `{tag}` | {a['n']} "
            f"| **{a['sim_shift_cm_mean']:.2f} ± {shift.std():.2f}** "
            f"| {a['sim_shift_cm_median']:.2f} "
            f"| {shift_p90:.2f} "
            f"| {a['target_mag_cm_mean']:.2f} "
            f"| {a['sim_dist_cm_mean']:.2f} "
            f"| {a['imag_dist_cm_mean']:.2f} |"
        )
    md.append("")
    md.append("`sim_dist` = final distance to commanded target (lower better; "
              "trivial 0 if planner stays put but cloud was already at target).")
    md.append("`imag_dist` = same but in the model's *imagined* world (model's "
              "self-belief; gap to sim_dist is the sim2real-ish error).")
    md.append("Target offsets are sampled uniformly in `[r_min=8 cm, r_max=14 cm]` "
              "per the scenario README.")
    (OUT_DIR / "summary.md").write_text("\n".join(md))


def make_plots(all_rows: dict[str, list[dict]]) -> None:
    # Per-config histogram of sim_cloud_shift (headline)
    for tag, _, _, _ in CONFIGS:
        rows = all_rows[tag]
        if not rows:
            continue
        shift = np.array([r["_sim_shift_cm"] for r in rows])
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(shift, bins=20, color="#55a868", alpha=0.85, edgecolor="white")
        ax.set_xlabel("sim_cloud_shift (cm) — cm actually moved")
        ax.set_ylabel("count")
        ax.set_title(f"{tag}  (n={len(rows)}, mean shift = {shift.mean():.2f} cm)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"hist_{tag}.png", dpi=110)
        plt.close(fig)

    # Per-config scatter: target magnitude vs sim_cloud_shift
    for tag, _, _, _ in CONFIGS:
        rows = all_rows[tag]
        if not rows:
            continue
        mag = np.array([r["_magnitude_cm"] for r in rows])
        shift = np.array([r["_sim_shift_cm"] for r in rows])
        sim_dist = np.array([r["_sim_dist_cm"] for r in rows])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sc = ax.scatter(mag, shift, c=sim_dist, cmap="viridis_r", s=25, alpha=0.85,
                        edgecolors="white", linewidths=0.4)
        ax.plot([mag.min(), mag.max()], [mag.min(), mag.max()],
                "--", color="grey", alpha=0.5, label="y=x (cloud reached target)")
        ax.set_xlabel("target |offset| (cm)")
        ax.set_ylabel("sim_cloud_shift (cm)  — cm actually moved")
        ax.set_title(f"{tag}  (n={len(rows)})")
        fig.colorbar(sc, ax=ax, label="sim_centroid_dist (cm)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"scatter_{tag}.png", dpi=110)
        plt.close(fig)

    # Cross-config comparison: box + strip on sim_cloud_shift
    tags = [t for t, _, _, _ in CONFIGS if all_rows[t]]
    if tags:
        data = [np.array([r["_sim_shift_cm"] for r in all_rows[t]]) for t in tags]
        fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(tags)), 4))
        bp = ax.boxplot(data, labels=tags, showmeans=True, widths=0.5,
                        patch_artist=True, meanline=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#c5d4e0"); patch.set_alpha(0.6)
        for i, d in enumerate(data, start=1):
            x = np.random.normal(i, 0.04, size=len(d))
            ax.scatter(x, d, s=12, color="#55a868", alpha=0.55,
                       edgecolors="white", linewidths=0.3)
        ax.set_ylabel("sim_cloud_shift (cm) — cm actually moved")
        ax.set_title("cloud_centroid PointWorld eval — cm moved per config")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "compare.png", dpi=110)
        plt.close(fig)


def main() -> None:
    all_rows: dict[str, list[dict]] = {}
    all_aggs: dict[str, dict] = {}
    for tag, dir_name, _seed, _tf in CONFIGS:
        rows = gather_one(ROOT / dir_name)
        all_rows[tag] = rows
        all_aggs[tag] = agg_stats(rows)
    write_summary(all_rows, all_aggs)
    make_plots(all_rows)

    # Console summary.
    total = sum(len(v) for v in all_rows.values())
    print(f"\n=== cloud_centroid PW eval summary  ({total}/200 scenarios) ===")
    for tag, _, _, _ in CONFIGS:
        a = all_aggs[tag]
        if a["n"] == 0:
            print(f"  {tag:14s}  0/50 scenarios")
            continue
        print(f"  {tag:14s}  {a['n']:>3d}/50    "
              f"sim_cloud_shift = {a['sim_shift_cm_mean']:.2f} cm  "
              f"(median {a['sim_shift_cm_median']:.2f}, "
              f"target mean {a['target_mag_cm_mean']:.2f} cm)    "
              f"sim_dist={a['sim_dist_cm_mean']:.2f}    "
              f"imag_dist={a['imag_dist_cm_mean']:.2f}")
    print(f"\noutputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
