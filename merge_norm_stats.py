#!/usr/bin/env python3
"""
merge_norm_stats.py — One PointWorld norm_stats.json for the three-domain dump.

TRAINING.md asks these three domains to share ONE ruler. A PWM normalization
artifact expresses that by giving yikai / pickup / rl_push identical entries;
a PointWorld stats file is keyed by domain in exactly the same way, so the
translation is: pool the three domains' Welford accumulators, then write the
pooled statistics under all three domain keys.

Input   <out>/<pw_domain>/stats_raw.json, written by pwm_three_to_pointworld_wds.py
        for the train split of each domain.
Output  <out>/norm_stats/norm_stats.json — the folder --norm_stats_path points at.

    python merge_norm_stats.py --out /data/group_data/katefgroup/VLA/3DPCWM/three_pointworld

--per-domain writes each domain's own statistics instead (the 2.4x-imbalance
alternative TRAINING.md prices out). The shared ruler is the default because it
is what the PWM run this baseline is compared against uses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pwm_three_to_pointworld_wds import CLIP_LEN, DOMAIN_NAMES, WelfordAccumulator


def load_raw(out: Path, pw_domain: str) -> dict:
    path = out / pw_domain / "stats_raw.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — dump that domain's train split first")
    return json.loads(path.read_text())


def accumulators(payload: dict):
    scene = WelfordAccumulator.from_json(payload["scene"])
    robot = WelfordAccumulator.from_json(payload["robot"])
    gt_rel = [WelfordAccumulator.from_json(p) for p in payload["gt_scene_flows_relative"]]
    return scene, robot, gt_rel


def entry(scene, robot, gt_rel) -> tuple[dict, dict]:
    stats = {
        "robot_features": {"mean": robot.mean.tolist(), "variance": robot.variance.tolist()},
        "scene_features": {"mean": scene.mean.tolist(), "variance": scene.variance.tolist()},
    }
    per_ts = {f"timestep_{t}": {"mean": gt_rel[t].mean.tolist(),
                                "variance": gt_rel[t].variance.tolist()}
              for t in range(CLIP_LEN)}
    return stats, {"gt_scene_flows_relative": per_ts}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", default="/data/group_data/katefgroup/VLA/3DPCWM/three_pointworld")
    p.add_argument("--per-domain", action="store_true",
                   help="Write each domain's own stats instead of the shared ruler")
    args = p.parse_args()

    out = Path(args.out)
    raws = {pw: load_raw(out, pw) for pw in DOMAIN_NAMES.values()}

    statistics, per_timestep = {}, {}
    if args.per_domain:
        for pw, payload in raws.items():
            statistics[pw], per_timestep[pw] = entry(*accumulators(payload))
        ruler = "per-domain"
    else:
        pooled_scene = WelfordAccumulator(len(raws[next(iter(raws))]["scene"]["mean"]))
        pooled_robot = WelfordAccumulator(len(raws[next(iter(raws))]["robot"]["mean"]))
        pooled_gt = [WelfordAccumulator(3) for _ in range(CLIP_LEN)]
        for payload in raws.values():
            scene, robot, gt_rel = accumulators(payload)
            pooled_scene.merge(scene)
            pooled_robot.merge(robot)
            for t in range(CLIP_LEN):
                pooled_gt[t].merge(gt_rel[t])
        shared_stats, shared_per_ts = entry(pooled_scene, pooled_robot, pooled_gt)
        for pw in raws:
            statistics[pw] = shared_stats
            per_timestep[pw] = shared_per_ts
        ruler = "shared (pooled over the three domains)"

    norm_dir = out / "norm_stats"
    norm_dir.mkdir(parents=True, exist_ok=True)
    (norm_dir / "norm_stats.json").write_text(json.dumps(
        {"statistics": statistics, "per_timestep_statistics": per_timestep}, indent=2))

    print(f"ruler: {ruler}")
    for pw, payload in raws.items():
        print(f"  {pw}: {payload['clips']} train clips accumulated")
    scene_var = statistics[next(iter(statistics))]["scene_features"]["variance"]
    robot_var = statistics[next(iter(statistics))]["robot_features"]["variance"]
    print(f"  scene feature dim {len(scene_var)}, robot feature dim {len(robot_var)}")
    print(f"Wrote {norm_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
