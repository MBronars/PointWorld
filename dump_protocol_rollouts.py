#!/usr/bin/env python3
"""
dump_protocol_rollouts.py — Page data from eval_protocol_pointworld.py receipts.

Reads, per snapshot, the protocol output folder (<run>/eval_protocol/<ckpt>_<schedule>/):
SUMMARY.json, every <dataset>/<set>.json receipt and <set>.cases.json sidecar, and the
headline set's saved predictions (<set>.predictions/<position>.npz: pred [H, rows, 3] m,
rows, predicted). Ground truth for the same cases is read through the evaluator's own
CaseDataset on the PWM run's resolved recipe, in manifest order, so position k of the
set is the k-th clip read.

Writes <out>/protocol.json (tables and curves) and <out>/rollouts_<dataset>.bin plus
<out>/rollouts_<dataset>.json (the selected cases' clouds: int16 millimetres, gripper
rows given, ground truth and each snapshot's prediction on the predicted rows, every
frame from the last context frame to the case's end). Selection: the first N of a
fixed-seed random draw over the set's cases, then the worst K by the LAST snapshot's
per-case mean error — the evaluator's own per-case curve.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--snapshots", nargs="+", required=True, help="label=protocol_out_dir, in budget order")
    p.add_argument("--datasets", default="pickup,rl_push")
    p.add_argument("--out", required=True)
    p.add_argument("--random", type=int, default=8)
    p.add_argument("--worst", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    sys.argv = [sys.argv[0]]
    import eval_protocol_pointworld as E
    E.bootstrap()
    import scripts.unified.eval_unified as EU
    from utils.eval_cases import CaseDataset
    from utils.evaluation import iter_clips

    cfg = E.resolved_recipe(E.DEFAULT_RECIPE)
    snaps = [s.split("=", 1) for s in a.snapshots]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [d for d in a.datasets.split(",") if d]

    page = {"snapshots": [], "datasets": {}}
    for label, folder in snaps:
        summary = json.loads((Path(folder) / "SUMMARY.json").read_text())
        page["snapshots"].append({"label": label, "folder": folder, "checkpoint": summary["checkpoint"],
                                  "checkpoint_sha256": summary["checkpoint_sha256"], "code": summary["code"],
                                  "sets": summary["sets"]})

    for ds_name in datasets:
        rows0 = [r for r in page["snapshots"][0]["sets"] if r["dataset"] == ds_name]
        headline = next(r for r in rows0 if r["headline"])
        case_set, manifest = headline["case_set"], headline["manifest"]
        # the evaluator's per-case curves for every snapshot
        per_case = {}
        for label, folder in snaps:
            side = json.loads((Path(folder) / ds_name / f"{case_set}.cases.json").read_text())
            per_case[label] = side["records"] if "records" in side else side["cases"]
        recs_last = per_case[snaps[-1][0]]
        mean = lambda c: float(np.mean(c)) if c else float("nan")
        n = len(recs_last)
        rand = np.random.default_rng(a.seed).choice(n, size=min(a.random, n), replace=False).tolist()
        order = np.argsort([-mean(r["predicted_mm"]) for r in recs_last]).tolist()
        worst = [k for k in order if k not in rand][: a.worst]
        pick = rand + worst

        # ground truth through the evaluator's own case reader, in manifest order
        domain = headline.get("domain", ds_name)
        dataset = EU.evaluation_dataset(cfg, domain, int(cfg.data.context_len), int(cfg.data.future_len))
        dataset = CaseDataset(dataset, str(E.PWM_ROOT / "data" / "manifests" / "eval_cases" / manifest), case_set)
        want = set(pick)
        gts = {}
        for k, (batch, _) in enumerate(iter_clips(dataset, None)):
            if k in want:
                gts[k] = batch
            if len(gts) == len(want):
                break

        blob, index = bytearray(), []
        def put(arr):
            q = np.clip(np.round(np.asarray(arr) * 1000.0), -32767, 32767).astype("<i2")
            off = len(blob); blob.extend(q.tobytes()); return [off, int(q.size)]
        for k in pick:
            batch = gts[k]
            C = batch.context_len
            preds = {}
            for label, folder in snaps:
                z = np.load(Path(folder) / ds_name / f"{case_set}.predictions" / f"{k:05d}.npz")
                preds[label] = z
            z0 = preds[snaps[-1][0]]
            rows, predicted = z0["rows"], z0["predicted"].astype(bool)
            H = z0["pred"].shape[0]
            pos = batch.positions[0].float().numpy()                     # [T, N, 3]
            act = batch.action_mask[0].numpy().astype(bool)
            srows, grows = rows[predicted], rows[~predicted]
            gt = pos[C - 1:C + H][:, srows]                              # anchor + H frames
            grip = pos[C - 1:C + H][:, grows]
            entry = {"position": int(k), "kind": "random" if k in rand else "worst", "H": int(H),
                     "ns": int(len(srows)), "nr": int(len(grows)),
                     "id": str(z0["id"]), "start": int(z0["start"]),
                     "gt": put(gt), "grip": put(grip), "pred": {},
                     "mean_mm": {lab: mean(per_case[lab][k]["predicted_mm"]) for lab, _ in snaps},
                     "zero_mm": mean(recs_last[k]["zero_predicted_mm"]),
                     "curve": {lab: per_case[lab][k]["predicted_mm"] for lab, _ in snaps},
                     "zero_curve": recs_last[k]["zero_predicted_mm"]}
            for label, _ in snaps:
                z = preds[label]
                if not np.array_equal(z["rows"], rows):
                    raise AssertionError(f"{ds_name} case {k}: saved rows differ between snapshots")
                anchor = pos[C - 1][srows][None]
                entry["pred"][label] = put(np.concatenate([anchor, z["pred"][:, predicted]], axis=0))
            index.append(entry)
        (out / f"rollouts_{ds_name}.bin").write_bytes(bytes(blob))
        (out / f"rollouts_{ds_name}.json").write_text(json.dumps({"case_set": case_set, "cases": index}))
        page["datasets"][ds_name] = {"headline": case_set, "manifest": manifest, "selection": {"random": rand, "worst": worst},
                                     "bin_bytes": len(blob)}
        print(f"{ds_name}: {case_set}, {len(pick)} cases, {len(blob)/1e6:.2f} MB")

    (out / "protocol.json").write_text(json.dumps(page))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
