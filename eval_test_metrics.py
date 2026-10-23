#!/usr/bin/env python3
"""
eval_test_metrics.py — Single-window evaluation on the held-out PointWorld test
WebDataset.

For each sample (one 11-frame clip from the `test/` shards), runs the trained
PointWorld model forward once and computes two metrics on the *predicted*
frames (indices 1..T-1, excluding the context frame 0):

  • Dist : per-frame mean Euclidean distance between predicted and GT
           particle positions, then SUMMED over the predicted frames.
           (i.e. sum_t mean_p ||pred[t,p] - gt[t,p]||_2)
  • MSE  : per-frame mean squared error between predicted and GT
           particle positions, then SUMMED over the predicted frames.
           (i.e. sum_t mean_{p,xyz} (pred[t,p,xyz] - gt[t,p,xyz])^2)

Reports mean ± std of these per-sample sums over N samples drawn from the
test loader.

Usage:
    python eval_test_metrics.py \\
        --checkpoint train_logs/pushing_no_normals/model-last.pt \\
        --data-dir   /data/group_data/katefgroup/VLA/3DPCWM/pushing_data_pointworld_05_03 \\
        --n-samples  256 \\
        --batch-size 32 \\
        --out-dir    eval_logs/pushing_no_normals_test
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress the project's own CLI from grabbing sys.argv when arguments.py imports.
_ORIGINAL_ARGV = sys.argv
sys.argv = [sys.argv[0]]

from arguments import parse_args as pw_parse_args  # noqa: E402
from dataset_components.dataloader import build_dataloader  # noqa: E402
from pointworld.base import BaseModel  # noqa: E402
from pointworld.checkpoint_contract import (  # noqa: E402
    apply_model_contract_to_args,
    read_checkpoint_contract,
)

sys.argv = _ORIGINAL_ARGV


# ─────────────────────────────────────────────────────────────────────────────
# Args + setup
# ─────────────────────────────────────────────────────────────────────────────

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True,
                   help="Path to model-last.pt")
    p.add_argument("--data-dir", required=True,
                   help="Dataset root containing test/, train/, norm_stats/")
    p.add_argument("--domain", default="pushing",
                   help="Domain string (default: pushing)")
    p.add_argument("--n-samples", "-N", type=int, default=64,
                   help="Number of test clips to evaluate (default: 64)")
    p.add_argument("--batch-size", "-b", type=int, default=32,
                   help="Inference batch size (default: 32)")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers (default: 2)")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed controls which test clips are drawn (default: 0)")
    p.add_argument("--device", default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--out-dir", default="eval_logs/test_metrics",
                   help="Where to write the JSON summary")
    p.add_argument("--amp", action="store_true",
                   help="Run forward under bfloat16 autocast (matches training eval).")
    return p.parse_args()


def build_args_for_eval(cli: argparse.Namespace, ckpt: dict) -> argparse.Namespace:
    """Build the full PointWorld args Namespace expected by build_dataloader+BaseModel.

    Starts from arguments.py defaults, overrides the dataset/domain/seed/etc.,
    then applies the checkpoint's model contract so that ptv3_size, predictor_dim,
    robot_features, scene_features all match what the model was trained with.
    """
    args = pw_parse_args(skip_command_line=True)

    # Dataset wiring
    args.data_dirs       = [cli.data_dir]
    args.domains         = [cli.domain]
    args.norm_stats_path = str(Path(cli.data_dir) / "norm_stats")

    # Run-time knobs
    args.batch_size       = cli.batch_size
    args.num_workers      = cli.num_workers
    args.eval_num_workers = cli.num_workers
    args.seed             = int(cli.seed)
    args.device           = cli.device
    args.distributed      = False
    # Resampled eval (force_resampled_eval=True below) needs deterministic_train=False
    # so that build_dataset doesn't force shardshuffle=False / detshuffle=True paths
    # incompatible with resampled mode.
    args.deterministic_train = False
    args.deterministic_data  = False

    # Apply model contract from the checkpoint — sets predictor_dim, ptv3_size,
    # robot_features, scene_features, grid_size, etc. to whatever the model
    # was trained with.
    context = f"eval checkpoint '{cli.checkpoint}'"
    model_contract, _data_contract = read_checkpoint_contract(ckpt, context=context)
    apply_model_contract_to_args(
        args, model_contract, context=context, explicit_cli_dests=set(),
    )
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_batch_metrics(
    pred: np.ndarray,   # (B, T, Ns, 3)  world frame, model output
    gt:   np.ndarray,   # (B, T, Ns, 3)  world frame, ground truth
    mask: np.ndarray,   # (B, T, Ns)     bool: True = real (non-padded) particle
) -> dict:
    """Per-sample masked metrics on the predicted frames (1..T-1).

    For each sample we first take the *per-frame* mean over particles (this is
    the standard per-step error), then SUM those per-frame errors across all
    predicted frames. So a clip's `sample_dist` is the total per-frame mean
    L2 distance accumulated across the 10-frame prediction horizon.

    Returns dict of numpy arrays, one entry per sample in the batch:
      • sample_dist    : (B,)      sum_t  mean_p ||pred - gt||_2
      • sample_mse     : (B,)      sum_t  mean_{p,xyz} (pred - gt)^2
      • per_frame_dist : (B, T-1)  mean_p ||pred - gt||_2  (for diagnostics)
      • per_frame_mse  : (B, T-1)  mean_{p,xyz} (pred - gt)^2
    """
    diff = pred[:, 1:] - gt[:, 1:]                 # (B, T-1, Ns, 3)
    mask_p = mask[:, 1:].astype(np.float32)         # (B, T-1, Ns)
    l2 = np.linalg.norm(diff, axis=-1)              # (B, T-1, Ns)
    sq_per_point = (diff ** 2).mean(axis=-1)        # (B, T-1, Ns)   xyz averaged

    n_pts_per_frame = mask_p.sum(axis=2).clip(min=1.0)         # (B, T-1)
    per_frame_dist  = (l2          * mask_p).sum(axis=2) / n_pts_per_frame
    per_frame_mse   = (sq_per_point * mask_p).sum(axis=2) / n_pts_per_frame

    sample_dist = per_frame_dist.sum(axis=1)        # (B,)  sum over predicted frames
    sample_mse  = per_frame_mse .sum(axis=1)        # (B,)

    return {
        "sample_dist":    sample_dist,
        "sample_mse":     sample_mse,
        "per_frame_dist": per_frame_dist,
        "per_frame_mse":  per_frame_mse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cli = parse_cli()
    device = cli.device if (cli.device == "cuda" and torch.cuda.is_available()) else "cpu"
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint  : {cli.checkpoint}")
    print(f"data_dir    : {cli.data_dir}")
    print(f"n_samples   : {cli.n_samples}")
    print(f"batch_size  : {cli.batch_size}")
    print(f"device      : {device}")
    print()

    # ── 1. Load checkpoint + build args ────────────────────────────────────
    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    args = build_args_for_eval(cli, ckpt)
    print(f"robot_features : {args.robot_features}")
    print(f"scene_features : {args.scene_features}")
    print(f"predictor_dim  : {args.predictor_dim}   ptv3_size: {args.ptv3_size}")
    print()

    # ── 2. Build test dataloader (force_resampled_eval=True → random order) ─
    test_loader, info = build_dataloader(
        args=args, mode="test", rank=0, world_size=1,
        force_resampled_eval=True,
    )
    assert test_loader is not None, "Test dataloader is None — check data_dir/test/ shards."
    print(f"test split: {info['total_samples']} clips, "
          f"{info['total_batches_needed']} batches at batch_size={args.batch_size}")

    # ── 3. Peek one batch to size the model ────────────────────────────────
    loader_iter = iter(test_loader)
    first_batch = next(loader_iter)
    data_info_dict = {
        "robot_features_dim": first_batch["robot_features"].shape[-1],
        "scene_features_dim": first_batch["scene_features"].shape[-1],
    }
    print(f"robot_features_dim={data_info_dict['robot_features_dim']}  "
          f"scene_features_dim={data_info_dict['scene_features_dim']}")

    # ── 4. Build model + load weights ──────────────────────────────────────
    model = BaseModel(args, data_info_dict, rank=0)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print()

    # ── 5. Iterate test loader and accumulate metrics ──────────────────────
    per_sample_dist: list[float] = []
    per_sample_mse:  list[float] = []
    per_frame_dist_sum: np.ndarray | None = None
    per_frame_count = 0
    samples_consumed = 0

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (cli.amp and device == "cuda")
        else torch.cuda.amp.autocast(enabled=False) if device == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    # Re-chain the peeked first batch so we don't waste it
    batch_stream = chain([first_batch], loader_iter)

    with torch.no_grad():
        for batch in batch_stream:
            if samples_consumed >= cli.n_samples:
                break

            batch_dev = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            with autocast_ctx:
                outputs = model(batch_dev, training=False)

            pred = outputs["scene_flows"].float().cpu().numpy()   # (B, T, Ns, 3)
            gt   = batch_dev["scene_flows"].float().cpu().numpy()  # (B, T, Ns, 3)
            mask = batch_dev["scene_exists"].bool().cpu().numpy()  # (B, T, Ns)

            stats = compute_batch_metrics(pred, gt, mask)

            # Trim to n_samples budget
            remaining = cli.n_samples - samples_consumed
            take = min(remaining, stats["sample_dist"].shape[0])

            per_sample_dist.extend(stats["sample_dist"][:take].tolist())
            per_sample_mse .extend(stats["sample_mse"] [:take].tolist())

            pf = stats["per_frame_dist"][:take]                    # (take, T-1)
            if per_frame_dist_sum is None:
                per_frame_dist_sum = np.zeros(pf.shape[1], dtype=np.float64)
            per_frame_dist_sum += pf.sum(axis=0)
            per_frame_count    += take

            samples_consumed += take
            print(f"  consumed {samples_consumed}/{cli.n_samples}  "
                  f"running Σ-Dist={np.mean(per_sample_dist):.6f}  "
                  f"Σ-MSE={np.mean(per_sample_mse):.6e}")

    if samples_consumed < cli.n_samples:
        print(f"WARNING: test loader exhausted after {samples_consumed} samples "
              f"(< requested {cli.n_samples})")

    # ── 6. Aggregate + report ──────────────────────────────────────────────
    dist_arr = np.array(per_sample_dist, dtype=np.float64)
    mse_arr  = np.array(per_sample_mse,  dtype=np.float64)
    per_frame_mean = (per_frame_dist_sum / max(per_frame_count, 1)).tolist()

    ddof = 1 if samples_consumed > 1 else 0
    summary = {
        "n_samples":          int(samples_consumed),
        "seed":               int(cli.seed),
        "checkpoint":         str(cli.checkpoint),
        "data_dir":           str(cli.data_dir),
        "dist_mean":          float(dist_arr.mean()),
        "dist_std":           float(dist_arr.std(ddof=ddof)),
        "mse_mean":           float(mse_arr.mean()),
        "mse_std":            float(mse_arr.std(ddof=ddof)),
        "per_frame_dist_mean": per_frame_mean,
    }

    n_pred_frames = len(per_frame_mean)
    print()
    print("=" * 60)
    print(f"  N samples                            : {summary['n_samples']}")
    print(f"  Predicted frames per clip            : {n_pred_frames}")
    print(f"  Σ-Dist over {n_pred_frames} frames (mean ± std) : "
          f"{summary['dist_mean']:.6f} ± {summary['dist_std']:.6f}")
    print(f"  Σ-MSE  over {n_pred_frames} frames (mean ± std) : "
          f"{summary['mse_mean']:.6e} ± {summary['mse_std']:.6e}")
    print(f"  Per-frame mean Dist (for reference, t=1..{n_pred_frames}):")
    for t, d in enumerate(per_frame_mean, start=1):
        print(f"    frame {t:2d}: {d:.6f}")
    print("=" * 60)

    summary_path = out_dir / "test_metrics_summary.json"
    summary_path.write_text(json.dumps(
        {
            "summary":          summary,
            "per_sample_dist":  per_sample_dist,
            "per_sample_mse":   per_sample_mse,
        },
        indent=2,
    ))
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
