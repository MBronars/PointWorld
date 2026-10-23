#!/usr/bin/env python3
"""
rollout_mould_viz.py — Long-horizon autoregressive rollout + side-by-side
PRED|GT visualization for the `mould` domain.

The mould training data lives under ``deformable_mould_10k/`` in the
``(B,T,1280,3)`` batched WDS format (gripper 256 + blob 1024). For a long-
horizon rollout we need the *full* per-demo trajectory, so this script reads
one test-shard entry directly with tarfile/pickle, splits by ``obj_ids``,
subsamples (stride 2 to match training), and rolls out the PointWorld
``mould`` model autoregressively.

Production-style architecture (no gripper-in-scene): the model predicts blob
particles only; the paddle is driven by GT as conditioning. Visualization
shows pred blob vs GT blob; the paddle is GT in both panels.

Usage:
    python rollout_mould_viz.py \\
        --checkpoint train_logs/mould_dense_grid_2gpu/model-last.pt \\
        --out        eval_logs/mould_dense_grid_2gpu_rollout_viz
"""
from __future__ import annotations

import argparse
import glob
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

_ORIGINAL_ARGV = sys.argv
sys.argv = [sys.argv[0]]
from arguments import parse_args as pw_parse_args            # noqa: E402
sys.argv = _ORIGINAL_ARGV

from pointworld.base import BaseModel                        # noqa: E402
from pointworld.checkpoint_contract import (                 # noqa: E402
    apply_model_contract_to_args,
    read_checkpoint_contract,
)
from dataset_components.transforms import center_shift       # noqa: E402
from dataset_components.robot import gather_features         # noqa: E402

from rollout_folding_viz import (                            # noqa: E402
    grid_select_indices, build_eval_args, load_model,
    MODEL_T, CONTEXT_LEN, PRED_FRAMES,
)

DOMAIN = "mould"
DEFAULT_MOULD_TEST_DIR = "/data/group_data/katefgroup/VLA/3DPCWM/deformable_mould_10k/test"
SKIP_FRAMES = 0     # original mould converter already dropped the first 2.
STRIDE      = 2     # matches mould_to_pointworld_wds.py default.


# ═══════════════════════════════════════════════════════════════════════════
# Read one full mould demo trajectory from a test shard
# ═══════════════════════════════════════════════════════════════════════════

def load_mould_demo(test_dir: str, demo_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (particle_pos (T,N,3), valid_particles (T,N), obj_ids (N,))."""
    tar_paths = sorted(glob.glob(f"{test_dir}/*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No tar files in {test_dir}")
    # Each test entry has B=1, so demo_idx maps to an absolute entry index.
    seen = 0
    for tar_path in tar_paths:
        with tarfile.open(tar_path) as tar:
            groups: dict[str, list] = {}
            for m in tar.getmembers():
                stem, _, ext = m.name.partition(".")
                groups.setdefault(stem, []).append((ext, m))
            for stem in sorted(groups.keys()):
                if seen == demo_idx:
                    data = {ext: pickle.loads(tar.extractfile(m).read())
                            for ext, m in groups[stem]}
                    pp  = data["particle_pos.pyd"]      # (1, T, N, 3)
                    vp  = data["valid_particles.pyd"]   # (1, T, N) bool
                    obj = data["obj_ids.pyd"]           # (1, N) int32
                    return pp[0], vp[0], obj[0]
                seen += 1
    raise IndexError(f"demo_idx={demo_idx} out of range (only {seen} test demos)")


# ═══════════════════════════════════════════════════════════════════════════
# Window batch + rollout — cloth-only scene (production-style)
# ═══════════════════════════════════════════════════════════════════════════

def build_window_batch_mould(ctx_cloth: np.ndarray,    # (Ns, 3)
                             robot_win: np.ndarray,    # (T, Nr, 3)
                             robot_features: list,
                             scene_features: list,
                             device: str) -> tuple[dict, np.ndarray]:
    Ns = ctx_cloth.shape[0]
    Nr = robot_win.shape[1]
    scene_flows = np.tile(ctx_cloth[None].astype(np.float32), (MODEL_T, 1, 1))
    robot_win = robot_win.astype(np.float32)

    sample = {
        "scene_flows":        scene_flows,
        "robot_flows":        robot_win,
        "robot_normals":      np.zeros_like(robot_win),
        # Mould has no left/right grippers; both placeholders are zeros, and
        # the robot_features list doesn't include `gripper_open` anyway.
        "right_gripper_open": np.zeros((MODEL_T, 1), dtype=np.float32),
        "left_gripper_open":  np.zeros((MODEL_T, 1), dtype=np.float32),
    }
    sample = center_shift(sample)
    shift = -np.asarray(sample["__shift_amount__"], dtype=np.float32)
    sample = gather_features(
        sample,
        robot_features=robot_features,
        scene_features=scene_features,
        random_context_mode="fixed",
        context_horizon=CONTEXT_LEN,
        has_bimanual_robot=False,
        domain=DOMAIN,
    )

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)

    batch = {
        "scene_flows":    t(sample["scene_flows"])[None],
        "scene_features": t(sample["scene_features"])[None],
        "scene_exists":   torch.ones(1, MODEL_T, Ns, dtype=torch.bool, device=device),
        "robot_flows":    t(sample["robot_flows"])[None],
        "robot_features": t(sample["robot_features"])[None],
        "robot_exists":   torch.ones(1, MODEL_T, Nr, dtype=torch.bool, device=device),
        "__domain__":     [DOMAIN],
    }
    return batch, shift


@torch.no_grad()
def rollout_mould(model: BaseModel,
                  cloth: np.ndarray,       # (T_sub, Ns, 3)
                  robot: np.ndarray,       # (T_sub, Nr, 3)
                  robot_features: list,
                  scene_features: list,
                  num_frames: int,
                  stride: int,
                  device: str) -> np.ndarray:
    assert 1 <= stride <= PRED_FRAMES
    T_sub, Ns, _ = cloth.shape

    pred = np.zeros((num_frames + 1, Ns, 3), dtype=np.float32)
    pred[0] = cloth[0]
    ctx = cloth[0].copy()

    next_commit = 1
    window = 0
    while next_commit <= num_frames:
        ctx_idx = next_commit - 1
        fut = list(range(next_commit, min(next_commit + PRED_FRAMES, T_sub)))
        idxs = [ctx_idx] + fut
        while len(idxs) < MODEL_T:
            idxs.append(idxs[-1])
        idxs = idxs[:MODEL_T]

        batch, shift = build_window_batch_mould(
            ctx, robot[idxs],
            robot_features, scene_features, device,
        )
        out = model(batch, training=False)
        pred_world = out["scene_flows"][0].float().cpu().numpy() + shift

        commit_F = min(stride, PRED_FRAMES, num_frames - next_commit + 1)
        for i in range(commit_F):
            pred[next_commit + i] = pred_world[1 + i]
        ctx = pred[next_commit + commit_F - 1].copy()

        ref = min(next_commit + commit_F - 1, T_sub - 1)
        err_mm = float(np.linalg.norm(
            pred[next_commit + commit_F - 1] - cloth[ref], axis=-1).mean() * 1000.0)
        print(f"  window {window:3d}  -> frame {next_commit + commit_F - 1:4d}"
              f"   blob L2 {err_mm:7.2f} mm")
        next_commit += commit_F
        window += 1
    return pred


# ═══════════════════════════════════════════════════════════════════════════
# Visualization (single gripper)
# ═══════════════════════════════════════════════════════════════════════════

def draw(ax, blob, paddle, lims, title):
    ax.clear()
    ax.scatter(blob[:, 0],   blob[:, 1],   blob[:, 2],   s=2.0, c="#1f77b4", alpha=0.55)
    ax.scatter(paddle[:, 0], paddle[:, 1], paddle[:, 2], s=2.5, c="#d62728", alpha=0.7)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect((lims[0][1] - lims[0][0],
                       lims[1][1] - lims[1][0],
                       lims[2][1] - lims[2][0]))
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def visualize_mould(pred_blob: np.ndarray,    # (T, Ns, 3)
                    gt_blob: np.ndarray,      # (T, Ns, 3)
                    gt_paddle: np.ndarray,    # (T, Nr_full, 3) — kept full for viz richness
                    out: Path, stride: int, fps: int) -> None:
    T = pred_blob.shape[0]
    allpts = np.concatenate([gt_blob.reshape(-1, 3),
                             gt_paddle.reshape(-1, 3),
                             pred_blob.reshape(-1, 3)], axis=0)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.05 * (hi - lo + 1e-6)
    lims = list(zip(lo - pad, hi + pad))

    err_mm = np.linalg.norm(pred_blob - gt_blob, axis=-1).mean(-1) * 1000.0
    frame_dir = out / "frames"; frame_dir.mkdir(parents=True, exist_ok=True)

    # animation
    idxs = list(range(0, T, stride))
    for k, f in enumerate(idxs):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        draw(ax1, pred_blob[f], gt_paddle[f], lims,
             f"PRED  frame {f}   blob L2 {err_mm[f]:.1f} mm")
        draw(ax2, gt_blob[f], gt_paddle[f], lims, f"GROUND TRUTH  frame {f}")
        fig.tight_layout()
        fig.savefig(frame_dir / f"{k:04d}.png", dpi=80)
        plt.close(fig)
        if (k + 1) % 20 == 0:
            print(f"  rendered {k + 1}/{len(idxs)} frames")

    video = out / "rollout.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-framerate", str(fps), "-i", str(frame_dir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                   check=False)
    print(f"video -> {video}")

    # strip (mould trajs ~150 sub frames; use coarser keys)
    keys = sorted({f for f in (0, 30, 60, 100, T - 1) if 0 <= f < T})
    fig = plt.figure(figsize=(4 * len(keys), 8))
    for j, f in enumerate(keys):
        a1 = fig.add_subplot(2, len(keys), j + 1, projection="3d")
        draw(a1, pred_blob[f], gt_paddle[f], lims, f"PRED f{f}  ({err_mm[f]:.1f} mm)")
        a2 = fig.add_subplot(2, len(keys), len(keys) + j + 1, projection="3d")
        draw(a2, gt_blob[f], gt_paddle[f], lims, f"GT f{f}")
    fig.tight_layout()
    fig.savefig(out / "rollout_strip.png", dpi=110)
    plt.close(fig)
    print(f"strip -> {out / 'rollout_strip.png'}")

    ref = [min(f, T - 1) for f in (10, 30, 60, 100, T - 1)]
    print(f"blob L2 (mm) @ frames {ref}: "
          + " / ".join(f"{err_mm[f]:.1f}" for f in ref))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="eval_logs/mould_rollout_viz")
    p.add_argument("--test-dir", default=DEFAULT_MOULD_TEST_DIR)
    p.add_argument("--demo-idx", type=int, default=0,
                   help="Which test demo to roll out on (0..N-1 in shard order).")
    p.add_argument("--data-dir", default=None,
                   help="Dataset root w/ norm_stats (default: from ckpt args).")
    p.add_argument("--num-frames", type=int, default=140,
                   help="Frames to roll out (test trajs have ~149 after stride-2).")
    p.add_argument("--commit-stride", type=int, default=PRED_FRAMES)
    p.add_argument("--viz-stride", type=int, default=5)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--no-grid-subsample", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    cli = parse_cli()
    device = cli.device if (cli.device == "cuda" and torch.cuda.is_available()) else "cpu"
    out = Path(cli.out); out.mkdir(parents=True, exist_ok=True)

    ckpt_meta = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    data_dir = cli.data_dir or ckpt_meta["args"].data_dirs[0]
    print(f"checkpoint : {cli.checkpoint}")
    print(f"data_dir   : {data_dir}")
    print(f"test_dir   : {cli.test_dir}  (demo_idx={cli.demo_idx})\n")

    args, ckpt = build_eval_args(cli.checkpoint, data_dir, device)
    # build_eval_args lives in rollout_folding_viz.py and hardcodes
    # args.domains = ["folding"]; override for mould so norm_stats lookup
    # finds the right key in norm_stats.json.
    args.domains = [DOMAIN]
    grid_size = float(args.grid_size)
    print(f"robot_features : {args.robot_features}")
    print(f"scene_features : {args.scene_features}")
    print(f"grid_size      : {grid_size}\n")

    # ── Load a full test demo trajectory ───────────────────────────────────
    pp, vp, obj = load_mould_demo(cli.test_dir, cli.demo_idx)
    print(f"loaded demo: T={pp.shape[0]} frames, N={pp.shape[1]} particles")
    grip_mask = (obj == 1)
    blob_mask = (obj == 0)

    # Apply the same temporal subsampling we used during training conversion.
    sub = slice(SKIP_FRAMES, None, STRIDE)
    pp_sub = pp[sub]
    T_sub = pp_sub.shape[0]
    print(f"subsampled (skip={SKIP_FRAMES}, stride={STRIDE}): {T_sub} frames")

    blob_full   = pp_sub[:, blob_mask]   # (T_sub, 1024, 3) -- blob particles
    paddle_full = pp_sub[:, grip_mask]   # (T_sub, 256, 3)  -- paddle points

    num_frames = min(cli.num_frames, T_sub - 1)
    if num_frames < cli.num_frames:
        print(f"WARNING: traj only has {T_sub} sub frames; capping rollout at {num_frames}")

    # Grid subsample once at frame 0 (matches training distribution).
    if cli.no_grid_subsample:
        blob_idx   = np.arange(blob_full.shape[1])
        paddle_idx = np.arange(paddle_full.shape[1])
    else:
        shift0 = np.concatenate([blob_full[0], paddle_full[0]], axis=0).mean(0)
        blob_idx   = grid_select_indices(blob_full[0]   - shift0, grid_size)
        paddle_idx = grid_select_indices(paddle_full[0] - shift0, grid_size)
    blob   = blob_full[:,   blob_idx]    # (T_sub, Ns, 3)
    paddle = paddle_full[:, paddle_idx]  # (T_sub, Nr, 3)  used as robot input
    print(f"Grid @ {grid_size} m: blob {blob_full.shape[1]}->{len(blob_idx)}, "
          f"paddle {paddle_full.shape[1]}->{len(paddle_idx)}")

    # ── Peek + build model ────────────────────────────────────────────────
    peek_batch, _ = build_window_batch_mould(
        blob[0], paddle[:MODEL_T],
        args.robot_features, args.scene_features, device,
    )
    data_info_dict = {
        "robot_features_dim": peek_batch["robot_features"].shape[-1],
        "scene_features_dim": peek_batch["scene_features"].shape[-1],
    }
    print(f"robot_features_dim={data_info_dict['robot_features_dim']}  "
          f"scene_features_dim={data_info_dict['scene_features_dim']}\n")
    model = load_model(args, ckpt, data_info_dict, device)

    # ── Rollout ───────────────────────────────────────────────────────────
    print(f"Rolling out {num_frames} frames (commit_stride={cli.commit_stride}) ...")
    pred_blob = rollout_mould(
        model, blob, paddle,
        args.robot_features, args.scene_features,
        num_frames, cli.commit_stride, device,
    )

    T = num_frames + 1
    gt_blob   = blob[:T]
    gt_paddle = paddle_full[:T]   # use full-resolution paddle for viz richness

    np.savez(out / "pred.npz",
             pred_blob=pred_blob, gt_blob=gt_blob, gt_paddle=gt_paddle,
             blob_idx=blob_idx, paddle_idx=paddle_idx)
    print(f"\narrays -> {out / 'pred.npz'}")

    print("Rendering visualization ...")
    visualize_mould(pred_blob, gt_blob, gt_paddle, out, cli.viz_stride, cli.fps)
    print(f"\nAll outputs -> {out}/")


if __name__ == "__main__":
    main()
