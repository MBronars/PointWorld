#!/usr/bin/env python3
"""
rollout_folding_viz.py — Long-horizon autoregressive rollout + pred-vs-GT
visualization for the PointWorld cloth-folding baseline.

This is the PointWorld analogue of pgnd's
``experiments/train/rollout_cloth.py`` + ``experiments/train/viz_rollout.py``:
it loads a trained ``folding`` checkpoint, autoregressively rolls it out on the
held-out OOD trajectory, and renders the same side-by-side PRED | GT panels:

  <out>/rollout.mp4         side-by-side pred | GT animation
  <out>/rollout_strip.png   static PRED/GT panels at a few key frames
  <out>/frames/NNNN.png     the individual animation frames
  <out>/pred.npz            raw arrays (pred_cloth, gt_cloth, gt_left, gt_right)

How the rollout works (mirrors rolling_infer_from_h5.py, adapted for folding):
  - The PointWorld model takes windows of T=11 frames (1 context + 10 predicted).
  - The cloth + gripper point clouds are grid-subsampled ONCE on the first frame
    (grid_size matches training) so per-point correspondence is stable for the
    whole rollout; the same index set is reused every window.
  - Each window: build the sample dict, run the repo's own `center_shift` +
    `gather_features` (so scene/robot features are byte-identical to training),
    forward the model, uncenter, commit `stride` predicted frames, and feed the
    last committed cloth prediction back as the next context.
  - Grippers are GT-driven the whole way (the model only predicts cloth).

Usage:
    python rollout_folding_viz.py \\
        --checkpoint train_logs/folding_dense_grid/model-last.pt \\
        --out        eval_logs/folding_dense_grid_rollout_viz
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress the project's own CLI from grabbing sys.argv when arguments.py imports.
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
from dataset_components.utils import fnv_hash_vec_nb         # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match the PointWorld release horizon for the `folding` domain
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN       = "folding"
MODEL_T      = 11      # CONTEXT_HORIZON(1) + PRED_HORIZON(10)
CONTEXT_LEN  = 1
PRED_FRAMES  = 10

# Default OOD evaluation trajectory (DATASET_SPEC sec. 6b) — never seen in train.
DEFAULT_OOD = ("/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/"
               "sim1_cloth_variant_0002/pointcloud_dataset/000086.npz")
# Preprocessing knobs the converter used (npz_to_pointworld_wds.py defaults).
SKIP_FRAMES = 30
STRIDE      = 2


# ═══════════════════════════════════════════════════════════════════════════
# Grid subsampling — copied verbatim from transforms.grid_sample_transform so the
# rollout point set matches the density the model was trained on.
# ═══════════════════════════════════════════════════════════════════════════

def grid_select_indices(points_t0: np.ndarray, grid_size: float) -> np.ndarray:
    """Indices kept by test-mode voxel downsampling (one — the first — per voxel)."""
    if points_t0.shape[0] == 0:
        return np.array([], dtype=np.int64)
    grid_coord = np.floor(points_t0 / grid_size).astype(np.int64)
    grid_coord -= grid_coord.min(axis=0)
    keys = fnv_hash_vec_nb(grid_coord)
    idx_sort = np.argsort(keys)
    keys_sorted = keys[idx_sort]
    _, _, counts = np.unique(keys_sorted, return_inverse=True, return_counts=True)
    cumsum_counts = np.cumsum(np.insert(counts, 0, 0)[:-1])
    offsets = np.zeros_like(counts, dtype=np.int64)   # 'test' mode: first per voxel
    return np.sort(idx_sort[cumsum_counts + offsets])


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def build_eval_args(checkpoint: str, data_dir: str, device: str) -> argparse.Namespace:
    """Full PointWorld args Namespace for inference, matching the checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    args = pw_parse_args(skip_command_line=True)
    args.domains          = [DOMAIN]
    args.data_dirs        = [data_dir]
    args.norm_stats_path  = str(Path(data_dir) / "norm_stats")
    args.device           = device
    args.distributed      = False
    args.deterministic_train = False
    args.deterministic_data  = False

    # robot_features / scene_features are stored on the checkpoint's args (the
    # model_contract only carries architecture knobs), so copy them across.
    ckpt_args = ckpt["args"]
    args.robot_features = list(ckpt_args.robot_features)
    args.scene_features = list(ckpt_args.scene_features)

    # Apply architecture contract (predictor_dim, ptv3_size, grid_size, ...).
    context = f"folding rollout checkpoint '{checkpoint}'"
    model_contract, _ = read_checkpoint_contract(ckpt, context=context)
    apply_model_contract_to_args(args, model_contract, context=context,
                                 explicit_cli_dests=set())
    return args, ckpt


def load_model(args: argparse.Namespace, ckpt: dict,
               data_info_dict: dict, device: str) -> BaseModel:
    model = BaseModel(args, data_info_dict, rank=0)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model = model.to(device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Per-window batch construction (mirrors the test-mode data pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def build_window_batch(ctx_cloth: np.ndarray,        # (Ns, 3) world frame
                       robot_win: np.ndarray,        # (MODEL_T, Nr, 3) world frame
                       right_open: np.ndarray,       # (MODEL_T, 1)
                       left_open: np.ndarray,        # (MODEL_T, 1)
                       robot_features: list,
                       scene_features: list,
                       device: str) -> tuple[dict, np.ndarray]:
    """One inference window → batch dict for model.forward(), plus the shift.

    Runs the repo's own `center_shift` + `gather_features`, so scene_features /
    robot_features are computed exactly as during training.
    """
    Ns = ctx_cloth.shape[0]

    # scene_flows is (T, Ns, 3); the model only reads frame 0, so echo the
    # context cloth across all T frames (also what the context mask would do).
    scene_flows = np.tile(ctx_cloth[None].astype(np.float32), (MODEL_T, 1, 1))

    robot_win = robot_win.astype(np.float32)
    sample = {
        "scene_flows":       scene_flows,                        # (T, Ns, 3)
        "robot_flows":       robot_win,                          # (T, Nr, 3)
        # folding stores no per-point gripper normals — zeros, exactly as
        # dataset_components.robot._get_robot_flows_folding synthesizes them.
        "robot_normals":     np.zeros_like(robot_win),           # (T, Nr, 3)
        "right_gripper_open": right_open.astype(np.float32),     # (T, 1)
        "left_gripper_open":  left_open.astype(np.float32),      # (T, 1)
    }

    # center_shift: subtract mean of scene+robot at t=0 (records -shift).
    sample = center_shift(sample)
    shift = -np.asarray(sample["__shift_amount__"], dtype=np.float32)

    # gather_features: builds scene_features (1,Ns,14) and robot_features
    # (T,Nr,14). folding is bimanual; scene_normals is silently skipped.
    sample = gather_features(
        sample,
        robot_features=robot_features,
        scene_features=scene_features,
        random_context_mode="fixed",
        context_horizon=CONTEXT_LEN,
        has_bimanual_robot=True,
        domain=DOMAIN,
    )

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)

    Nr = sample["robot_flows"].shape[1]
    batch = {
        "scene_flows":    t(sample["scene_flows"])[None],            # (1,T,Ns,3)
        "scene_features": t(sample["scene_features"])[None],         # (1,1,Ns,14)
        "scene_exists":   torch.ones(1, MODEL_T, Ns, dtype=torch.bool, device=device),
        "robot_flows":    t(sample["robot_flows"])[None],            # (1,T,Nr,3)
        "robot_features": t(sample["robot_features"])[None],         # (1,T,Nr,14)
        "robot_exists":   torch.ones(1, MODEL_T, Nr, dtype=torch.bool, device=device),
        "__domain__":     [DOMAIN],
    }
    return batch, shift


# ═══════════════════════════════════════════════════════════════════════════
# Autoregressive rollout
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rollout(model: BaseModel,
            cloth: np.ndarray,        # (T_sub, Ns, 3)  GT cloth, subsampled pts
            robot: np.ndarray,        # (T_sub, Nr, 3)  GT gripper pts (model input)
            openness: np.ndarray,     # (T_sub, 2)      [left, right]
            robot_features: list,
            scene_features: list,
            num_frames: int,
            stride: int,
            device: str) -> np.ndarray:
    """Roll the model out `num_frames` steps. Returns predicted cloth (num_frames+1, Ns, 3)."""
    assert 1 <= stride <= PRED_FRAMES
    T_sub, Ns, _ = cloth.shape

    pred = np.zeros((num_frames + 1, Ns, 3), dtype=np.float32)
    pred[0] = cloth[0]                       # frame 0 = GT context
    ctx = cloth[0].copy()

    next_commit = 1
    window = 0
    while next_commit <= num_frames:
        # Robot window: frame 0 = context robot, frames 1..10 = future robot.
        ctx_idx = next_commit - 1
        fut = list(range(next_commit, min(next_commit + PRED_FRAMES, T_sub)))
        idxs = [ctx_idx] + fut
        while len(idxs) < MODEL_T:           # pad short tail by repeating last
            idxs.append(idxs[-1])
        idxs = idxs[:MODEL_T]

        robot_win = robot[idxs]                       # (T, Nr, 3)
        left_open  = openness[idxs, 0:1]              # (T, 1)
        right_open = openness[idxs, 1:2]              # (T, 1)

        batch, shift = build_window_batch(
            ctx, robot_win, right_open, left_open,
            robot_features, scene_features, device,
        )
        out = model(batch, training=False)
        pred_world = out["scene_flows"][0].float().cpu().numpy() + shift   # (T, Ns, 3)

        commit_F = min(stride, PRED_FRAMES, num_frames - next_commit + 1)
        for i in range(commit_F):
            pred[next_commit + i] = pred_world[1 + i]   # index 0 = echoed context
        ctx = pred[next_commit + commit_F - 1].copy()

        cloth_err_mm = float(np.linalg.norm(
            pred[next_commit + commit_F - 1] - cloth[min(next_commit + commit_F - 1, T_sub - 1)],
            axis=-1).mean() * 1000.0)
        print(f"  window {window:3d}  committed -> frame {next_commit + commit_F - 1:4d}"
              f"   cloth L2 {cloth_err_mm:7.2f} mm")
        next_commit += commit_F
        window += 1
    return pred


# ═══════════════════════════════════════════════════════════════════════════
# Visualization — ported from pgnd's experiments/train/viz_rollout.py
# ═══════════════════════════════════════════════════════════════════════════

def draw(ax, cloth, left, right, lims, title):
    ax.clear()
    ax.scatter(cloth[:, 0], cloth[:, 1], cloth[:, 2], s=1.0, c="#1f77b4", alpha=0.5)
    ax.scatter(left[:, 0],  left[:, 1],  left[:, 2],  s=1.5, c="#d62728", alpha=0.6)
    ax.scatter(right[:, 0], right[:, 1], right[:, 2], s=1.5, c="#2ca02c", alpha=0.6)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect((lims[0][1] - lims[0][0],
                       lims[1][1] - lims[1][0],
                       lims[2][1] - lims[2][0]))
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def visualize(pred_cloth: np.ndarray,    # (T, Ns, 3)
              gt_cloth: np.ndarray,      # (T, Ns, 3)
              gt_left: np.ndarray,       # (T, 1000, 3)
              gt_right: np.ndarray,      # (T, 1000, 3)
              out: Path, stride: int, fps: int) -> None:
    T = pred_cloth.shape[0]

    # Shared fixed axis limits from GT (stable animation).
    allpts = np.concatenate([gt_cloth.reshape(-1, 3),
                             gt_left.reshape(-1, 3),
                             gt_right.reshape(-1, 3)], axis=0)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.05 * (hi - lo + 1e-6)
    lims = list(zip(lo - pad, hi + pad))

    # Per-frame cloth L2 error (mm), for the panel titles.
    cloth_err_mm = np.linalg.norm(pred_cloth - gt_cloth, axis=-1).mean(-1) * 1000.0

    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # --- animation ---
    idxs = list(range(0, T, stride))
    for k, f in enumerate(idxs):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        draw(ax1, pred_cloth[f], gt_left[f], gt_right[f], lims,
             f"PRED  frame {f}   cloth L2 {cloth_err_mm[f]:.1f} mm")
        draw(ax2, gt_cloth[f], gt_left[f], gt_right[f], lims,
             f"GROUND TRUTH  frame {f}")
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

    # --- static strip at key frames ---
    keys = sorted({f for f in (0, 100, 250, 500, T - 1) if 0 <= f < T})
    fig = plt.figure(figsize=(4 * len(keys), 8))
    for j, f in enumerate(keys):
        a1 = fig.add_subplot(2, len(keys), j + 1, projection="3d")
        draw(a1, pred_cloth[f], gt_left[f], gt_right[f], lims,
             f"PRED f{f}  ({cloth_err_mm[f]:.1f} mm)")
        a2 = fig.add_subplot(2, len(keys), len(keys) + j + 1, projection="3d")
        draw(a2, gt_cloth[f], gt_left[f], gt_right[f], lims, f"GT f{f}")
    fig.tight_layout()
    fig.savefig(out / "rollout_strip.png", dpi=110)
    plt.close(fig)
    print(f"strip -> {out / 'rollout_strip.png'}")

    ref = [min(f, T - 1) for f in (10, 100, 500, T - 1)]
    print(f"cloth L2 (mm) @ frames {ref}: "
          + " / ".join(f"{cloth_err_mm[f]:.1f}" for f in ref))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", required=True, help="folding model-last.pt")
    p.add_argument("--out", default="eval_logs/folding_rollout_viz",
                   help="Output directory for rollout.mp4 / rollout_strip.png / frames")
    p.add_argument("--ood-npz", default=DEFAULT_OOD,
                   help="Held-out OOD trajectory .npz (DATASET_SPEC sec. 6b)")
    p.add_argument("--data-dir", default=None,
                   help="Dataset root with norm_stats/ (default: read from checkpoint)")
    p.add_argument("--num-frames", type=int, default=1000,
                   help="Number of frames to roll out (default 1000)")
    p.add_argument("--commit-stride", type=int, default=PRED_FRAMES,
                   help=f"Predicted frames committed per window (<= {PRED_FRAMES})")
    p.add_argument("--viz-stride", type=int, default=10,
                   help="Frame subsample for the animation (default 10)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--no-grid-subsample", action="store_true",
                   help="Feed the full-resolution cloth/gripper clouds instead of "
                        "the training-density voxel-downsampled ones")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    cli = parse_cli()
    device = cli.device if (cli.device == "cuda" and torch.cuda.is_available()) else "cpu"
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Args + checkpoint ───────────────────────────────────────────────
    ckpt_meta = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    data_dir = cli.data_dir or ckpt_meta["args"].data_dirs[0]
    print(f"checkpoint : {cli.checkpoint}")
    print(f"data_dir   : {data_dir}")
    print(f"ood_npz    : {cli.ood_npz}")
    print(f"device     : {device}\n")

    args, ckpt = build_eval_args(cli.checkpoint, data_dir, device)
    grid_size = float(args.grid_size)
    print(f"robot_features : {args.robot_features}")
    print(f"scene_features : {args.scene_features}")
    print(f"predictor_dim  : {args.predictor_dim}   ptv3_size: {args.ptv3_size}   "
          f"grid_size: {grid_size}\n")

    # ── 2. Load OOD trajectory, apply skip + temporal stride ───────────────
    with np.load(cli.ood_npz) as d:
        cloth_full = np.asarray(d["cloth_points"], dtype=np.float32)[SKIP_FRAMES::STRIDE]
        left_full  = np.asarray(d["left_ee_points"], dtype=np.float32)[SKIP_FRAMES::STRIDE]
        right_full = np.asarray(d["right_ee_points"], dtype=np.float32)[SKIP_FRAMES::STRIDE]
        openness   = np.asarray(d["openness"], dtype=np.float32)[SKIP_FRAMES::STRIDE]
    robot_full = np.concatenate([left_full, right_full], axis=1)   # (T_sub, 2000, 3)
    T_sub = cloth_full.shape[0]
    print(f"OOD subsampled trajectory: {T_sub} frames "
          f"({cloth_full.shape[1]} cloth pts, {robot_full.shape[1]} gripper pts)")

    num_frames = min(cli.num_frames, T_sub - 1)
    if num_frames < cli.num_frames:
        print(f"WARNING: only {T_sub} frames available -> rolling out {num_frames}")

    # ── 3. Grid-subsample the cloth + gripper clouds ONCE (frame 0) ────────
    if cli.no_grid_subsample:
        cloth_idx = np.arange(cloth_full.shape[1])
        robot_idx = np.arange(robot_full.shape[1])
        print("Grid subsampling DISABLED — using full-resolution clouds")
    else:
        shift0 = np.concatenate([cloth_full[0], robot_full[0]], 0).mean(0)
        cloth_idx = grid_select_indices(cloth_full[0] - shift0, grid_size)
        robot_idx = grid_select_indices(robot_full[0] - shift0, grid_size)
        print(f"Grid subsample @ {grid_size} m: "
              f"cloth {cloth_full.shape[1]} -> {len(cloth_idx)}, "
              f"gripper {robot_full.shape[1]} -> {len(robot_idx)}")

    cloth = cloth_full[:, cloth_idx]            # (T_sub, Ns, 3)  model + viz
    robot = robot_full[:, robot_idx]            # (T_sub, Nr, 3)  model input only

    # ── 4. Build the model (peek one window to size the feature dims) ──────
    peek, _ = build_window_batch(
        cloth[0], robot[:MODEL_T], openness[:MODEL_T, 1:2], openness[:MODEL_T, 0:1],
        args.robot_features, args.scene_features, device,
    )
    data_info_dict = {
        "robot_features_dim": peek["robot_features"].shape[-1],
        "scene_features_dim": peek["scene_features"].shape[-1],
    }
    print(f"robot_features_dim={data_info_dict['robot_features_dim']}  "
          f"scene_features_dim={data_info_dict['scene_features_dim']}\n")
    model = load_model(args, ckpt, data_info_dict, device)

    # ── 5. Autoregressive rollout ──────────────────────────────────────────
    print(f"\nRolling out {num_frames} frames "
          f"(commit_stride={cli.commit_stride}) ...")
    pred_cloth = rollout(
        model, cloth, robot, openness,
        args.robot_features, args.scene_features,
        num_frames, cli.commit_stride, device,
    )

    # ── 6. Assemble GT, save, visualize ────────────────────────────────────
    T = num_frames + 1
    gt_cloth = cloth[:T]
    gt_left  = left_full[:T]
    gt_right = right_full[:T]

    np.savez(out / "pred.npz",
             pred_cloth=pred_cloth, gt_cloth=gt_cloth,
             gt_left=gt_left, gt_right=gt_right, cloth_idx=cloth_idx)
    print(f"\narrays -> {out / 'pred.npz'}")

    print("Rendering visualization ...")
    visualize(pred_cloth, gt_cloth, gt_left, gt_right,
              out, cli.viz_stride, cli.fps)
    print(f"\nAll outputs -> {out}/")


if __name__ == "__main__":
    main()
