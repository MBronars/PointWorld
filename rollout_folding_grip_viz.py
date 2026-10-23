#!/usr/bin/env python3
"""
rollout_folding_grip_viz.py — Long-horizon autoregressive rollout + viz for
the `--include_gripper_in_scene true` folding checkpoint.

Differences vs ``rollout_folding_viz.py``:

  - scene_flows is built as ``concat([ctx_cloth, ctx_gripper], axis=points)``
    each window, matching the merge transform that ran during training.
  - The model thus predicts ``(Nc + Nr)`` per-point trajectories per frame —
    we split the output back into ``pred_cloth`` (first Nc) and
    ``pred_gripper`` (last Nr).
  - Cloth context for the next window is taken from the *predicted* cloth (as
    before). Gripper context for the next window is taken from the *GT*
    gripper at that frame (consistent with how the model's robot conditioning
    sees it).
  - **PRED panel shows the predicted gripper (NOT the GT)** — user request:
    see what the model thinks the gripper is doing. GT panel still shows GT
    gripper.

Usage:
    python rollout_folding_grip_viz.py \\
        --checkpoint train_logs/folding_grippers_in_scene/model-last.pt \\
        --out        eval_logs/folding_grippers_in_scene_rollout_viz
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
    DOMAIN, MODEL_T, CONTEXT_LEN, PRED_FRAMES, SKIP_FRAMES, STRIDE,
    DEFAULT_OOD,
    grid_select_indices, build_eval_args, load_model,
)


# ═══════════════════════════════════════════════════════════════════════════
# Window batch (merged scene = cloth + gripper)
# ═══════════════════════════════════════════════════════════════════════════

def build_window_batch_grip(ctx_cloth: np.ndarray,        # (Nc, 3) world frame
                            ctx_gripper: np.ndarray,      # (Nr, 3) world frame
                            robot_win: np.ndarray,        # (T, Nr, 3) world frame
                            right_open: np.ndarray,       # (T, 1)
                            left_open: np.ndarray,        # (T, 1)
                            robot_features: list,
                            scene_features: list,
                            device: str) -> tuple[dict, np.ndarray, int]:
    """Build a model batch with scene = concat(cloth, gripper).

    Returns (batch_dict, world_shift, Nc) so the caller can split predictions.
    """
    Nc = ctx_cloth.shape[0]
    Nr = ctx_gripper.shape[0]
    Ns = Nc + Nr

    # Merged scene: cloth first then gripper, echoed across all T frames.
    scene_t0 = np.concatenate([ctx_cloth, ctx_gripper], axis=0).astype(np.float32)
    scene_flows = np.tile(scene_t0[None], (MODEL_T, 1, 1))

    robot_win = robot_win.astype(np.float32)
    sample = {
        "scene_flows":        scene_flows,                                # (T, Nc+Nr, 3)
        "robot_flows":        robot_win,                                  # (T, Nr, 3)
        "robot_normals":      np.zeros_like(robot_win),
        "right_gripper_open": right_open.astype(np.float32),
        "left_gripper_open":  left_open.astype(np.float32),
    }

    sample = center_shift(sample)
    shift = -np.asarray(sample["__shift_amount__"], dtype=np.float32)
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

    batch = {
        "scene_flows":    t(sample["scene_flows"])[None],            # (1,T,Ns,3)
        "scene_features": t(sample["scene_features"])[None],         # (1,1,Ns,14)
        "scene_exists":   torch.ones(1, MODEL_T, Ns, dtype=torch.bool, device=device),
        "robot_flows":    t(sample["robot_flows"])[None],            # (1,T,Nr,3)
        "robot_features": t(sample["robot_features"])[None],         # (1,T,Nr,14)
        "robot_exists":   torch.ones(1, MODEL_T, Nr, dtype=torch.bool, device=device),
        "__domain__":     [DOMAIN],
    }
    return batch, shift, Nc


# ═══════════════════════════════════════════════════════════════════════════
# Autoregressive rollout — split predictions into cloth + gripper
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rollout_grip(model: BaseModel,
                 cloth: np.ndarray,        # (T_sub, Nc, 3) GT cloth (subsampled idx)
                 robot: np.ndarray,        # (T_sub, Nr, 3) GT gripper (subsampled idx)
                 openness: np.ndarray,     # (T_sub, 2)    [left, right]
                 robot_features: list,
                 scene_features: list,
                 num_frames: int,
                 stride: int,
                 device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_cloth, pred_gripper) of shape (num_frames+1, N{c,r}, 3).

    Cloth context for next window comes from the prediction (true
    autoregressive). Gripper context comes from GT robot at that frame
    (consistent with the model's robot conditioning).
    """
    assert 1 <= stride <= PRED_FRAMES
    T_sub, Nc, _ = cloth.shape
    _, Nr, _ = robot.shape

    pred_cloth   = np.zeros((num_frames + 1, Nc, 3), dtype=np.float32)
    pred_gripper = np.zeros((num_frames + 1, Nr, 3), dtype=np.float32)
    pred_cloth[0]   = cloth[0]
    pred_gripper[0] = robot[0]
    ctx_cloth = cloth[0].copy()

    next_commit = 1
    window = 0
    while next_commit <= num_frames:
        ctx_idx = next_commit - 1
        fut = list(range(next_commit, min(next_commit + PRED_FRAMES, T_sub)))
        idxs = [ctx_idx] + fut
        while len(idxs) < MODEL_T:
            idxs.append(idxs[-1])
        idxs = idxs[:MODEL_T]

        robot_win = robot[idxs]                    # (T, Nr, 3)
        ctx_gripper = robot[ctx_idx]               # GT gripper at the context frame
        left_open  = openness[idxs, 0:1]
        right_open = openness[idxs, 1:2]

        batch, shift, _ = build_window_batch_grip(
            ctx_cloth, ctx_gripper,
            robot_win, right_open, left_open,
            robot_features, scene_features, device,
        )
        out = model(batch, training=False)
        pred_centered = out["scene_flows"][0].float().cpu().numpy()  # (T, Nc+Nr, 3)
        pred_world = pred_centered + shift                            # uncenter

        pred_world_cloth   = pred_world[:, :Nc]      # (T, Nc, 3)
        pred_world_gripper = pred_world[:, Nc:]      # (T, Nr, 3)

        commit_F = min(stride, PRED_FRAMES, num_frames - next_commit + 1)
        for i in range(commit_F):
            pred_cloth  [next_commit + i] = pred_world_cloth  [1 + i]
            pred_gripper[next_commit + i] = pred_world_gripper[1 + i]
        ctx_cloth = pred_cloth[next_commit + commit_F - 1].copy()

        ref_idx = min(next_commit + commit_F - 1, T_sub - 1)
        cl_err = float(np.linalg.norm(
            pred_cloth[next_commit + commit_F - 1] - cloth[ref_idx],
            axis=-1).mean() * 1000.0)
        gr_err = float(np.linalg.norm(
            pred_gripper[next_commit + commit_F - 1] - robot[ref_idx],
            axis=-1).mean() * 1000.0)
        print(f"  window {window:3d}  -> frame {next_commit + commit_F - 1:4d}"
              f"   cloth L2 {cl_err:7.2f} mm   gripper L2 {gr_err:7.2f} mm")
        next_commit += commit_F
        window += 1
    return pred_cloth, pred_gripper


# ═══════════════════════════════════════════════════════════════════════════
# Visualization — PRED panel uses predicted gripper, GT panel uses GT gripper
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


def visualize_grip(pred_cloth: np.ndarray,    # (T, Nc, 3)
                   pred_gripper: np.ndarray,  # (T, Nr, 3) — model's gripper prediction
                   gt_cloth: np.ndarray,      # (T, Nc, 3)
                   gt_left: np.ndarray,       # (T, 1000, 3) — for GT panel
                   gt_right: np.ndarray,      # (T, 1000, 3) — for GT panel
                   robot_idx: np.ndarray,     # (Nr,) original 0..2000 indices, for left/right split
                   out: Path, stride: int, fps: int) -> None:
    T = pred_cloth.shape[0]

    # The pred_gripper is the subsampled (after grid) gripper, in the same
    # index order as robot_full = [left(1000), right(1000)]. So split by
    # whether the kept index < 1000 (left) or >= 1000 (right).
    left_mask  = robot_idx < 1000
    right_mask = robot_idx >= 1000
    pred_left  = pred_gripper[:, left_mask]
    pred_right = pred_gripper[:, right_mask]

    # Axis limits from everything visible (pred + GT).
    allpts = np.concatenate([
        gt_cloth.reshape(-1, 3),
        gt_left.reshape(-1, 3),
        gt_right.reshape(-1, 3),
        pred_cloth.reshape(-1, 3),
        pred_gripper.reshape(-1, 3),
    ], axis=0)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.05 * (hi - lo + 1e-6)
    lims = list(zip(lo - pad, hi + pad))

    cloth_err_mm   = np.linalg.norm(pred_cloth - gt_cloth, axis=-1).mean(-1) * 1000.0

    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # --- animation ---
    idxs = list(range(0, T, stride))
    for k, f in enumerate(idxs):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        draw(ax1, pred_cloth[f], pred_left[f], pred_right[f], lims,
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

    # --- static strip ---
    keys = sorted({f for f in (0, 100, 250, 500, T - 1) if 0 <= f < T})
    fig = plt.figure(figsize=(4 * len(keys), 8))
    for j, f in enumerate(keys):
        a1 = fig.add_subplot(2, len(keys), j + 1, projection="3d")
        draw(a1, pred_cloth[f], pred_left[f], pred_right[f], lims,
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="eval_logs/folding_grippers_in_scene_rollout_viz")
    p.add_argument("--ood-npz", default=DEFAULT_OOD)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--num-frames", type=int, default=1000)
    p.add_argument("--commit-stride", type=int, default=PRED_FRAMES)
    p.add_argument("--viz-stride", type=int, default=10)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--no-grid-subsample", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    cli = parse_cli()
    device = cli.device if (cli.device == "cuda" and torch.cuda.is_available()) else "cpu"
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_meta = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    data_dir = cli.data_dir or ckpt_meta["args"].data_dirs[0]
    print(f"checkpoint : {cli.checkpoint}")
    print(f"data_dir   : {data_dir}")
    print(f"ood_npz    : {cli.ood_npz}\n")

    args, ckpt = build_eval_args(cli.checkpoint, data_dir, device)
    grid_size = float(args.grid_size)
    print(f"include_gripper_in_scene (from ckpt args) : "
          f"{getattr(ckpt['args'], 'include_gripper_in_scene', False)}")
    print(f"grid_size  : {grid_size}\n")

    # Load OOD trajectory.
    with np.load(cli.ood_npz) as d:
        cloth_full = np.asarray(d["cloth_points"],   dtype=np.float32)[SKIP_FRAMES::STRIDE]
        left_full  = np.asarray(d["left_ee_points"], dtype=np.float32)[SKIP_FRAMES::STRIDE]
        right_full = np.asarray(d["right_ee_points"],dtype=np.float32)[SKIP_FRAMES::STRIDE]
        openness   = np.asarray(d["openness"],       dtype=np.float32)[SKIP_FRAMES::STRIDE]
    robot_full = np.concatenate([left_full, right_full], axis=1)
    T_sub = cloth_full.shape[0]
    num_frames = min(cli.num_frames, T_sub - 1)
    print(f"OOD subsampled T={T_sub}; rolling out {num_frames} frames")

    if cli.no_grid_subsample:
        cloth_idx = np.arange(cloth_full.shape[1])
        robot_idx = np.arange(robot_full.shape[1])
    else:
        shift0 = np.concatenate([cloth_full[0], robot_full[0]], 0).mean(0)
        cloth_idx = grid_select_indices(cloth_full[0] - shift0, grid_size)
        robot_idx = grid_select_indices(robot_full[0] - shift0, grid_size)
    cloth = cloth_full[:, cloth_idx]
    robot = robot_full[:, robot_idx]
    print(f"Grid @ {grid_size} m: cloth {cloth_full.shape[1]}->{len(cloth_idx)}, "
          f"gripper {robot_full.shape[1]}->{len(robot_idx)}")

    # Peek + build model.
    peek_batch, _, _ = build_window_batch_grip(
        cloth[0], robot[0], robot[:MODEL_T],
        openness[:MODEL_T, 1:2], openness[:MODEL_T, 0:1],
        args.robot_features, args.scene_features, device,
    )
    data_info_dict = {
        "robot_features_dim": peek_batch["robot_features"].shape[-1],
        "scene_features_dim": peek_batch["scene_features"].shape[-1],
    }
    print(f"robot_features_dim={data_info_dict['robot_features_dim']}  "
          f"scene_features_dim={data_info_dict['scene_features_dim']}\n")
    model = load_model(args, ckpt, data_info_dict, device)

    print(f"\nRolling out (commit_stride={cli.commit_stride}) ...")
    pred_cloth, pred_gripper = rollout_grip(
        model, cloth, robot, openness,
        args.robot_features, args.scene_features,
        num_frames, cli.commit_stride, device,
    )

    T = num_frames + 1
    gt_cloth = cloth[:T]
    gt_left  = left_full[:T]
    gt_right = right_full[:T]

    np.savez(out / "pred.npz",
             pred_cloth=pred_cloth, pred_gripper=pred_gripper,
             gt_cloth=gt_cloth, gt_left=gt_left, gt_right=gt_right,
             cloth_idx=cloth_idx, robot_idx=robot_idx)
    print(f"\narrays -> {out / 'pred.npz'}")

    print("Rendering visualization (PRED panel uses predicted gripper) ...")
    visualize_grip(pred_cloth, pred_gripper, gt_cloth, gt_left, gt_right,
                   robot_idx, out, cli.viz_stride, cli.fps)
    print(f"\nAll outputs -> {out}/")


if __name__ == "__main__":
    main()
