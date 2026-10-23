#!/usr/bin/env python3
"""
rollout_pickup_viz.py — Long-horizon autoregressive rollout + side-by-side
PRED|GT visualization for the `pickup` domain.

The pickup source data lives under
``robert_pickup_data_{place,convex,pick,upright}_success/gpu0/*.tar``
with ``particle_pos[B=4, T=40, N=1183, 3]`` per WDS sample. For a long-
horizon rollout we want the *full* 40-frame demo, so this script reads
one batched sample directly with tarfile/pickle, picks a per-demo elem,
splits particles by ``obj_ids`` (0=object, 1=gripper), and autoregressively
rolls out the PointWorld ``pickup`` model.

Production-style architecture (no gripper-in-scene): the model predicts
object particles only; the gripper is driven by GT as conditioning.
Visualization shows pred object vs GT object; the gripper is GT in both
panels.

Usage:
    python rollout_pickup_viz.py \\
        --checkpoint train_logs/pickup_1gpu/model-last.pt \\
        --out        eval_logs/pickup_1gpu_rollout_viz \\
        --num-demos 4
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

DOMAIN = "pickup"
SOURCE_ROOTS = [
    "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_place_success",
    "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_convex_success",
    "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success",
    "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_upright_success",
]


# ═══════════════════════════════════════════════════════════════════════════
# Load one demo
# ═══════════════════════════════════════════════════════════════════════════

TELEOP_ROOT = "/data/group_data/katefgroup/VLA/3DPCWM/teleop_long_horizon_clean"


def load_teleop_demo(demo_path: str):
    """Load one teleop_long_horizon T=201 demo (.pkl).

    Returns (particle_pos (T,N,3), obj_ids (N,)).
    """
    with open(demo_path, "rb") as f:
        d = pickle.load(f)
    return (d["particle_pos"].astype(np.float32),
            d["obj_ids"].astype(np.int32))


def load_pickup_demo(root: str, sample_idx: int, elem: int = 0):
    """Original T=40 source loader (kept for backward compatibility)."""
    tar_paths = sorted(glob.glob(f"{root}/gpu0/*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"no tars in {root}/gpu0/")
    seen = 0
    for tp in tar_paths:
        with tarfile.open(tp) as tar:
            groups: dict[str, list] = {}
            for m in tar.getmembers():
                stem, _, ext = m.name.partition(".")
                groups.setdefault(stem, []).append((ext, m))
            for stem in sorted(groups.keys()):
                if seen != sample_idx:
                    seen += 1
                    continue
                d = {ext: pickle.loads(tar.extractfile(m).read())
                     for ext, m in groups[stem]}
                pp = d["particle_pos.pyd"]              # (B, T, N, 3)
                oid = d["obj_ids.pyd"]                  # (B, N)
                return (pp[elem].astype(np.float32),    # (T, N, 3)
                        oid[elem].astype(np.int32))     # (N,)
    raise IndexError(f"sample_idx={sample_idx} out of range")


# ═══════════════════════════════════════════════════════════════════════════
# Window batch + rollout — object-only scene (production-style)
# ═══════════════════════════════════════════════════════════════════════════

def build_window_batch_pickup(ctx_obj: np.ndarray,    # (Ns, 3)
                               robot_win: np.ndarray,  # (T, Nr, 3)
                               robot_features: list,
                               scene_features: list,
                               device: str):
    Ns = ctx_obj.shape[0]
    Nr = robot_win.shape[1]
    scene_flows = np.tile(ctx_obj[None].astype(np.float32), (MODEL_T, 1, 1))
    robot_win = robot_win.astype(np.float32)

    sample = {
        "scene_flows":        scene_flows,
        "robot_flows":        robot_win,
        "robot_normals":      np.zeros_like(robot_win),
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
def rollout_pickup(model: BaseModel,
                    obj_traj: np.ndarray,        # (T_full, Ns, 3)
                    robot_traj: np.ndarray,      # (T_full, Nr, 3)
                    robot_features: list,
                    scene_features: list,
                    num_frames: int,
                    stride: int,
                    device: str) -> np.ndarray:
    """Autoregressive rollout. Returns pred (num_frames+1, Ns, 3)."""
    assert 1 <= stride <= PRED_FRAMES
    T_full, Ns, _ = obj_traj.shape

    pred = np.zeros((num_frames + 1, Ns, 3), dtype=np.float32)
    pred[0] = obj_traj[0]
    ctx = obj_traj[0].copy()

    next_commit = 1
    window = 0
    while next_commit <= num_frames:
        ctx_idx = next_commit - 1
        fut = list(range(next_commit, min(next_commit + PRED_FRAMES, T_full)))
        idxs = [ctx_idx] + fut
        while len(idxs) < MODEL_T:
            idxs.append(idxs[-1])
        idxs = idxs[:MODEL_T]

        batch, shift = build_window_batch_pickup(
            ctx, robot_traj[idxs],
            robot_features, scene_features, device,
        )
        out = model(batch, training=False)
        pred_world = out["scene_flows"][0].float().cpu().numpy() + shift

        commit_F = min(stride, PRED_FRAMES, num_frames - next_commit + 1)
        for i in range(commit_F):
            pred[next_commit + i] = pred_world[1 + i]
        ctx = pred[next_commit + commit_F - 1].copy()

        ref = min(next_commit + commit_F - 1, T_full - 1)
        err_mm = float(np.linalg.norm(
            pred[next_commit + commit_F - 1] - obj_traj[ref], axis=-1).mean() * 1000.0)
        print(f"  window {window:3d}  -> frame {next_commit + commit_F - 1:4d}"
              f"   obj L2 {err_mm:7.2f} mm")
        next_commit += commit_F
        window += 1
    return pred


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def draw(ax, obj_pts, grip_pts, lims, title):
    ax.clear()
    ax.scatter(obj_pts[:, 0],  obj_pts[:, 1],  obj_pts[:, 2],
               s=4.0, c="#1f77b4", alpha=0.7, depthshade=False)
    ax.scatter(grip_pts[:, 0], grip_pts[:, 1], grip_pts[:, 2],
               s=2.0, c="#d62728", alpha=0.55, depthshade=False)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_box_aspect((lims[0][1]-lims[0][0],
                       lims[1][1]-lims[1][0],
                       lims[2][1]-lims[2][0]))
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=20, azim=-65)


def visualize_pickup(pred_obj: np.ndarray,   # (T, Ns, 3)
                      gt_obj: np.ndarray,     # (T, Ns, 3)
                      gt_grip: np.ndarray,    # (T, Nr, 3)
                      out: Path, fps: int = 8,
                      render_video: bool = True) -> None:
    T = pred_obj.shape[0]
    allpts = np.concatenate([gt_obj.reshape(-1, 3),
                             gt_grip.reshape(-1, 3),
                             pred_obj.reshape(-1, 3)], axis=0)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.05 * (hi - lo + 1e-6)
    lims = list(zip(lo - pad, hi + pad))

    # Per-frame per-particle L2 between pred and GT object particles.
    # Shape (T, N_obj); we save the full thing + several summary stats.
    per_part_l2 = np.linalg.norm(pred_obj - gt_obj, axis=-1)        # (T, N_obj) in m
    per_frame_mean_mm   = per_part_l2.mean(axis=-1) * 1000.0
    per_frame_median_mm = np.median(per_part_l2, axis=-1) * 1000.0
    per_frame_p90_mm    = np.percentile(per_part_l2, 90, axis=-1) * 1000.0
    per_frame_max_mm    = per_part_l2.max(axis=-1) * 1000.0
    np.savez(out / "per_frame_l2.npz",
             mean_mm=per_frame_mean_mm,
             median_mm=per_frame_median_mm,
             p90_mm=per_frame_p90_mm,
             max_mm=per_frame_max_mm,
             per_part_l2_m=per_part_l2.astype(np.float32))
    print(f"  -> {out / 'per_frame_l2.npz'}  "
          f"(t=0/{T-1}: mean={per_frame_mean_mm[0]:.1f}/{per_frame_mean_mm[-1]:.1f} mm, "
          f"max={per_frame_max_mm[-1]:.1f} mm)")
    if not render_video:
        return

    err_mm = per_frame_mean_mm
    frame_dir = out / "frames"; frame_dir.mkdir(parents=True, exist_ok=True)

    for f in range(T):
        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        draw(ax1, pred_obj[f], gt_grip[f], lims,
             f"PRED  frame {f}   obj L2 {err_mm[f]:.1f} mm")
        draw(ax2, gt_obj[f], gt_grip[f], lims, f"GROUND TRUTH  frame {f}")
        fig.tight_layout()
        fig.savefig(frame_dir / f"{f:04d}.png", dpi=90)
        plt.close(fig)

    video = out / "rollout.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-framerate", str(fps), "-i", str(frame_dir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                   check=False)
    gif = out / "rollout.gif"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-framerate", str(fps), "-i", str(frame_dir / "%04d.png"),
                    "-vf", "scale=720:-1:flags=lanczos", str(gif)],
                   check=False)
    print(f"  -> {video}")
    print(f"  -> {gif}")

    # Compact strip at key frames
    keys = sorted({f for f in (0, T // 4, T // 2, 3 * T // 4, T - 1) if 0 <= f < T})
    fig = plt.figure(figsize=(3.5 * len(keys), 7))
    for j, f in enumerate(keys):
        a1 = fig.add_subplot(2, len(keys), j + 1, projection="3d")
        a2 = fig.add_subplot(2, len(keys), len(keys) + j + 1, projection="3d")
        draw(a1, pred_obj[f], gt_grip[f], lims,
             f"PRED f{f}   {err_mm[f]:.1f} mm")
        draw(a2, gt_obj[f], gt_grip[f], lims, f"GT f{f}")
    fig.tight_layout()
    fig.savefig(out / "strip.png", dpi=120)
    plt.close(fig)
    print(f"  -> {out / 'strip.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out",        required=True,
                   help="Output directory root (will create per-demo subdirs).")
    p.add_argument("--num-demos",  type=int, default=4)
    p.add_argument("--data-source", choices=("teleop", "pickup_source"),
                   default="teleop",
                   help="teleop: long-horizon T=201 demos in teleop_long_horizon_clean. "
                        "pickup_source: original T=40 batched tars under "
                        "robert_pickup_data_*_success.")
    p.add_argument("--source-root", default=SOURCE_ROOTS[0],
                   help="Only used when --data-source=pickup_source.")
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--stride",     type=int, default=PRED_FRAMES,
                   help="Frames to commit per rollout window (1..PRED_FRAMES).")
    p.add_argument("--num-frames", type=int, default=200,
                   help="How many frames to predict. Default 200 (full teleop horizon).")
    p.add_argument("--no-render", action="store_true",
                   help="Skip mp4/gif/strip rendering — only save per_frame_l2.npz "
                        "(much faster for large sweeps).")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    cli = parse_cli()
    out_root = Path(cli.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Reuse rollout_folding_viz.build_eval_args (takes checkpoint, data_dir,
    # device) — point data_dir at the converted pickup wds so norm_stats and
    # the model contract resolve correctly.
    data_dir = "/data/group_data/katefgroup/VLA/3DPCWM/pickup_data_pointworld"
    eval_args, ckpt = build_eval_args(cli.checkpoint, data_dir, cli.device)
    eval_args.domains = [DOMAIN]
    eval_args.batch_size = 1
    eval_args.seed = cli.seed
    print(f"robot_features : {eval_args.robot_features}")
    print(f"scene_features : {eval_args.scene_features}")
    print(f"grid_size      : {eval_args.grid_size}\n")

    # Pre-resolve demo paths
    if cli.data_source == "teleop":
        demo_paths = sorted(glob.glob(f"{TELEOP_ROOT}/env*_teleop_T200.pkl"))
        if not demo_paths:
            raise FileNotFoundError(f"no demos at {TELEOP_ROOT}/env*_teleop_T200.pkl")
        demo_paths = demo_paths[cli.start_sample:
                                cli.start_sample + cli.num_demos]
        if not demo_paths:
            raise IndexError(f"start_sample={cli.start_sample} past end of demo list")
    else:
        demo_paths = None  # use load_pickup_demo per-iter

    for k in range(cli.num_demos):
        if cli.data_source == "teleop":
            demo_path = demo_paths[k]
            print(f"\n=== teleop demo {Path(demo_path).name} ===")
            pp, oid = load_teleop_demo(demo_path)
            tag = Path(demo_path).stem
        else:
            sample_idx = cli.start_sample + k
            print(f"\n=== pickup_source sample_idx={sample_idx}  "
                  f"{cli.source_root.split('/')[-1]} ===")
            pp, oid = load_pickup_demo(cli.source_root, sample_idx, elem=0)
            tag = f"demo_{sample_idx:04d}"
        T_full = pp.shape[0]
        obj_traj   = pp[:, oid == 0]    # (T, 512, 3)
        robot_traj = pp[:, oid == 1]    # (T, ~646-671, 3)
        print(f"  loaded T={T_full}, n_obj={obj_traj.shape[1]}, n_grip={robot_traj.shape[1]}")

        if k == 0:
            # Build model from a peek batch (need feature dims for load_model)
            peek, _ = build_window_batch_pickup(
                obj_traj[0], robot_traj[:MODEL_T],
                eval_args.robot_features, eval_args.scene_features,
                cli.device,
            )
            data_info_dict = {
                "robot_features_dim": peek["robot_features"].shape[-1],
                "scene_features_dim": peek["scene_features"].shape[-1],
            }
            print(f"  robot_features_dim={data_info_dict['robot_features_dim']}  "
                  f"scene_features_dim={data_info_dict['scene_features_dim']}")
            model = load_model(eval_args, ckpt, data_info_dict, cli.device)
            model.eval()

        num_frames = min(cli.num_frames, T_full - 1)
        pred = rollout_pickup(
            model, obj_traj, robot_traj,
            eval_args.robot_features, eval_args.scene_features,
            num_frames=num_frames, stride=cli.stride,
            device=cli.device,
        )
        out = out_root / tag
        out.mkdir(parents=True, exist_ok=True)
        visualize_pickup(pred[: num_frames + 1],
                          obj_traj[: num_frames + 1],
                          robot_traj[: num_frames + 1],
                          out, render_video=not cli.no_render)


if __name__ == "__main__":
    main()
