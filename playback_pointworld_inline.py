#!/usr/bin/env python3
"""
playback_pointworld_inline.py

Run the PointWorld model for one prediction window using actions from an HDF5
demo, then replay those same actions in the SAPIEN simulator and compare
predicted vs actual block motion.

Flow
----
  1. Load an HDF5 trajectory and extract one clip window of T frames.
  2. Build point-cloud data in the same format as h5_to_pointworld_wds
     (block surface points, gripper surface points, centering, normals, etc.).
  3. Write a temporary WebDataset shard so the existing dataloader pipeline
     handles feature extraction and normalisation identically to training.
  4. Load the PointWorld checkpoint and build the model via the Trainer infra.
  5. Run one forward pass → predicted scene_flows (block particle positions).
  6. Un-center predictions, recover block pose via Kabsch, extract actions.
  7. Step the SAPIEN simulator with the same actions.
  8. Compare and (optionally) visualise.

Usage
-----
    python playback_pointworld_inline.py \
        --model_path  checkpoints/model-last.pt \
        --h5          demos/PushBoundary/all_demos.h5 \
        --data_dir    data/push_pointworld \
        --traj        traj_0 \
        --start       0 \
        --output_dir  out/inline_eval

    Required: --data_dir must point to the training dataset root that contains
    norm_stats/norm_stats.json (produced by h5_to_pointworld_wds.py).
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch

# ── Local project imports ────────────────────────────────────────────────────
# Adjust sys.path if needed so the project root is importable.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# From the data-creation script (h5_to_pointworld_wds.py)
from vis_push_pcd_floating import (
    sample_templates,
    apply_pose,
    ACTOR_POS,
    ACTOR_QUAT,
    FLOAT_QPOS,
    FLOAT_ROOT_POS,
)
from h5_to_pointworld_wds import (
    block_normals_local,
    cylinder_normals_local,
    rotate_normals_world,
    quantize_normals,
    npy_bytes,
    IDENTITY_QUAT_WXYZ,
    DOMAIN,
)

# From the conversion / playback scripts
from output_conversions import (
    kabsch_batch,
    rotation_matrix_to_quaternion,
    _estimate_yaw_from_particles,
    CANONICAL_BLOCK_POS,
)
from playback_floating import (
    evaluate_predictions,
    create_comparison_gif,
    print_results,
    _build_gt_clouds,
    _BLOCK_HALF_SIZE,
)

# Model / training infrastructure
from arguments import parse_args as parse_train_args
from dataset_components.dataloader import build_dataloader
from training.trainer import Trainer
from pointworld.checkpoint_contract import (
    apply_model_contract_to_args,
    read_checkpoint_contract,
    train_domains_from_data_contract,
)
from arguments import DOMAIN_TO_DATA_DIR

try:
    import webdataset as wds
except ImportError:
    raise ImportError("webdataset is required: pip install webdataset")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Extract one clip from HDF5
# ─────────────────────────────────────────────────────────────────────────────

def extract_clip_from_h5(
    h5_path: str,
    traj_key: str,
    start: int,
    clip_len: int,
    n_block: int,
    n_hand: int,
    shape: str = "cube",
    subsample: int = 1,
    seed: int = 0,
) -> dict:
    """
    Pull one clip window from the HDF5 file and build the raw arrays that
    h5_to_pointworld_wds would write into a tar shard.

    Returns a dict with:
        block_world          (T, N_block, 3)  centred block positions
        block_normals        (T, N_block, 3)  world-frame normals
        hand_world           (T, N_hand,  3)  centred hand positions
        hand_normals         (T, N_hand,  3)  world-frame normals
        joint_pos            (T, 2)           prismatic joint positions
        grip_pose            (T, 7)           centred TCP pose
        grip_open            (T, 1)           zeros
        visibility           (T, N_block)     all-true
        center_shift         (3,)             translation applied (for un-centering)
        block_states_raw     (T, 13)          original block actor states (for sim init)
        tcp_poses_raw        (T, 7)           original (un-centred) TCP poses
        actions              (T-1, 2)         delta-XY gripper actions
        initial_block_pos    (3,)             world-frame block pos at t=0
        initial_block_quat   (4,)             [w,x,y,z] block quat at t=0
        initial_gripper_xy   (2,)             world-frame gripper XY at t=0
    """
    rng = np.random.default_rng(seed)
    T = clip_len
    use_T = (shape == "T")

    with h5py.File(h5_path, "r") as f:
        traj = f[traj_key]

        # ── Auto-discover key names (same logic as h5_to_pointworld_wds) ──
        art_keys   = list(traj["env_states/articulations"].keys())
        actor_keys = list(traj["env_states/actors"].keys())

        _robot_cands = ["floating_gripper", "robot", "panda_stick", "gripper"]
        art_key = next(
            (k for c in _robot_cands for k in art_keys if c in k.lower()),
            art_keys[0],
        )
        _block_cands = ["push_block", "block", "cube"]
        block_actor_key = next(
            (k for c in _block_cands for k in actor_keys if c in k.lower()),
            actor_keys[0],
        )

        block_states = traj[f"env_states/actors/{block_actor_key}"][:]
        art_states   = traj[f"env_states/articulations/{art_key}"][:]

    # Skip lag frame (same as data creation)
    block_states = block_states[1:]
    art_states   = art_states[1:]

    # Temporal subsampling
    sub          = slice(None, None, subsample)
    block_states = block_states[sub]
    art_states   = art_states[sub]

    Tsub = block_states.shape[0]
    assert start + clip_len <= Tsub, (
        f"Clip [{start}:{start+clip_len}] exceeds trajectory length {Tsub} "
        f"(after subsampling by {subsample})."
    )

    bs_clip  = block_states[start : start + clip_len]   # (T, 13)
    art_clip = art_states[start : start + clip_len]      # (T, 17)

    # ── TCP from articulation state ──────────────────────────────────────
    root_pos          = art_clip[:, FLOAT_ROOT_POS].copy()
    joint_xy          = art_clip[:, FLOAT_QPOS]
    root_pos[:, 0]   += joint_xy[:, 0]
    root_pos[:, 1]   += joint_xy[:, 1]
    tcp_quat          = np.tile(IDENTITY_QUAT_WXYZ, (T, 1))
    tcp_poses         = np.concatenate([root_pos, tcp_quat], axis=1)  # (T, 7)

    # ── Sample point-cloud templates ─────────────────────────────────────
    templates = sample_templates(n_block, n_hand, rng)
    block_tpl = templates["T"] if use_T else templates["cube"]  # (N_block, 3)
    hand_tpl  = templates["hand"]                                # (N_hand,  3)

    blk_norm_loc  = block_normals_local(block_tpl, shape)
    hand_norm_loc = cylinder_normals_local(hand_tpl)

    # ── Build world-space trajectories ───────────────────────────────────
    block_world      = np.empty((T, n_block, 3), dtype=np.float32)
    hand_world       = np.empty((T, n_hand,  3), dtype=np.float32)
    block_quats_wxyz = np.empty((T, 4),          dtype=np.float32)

    for t in range(T):
        bs  = bs_clip[t]
        tcp = tcp_poses[t]
        block_world[t]      = apply_pose(block_tpl, bs[ACTOR_POS], bs[ACTOR_QUAT])
        hand_world[t]       = apply_pose(hand_tpl,  tcp[:3],       tcp[3:7])
        block_quats_wxyz[t] = bs[ACTOR_QUAT]

    # ── Centre (mirror pointworld's center_shift) ────────────────────────
    shift = np.concatenate(
        [block_world[0], hand_world[0]], axis=0
    ).mean(0).astype(np.float32)

    block_world -= shift
    hand_world  -= shift
    tcp_centered       = tcp_poses.copy()
    tcp_centered[:, :3] -= shift

    # ── World-space normals ──────────────────────────────────────────────
    block_normals_w = rotate_normals_world(blk_norm_loc,  block_quats_wxyz)
    hand_normals_w  = rotate_normals_world(hand_norm_loc, tcp_poses[:, 3:7])

    # ── Joint positions ──────────────────────────────────────────────────
    joint_positions = art_clip[:, FLOAT_QPOS].astype(np.float32)

    # ── Gripper fields ───────────────────────────────────────────────────
    right_gripper_pose = tcp_centered.astype(np.float32)
    right_gripper_open = np.zeros((T, 1), dtype=np.float32)
    visibility         = np.ones((T, n_block), dtype=bool)

    # ── Actions: delta-XY from raw (un-centred) TCP ──────────────────────
    tcp_xy_raw = tcp_poses[:, :2].astype(np.float32)      # (T, 2)
    actions    = np.diff(tcp_xy_raw, axis=0)               # (T-1, 2)

    # ── Initial state for the simulator ──────────────────────────────────
    initial_block_pos  = bs_clip[0, ACTOR_POS].astype(np.float32)      # (3,)
    initial_block_quat = bs_clip[0, ACTOR_QUAT].astype(np.float32)     # (4,) wxyz
    initial_gripper_xy = tcp_poses[0, :2].astype(np.float32)           # (2,)

    return dict(
        # Arrays to write into the WebDataset shard
        block_world      = block_world,
        block_normals    = block_normals_w,
        hand_world       = hand_world,
        hand_normals     = hand_normals_w,
        joint_pos        = joint_positions,
        grip_pose        = right_gripper_pose,
        grip_open        = right_gripper_open,
        visibility       = visibility,
        # Bookkeeping for un-centering and sim
        center_shift     = shift,
        block_states_raw = bs_clip,
        tcp_poses_raw    = tcp_poses,
        actions          = actions,
        initial_block_pos  = initial_block_pos,
        initial_block_quat = initial_block_quat,
        initial_gripper_xy = initial_gripper_xy,
        # Templates (needed for Kabsch / visualisation)
        block_template   = block_tpl,
        hand_template    = hand_tpl,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Write a temporary WebDataset shard
# ─────────────────────────────────────────────────────────────────────────────

def write_temp_shard(clip: dict, out_dir: Path, include_normals: bool = False) -> None:
    """
    Write the clip as a single-sample WebDataset tar shard in out_dir/test/.
    Also writes metadata_rank0.json so the dataloader can find it.
    """
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    tar_path = str(test_dir / "test_00000.tar")
    writer = wds.TarWriter(tar_path)
    sample_fields = {
        "__key__": "test_00000000",
        "view0_scene_flows.npy":      npy_bytes(clip["block_world"]),
        "view0_scene_visibility.npy": npy_bytes(clip["visibility"]),
        "joint_positions.npy":        npy_bytes(clip["joint_pos"]),
        "right_gripper_pose.npy":     npy_bytes(clip["grip_pose"]),
        "right_gripper_open.npy":     npy_bytes(clip["grip_open"]),
    }
    if include_normals:
        sample_fields["view0_scene_normals.npy"] = npy_bytes(
            quantize_normals(clip["block_normals"])
        )
    writer.write(sample_fields)
    writer.close()

    # Minimal metadata so the dataloader knows how many samples exist.
    meta = {"train": {"processed_count": 0}, "test": {"processed_count": 1}}
    (out_dir / "metadata_rank0.json").write_text(json.dumps(meta, indent=2))


def symlink_norm_stats(src_data_dir: Path, tmp_data_dir: Path) -> None:
    """
    Symlink (or copy) the norm_stats directory from the real training dataset
    into the temporary dataset so the dataloader picks up the correct
    normalisation parameters.
    """
    src = src_data_dir / "norm_stats"
    dst = tmp_data_dir / "norm_stats"
    if dst.exists():
        return
    if src.is_dir():
        # Prefer symlink; fall back to copy on platforms that don't support it.
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copytree(str(src), str(dst))
    else:
        raise FileNotFoundError(
            f"norm_stats not found at {src}. "
            f"Make sure --data_dir points to the dataset root produced by "
            f"h5_to_pointworld_wds.py."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load model and run inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_infer(
    model_path: str,
    tmp_data_dir: str,
    real_data_dir: str,
    extra_args: list[str] | None = None,
) -> tuple[dict, dict, object]:
    """
    Load the PointWorld checkpoint, build the dataloader over the temporary
    single-sample dataset, run one forward pass, and return:
        (batch_np, outputs_np, args)
    where batch_np / outputs_np have numpy arrays on CPU.

    *real_data_dir* is used to resolve the training-domain data directory
    (required by the Trainer init for non-eval domains); the eval domain
    itself is pointed at *tmp_data_dir*.
    """
    # ── Build a synthetic argv for the Trainer ───────────────────────────
    # The Trainer / Tester reads args from the command line; we construct
    # the minimal set needed for inference.
    argv = [
        "--model_path", model_path,
        "--distributed", "false",
        "--eval_num_batches", "1",
        "--seed", "42",
    ]
    if extra_args:
        argv.extend(extra_args)

    # Parse using the project's argument parser.  This gives us all the
    # defaults that the Trainer expects.
    prev_argv = sys.argv
    sys.argv = ["playback_pointworld_inline.py"] + argv
    try:
        args = parse_train_args()
    finally:
        sys.argv = prev_argv

    args.model_path = model_path
    args.distributed = False

    # ── Read checkpoint contract to get model config + training domains ───
    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_contract, data_contract = read_checkpoint_contract(
        checkpoint, context="inline playback"
    )
    apply_model_contract_to_args(
        args, model_contract, context="inline playback",
        explicit_cli_dests=set(),
    )
    train_domains = train_domains_from_data_contract(data_contract, context="inline playback")
    if not train_domains:
        raise RuntimeError("Checkpoint has no training domains.")

    # Point every domain at its real data dir, except we will override the
    # eval split to use the temp shard.
    args.domains = list(train_domains)
    args.data_dirs = []
    for dom in args.domains:
        if dom == DOMAIN:
            # For the pushing domain, use the temp dir so the test shard is ours.
            args.data_dirs.append(tmp_data_dir)
        elif dom in DOMAIN_TO_DATA_DIR:
            args.data_dirs.append(DOMAIN_TO_DATA_DIR[dom])
        else:
            # Fallback: use real_data_dir (may not be correct for all domains).
            args.data_dirs.append(real_data_dir)

    # ── Build model (reusing Trainer init logic) ─────────────────────────
    os.environ["WANDB_MODE"] = "disabled"
    trainer = Trainer(args, inference_only=True, data_info_dict=None)

    # Load weights from the checkpoint we already parsed.
    trainer.model.load_state_dict(checkpoint["model"], strict=False)
    trainer.model.eval()
    device = trainer.device

    # ── Build dataloader for the temp test shard ─────────────────────────
    # We override eval to use the "test" split of the temp dataset.
    dl, info = build_dataloader(
        args, mode="test", rank=0, world_size=1,
        override_splits=["test"],
    )
    assert dl is not None, "Dataloader returned None for the temp shard."

    # ── Forward pass ─────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in dl:
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = trainer.model(batch, training=False)
            break  # only need one batch (one sample)

    # Move everything to numpy for downstream use.
    batch_np = {
        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    outputs_np = {
        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in outputs.items()
    }
    return batch_np, outputs_np, args


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Convert model predictions → world-frame block poses + actions
# ─────────────────────────────────────────────────────────────────────────────

def predictions_to_world(
    outputs_np: dict,
    batch_np: dict,
    center_shift: np.ndarray,
    clip: dict,
) -> dict:
    """
    Take the model's predicted scene_flows (centred block particle positions)
    and convert them to world-frame block positions, orientations, and
    gripper actions suitable for evaluate_predictions().

    Returns a dict with:
        predicted_block_pos     (T-1, 3)
        predicted_block_quat    (T-1, 4)  [w,x,y,z]
        predicted_actions       (T-1, 2)  delta-XY gripper
        block_particles_world   (T, N_block, 3)
        hand_particles_world    (T, N_hand, 3)
        block_template_centered (N_block, 3)
        hand_template_centered  (N_hand, 3)
    """
    # The model output scene_flows has shape (B, T, N_block, 3) in centred coords.
    # We take sample 0.
    pred_scene = outputs_np["scene_flows"][0]   # (T, N_block, 3)  centred
    T, N_block = pred_scene.shape[:2]

    # Un-center: add back the center_shift applied during data construction.
    pred_scene_world = pred_scene + center_shift  # (T, N_block, 3)

    # ── Block pose via Kabsch ────────────────────────────────────────────
    block_centroids = pred_scene_world.mean(axis=1)              # (T, 3)
    block_centred   = pred_scene_world - block_centroids[:, None, :]

    # Estimate initial yaw from frame-0 particle distribution.
    yaw_0 = _estimate_yaw_from_particles(block_centred[0], half_size=_BLOCK_HALF_SIZE)
    c0, s0 = np.cos(yaw_0), np.sin(yaw_0)
    R0 = np.array([[c0, -s0, 0.0],
                    [s0,  c0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float32)

    Rs_rel, _ = kabsch_batch(block_centred[0], block_centred)
    Rs_abs    = np.stack([Rs_rel[t] @ R0 for t in range(T)])

    block_quats = np.stack([rotation_matrix_to_quaternion(Rs_abs[t])
                            for t in range(T)])  # (T, 4)

    # ── Hand (gripper) particles un-centred ──────────────────────────────
    # The hand trajectory is known (GT conditioning), so we just un-center it.
    hand_world_centred = clip["hand_world"]                      # (T, N_hand, 3)
    hand_world         = hand_world_centred + center_shift       # (T, N_hand, 3)
    hand_centroids     = hand_world.mean(axis=1)                 # (T, 3)

    # ── Actions: delta-XY of gripper centroid ────────────────────────────
    actions = np.diff(hand_centroids[:, :2], axis=0).astype(np.float32)  # (T-1, 2)

    # ── Templates centred at origin (for playback GIF reconstruction) ────
    block_template_centered = (block_centred[0] @ R0).astype(np.float32)
    hand_template_centered  = (
        clip["hand_template"] - clip["hand_template"].mean(axis=0)
    ).astype(np.float32)

    return dict(
        predicted_block_pos     = block_centroids[1:].astype(np.float32),   # (T-1, 3)
        predicted_block_quat    = block_quats[1:].astype(np.float32),       # (T-1, 4)
        predicted_actions       = actions,                                    # (T-1, 2)
        block_particles_world   = pred_scene_world.astype(np.float32),       # (T, N_block, 3)
        hand_particles_world    = hand_world.astype(np.float32),             # (T, N_hand, 3)
        block_template_centered = block_template_centered,
        hand_template_centered  = hand_template_centered,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main: tie it all together
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="PointWorld inline playback: model prediction vs SAPIEN simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", required=True,
                   help="Path to trained PointWorld checkpoint (.pt)")
    p.add_argument("--h5", required=True,
                   help="Path to PushBoundary HDF5 demo file")
    p.add_argument("--data_dir", required=True,
                   help="Path to the training dataset root (must contain norm_stats/)")
    p.add_argument("--traj", default="traj_0",
                   help="Trajectory key inside the HDF5 file")
    p.add_argument("--start", type=int, default=0,
                   help="Starting frame index within the (subsampled) trajectory")
    p.add_argument("--clip_len", type=int, default=11,
                   help="Clip length T (must match training config)")
    p.add_argument("--n_block", type=int, default=128,
                   help="Number of block surface points (must match training)")
    p.add_argument("--n_hand", type=int, default=128,
                   help="Number of gripper surface points (must match training)")
    p.add_argument("--shape", default="cube", choices=["cube", "T"],
                   help="Block geometry type")
    p.add_argument("--subsample", type=int, default=1,
                   help="Temporal subsampling factor for raw trajectory")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for template sampling")
    p.add_argument("--playback_mode", default="set_pose",
                   choices=["action", "set_pose"],
                   help="How to replay actions in the simulator")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for video + GIF outputs")
    p.add_argument("--render", action="store_true",
                   help="Enable rendering even without --output_dir")
    p.add_argument("--gif_fps", type=int, default=8)
    p.add_argument("--include_normals", action="store_true",
                   help="Emit view0_scene_normals.npy in the temp shard "
                        "(default: off, model must not require scene_normals).")
    p.add_argument("--extra_args", nargs="*", default=[],
                   help="Extra arguments forwarded to the Trainer arg parser "
                        "(e.g. --log_dir /tmp/logs)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)

    # ── Step 1: Extract clip from HDF5 ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  HDF5       : {args.h5}")
    print(f"  Trajectory : {args.traj}")
    print(f"  Start frame: {args.start}  Clip length: {args.clip_len}")
    print(f"{'─'*60}\n")

    clip = extract_clip_from_h5(
        h5_path   = args.h5,
        traj_key  = args.traj,
        start     = args.start,
        clip_len  = args.clip_len,
        n_block   = args.n_block,
        n_hand    = args.n_hand,
        shape     = args.shape,
        subsample = args.subsample,
        seed      = args.seed,
    )
    print(f"Extracted clip: block_world {clip['block_world'].shape}, "
          f"hand_world {clip['hand_world'].shape}")
    print(f"Center shift : {clip['center_shift']}")
    print(f"Initial block: {clip['initial_block_pos']}")
    print(f"Initial grip : {clip['initial_gripper_xy']}")
    print(f"Actions shape: {clip['actions'].shape}")

    # ── Step 2: Write temporary WebDataset shard ─────────────────────────
    tmp_root = Path(tempfile.mkdtemp(prefix="pw_inline_"))
    print(f"\nTemp dataset dir: {tmp_root}")
    try:
        write_temp_shard(clip, tmp_root, include_normals=args.include_normals)
        symlink_norm_stats(data_dir, tmp_root)

        # ── Step 3: Load model + inference ───────────────────────────────
        print("\nLoading model and running inference …")
        batch_np, outputs_np, train_args = load_model_and_infer(
            model_path   = args.model_path,
            tmp_data_dir = str(tmp_root),
            real_data_dir = str(data_dir),
            extra_args   = args.extra_args or None,
        )

        print(f"Model output scene_flows shape: {outputs_np['scene_flows'].shape}")

        # ── Step 4: Convert predictions to world frame ───────────────────
        preds = predictions_to_world(
            outputs_np, batch_np, clip["center_shift"], clip,
        )
        print(f"Predicted block pos shape : {preds['predicted_block_pos'].shape}")
        print(f"Predicted actions shape   : {preds['predicted_actions'].shape}")

        # ── Step 5: Run the same actions in SAPIEN ───────────────────────
        # We use the *original* actions from the HDF5 (not the model's
        # gripper predictions, since the model is action-conditioned and
        # the gripper trajectory is GT conditioning).
        sim_actions = clip["actions"]  # (T-1, 2)
        T_pred = len(sim_actions)

        out_path = Path(args.output_dir) if args.output_dir else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        vid_dir  = str(out_path) if out_path else None
        vid_name = f"{Path(args.h5).stem}_{args.traj}_s{args.start}_tmp"
        do_render = args.render or (out_path is not None)

        print(f"\nReplaying {T_pred} actions in SAPIEN …")
        results = evaluate_predictions(
            initial_block_pos       = clip["initial_block_pos"],
            initial_block_quat      = clip["initial_block_quat"],
            initial_gripper_xy      = clip["initial_gripper_xy"],
            predicted_actions       = sim_actions,
            predicted_block_pos     = preds["predicted_block_pos"],
            predicted_block_quat    = preds["predicted_block_quat"],
            hand_particles_world    = preds["hand_particles_world"],
            block_particles_world   = preds["block_particles_world"],
            block_template_centered = preds["block_template_centered"],
            render                  = do_render,
            video_dir               = vid_dir,
            video_name              = vid_name,
            playback_mode           = args.playback_mode,
        )

        # ── Step 6: Print results ────────────────────────────────────────
        print_results(results)

        # ── Step 7: Save GIF (optional) ──────────────────────────────────
        if out_path is not None:
            stem = f"{Path(args.h5).stem}_{args.traj}_s{args.start}"
            err_cm  = results["max_position_error"] * 100
            n_pen   = results.get("penetration_frames_predicted") or 0
            n_phant = results.get("phantom_move_frames") or 0
            final_stem = f"{stem}_maxerr{err_cm:.1f}cm_pen{n_pen}_phant{n_phant}"

            # Rename the temp video
            from scripts.playback_floating import _find_and_rename_video
            renamed = _find_and_rename_video(str(out_path), vid_name.rstrip("_tmp"), final_stem)
            # Also try the exact temp name
            if not renamed:
                renamed = _find_and_rename_video(str(out_path), vid_name, final_stem)
            if renamed:
                print(f"  Video saved: {renamed}")

            gif_path = out_path / f"{final_stem}.gif"
            create_comparison_gif(
                results                 = results,
                initial_gripper_xy      = clip["initial_gripper_xy"],
                predicted_actions       = sim_actions,
                predicted_block_pos     = preds["predicted_block_pos"],
                predicted_block_quat    = preds["predicted_block_quat"],
                hand_particles_world    = preds["hand_particles_world"],
                hand_template_centered  = preds["hand_template_centered"],
                block_particles_world   = preds["block_particles_world"],
                block_template_centered = preds["block_template_centered"],
                save_path               = str(gif_path),
                fps                     = args.gif_fps,
            )

    finally:
        # Clean up the temp dataset directory.
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\nCleaned up temp dir: {tmp_root}")

    print("\nDone.")


if __name__ == "__main__":
    main()