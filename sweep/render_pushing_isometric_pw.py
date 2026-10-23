"""PointWorld port of pwm_clean/sweep/render_pushing_isometric.py.

PW saves block-pushing planner output as 2D xy trajectories
(imagined_block_xy, imagined_gripper_xy) + an init_block_quat, NOT as full
3D point clouds. To get the same visual style as the pwm_clean baseline we:

  1. Extract block + gripper *templates* (canonical local-frame point clouds,
     128 + 128 = 256 points) from a baseline ep's
     positions.npz['predicted_centered'] at the anchor frame.
  2. Pose those templates per-frame using:
        IMAGINED — translate by (imagined_block_xy, init_block_pos[2]) and
                   (imagined_gripper_xy, gripper_z_init); block rotation is
                   held at init_block_quat (PW doesn't predict it).
        SIM      — block: rotate+translate by sim_trajectory's
                   block_pos/block_quat; gripper: translate by gripper_xy.
  3. Reuse the baseline's axis limits + marker z so the two videos are
     directly visually comparable.

Generates per episode (in --out_dir):
   <ep>_final_imagined.mp4
   <ep>_final_sim.mp4

(The pwm_clean script also generates an _imagined_sequence.mp4 with multiple
optimization iterations — PW MPPI doesn't store iterations the same way, so
we skip that one.)

Usage:
    python sweep/render_pushing_isometric_pw.py \\
        --pw_root  /data/user_data/mbronars/packages/PointWorld/runs/seed42_calibrated \\
        --ref_root /data/group_data/katefgroup/datasets/mbronars/block_pushing_diff_priority_nopt1_pop3_L40S_random_seed42 \\
        --out_dir  /data/user_data/mbronars/packages/paper_final/video/block_pushing/PointWorld \\
        --episodes seed42_ep05 seed42_ep11 seed42_ep14 seed42_ep29 seed42_ep37
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.spatial.transform import Rotation

# Reuse the baseline helpers + constants. Override _render locally to drop
# title, legend, AND the block-centroid marker (paper-figure look).
PWM_CLEAN_SWEEP = Path("/data/user_data/mbronars/packages/pwm_clean/sweep")
sys.path.insert(0, str(PWM_CLEAN_SWEEP))
import render_pushing_isometric as _rpi  # noqa: E402
from render_pushing_isometric import (  # noqa: E402
    _save, _limits, _split,
    FPS, STRIDE, END_HOLD, N_HAND,
)
import matplotlib.pyplot as plt  # noqa: E402


def _render(block, gripper, target_xy, marker_z, lims, box, label):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=_rpi.ELEV, azim=_rpi.AZIM)
    ax.scatter(block[:, 0], block[:, 1], block[:, 2], s=8,
               c=_rpi.BLOCK_C, alpha=0.8, depthshade=True)
    if gripper is not None:
        ax.scatter(gripper[:, 0], gripper[:, 1], gripper[:, 2], s=6,
                   c=_rpi.GRIP_C, alpha=0.85, depthshade=True)
    # NO centroid marker (per user request for block-pushing)
    ax.scatter([target_xy[0]], [target_xy[1]], [marker_z], s=260, c="red", marker="*",
               alpha=0.55, edgecolors="black", linewidths=0.6, depthshade=False,
               zorder=11)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect(box)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # NO title, NO legend
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def _R(quat_wxyz):
    return Rotation.from_quat(np.asarray(quat_wxyz)[..., [1, 2, 3, 0]]).as_matrix()


def _baseline_templates(ref_ep: Path):
    """From a baseline ep, recover canonical block + gripper templates.

    Returns:
        block_local (n_block, 3)   block points in body frame (mean-centered)
        grip_local  (N_HAND, 3)    gripper points in local frame (mean-centered)
        block_center0 (3,)          block centroid in centered world at anchor
        grip_center0  (3,)          gripper centroid in centered world at anchor
        anchor (int)
    """
    pos = np.load(ref_ep / "positions.npz")
    committed = pos["predicted_centered"]                # (T, 256, 3) centered
    context_len = int(pos["context_len"])
    anchor = context_len - 1

    act = np.load(ref_ep / "actions.npz", allow_pickle=True)
    init_block_quat = np.asarray(act["init_block_quat"])

    grip_a, block_a = _split(committed[anchor])           # (128,3) each
    c_block0 = block_a.mean(0)
    # Undo the initial block rotation to get the body-frame template.
    local_block = (block_a - c_block0) @ _R(init_block_quat)
    c_grip0 = grip_a.mean(0)
    local_grip = grip_a - c_grip0
    return local_block, local_grip, c_block0, c_grip0, anchor


def _baseline_lims(ref_ep: Path):
    """Compute the exact axis lims the baseline pwm_clean render used."""
    pos = np.load(ref_ep / "positions.npz")
    committed = pos["predicted_centered"]
    offset = pos["offset_world"]
    context_len = int(pos["context_len"])

    act = np.load(ref_ep / "actions.npz", allow_pickle=True)
    target_xy = np.asarray(act["target_xy"], dtype=float) - offset[:2]

    # sim reconstruction (mirrors render_pushing_isometric._build_sim_clouds)
    sim = np.load(ref_ep / "replay_set_pose" / "sim_trajectory.npz", allow_pickle=True)
    anchor = context_len - 1
    grip_a, block_a = _split(committed[anchor])
    c_block0 = block_a.mean(0)
    local_block = (block_a - c_block0) @ _R(act["init_block_quat"])
    R_sim = _R(np.asarray(sim["block_quat"]))
    c_new = np.asarray(sim["block_pos"]) - offset
    sim_block = np.einsum("tij,pj->tpi", R_sim, local_block) + c_new[:, None, :]

    c_grip0 = grip_a.mean(0)
    local_grip = grip_a - c_grip0
    grip_xy = np.asarray(sim["gripper_xy"]) - offset[:2]
    T = grip_xy.shape[0]
    grip_c = np.concatenate([grip_xy, np.full((T, 1), c_grip0[2])], axis=1)
    sim_grip = local_grip[None] + grip_c[:, None, :]

    imag_g = committed[:, :N_HAND]; imag_b = committed[:, N_HAND:]
    # Ground plane = lowest z across all points across the full trajectory
    # (matches the visual floor in the isometric view). Puts the target star
    # on the table instead of floating at the block's mid-height.
    all_z = np.concatenate([
        imag_b[..., 2].ravel(), imag_g[..., 2].ravel(),
        sim_block[..., 2].ravel(), sim_grip[..., 2].ravel(),
    ])
    marker_z = float(all_z.min())
    tgt3 = np.array([target_xy[0], target_xy[1], marker_z])
    lims, box = _limits(
        [imag_b.reshape(-1, 3), imag_g.reshape(-1, 3),
         sim_block.reshape(-1, 3), sim_grip.reshape(-1, 3)], tgt3)
    return lims, box, marker_z, target_xy, offset


def _build_pw_clouds(pw_ep: Path, local_block, local_grip,
                    c_block0, c_grip0, offset, init_block_quat_baseline):
    """Reconstruct PW imagined + sim block+gripper point clouds.

    Returns (imag_block, imag_grip, sim_block, sim_grip, context_len) — all
    in the same CENTERED frame as the baseline lims (so the resulting clouds
    align with the precomputed lims).
    """
    act = np.load(pw_ep / "actions.npz", allow_pickle=True)
    init_block_pos  = np.asarray(act["init_block_pos"])
    init_block_quat = np.asarray(act["init_block_quat"])
    imagined_block_xy   = np.asarray(act["imagined_block_xy"])     # (T, 2)
    imagined_gripper_xy = np.asarray(act["imagined_gripper_xy"])   # (T, 2)
    T = imagined_block_xy.shape[0]

    # IMAGINED: block translates in xy, holds init_block_quat rotation;
    # block z stays at init_block_pos[2].
    R0 = _R(init_block_quat)                                       # (3,3)
    block_rotated = local_block @ R0.T                              # (n_b, 3)
    # Center the block at the new xy each frame, z = init block z (centered).
    init_block_z_centered = init_block_pos[2] - offset[2]
    bxyz = np.concatenate([
        imagined_block_xy - offset[:2],                              # (T, 2)
        np.full((T, 1), init_block_z_centered, dtype=np.float32),    # (T, 1)
    ], axis=1)                                                       # (T, 3)
    imag_block = block_rotated[None] + bxyz[:, None, :]              # (T, n_b, 3)

    # IMAGINED gripper: translate template by (imag_gripper_xy, c_grip0[2]).
    grip_z = c_grip0[2]
    gxyz = np.concatenate([
        imagined_gripper_xy - offset[:2],
        np.full((T, 1), grip_z, dtype=np.float32),
    ], axis=1)
    imag_grip = local_grip[None] + gxyz[:, None, :]                  # (T, n_g, 3)

    # SIM: block pose from sim_trajectory; gripper from sim_trajectory.gripper_xy.
    sim = np.load(pw_ep / "replay_set_pose" / "sim_trajectory.npz", allow_pickle=True)
    R_sim = _R(np.asarray(sim["block_quat"]))                        # (Ts, 3, 3)
    c_block_sim = np.asarray(sim["block_pos"]) - offset              # (Ts, 3)
    sim_block = np.einsum("tij,pj->tpi", R_sim, local_block) + c_block_sim[:, None, :]

    grip_xy_sim = np.asarray(sim["gripper_xy"]) - offset[:2]         # (Ts, 2)
    Ts = grip_xy_sim.shape[0]
    grip_xyz_sim = np.concatenate([grip_xy_sim,
                                    np.full((Ts, 1), grip_z, dtype=np.float32)], axis=1)
    sim_grip = local_grip[None] + grip_xyz_sim[:, None, :]            # (Ts, n_g, 3)

    # context_len for plotting (skip pre-context frames in imagined viz).
    plan_log = json.loads((pw_ep / "plan_log.json").read_text())
    context_len = int(plan_log.get("context_len", 2))
    return imag_block, imag_grip, sim_block, sim_grip, context_len


def _copy_sources(pw_ep: Path, out_dir: Path, ref_ep_name: str):
    """Co-locate the source NPZ files used to make the videos."""
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, dst_name in [
        (pw_ep / "actions.npz",                              f"{ref_ep_name}_actions.npz"),
        (pw_ep / "replay_set_pose" / "sim_trajectory.npz",   f"{ref_ep_name}_sim_trajectory.npz"),
        (pw_ep / "plan_log.json",                            f"{ref_ep_name}_plan_log.json"),
    ]:
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)


def render_episode(pw_ep: Path, ref_ep: Path, out_dir: Path):
    local_block, local_grip, c_block0, c_grip0, _ = _baseline_templates(ref_ep)
    lims, box, marker_z, target_xy, offset = _baseline_lims(ref_ep)
    init_block_quat = np.load(ref_ep / "actions.npz",
                              allow_pickle=True)["init_block_quat"]

    imag_b, imag_g, sim_b, sim_g, context_len = _build_pw_clouds(
        pw_ep, local_block, local_grip, c_block0, c_grip0, offset, init_block_quat,
    )

    # Name outputs after the baseline ep (seed42_epNN) so they pair 1:1 with
    # the baseline videos already in paper_final/video/block_pushing/.
    name = ref_ep.name

    # ── final imagined ────────────────────────────────────────────────
    frames = []
    for t in range(context_len, imag_b.shape[0], STRIDE):
        frames.append(_render(imag_b[t], imag_g[t], target_xy, marker_z, lims, box,
                              "Imagined: final plan (PointWorld)"))
    frames += [frames[-1]] * END_HOLD
    out_i = _save(frames, out_dir, f"{name}_final_imagined")

    # ── final sim ─────────────────────────────────────────────────────
    frames = []
    for t in range(sim_b.shape[0]):
        frames.append(_render(sim_b[t], sim_g[t], target_xy, marker_z, lims, box,
                              "Sim (sapien): final rollout (PointWorld)"))
    frames += [frames[-1]] * END_HOLD
    out_s = _save(frames, out_dir, f"{name}_final_sim")

    _copy_sources(pw_ep, out_dir, name)
    print(f"  [{name}] wrote {out_i.name}, {out_s.name}, + source NPZs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pw_root", required=True,
                   help="PointWorld run root (e.g. .../runs/seed42_calibrated)")
    p.add_argument("--ref_root", required=True,
                   help="Baseline pwm_clean run root with matching seed42_epXX/ dirs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--episodes", nargs="+", required=True,
                   help="Episode names — use baseline naming (seed42_epXX). "
                        "PW dir uses ep_0XX → script maps both.")
    args = p.parse_args()

    pw_root  = Path(args.pw_root)
    ref_root = Path(args.ref_root)
    out_dir  = Path(args.out_dir)
    print(f"pw_root  = {pw_root}")
    print(f"ref_root = {ref_root}")
    print(f"out_dir  = {out_dir}")

    for name in args.episodes:
        # baseline naming "seed42_ep05" → PW naming "ep_005"
        ep_num = int(name.split("ep")[-1])
        pw_ep = pw_root / f"ep_{ep_num:03d}"
        ref_ep = ref_root / name
        if not pw_ep.is_dir():
            print(f"  [WARN] no PW ep at {pw_ep}, skipping")
            continue
        if not ref_ep.is_dir():
            print(f"  [WARN] no ref ep at {ref_ep}, skipping")
            continue
        render_episode(pw_ep, ref_ep, out_dir)


if __name__ == "__main__":
    main()
