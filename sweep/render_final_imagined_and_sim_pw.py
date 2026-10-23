"""PointWorld port of pwm_efficient/sweep/render_final_imagined_and_sim.py.

Same output style (two standalone MP4s per episode: final-plan imagined +
Genesis-MPM sim rollout) but reads the PointWorld cc_pw output layout:

  ep_XXX/
    positions/batch000.npz   - predicted[0,0] is the full committed plan
                                (not lookaheads/ like pwm_efficient)
    replay/replay.pkl        - default Genesis playback (±10 cm bounds)
    replay_bigbounds/        - optional ±70 cm playback (T=150 only)

Uses the same _render helper from pwm_efficient/render_imagined_sequence so
visual style matches the existing paper figures exactly.

Usage:
    python render_final_imagined_and_sim_pw.py \\
        --sweep_root /data/group_data/katefgroup/datasets/mbronars/\\
                     cloud_centroid_pw_seed42_T150_K128_l40s/seed_42 \\
        --out_dir    /data/user_data/mbronars/packages/paper_final/\\
                     video/pile_manip/PointWorld \\
        --episodes ep_024 ep_036 ep_047 \\
        [--replay-subdir replay_bigbounds]   # default
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

# Reuse the rendering helper from pwm_efficient so visual style matches.
PWM_EFF_SWEEP = Path("/data/user_data/mbronars/packages/pwm_efficient/sweep")
sys.path.insert(0, str(PWM_EFF_SWEEP))
import render_imagined_sequence as _ris  # noqa: E402
from render_imagined_sequence import FPS, STRIDE, END_HOLD  # noqa: E402

# Local replacement for _render that drops the title + legend (clean look
# for paper figures). KEEPS the centroid marker — for the pile-manipulation
# task it's a useful visual reference for where the cloud's CoM is.
import matplotlib.pyplot as plt  # noqa: E402


def _render(pts, action_mask, show_grip, lims, box, label, target_xy, marker_z):
    cloud = pts[~action_mask]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=_ris.ELEV, azim=_ris.AZIM)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=4, c=_ris.CLOUD_C,
               alpha=0.75, depthshade=True)
    if show_grip:
        grip = pts[action_mask]
        ax.scatter(grip[:, 0], grip[:, 1], grip[:, 2], s=10, c=_ris.GRIP_C,
                   alpha=0.9, depthshade=True)
    c = cloud.mean(0)
    ax.plot([c[0], target_xy[0]], [c[1], target_xy[1]], [marker_z, marker_z],
            c="k", ls="--", lw=1.0, alpha=0.6, zorder=9)
    ax.scatter([c[0]], [c[1]], [marker_z], s=90, c="black", marker="o",
               edgecolors="white", linewidths=0.8, depthshade=False, zorder=10)
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

SIM_STRIDE = 25     # match pwm_efficient: 3750 sim substeps → 150 frames


def _baseline_lims(ref_ep: Path):
    """Compute axis lims using the *baseline* (pwm_efficient) episode's data
    so the PointWorld render uses the exact same camera bounds. Mirrors
    pwm_efficient/sweep/render_final_imagined_and_sim.py's lims calculation.
    """
    import json
    import pickle

    pos_npz = np.load(list((ref_ep / "positions").glob("*.npz"))[0])
    action_mask = pos_npz["action_mask"][0].astype(bool)
    cloud_mask = ~action_mask
    target_xy = np.asarray(
        json.loads((ref_ep / "replay" / "sim_centroid.json").read_text())["target_xy"],
        dtype=float)

    la_dir = sorted((ref_ep / "lookaheads").glob("batch*"))[0]
    entries = json.loads((la_dir / "manifest.json").read_text())["lookaheads"]
    final_entry = next(e for e in entries if e["is_final"])
    imag = np.load(la_dir / final_entry["file"])["positions"][0]   # (T,N,3)
    imag_cloud = imag[:, cloud_mask, :]
    imag_grip_raw = imag[:, action_mask, :]

    rep = pickle.load(open(ref_ep / "replay" / "replay.pkl", "rb"))
    local = _gripper_template(
        imag_grip_raw,
        np.asarray(rep["pred_pusher_pos"]),
        np.asarray(rep["pred_pusher_quat"]),
    )
    imag_grip = _pose_template(
        local, np.asarray(rep["pred_pusher_pos"]),
        np.asarray(rep["pred_pusher_quat"]),
    )
    sim_cloud = np.asarray(rep["sim_object_particles"])[::SIM_STRIDE]
    sim_grip = _pose_template(
        local,
        np.asarray(rep["sim_pusher_pos"])[::SIM_STRIDE],
        np.asarray(rep["sim_pusher_quat"])[::SIM_STRIDE],
    )

    stack = np.concatenate(
        [imag_cloud.reshape(-1, 3), imag_grip.reshape(-1, 3),
         sim_cloud.reshape(-1, 3), sim_grip.reshape(-1, 3)],
        axis=0,
    )
    marker_z = float(np.concatenate(
        [imag_cloud[..., 2].ravel(), sim_cloud[..., 2].ravel()]).mean())
    tgt3 = np.array([target_xy[0], target_xy[1], marker_z])
    stack = np.concatenate([stack, tgt3[None]], axis=0)
    mn, mx = stack.min(0), stack.max(0)
    pad = 0.05 * (mx - mn).max()
    lims = [(mn[i] - pad, mx[i] + pad) for i in range(3)]
    return lims, marker_z


def _R(quat_wxyz):
    return Rotation.from_quat(np.asarray(quat_wxyz)[..., [1, 2, 3, 0]]).as_matrix()


def _gripper_template(grip_pts, pos, quat_wxyz):
    """Recover canonical gripper template (local frame) by averaging over time."""
    R = _R(quat_wxyz)
    local = np.einsum("tji,tpj->tpi", R, grip_pts - pos[:, None, :])
    return local.mean(0)


def _pose_template(local, pos, quat_wxyz):
    R = _R(quat_wxyz)
    return np.einsum("tij,pj->tpi", R, local) + pos[:, None, :]


def _save(frames, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    gif = out_dir / f"{name}.gif"
    imageio.mimsave(str(gif), frames, fps=FPS, loop=0)
    mp4 = gif.with_suffix(".mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(gif),
                        "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-crf", "18", str(mp4)], check=True)
        gif.unlink()
        return mp4
    except Exception as e:
        print(f"    ffmpeg failed ({e}); keeping gif")
        return gif


def _copy_sources(ep: Path, replay_subdir: str, out_dir: Path):
    """Save copies of the source NPZ/PKL files used to generate this video,
    alongside the rendered MP4s, so a downstream consumer has everything.
    """
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    src_pos = ep / "positions" / "batch000.npz"
    src_pkl = ep / replay_subdir / "replay.pkl"
    src_score = ep / replay_subdir / "sim_centroid.json"
    if src_pos.exists():
        shutil.copy2(src_pos, out_dir / f"{ep.name}_positions.npz")
    if src_pkl.exists():
        shutil.copy2(src_pkl, out_dir / f"{ep.name}_replay.pkl")
    if src_score.exists():
        shutil.copy2(src_score, out_dir / f"{ep.name}_sim_centroid.json")


def render_episode(ep: Path, out_dir: Path, replay_subdir: str,
                   ref_lims=None, ref_marker_z=None):
    pos_npz = np.load(ep / "positions" / "batch000.npz")
    action_mask = pos_npz["action_mask"][0].astype(bool)        # (N,) True = gripper
    cloud_mask = ~action_mask
    context_len = int(pos_npz["context_len"])
    # Final committed imagined plan = predicted[0, 0]  (T_imag, N=1280, 3).
    imag = pos_npz["predicted"][0, 0]
    imag_cloud = imag[:, cloud_mask, :]                          # (T,1024,3)
    imag_grip_raw = imag[:, action_mask, :]                      # (T,256,3)

    # Target xy from the planner output (also stored in sim_centroid.json).
    target_xy = np.asarray(pos_npz["centroid_target_xy"], dtype=float)

    rep = pickle.load(open(ep / replay_subdir / "replay.pkl", "rb"))

    # Pose the gripper template (recovered from the imagined gripper) by both
    # the imagined pose and the sim pose, so both videos show the same
    # gripper *particles*, just at different poses.
    local = _gripper_template(
        imag_grip_raw,
        np.asarray(rep["pred_pusher_pos"]),
        np.asarray(rep["pred_pusher_quat"]),
    )
    imag_grip = _pose_template(
        local,
        np.asarray(rep["pred_pusher_pos"]),
        np.asarray(rep["pred_pusher_quat"]),
    )                                                            # (T,256,3)
    sim_cloud = np.asarray(rep["sim_object_particles"])[::SIM_STRIDE]    # (F,1024,3)
    sim_grip = _pose_template(
        local,
        np.asarray(rep["sim_pusher_pos"])[::SIM_STRIDE],
        np.asarray(rep["sim_pusher_quat"])[::SIM_STRIDE],
    )                                                            # (F,256,3)

    grip_mask = np.concatenate([np.zeros(imag_cloud.shape[1], bool),
                                np.ones(local.shape[0], bool)])

    if ref_lims is not None:
        # Reuse the BASELINE's axis bounds — direct visual comparability.
        # Pred/sim cloud or gripper that falls outside is just clipped, per
        # user request.
        lims = ref_lims
        marker_z = ref_marker_z
    else:
        # Fall back to computing from the PointWorld data itself.
        stack = np.concatenate(
            [imag_cloud.reshape(-1, 3), imag_grip.reshape(-1, 3),
             sim_cloud.reshape(-1, 3), sim_grip.reshape(-1, 3)],
            axis=0,
        )
        marker_z = float(np.concatenate(
            [imag_cloud[..., 2].ravel(), sim_cloud[..., 2].ravel()]).mean())
        tgt3 = np.array([target_xy[0], target_xy[1], marker_z])
        stack = np.concatenate([stack, tgt3[None]], axis=0)
        mn, mx = stack.min(0), stack.max(0)
        pad = 0.05 * (mx - mn).max()
        lims = [(mn[i] - pad, mx[i] + pad) for i in range(3)]
    rng = [l[1] - l[0] for l in lims]
    box = [r / max(rng) for r in rng]

    # ── Imagined ────────────────────────────────────────────────────────
    frames = []
    for t in range(context_len, imag_cloud.shape[0], STRIDE):
        pts = np.concatenate([imag_cloud[t], imag_grip[t]], axis=0)
        frames.append(_render(pts, grip_mask, True, lims, box,
                              "Imagined: final plan", target_xy, marker_z))
    frames += [frames[-1]] * END_HOLD
    out_i = _save(frames, out_dir, f"{ep.name}_final_imagined")

    # ── Sim ─────────────────────────────────────────────────────────────
    frames = []
    for t in range(sim_cloud.shape[0]):
        pts = np.concatenate([sim_cloud[t], sim_grip[t]], axis=0)
        frames.append(_render(pts, grip_mask, True, lims, box,
                              "Sim (Genesis MPM): final rollout", target_xy, marker_z))
    frames += [frames[-1]] * END_HOLD
    out_s = _save(frames, out_dir, f"{ep.name}_final_sim")

    _copy_sources(ep, replay_subdir, out_dir)
    print(f"  [{ep.name}] wrote {out_i.name}, {out_s.name}, + source NPZs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", required=True,
                   help="Path to .../<sweep>/seed_<S>/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--replay-subdir", default="replay_bigbounds",
                   help="Subdir under each ep to read replay.pkl from. "
                        "Use 'replay' for original ±10 cm playback, "
                        "'replay_bigbounds' for ±70 cm (default).")
    p.add_argument("--ref-sweep-root", default=None,
                   help="If given, axis limits are computed from this "
                        "baseline scenario root (one dir per episode under it) "
                        "rather than from the PointWorld data — so the "
                        "rendered video uses the EXACT same bounds as the "
                        "baseline for direct visual comparison.")
    args = p.parse_args()

    root = Path(args.sweep_root)
    out_dir = Path(args.out_dir)
    ref_root = Path(args.ref_sweep_root) if args.ref_sweep_root else None
    print(f"sweep_root    = {root}")
    print(f"out_dir       = {out_dir}")
    print(f"replay_subdir = {args.replay_subdir}")
    if ref_root:
        print(f"ref bounds    = from {ref_root}")
    for name in args.episodes:
        ref_lims = ref_marker_z = None
        if ref_root is not None:
            ref_ep = ref_root / name
            if not ref_ep.is_dir():
                print(f"  [WARN] no reference ep at {ref_ep} — computing lims from PW data")
            else:
                ref_lims, ref_marker_z = _baseline_lims(ref_ep)
                print(f"  [{name}] reusing baseline lims = "
                      f"{[(round(a,3), round(b,3)) for a,b in ref_lims]}")
        render_episode(root / name, out_dir, args.replay_subdir,
                       ref_lims=ref_lims, ref_marker_z=ref_marker_z)


if __name__ == "__main__":
    main()
