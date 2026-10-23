#!/usr/bin/env python3
"""
scripts/playback_floating.py

Playback and evaluation for block-pushing with the floating gripper.

Main entry point
----------------

evaluate_predictions(...)
   Takes predicted gripper actions and predicted block states, runs them in
   simulation, and returns per-step accuracy metrics and (optionally) a
   per-frame predicted-penetration mask.

   Always evaluates prediction accuracy.  Pass *target_xy* to also evaluate
   whether the block reached the commanded target.

   Inputs:
     initial_block_pos      (3,)    world-frame XYZ of the block at t=0
     initial_block_quat     (4,)    [w,x,y,z] block orientation at t=0
     initial_gripper_xy     (2,)    world-frame XY of the gripper at t=0
     predicted_actions      (T, 2)  gripper delta-XY actions to replay
     predicted_block_pos    (T, 3)  model's predicted block XYZ at each step
     predicted_block_quat   (T, 4)  model's predicted block orientation (optional)
     hand_particles_world   (T+1, N, 3)  full predicted gripper particle traj (optional)
     target_xy              (2,)    optional target block XY position

   Returns a dict with:
     actual_block_pos              (T, 3)
     actual_block_quat             (T, 4)
     position_errors               (T,)     L2 block position error (metres)
     yaw_errors                    (T,)     |Δyaw| per step (degrees)
     mean_position_error           float
     final_position_error          float
     max_position_error            float
     penetration_per_frame         (T,) bool or None
     penetration_frames_predicted  int or None
     out_of_bounds_steps           int
     trajectory_success            bool
     target_success                bool    (only when target_xy provided)
     min_distance_to_target        float   (only when target_xy provided)
     final_distance_to_target      float   (only when target_xy provided)

CLI usage
---------
    # Replay a single standardised NPZ; save video + GIF to out/
    python scripts/playback_floating.py --npz path/to/predictions.npz \\
        --output_dir out/

    # Process an entire directory of raw NPZ files
    python scripts/playback_floating.py --raw_npz_dir results/ \\
        --conversion_mode 3d --output_dir out/

    # Replay from HDF5 (no GIF — no particle data available)
    python scripts/playback_floating.py --h5 demos/all_demos.h5 --traj traj_0

Output file naming (when --output_dir is set)
---------------------------------------------
    {stem}_maxerr{max_pos_err_cm:.1f}cm.mp4
    {stem}_maxerr{max_pos_err_cm:.1f}cm.gif

GIF layout
----------
    1 row × 3 columns: Perspective | Top-down | Side
    GT     (simulation actual) : steelblue block + tomato gripper, alpha=0.25
    Pred   (model output)      : deepskyblue block + darkorange gripper, alpha=0.65
    Suptitle per frame shows block-pos error, yaw error, and penetration status.

NPZ format (for --npz / --npz_dir)
------------------------------------
    initial_block_pos       (3,)
    initial_block_quat      (4,)
    initial_gripper_xy      (2,)
    predicted_actions       (T, 2)
    predicted_block_pos     (T, 3)          optional
    predicted_block_quat    (T, 4)          optional
    hand_particles_world    (T+1, N, 3)     optional — produced by convert_from_3D
    hand_template_centered  (N, 3)          optional — produced by convert_from_3D
    target_xy               (2,)            optional
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import sapien
import gymnasium as gym
import torch

import envs  # noqa: F401 — registers PushBoundary + FloatingGripper

from utils.output_conversions import convert, CANONICAL_BLOCK_POS

from envs.push_boundary import (
    BOUNDARY_CENTER_X,
    BOUNDARY_CENTER_Y,
    BOUNDARY_HALF_X,
    BOUNDARY_HALF_Y,
    GRIPPER_Z_FIXED,
    OUT_MARGIN,
)

# Optional visualisation dependencies — graceful fallback if absent.
try:
    import matplotlib
    matplotlib.use("Agg")   # headless-safe; must precede pyplot import
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False
    
import json


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum block XY displacement (m) for a file to be included in aggregate stats.
_BLOCK_MOVE_THRESHOLD = 0.01   # 1 cm

# Block geometry used for penetration detection and cube-cloud reconstruction.
_BLOCK_HALF_SIZE = 0.025   # 2.5 cm  (cube side = 5 cm)
_PENETRATION_EPS = -0.002   # 2 mm slack — grazing contact is not penetration

# Min predicted block XY displacement per step to count as 'moving' (phantom-move check).
_BLOCK_MOVE_PER_STEP = 0.002  # 1 mm

# How far below the block bottom face the target plane sits in GIF visualisations.
_TARGET_Z_EPS = 0.002  # 2 mm

# Camera views for GIFs: (elevation_deg, azimuth_deg, label)
_VIEWS = [
    (25,  45,  "Perspective"),
    (90,   0,  "Top-down"),
    ( 5,  90,  "Side"),
]

_GT_ALPHA   = 0.25   # GT cloud alpha   — more transparent (reference background)
_PRED_ALPHA = 0.65   # Pred cloud alpha — less transparent (foreground subject)


# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _unwrap(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Extract yaw (radians) from a [w, x, y, z] quaternion."""
    q = np.asarray(q, dtype=np.float64).ravel()
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])   # scipy uses [x,y,z,w]
    return float(r.as_euler("xyz")[2])


def _quat_wxyz_to_sapien(q: np.ndarray) -> list:
    q = np.asarray(q, dtype=np.float64).ravel()
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _block_out_of_bounds(block_xy: np.ndarray) -> bool:
    x, y = float(block_xy[0]), float(block_xy[1])
    return (
        x < BOUNDARY_CENTER_X - BOUNDARY_HALF_X - OUT_MARGIN or
        x > BOUNDARY_CENTER_X + BOUNDARY_HALF_X + OUT_MARGIN or
        y < BOUNDARY_CENTER_Y - BOUNDARY_HALF_Y - OUT_MARGIN or
        y > BOUNDARY_CENTER_Y + BOUNDARY_HALF_Y + OUT_MARGIN
    )


def _block_moved(
    block_pos_seq: np.ndarray,
    initial_pos:   Optional[np.ndarray] = None,
    threshold:     float = _BLOCK_MOVE_THRESHOLD,
) -> bool:
    """True if block XY moves more than *threshold* metres over the trajectory."""
    if len(block_pos_seq) == 0:
        return False
    start = (np.asarray(initial_pos, dtype=np.float32)[:2]
             if initial_pos is not None else block_pos_seq[0, :2])
    return bool(np.linalg.norm(block_pos_seq[-1, :2] - start) > threshold)



def _mean_nn_dist(query_pts: np.ndarray, ref_pts: np.ndarray) -> float:
    """
    One-sided mean nearest-neighbour distance: for each point in *query_pts*,
    find the closest point in *ref_pts* and return the mean of those distances.
    """
    # (Q, R, 3) → (Q, R) → (Q,)
    diff  = query_pts[:, None, :] - ref_pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    return float(dists.min(axis=1).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Penetration helpers
# (prediction-space only — rigid-body physics guarantees no actual penetration,
# so checking the actual rollout is always zero by construction)
# ─────────────────────────────────────────────────────────────────────────────

def _max_pen_depth_per_frame(
    gripper_pts_seq: np.ndarray,   # (T, N, 3)
    block_pos_seq:   np.ndarray,   # (T, 3)
    block_quat_seq:  np.ndarray,   # (T, 4)  [w,x,y,z]
    half_size:       float = _BLOCK_HALF_SIZE,
) -> np.ndarray:                   # (T,) float32
    """
    Per-frame signed depth of the most-penetrating gripper particle.

    Positive = deepest particle is this far INSIDE the block surface.
    Negative = closest particle is this far OUTSIDE the block surface.

    Computed as max over all particles of
      min(half_size - |x_loc|, half_size - |y_loc|, half_size - |z_loc|)
    i.e. the L-inf signed distance to the box surface (positive = inside).
    """
    T      = len(gripper_pts_seq)
    depths = np.zeros(T, dtype=np.float32)
    for t in range(T):
        yaw = _yaw_from_quat_wxyz(block_quat_seq[t])
        cy, sy = np.cos(yaw), np.sin(yaw)
        delta  = gripper_pts_seq[t] - block_pos_seq[t]
        x_loc  =  cy * delta[:, 0] + sy * delta[:, 1]
        y_loc  = -sy * delta[:, 0] + cy * delta[:, 1]
        z_loc  =                          delta[:, 2]
        per_particle = np.minimum(
            half_size - np.abs(x_loc),
            np.minimum(half_size - np.abs(y_loc), half_size - np.abs(z_loc)),
        )
        depths[t] = per_particle.max()
    return depths


def _penetration_mask(
    gripper_pts_seq: np.ndarray,   # (T, N, 3)  predicted gripper cloud per step
    block_pos_seq:   np.ndarray,   # (T, 3)
    block_quat_seq:  np.ndarray,   # (T, 4)     [w,x,y,z]
    half_size:       float = _BLOCK_HALF_SIZE,
    eps:             float = _PENETRATION_EPS,
) -> np.ndarray:                   # (T,) bool
    """
    Per-frame boolean: True where the closest gripper particle is within
    eps of the block surface (depth > -eps).

    Negative eps tightens the threshold (particle must be eps inside the
    surface to flag); positive eps loosens it (particle flagged even when
    eps outside).  Only yaw rotation is applied.
    """
    return _max_pen_depth_per_frame(
        gripper_pts_seq, block_pos_seq, block_quat_seq, half_size
    ) > -eps


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _transform_template(
    template_centered: np.ndarray, # (N, 3)
    pos:               np.ndarray, # (3,)
    quat_wxyz:         np.ndarray  # (4,)
) -> np.ndarray:
    """Rotates and translates a centered point cloud template."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    # Scipy uses [x, y, z, w]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return (template_centered @ R.T + np.asarray(pos, dtype=np.float32)).astype(np.float32)


def _build_gt_clouds(
    actual_block_pos:       np.ndarray,            # (T, 3)
    actual_block_quat:      np.ndarray,            # (T, 4)
    initial_gripper_xy:     np.ndarray,            # (2,)
    actions:                np.ndarray,            # (T, 2)
    hand_template_centered: Optional[np.ndarray],  # (N, 3) or None
    block_template_centered: np.ndarray,            # (N, 3)
    block_half_size:        float = _BLOCK_HALF_SIZE,
    n_block_pts:            int   = 300,
) -> tuple:
    """
    Build per-step GT point clouds from simulation outputs.

    GT block cloud   : cube surface points at the simulated block pose.
    GT gripper cloud : centred template translated to the known actual centroid
                       (initial_gripper_xy + cumsum(actions)).  The floating
                       gripper never rotates, so translation is sufficient.
                       Returns None when hand_template_centered is unavailable.

    Returns
    -------
    gt_block_clouds   : list[np.ndarray (n_pts, 3)],  length T
    gt_gripper_clouds : list[np.ndarray (N, 3)] or None, length T
    """
    T      = len(actual_block_pos)
    ini_xy = np.asarray(initial_gripper_xy, dtype=np.float32)[:2]
    # Actual gripper centroid XY after each action (T, 2)
    grip_xy = ini_xy + np.cumsum(
        np.asarray(actions, dtype=np.float32)[:, :2], axis=0
    )
    
    gt_block_clouds = [
        _transform_template(
            block_template_centered,
            actual_block_pos[t], 
            actual_block_quat[t]
        )
        for t in range(T)
    ]

    gt_gripper_clouds = None
    if hand_template_centered is not None:
        tc = np.asarray(hand_template_centered, dtype=np.float32)
        gt_gripper_clouds = []
        for t in range(T):
            pts = tc.copy()
            pts[:, 0] += grip_xy[t, 0]
            pts[:, 1] += grip_xy[t, 1]
            # Template is centred at origin; add the fixed gripper height back.
            pts[:, 2] += GRIPPER_Z_FIXED
            gt_gripper_clouds.append(pts)

    return gt_block_clouds, gt_gripper_clouds

def create_comparison_gif(
    results:                dict,
    initial_gripper_xy:     np.ndarray,
    predicted_actions:      np.ndarray,
    predicted_block_pos:    np.ndarray,
    predicted_block_quat:   Optional[np.ndarray],
    hand_particles_world:   Optional[np.ndarray],   # (T+1, N, 3)
    hand_template_centered: Optional[np.ndarray],   # (N, 3)
    block_particles_world: Optional[np.ndarray],   # (T+1, N, 3)
    block_template_centered: Optional[np.ndarray],           # (N, 3)
    save_path:              str,
    fps:                    int   = 8,
    block_half_size:        float = _BLOCK_HALF_SIZE,
    n_block_surface_pts:    int   = 300,
    gt_alpha:               float = _GT_ALPHA,
    pred_alpha:             float = _PRED_ALPHA,
) -> None:
    """
    Save a multi-view GIF comparing GT (simulation actuals) vs Predicted
    (model output) particle clouds, frame by frame.

    ... [Docstring remains the same] ...
    """
    if not _VIZ_AVAILABLE:
        print("  GIF skipped — matplotlib and/or imageio not installed.")
        return

    T                 = len(results["actual_block_pos"])
    actual_block_pos  = results["actual_block_pos"]   # (T, 3)
    actual_block_quat = results["actual_block_quat"]  # (T, 4)
    pos_errors        = results["position_errors"]    # (T,)
    yaw_errors        = results["yaw_errors"]         # (T,)
    pen_per_frame     = results.get("penetration_per_frame")  # (T,) bool or None
    phantom_per_frame = results.get("phantom_move_per_frame")
    
    if phantom_per_frame is None:
        phantom_per_frame = np.zeros(T, dtype=bool)

    # ── GT clouds ─────────────────────────────────────────────────────────────
    gt_block_clouds, gt_gripper_clouds = _build_gt_clouds(
        actual_block_pos, actual_block_quat,
        initial_gripper_xy, predicted_actions,
        hand_template_centered, block_template_centered, block_half_size, n_block_surface_pts,
    )

    # ── Predicted block clouds ────────────────────────────────────────────────
    pred_block_clouds = None
    # if predicted_block_quat is not None:
    #     pred_block_clouds = [
    #         _make_cube_surface_pts(
    #             predicted_block_pos[t], predicted_block_quat[t],
    #             block_half_size, n_block_surface_pts,
    #         )
    #         for t in range(T)
    #     ]
    if block_particles_world is not None and block_particles_world.shape[0] >= T + 1:
        # Index 0 = frame before action; index t+1 = after action t.
        pred_block_clouds = [block_particles_world[t + 1] for t in range(T)]

    # ── Predicted gripper clouds ──────────────────────────────────────────────
    pred_gripper_clouds = None
    if hand_particles_world is not None and hand_particles_world.shape[0] >= T + 1:
        # Index 0 = frame before any action; index t+1 = after action t.
        pred_gripper_clouds = [hand_particles_world[t + 1] for t in range(T)]
        


    # ── Per-frame penetration ─────────────────────────────────────────────────
    if pen_per_frame is None and pred_gripper_clouds is not None \
            and predicted_block_quat is not None:
        pred_grip_arr = np.stack(pred_gripper_clouds, axis=0)   # (T, N, 3)
        pen_per_frame = _penetration_mask(
            pred_grip_arr,
            predicted_block_pos,
            np.asarray(predicted_block_quat, dtype=np.float32),
            half_size=block_half_size,
            eps=_PENETRATION_EPS,
        )
    if pen_per_frame is None:
        pen_per_frame = np.zeros(T, dtype=bool)

    # ── Target visualisation geometry (built once if target present) ──────────
    target_bbox_vis    = results.get("target_bbox_xy")      # (4, 2) or None
    target_part_vis    = results.get("target_particles_world")  # (N, 3) or None
    target_square_corners: Optional[list]   = None
    target_particles_arr:  Optional[np.ndarray] = None
    
    if target_bbox_vis is not None or target_part_vis is not None:
        z_block_bottom = float(actual_block_pos[:, 2].min()) - block_half_size
        z_tgt = z_block_bottom - _TARGET_Z_EPS

    # ────────────────────────────────────────────────────────────────────────
    # NEW LOGIC: Iterating over the 4 oriented corners instead of min/max flat array
    # ────────────────────────────────────────────────────────────────────────
    if target_bbox_vis is not None:
        target_square_corners = [
            [float(pt[0]), float(pt[1]), z_tgt] for pt in target_bbox_vis
        ]

    if target_part_vis is not None:
        target_particles_arr = np.asarray(target_part_vis, dtype=np.float32)

    # ── Global axis limits (cube aspect ratio, consistent across all frames) ──
    all_clouds: list = [np.array(gt_block_clouds).reshape(-1, 3)]
    if gt_gripper_clouds is not None:
        all_clouds.append(np.array(gt_gripper_clouds).reshape(-1, 3))
    if pred_block_clouds is not None:
        all_clouds.append(np.array(pred_block_clouds).reshape(-1, 3))
    if pred_gripper_clouds is not None:
        all_clouds.append(np.array(pred_gripper_clouds).reshape(-1, 3))
    all_pts = np.concatenate(all_clouds, axis=0)

    x_mid     = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
    y_mid     = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
    z_mid     = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2
    half_span = max(
        all_pts[:, 0].max() - all_pts[:, 0].min(),
        all_pts[:, 1].max() - all_pts[:, 1].min(),
        all_pts[:, 2].max() - all_pts[:, 2].min(),
    ) / 2 * 1.2
    x_lim = (x_mid - half_span, x_mid + half_span)
    y_lim = (y_mid - half_span, y_mid + half_span)
    z_lim = (z_mid - half_span, z_mid + half_span)

    # ── Render frames ─────────────────────────────────────────────────────────
    # ── Render frames ─────────────────────────────────────────────────────────
    n_views = len(_VIEWS)
    frames: list = []

    for t in range(T):
        pen     = bool(pen_per_frame[t])
        phantom = bool(phantom_per_frame[t])
        
        # Build a combined status string
        status_flags = []
        if pen:
            status_flags.append("⚠ PENETRATION")
        if phantom:
            status_flags.append("👻 PHANTOM MOVE")
            
        if not status_flags:
            status_str  = "Normal"
            title_color = "black"
            title_weight = "normal"
        else:
            status_str   = " & ".join(status_flags)
            title_weight = "bold"
            # Prioritize red for penetration, otherwise use purple for phantom
            title_color  = "red" if pen else "purple"

        suptitle = (
            f"Frame {t + 1}/{T}  │  "
            f"Block err: {pos_errors[t] * 100:.2f} cm  │  "
            f"Yaw err: {yaw_errors[t]:.1f}°  │  "
            f"Status: {status_str}"
        )

        fig, axes = plt.subplots(
            1, n_views,
            figsize=(5.5 * n_views, 4.8),
            subplot_kw={"projection": "3d"},
        )
        if n_views == 1:
            axes = [axes]

        fig.suptitle(suptitle, fontsize=9, color=title_color, fontweight=title_weight)

        for col, (elev, azim, view_label) in enumerate(_VIEWS):
            ax = axes[col]
            ax.view_init(elev=elev, azim=azim)

            # ── GT block ─────────────────────────────────────────────────────
            bp = gt_block_clouds[t]
            ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2],
                       c="steelblue", s=1.5, alpha=gt_alpha,
                       label="GT block", depthshade=False)

            # ── GT gripper ───────────────────────────────────────────────────
            if gt_gripper_clouds is not None:
                gp = gt_gripper_clouds[t]
                ax.scatter(gp[:, 0], gp[:, 1], gp[:, 2],
                           c="tomato", s=3, alpha=gt_alpha,
                           label="GT gripper", depthshade=False)

            # ── Predicted block ──────────────────────────────────────────────
            if pred_block_clouds is not None:
                pb = pred_block_clouds[t]
                ax.scatter(pb[:, 0], pb[:, 1], pb[:, 2],
                           c="deepskyblue", s=1.5, alpha=pred_alpha,
                           label="Pred block", depthshade=False)

            # # ── Predicted gripper ────────────────────────────────────────────
            # if pred_gripper_clouds is not None:
            #     pg        = pred_gripper_clouds[t]
            #     edge_col  = "red" if pen else None
            #     edge_lw   = 0.5   if pen else 0
            #     ax.scatter(pg[:, 0], pg[:, 1], pg[:, 2],
            #                c="darkorange", s=3, alpha=pred_alpha,
            #                edgecolors=edge_col, linewidths=edge_lw,
            #                label="Pred gripper", depthshade=False)

            # ── Target particles ─────────────────────────────────────────────
            # if target_particles_arr is not None:
            #     ax.scatter(
            #         target_particles_arr[:, 0],
            #         target_particles_arr[:, 1],
            #         target_particles_arr[:, 2],
            #         c="limegreen", s=2, alpha=0.25,
            #         label="Target", depthshade=False,
            #     )

            # ── Target bbox square (now an oriented polygon) ──────────────────
            # if target_square_corners is not None:
            #     poly = Poly3DCollection(
            #         [target_square_corners],
            #         facecolor="forestgreen", edgecolor="darkgreen",
            #         linewidth=0.2, alpha=0.1,
            #     )
            #     ax.add_collection3d(poly)

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.set_xlabel("X", fontsize=6)
            ax.set_ylabel("Y", fontsize=6)
            ax.set_zlabel("Z", fontsize=6)
            ax.tick_params(labelsize=5)
            ax.set_title(view_label, fontsize=8, pad=2)

            if col == n_views - 1:
                ax.legend(loc="upper right", fontsize=6, markerscale=2.5,
                          framealpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close(fig)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"  GIF  saved : {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_env(render: bool,
               target_xy:  Optional[np.ndarray] = None,
               video_dir:  Optional[str] = None,
               video_name: str = "playback"):
    needs_render = render or (video_dir is not None)

    kwargs: dict = dict(
        obs_mode     = "state_dict",
        control_mode = "floating_vel",
        render_mode  = "all" if needs_render else None,
        sim_backend  = "cpu",
        num_envs     = 1,
        robot_uids   = "floating_gripper",
    )
    if target_xy is not None:
        kwargs["target_xy"] = (float(target_xy[0]), float(target_xy[1]))

    env = gym.make("PushBoundary", **kwargs)

    if video_dir is not None:
        from mani_skill.utils.wrappers.record import RecordEpisode
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir      = video_dir,
            save_trajectory = False,
            save_video      = True,
            video_fps       = 20,
            trajectory_name = video_name,
        )

    return env


def _set_gripper_xy(base_env, gx: float, gy: float) -> None:
    robot = base_env.agent.robot
    try:
        n_dof = robot.dof if hasattr(robot, "dof") else len(robot.get_qpos())
        if hasattr(n_dof, "item"):
            n_dof = int(n_dof.item())
        qpos    = torch.zeros(n_dof, dtype=torch.float32)
        qpos[0] = gx - BOUNDARY_CENTER_X
        qpos[1] = gy - BOUNDARY_CENTER_Y
        robot.set_qpos(qpos.unsqueeze(0).expand(base_env.num_envs, -1))
        robot.set_qvel(
            torch.zeros_like(qpos).unsqueeze(0).expand(base_env.num_envs, -1)
        )
    except Exception:
        robot.set_pose(sapien.Pose(p=[gx, gy, GRIPPER_Z_FIXED], q=[1, 0, 0, 0]))


def _setup_episode(base_env,
                   initial_block_pos:  np.ndarray,
                   initial_block_quat: np.ndarray,
                   initial_gripper_xy: np.ndarray) -> None:
    block_pos  = np.asarray(initial_block_pos,  dtype=np.float32).ravel()
    block_quat = np.asarray(initial_block_quat, dtype=np.float32).ravel()
    gxy        = np.asarray(initial_gripper_xy, dtype=np.float32).ravel()

    base_env.block.set_pose(
        sapien.Pose(
            p=[float(block_pos[0]), float(block_pos[1]), float(block_pos[2])],
            q=_quat_wxyz_to_sapien(block_quat),
        )
    )
    try:
        base_env.block.set_velocity([0.0, 0.0, 0.0])
        base_env.block.set_angular_velocity([0.0, 0.0, 0.0])
    except Exception:
        pass

    _set_gripper_xy(base_env, float(gxy[0]), float(gxy[1]))


def _run_actions(env, base_env, actions: np.ndarray,
                 initial_gripper_xy: np.ndarray,
                 playback_mode: str = "action") -> tuple:
    """
    Step the environment through `actions` (T, 2) and collect per-step
    block positions and quaternions.

    Returns
    -------
    block_pos  (T, 3)
    block_quat (T, 4)  [w,x,y,z]
    oob_steps  int
    """
    T = len(actions)
    block_pos  = np.zeros((T, 3), dtype=np.float32)
    block_quat = np.zeros((T, 4), dtype=np.float32)
    oob_steps  = 0

    gripper_xy = np.asarray(initial_gripper_xy, dtype=np.float32).ravel()[:2].copy()

    for t in range(T):
        action = np.asarray(actions[t], dtype=np.float32).ravel()[:2]

        if playback_mode == "set_pose":
            gripper_xy = gripper_xy + action
            _set_gripper_xy(base_env, float(gripper_xy[0]), float(gripper_xy[1]))
            env.step(np.zeros(2, dtype=np.float32))
        else:
            env.step(action)

        bp = _to_np(base_env.block.pose.p).ravel()
        bq = _to_np(base_env.block.pose.q).ravel()
        block_pos[t]  = bp[:3]
        block_quat[t] = bq[:4]

        if _block_out_of_bounds(bp[:2]):
            oob_steps += 1

    return block_pos, block_quat, oob_steps


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


import time
import numpy as np
from typing import Optional
from matplotlib.path import Path as MplPath

def evaluate_predictions(
    initial_block_pos:      np.ndarray,
    initial_block_quat:     np.ndarray,
    initial_gripper_xy:     np.ndarray,
    predicted_actions:      np.ndarray,
    predicted_block_pos:    np.ndarray,
    predicted_block_quat:   Optional[np.ndarray] = None,
    # ── Particle extras (from convert_from_3D) ────────────────────────────────
    hand_particles_world:   Optional[np.ndarray] = None,   # (T+1, N, 3)
    block_particles_world:  Optional[np.ndarray] = None,   # (T+1, N, 3)
    block_template_centered: Optional[np.ndarray] = None,   # (N, 3)
    # ── Penetration tuning ────────────────────────────────────────────────────
    block_half_size:        float = _BLOCK_HALF_SIZE,
    penetration_eps:        float = _PENETRATION_EPS,
    # ── Target evaluation (optional) ──────────────────────────────────────────
    target_bbox_xy:         Optional[np.ndarray] = None,  # (4, 2) [4 corners of OBB]
    target_particles_world: Optional[np.ndarray] = None,  # (N, 3)
    # ── Rendering / recording ─────────────────────────────────────────────────
    render:                 bool = False,
    video_dir:              Optional[str] = None,
    video_name:             str = "playback",
    playback_mode:          str = "action",
    pause_on_done:          float = 0.0,
) -> dict:
    """
    Replay predicted gripper actions in simulation and compute evaluation metrics.
    
    ... [Docstring remains the same] ...
    """
    predicted_actions   = np.asarray(predicted_actions,   dtype=np.float32)
    predicted_block_pos = np.asarray(predicted_block_pos, dtype=np.float32)
    T = len(predicted_actions)

    # 1. Load bbox without flattening it to keep the (4, 2) shape
    bbox = (np.asarray(target_bbox_xy, dtype=np.float32)
            if target_bbox_xy is not None else None)

    # 2. Pass bbox centre to env so the target marker renders correctly
    # Taking the mean of the 4 corners gives the exact center of the OBB
    tgt_center = (bbox.mean(axis=0) if bbox is not None else None)

    env      = _build_env(render, target_xy=tgt_center, video_dir=video_dir, video_name=video_name)
    base_env = _unwrap(env)
    env.reset()
    _setup_episode(base_env, initial_block_pos, initial_block_quat, initial_gripper_xy)

    actual_pos, actual_quat, oob_steps = _run_actions(
        env, base_env, predicted_actions, initial_gripper_xy, playback_mode
    )

    if render and pause_on_done > 0:
        time.sleep(pause_on_done)
    env.close()

    # ── Block position errors ─────────────────────────────────────────────────
    pos_errors = np.linalg.norm(actual_pos - predicted_block_pos, axis=1)   # (T,)

    # ── Yaw errors ────────────────────────────────────────────────────────────
    yaw_errors = np.zeros(T, dtype=np.float32)
    if predicted_block_quat is not None:
        pq = np.asarray(predicted_block_quat, dtype=np.float32)
        for t in range(T):
            pred_yaw   = _yaw_from_quat_wxyz(pq[t])
            actual_yaw = _yaw_from_quat_wxyz(actual_quat[t])
            diff_deg   = abs(np.degrees(actual_yaw - pred_yaw))
            yaw_errors[t] = min(diff_deg, 360.0 - diff_deg)

    # ── Penetration + phantom moves (prediction space only) ──────────────────
    penetration_per_frame        = None
    penetration_frames_predicted = None
    mean_penetration_depth       = None
    max_penetration_depth        = None
    phantom_move_per_frame       = None
    phantom_move_frames          = None
    mean_phantom_dist            = None
    max_phantom_dist             = None

    if hand_particles_world is not None and predicted_block_quat is not None:
        hw = np.asarray(hand_particles_world, dtype=np.float32)
        pq = np.asarray(predicted_block_quat, dtype=np.float32)
        if hw.shape[0] >= T + 1:
            hw_steps  = hw[1: T + 1]   # (T, N, 3)

            depth_seq = _max_pen_depth_per_frame(
                hw_steps, predicted_block_pos, pq, half_size=block_half_size
            )

            penetration_per_frame        = depth_seq > -penetration_eps
            penetration_frames_predicted = int(penetration_per_frame.sum())
            if penetration_frames_predicted > 0:
                pen_depths             = depth_seq[penetration_per_frame]
                mean_penetration_depth = float(pen_depths.mean())
                max_penetration_depth  = float(pen_depths.max())

            block_disp = np.zeros(T, dtype=np.float32)
            if T > 1:
                block_disp[1:] = np.linalg.norm(
                    np.diff(predicted_block_pos, axis=0), axis=1
                )
            block_moving           = block_disp > _BLOCK_MOVE_PER_STEP
            not_in_contact         = depth_seq < (penetration_eps - 0.001)
            phantom_move_per_frame = block_moving & not_in_contact
            phantom_move_frames    = int(phantom_move_per_frame.sum())
            if phantom_move_frames > 0:
                phantom_dists     = -depth_seq[phantom_move_per_frame]  # positive
                mean_phantom_dist = float(phantom_dists.mean())
                max_phantom_dist  = float(phantom_dists.max())
                
        else:
            print(f"  Warning: hand_particles_world has {hw.shape[0]} frames "
                  f"but T+1={T + 1} needed; skipping penetration check.")

    result = dict(
        actual_block_pos             = actual_pos,
        actual_block_quat            = actual_quat,
        position_errors              = pos_errors,
        yaw_errors                   = yaw_errors,
        mean_position_error          = float(pos_errors.mean()),
        final_position_error         = float(pos_errors[-1]) if T > 0 else 0.0,
        max_position_error           = float(pos_errors.max()) if T > 0 else 0.0,
        penetration_per_frame        = penetration_per_frame,
        penetration_frames_predicted = penetration_frames_predicted,
        mean_penetration_depth       = mean_penetration_depth,
        max_penetration_depth        = max_penetration_depth,
        phantom_move_per_frame       = phantom_move_per_frame,
        phantom_move_frames          = phantom_move_frames,
        mean_phantom_dist            = mean_phantom_dist,
        max_phantom_dist             = max_phantom_dist,
        out_of_bounds_steps          = oob_steps,
        trajectory_success           = (oob_steps == 0),
    )

    # ── Target metrics (only when target_bbox_xy was supplied) ───────────────
    if bbox is not None:
        poly_path = MplPath(bbox)
        
        actual_target_success = False
        pred_target_success   = False
        first_success_step    = -1
        
        if T > 0:
            # 1. Actual final success (Using the frame-0 block template!)
            actual_final_cloud = _transform_template(
                block_template_centered, actual_pos[-1], actual_quat[-1]
            )
            actual_target_success = bool(poly_path.contains_points(actual_final_cloud[:, :2]).all())
            
            # 2. Predicted final success (Using the actual predicted block points!)
            if block_particles_world is not None and block_particles_world.shape[0] >= T:
                # We use the raw prediction for the final step
                pred_final_cloud = block_particles_world[-1] 
                # for pred target success only 95% of the points need to be in the target to account for prediction noise
                pred_target_success = np.sum(poly_path.contains_points(pred_final_cloud[:, :2])) >= 0.95 * len(pred_final_cloud)
                # pred_target_success = bool(poly_path.contains_points(pred_final_cloud[:, :2]).all())
            
            # 3. Keep tracking when the actual centroid first crossed into the box
            centroid_in_bbox = poly_path.contains_points(actual_pos[:, :2])
            first_success_step = int(np.argmax(centroid_in_bbox)) if centroid_in_bbox.any() else -1

        tgt_pts = (np.asarray(target_particles_world, dtype=np.float32)
                   if target_particles_world is not None else None)
        
        result.update(
            target_bbox_xy         = bbox,
            target_particles_world = tgt_pts,
            target_success         = actual_target_success,  # Kept for backward compatibility
            actual_target_success  = actual_target_success,
            pred_target_success    = pred_target_success,
            first_success_step     = first_success_step,
        )
        
        if tgt_pts is not None:
            # Use the transformed template for actual rollout distances
            actual_dists = np.array([
                _mean_nn_dist(
                    _transform_template(block_template_centered, actual_pos[t], actual_quat[t]),
                    tgt_pts,
                )
                for t in range(T)
            ], dtype=np.float32)
            
            result.update(
                actual_dist_to_target        = actual_dists,
                actual_mean_dist_to_target   = float(actual_dists.mean()),
                actual_final_dist_to_target  = float(actual_dists[-1]) if T > 0 else float("inf"),
                actual_min_dist_to_target    = float(actual_dists.min()),
            )

            if block_particles_world is not None and block_particles_world.shape[0] >= T + 1:
                # Use the raw dense predictions for predicted distances!
                pred_steps = block_particles_world[1 : T + 1]
                pred_dists = np.array([
                    _mean_nn_dist(pred_steps[t], tgt_pts) for t in range(T)
                ], dtype=np.float32)
                result.update(
                    pred_dist_to_target        = pred_dists,
                    pred_mean_dist_to_target   = float(pred_dists.mean()),
                    pred_final_dist_to_target  = float(pred_dists[-1]) if T > 0 else float("inf"),
                    pred_min_dist_to_target    = float(pred_dists.min()),
                )

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_h5_trajectory(h5_path: str, traj_key: str = "traj_0") -> dict:
    """Load a single trajectory from a RecordEpisode HDF5 file."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        traj        = f[traj_key]
        actions     = np.array(traj["actions"], dtype=np.float32)
        extra       = traj["obs"]["extra"]
        block_raw   = np.array(extra["block_pose"], dtype=np.float32)
        gripper_raw = np.array(extra["tcp_pose"],   dtype=np.float32)

    T = len(actions)
    return dict(
        actions            = actions,
        block_pos          = block_raw[:T, :3],
        block_quat         = block_raw[:T, 3:7],
        gripper_pos        = gripper_raw[:T, :3],
        initial_block_pos  = block_raw[0, :3],
        initial_block_quat = block_raw[0, 3:7],
        initial_gripper_xy = gripper_raw[0, :2],
    )


def load_npz_predictions(npz_path: str) -> dict:
    """
    Load a standardised predictions NPZ.  All arrays are returned; optional
    keys (hand_particles_world, hand_template_centered, target_xy, …) are
    included when present.
    """
    return dict(np.load(npz_path, allow_pickle=False))


def load_raw_npz(npz_path: str, conversion_mode: str) -> dict:
    """Load a raw model-output NPZ and convert it to the standardised playback dict."""
    raw = dict(np.load(npz_path, allow_pickle=True))
    for key in ("hand_id", "block_id"):
        if key in raw and isinstance(raw[key], np.ndarray) and raw[key].ndim == 0:
            raw[key] = raw[key].item()
    return convert(raw, conversion_mode)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_and_rename_video(
    video_dir:  str,
    tmp_name:   str,
    final_name: str,
) -> Optional[Path]:
    """
    Locate the MP4 file RecordEpisode wrote for *tmp_name* and rename it to
    *final_name*.  Returns the new path or None if no file was found.

    RecordEpisode typically writes `{output_dir}/{trajectory_name}.mp4`; the
    `{trajectory_name}_0.mp4` variant produced by some versions is also handled.
    """
    candidates = list(Path(video_dir).glob(f"{tmp_name}*.mp4"))
    if not candidates:
        return None
    src = max(candidates, key=lambda p: p.stat().st_mtime)
    dst = src.parent / (final_name + ".mp4")
    src.rename(dst)
    return dst


def _save_file_outputs(
    results:                dict,
    stem:                   str,
    output_dir:             Path,
    initial_gripper_xy:     np.ndarray,
    predicted_actions:      np.ndarray,
    predicted_block_pos:    np.ndarray,
    predicted_block_quat:   Optional[np.ndarray],
    hand_particles_world:   Optional[np.ndarray],
    hand_template_centered: Optional[np.ndarray],
    block_particles_world:   Optional[np.ndarray],
    block_template_centered: Optional[np.ndarray],
    gif_fps:                int = 8,
) -> None:
    """
    Rename the simulation video and create the comparison GIF for one file.
    Output filenames embed the max block position error so results are
    self-documenting at a glance.
    """
    err_cm  = results["max_position_error"] * 100
    n_pen   = results.get("penetration_frames_predicted") or 0
    n_phant = results.get("phantom_move_frames") or 0
    final_stem = f"{stem}_maxerr{err_cm:.1f}cm_pen{n_pen}_phant{n_phant}"

    # Rename the video that RecordEpisode wrote under the temp name
    renamed = _find_and_rename_video(str(output_dir), f"{stem}_tmp", final_stem)
    if renamed:
        print(f"  Video saved: {renamed}")
    else:
        print(f"  Warning: no video found for '{stem}_tmp' in {output_dir}")

    # Create the GT-vs-Pred comparison GIF
    gif_path = output_dir / f"{final_stem}.gif"
    create_comparison_gif(
        results                 = results,
        initial_gripper_xy      = initial_gripper_xy,
        predicted_actions       = predicted_actions,
        predicted_block_pos     = predicted_block_pos,
        predicted_block_quat    = predicted_block_quat,
        hand_particles_world    = hand_particles_world,
        hand_template_centered  = hand_template_centered,
        block_particles_world   = block_particles_world,
        block_template_centered = block_template_centered,
        save_path               = str(gif_path),
        fps                     = gif_fps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_results(results_list: list) -> dict:
    """Mean ± std for each scalar metric across all successfully evaluated files."""

    def _stat(values):
        arr = np.array(
            [v for v in values if v is not None and np.isfinite(float(v))],
            dtype=np.float64,
        )
        if len(arr) == 0:
            return dict(mean=float("nan"), std=float("nan"), n=0)
        return dict(mean=float(arr.mean()), std=float(arr.std(ddof=0)), n=len(arr))

    scalar_keys = [
        "mean_position_error",
        "final_position_error",
        "max_position_error",
        "penetration_frames_predicted",
        "mean_penetration_depth",
        "max_penetration_depth",
        "phantom_move_frames",
        "mean_phantom_dist",
        "max_phantom_dist",
        "out_of_bounds_steps",
        "actual_mean_dist_to_target",
        "actual_final_dist_to_target",
        "actual_min_dist_to_target",
        "pred_mean_dist_to_target",
        "pred_final_dist_to_target",
        "pred_min_dist_to_target",
    ]

    agg = {"n_files": len(results_list)}
    for k in scalar_keys:
        agg[k] = _stat([r.get(k) for r in results_list])

    agg["mean_yaw_error"] = _stat([
        float(r["yaw_errors"].mean())
        if r.get("yaw_errors") is not None and r["yaw_errors"].any()
        else None
        for r in results_list
    ])

    agg["trajectory_success_rate"] = float(
        np.mean([r["trajectory_success"] for r in results_list])
    )

    target_results = [r for r in results_list if "target_success" in r]
    if target_results:
        agg["actual_target_success_rate"] = float(
            np.mean([r["actual_target_success"] for r in target_results])
        )
        agg["pred_target_success_rate"] = float(
            np.mean([r["pred_target_success"] for r in target_results])
        )
        agg["n_target_files"] = len(target_results)

    return agg


def _process_npz_file(
    npz_path:             Path,
    conversion_mode:      Optional[str],
    render:               bool,
    playback_mode:        str,
    max_steps:            Optional[int],
    block_move_threshold: float = _BLOCK_MOVE_THRESHOLD,
    output_dir:           Optional[Path] = None,
    gif_fps:              int = 8,
) -> Optional[dict]:
    """
    Load, evaluate, filter, and (when output_dir is set) save video + GIF for
    a single NPZ file.

    Always evaluates prediction accuracy metrics.  Target metrics are included
    when target_bbox_xy is present in the file.

    Returns the results dict, or None when the file is skipped (block static
    in both trajectories, or an unrecoverable error occurred).
    """
    stem = npz_path.stem

    try:
        # ── Load ──────────────────────────────────────────────────────────────
        if conversion_mode is not None:
            data = load_raw_npz(str(npz_path), conversion_mode)
        else:
            data = load_npz_predictions(str(npz_path))
            data.setdefault("initial_block_pos",  CANONICAL_BLOCK_POS.copy())
            data.setdefault("initial_block_quat",
                            np.array([1., 0., 0., 0.], dtype=np.float32))
            data.setdefault("initial_gripper_xy", CANONICAL_BLOCK_POS[:2].copy())

        actions              = data["predicted_actions"]
        initial_block_pos    = data.get("initial_block_pos",  CANONICAL_BLOCK_POS.copy())
        initial_block_quat   = data.get("initial_block_quat", np.array([1., 0., 0., 0.]))
        initial_gripper_xy   = data.get("initial_gripper_xy", CANONICAL_BLOCK_POS[:2].copy())
        predicted_block_pos  = data.get("predicted_block_pos",  None)
        predicted_block_quat = data.get("predicted_block_quat", None)
        hand_particles_world    = data.get("hand_particles_world",   None)
        hand_template_centered  = data.get("hand_template_centered", None)
        block_particles_world   = data.get("block_particles_world",  None)
        block_template_centered = data.get("block_template_centered", None)
        
        if max_steps is not None:
            actions = actions[:max_steps]
            if predicted_block_pos  is not None:
                predicted_block_pos  = predicted_block_pos[:max_steps]
            if predicted_block_quat is not None:
                predicted_block_quat = predicted_block_quat[:max_steps]
            if hand_particles_world is not None:
                hand_particles_world = hand_particles_world[:max_steps + 1]

        if predicted_block_pos is None:
            predicted_block_pos = np.zeros((len(actions), 3), dtype=np.float32)

        T = len(actions)
        if T == 0:
            print(f"  [{stem}] Skipped — 0 actions.")
            return None

        # ── Evaluate ──────────────────────────────────────────────────────────
        # output_dir triggers rendering + video save; tmp name is renamed
        # after we learn the max error.
        vid_dir  = str(output_dir) if output_dir is not None else None
        vid_name = f"{stem}_tmp"

        results = evaluate_predictions(
            initial_block_pos      = initial_block_pos,
            initial_block_quat     = initial_block_quat,
            initial_gripper_xy     = initial_gripper_xy,
            predicted_actions      = actions,
            predicted_block_pos    = predicted_block_pos,
            predicted_block_quat   = predicted_block_quat,
            hand_particles_world    = hand_particles_world,
            block_particles_world   = block_particles_world,
            block_template_centered = block_template_centered,
            target_bbox_xy          = data.get("target_bbox_xy",         None),
            target_particles_world  = data.get("target_particles_world", None),
            render                  = render or (output_dir is not None),
            video_dir              = vid_dir,
            video_name             = vid_name,
            playback_mode          = playback_mode,
        )
        
        # ── Block-moved filter ────────────────────────────────────────────────
        pred_moved   = _block_moved(predicted_block_pos,
                                    initial_pos=initial_block_pos,
                                    threshold=block_move_threshold)
        actual_moved = _block_moved(results["actual_block_pos"],
                                    threshold=block_move_threshold)

        if not pred_moved and not actual_moved:
            pred_d = np.linalg.norm(
                predicted_block_pos[-1, :2]
                - np.asarray(initial_block_pos)[:2]
            ) * 100
            act_d = np.linalg.norm(
                results["actual_block_pos"][-1, :2]
                - results["actual_block_pos"][0, :2]
            ) * 100
            print(f"  [{stem}] Skipped — block static in both "
                  f"predicted ({pred_d:.1f} cm) and actual ({act_d:.1f} cm).")
            if output_dir is not None:
                for tmp in output_dir.glob(f"{stem}_tmp*.mp4"):
                    tmp.unlink(missing_ok=True)
            return None

        # ── Save video + GIF ──────────────────────────────────────────────────
        if output_dir is not None:
            _save_file_outputs(
                results                 = results,
                stem                    = stem,
                output_dir              = output_dir,
                initial_gripper_xy      = initial_gripper_xy,
                predicted_actions       = actions,
                predicted_block_pos     = predicted_block_pos,
                predicted_block_quat    = predicted_block_quat,
                hand_particles_world    = hand_particles_world,
                hand_template_centered  = hand_template_centered,
                block_particles_world   = block_particles_world,
                block_template_centered = block_template_centered,
                gif_fps                 = gif_fps,
            )

        results["filename"] = npz_path.name
        return results

    except Exception as exc:
        print(f"  [{stem}] Error: {exc}")
        return None


def process_npz_directory(
    npz_dir:              str,
    conversion_mode:      Optional[str] = None,
    render:               bool = False,
    playback_mode:        str = "action",
    max_steps:            Optional[int] = None,
    block_move_threshold: float = _BLOCK_MOVE_THRESHOLD,
    output_dir:           Optional[str] = None,
    gif_fps:              int = 8,
    max_demos:            Optional[int] = None,
) -> dict:
    """
    Process *.npz files in *npz_dir*, filter static-block trajectories,
    save per-file video + GIF (when output_dir is set), and return aggregate stats.

    Always evaluates prediction accuracy.  Target metrics are included per-file
    when target_bbox_xy is present in the file.

    If *max_demos* is set, files are shuffled and processing stops once that many
    valid (non-skipped) demos have been collected.
    """
    npz_files = sorted(Path(npz_dir).glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {npz_dir}")
        return {}

    if max_demos is not None:
        rng = np.random.default_rng()
        npz_files = rng.permutation(npz_files).tolist()

    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    total_files = len(npz_files)
    print(f"\nFound {total_files} NPZ files in {npz_dir}"
          + (f"  (target: {max_demos} valid demos)" if max_demos is not None else ""))
    print(f"  conversion={conversion_mode or 'standardised'}  "
          f"playback={playback_mode}  move_threshold={block_move_threshold * 100:.1f} cm"
          + (f"\n  output_dir={output_dir}" if output_dir else ""))
    print()

    all_results = []
    n_skipped   = 0

    for npz_file in npz_files:
        if max_demos is not None and len(all_results) >= max_demos:
            break
        i = len(all_results) + n_skipped + 1
        print(f"[{i}] {npz_file.name}")
        result = _process_npz_file(
            npz_path             = npz_file,
            conversion_mode      = conversion_mode,
            render               = render,
            playback_mode        = playback_mode,
            max_steps            = max_steps,
            block_move_threshold = block_move_threshold,
            output_dir           = out_path,
            gif_fps              = gif_fps,
        )
        if result is None:
            n_skipped += 1
            continue
        all_results.append(result)
        print_results(result)

    print(f"\n{'─' * 66}")
    print(f"  Files evaluated : {len(all_results)}")
    print(f"  Files skipped   : {n_skipped}  (static block or error)")
    print(f"{'─' * 66}")

    if not all_results:
        print("No files passed the block-moved filter — no aggregate stats.")
        return {}

    agg = _aggregate_results(all_results)
    print_aggregate_results(agg)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    T = len(results["position_errors"])
    print("\n── Prediction Accuracy ─────────────────────────────────────────")
    print(f"  Steps evaluated              : {T}")
    print(f"  Mean block pos error         : {results['mean_position_error'] * 100:.2f} cm")
    print(f"  Final block pos error        : {results['final_position_error'] * 100:.2f} cm")
    print(f"  Max block pos error          : {results['max_position_error'] * 100:.2f} cm")
    if results["yaw_errors"].any():
        print(f"  Mean yaw error               : {results['yaw_errors'].mean():.1f}°")
        print(f"  Max yaw error                : {results['yaw_errors'].max():.1f}°")
    if results.get("penetration_frames_predicted") is not None:
        n_pen = results['penetration_frames_predicted']
        print(f"  Penetration frames           : {n_pen}")
        if results.get("mean_penetration_depth") is not None:
            print(f"    Mean pen depth             : {results['mean_penetration_depth']*100:.2f} cm")
            print(f"    Max  pen depth             : {results['max_penetration_depth']*100:.2f} cm")
    if results.get("phantom_move_frames") is not None:
        n_ph = results['phantom_move_frames']
        print(f"  Phantom move frames          : {n_ph}")
        if results.get("mean_phantom_dist") is not None:
            print(f"    Mean gripper->block dist   : {results['mean_phantom_dist']*100:.2f} cm")
            print(f"    Max  gripper->block dist   : {results['max_phantom_dist']*100:.2f} cm")
    print(f"  Out-of-bounds steps          : {results['out_of_bounds_steps']}")
    print(f"  Boundary success             : {'✓' if results['trajectory_success'] else '✗'}")
    if "target_success" in results:
        print(f"\n── Target Evaluation ───────────────────────────────────────────")
        print(f"  Actual Target reached (bbox)        : {'✓' if results['actual_target_success'] else '✗'}")
        print(f"  Predicted Target reached (bbox)        : {'✓' if results['pred_target_success'] else '✗'}")
        if results["first_success_step"] >= 0:
            print(f"  First success at step        : {results['first_success_step']}")
        if "actual_mean_dist_to_target" in results:
            print(f"  Actual   mean dist→target    : {results['actual_mean_dist_to_target'] * 100:.2f} cm")
            print(f"  Actual   final dist→target   : {results['actual_final_dist_to_target'] * 100:.2f} cm")
            print(f"  Actual   min dist→target     : {results['actual_min_dist_to_target'] * 100:.2f} cm")
        if "pred_mean_dist_to_target" in results:
            print(f"  Predicted mean dist→target   : {results['pred_mean_dist_to_target'] * 100:.2f} cm")
            print(f"  Predicted final dist→target  : {results['pred_final_dist_to_target'] * 100:.2f} cm")
            print(f"  Predicted min dist→target    : {results['pred_min_dist_to_target'] * 100:.2f} cm")
    print("─" * 66)


def print_aggregate_results(agg: dict) -> None:
    def _row(label: str, stat: dict, scale: float = 1.0, unit: str = "") -> str:
        m = stat.get("mean", float("nan")) * scale
        s = stat.get("std",  float("nan")) * scale
        n = stat.get("n", 0)
        if np.isnan(m):
            return f"  {label:<32}: N/A"
        return f"  {label:<32}: {m:.4f} ± {s:.4f} {unit}  (n={n})"

    print(f"\n{'═' * 66}")
    print(f"  Aggregate Results  —  {agg['n_files']} files")
    print(f"{'═' * 66}")
    print(_row("Mean block pos error",        agg["mean_position_error"],  scale=100, unit="cm"))
    print(_row("Final block pos error",        agg["final_position_error"], scale=100, unit="cm"))
    print(_row("Max block pos error",          agg["max_position_error"],   scale=100, unit="cm"))
    if not np.isnan(agg["mean_yaw_error"].get("mean", float("nan"))):
        print(_row("Mean yaw error",           agg["mean_yaw_error"],                  unit="°"))
    if not np.isnan(agg["penetration_frames_predicted"].get("mean", float("nan"))):
        print(_row("Penetration frames",        agg["penetration_frames_predicted"],   unit="frames"))
        if not np.isnan(agg["mean_penetration_depth"].get("mean", float("nan"))):
            print(_row("  Mean pen depth",      agg["mean_penetration_depth"], scale=100, unit="cm"))
            print(_row("  Max  pen depth",      agg["max_penetration_depth"],  scale=100, unit="cm"))
    if not np.isnan(agg["phantom_move_frames"].get("mean", float("nan"))):
        print(_row("Phantom move frames",        agg["phantom_move_frames"],            unit="frames"))
        if not np.isnan(agg["mean_phantom_dist"].get("mean", float("nan"))):
            print(_row("  Mean phantom dist",   agg["mean_phantom_dist"],      scale=100, unit="cm"))
            print(_row("  Max  phantom dist",   agg["max_phantom_dist"],       scale=100, unit="cm"))
    print(_row("Out-of-bounds steps",          agg["out_of_bounds_steps"],             unit="steps"))
    print(f"  {'Boundary success rate':<32}: {agg['trajectory_success_rate'] * 100:.1f}%")
    if "actual_target_success_rate" in agg:
        n = agg["n_target_files"]
        print(f"\n── Target Evaluation  ({n} files with target) ──────────────────")
        print(f"  {'Actual target success rate (bbox)':<32}: {agg['actual_target_success_rate'] * 100:.1f}%")
        print(f"  {'Predicted target success rate (bbox)':<32}: {agg['pred_target_success_rate'] * 100:.1f}%")

        if not np.isnan(agg["actual_mean_dist_to_target"].get("mean", float("nan"))):
            print(_row("Actual   mean dist→target",  agg["actual_mean_dist_to_target"],  scale=100, unit="cm"))
            print(_row("Actual   final dist→target", agg["actual_final_dist_to_target"], scale=100, unit="cm"))
            print(_row("Actual   min dist→target",   agg["actual_min_dist_to_target"],   scale=100, unit="cm"))
        if not np.isnan(agg["pred_mean_dist_to_target"].get("mean", float("nan"))):
            print(_row("Predicted mean dist→target",  agg["pred_mean_dist_to_target"],  scale=100, unit="cm"))
            print(_row("Predicted final dist→target", agg["pred_final_dist_to_target"], scale=100, unit="cm"))
            print(_row("Predicted min dist→target",   agg["pred_min_dist_to_target"],   scale=100, unit="cm"))
    print(f"{'═' * 66}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Playback and evaluate floating-gripper push predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input source (mutually exclusive) ─────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz",         type=str, default=None,
                     help="Single standardised predictions NPZ.")
    src.add_argument("--raw_npz",     type=str, default=None,
                     help="Single raw model-output NPZ (auto-converted).")
    src.add_argument("--h5",          type=str, default=None,
                     help="HDF5 demo file (ground-truth actions replayed).")
    src.add_argument("--npz_dir",     type=str, default=None,
                     help="Directory of standardised NPZ files.")
    src.add_argument("--raw_npz_dir", type=str, default=None,
                     help="Directory of raw model-output NPZ files (auto-converted).")

    # ── Conversion ────────────────────────────────────────────────────────────
    p.add_argument("--conversion_mode", type=str, default="3d",
                   choices=["3d", "2d"],
                   help="Converter for --raw_npz / --raw_npz_dir.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    p.add_argument("--traj",           type=str,   default="traj_0")
    p.add_argument("--playback_mode",  type=str,   default="set_pose",
                   choices=["action", "set_pose"])
    p.add_argument("--block_move_threshold", type=float,
                   default=_BLOCK_MOVE_THRESHOLD,
                   help="Min block XY displacement (m) to include a file in "
                        "aggregate stats.  Files where BOTH trajectories are "
                        "below this threshold are excluded.")
    p.add_argument("--max_steps",      type=int,   default=None,
                   help="Truncate each trajectory to this many steps.")
    p.add_argument("--max_demos",      type=int,   default=50,
                   help="Stop after collecting this many valid (non-skipped) demos "
                        "from a directory.  Files are processed in random order.")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",  type=str, default=None,
                   help="Directory for per-file rendered video AND comparison GIF. "
                        "Files are named {stem}_maxerr{err_cm:.1f}cm.{mp4|gif}. "
                        "Rendering is automatically enabled when this is set.")
    p.add_argument("--video_dir",   type=str, default=None,
                   help="(Single-file) directory to save the rendered video only. "
                        "Use --output_dir to also generate a GIF.")
    p.add_argument("--render",      action="store_true",
                   help="Enable offscreen rendering (single-file mode, no output_dir).")
    p.add_argument("--gif_fps",     type=int,   default=8,
                   help="Frames per second for output GIFs.")
    p.add_argument("--pause",       type=float, default=0.0,
                   help="Seconds to pause after rollout (single-file mode).")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Directory modes ───────────────────────────────────────────────────────
    if args.npz_dir is not None or args.raw_npz_dir is not None:
        npz_dir         = args.npz_dir or args.raw_npz_dir
        conversion_mode = args.conversion_mode if args.raw_npz_dir else None

        # --video_dir acts as the output directory in directory mode when
        # --output_dir is not explicitly given; both video and GIF go there.
        effective_output_dir = args.output_dir or args.video_dir

        agg = process_npz_directory(
            npz_dir              = npz_dir,
            conversion_mode      = conversion_mode,
            render               = args.render,
            playback_mode        = args.playback_mode,
            max_steps            = args.max_steps,
            block_move_threshold = args.block_move_threshold,
            output_dir           = effective_output_dir,
            gif_fps              = args.gif_fps,
            max_demos            = args.max_demos,
        )
        
        # save aggregate stats to a JSON file in the output directory when processing a directory
        if effective_output_dir is not None:
            agg_stats_path = Path(effective_output_dir) / "aggregate_stats.json"
            with agg_stats_path.open("w") as f:
                json.dump(agg, f, indent=4)
            print(f"\nAggregate stats saved to {agg_stats_path}")
        
        return

    # ── Single-file modes ─────────────────────────────────────────────────────
    if args.h5 is not None:
        print(f"Loading H5: {args.h5}  traj={args.traj}")
        data = load_h5_trajectory(args.h5, args.traj)
        actions              = data["actions"]
        initial_block_pos    = data["initial_block_pos"]
        initial_block_quat   = data["initial_block_quat"]
        initial_gripper_xy   = data["initial_gripper_xy"]
        predicted_block_pos  = data["block_pos"]
        predicted_block_quat = data["block_quat"]
        hand_particles_world    = None
        hand_template_centered  = None
        target_bbox_xy = None
        stem = Path(args.h5).stem + f"_{args.traj}"

    elif args.raw_npz is not None:
        print(f"Loading raw NPZ: {args.raw_npz}  mode={args.conversion_mode}")
        data = load_raw_npz(args.raw_npz, args.conversion_mode)
        actions              = data["predicted_actions"]
        initial_block_pos    = data["initial_block_pos"]
        initial_block_quat   = data["initial_block_quat"]
        initial_gripper_xy   = data["initial_gripper_xy"]
        predicted_block_pos  = data.get("predicted_block_pos",  None)
        predicted_block_quat = data.get("predicted_block_quat", None)
        hand_particles_world    = data.get("hand_particles_world",    None)
        hand_template_centered  = data.get("hand_template_centered",  None)
        target_bbox_xy          = data.get("target_bbox_xy",          None)
        target_particles_world  = data.get("target_particles_world",  None)
        stem = Path(args.raw_npz).stem
        print(f"  Converted {args.conversion_mode.upper()} → {len(actions)} steps")

    else:  # --npz
        print(f"Loading NPZ: {args.npz}")
        data = load_npz_predictions(args.npz)
        actions = data["predicted_actions"]
        initial_block_pos  = CANONICAL_BLOCK_POS.copy()
        initial_block_quat = np.asarray(
            data.get("initial_block_quat", np.array([1., 0., 0., 0.])),
            dtype=np.float32,
        )
        initial_gripper_xy = np.asarray(
            data.get("initial_gripper_xy", CANONICAL_BLOCK_POS[:2].copy()),
            dtype=np.float32,
        )
        predicted_block_pos  = data.get("predicted_block_pos",  None)
        predicted_block_quat = data.get("predicted_block_quat", None)
        hand_particles_world    = data.get("hand_particles_world",   None)
        hand_template_centered  = data.get("hand_template_centered", None)
        target_bbox_xy = data.get("target_bbox_xy", None)
        stem = Path(args.npz).stem

    # ── Optional truncation ───────────────────────────────────────────────────
    if args.max_steps is not None:
        actions = actions[:args.max_steps]
        if predicted_block_pos  is not None:
            predicted_block_pos  = predicted_block_pos[:args.max_steps]
        if predicted_block_quat is not None:
            predicted_block_quat = predicted_block_quat[:args.max_steps]
        if hand_particles_world is not None:
            hand_particles_world = hand_particles_world[:args.max_steps + 1]

    if predicted_block_pos is None:
        predicted_block_pos = np.zeros((len(actions), 3), dtype=np.float32)

    print(f"  T={len(actions)} steps")
    print(f"  Block start  : {initial_block_pos}")
    print(f"  Gripper start: {initial_gripper_xy}")

    # --output_dir drives both video and GIF; --video_dir is video-only fallback
    out_path = Path(args.output_dir) if args.output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    vid_dir  = str(out_path) if out_path is not None else args.video_dir
    vid_name = f"{stem}_tmp" if out_path is not None else stem
    do_render = args.render or (out_path is not None)

    if target_bbox_xy is not None:
        print(f"  Target bbox XY : {target_bbox_xy}")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = evaluate_predictions(
        initial_block_pos      = initial_block_pos,
        initial_block_quat     = initial_block_quat,
        initial_gripper_xy     = initial_gripper_xy,
        predicted_actions      = actions,
        predicted_block_pos    = predicted_block_pos,
        predicted_block_quat   = predicted_block_quat,
        hand_particles_world   = hand_particles_world,
        target_bbox_xy         = target_bbox_xy,
        render                 = do_render,
        video_dir              = vid_dir,
        video_name             = vid_name,
        playback_mode          = args.playback_mode,
        pause_on_done          = args.pause,
    )
    print_results(results)

    if out_path is not None:
        _save_file_outputs(
            results                 = results,
            stem                    = stem,
            output_dir              = out_path,
            initial_gripper_xy      = initial_gripper_xy,
            predicted_actions       = actions,
            predicted_block_pos     = predicted_block_pos,
            predicted_block_quat    = predicted_block_quat,
            hand_particles_world    = hand_particles_world,
            hand_template_centered  = hand_template_centered,
            gif_fps                 = args.gif_fps,
        )


if __name__ == "__main__":
    main()