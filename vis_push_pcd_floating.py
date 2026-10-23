#!/usr/bin/env python3
"""
vis_push_pcd.py

Shared geometry / point-cloud utilities for PushBoundary demos collected with
the FloatingGripper robot.

Floating-gripper geometry (from URDF)
--------------------------------------
  Cylinder: radius=0.01 m, length=0.12 m, axis along Z (upright).
  The cylinder is centred at the TCP origin in the gripper link frame —
  no additional Z offset needed.

Block geometry
--------------
  Cube: half-extent = 0.025 m on all sides (CUBE_HALF from push_boundary.py).

H5 articulation state layout (FloatingGripper, n_dof=2)
---------------------------------------------------------
  ManiSkill saves the articulation state as:
    root_pos  (3)   — anchored at (BCX, BCY, gripper_z)
    root_quat (4)   — always identity for this robot
    qpos      (2)   — [joint_x, joint_y]
    qvel      (2)   — joint velocities
    lin_vel   (3)   — root linear velocity
    ang_vel   (3)   — root angular velocity
    total     17

  TCP world position = root_pos + [qpos[0], qpos[1], 0]
  (root quaternion is identity, so no rotation needed)

H5 actor state layout (push_block)
------------------------------------
    pos       (3)
    quat_wxyz (4)
    lin_vel   (3)
    ang_vel   (3)
    total     13
"""

from __future__ import annotations

import time

import numpy as np
import fpsample


# ─────────────────────────────────────────────────────────────────────────────
# H5 state slices
# ─────────────────────────────────────────────────────────────────────────────

ACTOR_POS  = slice(0, 3)
ACTOR_QUAT = slice(3, 7)

# FloatingGripper articulation state
# FLOAT_ROOT_POS  = slice(0, 3)    # world pos of the root link
# FLOAT_ROOT_QUAT = slice(3, 7)    # root quaternion (identity in practice)
# FLOAT_QPOS      = slice(7, 9)    # [joint_x, joint_y]

FLOAT_ROOT_POS = slice(0, 3)     # Correct
FLOAT_ROOT_QUAT = slice(3, 7)    # root quaternion (identity in practice)
FLOAT_QPOS     = slice(13, 15)   # Update from slice(7, 9)
FLOAT_QVEL     = slice(15, 17)   # If you need velocities


# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants — must match push_boundary.py and the floating_gripper URDF
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_HALF      = 0.025   # cube half-extent (metres)

GRIPPER_RADIUS  = 0.01    # cylinder radius  (from URDF)
GRIPPER_LENGTH  = 0.12    # cylinder length  (from URDF)
# The URDF places the cylinder centred at the gripper link origin with rpy=0 0 0,
# so the cylinder axis is along Z and its centre is at the TCP.  No Z offset.

# T-block geometry (kept for completeness; not used by default)
_T_box1_hw = 0.10 / 2
_T_box1_hh = 0.025 / 2
_T_com_y   = 0.0375 / 2
_T_half_t  = 0.02
T_BOXES = [
    (np.array([0.0, -_T_com_y, 0.0]),               np.array([_T_box1_hw, _T_box1_hh, _T_half_t])),
    (np.array([0.0, 4*_T_box1_hh - _T_com_y, 0.0]), np.array([_T_box1_hh, (3/4)*_T_box1_hw, _T_half_t])),
]

# Viewer colours
BLOCK_COLOR = np.array([0.20, 0.47, 0.96])   # blue
HAND_COLOR  = np.array([0.91, 0.47, 0.10])   # orange


# ─────────────────────────────────────────────────────────────────────────────
# FloatingGripperFK — trivial; just reads joint values from articulation state
# ─────────────────────────────────────────────────────────────────────────────

class FloatingGripperFK:
    """
    Computes the TCP world position for a 2-DOF prismatic floating gripper.

    The robot root is anchored at a fixed world pose.  The two prismatic joints
    (joint_x, joint_y) translate the gripper link in X and Y respectively.

    TCP = root_pos + [qpos[0], qpos[1], 0]

    No URDF or pinocchio required.
    """

    def tcp_pose(self, art_state: np.ndarray):
        """
        Parameters
        ----------
        art_state : (17,) float array — one row from the articulation h5 dataset

        Returns
        -------
        pos  : (3,) float32 — TCP world position
        quat : (4,) float32 — TCP orientation (w,x,y,z) — always identity
        """
        root_pos = art_state[FLOAT_ROOT_POS].astype(np.float32)
        qpos     = art_state[FLOAT_QPOS].astype(np.float32)
        tcp_pos  = root_pos + np.array([qpos[0], qpos[1], 0.0], dtype=np.float32)
        tcp_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return tcp_pos, tcp_quat


# ─────────────────────────────────────────────────────────────────────────────
# Local-frame surface samplers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_box(offset: np.ndarray, half: np.ndarray,
                n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points on the surface of an axis-aligned box."""
    n_initial = n * 20
    hx, hy, hz = half
    areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
    counts = np.round(areas / areas.sum() * n_initial).astype(int)
    counts[-1] += n_initial - counts.sum()
    pts = []
    if counts[0] > 0:
        signs = rng.choice([-1.0, 1.0], counts[0])
        pts.append(np.stack([signs*hx,
                             rng.uniform(-hy, hy, counts[0]),
                             rng.uniform(-hz, hz, counts[0])], 1))
    if counts[1] > 0:
        signs = rng.choice([-1.0, 1.0], counts[1])
        pts.append(np.stack([rng.uniform(-hx, hx, counts[1]),
                             signs*hy,
                             rng.uniform(-hz, hz, counts[1])], 1))
    if counts[2] > 0:
        signs = rng.choice([-1.0, 1.0], counts[2])
        pts.append(np.stack([rng.uniform(-hx, hx, counts[2]),
                             rng.uniform(-hy, hy, counts[2]),
                             signs*hz], 1))
    pts_all = np.concatenate(pts).astype(np.float32) + offset.astype(np.float32)
    return pts_all[fpsample.fps_sampling(pts_all, n)]


def _sample_cylinder(radius: float, length: float,
                     n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n points on the surface of a cylinder whose axis is along Z,
    centred at the origin (spans z ∈ [-length/2, +length/2]).
    """
    n_initial = n * 20
    lat_a  = 2 * np.pi * radius * length
    cap_a  = 2 * np.pi * radius ** 2
    total  = lat_a + cap_a
    n_lat  = max(1, int(n_initial * lat_a / total)) if total > 0 else 0
    n_cap  = n_initial - n_lat
    n_top, n_bot = n_cap // 2, n_cap - n_cap // 2

    pts = []
    if n_lat > 0:
        th = rng.uniform(0, 2*np.pi, n_lat)
        z  = rng.uniform(-length/2, length/2, n_lat)
        pts.append(np.stack([radius*np.cos(th), radius*np.sin(th), z], 1))
    for nz, z0 in [(n_top, length/2), (n_bot, -length/2)]:
        if nz > 0:
            r  = radius * np.sqrt(rng.uniform(0, 1, nz))
            th = rng.uniform(0, 2*np.pi, nz)
            pts.append(np.stack([r*np.cos(th), r*np.sin(th), np.full(nz, z0)], 1))

    pts_all = np.concatenate(pts).astype(np.float32)
    return pts_all[fpsample.fps_sampling(pts_all, n)]


# ─────────────────────────────────────────────────────────────────────────────
# Template sampler — call once per trajectory for varied point patterns
# ─────────────────────────────────────────────────────────────────────────────

def sample_templates(n_block: int, n_hand: int,
                     rng: np.random.Generator) -> dict[str, np.ndarray]:
    """
    Sample fresh local-frame surface point clouds for one trajectory.

    Call this once per trajectory (not once globally) so each trajectory
    has a differently sampled template, improving geometric diversity.

    Returns
    -------
    dict with keys:
      "cube"  : (n_block, 3) float32 — cube surface in local frame
      "T"     : (n_T,    3) float32 — T-block surface in local frame
      "hand"  : (n_hand, 3) float32 — gripper cylinder in local frame
    """
    cube_tpl = _sample_box(np.zeros(3), np.full(3, BLOCK_HALF), n_block, rng)

    areas  = [2 * (4*hy*hz + 4*hx*hz + 4*hx*hy) for _, (hx, hy, hz) in T_BOXES]
    total  = sum(areas)
    t_tpl  = np.concatenate([
        _sample_box(off, half, max(1, int(n_block * a / total)), rng)
        for (off, half), a in zip(T_BOXES, areas)
    ])

    hand_tpl = _sample_cylinder(GRIPPER_RADIUS, GRIPPER_LENGTH, n_hand, rng)

    return {"cube": cube_tpl, "T": t_tpl, "hand": hand_tpl}


# Keep the old name as an alias so existing imports don't break immediately,
# but users should prefer sample_templates.
build_templates = sample_templates


# ─────────────────────────────────────────────────────────────────────────────
# Pose math
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) → (3,3) rotation matrix."""
    w, x, y, z = q.astype(np.float64)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → (w,x,y,z)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def apply_pose(local_pts: np.ndarray,
               pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate + translate local_pts into world frame given (w,x,y,z) quat."""
    R = _quat_to_rot(quat)
    return (local_pts.astype(np.float64) @ R.T + pos.astype(np.float64)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-step extraction (for visualiser)
# ─────────────────────────────────────────────────────────────────────────────

def extract_step(traj, step: int, templates: dict,
                 fk: FloatingGripperFK, use_T: bool) -> dict:
    actors = traj["env_states"]["actors"]
    arts   = traj["env_states"]["articulations"]

    bs    = actors["push_block"][step]
    block = apply_pose(templates["T"] if use_T else templates["cube"],
                       bs[ACTOR_POS], bs[ACTOR_QUAT])

    ar       = arts["floating_gripper"][step]
    tcp_pos, tcp_quat = fk.tcp_pose(ar)
    hand     = apply_pose(templates["hand"], tcp_pos, tcp_quat)

    return {"block": block, "hand": hand}


# ─────────────────────────────────────────────────────────────────────────────
# Shared limit computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_limits(all_clouds: list[dict]) -> list[tuple]:
    pts = np.concatenate([c[k] for c in all_clouds for k in ("block", "hand")])
    pad = 0.05
    c   = pts.mean(0)
    r   = max((pts.max(0) - pts.min(0)).max() / 2 + pad, 0.1)
    return [(c[i]-r, c[i]+r) for i in range(3)]


# ─────────────────────────────────────────────────────────────────────────────
# GIF rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(clouds: dict, title: str, elev: float, azim: float,
                 lims: list, dpi: int = 100) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")
    b   = clouds["block"]
    ax.scatter(b[:,0], b[:,1], b[:,2], s=1.5, c="#3478f5", label="block", depthshade=True)
    h   = clouds["hand"]
    ax.scatter(h[:,0], h[:,1], h[:,2], s=1.5, c="#e8781a", label="hand",  depthshade=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7); ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf  = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h_px = fig.canvas.get_width_height()
    img  = buf.reshape(h_px, w, 4).copy()
    plt.close(fig)
    return img


def save_gif(frames: list, path: str, fps: float) -> None:
    try:
        import imageio
        imageio.mimsave(path, frames, duration=1.0/fps, loop=0)
        return
    except ImportError:
        pass
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(1000/fps), loop=0)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive viewer (open3d)
# ─────────────────────────────────────────────────────────────────────────────

def _make_pcd(pts, color):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
    return pcd


def _update_pcd(pcd, pts, color):
    import open3d as o3d
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))


def run_viewer(all_clouds: list, steps: list,
               fps: float = 10.0, start_paused: bool = False) -> None:
    import open3d as o3d

    n   = len(all_clouds)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer", width=800, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.12, 0.12, 0.12])
    opt.point_size = 2.5

    pcd_block = _make_pcd(all_clouds[0]["block"], BLOCK_COLOR)
    pcd_hand  = _make_pcd(all_clouds[0]["hand"],  HAND_COLOR)
    vis.add_geometry(pcd_block)
    vis.add_geometry(pcd_hand)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    state = {"frame": 0, "paused": start_paused, "quit": False, "last_t": time.time()}

    def _refresh(vis):
        c = all_clouds[state["frame"]]
        _update_pcd(pcd_block, c["block"], BLOCK_COLOR)
        _update_pcd(pcd_hand,  c["hand"],  HAND_COLOR)
        vis.update_geometry(pcd_block)
        vis.update_geometry(pcd_hand)
        vis.get_view_control()

    def on_space(vis): state["paused"] = not state["paused"]; print("Paused" if state["paused"] else "Playing")
    def on_left(vis):
        if state["paused"]: state["frame"] = (state["frame"] - 1) % n; _refresh(vis)
    def on_right(vis):
        if state["paused"]: state["frame"] = (state["frame"] + 1) % n; _refresh(vis)
    def on_quit(vis): state["quit"] = True

    vis.register_key_callback(32,       on_space)
    vis.register_key_callback(263,      on_left)
    vis.register_key_callback(262,      on_right)
    vis.register_key_callback(ord("Q"), on_quit)
    vis.register_key_callback(27,       on_quit)

    frame_dt = 1.0 / fps
    print(f"\nViewer: {n} frames @ {fps} fps   SPACE=pause  ←→=step  Q/ESC=quit\n")

    while not state["quit"]:
        now = time.time()
        if not state["paused"] and (now - state["last_t"]) >= frame_dt:
            state["frame"]  = (state["frame"] + 1) % n
            state["last_t"] = now
            _refresh(vis)
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


# ─────────────────────────────────────────────────────────────────────────────
# CLI (stand-alone visualiser)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import h5py

    def parse_args():
        p = argparse.ArgumentParser(
            description="Visualize PushBoundary (floating gripper) point clouds."
        )
        p.add_argument("--h5",     required=True)
        p.add_argument("--traj",   default="traj_0")
        p.add_argument("--shape",  default="cube", choices=["cube", "T"])
        p.add_argument("--every",  type=int,   default=1)
        p.add_argument("--fps",    type=float, default=20.0)
        p.add_argument("--n_block",type=int,   default=512)
        p.add_argument("--n_hand", type=int,   default=256)
        p.add_argument("--render", action="store_true")
        p.add_argument("--paused", action="store_true")
        p.add_argument("--gif",    default="pointclouds.gif")
        p.add_argument("--elev",   type=float, default=30.0)
        p.add_argument("--azim",   type=float, default=-60.0)
        p.add_argument("--dpi",    type=int,   default=100)
        return p.parse_args()

    args      = parse_args()
    rng       = np.random.default_rng(0)
    fk        = FloatingGripperFK()
    templates = sample_templates(args.n_block, args.n_hand, rng)

    with h5py.File(args.h5, "r") as f:
        traj     = f[args.traj]
        n_states = traj["env_states"]["actors"]["push_block"].shape[0]
        steps    = list(range(0, n_states, args.every))
        print(f"'{args.traj}': {n_states} states → {len(steps)} frames")
        all_clouds = [
            extract_step(traj, s, templates, fk, args.shape == "T")
            for s in steps
        ]

    if args.render:
        run_viewer(all_clouds, steps, fps=args.fps, start_paused=args.paused)
    else:
        lims = _compute_limits(all_clouds)
        frames = [
            render_frame(c, f"step {s}", args.elev, args.azim, lims, args.dpi)
            for s, c in zip(steps, all_clouds)
        ]
        save_gif(frames, args.gif, args.fps)
        print(f"Saved {len(frames)} frames → {args.gif}")