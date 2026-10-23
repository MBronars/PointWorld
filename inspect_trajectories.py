#!/usr/bin/env python3
"""
inspect_robot_pcd_gif.py
Reads a generated WebDataset shard, uses the RobotSampler to generate 
dense robot point clouds, and generates an animated GIF with a locked camera.
"""

import argparse
import io
import tarfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch

from robot_sampler import RobotSampler as TorchRobotSampler
from dataset_components.robot import _get_robot_flows 

VIEWS = [
    (30, 45, "Perspective"),
    (90, -90, "Top-Down"),
    (0, -90, "Side View")
]

def load_sample_from_tar(tar_path: str, sample_idx: int) -> dict:
    sample_data = {}
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        prefixes = sorted(list(set([m.name.split('.')[0] for m in members])))
        
        if sample_idx >= len(prefixes):
            raise ValueError(f"Sample index {sample_idx} out of bounds.")
        
        target_prefix = prefixes[sample_idx]
        print(f"Loading sample: {target_prefix}")
        
        for m in members:
            if m.name.startswith(target_prefix) and m.name.endswith('.npy'):
                f = tar.extractfile(m)
                if f is not None:
                    key = m.name.split('.', 1)[1]
                    clean_key = key.replace('.npy', '')
                    sample_data[clean_key] = np.load(io.BytesIO(f.read()))
    return sample_data

def create_trajectory_gif(
    sample_data: dict, 
    robot_sampler: TorchRobotSampler,
    domain: str,
    max_robot_points: int,
    save_path: str, 
    fps: int = 5, 
    multi_view: bool = False
):
    block_flows = sample_data.get('view0_scene_flows', sample_data.get('scene_flows'))
    if block_flows is None:
        raise KeyError("Could not find scene flows in the extracted sample.")
    
    T = block_flows.shape[0]

    print(f"Generating robot point clouds for domain '{domain}'...")
    robot_data = _get_robot_flows(
        sample=sample_data, 
        robot_sampler=robot_sampler, 
        max_robot_points=max_robot_points, 
        domain=domain
    )
    robot_flows = robot_data['robot_flows']

    # 1. Calculate strict global limits
    all_pts = np.vstack([block_flows.reshape(-1, 3), robot_flows.reshape(-1, 3)])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5

    x_lim = (mid_x - max_range, mid_x + max_range)
    y_lim = (mid_y - max_range, mid_y + max_range)
    z_lim = (mid_z - max_range, mid_z + max_range)

    block_centroids = block_flows.mean(axis=1)
    robot_centroids = robot_flows.mean(axis=1)

    print(f"Rendering {T} frames...")
    frames = []
    
    # 2. Setup fixed layout outside the loop
    num_cols = len(VIEWS) if multi_view else 1
    figsize = (18, 6) if multi_view else (8, 8)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.90, wspace=0.1)
    
    views_to_render = VIEWS if multi_view else [VIEWS[0]]

    for t in range(T):
        fig.clf()
        
        for col, (elev, azim, view_label) in enumerate(views_to_render):
            ax = fig.add_subplot(1, num_cols, col + 1, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            
            ax.scatter(block_flows[t, :, 0], block_flows[t, :, 1], block_flows[t, :, 2],
                       color='blue', s=15, alpha=0.8, label='Object Cloud')
            
            ax.scatter(robot_flows[t, :, 0], robot_flows[t, :, 1], robot_flows[t, :, 2],
                       color='red', s=15, alpha=0.8, label='Robot Cloud')

            if t > 0:
                ax.plot(block_centroids[:t+1, 0], block_centroids[:t+1, 1], block_centroids[:t+1, 2], 
                        color='lightblue', linewidth=1.5, linestyle='--')
                ax.plot(robot_centroids[:t+1, 0], robot_centroids[:t+1, 1], robot_centroids[:t+1, 2], 
                        color='lightcoral', linewidth=1.5, linestyle='--')

            # 3. Apply strict 3D limits and box aspect ratio
            ax.set_xlim3d(x_lim)
            ax.set_ylim3d(y_lim)
            ax.set_zlim3d(z_lim)
            ax.set_box_aspect([1, 1, 1])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{view_label} (Frame {t+1}/{T})', fontsize=12)
            
            if col == 0:
                ax.legend(loc='upper right')

        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)
        
    plt.close(fig)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"Saved GIF to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=str, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--domain", type=str, default="pushing")
    parser.add_argument("--urdf", type=str, default="/data/user_data/mbronars/packages/PointWorld/assets/pushing_robot/pushing_robot.urdf")
    parser.add_argument("--max-robot-points", type=int, default=128)
    parser.add_argument("--out", type=str, default="pcd_trajectory.gif")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--multi-view", action="store_true")
    args = parser.parse_args()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        robot_sampler = TorchRobotSampler(args.urdf, device=device)
        
        sample_data = load_sample_from_tar(args.tar, args.sample)
        create_trajectory_gif(
            sample_data=sample_data, 
            robot_sampler=robot_sampler, 
            domain=args.domain,
            max_robot_points=args.max_robot_points,
            save_path=args.out, 
            fps=args.fps, 
            multi_view=args.multi_view
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()