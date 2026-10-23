#!/bin/bash
#SBATCH --job-name=folding_training
#SBATCH --output=/data/user_data/mbronars/slurm/output/folding_training_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/folding_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=general

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env


cd /data/user_data/mbronars/packages/PointWorld

# robot_features = the standard PointWorld bimanual list (matches the
# `behavior` domain). scene_features keeps the trimmed default
# (scene_flows,scene_normals,dist2robot) since the cloth shards carry no
# real scene colors/normals -- scene_normals is silently skipped.
#
# Single-GPU run: --batch_size is the global batch.
python train.py \
    --domains folding \
    --data_dirs /data/group_data/katefgroup/VLA/3DPCWM/folding_data_pointworld \
    --norm_stats_path /data/group_data/katefgroup/VLA/3DPCWM/folding_data_pointworld/norm_stats \
    --robot_features robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration \
    --num_workers 8 \
    --exp_name folding_dense_grid \
    --batch_size 16 \
    --grid_size 0.01 \
