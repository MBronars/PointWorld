#!/bin/bash
#SBATCH --job-name=mould_new_6k_data
#SBATCH --output=/data/user_data/mbronars/slurm/output/mould_new_6k_data_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/mould_new_6k_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd /data/user_data/mbronars/packages/PointWorld

# Same per-demo layout as deformable_mould_10k (1280 particles = 256 gripper +
# 1024 blob, gripper-first), just 240 frames per traj instead of 300 and 304
# train shards instead of 509. Reusing mould_to_pointworld_wds.py unchanged;
# only --input / --out differ.
python mould_to_pointworld_wds.py \
    --input /data/group_data/katefgroup/VLA/3DPCWM/deformable_mould_new_6k_wds \
    --out   /data/group_data/katefgroup/VLA/3DPCWM/mould_new_6k_data_pointworld \
    --robot-features robot_flows,robot_normals,robot_velocity,robot_acceleration \
    --stride 2 \
    --clip-spacing 15
