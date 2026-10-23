#!/bin/bash
#SBATCH --job-name=folding_data
#SBATCH --output=/data/user_data/mbronars/slurm/output/folding_data_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/folding_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# CPU-only job: the converter is pure numpy/webdataset, no GPU needed.
# The `cpu` partition has a 2-day limit; requesting no --gres keeps it CPU-only.
#SBATCH --partition=cpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env


cd /data/user_data/mbronars/packages/PointWorld

# --robot-features MUST match train_folding.sbatch's --robot_features exactly
# (same names, same order) -- norm_stats is a flat positional vector.
python npz_to_pointworld_wds.py \
    --root /data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1 \
    --out  /data/group_data/katefgroup/VLA/3DPCWM/folding_data_pointworld \
    --robot-features robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration
