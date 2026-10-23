#!/bin/bash
#SBATCH --job-name=pickup_data
#SBATCH --output=/data/user_data/mbronars/slurm/output/pickup_data_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pickup_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd /data/user_data/mbronars/packages/PointWorld

# Pool all 4 success roots (matches v9 / PhysCtrl). Blocklist is in robertre's
# user dir and read-only to us — the converter gracefully proceeds without it
# (0.56% extra trajectories included, no harm).
# T=40 native, stride=1 (per pickup docs), clip_spacing=15 → ~2 clips/demo.
python pickup_to_pointworld_wds.py \
    --out /data/group_data/katefgroup/VLA/3DPCWM/pickup_data_pointworld \
    --robot-features robot_flows,robot_normals,robot_velocity,robot_acceleration \
    --stride 1 \
    --clip-spacing 15
