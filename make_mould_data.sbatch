#!/bin/bash
#SBATCH --job-name=mould_data
#SBATCH --output=/data/user_data/mbronars/slurm/output/mould_data_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/mould_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd /data/user_data/mbronars/packages/PointWorld

# robot-features MUST match train_mould.sbatch --robot_features exactly (no
# gripper_open since the mould paddle doesn't open/close). norm_stats is a
# flat positional vector, so order matters too.
python mould_to_pointworld_wds.py \
    --input /data/group_data/katefgroup/VLA/3DPCWM/deformable_mould_10k \
    --out   /data/group_data/katefgroup/VLA/3DPCWM/mould_data_pointworld \
    --robot-features robot_flows,robot_normals,robot_velocity,robot_acceleration \
    --stride 2 \
    --clip-spacing 15
