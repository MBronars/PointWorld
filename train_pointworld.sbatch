#!/bin/bash
#SBATCH --job-name=pointworld_training
#SBATCH --output=/data/user_data/mbronars/slurm/output/pointworld_training_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pointworld_training_%j.err
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

python train.py --domains pushing --data_dirs /data/group_data/katefgroup/VLA/3DPCWM/pushing_data_pointworld_05_03 --norm_stats_path=/data/group_data/katefgroup/VLA/3DPCWM/pushing_data_pointworld_05_03/norm_stats --exp_name pushing_no_normals --batch_size=128