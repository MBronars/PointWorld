#!/bin/bash
#SBATCH --job-name=data_collection
#SBATCH --output=/data/user_data/mbronars/slurm/output/data_collection_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/data_collection_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=general

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env


cd /data/user_data/mbronars/packages/PointWorld

python h5_to_pointworld_wds.py --h5 /data/user_data/mbronars/packages/Planning_wrapper/demos/batch_200k_04_16_obstacles0/all_demos.h5 --out /data/group_data/katefgroup/VLA/3DPCWM/pushing_data_pointworld_05_03