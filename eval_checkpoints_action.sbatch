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

python rolling_infer_from_h5.py --out-dir rollout_results_interpolants/pointworld_new --checkpoint /data/user_data/mbronars/packages/PointWorld/logs/pushing/push_v1/model-last.pt --save-gifs --total-frames 100 --seed 42 --h5 /data/user_data/mbronars/packages/Planning_wrapper/demos/demos_old/PushBoundary/face_push_batch_v8/all_demos.h5 --n-block 128 --n-hand 128 --data-dir /data/group_data/katefgroup/VLA/3DPCWM/pushing_data_pointworld --num-samples 50