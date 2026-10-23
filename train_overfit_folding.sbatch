#!/bin/bash
#SBATCH --job-name=folding_overfit
#SBATCH --output=/data/user_data/mbronars/slurm/output/folding_overfit_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/folding_overfit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=general

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env


cd /data/user_data/mbronars/packages/PointWorld

# Overfitting / debug run: clips from a SINGLE folding trajectory
# (ik_push/000028.npz — short_shirt = 7021 verts, 409 native frames, high
# cloth motion). train/ and test/ hold the identical clip set, so val_loss
# dropping alongside train_loss confirms the model can fit the data.
#
#   - grid_size 0.003 : near-full point density (cloth ~7000, gripper ~1900),
#                       a big drop from the 0.01 production run.
#   - NOTE: deterministic_train is intentionally NOT set. It makes the training
#     stream finite, but trainer.py creates the train iterator once and shares
#     it with eval (assuming the normal resampled/infinite stream) -> a finite
#     stream exhausts and raises StopIteration. So we use the standard
#     resampled path; augmentation is on but a 410M model still overfits 180
#     clips hard.
#   - max_train_steps caps the run; num_epochs is set high so it is the limiter.
python train.py \
    --domains folding \
    --data_dirs /data/user_data/mbronars/folding_overfit_ikpush28 \
    --norm_stats_path /data/user_data/mbronars/folding_overfit_ikpush28/norm_stats \
    --robot_features robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration \
    --num_workers 4 \
    --exp_name folding_overfit_ikpush28 \
    --batch_size 8 \
    --grid_size 0.003 \
    --num_epochs 1000 \
    --max_train_steps 8000 \
    --eval_freq 100 \
    --save_freq 500 \
    --num_eval_batches 25
