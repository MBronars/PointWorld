#!/bin/bash
#SBATCH --job-name=folding_cloth_dense
#SBATCH --output=/data/user_data/mbronars/slurm/output/folding_cloth_dense_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/folding_cloth_dense_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --signal=B:USR1@180

# ── Fallback 2-GPU training: cloth-only scene at "dense grid" ──────────────
# This is the production architecture (cloth in scene, gripper in robot ONLY)
# but with grid_size 0.001 so essentially every cloth vertex survives the
# voxel downsample (4-7k -> ~all 4952/7021/10580). No gripper-in-scene
# changes; norm_stats are the production stats (cloth-only).
#
# Use as a fallback if the gripper-in-scene run blows up — this isolates
# "more cloth points" from "predicting gripper too."

SCRIPT=/data/user_data/mbronars/packages/PointWorld/train_cloth_dense_grid.sbatch
EXP=folding_cloth_dense_grid
CKPT=/data/user_data/mbronars/packages/PointWorld/train_logs/${EXP}/model-last.pt

_resubmitted=0
_on_timeout() {
    if [ "${_resubmitted}" -eq 0 ]; then
        _resubmitted=1
        echo "[$(date)] USR1: time limit near -> chaining dependent resume job"
        sbatch --dependency=afterany:${SLURM_JOB_ID} "${SCRIPT}"
    fi
}
trap _on_timeout USR1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env
cd /data/user_data/mbronars/packages/PointWorld

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESUME=""
if [ -f "${CKPT}" ]; then
    RESUME="--model_path ${CKPT}"
    echo "[$(date)] Resuming from ${CKPT}"
else
    echo "[$(date)] Fresh start (no checkpoint found)"
fi

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} train.py \
    --domains folding \
    --data_dirs /data/group_data/katefgroup/VLA/3DPCWM/folding_data_pointworld \
    --norm_stats_path /data/group_data/katefgroup/VLA/3DPCWM/folding_data_pointworld/norm_stats \
    --robot_features robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration \
    --distributed true \
    --num_workers 6 \
    --exp_name ${EXP} \
    --batch_size 4 \
    --grid_size 0.001 \
    --max_scene_points 16000 \
    --num_epochs 25 \
    --eval_freq 500 \
    --save_freq 1000 \
    --num_eval_batches 25 \
    ${RESUME} &

PID=$!
while true; do
    wait "${PID}"
    status=$?
    kill -0 "${PID}" 2>/dev/null || break
done
echo "[$(date)] train.py exited with status ${status}"
exit ${status}
