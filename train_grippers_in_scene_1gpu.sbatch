#!/bin/bash
#SBATCH --job-name=folding_grip_1gpu
#SBATCH --output=/data/user_data/mbronars/slurm/output/folding_grip_1gpu_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/folding_grip_1gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --signal=B:USR1@180

# ── 1-GPU folding training: gripper-in-scene + dense grid (0.001) ──────────
# Fallback when 2-GPU DDP × grid 0.001 keeps crashing (verified consistent
# across multiple attempts). Single-GPU at the same grid is known to work
# (the ik_push smoke trained for 250 steps cleanly, and the 1000-epoch
# overfit on new_000019 at grid 0.003 has been training cleanly for 20h+).
# So we drop DDP and accept ~2x slower wall-clock to actually train.
#
# Same self-chaining USR1 trap; expect 2-3 sequential 48h jobs.

SCRIPT=/data/user_data/mbronars/packages/PointWorld/train_grippers_in_scene_1gpu.sbatch
EXP=folding_grippers_in_scene_1gpu
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

python train.py \
    --domains folding \
    --data_dirs /data/group_data/katefgroup/VLA/3DPCWM/folding_data_pointworld \
    --norm_stats_path /data/user_data/mbronars/folding_grippers_norm_stats \
    --robot_features robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration \
    --num_workers 6 \
    --exp_name ${EXP} \
    --batch_size 2 \
    --grid_size 0.001 \
    --include_gripper_in_scene true \
    --max_scene_points 16000 \
    --num_epochs 25 \
    --eval_freq 500 \
    --save_freq 2000 \
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
