#!/bin/bash
#SBATCH --job-name=folding_2gpu
#SBATCH --output=/data/user_data/mbronars/slurm/output/folding_2gpu_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/folding_2gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --signal=B:USR1@180

# ── 2-GPU DDP folding training, production config ───────────────────────────
# Why this config: a disambiguation test showed DDP × grid_size 0.001 is
# unstable (PTV3 sparse-conv path crashes under DDP after ~10-16 steps with
# huge initial loss). DDP itself works fine at production grid 0.01 — clean
# loss trajectory in the disambiguation, model converges as expected.
#
# So this is just production-folding-dense-grid (grid 0.01) but on 2 GPUs:
# matching global batch = 16 (8 per GPU × 2), which is what production used
# at batch_size 16 on 1 GPU. Should be ~2x faster than a single-GPU run.

SCRIPT=/data/user_data/mbronars/packages/PointWorld/train_folding_2gpu.sbatch
EXP=folding_dense_grid_2gpu
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
    --batch_size 8 \
    --grid_size 0.01 \
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
