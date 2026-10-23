#!/bin/bash
#SBATCH --job-name=mould_2gpu
#SBATCH --output=/data/user_data/mbronars/slurm/output/mould_2gpu_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/mould_2gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --signal=B:USR1@180

# ── 2-GPU DDP folding-style training on the new `mould` domain ──────────────
# Data: deformable mould blob (1024 particles) pushed by a rigid paddle
# (256 gripper points). grid_size 0.001 captures essentially all particles
# (measured: 99.8% retention; 0.0005 would be 100%). Scene is tiny vs cloth
# (~1280 tokens) so memory is comfortable even at the fine grid.
#
# Robot features: robot_flows + robot_normals (zeros) + robot_velocity +
# robot_acceleration (no gripper_open — paddle is solid). Scene features:
# scene_flows + dist2robot (scene_normals silently skipped for non-normal
# domains).
#
# Same self-chaining USR1 trap pattern as the folding 2-GPU run.

SCRIPT=/data/user_data/mbronars/packages/PointWorld/train_mould.sbatch
EXP=mould_dense_grid_2gpu
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
    --domains mould \
    --data_dirs /data/group_data/katefgroup/VLA/3DPCWM/mould_data_pointworld \
    --norm_stats_path /data/group_data/katefgroup/VLA/3DPCWM/mould_data_pointworld/norm_stats \
    --robot_features robot_flows,robot_normals,robot_velocity,robot_acceleration \
    --distributed true \
    --num_workers 6 \
    --exp_name ${EXP} \
    --batch_size 8 \
    --grid_size 0.001 \
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
