#!/bin/bash
#SBATCH --job-name=pickup_1gpu
#SBATCH --output=/data/user_data/mbronars/slurm/output/pickup_1gpu_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/pickup_1gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --signal=B:USR1@180

# ── 1-GPU folding-style training on the Pickup-Pose 4-success data ─────────
# Same per-demo layout as folding/mould: object particles in scene
# (512/demo), Panda gripper in robot conditioning (671/demo). grid_size 0.001
# keeps essentially all particles. Pickup scenes are small (~7900
# tokens/sample at batch 1 / ~126k at batch 16) — well within a single L40S.
#
# Long run: 200 epochs at production defaults, no early-stopping. Self-
# chaining across the 48 h slurm wall; expect ~3-5 sequential jobs given
# ~193k train clips × 200 epochs at batch 16.

SCRIPT=/data/user_data/mbronars/packages/PointWorld/train_pickup.sbatch
EXP=pickup_1gpu
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
    --domains pickup \
    --data_dirs /data/group_data/katefgroup/VLA/3DPCWM/pickup_data_pointworld \
    --norm_stats_path /data/group_data/katefgroup/VLA/3DPCWM/pickup_data_pointworld/norm_stats \
    --robot_features robot_flows,robot_normals,robot_velocity,robot_acceleration \
    --num_workers 6 \
    --exp_name ${EXP} \
    --batch_size 8 \
    --grid_size 0.001 \
    --num_epochs 200 \
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
