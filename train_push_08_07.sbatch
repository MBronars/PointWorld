#!/bin/bash
#SBATCH --job-name=push_08_07
#SBATCH --output=/data/user_data/mbronars/slurm/output/push_08_07_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/push_08_07_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --signal=B:USR1@180

# ── 1-GPU PointWorld pushing training on push_08_07 (no-normals variant) ───
# Data was converted by push_08_07_to_pointworld_wds.py.
#   - 178,176 train clips / 8,000 test clips
#   - 128 block particles + 128 gripper particles per scene (obj_ids split)
#   - grid_size 0.015 (15 mm) matches the reference pushing_no_normals config;
#     pushing scenes are low-density enough that this captures everything.
#   - No normals: robot_features excludes robot_normals; scene_features
#     excludes scene_normals. Both are stored in norm_stats.json accordingly.
#
# Self-chains across the 48 h wall via USR1 → sbatch --dependency=afterany.

SCRIPT=/data/user_data/mbronars/packages/PointWorld/train_push_08_07.sbatch
EXP=push_08_07
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
    --domains pushing \
    --data_dirs /data/group_data/katefgroup/VLA/3DPCWM/push_08_07_pointworld \
    --norm_stats_path /data/group_data/katefgroup/VLA/3DPCWM/push_08_07_pointworld/norm_stats \
    --robot_features robot_flows,robot_velocity,robot_acceleration \
    --scene_features scene_flows,dist2robot \
    --num_workers 16 \
    --exp_name ${EXP} \
    --batch_size 128 \
    --grid_size 0.015 \
    --num_epochs 200 \
    --eval_freq 600 \
    --save_freq 1800 \
    --num_eval_batches 10 \
    ${RESUME} &

PID=$!
while true; do
    wait "${PID}"
    status=$?
    kill -0 "${PID}" 2>/dev/null || break
done
echo "[$(date)] train.py exited with status ${status}"
exit ${status}
