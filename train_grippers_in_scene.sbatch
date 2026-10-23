#!/bin/bash
#SBATCH --job-name=folding_grip_2gpu
#SBATCH --output=/data/user_data/mbronars/slurm/output/folding_grip_2gpu_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/folding_grip_2gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --signal=B:USR1@180

# ── Full 2-GPU training: gripper points merged into scene (post-hoc) ────────
# Architectural diffs vs the production folding_dense_grid checkpoint:
#   - --include_gripper_in_scene true : a transform in the data pipeline
#     concatenates robot_flows into scene_flows so the gripper is part of the
#     predicted particle set. Robot conditioning is unchanged.
#   - --grid_size 0.001 : ~99.8% of points kept (cloth 100%, gripper ~99%),
#     vs production's 0.01 which kept ~60-75% of cloth and ~40% of gripper.
#   - 2 L40S GPUs via torchrun + --distributed true.
#
# Same self-chaining USR1 trap as train_overfit_folding_new19.sbatch: ~180 s
# before the 48 h wall, sbatch a dependent resume job. Chain stops on success
# (exit 0, no USR1) or crash (no USR1).
#
# norm_stats: re-estimated from the production train shards under the merge
# transform (recompute_norm_stats_with_grippers.py, ~8k samples), so the
# scene_features mean/var and per-timestep gt_scene_flows_relative stats
# match the actual cloth+gripper distribution the model will see. Robot
# stats are copied from the production norm_stats unchanged.

SCRIPT=/data/user_data/mbronars/packages/PointWorld/train_grippers_in_scene.sbatch
EXP=folding_grippers_in_scene
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

# Fragmentation-friendly allocator (the smoke needed it at grid 0.001).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume if a previous job in the chain checkpointed.
RESUME=""
if [ -f "${CKPT}" ]; then
    RESUME="--model_path ${CKPT}"
    echo "[$(date)] Resuming from ${CKPT}"
else
    echo "[$(date)] Fresh start (no checkpoint found)"
fi

# Unique torchrun rendezvous port per slurm job (avoid clashes if multiple
# 2-GPU jobs share a node).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} train.py \
    --domains folding \
    --data_dirs /data/group_data/katefgroup/VLA/3DPCWM/folding_data_pointworld \
    --norm_stats_path /data/user_data/mbronars/folding_grippers_norm_stats \
    --robot_features robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration \
    --distributed true \
    --num_workers 6 \
    --exp_name ${EXP} \
    --batch_size 2 \
    --grid_size 0.001 \
    --include_gripper_in_scene true \
    --max_scene_points 16000 \
    --num_epochs 25 \
    --eval_freq 500 \
    --save_freq 1000 \
    --num_eval_batches 25 \
    ${RESUME} &

# Re-enter wait if the USR1 trap interrupts it.
PID=$!
while true; do
    wait "${PID}"
    status=$?
    kill -0 "${PID}" 2>/dev/null || break
done
echo "[$(date)] train.py exited with status ${status}"
exit ${status}
