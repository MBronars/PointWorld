#!/bin/bash
#SBATCH --job-name=folding_overfit19
#SBATCH --output=/data/user_data/mbronars/slurm/output/folding_overfit19_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/folding_overfit19_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --signal=B:USR1@180

# ── Overfit / debug run #2 ──────────────────────────────────────────────────
# A single FULL cloth-folding trajectory: bimanual/new_000019.npz
# (short_shirt = 7021 verts, 1571 native frames -> 761 window-clips). train/
# and test/ hold the identical clip set.
#
# Goal: 1000 true epochs over all 761 clips ~= 96k optimizer steps (~53 h at
# ~2 s/step). That exceeds the 48 h slurm limit, so this script is self-
# chaining: ~180 s before the time limit slurm sends USR1, the trap below
# submits a DEPENDENT resume job (starts only after this one ends, so the two
# never write the checkpoint concurrently). The chain stops automatically when
# train.py reaches 1000 epochs (exits 0, no USR1) or crashes (no USR1) -- it
# only re-chains on a genuine timeout. Expected: 2 jobs total.

SCRIPT=/data/user_data/mbronars/packages/PointWorld/train_overfit_folding_new19.sbatch
EXP=folding_overfit_new19
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

# Augmentation is on in this run (no --deterministic_train, see comment above),
# so per-sample memory varies; expandable_segments reduces fragmentation that
# OOM'd the first attempt right at the 44 GB edge.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume from the latest checkpoint if a previous job in the chain left one.
RESUME=""
if [ -f "${CKPT}" ]; then
    RESUME="--model_path ${CKPT}"
    echo "[$(date)] Resuming from ${CKPT}"
else
    echo "[$(date)] Fresh start (no checkpoint found)"
fi

python train.py \
    --domains folding \
    --data_dirs /data/user_data/mbronars/folding_overfit_new19 \
    --norm_stats_path /data/user_data/mbronars/folding_overfit_new19/norm_stats \
    --robot_features robot_flows,robot_normals,gripper_open,robot_velocity,robot_acceleration \
    --num_workers 4 \
    --exp_name ${EXP} \
    --batch_size 4 \
    --grid_size 0.003 \
    --num_epochs 1000 \
    --eval_freq 500 \
    --save_freq 1000 \
    --num_eval_batches 25 \
    ${RESUME} &

# wait for train.py, re-entering wait if the USR1 trap interrupts it
PID=$!
while true; do
    wait "${PID}"
    status=$?
    kill -0 "${PID}" 2>/dev/null || break
done
echo "[$(date)] train.py exited with status ${status}"
exit ${status}
