#!/bin/bash
#SBATCH --job-name=pw_dex
#SBATCH --output=/data/user_data/mbronars/logs/pw_dex_%j.out
#SBATCH --error=/data/user_data/mbronars/logs/pw_dex_%j.err
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-23:30:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32

# ── PointWorld baseline on dex_v2_filtered2, matched to the PWM dex specialist ──
#
# The PWM run it is matched against: pwm_v1_dex_v2_filtered2_100m (runs_pwm_robertre),
# recipe unet_dex_v2_filtered2_100m.yaml (sha256 6d17c8b0d29a), 16 windows per optimizer
# step (8 per GPU x 2 GPUs), 200,000 steps = 3.2 M windows, 4 context + 16 predicted
# frames at 20 Hz.
#
# DATA. Streamed live from the PWM loader with a frozen, sha-verified copy of that recipe
# (pwm_stream.py; nothing copied). Dex windows: the object (512 points) is the scene and
# the only predicted body; the hand (13 or 23 links, 1,646 or 1,682 points) is the given
# robot. dex_v2_filtered2/train: 16,999 episodes; held-out test: 174.
#
# HORIZON. --pred_horizon 16: PointWorld predicts 16 frames from 1 context frame, so the
# model is built at T = 17 (release default 10). Recorded in the checkpoint contract, so
# evaluation rebuilds the same model. Under the protocol each 16-frame window is exactly
# one PointWorld call.
#
# BATCH. 4 windows per micro-batch x 4 accumulation = 16 windows per optimizer step, the
# specialist's. Measured on an A100 40 GB (2026-09-22): micro-batch 4 peaks at 23.4 GB and
# runs 4.29 windows/s with 4 CPUs (GPU ~45% busy); micro-batch 8 and the full batch of 16
# run out of memory on 40 GB.
#
# BUDGET. --budget_windows 3,200,000 stops at exactly 200,000 optimizer steps (it fires only
# on an accumulation boundary). --num_epochs is set out of the way. At the measured 4.29
# windows/s that is ~207 h, about three 71.5 h legs; resubmit this script to resume.
#
# CHECKPOINTS (group_data; ~5 GB each). latest.pt every 2,000 micro-batches (~31 min),
# atomic, the file a resubmitted leg resumes from. matched_step_NNNNNN.pt every 10,000
# optimizer steps (160,000 windows, --matched_windows_per_step 16), 20 over the budget.
#
# STATS. Fitted on 20,000 dex train windows of 17 frames through the same stream
# (fit_pwm_norm_stats.py); single domain.
#
# This job does not self-chain.

set -uo pipefail

MICRO=${MICRO:-4}
ACCUM=${ACCUM:-4}
RECIPE=${RECIPE:-/data/user_data/mbronars/launch_pw_dex/configs/unet_dex_v2_filtered2_100m.yaml}
RECIPE_SHA256=6d17c8b0d29a769cae712743ba18e843739fff2c7f54199c88f68ec37fc59269
STATS=${STATS:-/data/group_data/katefgroup/runs_pw_mbronars/dex_stream/norm_stats}
EXP=${EXP:-pointworld_dex}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pw_mbronars}
CKPT=${LOG_DIR}/${EXP}/latest.pt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env
cd /data/user_data/mbronars/packages/PointWorld

echo "${RECIPE_SHA256}  ${RECIPE}" | sha256sum --check --quiet \
    || { echo "recipe ${RECIPE} differs from the dex specialist's (${RECIPE_SHA256:0:12})" >&2; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# pointworld-env has no pytorch3d; PWM's adapters reach two pure-tensor functions in it for
# pose channels PointWorld discards. See pwm_shims/README.md.
export PWM_PYTORCH3D=/data/user_data/mbronars/packages/PointWorld/pwm_shims
export WANDB_DIR=${LOG_DIR}/${EXP}
mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "PointWorld baseline -- dex_v2_filtered2 (streamed), 16-frame horizon"
echo "Date: $(date)   Job: ${SLURM_JOB_ID}   Node: $(hostname)"
echo "micro-batch ${MICRO} x accumulation ${ACCUM} = $((MICRO * ACCUM)) windows/step; budget 3,200,000 windows = 200,000 steps"
echo "Recipe: ${RECIPE}"
echo "Stats: ${STATS}"
echo "Checkpoints: ${LOG_DIR}/${EXP}"
nvidia-smi -L
echo "============================================================"

RESUME=""
if [ -f "${CKPT}" ]; then
    RESUME="--model_path ${CKPT}"
    echo "[$(date)] Resuming from ${CKPT}"
else
    echo "[$(date)] Fresh start (no checkpoint found)"
fi

python train.py \
    --domains pickup_dex \
    --data_dirs pwm://dex@${RECIPE} \
    --norm_stats_path ${STATS} \
    --robot_features robot_flows,robot_velocity,robot_acceleration \
    --scene_features scene_flows,dist2robot \
    --exp_name ${EXP} \
    --log_dir ${LOG_DIR} \
    --pred_horizon 16 \
    --batch_size ${MICRO} \
    --grad_accum_steps ${ACCUM} \
    --budget_windows 3200000 \
    --num_epochs 100000 \
    --grid_size 0.001 \
    --num_workers 6 \
    --eval_freq 500 \
    --save_freq 2000 \
    --num_eval_batches 25 \
    --snapshot_matched_steps 10000 \
    --matched_windows_per_step 16 \
    ${RESUME}

status=$?
echo "[$(date)] train.py exited with status ${status}"
exit ${status}
