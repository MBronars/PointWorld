#!/bin/bash
#SBATCH --job-name=pw_two
#SBATCH --output=/data/user_data/mbronars/logs/pw_two_%j.out
#SBATCH --error=/data/user_data/mbronars/logs/pw_two_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32

# ── PointWorld baseline on pickup + yikai_small_fs0 (no rl_push) ──────────────
#
# Matched to the PWM run pwm_v1_two_ctxgen_100m_uncond (job 10537059): the same two
# corpora, 32 windows per optimizer step, 200,000 steps from scratch, 4 + 16 frames.
# Its README is
# /data/group_data/katefgroup/runs_pwm_mbronars/datasets/yikai_small_fs0/README.md.
#
# DATA. Streamed live from the PWM loader with that run's own recipe (nothing copied):
# pickup with its exclusion manifest and shift_m, and yikai_small_fs0 — the index that
# drops the 10-frame skip, so each segment keeps its approach (951,512 distinct 20-frame
# training windows against 780,425 before). No data.weights, so the interleave draws a
# domain per batch: 50/50, batches domain-pure.
#
# HORIZON. --pred_horizon 16: 16 predicted frames from 1 context frame (T = 17), recorded
# in the checkpoint contract. Under the protocol each 16-frame window is one call.
#
# BATCH. 4 windows per micro-batch x 8 accumulation = 32 per optimizer step, the PWM run's
# number. Measured at T = 17 on an A100 40 GB (2026-09-23): micro-batch 4 peaks at 21.5 GB
# and runs 3.56 windows/s; micro-batch 8 runs out of memory at 34 GB. An L40S is 46 GB, so
# MICRO=8 ACCUM=4 may fit there and run faster — unmeasured.
#
# BUDGET. --budget_windows 6,400,000 = exactly 200,000 optimizer steps (it fires only on an
# accumulation boundary). At 3.56 windows/s that is ~500 h, so several 48 h legs; resubmit
# this script to resume from latest.pt.
#
# RULER. Its own pooled two-domain fit (pickup + yikai_small_fs0, 20,000 windows each of 17
# frames). The three-domain ruler does not apply: rl_push set its xy scale.
#
# CHECKPOINTS. latest.pt every 2,000 micro-batches; matched_step_NNNNNN.pt every 10,000
# optimizer steps (320,000 windows, --matched_windows_per_step 32), 20 over the budget.
#
# This job does not self-chain.

set -uo pipefail

MICRO=${MICRO:-4}
ACCUM=${ACCUM:-8}
RECIPE=${RECIPE:-/data/user_data/mbronars/launch_pwm_data_v1_two/configs/unet_two_ctxgen_100m_uncond.yaml}
RECIPE_SHA256=b455a5e5f797fc9d8fb3b09cabb4baf64a0a818751e4618601e10052026dd229
STATS=${STATS:-/data/group_data/katefgroup/runs_pw_mbronars/two_stream/norm_stats}
EXP=${EXP:-pointworld_two}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pw_mbronars}
CKPT=${LOG_DIR}/${EXP}/latest.pt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env
cd /data/user_data/mbronars/packages/PointWorld

[ -n "${RECIPE_SHA256}" ] && { echo "${RECIPE_SHA256}  ${RECIPE}" | sha256sum --check --quiet \
    || { echo "recipe ${RECIPE} changed since this script was written" >&2; exit 2; }; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# pointworld-env has no pytorch3d; PWM's adapters reach two pure-tensor functions in it for
# pose channels PointWorld discards. See pwm_shims/README.md.
export PWM_PYTORCH3D=/data/user_data/mbronars/packages/PointWorld/pwm_shims
export WANDB_DIR=${LOG_DIR}/${EXP}
mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "PointWorld baseline -- pickup + yikai_small_fs0 (streamed), 16-frame horizon"
echo "Date: $(date)   Job: ${SLURM_JOB_ID}   Node: $(hostname)"
echo "micro-batch ${MICRO} x accumulation ${ACCUM} = $((MICRO * ACCUM)) windows/step; budget 6,400,000 windows = 200,000 steps"
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
    --domains pickup_yikai_small,pickup_release \
    --data_dirs pwm://yikai@${RECIPE},pwm://pickup@${RECIPE} \
    --norm_stats_path ${STATS} \
    --robot_features robot_flows,robot_velocity,robot_acceleration \
    --scene_features scene_flows,dist2robot \
    --exp_name ${EXP} \
    --log_dir ${LOG_DIR} \
    --pred_horizon 16 \
    --batch_size ${MICRO} \
    --grad_accum_steps ${ACCUM} \
    --budget_windows 6400000 \
    --num_epochs 100000 \
    --grid_size 0.001 \
    --num_workers 6 \
    --eval_freq 500 \
    --save_freq 2000 \
    --num_eval_batches 25 \
    --snapshot_matched_steps 10000 \
    --matched_windows_per_step 32 \
    ${RESUME}

status=$?
echo "[$(date)] train.py exited with status ${status}"
exit ${status}
