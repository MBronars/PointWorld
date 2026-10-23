#!/bin/bash
#SBATCH --job-name=pw_folding_med
#SBATCH --output=/data/group_data/katefgroup/runs_pw_mbronars/slurm/pw_folding_med_%j.out
#SBATCH --error=/data/group_data/katefgroup/runs_pw_mbronars/slurm/pw_folding_med_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32,babel-q5-28,babel-n5-28
# ── PointWorld baseline on cloth folding, 99M "medium" backbone ────────────────
#
# DATA. Streamed live from the PWM loader (pwm://folding) with the folding settings of the 500M
# clean4 recipe. Nominal 4,510 train / 236 test windows of 17 frames (1 given + 16 predicted).
# Training starts use the recipe's data.window_motion for folding, as PWM training does:
# p(start) ∝ (mean cloth step over the 16 predicted frames + 0.02 mm)^1.0 (pwm_stream.build_pwm_dataset).
# Measured over 300 training windows: mean cloth step 0.167 mm/frame weighted vs 0.072 uniform.
# Held-out windows stay uniform.
#
# MODEL. --ptv3_size medium --predictor_dim 192 = 99.1M parameters (small layout, wider;
# ptv3/ptv3_arch.yaml 'medium'). Inputs as released: 1 mm grid, no normals or colours.
#   gripper 9  = position 3 + velocity 3 + acceleration 3
#   scene   20 = position 3 + distance to the gripper 17
# --sphere_crop false: no training-time sphere crop (user, 2026-10-22).
# --param_nan_checks false: skips the two per-step whole-model parameter NaN scans; the loss NaN
#   check stays (~6% of a step on a shared A6000).
#
# BATCH. 2 GPUs x micro-batch 4 x 1 accumulation = 8 windows per optimizer step, the same as the PWM
# folding model it is compared with (runs_pwm_robertre/pwm_v1_folding_8mm_100m_unet_bidir, job 10507043:
# 2 GPUs x batch 4, 200,000 steps). Micro-batch 4 peaked at 24-29 GiB on one A6000; 8 does not fit.
#
# BUDGET. --budget_windows 1,600,000 = 200,000 optimizer steps of 8: same windows AND same steps as
# that model. Estimated ~12 windows/s on 2 L40S (not measured on L40S) = ~37 h; if a leg times out,
# resubmit this script to resume from latest.pt.
#
# RULER. Folding-only fit (20,000 windows, fit_pwm_norm_stats.py, job 10888082).
#
# CHECKPOINTS. latest.pt every 2,000 micro-batches per GPU (16,000 windows = 2,000 steps);
# matched_step_NNNNNN.pt every 10,000 steps of 8 windows, 20 in all.
#
# This job does not self-chain.
set -uo pipefail
MICRO=${MICRO:-4}
ACCUM=${ACCUM:-1}
RECIPE=${RECIPE:-/data/user_data/mbronars/launch_pwm_data_v1_clean4/configs/unet_joint8_500m_clean4_muon.yaml}
RECIPE_SHA256=1e308cbd62d5aab5bf0dda530d3f5384a9a57a1af3d7d4035df853bb7e49f4d2
STATS=${STATS:-/data/group_data/katefgroup/runs_pw_mbronars/folding_stream/norm_stats}
EXP=${EXP:-pointworld_folding_medium}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pw_mbronars}
CKPT=${LOG_DIR}/${EXP}/latest.pt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env
cd /data/user_data/mbronars/packages/PointWorld
[ -n "${RECIPE_SHA256}" ] && { echo "${RECIPE_SHA256}  ${RECIPE}" | sha256sum --check --quiet \
    || { echo "recipe ${RECIPE} changed since this script was written" >&2; exit 2; }; }
[ -f "${STATS}/norm_stats.json" ] || { echo "no norm_stats.json in ${STATS}" >&2; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Leg 1 (job 10892782) died at ~8.7k steps: rank 1 timed out (10 min) on the per-forward buffer broadcast while
# rank 0 had stalled mid-training with the job at its 100 GB host-memory limit (MaxRSS 100.0 GiB; one folding
# loader worker holds 0.8-0.96 GiB flat, so the rest is the two trainers + page cache of the clips read).
# Leg 2 on: 160 GB host memory and a 60 min collective timeout so a transient stall does not kill the run.
export PW_NCCL_TIMEOUT_MIN=60
export PWM_PYTORCH3D=/data/user_data/mbronars/packages/PointWorld/pwm_shims
export WANDB_DIR=${LOG_DIR}/${EXP}
mkdir -p "${WANDB_DIR}"
echo "============================================================"
echo "PointWorld baseline -- cloth folding (streamed), medium 99M, 16-frame horizon, no sphere crop"
echo "Date: $(date)   Job: ${SLURM_JOB_ID}   Node: $(hostname)"
echo "2 GPUs x micro-batch ${MICRO} x accumulation ${ACCUM} = $((2 * MICRO * ACCUM)) windows/step; budget 1,600,000 windows = 200,000 steps"
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
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} train.py \
    --distributed true \
    --domains pickup_folding \
    --data_dirs pwm://folding@${RECIPE} \
    --norm_stats_path ${STATS} \
    --robot_features robot_flows,robot_velocity,robot_acceleration \
    --scene_features scene_flows,dist2robot \
    --ptv3_size medium \
    --predictor_dim 192 \
    --sphere_crop false \
    --param_nan_checks false \
    --exp_name ${EXP} \
    --log_dir ${LOG_DIR} \
    --pred_horizon 16 \
    --batch_size ${MICRO} \
    --grad_accum_steps ${ACCUM} \
    --budget_windows 1600000 \
    --num_epochs 100000 \
    --grid_size 0.001 \
    --num_workers 6 \
    --eval_freq 500 \
    --save_freq 2000 \
    --num_eval_batches 25 \
    --snapshot_matched_steps 10000 \
    --matched_windows_per_step 8 \
    ${RESUME}
