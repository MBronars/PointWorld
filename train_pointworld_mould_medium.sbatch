#!/bin/bash
#SBATCH --job-name=pw_mould_med
#SBATCH --output=/data/group_data/katefgroup/runs_pw_mbronars/slurm/pw_mould_med_%j.out
#SBATCH --error=/data/group_data/katefgroup/runs_pw_mbronars/slurm/pw_mould_med_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32,babel-q5-28,babel-n5-28
# ── PointWorld baseline on pile manipulation (mould), 99M "medium" backbone ──────
#
# DATA. Streamed live from gpwm's loader (pwm://mould) with the data roots of the pile gpwm run's recipe
# (unet_mould_drop50_100m.yaml): deformable_mould_new_6k_wds, 5,904 train / 96 held-out episodes of 240 frames.
# gpwm checkout = pwm_final_wt/baselines (mb/baselines has the mould domain). Windows of 17 frames (1 given +
# 16 predicted); scene = the 1,024 dough points, gripper = the 256 paddle points. Uniform window starts (the pile
# gpwm run has no motion weighting either).
#
# MODEL. --ptv3_size medium --predictor_dim 192 = 99.1M parameters. 1 mm grid (0.25% of dough points share a
# cell; every point keeps its own position). No sphere crop, no per-step parameter NaN scans.
#   gripper 9  = position 3 + velocity 3 + acceleration 3
#   scene   20 = position 3 + distance to the gripper 17
#
# BATCH / BUDGET (user, 2026-10-24). 2 GPUs x micro-batch 8 = 16 windows per step x 100,000 steps = 1.6M windows,
# the action-conditioned windows of the pile gpwm run (200k x 16 at 50% action dropout).
#
# RULER. Pile-only fit (20,000 windows, fit_pwm_norm_stats.py, job 10911018).
#
# CHECKPOINTS. latest.pt every 2,000 micro-batches per GPU (2,000 steps); matched_step_NNNNNN.pt every 10,000 steps.
# 160 GB host memory and PW_NCCL_TIMEOUT_MIN=60 (the folding run's fixes). This job does not self-chain.
set -uo pipefail
MICRO=${MICRO:-8}
ACCUM=${ACCUM:-1}
RECIPE=${RECIPE:-/data/user_data/mbronars/launch_pwm_data_v1_baselines/configs/unet_mould_drop50_100m.yaml}
RECIPE_SHA256=6e41965cba8ee869cf5be03700d9b57977a4ea976f6f47777d9bd7f3bfe7e5d1
STATS=${STATS:-/data/group_data/katefgroup/runs_pw_mbronars/mould_stream/norm_stats}
EXP=${EXP:-pointworld_mould_medium}
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
export PWM_ROOT=/data/user_data/mbronars/packages/pwm_final_wt/baselines    # gpwm checkout with the mould domain
export WANDB_DIR=${LOG_DIR}/${EXP}
mkdir -p "${WANDB_DIR}"
echo "============================================================"
echo "PointWorld baseline -- pile manipulation (mould, streamed), medium 99M, 16-frame horizon, no sphere crop"
echo "Date: $(date)   Job: ${SLURM_JOB_ID}   Node: $(hostname)"
echo "2 GPUs x micro-batch ${MICRO} x accumulation ${ACCUM} = $((2 * MICRO * ACCUM)) windows/step; budget 1,600,000 windows = 100,000 steps"
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
    --domains pickup_mould \
    --data_dirs pwm://mould@${RECIPE} \
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
    --matched_windows_per_step 16 \
    ${RESUME}
