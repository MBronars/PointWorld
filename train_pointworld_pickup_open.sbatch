#!/bin/bash
#SBATCH --job-name=pw_pickup_open
#SBATCH --output=/data/group_data/katefgroup/runs_pw_mbronars/slurm/pw_pickup_open_%j.out
#SBATCH --error=/data/group_data/katefgroup/runs_pw_mbronars/slurm/pw_pickup_open_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32,babel-n5-32,babel-q5-28
# ── PointWorld baseline on pickup only, with the binary gripper label ─────────
#
# The reference PWM model on the 25-problem pick benchmark is pickup-only, so this is its
# PointWorld counterpart: pickup alone, 32 windows per optimizer step, 200,000 steps from scratch.
#
# DATA. Streamed live from the PWM loader (nothing copied) with the pickup settings of the
# two-domain recipe: the same train root, exclusion manifest and shift_m. yikai is in that recipe
# but is not read: --data_dirs names pwm://pickup only.
#
# FEATURES. Upstream's gripper_open is back, as a BINARY label: 1 open, 0 closed, closed when the
# finger-centroid gap is under 97 mm (the rule the Isaac replay commands the gripper by: mean
# finger joint q < 0.035 m; pwm_stream.GRIPPER_OPEN_MIN_GAP_M). It sits on every gripper point
# (that frame's value) and on every scene point (the window's 17 values). No normals, no colours.
#   gripper 10 = position 3 + gripper_open 1 + velocity 3 + acceleration 3
#   scene   37 = position 3 + gripper_open 17 + distance to the gripper 17
#
# HORIZON. --pred_horizon 16: 16 predicted frames from 1 given frame (T = 17).
#
# BATCH. 2 GPUs x micro-batch 8 x 2 accumulation = 32 windows per optimizer step. Measured on
# 2 L40S (2026-10-09, babel-u5-24): 13.37 windows/s, 42.1 GB of 46 per GPU. The other split,
# micro-batch 4 x 4 accumulation, ran 11.79 windows/s at 26.4 GB: MICRO=4 ACCUM=4 if memory is short.
#
# BUDGET. --budget_windows 6,400,000 = exactly 200,000 optimizer steps. At 13.37 windows/s that is
# ~133 h, so three 48 h legs; resubmit this script to resume from latest.pt.
#
# RULER. Its own pickup-only fit with the label's layout (20,000 windows of 17 frames;
# fit_pwm_norm_stats.py --gripper-open). The label's columns are mean 0 / variance 1: read as 0/1.
#
# CHECKPOINTS. latest.pt every 2,000 micro-batches per GPU (32,000 windows, 1,000 steps);
# matched_step_NNNNNN.pt every 10,000 optimizer steps (--matched_windows_per_step 32), 20 in all.
#
# This job does not self-chain.
set -uo pipefail
MICRO=${MICRO:-8}
ACCUM=${ACCUM:-2}
RECIPE=${RECIPE:-/data/user_data/mbronars/launch_pwm_data_v1_two/configs/unet_two_ctxgen_100m_uncond.yaml}
RECIPE_SHA256=b455a5e5f797fc9d8fb3b09cabb4baf64a0a818751e4618601e10052026dd229
STATS=${STATS:-/data/group_data/katefgroup/runs_pw_mbronars/pickup_open_stream/norm_stats}
EXP=${EXP:-pointworld_pickup_open}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pw_mbronars}
CKPT=${LOG_DIR}/${EXP}/latest.pt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env
cd /data/user_data/mbronars/packages/PointWorld
[ -n "${RECIPE_SHA256}" ] && { echo "${RECIPE_SHA256}  ${RECIPE}" | sha256sum --check --quiet \
    || { echo "recipe ${RECIPE} changed since this script was written" >&2; exit 2; }; }
[ -f "${STATS}/norm_stats.json" ] || { echo "no norm_stats.json in ${STATS}" >&2; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# pointworld-env has no pytorch3d; PWM's adapters reach two pure-tensor functions in it for
# pose channels PointWorld discards. See pwm_shims/README.md.
export PWM_PYTORCH3D=/data/user_data/mbronars/packages/PointWorld/pwm_shims
export WANDB_DIR=${LOG_DIR}/${EXP}
mkdir -p "${WANDB_DIR}"
echo "============================================================"
echo "PointWorld baseline -- pickup only (streamed), binary gripper label, 16-frame horizon"
echo "Date: $(date)   Job: ${SLURM_JOB_ID}   Node: $(hostname)"
echo "2 GPUs x micro-batch ${MICRO} x accumulation ${ACCUM} = $((2 * MICRO * ACCUM)) windows/step; budget 6,400,000 windows = 200,000 steps"
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
    --domains pickup_release \
    --data_dirs pwm://pickup@${RECIPE} \
    --norm_stats_path ${STATS} \
    --robot_features robot_flows,gripper_open,robot_velocity,robot_acceleration \
    --scene_features scene_flows,gripper_open,dist2robot \
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
