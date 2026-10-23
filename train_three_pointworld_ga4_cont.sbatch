#!/bin/bash
#SBATCH --job-name=pw_three_ga4_cont
#SBATCH --output=/data/user_data/mbronars/logs/pw_three_ga4_cont_%j.out
#SBATCH --error=/data/user_data/mbronars/logs/pw_three_ga4_cont_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32

# ── CONTINUATION of the batch-8 PointWorld run at 32 windows per step ────────
#
# Continues three_pointworld (job 10517218, batch 8, one optimizer step per 8
# windows; stopped by the user 2026-09-22 right after a latest.pt save) at
# --grad_accum_steps 4 = 32 windows per optimizer step, in its OWN folder so the
# batch-8 run's folder is never written to (user decision 2026-09-22).
#   First leg: no latest.pt here yet, so it initialises from INIT_FROM — the
#   batch-8 run's latest.pt, read in place: weights, AdamW state and the window
#   counter, so the 6.42 M-window budget continues where it stopped. Later legs
#   resume from this folder's own latest.pt.
#   Matched snapshots continue here (matched_step_030000 on); 010000 and 020000
#   stay in the batch-8 folder.
#   Verified: a batch-8 checkpoint resumed at accumulation 4 restores 80,000
#   optimizer steps / 640,000 windows, steps only on boundaries, and schedules
#   its next snapshot at 960,000 windows without duplicating one.
#
# The twin of train_three_pointworld.sbatch (job 10517218, batch 8, one optimizer
# step per 8 windows) with --grad_accum_steps 4: four micro-batches of 8 per
# optimizer step = 32 windows per step, the PWM run's own step size, so the
# 6.42 M-window budget is ~200.7 k optimizer steps instead of ~803 k (user
# decision 2026-09-22: batch 16 does not fit an L40S, so accumulate).
# Everything else is unchanged, including the constant AdamW learning rate 1e-4.
# eval_freq / save_freq still count MICRO-batches (500 / 2,000 = 125 / 500
# optimizer steps, the same wall-clock cadence as the batch-8 run); evals, saves
# and matched snapshots fire only on accumulation boundaries, so no checkpoint
# holds a half-accumulated gradient. Own output folder: the batch-8 run is left
# running, untouched.
#
# The baseline for the PWM run pwm_v1_three_ctxgen_100m (job 10515960): same
# three corpora, same round robin, same shared ruler, per
# /data/group_data/katefgroup/runs_pwm_mbronars/datasets/yikai_small/TRAINING.md.
#
# DATA IS NOT COPIED. Each --data_dirs entry is a `pwm://<domain>@<recipe>` spec,
# so windows are read live from the PWM loader driven by that run's own recipe
# (pwm_stream.py). The exclusion manifests, pickup's shift_m, yikai_small's
# per-episode table height and rl_push's load-time point construction are the PWM
# run's, not a re-implementation, and they cannot drift. Windows are drawn fresh
# every epoch rather than frozen at dump time.
#   Measured: 1.13 it/s streaming against 1.14 it/s from dumped shards — the
#   loader is not the bottleneck (one reader makes ~59 windows/s, training eats 9).
#   pwm_three_to_pointworld_wds.py remains the way to freeze a corpus to shards.
#
# ROUND ROBIN. PointWorld mixes the three with wds.RandomMix at equal frequency,
# which is the equal expected share of batches TRAINING.md asks for. Batches stay
# domain-pure.
#
# DOMAIN NAMES contain "pickup" on purpose: PointWorld dispatches on that
# substring in four places and the pickup path is the one these windows want —
# gripper points in the sample, no URDF sampler, no cameras.
#
# SHARED RULER. norm_stats/norm_stats.json holds ONE pooled set of statistics
# written under all three domain keys (fit_pwm_norm_stats.py) — the PointWorld
# spelling of the shared ruler. --per-domain fits the 2.4x alternative.
#
# HORIZON. PointWorld's release horizon is fixed at 1 context + 10 prediction;
# the PWM run predicts 16. The baseline trains at its own horizon and is chained
# at evaluation time to reach the protocol's.
#
# BUDGET. The PWM run sees 200,000 steps x 32 windows = 6.4 M windows. A nominal
# epoch here is 110,729 windows (one per indexed clip: 17,661 + 42,253 + 50,815),
# so NUM_EPOCHS=58 is 6.42 M windows. The trainer counts `sample_count` in
# WINDOWS, carries it in the checkpoint and recomputes epoch_count from it, so
# the budget is independent of batch size and GPU count.
#
# CHECKPOINTS. latest.pt is the rolling file a resubmitted leg resumes from,
# rewritten every 2,000 steps of the leg (~29 min at 1.13 it/s) through a
# temporary file so a death mid-write cannot destroy it; model-last.pt is a
# symlink to it for the repo's other scripts. Permanent snapshots land every
# 10,000 steps of MATCHED budget — 320,000 windows, the PWM run's 10k x 32 — as
# matched_step_NNNNNN.pt, 20 of them over the full budget at ~4.9 GB each. The
# cadence is on windows seen, so it is unmoved by batch size or GPU count.
#
# LEGS AND RESUME. This job does not self-chain: it runs to the wall and stops.
# Resubmit to resume from latest.pt. A leg may use a different batch size than
# the one before it — the optimizer is a plain constant-LR AdamW with no schedule
# to desynchronize, and the budget counter is in windows. Verified by resuming one
# checkpoint at batch 8 -> 4 (sample_count continued 64 -> 68 -> 72 ...).
#   MULTI-GPU IS NOT VERIFIED: a 2-GPU DDP start hangs in this checkout — rank 0
#   in dist.new_group (trainer.py:111), rank 1 in broadcast_object_list
#   (trainer.py:223) — and it reproduces on dumped shards, so it is not the
#   pwm:// source. Fix that before a GPUS=2 leg (which also needs
#   --gres=gpu:L40S:2 and --cpus-per-task=16).
#
# At 1.13 it/s and batch 8 on one L40S, a 48 h leg is ~195k steps = 1.56 M
# windows, so the full budget is about 4 legs. Batch 16 does not fit an L40S
# (45.4 of 46 GB at the peak).

set -uo pipefail

GPUS=${GPUS:-1}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-4}
NUM_EPOCHS=${NUM_EPOCHS:-58}
RECIPE=${RECIPE:-/data/user_data/mbronars/launch_pwm_data_v1_three/configs/unet_three_ctxgen_100m.yaml}
STATS=${STATS:-/data/group_data/katefgroup/runs_pw_mbronars/three_stream/norm_stats}
EXP=${EXP:-three_pointworld_ga4_cont}
INIT_FROM=${INIT_FROM:-/data/group_data/katefgroup/runs_pw_mbronars/three_pointworld/latest.pt}
# Checkpoints go to group_data: this model is 409.8 M parameters, so model +
# AdamW moments is ~5 GB a file, and user_data is at 98%.
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pw_mbronars}
CKPT=${LOG_DIR}/${EXP}/latest.pt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env
cd /data/user_data/mbronars/packages/PointWorld

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# pointworld-env has no pytorch3d; PWM's yikai adapter reaches two pure-tensor
# functions in it for pose channels PointWorld discards. See pwm_shims/README.md.
export PWM_PYTORCH3D=/data/user_data/mbronars/packages/PointWorld/pwm_shims
export WANDB_DIR=${LOG_DIR}/${EXP}
mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "PointWorld baseline -- yikai_small + pickup + rl_push (streamed)"
echo "Date: $(date)   Job: ${SLURM_JOB_ID}   Node: $(hostname)"
echo "GPUs: ${GPUS}   batch/GPU: ${BATCH}   accumulation: ${ACCUM} -> $((GPUS * BATCH * ACCUM)) windows/step   epochs: ${NUM_EPOCHS} (x 110,729 windows)"
echo "Recipe: ${RECIPE}"
echo "Checkpoints: ${LOG_DIR}/${EXP}"
nvidia-smi -L
echo "============================================================"

RESUME=""
if [ -f "${CKPT}" ]; then
    RESUME="--model_path ${CKPT}"
    echo "[$(date)] Resuming from ${CKPT}"
elif [ -f "${INIT_FROM}" ]; then
    RESUME="--model_path ${INIT_FROM}"
    mkdir -p "${LOG_DIR}/${EXP}"
    # provenance of the starting point, written once, beside this run's checkpoints
    [ -f "${LOG_DIR}/${EXP}/init_from.txt" ] || \
        echo "initialised from ${INIT_FROM} ($(stat -c '%s bytes, modified %y' "${INIT_FROM}")) by job ${SLURM_JOB_ID}" \
        > "${LOG_DIR}/${EXP}/init_from.txt"
    echo "[$(date)] First leg: initialising from ${INIT_FROM}"
else
    echo "[$(date)] Neither ${CKPT} nor INIT_FROM=${INIT_FROM} exists: refusing a silent fresh start" >&2
    exit 2
fi

ARGS="--domains pickup_yikai_small,pickup_release,pickup_rl_push \
    --data_dirs pwm://yikai@${RECIPE},pwm://pickup@${RECIPE},pwm://rl_push@${RECIPE} \
    --norm_stats_path ${STATS} \
    --robot_features robot_flows,robot_velocity,robot_acceleration \
    --scene_features scene_flows,dist2robot \
    --exp_name ${EXP} \
    --log_dir ${LOG_DIR} \
    --batch_size ${BATCH} \
    --grad_accum_steps ${ACCUM} \
    --grid_size 0.001 \
    --num_workers 6 \
    --num_epochs ${NUM_EPOCHS} \
    --eval_freq 500 \
    --save_freq 2000 \
    --snapshot_matched_steps 10000 \
    --num_eval_batches 25 \
    ${RESUME}"

if [ "${GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${GPUS}" --master_port=$((29500 + SLURM_JOB_ID % 1000)) \
        train.py ${ARGS} --distributed true
else
    python train.py ${ARGS}
fi

status=$?
echo "[$(date)] train.py exited with status ${status}"
exit ${status}
