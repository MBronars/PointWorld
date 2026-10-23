#!/bin/bash
#SBATCH --job-name=cc_pw
#SBATCH --output=/data/user_data/mbronars/logs/cc_pw_%A_%a.out
#SBATCH --error=/data/user_data/mbronars/logs/cc_pw_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --array=0-3

# ── PointWorld cloud_centroid eval worker ──────────────────────────────────
# Claim-and-go parallel worker for one (SEEDS, TOTAL_FRAMES) configuration.
# Each array task is one worker; they race for the same scenarios via mkdir-
# atomic lock files, so the work is naturally distributed.
#
# Required env (set via `sbatch --export=ALL,SEEDS=...,TOTAL_FRAMES=...`):
#   SEEDS           e.g. "0" or "42"  (which seed sweep this job covers)
#   TOTAL_FRAMES    e.g. "100" or "150"  (planner horizon)
# Optional env:
#   N_PER_SEED      default 50
#   POPULATION_SIZE default 128
#   PLANNER_SEED    default 0
#   RUN_REPLAY      "1" (default — Genesis MPM playback for sim metric)
#
# Headline metric: sim_centroid_dist (m) per scenario in
#   /data/group_data/katefgroup/datasets/mbronars/cloud_centroid_pw_seed<S>_T<T>/
#   seed_<S>/ep_<NN>/replay/sim_centroid.json
#
# A symlink under PointWorld/inference_results/<sweep>/ also points there.

set -euo pipefail
mkdir -p /data/user_data/mbronars/logs

cd /data/user_data/mbronars/packages/PointWorld

# Default per-config sweep name embeds the seed and horizon for clarity.
: "${SEEDS:?SEEDS env var required}"
: "${TOTAL_FRAMES:?TOTAL_FRAMES env var required}"
export N_PER_SEED="${N_PER_SEED:-50}"
export POPULATION_SIZE="${POPULATION_SIZE:-128}"
export CHUNK_SIZE="${CHUNK_SIZE:-128}"
# Sweep name now embeds K and the GPU class so we don't clobber prior runs.
export SWEEP_NAME="${SWEEP_NAME:-cloud_centroid_pw_seed${SEEDS}_T${TOTAL_FRAMES}_K${POPULATION_SIZE}_l40s}"
export TEMPERATURE="${TEMPERATURE:-0.3}"
export LOOKAHEAD_COMMIT_SIZE="${LOOKAHEAD_COMMIT_SIZE:-5}"
export PLANNER_SEED="${PLANNER_SEED:-0}"
export RUN_REPLAY="${RUN_REPLAY:-1}"
export CHECKPOINT="${CHECKPOINT:-/data/user_data/mbronars/packages/PointWorld/train_logs/mould_new_6k_1gpu/model-eval-cc.pt}"

echo "=== array task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT} ==="
echo "host        : $(hostname)"
echo "GPU         :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "SEEDS       : ${SEEDS}"
echo "TOTAL_FRAMES: ${TOTAL_FRAMES}"
echo "N_PER_SEED  : ${N_PER_SEED}"
echo "SWEEP_NAME  : ${SWEEP_NAME}"
echo "POPULATION  : ${POPULATION_SIZE}"
echo "CHUNK_SIZE  : ${CHUNK_SIZE}"
echo "CLAMP_XY    : ${CENTROID_CLAMP_XY:-(none)}"
echo "CHECKPOINT  : ${CHECKPOINT}"
echo "================================="

/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python \
    sweep/run_cloud_centroid_eval_worker_pw.py
