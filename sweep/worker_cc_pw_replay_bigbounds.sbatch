#!/bin/bash
#SBATCH --job-name=cc_pw_bb
#SBATCH --output=/data/user_data/mbronars/logs/cc_pw_bb_%A_%a.out
#SBATCH --error=/data/user_data/mbronars/logs/cc_pw_bb_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=0-15

# ── cc_pw bigbounds REPLAY (no planning) ──────────────────────────────────
# Re-runs the Genesis playback for all seed42 L40S cc_pw scenarios using
# expanded MPM bounds (±0.50 m xy, [-0.06, 0.30] z) so MPM boundary
# clamping can't affect the sim metrics. Reads existing
# positions/batch000.npz; writes to <scn>/replay_bigbounds/ without
# touching the original <scn>/replay/.
#
# 16 parallel workers race over 200 scenarios via mkdir-atomic .lock_bb.

set -euo pipefail
mkdir -p /data/user_data/mbronars/logs
cd /data/user_data/mbronars/packages/PointWorld

echo "=== bigbounds replay worker  array task ${SLURM_ARRAY_TASK_ID} ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python \
    sweep/run_cc_pw_replay_bigbounds_worker.py
