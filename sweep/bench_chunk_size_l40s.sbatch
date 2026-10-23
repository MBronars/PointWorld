#!/bin/bash
#SBATCH --job-name=cc_chunkbench
#SBATCH --output=/data/user_data/mbronars/logs/cc_chunkbench_%j.out
#SBATCH --error=/data/user_data/mbronars/logs/cc_chunkbench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --partition=general

# Walk chunk_size on a single cc scenario (T=150, K=128 — worst case) on a
# fresh L40S 46GB. For each chunk size:
#   - run once to warm spconv JIT / kernel caches  (discard timing)
#   - run again, report wall (this is the production-relevant number)
# Stop at the first OOM. Final summary table at the end.

set -uo pipefail
mkdir -p /data/user_data/mbronars/logs
cd /data/user_data/mbronars/packages/PointWorld

echo "=== L40S chunk_size benchmark  job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

PYTHON=/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python
CKPT=/data/user_data/mbronars/packages/PointWorld/train_logs/mould_new_6k_1gpu/model-eval-cc.pt
DATA_DIR=/data/group_data/katefgroup/VLA/3DPCWM/mould_new_6k_data_pointworld
TEST_DIR=/data/group_data/katefgroup/VLA/3DPCWM/deformable_mould_new_6k_wds/test
SCN=/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid/seed_42/ep_001.npz

read DX DY DEMO < <($PYTHON -c "
import numpy as np
d = np.load('$SCN')
print(f'{d[\"target_offset_xy\"][0]:.6f} {d[\"target_offset_xy\"][1]:.6f} {int(d[\"batch_idx\"])}')
")
echo "scenario: demo_idx=$DEMO  offset=($DX,$DY)  T=150  K=128"
echo ""

# Track warm wall + peak GPU mem per chunk, write to a summary file.
SUMMARY=/data/user_data/mbronars/logs/cc_chunkbench_${SLURM_JOB_ID}.summary
echo "chunk warm_s peak_mem_MiB rc" > "$SUMMARY"

run_one () {
  local CHUNK=$1
  local TAG=$2
  local OUTNAME=/tmp/cc_chunkbench_${SLURM_JOB_ID}/chunk${CHUNK}_${TAG}
  local START=$(date +%s)
  $PYTHON mppi_mould.py \
    --checkpoint    "$CKPT" \
    --data_dir      "$DATA_DIR" \
    --test_dir      "$TEST_DIR" \
    --output_name   "$OUTNAME" \
    --inference_root /tmp \
    --demo_idx      "$DEMO" \
    --seed 0 \
    --objective centroid \
    --centroid_offset_x "$DX" \
    --centroid_offset_y "$DY" \
    --total_frames 150 \
    --lookahead_commit_size 5 \
    --population_size 128 \
    --chunk_size "$CHUNK" \
    --temperature 0.3 \
    2>&1 | tail -3
  local RC=${PIPESTATUS[0]}
  local END=$(date +%s)
  echo "  [chunk=$CHUNK $TAG] rc=$RC wall=$((END - START))s"
  echo $((END - START))
  return $RC
}

for CHUNK in 32 64 96 128; do
  echo "================================"
  echo "  chunk_size = $CHUNK"
  echo "================================"

  # Cold pass (warmup) — discard timing.
  COLD_S=$(run_one "$CHUNK" cold | tail -1)
  RC_COLD=$?
  if [ $RC_COLD -ne 0 ]; then
    PEAK=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "  cold OOM/fail — stopping. peak mem snapshot=${PEAK} MiB"
    echo "$CHUNK NA $PEAK $RC_COLD" >> "$SUMMARY"
    break
  fi

  # Warm pass — record this as the production number.
  WARM_S=$(run_one "$CHUNK" warm | tail -1)
  RC_WARM=$?
  PEAK=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ $RC_WARM -ne 0 ]; then
    echo "  warm OOM/fail (after cold OK) — stopping. peak=${PEAK} MiB"
    echo "$CHUNK NA $PEAK $RC_WARM" >> "$SUMMARY"
    break
  fi
  echo "  ✓ chunk=$CHUNK  warm=${WARM_S}s  peak_mem=${PEAK} MiB"
  echo "$CHUNK $WARM_S $PEAK $RC_WARM" >> "$SUMMARY"
done

echo ""
echo "==================== SUMMARY ===================="
column -t "$SUMMARY"
echo ""
echo "(warm = second run at that chunk size; spconv JIT cache + kernel cache populated)"
