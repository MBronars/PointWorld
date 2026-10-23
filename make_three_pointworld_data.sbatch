#!/bin/bash
#SBATCH --job-name=pw_three_data
#SBATCH --output=/data/user_data/mbronars/logs/pw_three_data_%j.out
#SBATCH --error=/data/user_data/mbronars/logs/pw_three_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --exclude=babel-o9-20,babel-o9-24,babel-p9-32

# ── Dump yikai_small + pickup + rl_push into PointWorld WDS shards ───────────
#
# CPU only: pwm_three_to_pointworld_wds.py drives the PWM loader and writes tars,
# no GPU involved. Three domains run concurrently, 4 reader workers each (12 of
# the 15 CPUs), because one reader process peaks around 30 clips/s and the three
# are independent.
#
# Measured on the smoke dump (2026-09-20, babel-q5-20): 28 clips/s yikai_small,
# 38 pickup, 35 rl_push with 4 workers, and 18 GB of shards per 100k clips.
# 150k clips per domain is therefore about 1.5 h wall and 81 GB, on group_data
# (user_data is at 97%).
#
# Normalization statistics are accumulated on every 8th clip (--stats-every) and
# written per domain as stats_raw.json; merge_norm_stats.py pools the three into
# the one shared ruler afterwards. That last step is part of this job.

set -euo pipefail

OUT=/data/group_data/katefgroup/VLA/3DPCWM/three_pointworld
TRAIN_CLIPS=${TRAIN_CLIPS:-150000}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pwm211
cd /data/user_data/mbronars/packages/PointWorld

echo "============================================================"
echo "PointWorld three-domain dump"
echo "Date: $(date)   Job: ${SLURM_JOB_ID}   Node: $(hostname)"
echo "Out: ${OUT}   train clips per domain: ${TRAIN_CLIPS}"
echo "============================================================"

for d in yikai pickup rl_push; do
    (
        python pwm_three_to_pointworld_wds.py --domain "$d" --split train \
            --target-clips "${TRAIN_CLIPS}" --num-workers 4 --out "${OUT}" --log-every 20000
        # The held-out split is one deterministic pass: one pinned window per
        # episode, so no clip budget applies (yikai_small 334, pickup 200,
        # rl_push 200 episodes).
        python pwm_three_to_pointworld_wds.py --domain "$d" --split test \
            --num-workers 2 --out "${OUT}" --log-every 20000
    ) > "/data/user_data/mbronars/logs/pw_three_data_${SLURM_JOB_ID}_${d}.log" 2>&1 &
done
wait

echo "--- per-domain logs ---"
for d in yikai pickup rl_push; do
    tail -6 "/data/user_data/mbronars/logs/pw_three_data_${SLURM_JOB_ID}_${d}.log"
done

echo "--- shared ruler ---"
python merge_norm_stats.py --out "${OUT}"

echo "--- on disk ---"
du -sh "${OUT}"/*
echo "[$(date)] done"
