#!/bin/bash
#SBATCH --job-name=pw_fit_stats
#SBATCH --account=cis260367p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=62G
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pointworld/slurm/%x_%j.out
# Per-domain PointWorld normalization statistics for the PSC specialists (as psc/norm_stats/folding: the unified recipe,
# 17-frame clips = 1 + 16 predicted, 20,000 windows). DOMAINS: PointWorld domain names, one fit each, in parallel.
#   sbatch --export=ALL,DOMAINS="pickup_rl_push:rl_push pickup_yikai_small:yikai" psc/fit_norm_stats.psc.sbatch
set -uo pipefail
PW_REPO=${PW_REPO:-/ocean/projects/cis260367p/mbronars/projects/GPWM/baselines/PointWorld}
GPWM_REPO=${GPWM_REPO:-/ocean/projects/cis260367p/mbronars/projects/GPWM/gpwm_worktrees/psc-port}
COMPACT=${COMPACT:-/ocean/projects/cis260367p/mbronars/projects/GPWM/data}
DOMAINS=${DOMAINS:?set DOMAINS="<pointworld domain>:<out folder> ..."}
RECIPE=$GPWM_REPO/launch/psc_root/configs/unet_joint8_500m_clean4_muon.yaml
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate pointworld-env
cd "$PW_REPO"
export PWM_ROOT=$GPWM_REPO PWM_PYTORCH3D=$PW_REPO/pwm_shims
export PWM_RELEASE_FROM=/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1 PWM_RELEASE_TO=$COMPACT
n=$(echo $DOMAINS | wc -w); w=$(( SLURM_CPUS_PER_TASK / n ))
for spec in $DOMAINS; do
  d=${spec%%:*}; out=${spec#*:}
  python fit_pwm_norm_stats.py --domains $d --recipe "$RECIPE" --pred-horizon 16 --windows 20000 --workers $w \
    --out psc/norm_stats/$out > "psc/norm_stats/$out/fit.log" 2>&1 &
done
wait; rc=0
for spec in $DOMAINS; do out=${spec#*:}; tail -3 "psc/norm_stats/$out/fit.log"; [ -f "psc/norm_stats/$out/norm_stats.json" ] || rc=1; done
exit $rc
