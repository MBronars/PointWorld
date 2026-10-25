#!/bin/bash
#SBATCH --job-name=pw_gpwm
#SBATCH --account=cis260367p
#SBATCH --partition=GPU-shared
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l40s-48:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pointworld/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pointworld/slurm/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# PointWorld baseline for gpwm on PSC Bridges-2. One script, two runs (RUN=folding_medium | pickup_open); every
# setting matches the babel launcher of the same run (train_pointworld_folding_medium.sbatch /
# train_pointworld_pickup_open.sbatch), only paths and the scheduler lines differ. CONFIRM the partition and gres
# lines on Bridges-2 before the first submit (PSC_SETUP.md). Resubmit the same command to resume from latest.pt.
#   sbatch --job-name=pw_folding_medium --export=ALL,RUN=folding_medium psc/train_pointworld.psc.sbatch
set -uo pipefail
RUN=${RUN:?set RUN=folding_medium or RUN=pickup_open}
PW_REPO=${PW_REPO:-/ocean/projects/cis260367p/mbronars/projects/GPWM/baselines/PointWorld}
GPWM_REPO=${GPWM_REPO:-/ocean/projects/cis260367p/mbronars/projects/GPWM/gpwm_worktrees/psc-port}     # gpwm, branch mb/psc-port
COMPACT=${COMPACT:-/ocean/projects/cis260367p/mbronars/projects/GPWM/data}             # pwm-data-v1-compact
LOG_DIR=${LOG_DIR:-/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pointworld}
CONDA_ENV=${CONDA_ENV:-pointworld-env}
NPROC=${NPROC:-2}   # GPUs (the windows per step assume 2)
RECIPE=$GPWM_REPO/launch/psc_root/configs/unet_joint8_500m_clean4_muon.yaml
RECIPE_SHA256=1e308cbd62d5aab5bf0dda530d3f5384a9a57a1af3d7d4035df853bb7e49f4d2
case $RUN in
  folding_medium)  # 99M medium, no sphere crop, 2 GPUs x 4 = 8 windows/step x 200k = 1.6M windows (= gpwm folding)
    EXP=pointworld_folding_medium; DOMAIN=pickup_folding; DATA="pwm://folding@$RECIPE"; STATS=$PW_REPO/psc/norm_stats/folding
    MICRO=4; ACCUM=1; BUDGET=1600000; MATCHED=8
    EXTRA="--ptv3_size medium --predictor_dim 192 --sphere_crop false --param_nan_checks false
           --robot_features robot_flows,robot_velocity,robot_acceleration --scene_features scene_flows,dist2robot" ;;
  pickup_open)     # released backbone, binary gripper_open, 2 GPUs x 8 x 2 = 32 windows/step x 200k = 6.4M windows
    EXP=pointworld_pickup_open; DOMAIN=pickup_release; DATA="pwm://pickup@$RECIPE"; STATS=$PW_REPO/psc/norm_stats/pickup_open
    MICRO=8; ACCUM=2; BUDGET=6400000; MATCHED=32
    EXTRA="--robot_features robot_flows,gripper_open,robot_velocity,robot_acceleration --scene_features scene_flows,gripper_open,dist2robot" ;;
  *) echo "unknown RUN=$RUN" >&2; exit 2 ;;
esac
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
cd "$PW_REPO"
echo "${RECIPE_SHA256}  ${RECIPE}" | sha256sum --check --quiet || { echo "recipe changed: $RECIPE" >&2; exit 2; }
[ -f "$STATS/norm_stats.json" ] || { echo "no $STATS/norm_stats.json" >&2; exit 2; }
export PWM_ROOT=$GPWM_REPO PWM_PYTORCH3D=$PW_REPO/pwm_shims
export PWM_RELEASE_FROM=/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1 PWM_RELEASE_TO=$COMPACT
export PW_NCCL_TIMEOUT_MIN=60 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_DIR=$LOG_DIR/$EXP
mkdir -p "$LOG_DIR/$EXP" "$LOG_DIR/slurm"
RESUME=""; [ -f "$LOG_DIR/$EXP/latest.pt" ] && RESUME="--model_path $LOG_DIR/$EXP/latest.pt"
RESTART=${SLURM_RESTART_COUNT:-0}; MAX_RESTARTS=${MAX_RESTARTS:-3}
echo "$(date) job ${SLURM_JOB_ID:-local} $(hostname) RUN=$RUN EXP=$EXP restart=$RESTART resume=${RESUME:-none}"; nvidia-smi -L
torchrun --standalone --nproc_per_node=$NPROC train.py --distributed true \
  --domains $DOMAIN --data_dirs "$DATA" --norm_stats_path "$STATS" \
  --exp_name $EXP --log_dir "$LOG_DIR" --pred_horizon 16 \
  --batch_size $MICRO --grad_accum_steps $ACCUM --budget_windows ${BUDGET_OVERRIDE:-$BUDGET} --num_epochs 100000 \
  --grid_size 0.001 --num_workers 6 --eval_freq 500 --save_freq 2000 --num_eval_batches 25 \
  --snapshot_matched_steps 10000 --matched_windows_per_step $MATCHED $EXTRA $RESUME
rc=$?
# A crash requeues the job, which resumes from latest.pt (saved every ~28 min). 2026-10-24: rank 1 took a SIGBUS after
# 8.6 h (the only one on PSC so far; not memory, /dev/shm was empty), most likely a Lustre read failing on a mapped
# page. The time limit kills this script and is not requeued; MAX_RESTARTS stops a crash loop.
if [ $rc -ne 0 ] && [ -n "${SLURM_JOB_ID:-}" ] && [ "$RESTART" -lt "$MAX_RESTARTS" ]; then
  echo "$(date) train.py exited $rc: requeue $((RESTART + 1)) of $MAX_RESTARTS"; scontrol requeue "$SLURM_JOB_ID"
fi
exit $rc
