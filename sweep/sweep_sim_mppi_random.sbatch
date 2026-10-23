#!/bin/bash
#SBATCH --job-name=sim_mppi_random
#SBATCH --output=/data/user_data/mbronars/logs/sim_mppi_random_%A_%a.out
#SBATCH --error=/data/user_data/mbronars/logs/sim_mppi_random_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-3

set -euo pipefail
cd /data/user_data/mbronars/packages/PointWorld
mkdir -p /data/user_data/mbronars/logs

export SWEEP_NAME=block_pushing_sim_mppi_seed42_random
export SCENARIO_DIFFICULTY=random
export SEEDS=42
export EPISODES_PER_SEED=50
export POPULATION_SIZE=256
export HORIZON=20
export REPLAN_EVERY=5
export TOTAL_STEPS=60

/data/user_data/mbronars/.miniconda3/envs/planning_wrapper/bin/python \
    sweep/run_sim_mppi_baseline_worker.py
