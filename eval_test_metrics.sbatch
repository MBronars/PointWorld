#!/bin/bash
#SBATCH --job-name=eval_test_metrics
#SBATCH --output=/data/user_data/mbronars/slurm/output/eval_test_metrics_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/eval_test_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=general

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd /data/user_data/mbronars/packages/PointWorld

# Override N from CLI:  sbatch eval_test_metrics.sbatch 1024
N="${1:-256}"

python eval_test_metrics.py \
    --checkpoint train_logs/pushing_no_normals/model-last.pt \
    --data-dir   /data/group_data/katefgroup/VLA/3DPCWM/pushing_data_pointworld_05_03 \
    --domain     pushing \
    --n-samples  "${N}" \
    --batch-size 32 \
    --num-workers 2 \
    --seed       0 \
    --out-dir    eval_logs/pushing_no_normals_test_N${N}
