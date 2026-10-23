#!/bin/bash
#SBATCH --job-name=mppi_mould_alpha
#SBATCH --output=/data/user_data/mbronars/slurm/output/mppi_mould_alpha_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/mppi_mould_alpha_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=60G
#SBATCH --time=8:00:00
#SBATCH --partition=preempt

# ── Apples-to-apples mppi-in-WM alphabet baseline. Mirrors
#    pwm_efficient/sweep/run_mppi_eval_worker.py's protocol exactly:
#      MASTER_SEED = 42  → per-letter rotation drawn from
#                         np.random.default_rng(SeedSequence([42, ord(L)]))
#                         .uniform(-40, 40)            ← bit-identical, verified
#      planner seed = 0
#      one demo per letter (--demo_idx 0)
#      K=128, chunk=32, T=150, K_la=5, temperature=0.3
#      letter_density_scale=0.75, letter_coverage_frac=0.75
#      letter_fit_rot_range=40, letter_coverage_weight=1.0
#      letter_patch_frac=0.2
#    The same chosen_rot_deg flows through fit_letter_field → identical
#    target field → directly comparable final_chamfer between the two
#    planners on a per-letter basis.

set -e
mkdir -p /data/user_data/mbronars/slurm/output /data/user_data/mbronars/slurm/error

PW=/data/user_data/mbronars/packages/PointWorld
PY=/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python

cd $PW

SWEEP=mould_alphabet_seed42
LETTERS="A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"

echo "=== mppi_mould alphabet sweep ==="
echo "start    : $(date)"
echo "letters  : $LETTERS"
echo "checkpoint: $PW/train_logs/mould_new_6k_1gpu/model-last-copy.pt"
echo "output    : $PW/inference_results/$SWEEP/letter_<L>/"
echo "GPU       :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

for LETTER in $LETTERS; do
  # Derive chosen_rot_deg exactly the way
  # pwm_efficient/sweep/run_mppi_eval_worker.chosen_rot_for_letter does:
  #   np.random.default_rng(SeedSequence([42, ord(L)])).uniform(-40, 40)
  ROT=$($PY -c "
import numpy as np
rng = np.random.default_rng(np.random.SeedSequence([42] + [ord(c) for c in '$LETTER']))
print(f'{rng.uniform(-40.0, 40.0):.6f}')
")

  OUT_NAME=$SWEEP/letter_$LETTER
  echo
  echo "── letter $LETTER   chosen_rot=$ROT   $(date +%T) ──"

  $PY mppi_mould.py \
    --checkpoint $PW/train_logs/mould_new_6k_1gpu/model-last-copy.pt \
    --data_dir   /data/group_data/katefgroup/VLA/3DPCWM/mould_new_6k_data_pointworld \
    --test_dir   /data/group_data/katefgroup/VLA/3DPCWM/deformable_mould_10k/test \
    --output_name $OUT_NAME \
    --inference_root $PW/inference_results \
    --demo_idx 0 \
    --seed 0 \
    --population_size 128 --chunk_size 32 \
    --temperature 0.3 \
    --total_frames 150 --lookahead_commit_size 5 \
    --letter $LETTER \
    --letter_chosen_rot_deg $ROT \
    --letter_density_scale 0.75 \
    --letter_coverage_frac 0.75 \
    --letter_coverage_weight 1.0 \
    --letter_fit_rot_range 40.0 \
    --letter_patch_frac 0.2
done

echo
echo "finish: $(date)"
