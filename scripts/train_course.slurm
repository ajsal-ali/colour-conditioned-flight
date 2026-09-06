#!/bin/bash
#SBATCH --job-name=mavrl_course
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=50G
#SBATCH --time=36:00:00
#SBATCH --output=logs/course_%j.out
#SBATCH --error=logs/course_%j.err

# Stage 4: PPO on the full course, warm-started from runs/course/hamm.pt.
#
# hamm.pt is 3.58M steps into the run that failed by memorising one colour
# order: entry window clean, then a fixed altitude held straight through every
# bar. The encoder and the value head are worth keeping; what it never learned
# is the rule. Three things in this launch are aimed at that, and nothing else
# about the task changes:
#
#   --stage 3 --curriculum   25 render workers, 25 DIFFERENT layouts, redrawn
#                            every 5 rollouts. The old run had one layout for
#                            200 rollouts, which made memorising it optimal.
#   --ent-coef 0.005         entropy was hardcoded to 0 and decayed 1.68 -> 0.21
#                            with no plateau. hamm.pt already starts at 0.60.
#   --expert-frac 0.25       a quarter of the envs flown by the scripted pilot,
#                            decaying to zero over 1M steps, their transitions
#                            trained under the self-imitation term (--sil-coef).
#
# --memory-type MUST be attention: that is what hamm.pt was trained with, and
# the checkpoint load is strict=False, so lstm would start from a random memory
# without complaining. train_course.py now refuses the mismatch outright.
#
# Sizing, from `python -m mavrl.glcheck --profile` on gpu016 (V100, Xeon 6148):
# one env step is 55.5 ms and only 5% of that is the render, so the run is
# CPU-bound and --cpus-per-task is the throughput knob, not the GPU.
#
#   100 envs / 25 render workers = 4 envs per worker, stepped serially
#   4 x 55 ms  =  ~220 ms per policy step across all 100 envs
#              =  ~460 env-steps/s,  ~30-40M steps/day
#
# 25 workers + the trainer is 26 busy cores; 40 leaves room for the observation
# pipeline, which the noise and downsample rework cut from 90 to 57 ms/step.
# If throughput lands well under 460, drop to --n-envs 80 --render-workers 20
# rather than asking for more cores -- the GPU is already near its render
# ceiling (~500 Hz without the segmentation pass, which training does not use).

set -e

echo "===== JOB ====="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/.bashrc
conda activate mavrl
echo "Python: $(which python)"
python --version

cd /scratch/scratch26/23me36008/mujoco_cor
mkdir -p logs

export MUJOCO_GL=egl
unset DISPLAY
# One thread per process. Without this each of the 25 render workers spawns a
# 40-thread OpenMP pool on 40 cores and they spend the run fighting each other.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "===== GL ====="
#python -m mavrl.glcheck

RUN=runs2/course_${SLURM_JOB_ID}

echo "===== TRAIN ====="
python -m mavrl.train_course \
    --out            "$RUN" \
    --init           runs/course/hamm.pt \
    --memory-type    attention \
    --stage          3 \
    --curriculum \
    --split          train \
    --n-envs         100 \
    --render-workers 25 \
    --envs-per-batch 15 \
    --n-steps        128 \
    --ent-coef       0.02 \
    --expert-frac    0.6 \
    --expert-decay   10000000 \
    --sil-coef       0.1 \
    --critic-warmup  100000 \
    --lr             3e-4 \
    --device         cuda \
    --seed           0 \
    --plot-every     10

echo "===== DONE ====="
