#!/bin/bash
#SBATCH --job-name=surv_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.log

set -euo pipefail

PROJECT_DIR=/project/c_gnn42/alexa_thesis
SIF=${PROJECT_DIR}/containers/thesis.sif

CFG=$1

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/runs

SEEDS=(42 43 44 45 46 47 48 49 50 51)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Running config: ${CFG}"
echo "Seed: ${SEED}"
echo "Project dir: ${PROJECT_DIR}"
echo "Container: ${SIF}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

module load singularity

srun singularity exec --nv \
  --bind ${PROJECT_DIR}:/workspace \
  ${SIF} \
  bash -lc "cd /workspace && python -u train.py --config '${CFG}' --seed ${SEED}"
