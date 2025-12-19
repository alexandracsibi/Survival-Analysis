#!/bin/bash
#SBATCH --job-name=surv_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

PROJECT_DIR=/project/c_gnn42/alexa_thesis
SIF=${PROJECT_DIR}/containers/thesis.sif

CFG=$1

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/runs

echo "Running config: ${CFG}"
echo "Project dir: ${PROJECT_DIR}"
echo "Container: ${SIF}"

module load singularity

srun singularity exec --nv \
  --bind ${PROJECT_DIR}:/workspace \
  ${SIF} \
  bash -lc "cd /workspace && python -u train.py --config '${CFG}'"
