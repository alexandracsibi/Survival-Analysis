#!/bin/bash
#SBATCH --job-name=surv_agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --output=/project/c_gnn42/alexa_thesis/logs/%x_%j.log

set -euo pipefail

PROJECT_DIR=/project/c_gnn42/alexa_thesis
SIF=${PROJECT_DIR}/containers/thesis.sif
BASE_NAME=$1

module load singularity

singularity exec \
  --bind ${PROJECT_DIR}:/workspace \
  ${SIF} \
  bash -lc "cd /workspace && python -u tools/aggregate_multiseed.py --base_name ${BASE_NAME}"
