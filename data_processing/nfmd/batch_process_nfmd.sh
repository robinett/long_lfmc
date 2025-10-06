#!/bin/bash
#SBATCH --job-name=nfmd_process
#SBATCH --partition=serc,konings
#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH --mail-user=trobinet@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Run from the directory where you call `sbatch`
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Activate your env (adjust if your path differs)
source ~/poetry_activations/activate_long_lfmc_process_env.sh

# Helpful run info
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"

# Unbuffered Python so logs stream
export PYTHONUNBUFFERED=1

# Run
srun python3 main.py

echo "End: $(date)"

