#!/bin/bash

#SBATCH --job-name=daymet_vars_and_anoms
#SBATCH --output=./logs/daymet_vars_and_anoms_%j.out
#SBATCH --error=./logs/daymet_vars_and_anoms_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
cd "$SCRIPT_DIR"

mkdir -p ./logs

source ~/uv_activations/activate_lfmc_process_py312.sh

python3 -u build_daymet_derived_products.py "$@"

echo Daymet vars-and-anoms workflow step complete
