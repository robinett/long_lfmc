#!/bin/bash

#SBATCH --job-name=soilgrids_build
#SBATCH --output=./logs/soilgrids_build_%j.out
#SBATCH --error=./logs/soilgrids_build_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

set -euo pipefail

source ~/uv_activations/activate_lfmc_process_py312.sh

python3 -u build_soilgrids.py

echo SoilGrids build complete
