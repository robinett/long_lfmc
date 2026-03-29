#!/bin/bash

#SBATCH --job-name=gedi_canopy_height
#SBATCH --output=./logs/gedi_canopy_height_%j.out
#SBATCH --error=./logs/gedi_canopy_height_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB

set -euo pipefail

source ~/uv_activations/activate_lfmc_process_py312.sh

python3 -u build_gedi_canopy_height.py

echo GEDI canopy-height build complete
