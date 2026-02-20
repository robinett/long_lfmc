#!/bin/bash

#SBATCH --job-name=daymet_to_daily
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

source ~/.bashrc
source ~/uv_activations/activate_lfmc_process_py312.sh

export YEAR=$1
python3 -u to_daily.py

echo Daily conversion complete for year $YEAR
