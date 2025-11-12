#!/bin/bash

#SBATCH --job-name=stats  # Job name
#SBATCH --output=./logs/slurm-%j.out       # Output log file (%j = job ID)
#SBATCH --error=./logs/slurm-%j.err        # Error log file
#SBATCH --time=06:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc            # Partition name
#SBATCH --mem=128GB                 # Memory per node
#SBATCH --mail-user=trobinet@stanford.edu

source ~/.bashrc
source ~/uv_activations/activate_lfmc_process.sh

python3 -u compute_krisha_stats.py

echo Processing Complete
