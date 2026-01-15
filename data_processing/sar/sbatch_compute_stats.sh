#!/bin/bash

#SBATCH --job-name=sar_stats  # Job name
#SBATCH --output=./logs/slurm-%j.out       # Output log file (%j = job ID)
#SBATCH --error=./logs/slurm-%j.err        # Error log file
#SBATCH --time=12:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc,konings    # Partition name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --mem=200GB                  # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL  # email me
#SBATCH --mail-user=trobinet@stanford.edu

source ~/.bashrc
source ~/uv_activations/activate_lfmc_process.sh

python3 -u compute_sar_stats.py

echo Processing Complete
