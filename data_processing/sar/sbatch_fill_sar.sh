#!/bin/bash

#SBATCH --job-name=fill_sar  # Job name
#SBATCH --output=./logs/sar_stats_%j.out       # Output log file (%j = job ID)
#SBATCH --error=./logs/sar_stats_%j.err        # Error log file
#SBATCH --time=12:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc    # Partition name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --mem=256GB                  # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL  # email me
#SBATCH --mail-user=trobinet@stanford.edu

source ~/.bashrc
source ~/uv_activations/activate_lfmc_process.sh

python3 -u fill_sar.py

echo Processing Complete
