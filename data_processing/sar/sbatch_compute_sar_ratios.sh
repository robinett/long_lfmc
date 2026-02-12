#!/bin/bash

#SBATCH --job-name=sar_ratios  # Job name
#SBATCH --output=./logs/sar_ratios_%j.out       # Output log file (%j = job ID)
#SBATCH --error=./logs/sar_ratios_%j.err        # Error log file
#SBATCH --time=24:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc    # Partition name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --mem=256GB                  # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL  # email me
#SBATCH --mail-user=trobinet@stanford.edu

source ~/.bashrc
source ~/uv_activations/activate_lfmc_model.sh

python3 -u compute_sar_ratios.py

echo Processing Complete
