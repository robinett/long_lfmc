#!/bin/bash

#SBATCH --job-name=compile  # Job name
#SBATCH --output=./logs/slurm-%j.out       # Output log file (%j = job ID)
#SBATCH --error=./logs/slurm-%j.err        # Error log file
#SBATCH --time=04:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc            # Partition name
#SBATCH --mem=512GB                 # Memory per node
#SBATCH --mail-user=trobinet@stanford.edu

source ~/.bashrc
source ~/poetry_activations/activate_long_lfmc_process_env.sh

python3 -u interpolate_ds_time.py --start_date $1 --end_date $2

echo Processing Complete
