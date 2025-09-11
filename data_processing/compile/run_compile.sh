#!/bin/bash

#SBATCH --job-name=compile  # Job name
#SBATCH --output=./logs/slurm-%j.out       # Output log file (%j = job ID)
#SBATCH --error=./logs/slurm-%j.err        # Error log file
#SBATCH --time=03:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc    # Partition name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --mem=128GB                  # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL  # email me
#SBATCH --mail-user=trobinet@stanford.edu

source ~/.bashrc
source ~/poetry_activations/activate_long_lfmc_process_env.sh

python3 -u main.py --start_date $1 --end_date $2

echo Processing Complete
