#!/bin/bash

#SBATCH --job-name=regridder_batch  # Job name
#SBATCH --output=./logs/slurm-%j.out       # Output log file (%j = job ID)
#SBATCH --error=./logs/slurm-%j.err        # Error log file
#SBATCH --time=02:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc    # Partition name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --mem=128GB                  # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL  # email me
#SBATCH --mail-user=trobinet@stanford.edu

source ~/.bashrc
source ~/uv_activations/activate_lfmc_process_py312.sh

# Generic wrapper for a single regridding invocation.
python3 -u main.py --target_grid "$1" --src_dir "$2" --target_dir "$3" --src_crs "$4" --target_crs "$5" --chunk_buffer "$6"

echo Processing Complete
