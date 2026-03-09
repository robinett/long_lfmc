#!/bin/bash

#SBATCH --job-name=sar_stats  # Job name
#SBATCH --output=/home/users/trobinet/long_lfmc/data_processing/sar/logs/sar_samples_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/data_processing/sar/logs/sar_samples_%j.err
#SBATCH --time=12:00:00             # Wall time limit (hh:mm:ss)
#SBATCH --partition=serc    # Partition name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --mem=200GB                  # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL  # email me
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

cd /home/users/trobinet/long_lfmc/data_processing/sar
mkdir -p /home/users/trobinet/long_lfmc/data_processing/sar/logs

source ~/uv_activations/activate_lfmc_process_py312.sh

python3 -u select_sar_samples.py "$@"

echo Processing Complete
