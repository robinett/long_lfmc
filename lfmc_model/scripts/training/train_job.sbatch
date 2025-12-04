#!/usr/bin/env bash
#SBATCH --job-name=lfmc_train
#SBATCH --partition=serc
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/lfmc_%j.out
#SBATCH --error=logs/lfmc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL  # email me
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

# Activate your environment
source /home/users/trobinet/uv_activations/activate_lfmc_model.sh

# Forward *all* args to the Python script
python3 -u train_multitarget_longweather_vvvh.py "$@"
