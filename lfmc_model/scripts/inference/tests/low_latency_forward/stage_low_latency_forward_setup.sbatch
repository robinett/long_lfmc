#!/bin/bash

#SBATCH --job-name=stage_ll_fwd
#SBATCH --output=/home/users/trobinet/long_lfmc/logs/low_latency_forward_setup/stage_low_latency_forward_setup_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/logs/low_latency_forward_setup/stage_low_latency_forward_setup_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/tests/low_latency_forward"
source /home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh
python3 -u "${script_dir}/prepare_low_latency_forward_setup.py"
