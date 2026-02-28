#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs/lfmc_longweather_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs/lfmc_longweather_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

cd "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training"
mkdir -p "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model.sh

lock_file="${LOCK_FILE:-}"
if [[ -n "${lock_file}" ]]; then
  cleanup() { rm -f "${lock_file}"; }
  trap cleanup EXIT INT TERM
fi

python3 -u "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/train_multitarget_longweather_vvvh.py" "$@"
