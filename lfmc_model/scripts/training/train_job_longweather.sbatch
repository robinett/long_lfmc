#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/lfmc_longweather_%j.out
#SBATCH --error=logs/lfmc_longweather_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
mkdir -p logs

source /home/users/trobinet/uv_activations/activate_lfmc_model.sh

lock_file="${LOCK_FILE:-}"
if [[ -n "${lock_file}" ]]; then
  cleanup() { rm -f "${lock_file}"; }
  trap cleanup EXIT INT TERM
fi

python3 -u train_multitarget_longweather_vvvh.py "$@"

