#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/cleanup_prepared_tensors_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/cleanup_prepared_tensors_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
cd "${script_dir}"
mkdir -p "${script_dir}/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

if [[ -z "${MANIFEST_PATH:-}" ]]; then
    echo "MANIFEST_PATH must be set before submitting ${BASH_SOURCE[0]}" >&2
    exit 1
fi

prepared_dir="$(
    MANIFEST_PATH="${MANIFEST_PATH}" python3 - <<'PY'
import json
import os

run_dir = os.path.dirname(os.environ["MANIFEST_PATH"])
run_config_path = os.path.join(run_dir, "run_config.json")
with open(run_config_path, "r") as f:
    run_config = json.load(f)
print(run_config["prepared_dir"])
PY
)"

if [[ -z "${prepared_dir}" ]]; then
    echo "Could not resolve prepared_dir from ${MANIFEST_PATH}" >&2
    exit 1
fi

echo "Cleaning prepared tensors at ${prepared_dir}"
if [[ -d "${prepared_dir}" ]]; then
    rm -rf "${prepared_dir}"
    echo "Removed ${prepared_dir}"
else
    echo "Prepared tensor directory already absent: ${prepared_dir}"
fi
