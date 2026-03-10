#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/merge_maps_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/logs/merge_maps_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
cd "${script_dir}"
mkdir -p "${script_dir}/logs"

source /home/users/trobinet/uv_activations/activate_lfmc_model.sh

if [[ -z "${MANIFEST_PATH:-}" ]]; then
    echo "MANIFEST_PATH must be set before submitting ${BASH_SOURCE[0]}" >&2
    exit 1
fi

overwrite_flag=""
if [[ "${OVERWRITE_MERGE:-0}" == "1" ]]; then
    overwrite_flag="--overwrite"
fi

echo "Merging manifest ${MANIFEST_PATH}"
python3 -u "${script_dir}/merge_map_shards.py" \
    --manifest_path "${MANIFEST_PATH}" \
    ${overwrite_flag}

run_root="$(dirname "${MANIFEST_PATH}")"
echo "Validating run ${run_root}"
python3 -u "${script_dir}/validate_map_outputs.py" \
    --run_root "${run_root}"
