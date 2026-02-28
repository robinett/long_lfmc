#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=./logs/build_dataset_longweather_%A_%a.out
#SBATCH --error=./logs/build_dataset_longweather_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

sample_index_paths=(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/sample_index_longweather_2000_2024_lfmc.parquet"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/sample_index_longweather_2000_2024_lfmc_vh.parquet"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/sample_index_longweather_2000_2024_lfmc_vv.parquet"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/sample_index_longweather_2000_2024_lfmc_vh_vv.parquet"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/sample_index_longweather_2000_2024_lfmc_vv_minus_vh.parquet"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/indexes/sample_index_longweather_2000_2024_lfmc_vv_over_vh.parquet"
)
save_dirs=(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_vh"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_vv"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_vh_vv"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_vv_minus_vh"
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_vv_over_vh"
)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
mkdir -p ./logs

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

if [[ ${#sample_index_paths[@]} -ne ${#save_dirs[@]} ]]; then
    echo "Error: sample_index_paths and save_dirs lengths do not match." >&2
    exit 1
fi

idx="${SLURM_ARRAY_TASK_ID}"
if [[ "${idx}" -lt 0 || "${idx}" -ge "${#sample_index_paths[@]}" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID=${idx} out of range." >&2
    exit 1
fi

sample_index_path="${sample_index_paths[$idx]}"
save_dir="${save_dirs[$idx]}"

echo "Running task ${idx} / $(( ${#sample_index_paths[@]} - 1 ))"
echo "sample_index_path=${sample_index_path}"
echo "save_dir=${save_dir}"

python3 -u build_dataset_longweather_direct.py "${sample_index_path}" "${save_dir}"
