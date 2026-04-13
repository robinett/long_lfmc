#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857"
dataset_sbatch="${script_dir}/build_viewer_dataset_3857.sbatch"
assets_sbatch="${script_dir}/build_viewer_assets.sbatch"
logs_dir="${script_dir}/logs"

mkdir -p "${logs_dir}"
cd "${script_dir}"

dataset_job_id="$(sbatch --parsable "${dataset_sbatch}")"
assets_job_id="$(sbatch --parsable --dependency=afterok:${dataset_job_id} "${assets_sbatch}")"

echo "Submitted viewer dataset build job ${dataset_job_id}"
echo "Submitted viewer asset build job ${assets_job_id} (afterok:${dataset_job_id})"
