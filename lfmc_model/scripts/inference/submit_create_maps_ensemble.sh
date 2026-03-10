#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model.sh

validation_test="${VALIDATION_TEST:-true}"
max_tiles="${MAX_TILES:-20}"
months_per_block="${MONTHS_PER_BLOCK:-1}"
array_concurrency="${ARRAY_CONCURRENCY:-32}"
time_chunk_days="${TIME_CHUNK_DAYS:-31}"
y_chunk="${Y_CHUNK:-100}"
x_chunk="${X_CHUNK:-100}"
requested_start_date="${REQUESTED_START_DATE:-}"
requested_end_date="${REQUESTED_END_DATE:-2024-12-31}"

manifest_args=(
    --months_per_block "${months_per_block}"
    --time_chunk_days "${time_chunk_days}"
    --y_chunk "${y_chunk}"
    --x_chunk "${x_chunk}"
    --requested_end_date "${requested_end_date}"
)

if [[ -n "${requested_start_date}" ]]; then
    manifest_args+=(--requested_start_date "${requested_start_date}")
fi

if [[ "${validation_test}" == "true" ]]; then
    manifest_args+=(--validation_test)
fi

if [[ -n "${max_tiles}" ]]; then
    manifest_args+=(--max_tiles "${max_tiles}")
fi

echo "Building map manifest..."
python3 -u "${script_dir}/create_map_manifest.py" "${manifest_args[@]}"

latest_run_dir="$(python3 - <<'PY'
import os
from map_runtime_utils import DEFAULT_MAP_RUN_ROOT, latest_run_dir
print(latest_run_dir(DEFAULT_MAP_RUN_ROOT))
PY
)"
manifest_path="${latest_run_dir}/manifest.csv"

num_tasks="$(python3 - <<PY
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(len(df))
PY
)"

if [[ "${num_tasks}" -le 0 ]]; then
    echo "Manifest ${manifest_path} has no tasks" >&2
    exit 1
fi

echo "Submitting ${num_tasks} array tasks from manifest ${manifest_path}"
array_job_id="$(
    sbatch \
        --parsable \
        --array="0-$(( num_tasks - 1 ))%${array_concurrency}" \
        --export=ALL,MANIFEST_PATH="${manifest_path}",MODEL_TYPE=standard \
        "${script_dir}/create_maps_ensemble.sbatch"
)"
echo "Submitted worker array job ${array_job_id}"

merge_job_id="$(
    sbatch \
        --parsable \
        --dependency=afterok:${array_job_id} \
        --export=ALL,MANIFEST_PATH="${manifest_path}",OVERWRITE_MERGE=1 \
        "${script_dir}/merge_maps_ensemble.sbatch"
)"
echo "Submitted merge/validation job ${merge_job_id}"
