#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

if [[ -z "${CONFIG_PATH:-}" ]]; then
    CONFIG_PATH="${script_dir}/map_configs.yaml"
fi

cfg_value() {
    local section="$1"
    local key="$2"
    local default_value="$3"
    CONFIG_PATH="${CONFIG_PATH}" CFG_SECTION="${section}" CFG_KEY="${key}" CFG_DEFAULT="${default_value}" python3 - <<'PY'
import os
from map_config import get_cfg, load_map_config

cfg = load_map_config(os.environ["CONFIG_PATH"])
section = os.environ["CFG_SECTION"]
key = os.environ["CFG_KEY"]
default_value = os.environ["CFG_DEFAULT"]
value = get_cfg(cfg, section, key, default=default_value)
if value is None:
    print("")
elif isinstance(value, bool):
    print(str(value).lower())
else:
    print(value)
PY
}

validation_test="${VALIDATION_TEST:-$(cfg_value submission validation_test true)}"
max_tiles="${MAX_TILES:-$(cfg_value submission max_tiles 20)}"
months_per_block="${MONTHS_PER_BLOCK:-$(cfg_value chunking months_per_block 1)}"
time_chunk_days="${TIME_CHUNK_DAYS:-$(cfg_value chunking time_chunk_days 31)}"
y_chunk="${Y_CHUNK:-$(cfg_value chunking y_chunk 100)}"
x_chunk="${X_CHUNK:-$(cfg_value chunking x_chunk 100)}"
requested_start_date="${REQUESTED_START_DATE:-$(cfg_value data requested_start_date '')}"
requested_end_date="${REQUESTED_END_DATE:-$(cfg_value data requested_end_date 2024-12-31)}"
array_concurrency="${ARRAY_CONCURRENCY:-$(cfg_value submission array_concurrency 32)}"
tasks_per_job="${TASKS_PER_JOB:-$(cfg_value submission tasks_per_job 1)}"
use_gpu_forward="${USE_GPU_FORWARD:-$(cfg_value submission use_gpu_forward false)}"
gpu_fine_tasks_per_job="${GPU_FINE_TASKS_PER_JOB:-$(cfg_value submission gpu_fine_tasks_per_job 1)}"
gpu_max_jobs="${GPU_MAX_JOBS:-$(cfg_value submission gpu_max_jobs 8)}"
gpu_lock_dir="${GPU_LOCK_DIR:-$(cfg_value submission gpu_lock_dir /scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/gpu_locks)}"
gpu_submit_sleep_seconds="${GPU_SUBMIT_SLEEP_SECONDS:-$(cfg_value submission gpu_submit_sleep_seconds 30)}"
gpu_time_limit="${GPU_TIME_LIMIT:-$(cfg_value submission gpu_time_limit 02:00:00)}"
gpu_cpus_per_task="${GPU_CPUS_PER_TASK:-$(cfg_value submission gpu_cpus_per_task 4)}"
gpu_mem="${GPU_MEM:-$(cfg_value submission gpu_mem 32G)}"
merge_blocks_per_job="${MERGE_BLOCKS_PER_JOB:-$(cfg_value submission merge_blocks_per_job 1)}"
model_type="${MODEL_TYPE:-$(cfg_value ensemble model_type standard)}"

manifest_args=(
    --config_path "${CONFIG_PATH}"
    --months_per_block "${months_per_block}"
    --time_chunk_days "${time_chunk_days}"
    --y_chunk "${y_chunk}"
    --x_chunk "${x_chunk}"
    --requested_end_date "${requested_end_date}"
)

if [[ -n "${requested_start_date}" ]]; then
    manifest_args+=(--requested_start_date "${requested_start_date}")
fi

if [[ -n "${ENSEMBLE_ROOT:-}" ]]; then
    manifest_args+=(--ensemble_root "${ENSEMBLE_ROOT}")
fi

if [[ -n "${INPUT_DATA_NAME:-}" ]]; then
    manifest_args+=(--input_data_name "${INPUT_DATA_NAME}")
fi

if [[ -n "${INPUTS_ROOT:-}" ]]; then
    manifest_args+=(--inputs_root "${INPUTS_ROOT}")
fi

if [[ -n "${RUN_ROOT:-}" ]]; then
    manifest_args+=(--run_root "${RUN_ROOT}")
fi

if [[ -n "${GRID_PATH:-}" ]]; then
    manifest_args+=(--grid_path "${GRID_PATH}")
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
from map_config import load_map_config, get_cfg
from map_runtime_utils import latest_run_dir
cfg = load_map_config(os.environ["CONFIG_PATH"])
run_root = os.environ.get("RUN_ROOT") or get_cfg(cfg, "paths", "run_root")
print(latest_run_dir(run_root))
PY
)"
manifest_path="${latest_run_dir}/manifest.csv"

num_fine_tasks="$(python3 - <<PY
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(len(df))
PY
)"

if [[ "${num_fine_tasks}" -le 0 ]]; then
    echo "Manifest ${manifest_path} has no tasks" >&2
    exit 1
fi

num_job_tasks="$(python3 - <<PY
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(int(df["job_task_id"].max()) + 1)
PY
)"

num_gpu_job_tasks="$(python3 - <<PY
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(int(df["gpu_job_task_id"].max()) + 1)
PY
)"

num_merge_tasks="$(python3 - <<PY
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(int(df["merge_task_id"].max()) + 1)
PY
)"

echo "Submitting ${num_job_tasks} array jobs covering ${num_fine_tasks} fine tasks from manifest ${manifest_path}"
echo "tasks_per_job=${tasks_per_job}"
echo "use_gpu_forward=${use_gpu_forward}; gpu_fine_tasks_per_job=${gpu_fine_tasks_per_job}; num_gpu_job_tasks=${num_gpu_job_tasks}"
echo "merge_blocks_per_job=${merge_blocks_per_job}; num_merge_tasks=${num_merge_tasks}"

upstream_dependency=""
if [[ "${use_gpu_forward}" == "true" ]]; then
    prepare_job_id="$(
        sbatch \
            --parsable \
            --array="0-$(( num_job_tasks - 1 ))%${array_concurrency}" \
            --export=ALL,MANIFEST_PATH="${manifest_path}" \
            "${script_dir}/prepare_maps_ensemble.sbatch"
    )"
    echo "Submitted CPU prepare array job ${prepare_job_id}"
    mkdir -p "${gpu_lock_dir}"
    gpu_job_ids=()
    for (( gpu_task_id=0; gpu_task_id<num_gpu_job_tasks; gpu_task_id++ )); do
        while [[ "$(find "${gpu_lock_dir}" -type f | wc -l)" -ge "${gpu_max_jobs}" ]]; do
            echo "Found $(find "${gpu_lock_dir}" -type f | wc -l) active GPU locks. Waiting."
            sleep "${gpu_submit_sleep_seconds}"
        done
        lock_file="${gpu_lock_dir}/lock_$(basename "$(dirname "${manifest_path}")")_gpu_${gpu_task_id}.lock"
        touch "${lock_file}"
        gpu_job_id="$(
            sbatch \
                --parsable \
                --dependency=afterok:${prepare_job_id} \
                --time="${gpu_time_limit}" \
                --cpus-per-task="${gpu_cpus_per_task}" \
                --mem="${gpu_mem}" \
                --job-name="map_gpu_${gpu_task_id}" \
                --export=ALL,MANIFEST_PATH="${manifest_path}",GPU_TASK_ID="${gpu_task_id}",MODEL_TYPE="${model_type}",LOCK_FILE="${lock_file}" \
                "${script_dir}/run_maps_gpu_ensemble.sbatch"
        )"
        gpu_job_ids+=("${gpu_job_id}")
        echo "Submitted GPU forward job ${gpu_job_id} for gpu_job_task_id=${gpu_task_id}"
        sleep "${gpu_submit_sleep_seconds}"
    done
    upstream_dependency="$(IFS=:; echo "${gpu_job_ids[*]}")"
else
    array_job_id="$(
        sbatch \
            --parsable \
            --array="0-$(( num_job_tasks - 1 ))%${array_concurrency}" \
            --export=ALL,MANIFEST_PATH="${manifest_path}",MODEL_TYPE="${model_type}" \
            "${script_dir}/create_maps_ensemble.sbatch"
    )"
    echo "Submitted worker array job ${array_job_id}"
    upstream_dependency="${array_job_id}"
fi

merge_init_job_id="$(
    sbatch \
        --parsable \
        --dependency=afterok:${upstream_dependency} \
        --export=ALL,MANIFEST_PATH="${manifest_path}",OVERWRITE_MERGE=1,MERGE_INITIALIZE_ONLY=1 \
        "${script_dir}/merge_maps_ensemble.sbatch"
)"
echo "Submitted merge initialization job ${merge_init_job_id}"

merge_array_job_id="$(
    sbatch \
        --parsable \
        --dependency=afterok:${merge_init_job_id} \
        --array="0-$(( num_merge_tasks - 1 ))%${array_concurrency}" \
        --export=ALL,MANIFEST_PATH="${manifest_path}" \
        "${script_dir}/merge_maps_ensemble.sbatch"
)"
echo "Submitted merge array job ${merge_array_job_id}"

validate_job_id="$(
    sbatch \
        --parsable \
        --dependency=afterok:${merge_array_job_id} \
        --export=ALL,MANIFEST_PATH="${manifest_path}" \
        "${script_dir}/validate_maps_ensemble.sbatch"
)"
echo "Submitted validation job ${validate_job_id}"
