#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"
cd "${script_dir}"

source /home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh

if [[ -z "${CONFIG_PATH:-}" ]]; then
    CONFIG_PATH="${script_dir}/map_configs_full_2024.yaml"
fi
export CONFIG_PATH

cfg_value() {
    local section="$1"
    local key="$2"
    local default_value="$3"
    CONFIG_PATH="${CONFIG_PATH}" CFG_SECTION="${section}" CFG_KEY="${key}" CFG_DEFAULT="${default_value}" python3 - <<'PYCFG'
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
PYCFG
}

job_state() {
    local job_id="$1"
    local state=""
    state="$({ sacct -j "${job_id}" --format=State -n -P 2>/dev/null || true; } | awk -F'|' 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1; exit}')"
    if [[ -n "${state}" ]]; then
        printf '%s\n' "${state}"
        return 0
    fi
    state="$({ squeue -h -j "${job_id}" -o "%T" 2>/dev/null || true; } | awk 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1; exit}')"
    if [[ -n "${state}" ]]; then
        printf '%s\n' "${state}"
        return 0
    fi
    printf 'UNKNOWN\n'
}

job_failed() {
    case "$1" in
        BOOT_FAIL|CANCELLED|CANCELLED+|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|TIMEOUT)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

validation_test="${VALIDATION_TEST:-$(cfg_value submission validation_test true)}"
max_tiles="${MAX_TILES:-$(cfg_value submission max_tiles '')}"
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
gpu_submit_sleep_seconds="${GPU_SUBMIT_SLEEP_SECONDS:-$(cfg_value submission gpu_submit_sleep_seconds 60)}"
gpu_time_limit="${GPU_TIME_LIMIT:-$(cfg_value submission gpu_time_limit 02:00:00)}"
gpu_cpus_per_task="${GPU_CPUS_PER_TASK:-$(cfg_value submission gpu_cpus_per_task 4)}"
gpu_mem="${GPU_MEM:-$(cfg_value submission gpu_mem 32G)}"
gpu_constraint="${GPU_CONSTRAINT:-$(cfg_value submission gpu_constraint '')}"
merge_blocks_per_job="${MERGE_BLOCKS_PER_JOB:-$(cfg_value submission merge_blocks_per_job 1)}"
model_type="${MODEL_TYPE:-$(cfg_value ensemble model_type standard)}"

start_from_gpu=false
existing_run_dir=
cleanup_prepared_tensors_after_gpu=false

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

if [[ "${start_from_gpu}" == "true" ]]; then
    latest_run_dir="${existing_run_dir}"
    manifest_path="${latest_run_dir}/manifest.csv"
    run_name="$(basename "${latest_run_dir}")"
    echo "GPU-only restart mode active; reusing run directory ${latest_run_dir}"
    if [[ ! -d "${latest_run_dir}" ]]; then
        echo "Existing run directory does not exist: ${latest_run_dir}" >&2
        exit 1
    fi
    if [[ ! -f "${manifest_path}" ]]; then
        echo "Existing manifest not found: ${manifest_path}" >&2
        exit 1
    fi
    if [[ ! -f "${latest_run_dir}/run_config.json" ]]; then
        echo "Existing run_config.json not found in ${latest_run_dir}" >&2
        exit 1
    fi
else
    echo "Submitting manifest build job..."
    manifest_job_id="$({
        sbatch --parsable --export=ALL,CONFIG_PATH="${CONFIG_PATH}" "${script_dir}/build_map_manifest_ensemble.sbatch" "${manifest_args[@]}"
    })"
    echo "Submitted manifest build job ${manifest_job_id}"

    while true; do
        manifest_job_state="$(job_state "${manifest_job_id}")"
        if job_failed "${manifest_job_state}"; then
            echo "Manifest build job ${manifest_job_id} failed with state ${manifest_job_state}" >&2
            exit 1
        fi
        if [[ "${manifest_job_state}" == "COMPLETED" ]]; then
            break
        fi
        echo "Manifest build monitor: job=${manifest_job_id} state=${manifest_job_state}; sleeping ${gpu_submit_sleep_seconds}s"
        sleep "${gpu_submit_sleep_seconds}"
    done

    echo "Manifest build job ${manifest_job_id} completed"
    latest_run_dir="$(python3 - <<'PYRUN'
import os
from map_config import load_map_config, get_cfg
from map_runtime_utils import latest_run_dir
cfg = load_map_config(os.environ["CONFIG_PATH"])
run_root = os.environ.get("RUN_ROOT") or get_cfg(cfg, "paths", "run_root")
print(latest_run_dir(run_root))
PYRUN
    )"
    manifest_path="${latest_run_dir}/manifest.csv"
    run_name="$(basename "${latest_run_dir}")"
fi
mkdir -p "${gpu_lock_dir}"

lock_count() {
    find "${gpu_lock_dir}" -maxdepth 1 -type f | wc -l
}

num_fine_tasks="$(python3 - <<PYCOUNT
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(len(df))
PYCOUNT
)"

if [[ "${num_fine_tasks}" -le 0 ]]; then
    echo "Manifest ${manifest_path} has no tasks" >&2
    exit 1
fi

num_job_tasks="$(python3 - <<PYCOUNT
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(int(df["job_task_id"].max()) + 1)
PYCOUNT
)"

num_gpu_job_tasks="$(python3 - <<PYCOUNT
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(int(df["gpu_job_task_id"].max()) + 1)
PYCOUNT
)"

num_merge_tasks="$(python3 - <<PYCOUNT
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(int(df["merge_task_id"].max()) + 1)
PYCOUNT
)"

prepared_tensor_count="$(find "${latest_run_dir}/prepared_tensors" -maxdepth 1 -type f | wc -l)"
if [[ "${start_from_gpu}" == "true" ]]; then
    if [[ "${prepared_tensor_count}" -ne "${num_fine_tasks}" ]]; then
        echo "GPU-only restart requested, but prepared tensor count ${prepared_tensor_count} does not match manifest tasks ${num_fine_tasks}" >&2
        exit 1
    fi
    echo "Validated GPU-only restart inputs: prepared_tensors=${prepared_tensor_count}, manifest_tasks=${num_fine_tasks}"
fi

echo "Submitting ${num_job_tasks} array jobs covering ${num_fine_tasks} fine tasks from manifest ${manifest_path}"
echo "tasks_per_job=${tasks_per_job}"
echo "use_gpu_forward=${use_gpu_forward}; gpu_fine_tasks_per_job=${gpu_fine_tasks_per_job}; num_gpu_job_tasks=${num_gpu_job_tasks}"
echo "merge_blocks_per_job=${merge_blocks_per_job}; num_merge_tasks=${num_merge_tasks}"

after_gpu_dependency=""
if [[ "${use_gpu_forward}" == "true" ]]; then
    declare -a gpu_prepare_task_ids
    declare -a gpu_job_ids
    declare -a gpu_submitted_flags

    if [[ "${start_from_gpu}" == "true" ]]; then
        echo "GPU-only restart mode: skipping CPU prepare array and starting directly from prepared tensors in ${latest_run_dir}"
        for (( gpu_task_id=0; gpu_task_id<num_gpu_job_tasks; gpu_task_id++ )); do
            gpu_prepare_task_ids[${gpu_task_id}]="PREBUILT"
            gpu_submitted_flags[${gpu_task_id}]=0
        done
    else
        prepare_job_id="$({
            sbatch --parsable --array="0-$(( num_job_tasks - 1 ))%${array_concurrency}" --export=ALL,MANIFEST_PATH="${manifest_path}" "${script_dir}/prepare_maps_ensemble.sbatch"
        })"
        echo "Submitted CPU prepare array job ${prepare_job_id}"
        echo "Monitoring prepare tasks and submitting GPU jobs under shared lock budget ${gpu_max_jobs} from ${gpu_lock_dir}"

        mapfile -t gpu_dependency_lines < <(python3 - <<PYMAP
import pandas as pd

df = pd.read_csv("${manifest_path}")
grouped = (
    df[["gpu_job_task_id", "job_task_id"]]
    .drop_duplicates()
    .sort_values(["gpu_job_task_id", "job_task_id"])
    .groupby("gpu_job_task_id")["job_task_id"]
)
for gpu_job_task_id, job_task_ids in grouped:
    csv_ids = ",".join(str(int(v)) for v in job_task_ids.astype(int).tolist())
    print(f"{int(gpu_job_task_id)}|{csv_ids}")
PYMAP
        )

        for line in "${gpu_dependency_lines[@]}"; do
            IFS='|' read -r gpu_task_id prepare_task_csv <<< "${line}"
            gpu_prepare_task_ids[${gpu_task_id}]="${prepare_task_csv}"
            gpu_submitted_flags[${gpu_task_id}]=0
        done
    fi

    submitted_gpu_jobs=0
    while [[ "${submitted_gpu_jobs}" -lt "${num_gpu_job_tasks}" ]]; do
        progress_made=0
        blocked_by_prepare=0
        blocked_by_lock=0

        for (( gpu_task_id=0; gpu_task_id<num_gpu_job_tasks; gpu_task_id++ )); do
            if [[ "${gpu_submitted_flags[$gpu_task_id]:-0}" == "1" ]]; then
                continue
            fi

            prepare_task_csv="${gpu_prepare_task_ids[$gpu_task_id]:-}"
            if [[ -z "${prepare_task_csv}" ]]; then
                echo "Missing prepare-task mapping for gpu_job_task_id=${gpu_task_id}" >&2
                exit 1
            fi

            prereqs_ready=1
            if [[ "${start_from_gpu}" != "true" ]]; then
                IFS=',' read -r -a prepare_task_ids <<< "${prepare_task_csv}"
                for prepare_task_id in "${prepare_task_ids[@]}"; do
                    prepare_task_state="$(job_state "${prepare_job_id}_${prepare_task_id}")"
                    if job_failed "${prepare_task_state}"; then
                        echo "Prepare array task ${prepare_job_id}_${prepare_task_id} failed with state ${prepare_task_state}; aborting GPU submission monitor." >&2
                        exit 1
                    fi
                    if [[ "${prepare_task_state}" != "COMPLETED" ]]; then
                        prereqs_ready=0
                        break
                    fi
                done
            fi

            if [[ "${prereqs_ready}" != "1" ]]; then
                blocked_by_prepare=$(( blocked_by_prepare + 1 ))
                continue
            fi

            active_locks="$(lock_count)"
            if [[ "${active_locks}" -ge "${gpu_max_jobs}" ]]; then
                blocked_by_lock=$(( blocked_by_lock + 1 ))
                continue
            fi

            lock_file="${gpu_lock_dir}/lock_${run_name}_gpu_${gpu_task_id}.lock"
            touch "${lock_file}"
            sbatch_gpu_constraint_args=()
            if [[ -n "${gpu_constraint}" ]]; then
                sbatch_gpu_constraint_args+=(--constraint="${gpu_constraint}")
            fi
            if ! gpu_job_id="$({
                sbatch \
                    --parsable \
                    --time="${gpu_time_limit}" \
                    --cpus-per-task="${gpu_cpus_per_task}" \
                    --mem="${gpu_mem}" \
                    "${sbatch_gpu_constraint_args[@]}" \
                    --job-name="map_gpu_${gpu_task_id}" \
                    --export=ALL,MANIFEST_PATH="${manifest_path}",MODEL_TYPE="${model_type}",GPU_TASK_ID="${gpu_task_id}",LOCK_FILE="${lock_file}" \
                    "${script_dir}/run_maps_gpu_ensemble.sbatch"
            })"; then
                rm -f "${lock_file}"
                echo "Failed to submit GPU job for gpu_job_task_id=${gpu_task_id}" >&2
                exit 1
            fi

            gpu_job_ids[$gpu_task_id]="${gpu_job_id}"
            gpu_submitted_flags[$gpu_task_id]=1
            submitted_gpu_jobs=$(( submitted_gpu_jobs + 1 ))
            progress_made=1
            echo "Submitted GPU job ${gpu_job_id} for gpu_job_task_id=${gpu_task_id}; prepare_task_ids=${prepare_task_csv}; active_locks=$(lock_count)/${gpu_max_jobs}; submitted=${submitted_gpu_jobs}/${num_gpu_job_tasks}"
        done

        if [[ "${submitted_gpu_jobs}" -lt "${num_gpu_job_tasks}" ]]; then
            echo "GPU submission monitor: submitted=${submitted_gpu_jobs}/${num_gpu_job_tasks}; blocked_by_prepare=${blocked_by_prepare}; blocked_by_lock=${blocked_by_lock}; active_locks=$(lock_count)/${gpu_max_jobs}; sleeping ${gpu_submit_sleep_seconds}s"
            sleep "${gpu_submit_sleep_seconds}"
        fi
    done

    echo "All GPU jobs submitted. Monitoring for completion before scheduling downstream steps."
    while true; do
        completed_gpu_jobs=0
        running_gpu_jobs=0
        pending_gpu_jobs=0

        for (( gpu_task_id=0; gpu_task_id<num_gpu_job_tasks; gpu_task_id++ )); do
            gpu_job_id="${gpu_job_ids[$gpu_task_id]:-}"
            if [[ -z "${gpu_job_id}" ]]; then
                echo "Missing GPU job id for gpu_job_task_id=${gpu_task_id}" >&2
                exit 1
            fi
            gpu_job_state="$(job_state "${gpu_job_id}")"
            if job_failed "${gpu_job_state}"; then
                echo "GPU job ${gpu_job_id} for gpu_job_task_id=${gpu_task_id} failed with state ${gpu_job_state}; aborting downstream submission." >&2
                exit 1
            fi
            if [[ "${gpu_job_state}" == "COMPLETED" ]]; then
                completed_gpu_jobs=$(( completed_gpu_jobs + 1 ))
            elif [[ "${gpu_job_state}" == "PENDING" || "${gpu_job_state}" == "CONFIGURING" ]]; then
                pending_gpu_jobs=$(( pending_gpu_jobs + 1 ))
            else
                running_gpu_jobs=$(( running_gpu_jobs + 1 ))
            fi
        done

        if [[ "${completed_gpu_jobs}" -eq "${num_gpu_job_tasks}" ]]; then
            break
        fi

        echo "GPU completion monitor: completed=${completed_gpu_jobs}/${num_gpu_job_tasks}; running=${running_gpu_jobs}; pending=${pending_gpu_jobs}; active_locks=$(lock_count)/${gpu_max_jobs}; sleeping ${gpu_submit_sleep_seconds}s"
        sleep "${gpu_submit_sleep_seconds}"
    done

    if [[ "${cleanup_prepared_tensors_after_gpu}" == "true" ]]; then
        cleanup_job_id="$({
            sbatch --parsable --job-name="cleanup_prepared_tensors" --export=ALL,MANIFEST_PATH="${manifest_path}" "${script_dir}/cleanup_prepared_tensors.sbatch"
        })"
        echo "Submitted prepared-tensor cleanup job ${cleanup_job_id}"
    else
        echo "Skipping prepared-tensor cleanup after GPU stage."
    fi
else
    array_job_id="$({
        sbatch --parsable --array="0-$(( num_job_tasks - 1 ))%${array_concurrency}" --export=ALL,MANIFEST_PATH="${manifest_path}",MODEL_TYPE="${model_type}" "${script_dir}/create_maps_ensemble.sbatch"
    })"
    echo "Submitted worker array job ${array_job_id}"
    after_gpu_dependency="${array_job_id}"
fi

merge_init_args=(
    --parsable
)
if [[ -n "${after_gpu_dependency}" ]]; then
    merge_init_args+=(--dependency=afterok:${after_gpu_dependency})
fi
merge_init_args+=(
    --export=ALL,MANIFEST_PATH="${manifest_path}",OVERWRITE_MERGE=1,MERGE_INITIALIZE_ONLY=1
    "${script_dir}/merge_maps_ensemble.sbatch"
)
merge_init_job_id="$(sbatch "${merge_init_args[@]}")"
echo "Submitted merge initialization job ${merge_init_job_id}"

merge_array_job_id="$({
    sbatch --parsable --dependency=afterok:${merge_init_job_id} --array="0-$(( num_merge_tasks - 1 ))%${array_concurrency}" --export=ALL,MANIFEST_PATH="${manifest_path}" "${script_dir}/merge_maps_ensemble.sbatch"
})"
echo "Submitted merge array job ${merge_array_job_id}"

validate_job_id="$({
    sbatch --parsable --dependency=afterok:${merge_array_job_id} --export=ALL,MANIFEST_PATH="${manifest_path}" "${script_dir}/validate_maps_ensemble.sbatch"
})"
echo "Submitted validation job ${validate_job_id}"
