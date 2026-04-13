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

SBATCH_SUBMIT_MAX_ATTEMPTS="${SBATCH_SUBMIT_MAX_ATTEMPTS:-5}"
SBATCH_SUBMIT_RETRY_SLEEP_SECONDS="${SBATCH_SUBMIT_RETRY_SLEEP_SECONDS:-15}"

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

timestamp_slug() {
    date +"%Y%m%d_%H%M%S"
}

submit_with_retry() {
    local label="$1"
    shift
    local attempt=""
    local stdout_log=""
    local stderr_log=""
    local output=""
    local status=0
    for (( attempt=1; attempt<=SBATCH_SUBMIT_MAX_ATTEMPTS; attempt++ )); do
        stdout_log="${logs_dir}/sbatch_${label}_attempt${attempt}_$(timestamp_slug).out"
        stderr_log="${logs_dir}/sbatch_${label}_attempt${attempt}_$(timestamp_slug).err"
        if "$@" >"${stdout_log}" 2>"${stderr_log}"; then
            output="$(awk 'NF {print; exit}' "${stdout_log}")"
            if [[ -z "${output}" ]]; then
                echo "sbatch returned success for ${label} attempt ${attempt}, but no job id was captured. stdout=${stdout_log} stderr=${stderr_log}" >&2
                status=1
            else
                echo "Submitted ${label} on attempt ${attempt}; stdout=${stdout_log} stderr=${stderr_log}" >&2
                printf '%s\n' "${output}"
                return 0
            fi
        else
            status=$?
            echo "sbatch failed for ${label} attempt ${attempt}/${SBATCH_SUBMIT_MAX_ATTEMPTS}; status=${status}; stdout=${stdout_log} stderr=${stderr_log}" >&2
        fi
        if [[ "${attempt}" -lt "${SBATCH_SUBMIT_MAX_ATTEMPTS}" ]]; then
            echo "Retrying ${label} submission after ${SBATCH_SUBMIT_RETRY_SLEEP_SECONDS}s" >&2
            sleep "${SBATCH_SUBMIT_RETRY_SLEEP_SECONDS}"
        fi
    done
    echo "Exhausted sbatch retries for ${label}" >&2
    return 1
}

refresh_runtime_scheduler_limits() {
    local new_gpu_max_jobs=""
    local new_owners_gpu_max_jobs=""
    local new_serc_endgame_minutes_threshold=""

    new_gpu_max_jobs="$(cfg_value submission gpu_max_jobs "${gpu_max_jobs}")"
    new_owners_gpu_max_jobs="$(cfg_value submission owners_gpu_max_jobs "${owners_gpu_max_jobs}")"
    new_serc_endgame_minutes_threshold="$(cfg_value submission serc_endgame_minutes_threshold "${serc_endgame_minutes_threshold}")"

    if [[ "${new_gpu_max_jobs}" != "${gpu_max_jobs}" || "${new_owners_gpu_max_jobs}" != "${owners_gpu_max_jobs}" || "${new_serc_endgame_minutes_threshold}" != "${serc_endgame_minutes_threshold}" ]]; then
        echo "Updated runtime scheduler limits from config: serc_cap=${gpu_max_jobs}->${new_gpu_max_jobs}; owners_cap=${owners_gpu_max_jobs}->${new_owners_gpu_max_jobs}; serc_endgame_minutes_threshold=${serc_endgame_minutes_threshold}->${new_serc_endgame_minutes_threshold}"
    fi

    gpu_max_jobs="${new_gpu_max_jobs}"
    owners_gpu_max_jobs="${new_owners_gpu_max_jobs}"
    serc_endgame_minutes_threshold="${new_serc_endgame_minutes_threshold}"
}

job_state() {
    local job_id="$1"
    local state=""
    state="$({ sacct -j "${job_id}" --format=State -n -P 2>/dev/null || true; } | awk -F'|' 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1; exit}')"
    if [[ -n "${state}" ]]; then
        printf '%s\n' "${state%% *}"
        return 0
    fi
    state="$({ squeue -h -j "${job_id}" -o "%T" 2>/dev/null || true; } | awk 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1; exit}')"
    if [[ -n "${state}" ]]; then
        printf '%s\n' "${state%% *}"
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

state_in_csv() {
    local state="$1"
    local csv_states="$2"
    local retry_state=""
    local retryable_states=()
    IFS=',' read -r -a retryable_states <<< "${csv_states}"
    for retry_state in "${retryable_states[@]}"; do
        retry_state="$(echo "${retry_state}" | tr -d '[:space:]')"
        if [[ -n "${retry_state}" && "${state}" == "${retry_state}" ]]; then
            return 0
        fi
    done
    return 1
}

job_retryable() {
    state_in_csv "$1" "${gpu_retry_states}"
}

prepare_retryable() {
    state_in_csv "$1" "${prepare_retry_states}"
}

hhmmss_to_seconds() {
    local time_value="$1"
    IFS=':' read -r hours minutes seconds <<< "${time_value}"
    printf '%s\n' "$(( 10#${hours} * 3600 + 10#${minutes} * 60 + 10#${seconds} ))"
}

format_dhms() {
    local total_seconds="$1"
    local days=""
    local hours=""
    local minutes=""
    local seconds=""
    if [[ -z "${total_seconds}" || "${total_seconds}" == "NA" ]]; then
        printf 'NA\n'
        return 0
    fi
    total_seconds="${total_seconds%.*}"
    if [[ -z "${total_seconds}" || "${total_seconds}" =~ [^0-9-] || "${total_seconds}" -lt 0 ]]; then
        printf 'NA\n'
        return 0
    fi
    days=$(( total_seconds / 86400 ))
    hours=$(( (total_seconds % 86400) / 3600 ))
    minutes=$(( (total_seconds % 3600) / 60 ))
    seconds=$(( total_seconds % 60 ))
    printf '%d:%02d:%02d:%02d\n' "${days}" "${hours}" "${minutes}" "${seconds}"
}

downshift_gpu_batch_cache_for_job() {
    local job_id="$1"
    local metadata_path="${latest_run_dir}/gpu_work_queue/worker_runtime/job_${job_id}.json"
    if [[ ! -f "${metadata_path}" ]]; then
        return 1
    fi
    JOB_METADATA_PATH="${metadata_path}" python3 - <<'PYDOWN'
import json
import math
import os
import time

metadata_path = os.environ["JOB_METADATA_PATH"]
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

cache_path = metadata.get("cache_path")
selected_batch_size = int(metadata.get("selected_batch_size", 0))
configured_batch_size = int(metadata.get("configured_batch_size", 512))
if not cache_path or selected_batch_size <= 0:
    raise SystemExit(1)

cache_payload = {}
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache_payload = json.load(f)

new_batch_size = max(configured_batch_size, int(math.floor(selected_batch_size * 0.5)))
if new_batch_size >= selected_batch_size:
    new_batch_size = max(configured_batch_size, selected_batch_size - 256)
if new_batch_size <= 0:
    new_batch_size = configured_batch_size

cache_payload["selected_batch_size"] = int(new_batch_size)
cache_payload["last_downshift_reason"] = "OUT_OF_MEMORY"
cache_payload["last_downshift_from"] = int(selected_batch_size)
cache_payload["last_downshift_job_id"] = str(metadata.get("job_id", ""))
cache_payload["updated_at_epoch"] = float(time.time())

os.makedirs(os.path.dirname(cache_path), exist_ok=True)
tmp_path = cache_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(cache_payload, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp_path, cache_path)
print(f"{selected_batch_size}->{new_batch_size}")
PYDOWN
}

estimate_recent_shard_rate_per_second() {
    local completion_log_path
    completion_log_path="$(gpu_completion_log_path)"
    if [[ ! -f "${completion_log_path}" ]]; then
        return 1
    fi
    COMPLETION_LOG_PATH="${completion_log_path}" python3 - <<'PYRATE'
import os
from pathlib import Path

path = Path(os.environ["COMPLETION_LOG_PATH"])
timestamps = []
for line in path.read_text(errors="ignore").splitlines():
    parts = line.split("\t")
    if len(parts) < 5:
        continue
    try:
        timestamps.append(float(parts[0]))
    except ValueError:
        continue

timestamps = timestamps[-128:]
if len(timestamps) < 4:
    raise SystemExit(1)

window_seconds = timestamps[-1] - timestamps[0]
if window_seconds <= 0:
    raise SystemExit(1)

print(len(timestamps) / window_seconds)
PYRATE
}

estimate_multiyear_future_years_and_median_seconds() {
    if [[ -z "${multiyear_status_dir}" || -z "${multiyear_current_year}" || -z "${multiyear_start_year}" ]]; then
        return 1
    fi
    MULTIYEAR_STATUS_DIR="${multiyear_status_dir}" \
    MULTIYEAR_CURRENT_YEAR="${multiyear_current_year}" \
    MULTIYEAR_START_YEAR="${multiyear_start_year}" \
    python3 - <<'PYMULTI'
import json
import os
import statistics
from pathlib import Path

status_dir = Path(os.environ["MULTIYEAR_STATUS_DIR"])
current_year = int(os.environ["MULTIYEAR_CURRENT_YEAR"])
start_year = int(os.environ["MULTIYEAR_START_YEAR"])

remaining_future_years = 0
for year in range(start_year, current_year):
    if not (status_dir / f"year_{year}_completed.json").exists():
        remaining_future_years += 1

elapsed_samples = []
for marker_path in sorted(status_dir.glob("year_*_completed.json")):
    try:
        payload = json.loads(marker_path.read_text())
    except Exception:
        continue
    elapsed = payload.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        elapsed_samples.append(float(elapsed))

median_elapsed = statistics.median(elapsed_samples) if elapsed_samples else None
if median_elapsed is None:
    print(f"{remaining_future_years}\tNA")
else:
    print(f"{remaining_future_years}\t{median_elapsed:.3f}")
PYMULTI
}

count_prepared_reference_payloads() {
    local prepared_dir="${latest_run_dir}/prepared_tensors"
    if [[ ! -d "${prepared_dir}" ]]; then
        printf '0\n'
        return 0
    fi
    find "${prepared_dir}" -maxdepth 1 -type f -name '*_reference.pt' | wc -l
}

count_shards_on_disk() {
    find "${latest_run_dir}/shards" -maxdepth 1 -type f -name '*.npz' | wc -l
}

array_has_failed_tasks() {
    local array_job_id="$1"
    local bad_state=""
    bad_state="$({
        sacct -j "${array_job_id}" --format=State -n -P 2>/dev/null || true
    } | awk -F'|' '
        NF {
            state=$1
            sub(/[[:space:]].*$/, "", state)
            gsub(/^[ \t]+|[ \t]+$/, "", state)
            if (state ~ /^(BOOT_FAIL|CANCELLED|CANCELLED\+|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|REVOKED|TIMEOUT)$/) {
                print state
                exit
            }
        }
    ')"
    if [[ -n "${bad_state}" ]]; then
        printf '%s\n' "${bad_state}"
        return 0
    fi
    return 1
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
max_prepared_ahead_of_completed_shards="${MAX_PREPARED_AHEAD_OF_COMPLETED_SHARDS:-$(cfg_value submission max_prepared_ahead_of_completed_shards 1000)}"
use_gpu_forward="${USE_GPU_FORWARD:-$(cfg_value submission use_gpu_forward false)}"
dynamic_gpu_work_queue="${DYNAMIC_GPU_WORK_QUEUE:-$(cfg_value submission dynamic_gpu_work_queue false)}"
gpu_fine_tasks_per_job="${GPU_FINE_TASKS_PER_JOB:-$(cfg_value submission gpu_fine_tasks_per_job 1)}"
gpu_max_jobs="${GPU_MAX_JOBS:-$(cfg_value submission gpu_max_jobs 8)}"
gpu_lock_dir="${GPU_LOCK_DIR:-$(cfg_value submission gpu_lock_dir /scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/gpu_locks)}"
gpu_submit_sleep_seconds="${GPU_SUBMIT_SLEEP_SECONDS:-$(cfg_value submission gpu_submit_sleep_seconds 60)}"
gpu_time_limit="${GPU_TIME_LIMIT:-$(cfg_value submission gpu_time_limit 02:00:00)}"
gpu_cpus_per_task="${GPU_CPUS_PER_TASK:-$(cfg_value submission gpu_cpus_per_task 4)}"
gpu_mem="${GPU_MEM:-$(cfg_value submission gpu_mem 32G)}"
gpu_constraint="${GPU_CONSTRAINT:-$(cfg_value submission gpu_constraint '')}"
serc_partition="${SERC_PARTITION:-$(cfg_value submission serc_partition serc)}"
owners_gpu_max_jobs="${OWNERS_GPU_MAX_JOBS:-$(cfg_value submission owners_gpu_max_jobs 0)}"
owners_partition="${OWNERS_PARTITION:-$(cfg_value submission owners_partition owners)}"
owners_gpu_time_limit="${OWNERS_GPU_TIME_LIMIT:-$(cfg_value submission owners_gpu_time_limit "${gpu_time_limit}")}"
owners_gpu_cpus_per_task="${OWNERS_GPU_CPUS_PER_TASK:-$(cfg_value submission owners_gpu_cpus_per_task "${gpu_cpus_per_task}")}"
owners_gpu_mem="${OWNERS_GPU_MEM:-$(cfg_value submission owners_gpu_mem "${gpu_mem}")}"
owners_gpu_constraint="${OWNERS_GPU_CONSTRAINT:-$(cfg_value submission owners_gpu_constraint "${gpu_constraint}")}"
gpu_time_margin_seconds="${GPU_TIME_MARGIN_SECONDS:-$(cfg_value submission gpu_time_margin_seconds 600)}"
gpu_claim_stale_seconds="${GPU_CLAIM_STALE_SECONDS:-$(cfg_value submission gpu_claim_stale_seconds 1800)}"
gpu_idle_sleep_seconds="${GPU_IDLE_SLEEP_SECONDS:-$(cfg_value submission gpu_idle_sleep_seconds 15)}"
gpu_next_task_safety_factor="${GPU_NEXT_TASK_SAFETY_FACTOR:-$(cfg_value submission gpu_next_task_safety_factor 1.15)}"
serc_only_endgame="${SERC_ONLY_ENDGAME:-$(cfg_value submission serc_only_endgame false)}"
serc_endgame_minutes_threshold="${SERC_ENDGAME_MINUTES_THRESHOLD:-$(cfg_value submission serc_endgame_minutes_threshold 60)}"
serc_endgame_min_samples="${SERC_ENDGAME_MIN_SAMPLES:-$(cfg_value submission serc_endgame_min_samples 4)}"
serc_endgame_recent_completion_count="${SERC_ENDGAME_RECENT_COMPLETION_COUNT:-$(cfg_value submission serc_endgame_recent_completion_count 32)}"
gpu_max_retries="${GPU_MAX_RETRIES:-$(cfg_value submission gpu_max_retries 20)}"
gpu_retry_states="${GPU_RETRY_STATES:-$(cfg_value submission gpu_retry_states PREEMPTED,REVOKED,BOOT_FAIL,NODE_FAIL)}"
prepare_time_limit="${PREPARE_TIME_LIMIT:-$(cfg_value submission prepare_time_limit 01:00:00)}"
prepare_cpus_per_task="${PREPARE_CPUS_PER_TASK:-$(cfg_value submission prepare_cpus_per_task 8)}"
prepare_mem="${PREPARE_MEM:-$(cfg_value submission prepare_mem 128G)}"
prepare_max_retries="${PREPARE_MAX_RETRIES:-$(cfg_value submission prepare_max_retries 8)}"
prepare_retry_states="${PREPARE_RETRY_STATES:-$(cfg_value submission prepare_retry_states PREEMPTED,REVOKED,BOOT_FAIL,NODE_FAIL,TIMEOUT,OUT_OF_MEMORY)}"
merge_blocks_per_job="${MERGE_BLOCKS_PER_JOB:-$(cfg_value submission merge_blocks_per_job 1)}"
model_type="${MODEL_TYPE:-$(cfg_value ensemble model_type standard)}"
cleanup_prepared_tensors_after_success="${CLEANUP_PREPARED_TENSORS_AFTER_SUCCESS:-$(cfg_value submission cleanup_prepared_tensors_after_success false)}"
wait_for_validation_completion="${WAIT_FOR_VALIDATION_COMPLETION:-$(cfg_value submission wait_for_validation_completion false)}"
multiyear_current_year="${MULTIYEAR_CURRENT_YEAR:-NA}"
multiyear_start_year="${MULTIYEAR_START_YEAR:-}"
multiyear_end_year="${MULTIYEAR_END_YEAR:-}"
multiyear_total_years="${MULTIYEAR_TOTAL_YEARS:-}"
multiyear_year_ordinal="${MULTIYEAR_YEAR_ORDINAL:-}"
multiyear_status_dir="${MULTIYEAR_STATUS_DIR:-}"
multiyear_year_started_at_epoch="${MULTIYEAR_YEAR_STARTED_AT_EPOCH:-}"

start_from_gpu="${START_FROM_GPU:-false}"
existing_run_dir="${EXISTING_RUN_DIR:-}"
auto_resume_complete_run="${AUTO_RESUME_COMPLETE_RUN:-true}"
resume_partial_run="false"
resume_completed_run="false"

find_resumable_run_dir() {
    local run_root="$1"
    RUN_ROOT_FOR_RESUME="${run_root}" python3 - <<'PYRESUME'
import csv
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT_FOR_RESUME"])
if not run_root.exists():
    raise SystemExit(1)

run_dirs = sorted(
    [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not run_dirs:
    raise SystemExit(1)

run_dir = run_dirs[0]
manifest_path = run_dir / "manifest.csv"
run_config_path = run_dir / "run_config.json"
prepared_dir = run_dir / "prepared_tensors"
shards_dir = run_dir / "shards"
completed_log = run_dir / "gpu_work_queue" / "completed_tasks.tsv"

if not manifest_path.exists() or not run_config_path.exists():
    raise SystemExit(1)

with open(run_config_path, "r", encoding="utf-8") as f:
    run_config = json.load(f)

out_zarr_path = run_config.get("out_zarr_path") or run_config.get("persistent_out_zarr_path")
persistent_out_zarr_path = run_config.get("persistent_out_zarr_path")
if (not persistent_out_zarr_path) and out_zarr_path and Path(out_zarr_path).exists():
    raise SystemExit(1)

with open(manifest_path, newline="", encoding="utf-8") as f:
    task_count = sum(1 for _ in csv.DictReader(f))

shard_count = sum(1 for p in shards_dir.glob("*.npz")) if shards_dir.exists() else 0
completed_count = sum(1 for _ in completed_log.open("r", encoding="utf-8")) if completed_log.exists() else 0

if task_count <= 0:
    raise SystemExit(1)
if shard_count != task_count or completed_count != task_count:
    raise SystemExit(1)

print(str(run_dir))
PYRESUME
}

find_partial_run_dir() {
    local run_root="$1"
    RUN_ROOT_FOR_RESUME="${run_root}" python3 - <<'PYPARTIAL'
import csv
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT_FOR_RESUME"])
if not run_root.exists():
    raise SystemExit(1)

run_dirs = sorted(
    [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not run_dirs:
    raise SystemExit(1)

run_dir = run_dirs[0]
manifest_path = run_dir / "manifest.csv"
run_config_path = run_dir / "run_config.json"
prepared_dir = run_dir / "prepared_tensors"
shards_dir = run_dir / "shards"

if not manifest_path.exists() or not run_config_path.exists():
    raise SystemExit(1)

with open(run_config_path, "r", encoding="utf-8") as f:
    run_config = json.load(f)

out_zarr_path = run_config.get("out_zarr_path") or run_config.get("persistent_out_zarr_path")
persistent_out_zarr_path = run_config.get("persistent_out_zarr_path")
if (not persistent_out_zarr_path) and out_zarr_path and Path(out_zarr_path).exists():
    raise SystemExit(1)

with open(manifest_path, newline="", encoding="utf-8") as f:
    task_count = sum(1 for _ in csv.DictReader(f))

prepared_count = sum(1 for p in prepared_dir.glob("*_reference.pt")) if prepared_dir.exists() else 0
shard_count = sum(1 for p in shards_dir.glob("*.npz")) if shards_dir.exists() else 0

if task_count <= 0:
    raise SystemExit(1)
if shard_count <= 0 and prepared_count <= 0:
    raise SystemExit(1)
if shard_count >= task_count:
    raise SystemExit(1)

print(str(run_dir))
PYPARTIAL
}

configured_run_root="${RUN_ROOT:-$(cfg_value paths run_root '')}"
if [[ "${start_from_gpu}" != "true" && -z "${existing_run_dir}" && "${auto_resume_complete_run}" == "true" && -n "${configured_run_root}" ]]; then
    if resumable_run_dir="$(find_resumable_run_dir "${configured_run_root}")"; then
        resume_completed_run="true"
        existing_run_dir="${resumable_run_dir}"
        echo "Detected completed-shard run ready for resume: ${existing_run_dir}"
        echo "Skipping manifest/prepare/GPU submission and resuming at merge/validation."
    elif partial_run_dir="$(find_partial_run_dir "${configured_run_root}")"; then
        existing_run_dir="${partial_run_dir}"
        resume_partial_run="true"
        echo "Detected partially completed run ready for resume: ${existing_run_dir}"
        echo "Skipping manifest rebuild and resuming prepare/GPU scheduling from the existing run."
    fi
fi

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

if [[ -n "${ENSEMBLE_MEMBER_NAME_PREFIX:-}" ]]; then
    manifest_args+=(--ensemble_member_name_prefix "${ENSEMBLE_MEMBER_NAME_PREFIX}")
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

if [[ "${start_from_gpu}" == "true" || "${resume_partial_run}" == "true" || "${resume_completed_run}" == "true" ]]; then
    latest_run_dir="${existing_run_dir}"
    manifest_path="${latest_run_dir}/manifest.csv"
    run_name="$(basename "${latest_run_dir}")"
    if [[ "${resume_completed_run}" == "true" ]]; then
        echo "Completed-run resume mode active; reusing run directory ${latest_run_dir}"
    elif [[ "${start_from_gpu}" == "true" ]]; then
        echo "GPU-only restart mode active; reusing run directory ${latest_run_dir}"
    else
        echo "Partial-run resume active; reusing run directory ${latest_run_dir}"
    fi
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
    persistent_out_zarr_path="$(
        python3 - <<PYRUNCFG
import json
from pathlib import Path
run_config = json.loads(Path("${latest_run_dir}/run_config.json").read_text())
print(run_config.get("persistent_out_zarr_path") or "")
PYRUNCFG
    )"
    out_zarr_path="$(
        python3 - <<PYRUNCFG
import json
from pathlib import Path
run_config = json.loads(Path("${latest_run_dir}/run_config.json").read_text())
print(run_config.get("out_zarr_path") or "")
PYRUNCFG
    )"
else
    echo "Submitting manifest build job..."
    manifest_job_id="$(submit_with_retry \
        "build_map_manifest" \
        sbatch --parsable --export=ALL,CONFIG_PATH="${CONFIG_PATH}" "${script_dir}/build_map_manifest_ensemble.sbatch" "${manifest_args[@]}")"
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
    find "${gpu_lock_dir}" -maxdepth 1 -type f -name "lock_${run_name}_gpu_*.lock" | wc -l
}

gpu_completion_log_path() {
    printf '%s/gpu_work_queue/completed_tasks.tsv\n' "${latest_run_dir}"
}

serc_endgame_flag_path() {
    printf '%s/gpu_work_queue/serc_only_endgame.flag\n' "${latest_run_dir}"
}

estimate_serc_task_seconds() {
    local completion_log_path
    completion_log_path="$(gpu_completion_log_path)"
    if [[ ! -f "${completion_log_path}" ]]; then
        return 1
    fi
    COMPLETION_LOG_PATH="${completion_log_path}" \
    SERC_ENDGAME_MIN_SAMPLES="${serc_endgame_min_samples}" \
    SERC_ENDGAME_RECENT_COMPLETION_COUNT="${serc_endgame_recent_completion_count}" \
    python3 - <<'PYRATE'
import os
import statistics
from pathlib import Path

path = Path(os.environ["COMPLETION_LOG_PATH"])
min_samples = int(os.environ["SERC_ENDGAME_MIN_SAMPLES"])
recent_count = int(os.environ["SERC_ENDGAME_RECENT_COMPLETION_COUNT"])

serc_elapsed = []
for line in path.read_text(errors="ignore").splitlines():
    parts = line.split("\t")
    if len(parts) < 5 or parts[1] != "serc":
        continue
    try:
        serc_elapsed.append(float(parts[4]))
    except ValueError:
        continue

if recent_count > 0:
    serc_elapsed = serc_elapsed[-recent_count:]

if len(serc_elapsed) < min_samples:
    raise SystemExit(1)

print(int(round(statistics.median(serc_elapsed))))
PYRATE
}

activate_serc_only_endgame() {
    local remaining_shards="$1"
    local running_serc_jobs="$2"
    local estimated_task_seconds="$3"
    local estimated_finish_seconds="$4"
    local flag_path
    flag_path="$(serc_endgame_flag_path)"
    mkdir -p "$(dirname "${flag_path}")"
    {
        echo "activated_at_epoch=$(date +%s)"
        echo "remaining_shards=${remaining_shards}"
        echo "running_serc_jobs=${running_serc_jobs}"
        echo "estimated_serc_task_seconds=${estimated_task_seconds}"
        echo "estimated_finish_seconds=${estimated_finish_seconds}"
    } > "${flag_path}"
    echo "Activated serc-only endgame: remaining_shards=${remaining_shards}; running_serc_jobs=${running_serc_jobs}; estimated_serc_task_seconds=${estimated_task_seconds}; estimated_finish_seconds=${estimated_finish_seconds}; flag=${flag_path}"
}

gpu_lock_file_for_task() {
    local gpu_task_id="$1"
    printf '%s/lock_%s_gpu_%s.lock\n' "${gpu_lock_dir}" "${run_name}" "${gpu_task_id}"
}

cleanup_lock_for_task() {
    local gpu_task_id="$1"
    rm -f "$(gpu_lock_file_for_task "${gpu_task_id}")"
}

submit_gpu_job() {
    local gpu_task_id="$1"
    local pool="$2"
    local partition=""
    local time_limit=""
    local time_limit_seconds=""
    local cpus_per_task=""
    local mem=""
    local constraint=""
    local lock_file=""
    local sbatch_constraint_args=()
    local sbatch_export=""
    local gpu_job_id=""

    if [[ "${pool}" == "serc" ]]; then
        partition="${serc_partition}"
        time_limit="${gpu_time_limit}"
        time_limit_seconds="$(hhmmss_to_seconds "${time_limit}")"
        cpus_per_task="${gpu_cpus_per_task}"
        mem="${gpu_mem}"
        constraint="${gpu_constraint}"
        lock_file="$(gpu_lock_file_for_task "${gpu_task_id}")"
        touch "${lock_file}"
    elif [[ "${pool}" == "owners" ]]; then
        partition="${owners_partition}"
        time_limit="${owners_gpu_time_limit}"
        time_limit_seconds="$(hhmmss_to_seconds "${time_limit}")"
        cpus_per_task="${owners_gpu_cpus_per_task}"
        mem="${owners_gpu_mem}"
        constraint="${owners_gpu_constraint}"
    else
        echo "Unsupported GPU pool ${pool}" >&2
        return 1
    fi

    sbatch_export="ALL,MANIFEST_PATH=${manifest_path},MODEL_TYPE=${model_type},GPU_TASK_ID=${gpu_task_id},GPU_POOL=${pool},DYNAMIC_GPU_WORK_QUEUE=${dynamic_gpu_work_queue},GPU_JOB_TIME_LIMIT_SECONDS=${time_limit_seconds},GPU_TIME_MARGIN_SECONDS=${gpu_time_margin_seconds},GPU_CLAIM_STALE_SECONDS=${gpu_claim_stale_seconds},GPU_IDLE_SLEEP_SECONDS=${gpu_idle_sleep_seconds},GPU_NEXT_TASK_SAFETY_FACTOR=${gpu_next_task_safety_factor}"
    if [[ "${pool}" == "serc" ]]; then
        sbatch_export="${sbatch_export},LOCK_FILE=${lock_file}"
    else
        sbatch_export="${sbatch_export},LOCK_FILE="
    fi

    if [[ -n "${constraint}" ]]; then
        sbatch_constraint_args+=(--constraint="${constraint}")
    fi

    if [[ -n "${constraint}" ]]; then
        if ! gpu_job_id="$(submit_with_retry \
            "run_maps_gpu_${pool}_${gpu_task_id}" \
            sbatch \
                --parsable \
                --partition="${partition}" \
                --time="${time_limit}" \
                --cpus-per-task="${cpus_per_task}" \
                --mem="${mem}" \
                --constraint="${constraint}" \
                --job-name="map_gpu_${pool}_${gpu_task_id}" \
                --export="${sbatch_export}" \
                "${script_dir}/run_maps_gpu_ensemble.sbatch")"; then
            if [[ "${pool}" == "serc" ]]; then
                rm -f "${lock_file}"
            fi
            return 1
        fi
    elif ! gpu_job_id="$(submit_with_retry \
        "run_maps_gpu_${pool}_${gpu_task_id}" \
        sbatch \
            --parsable \
            --partition="${partition}" \
            --time="${time_limit}" \
            --cpus-per-task="${cpus_per_task}" \
            --mem="${mem}" \
            --job-name="map_gpu_${pool}_${gpu_task_id}" \
            --export="${sbatch_export}" \
            "${script_dir}/run_maps_gpu_ensemble.sbatch")"; then
        if [[ "${pool}" == "serc" ]]; then
            rm -f "${lock_file}"
        fi
        return 1
    fi

    printf '%s\n' "${gpu_job_id}"
}

submit_prepare_job() {
    local prepare_task_id="$1"
    local prepare_job_id=""
    if ! prepare_job_id="$(submit_with_retry \
        "prepare_maps_${prepare_task_id}" \
        sbatch \
            --parsable \
            --job-name="prepare_maps_${prepare_task_id}" \
            --time="${prepare_time_limit}" \
            --cpus-per-task="${prepare_cpus_per_task}" \
            --mem="${prepare_mem}" \
            --export=ALL,MANIFEST_PATH="${manifest_path}",PREPARE_TASK_ID="${prepare_task_id}" \
            "${script_dir}/prepare_maps_ensemble.sbatch")"; then
        return 1
    fi
    printf '%s\n' "${prepare_job_id}"
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

declare -a prepare_task_fine_counts
mapfile -t prepare_task_count_lines < <(python3 - <<PYCOUNTS
import pandas as pd

df = pd.read_csv("${manifest_path}")
counts = (
    df.groupby("job_task_id")
    .size()
    .sort_index()
)
for job_task_id, count in counts.items():
    print(f"{int(job_task_id)}|{int(count)}")
PYCOUNTS
)
for line in "${prepare_task_count_lines[@]}"; do
    IFS='|' read -r prepare_task_id prepare_fine_count <<< "${line}"
    prepare_task_fine_counts[${prepare_task_id}]="${prepare_fine_count}"
done

num_gpu_job_tasks=0
if [[ "${dynamic_gpu_work_queue}" != "true" ]]; then
    num_gpu_job_tasks="$(python3 - <<PYCOUNT
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(int(df["gpu_job_task_id"].max()) + 1)
PYCOUNT
    )"
fi

num_merge_tasks="$(python3 - <<PYCOUNT
import pandas as pd
df = pd.read_csv("${manifest_path}")
print(int(df["merge_task_id"].max()) + 1)
PYCOUNT
)"

prepared_tensor_count="$(count_prepared_reference_payloads)"
if [[ "${start_from_gpu}" == "true" && "${resume_completed_run}" != "true" ]]; then
    if [[ "${prepared_tensor_count}" -ne "${num_fine_tasks}" ]]; then
        echo "GPU-only restart requested, but prepared tensor count ${prepared_tensor_count} does not match manifest tasks ${num_fine_tasks}" >&2
        exit 1
    fi
    echo "Validated GPU-only restart inputs: prepared_tensors=${prepared_tensor_count}, manifest_tasks=${num_fine_tasks}"
fi

echo "Submitting ${num_job_tasks} array jobs covering ${num_fine_tasks} fine tasks from manifest ${manifest_path}"
echo "tasks_per_job=${tasks_per_job}"
if [[ "${dynamic_gpu_work_queue}" == "true" ]]; then
    echo "use_gpu_forward=${use_gpu_forward}; dynamic_gpu_work_queue=true; gpu_time_limit=${gpu_time_limit}; owners_gpu_time_limit=${owners_gpu_time_limit}"
else
    echo "use_gpu_forward=${use_gpu_forward}; gpu_fine_tasks_per_job=${gpu_fine_tasks_per_job}; num_gpu_job_tasks=${num_gpu_job_tasks}"
fi
echo "merge_blocks_per_job=${merge_blocks_per_job}; num_merge_tasks=${num_merge_tasks}"

after_gpu_dependency=""
if [[ "${use_gpu_forward}" == "true" ]]; then
    if [[ "${dynamic_gpu_work_queue}" == "true" ]]; then
        if [[ "${resume_completed_run}" == "true" ]]; then
            echo "Completed-run resume mode: skipping CPU prepare and GPU scheduling because all shards are already complete in ${latest_run_dir}"
        else
        declare -a gpu_job_ids
        declare -a gpu_job_pool
        declare -a gpu_retry_counts
        declare -a gpu_submission_counts
        declare -a gpu_endgame_cancelled
        declare -a prepare_job_ids
        declare -a prepare_completed_flags
        declare -a prepare_retry_counts
        declare -a prepare_submission_counts
        next_prepare_task_id=0
        if [[ "${start_from_gpu}" == "true" ]]; then
            echo "GPU-only restart mode: skipping CPU prepare array and starting directly from prepared tensors in ${latest_run_dir}"
        elif [[ "${resume_partial_run}" == "true" ]]; then
            echo "Partial-run resume mode: rebuilding prepare submission state from ${latest_run_dir}"
            mapfile -t prepare_resume_lines < <(python3 - <<PYRESUME_PREP
import csv
from pathlib import Path

manifest_path = Path("${manifest_path}")
prepared_dir = Path("${latest_run_dir}") / "prepared_tensors"

groups = {}
with manifest_path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        job_task_id = int(row["job_task_id"])
        fine_task_id = int(row["task_id"])
        shard_exists = Path(row["shard_path"]).exists()
        prepared_exists = (prepared_dir / f"task_{fine_task_id:06d}_reference.pt").exists()
        groups.setdefault(job_task_id, []).append(shard_exists or prepared_exists)

for job_task_id in sorted(groups):
    materialized = int(all(groups[job_task_id]))
    print(f"{job_task_id}|{materialized}")
PYRESUME_PREP
            )
            for line in "${prepare_resume_lines[@]}"; do
                IFS='|' read -r prepare_task_id materialized_flag <<< "${line}"
                prepare_completed_flags[${prepare_task_id}]="${materialized_flag}"
            done
            while [[ "${next_prepare_task_id}" -lt "${num_job_tasks}" && "${prepare_completed_flags[$next_prepare_task_id]:-0}" == "1" ]]; do
                next_prepare_task_id=$(( next_prepare_task_id + 1 ))
            done
        else
            echo "Monitoring incremental CPU prepare jobs and submitting GPU jobs under shared lock budget ${gpu_max_jobs} from ${gpu_lock_dir}"
        fi

        echo "GPU scheduler config: serc_partition=${serc_partition}; serc_cap=${gpu_max_jobs}; owners_partition=${owners_partition}; owners_cap=${owners_gpu_max_jobs}; retry_states=${gpu_retry_states}; max_retries=${gpu_max_retries}; dynamic_gpu_work_queue=${dynamic_gpu_work_queue}"

        next_gpu_task_id=0
        retry_waiting_total=0
        serc_retry_events_total=0
        owners_failure_events_total=0
        scheduler_iteration=0
        serc_only_endgame_active=0
        while true; do
            scheduler_iteration=$(( scheduler_iteration + 1 ))
            refresh_runtime_scheduler_limits
            progress_made=0
            blocked_by_prepare=0
            blocked_by_capacity=0
            running_gpu_jobs=0
            pending_gpu_jobs=0
            running_serc_jobs=0
            pending_serc_jobs=0
            active_owner_jobs=0
            active_prepare_jobs=0
            running_prepare_jobs=0
            pending_prepare_jobs=0

            completed_shards="$(count_shards_on_disk)"
            prepared_reference_count="$(count_prepared_reference_payloads)"
            remaining_shards=$(( num_fine_tasks - completed_shards ))
            if [[ "${remaining_shards}" -lt 0 ]]; then
                remaining_shards=0
            fi
            if [[ -f "$(serc_endgame_flag_path)" ]]; then
                serc_only_endgame_active=1
            fi
            if [[ "${start_from_gpu}" != "true" ]]; then
                for (( prepare_task_id=0; prepare_task_id<next_prepare_task_id; prepare_task_id++ )); do
                    prepare_job_id="${prepare_job_ids[$prepare_task_id]:-}"
                    if [[ -z "${prepare_job_id}" ]]; then
                        continue
                    fi
                    prepare_job_state="$(job_state "${prepare_job_id}")"
                    if [[ "${prepare_job_state}" == "COMPLETED" ]]; then
                        prepare_job_ids[$prepare_task_id]=""
                        prepare_completed_flags[$prepare_task_id]=1
                        progress_made=1
                        echo "Prepare worker job ${prepare_job_id} for prepare_task_id=${prepare_task_id} completed"
                        continue
                    fi
                    if job_failed "${prepare_job_state}"; then
                        if prepare_retryable "${prepare_job_state}"; then
                            retry_count=$(( ${prepare_retry_counts[$prepare_task_id]:-0} + 1 ))
                            prepare_retry_counts[$prepare_task_id]="${retry_count}"
                            if [[ "${retry_count}" -gt "${prepare_max_retries}" ]]; then
                                echo "Prepare worker job ${prepare_job_id} for prepare_task_id=${prepare_task_id} exceeded retry cap ${prepare_max_retries} after state ${prepare_job_state}" >&2
                                exit 1
                            fi
                            prepare_job_ids[$prepare_task_id]=""
                            progress_made=1
                            echo "Prepare worker job ${prepare_job_id} for prepare_task_id=${prepare_task_id} ended with retryable state ${prepare_job_state}; retry ${retry_count}/${prepare_max_retries} queued"
                            continue
                        fi
                        echo "Prepare worker job ${prepare_job_id} for prepare_task_id=${prepare_task_id} failed with non-retryable state ${prepare_job_state}; aborting GPU submission monitor." >&2
                        exit 1
                    fi
                    active_prepare_jobs=$(( active_prepare_jobs + 1 ))
                    if [[ "${prepare_job_state}" == "PENDING" || "${prepare_job_state}" == "CONFIGURING" ]]; then
                        pending_prepare_jobs=$(( pending_prepare_jobs + 1 ))
                    else
                        running_prepare_jobs=$(( running_prepare_jobs + 1 ))
                    fi
                done
            fi
            prepare_complete=0
            if [[ "${start_from_gpu}" == "true" ]]; then
                prepare_complete=1
            elif [[ "${next_prepare_task_id}" -ge "${num_job_tasks}" && "${active_prepare_jobs}" -eq 0 ]]; then
                prepare_complete=1
            fi

            for (( gpu_task_id=0; gpu_task_id<next_gpu_task_id; gpu_task_id++ )); do
                gpu_job_id="${gpu_job_ids[$gpu_task_id]:-}"
                if [[ -z "${gpu_job_id}" ]]; then
                    continue
                fi

                gpu_job_state="$(job_state "${gpu_job_id}")"
                gpu_pool="${gpu_job_pool[$gpu_task_id]:-unknown}"
                if [[ "${gpu_endgame_cancelled[$gpu_task_id]:-0}" == "1" ]]; then
                    if [[ "${gpu_job_state}" == "CANCELLED" || "${gpu_job_state}" == "CANCELLED+" || "${gpu_job_state}" == "COMPLETED" || "${gpu_job_state}" == "UNKNOWN" ]]; then
                        gpu_job_ids[$gpu_task_id]=""
                        gpu_job_pool[$gpu_task_id]=""
                        progress_made=1
                        echo "GPU worker job ${gpu_job_id} for worker_id=${gpu_task_id} retired for serc-only endgame on pool=${gpu_pool}; final_state=${gpu_job_state}"
                        continue
                    fi
                fi
                if [[ "${gpu_job_state}" == "COMPLETED" ]]; then
                    if [[ "${gpu_pool}" == "serc" ]]; then
                        cleanup_lock_for_task "${gpu_task_id}"
                    fi
                    gpu_job_ids[$gpu_task_id]=""
                    gpu_job_pool[$gpu_task_id]=""
                    progress_made=1
                    echo "GPU worker job ${gpu_job_id} for worker_id=${gpu_task_id} completed on pool=${gpu_pool}; shards=${completed_shards}/${num_fine_tasks}"
                    continue
                fi

                if job_failed "${gpu_job_state}"; then
                    if [[ "${gpu_pool}" == "serc" ]]; then
                        cleanup_lock_for_task "${gpu_task_id}"
                    fi
                    if [[ "${gpu_pool}" == "owners" ]]; then
                        gpu_job_ids[$gpu_task_id]=""
                        gpu_job_pool[$gpu_task_id]=""
                        owners_failure_events_total=$(( owners_failure_events_total + 1 ))
                        if [[ "${gpu_job_state}" == "OUT_OF_MEMORY" ]]; then
                            if downshift_note="$(downshift_gpu_batch_cache_for_job "${gpu_job_id}")"; then
                                echo "GPU worker job ${gpu_job_id} hit OUT_OF_MEMORY; downshifted cached batch size ${downshift_note}"
                            else
                                echo "GPU worker job ${gpu_job_id} hit OUT_OF_MEMORY; no batch-cache metadata was available to downshift"
                            fi
                        fi
                        if job_retryable "${gpu_job_state}"; then
                            retry_waiting_total=$(( retry_waiting_total + 1 ))
                            echo "GPU worker job ${gpu_job_id} for worker_id=${gpu_task_id} on pool=owners ended with retryable state ${gpu_job_state}; tolerated_owners_failures=${owners_failure_events_total}; unfinished shards remain claimable"
                        else
                            echo "GPU worker job ${gpu_job_id} for worker_id=${gpu_task_id} on pool=owners ended with non-retryable state ${gpu_job_state}; tolerated_owners_failures=${owners_failure_events_total}; unfinished shards remain claimable"
                        fi
                        progress_made=1
                        continue
                    fi
                    if [[ "${gpu_job_state}" == "OUT_OF_MEMORY" ]]; then
                        if downshift_note="$(downshift_gpu_batch_cache_for_job "${gpu_job_id}")"; then
                            echo "GPU worker job ${gpu_job_id} hit OUT_OF_MEMORY; downshifted cached batch size ${downshift_note}"
                        else
                            echo "GPU worker job ${gpu_job_id} hit OUT_OF_MEMORY; no batch-cache metadata was available to downshift"
                        fi
                    fi
                    if job_retryable "${gpu_job_state}"; then
                        retry_count=$(( ${gpu_retry_counts[$gpu_task_id]:-0} + 1 ))
                        gpu_retry_counts[$gpu_task_id]="${retry_count}"
                        gpu_job_ids[$gpu_task_id]=""
                        gpu_job_pool[$gpu_task_id]=""
                        serc_retry_events_total=$(( serc_retry_events_total + 1 ))
                        if [[ "${serc_retry_events_total}" -gt 50 ]]; then
                            echo "SERC retry budget exceeded for this year: serc_retry_events_total=${serc_retry_events_total}/50 after worker_id=${gpu_task_id} state=${gpu_job_state}" >&2
                            exit 1
                        fi
                        retry_waiting_total=$(( retry_waiting_total + 1 ))
                        progress_made=1
                        echo "GPU worker job ${gpu_job_id} for worker_id=${gpu_task_id} on pool=serc ended with retryable state ${gpu_job_state}; unfinished shards remain claimable; retry_events=${retry_waiting_total}; serc_retry_events=${serc_retry_events_total}/50"
                        continue
                    fi
                    echo "GPU worker job ${gpu_job_id} for worker_id=${gpu_task_id} on pool=serc failed with non-retryable state ${gpu_job_state}; aborting downstream submission." >&2
                    exit 1
                fi

                if [[ "${gpu_pool}" == "owners" ]]; then
                    active_owner_jobs=$(( active_owner_jobs + 1 ))
                elif [[ "${gpu_pool}" == "serc" ]]; then
                    if [[ "${gpu_job_state}" == "PENDING" || "${gpu_job_state}" == "CONFIGURING" ]]; then
                        pending_serc_jobs=$(( pending_serc_jobs + 1 ))
                    else
                        running_serc_jobs=$(( running_serc_jobs + 1 ))
                    fi
                fi
                if [[ "${gpu_job_state}" == "PENDING" || "${gpu_job_state}" == "CONFIGURING" ]]; then
                    pending_gpu_jobs=$(( pending_gpu_jobs + 1 ))
                else
                    running_gpu_jobs=$(( running_gpu_jobs + 1 ))
                fi
            done

            active_gpu_jobs=$(( running_gpu_jobs + pending_gpu_jobs ))
            if [[ "${completed_shards}" -ge "${num_fine_tasks}" && "${prepare_complete}" -eq 1 && "${active_gpu_jobs}" -eq 0 ]]; then
                break
            fi

            serc_estimated_task_seconds=""
            serc_estimated_finish_seconds=""
            year_eta_seconds="NA"
            year_eta_dhms="NA"
            overall_eta_seconds="NA"
            overall_eta_dhms="NA"
            if [[ "${remaining_shards}" -le 0 ]]; then
                year_eta_seconds="0"
                year_eta_dhms="0:00:00:00"
                overall_eta_seconds="0"
                overall_eta_dhms="0:00:00:00"
            else
                if [[ "${serc_only_endgame}" == "true" && "${serc_only_endgame_active}" != "1" && "${gpu_max_jobs}" -gt 0 ]]; then
                    if serc_estimated_task_seconds="$(estimate_serc_task_seconds)"; then
                        serc_estimated_finish_seconds=$(( (remaining_shards * serc_estimated_task_seconds + gpu_max_jobs - 1) / gpu_max_jobs ))
                        if [[ "${serc_estimated_finish_seconds}" -le $(( serc_endgame_minutes_threshold * 60 )) ]]; then
                            serc_only_endgame_active=1
                            activate_serc_only_endgame "${remaining_shards}" "${gpu_max_jobs}" "${serc_estimated_task_seconds}" "${serc_estimated_finish_seconds}"
                            progress_made=1
                        fi
                    fi
                fi
                recent_shard_rate_per_second=""
                if recent_shard_rate_per_second="$(estimate_recent_shard_rate_per_second)"; then
                    year_eta_seconds="$(python3 - <<PYETA
import math
rate = float("${recent_shard_rate_per_second}")
remaining = int("${remaining_shards}")
if rate <= 0 or remaining <= 0:
    raise SystemExit(1)
print(int(math.ceil(remaining / rate)))
PYETA
                    )"
                    year_eta_dhms="$(format_dhms "${year_eta_seconds}")"
                fi

                current_year_elapsed_seconds=""
                current_year_full_estimate_seconds=""
                if [[ -n "${multiyear_year_started_at_epoch}" && "${multiyear_year_started_at_epoch}" != "NA" ]]; then
                    current_year_elapsed_seconds="$(python3 - <<PYELAPSED
import time
started = float("${multiyear_year_started_at_epoch}")
print(int(max(time.time() - started, 0)))
PYELAPSED
                    )"
                fi
                if [[ "${year_eta_seconds}" != "NA" && -n "${current_year_elapsed_seconds}" ]]; then
                    current_year_full_estimate_seconds=$(( current_year_elapsed_seconds + year_eta_seconds ))
                fi

                remaining_years_including_current=""
                if [[ -n "${multiyear_year_ordinal}" && -n "${multiyear_total_years}" ]]; then
                    remaining_years_including_current=$(( multiyear_total_years - multiyear_year_ordinal + 1 ))
                fi
                if [[ "${year_eta_seconds}" != "NA" && -n "${current_year_full_estimate_seconds}" && -n "${current_year_elapsed_seconds}" && -n "${remaining_years_including_current}" && "${remaining_years_including_current}" -gt 0 ]]; then
                    overall_eta_seconds=$(( remaining_years_including_current * current_year_full_estimate_seconds - current_year_elapsed_seconds ))
                    overall_eta_dhms="$(format_dhms "${overall_eta_seconds}")"
                fi
            fi

            if [[ "${serc_only_endgame_active}" == "1" ]]; then
                for (( gpu_task_id=0; gpu_task_id<next_gpu_task_id; gpu_task_id++ )); do
                    if [[ "${gpu_endgame_cancelled[$gpu_task_id]:-0}" == "1" ]]; then
                        continue
                    fi
                    gpu_job_id="${gpu_job_ids[$gpu_task_id]:-}"
                    gpu_pool="${gpu_job_pool[$gpu_task_id]:-unknown}"
                    if [[ -z "${gpu_job_id}" || "${gpu_pool}" != "owners" ]]; then
                        continue
                    fi
                    gpu_job_state="$(job_state "${gpu_job_id}")"
                    if [[ "${gpu_job_state}" == "PENDING" || "${gpu_job_state}" == "CONFIGURING" ]]; then
                        scancel "${gpu_job_id}" >/dev/null 2>&1 || true
                        gpu_endgame_cancelled[$gpu_task_id]=1
                        progress_made=1
                        echo "Canceled owners GPU worker job ${gpu_job_id} for worker_id=${gpu_task_id} during serc-only endgame; state=${gpu_job_state}"
                    fi
                done
            fi

            if [[ "${completed_shards}" -ge "${num_fine_tasks}" && "${prepare_complete}" -eq 1 && "${active_gpu_jobs}" -eq 0 ]]; then
                break
            fi

            effective_prepare_backlog="${prepared_reference_count}"
            if [[ "${start_from_gpu}" != "true" ]]; then
                while [[ "${next_prepare_task_id}" -lt "${num_job_tasks}" && "${prepare_completed_flags[$next_prepare_task_id]:-0}" == "1" ]]; do
                    next_prepare_task_id=$(( next_prepare_task_id + 1 ))
                done
                for (( prepare_task_id=0; prepare_task_id<next_prepare_task_id; prepare_task_id++ )); do
                    prepare_job_id="${prepare_job_ids[$prepare_task_id]:-}"
                    if [[ -z "${prepare_job_id}" ]]; then
                        continue
                    fi
                    prepare_fine_count="${prepare_task_fine_counts[$prepare_task_id]:-${tasks_per_job}}"
                    effective_prepare_backlog=$(( effective_prepare_backlog + prepare_fine_count ))
                done

                for (( prepare_task_id=0; prepare_task_id<next_prepare_task_id; prepare_task_id++ )); do
                    if [[ "${prepare_completed_flags[$prepare_task_id]:-0}" == "1" ]]; then
                        continue
                    fi
                    if [[ -n "${prepare_job_ids[$prepare_task_id]:-}" ]]; then
                        continue
                    fi
                    prepare_fine_count="${prepare_task_fine_counts[$prepare_task_id]:-${tasks_per_job}}"
                    if ! prepare_job_id="$(submit_prepare_job "${prepare_task_id}")"; then
                        echo "Failed to resubmit prepare worker for prepare_task_id=${prepare_task_id}" >&2
                        exit 1
                    fi
                    prepare_job_ids[$prepare_task_id]="${prepare_job_id}"
                    prepare_submission_counts[$prepare_task_id]=$(( ${prepare_submission_counts[$prepare_task_id]:-0} + 1 ))
                    active_prepare_jobs=$(( active_prepare_jobs + 1 ))
                    pending_prepare_jobs=$(( pending_prepare_jobs + 1 ))
                    effective_prepare_backlog=$(( effective_prepare_backlog + prepare_fine_count ))
                    progress_made=1
                    echo "Resubmitted prepare worker job ${prepare_job_id} for prepare_task_id=${prepare_task_id}; prepare_fine_tasks=${prepare_fine_count}; attempts=${prepare_submission_counts[$prepare_task_id]}"
                done

                while [[ "${next_prepare_task_id}" -lt "${num_job_tasks}" && "${effective_prepare_backlog}" -lt "${max_prepared_ahead_of_completed_shards}" ]]; do
                    prepare_task_id="${next_prepare_task_id}"
                    prepare_fine_count="${prepare_task_fine_counts[$prepare_task_id]:-${tasks_per_job}}"
                    if ! prepare_job_id="$(submit_prepare_job "${prepare_task_id}")"; then
                        echo "Failed to submit prepare worker for prepare_task_id=${prepare_task_id}" >&2
                        exit 1
                    fi
                    prepare_job_ids[$prepare_task_id]="${prepare_job_id}"
                    prepare_completed_flags[$prepare_task_id]=0
                    prepare_submission_counts[$prepare_task_id]=$(( ${prepare_submission_counts[$prepare_task_id]:-0} + 1 ))
                    next_prepare_task_id=$(( next_prepare_task_id + 1 ))
                    active_prepare_jobs=$(( active_prepare_jobs + 1 ))
                    pending_prepare_jobs=$(( pending_prepare_jobs + 1 ))
                    effective_prepare_backlog=$(( effective_prepare_backlog + prepare_fine_count ))
                    progress_made=1
                    echo "Submitted prepare worker job ${prepare_job_id} for prepare_task_id=${prepare_task_id}; prepare_fine_tasks=${prepare_fine_count}; effective_prepare_backlog=${effective_prepare_backlog}/${max_prepared_ahead_of_completed_shards}; submitted_prepare_jobs=${next_prepare_task_id}/${num_job_tasks}"
                done
            fi

            ready_unsharded="${prepared_reference_count}"

            while [[ "${ready_unsharded}" -gt "${active_gpu_jobs}" ]]; do
                target_pool=""
                active_locks="$(lock_count)"
                if [[ "${active_locks}" -lt "${gpu_max_jobs}" ]]; then
                    target_pool="serc"
                elif [[ "${serc_only_endgame_active}" != "1" && "${owners_gpu_max_jobs}" -gt 0 && "${active_owner_jobs}" -lt "${owners_gpu_max_jobs}" ]]; then
                    target_pool="owners"
                fi

                if [[ -z "${target_pool}" ]]; then
                    blocked_by_capacity=$(( blocked_by_capacity + 1 ))
                    break
                fi

                gpu_task_id="${next_gpu_task_id}"
                next_gpu_task_id=$(( next_gpu_task_id + 1 ))
                if ! gpu_job_id="$(submit_gpu_job "${gpu_task_id}" "${target_pool}")"; then
                    echo "Failed to submit GPU worker for worker_id=${gpu_task_id} on pool=${target_pool}" >&2
                    exit 1
                fi

                gpu_job_ids[$gpu_task_id]="${gpu_job_id}"
                gpu_job_pool[$gpu_task_id]="${target_pool}"
                gpu_submission_counts[$gpu_task_id]=$(( ${gpu_submission_counts[$gpu_task_id]:-0} + 1 ))
                progress_made=1
                active_gpu_jobs=$(( active_gpu_jobs + 1 ))
                if [[ "${target_pool}" == "owners" ]]; then
                    active_owner_jobs=$(( active_owner_jobs + 1 ))
                fi
                echo "Submitted GPU worker job ${gpu_job_id} for worker_id=${gpu_task_id} on pool=${target_pool}; ready_unsharded=${ready_unsharded}; shards=${completed_shards}/${num_fine_tasks}; prepared_refs=${prepared_reference_count}; serc_locks=$(lock_count)/${gpu_max_jobs}; owners_active=${active_owner_jobs}/${owners_gpu_max_jobs}; attempts=${gpu_submission_counts[$gpu_task_id]}"
            done

            if [[ "${prepare_complete}" -ne 1 && "${ready_unsharded}" -le 0 ]]; then
                blocked_by_prepare=$(( blocked_by_prepare + 1 ))
            fi

            echo "GPU scheduler: iteration=${scheduler_iteration}; current_year=${multiyear_current_year}; year_ordinal=${multiyear_year_ordinal:-NA}/${multiyear_total_years:-NA}; year_eta=${year_eta_dhms}; overall_eta=${overall_eta_dhms}; shards=${completed_shards}/${num_fine_tasks}; prepared_refs=${prepared_reference_count}/${num_fine_tasks}; effective_prepare_backlog=${effective_prepare_backlog}/${max_prepared_ahead_of_completed_shards}; prepare_running=${running_prepare_jobs}; prepare_pending=${pending_prepare_jobs}; prepare_submitted=${next_prepare_task_id}/${num_job_tasks}; running=${running_gpu_jobs}; pending=${pending_gpu_jobs}; total_retries=${retry_waiting_total}; serc_retry_events=${serc_retry_events_total}/50; owners_failure_events=${owners_failure_events_total}; blocked_by_prepare=${blocked_by_prepare}; blocked_by_capacity=${blocked_by_capacity}; serc_locks=$(lock_count)/${gpu_max_jobs}; serc_running=${running_serc_jobs}; owners_active=${active_owner_jobs}/${owners_gpu_max_jobs}; serc_only_endgame=${serc_only_endgame_active}; serc_estimated_task_seconds=${serc_estimated_task_seconds:-NA}; serc_estimated_finish_seconds=${serc_estimated_finish_seconds:-NA}; sleeping ${gpu_submit_sleep_seconds}s"
            sleep "${gpu_submit_sleep_seconds}"
        done
        echo "All GPU work completed."
        fi
    else
        declare -a gpu_prepare_task_ids
        declare -a gpu_job_ids
        declare -a gpu_job_pool
        declare -a gpu_completed_flags
        declare -a gpu_retry_counts
        declare -a gpu_submission_counts

        if [[ "${resume_completed_run}" == "true" ]]; then
            echo "Completed-run resume mode: skipping CPU prepare and GPU scheduling because all shards are already complete in ${latest_run_dir}"
        elif [[ "${start_from_gpu}" == "true" ]]; then
            echo "GPU-only restart mode: skipping CPU prepare array and starting directly from prepared tensors in ${latest_run_dir}"
            for (( gpu_task_id=0; gpu_task_id<num_gpu_job_tasks; gpu_task_id++ )); do
                gpu_prepare_task_ids[${gpu_task_id}]="PREBUILT"
                gpu_completed_flags[${gpu_task_id}]=0
                gpu_retry_counts[${gpu_task_id}]=0
                gpu_submission_counts[${gpu_task_id}]=0
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
                gpu_completed_flags[${gpu_task_id}]=0
                gpu_retry_counts[${gpu_task_id}]=0
                gpu_submission_counts[${gpu_task_id}]=0
            done
        fi

        echo "GPU scheduler config: serc_partition=${serc_partition}; serc_cap=${gpu_max_jobs}; owners_partition=${owners_partition}; owners_cap=${owners_gpu_max_jobs}; retry_states=${gpu_retry_states}; max_retries=${gpu_max_retries}; dynamic_gpu_work_queue=${dynamic_gpu_work_queue}"

        completed_gpu_jobs=0
        scheduler_iteration=0
        while [[ "${completed_gpu_jobs}" -lt "${num_gpu_job_tasks}" ]]; do
            scheduler_iteration=$(( scheduler_iteration + 1 ))
            progress_made=0
            blocked_by_prepare=0
            blocked_by_capacity=0
            running_gpu_jobs=0
            pending_gpu_jobs=0
            active_owner_jobs=0
            retry_waiting=0

            for (( gpu_task_id=0; gpu_task_id<num_gpu_job_tasks; gpu_task_id++ )); do
                if [[ "${gpu_completed_flags[$gpu_task_id]:-0}" == "1" ]]; then
                    continue
                fi
                gpu_job_id="${gpu_job_ids[$gpu_task_id]:-}"
                if [[ -z "${gpu_job_id}" ]]; then
                    continue
                fi

                gpu_job_state="$(job_state "${gpu_job_id}")"
                gpu_pool="${gpu_job_pool[$gpu_task_id]:-unknown}"
                if [[ "${gpu_job_state}" == "COMPLETED" ]]; then
                    if [[ "${gpu_pool}" == "serc" ]]; then
                        cleanup_lock_for_task "${gpu_task_id}"
                    fi
                    gpu_job_ids[$gpu_task_id]=""
                    gpu_job_pool[$gpu_task_id]=""
                    gpu_completed_flags[$gpu_task_id]=1
                    completed_gpu_jobs=$(( completed_gpu_jobs + 1 ))
                    progress_made=1
                    echo "GPU job ${gpu_job_id} for gpu_job_task_id=${gpu_task_id} completed on pool=${gpu_pool}; completed=${completed_gpu_jobs}/${num_gpu_job_tasks}"
                    continue
                fi

                if job_failed "${gpu_job_state}"; then
                    if [[ "${gpu_pool}" == "serc" ]]; then
                        cleanup_lock_for_task "${gpu_task_id}"
                    fi
                    if [[ "${gpu_job_state}" == "OUT_OF_MEMORY" ]]; then
                        if downshift_note="$(downshift_gpu_batch_cache_for_job "${gpu_job_id}")"; then
                            echo "GPU job ${gpu_job_id} hit OUT_OF_MEMORY; downshifted cached batch size ${downshift_note}"
                        else
                            echo "GPU job ${gpu_job_id} hit OUT_OF_MEMORY; no batch-cache metadata was available to downshift"
                        fi
                    fi
                    if job_retryable "${gpu_job_state}"; then
                        retry_count=$(( ${gpu_retry_counts[$gpu_task_id]:-0} + 1 ))
                        gpu_retry_counts[$gpu_task_id]="${retry_count}"
                        if [[ "${retry_count}" -gt "${gpu_max_retries}" ]]; then
                            echo "GPU job ${gpu_job_id} for gpu_job_task_id=${gpu_task_id} exceeded retry cap ${gpu_max_retries} after state ${gpu_job_state}" >&2
                            exit 1
                        fi
                        gpu_job_ids[$gpu_task_id]=""
                        gpu_job_pool[$gpu_task_id]=""
                        retry_waiting=$(( retry_waiting + 1 ))
                        progress_made=1
                        echo "GPU job ${gpu_job_id} for gpu_job_task_id=${gpu_task_id} ended with retryable state ${gpu_job_state}; retry ${retry_count}/${gpu_max_retries} queued"
                        continue
                    fi
                    echo "GPU job ${gpu_job_id} for gpu_job_task_id=${gpu_task_id} failed with non-retryable state ${gpu_job_state}; aborting downstream submission." >&2
                    exit 1
                fi

                if [[ "${gpu_pool}" == "owners" ]]; then
                    active_owner_jobs=$(( active_owner_jobs + 1 ))
                fi
                if [[ "${gpu_job_state}" == "PENDING" || "${gpu_job_state}" == "CONFIGURING" ]]; then
                    pending_gpu_jobs=$(( pending_gpu_jobs + 1 ))
                else
                    running_gpu_jobs=$(( running_gpu_jobs + 1 ))
                fi
            done

            for (( gpu_task_id=0; gpu_task_id<num_gpu_job_tasks; gpu_task_id++ )); do
                if [[ "${gpu_completed_flags[$gpu_task_id]:-0}" == "1" ]]; then
                    continue
                fi
                if [[ -n "${gpu_job_ids[$gpu_task_id]:-}" ]]; then
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

                target_pool=""
                active_locks="$(lock_count)"
                if [[ "${active_locks}" -lt "${gpu_max_jobs}" ]]; then
                    target_pool="serc"
                elif [[ "${owners_gpu_max_jobs}" -gt 0 && "${active_owner_jobs}" -lt "${owners_gpu_max_jobs}" ]]; then
                    target_pool="owners"
                fi

                if [[ -z "${target_pool}" ]]; then
                    blocked_by_capacity=$(( blocked_by_capacity + 1 ))
                    continue
                fi

                if ! gpu_job_id="$(submit_gpu_job "${gpu_task_id}" "${target_pool}")"; then
                    echo "Failed to submit GPU job for gpu_job_task_id=${gpu_task_id} on pool=${target_pool}" >&2
                    exit 1
                fi

                gpu_job_ids[$gpu_task_id]="${gpu_job_id}"
                gpu_job_pool[$gpu_task_id]="${target_pool}"
                gpu_submission_counts[$gpu_task_id]=$(( ${gpu_submission_counts[$gpu_task_id]:-0} + 1 ))
                progress_made=1
                if [[ "${target_pool}" == "owners" ]]; then
                    active_owner_jobs=$(( active_owner_jobs + 1 ))
                fi
                echo "Submitted GPU job ${gpu_job_id} for gpu_job_task_id=${gpu_task_id} on pool=${target_pool}; prepare_task_ids=${prepare_task_csv}; serc_locks=$(lock_count)/${gpu_max_jobs}; owners_active=${active_owner_jobs}/${owners_gpu_max_jobs}; attempts=${gpu_submission_counts[$gpu_task_id]}"
            done

            if [[ "${completed_gpu_jobs}" -eq "${num_gpu_job_tasks}" ]]; then
                break
            fi

            echo "GPU scheduler: iteration=${scheduler_iteration}; completed=${completed_gpu_jobs}/${num_gpu_job_tasks}; running=${running_gpu_jobs}; pending=${pending_gpu_jobs}; retry_waiting=${retry_waiting}; blocked_by_prepare=${blocked_by_prepare}; blocked_by_capacity=${blocked_by_capacity}; serc_locks=$(lock_count)/${gpu_max_jobs}; owners_active=${active_owner_jobs}/${owners_gpu_max_jobs}; sleeping ${gpu_submit_sleep_seconds}s"
            sleep "${gpu_submit_sleep_seconds}"
        done
        echo "All GPU jobs completed."
    fi

else
    array_job_id="$(submit_with_retry \
        "create_maps_ensemble" \
        sbatch --parsable --array="0-$(( num_job_tasks - 1 ))%${array_concurrency}" --export=ALL,MANIFEST_PATH="${manifest_path}",MODEL_TYPE="${model_type}" "${script_dir}/create_maps_ensemble.sbatch")"
    echo "Submitted worker array job ${array_job_id}"
    after_gpu_dependency="${array_job_id}"
fi

merge_overwrite_flag="1"
if [[ -n "${persistent_out_zarr_path:-}" && "${persistent_out_zarr_path}" != "None" ]]; then
    merge_overwrite_flag="0"
fi

merge_init_args=(
    --parsable
)
if [[ -n "${after_gpu_dependency}" ]]; then
    merge_init_args+=(--dependency=afterok:${after_gpu_dependency})
fi
merge_init_args+=(
    --export=ALL,MANIFEST_PATH="${manifest_path}",OVERWRITE_MERGE="${merge_overwrite_flag}",MERGE_INITIALIZE_ONLY=1
    "${script_dir}/merge_maps_ensemble.sbatch"
)
merge_init_job_id="$(submit_with_retry "merge_init" sbatch "${merge_init_args[@]}")"
echo "Submitted merge initialization job ${merge_init_job_id}; overwrite_merge=${merge_overwrite_flag}"

merge_array_job_id="$(submit_with_retry \
    "merge_all" \
    sbatch --parsable --dependency=afterok:${merge_init_job_id} --export=ALL,MANIFEST_PATH="${manifest_path}",MERGE_ALL_TASKS=1 "${script_dir}/merge_maps_ensemble.sbatch")"
echo "Submitted single-writer merge job ${merge_array_job_id}"

validate_job_id="$(submit_with_retry \
    "validate_maps" \
    sbatch --parsable --dependency=afterok:${merge_array_job_id} --export=ALL,MANIFEST_PATH="${manifest_path}" "${script_dir}/validate_maps_ensemble.sbatch")"
echo "Submitted validation job ${validate_job_id}"

if [[ "${wait_for_validation_completion}" == "true" ]]; then
    echo "Waiting for validation job ${validate_job_id} to complete"
    while true; do
        validate_job_state="$(job_state "${validate_job_id}")"
        if job_failed "${validate_job_state}"; then
            echo "Validation job ${validate_job_id} failed with state ${validate_job_state}" >&2
            exit 1
        fi
        if [[ "${validate_job_state}" == "COMPLETED" ]]; then
            break
        fi
        echo "Validation monitor: job=${validate_job_id} state=${validate_job_state}; sleeping ${gpu_submit_sleep_seconds}s"
        sleep "${gpu_submit_sleep_seconds}"
    done
    echo "Validation job ${validate_job_id} completed"
fi
