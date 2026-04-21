#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
repo_root="/home/users/trobinet/long_lfmc"
config_path="${CONFIG_PATH:-${script_dir}/map_configs_low_latency_update.yaml}"
registry_path="${REGISTRY_PATH:-${script_dir}/source_registry.yaml}"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"
current_model_script="${script_dir}/current_model_family_utils.py"

modis_update_script="${repo_root}/data_processing/modis/update_modis_range.py"
climate_update_script="${repo_root}/data_processing/climate_low_latency/update_low_latency_climate_range.py"
combined_weather_update_script="${repo_root}/data_processing/climate_low_latency/append_low_latency_range_from_saved_climatology.py"
modis_worker_script="${repo_root}/data_processing/modis/run_modis_range_worker.sbatch"
climate_worker_script="${repo_root}/data_processing/climate_low_latency/run_low_latency_climate_update_worker.sbatch"
promotion_script="${script_dir}/submit_low_latency_range_promotion.sh"
skip_oak_sync="${SKIP_OAK_SYNC:-0}"
today_override="${TODAY_OVERRIDE:-}"
safe_end_date_override="${SAFE_END_DATE_OVERRIDE:-}"

source "${process_env}"

readarray -t model_values < <(python3 "${current_model_script}" --variant multitask --format lines)
active_model_label="${model_values[0]}"
ensemble_outputs_root="${model_values[1]}"
input_data_name="${model_values[2]}"

readarray -t registry_values < <(REGISTRY_PATH="${registry_path}" CONFIG_PATH="${config_path}" python3 - <<'PY1'
import os
import yaml

with open(os.environ['REGISTRY_PATH'], 'r') as f:
    registry = yaml.safe_load(f)
with open(os.environ['CONFIG_PATH'], 'r') as f:
    cfg = yaml.safe_load(f)
values = [
    registry['storage']['scratch_root'],
    registry['storage']['oak_root'],
    registry['sources']['modis']['path'],
    registry['processing']['modis']['regrid_root'],
    registry['sources']['climate_low_latency']['path'],
    registry['sources']['daymet']['combined_path'],
    registry['sources']['daymet']['climatology_path'],
    registry['sources']['nlcd']['annual_path'],
    registry['sources']['static']['path'],
    registry['sources']['soils']['path'],
    registry['sources']['canopy_height']['path'],
    registry['sources']['scientific_current']['zarr_path'],
    registry['sources']['production']['zarr_path'],
    registry['sources']['production']['metadata_dir'],
    cfg['data']['grid_path'],
    str(cfg['data']['requested_start_date']),
    str(registry['processing']['prism']['release_latency_days']),
    str(registry['processing']['modis']['low_latency_tail_context_days']),
]
for value in values:
    print(value)
PY1
)

scratch_root="${registry_values[0]}"
oak_root="${registry_values[1]}"
modis_path="${registry_values[2]}"
modis_regrid_root="${registry_values[3]}"
low_latency_climate_path="${registry_values[4]}"
daymet_combined_path="${registry_values[5]}"
daymet_climatology_path="${registry_values[6]}"
nlcd_annual_path="${registry_values[7]}"
static_path="${registry_values[8]}"
soils_path="${registry_values[9]}"
canopy_height_path="${registry_values[10]}"
scientific_current_zarr="${registry_values[11]}"
production_zarr="${registry_values[12]}"
metadata_dir="${registry_values[13]}"
grid_path="${registry_values[14]}"
default_start_date="${registry_values[15]}"
prism_latency_days="${registry_values[16]}"
modis_tail_context_days="${registry_values[17]}"

batch_stamp="$(date +%Y%m%d_%H%M%S)"
status_dir="${metadata_dir}/status_reports"
lock_dir="${metadata_dir}/locks/low_latency_daily.lock"
mkdir -p "${status_dir}" "${metadata_dir}/locks"
status_path="${status_dir}/low_latency_daily_update_${batch_stamp}.json"

bootstrap_modis="not_started"
bootstrap_modis_regrid="not_started"
bootstrap_low_latency_climate="not_started"
bootstrap_daymet_combined="not_started"
bootstrap_daymet_climatology="not_started"
bootstrap_nlcd="not_started"
bootstrap_static="not_started"
bootstrap_soils="not_started"
bootstrap_canopy_height="not_started"
bootstrap_grid="not_started"
bootstrap_models="not_started"
bootstrap_production="not_started"
modis_check_status="not_started"
climate_check_status="not_started"
modis_update_status="not_started"
climate_update_status="not_started"
combined_weather_update_status="not_started"
inference_status="not_started"
sync_back_status="not_started"
final_status="started"
message="started"
requested_start_date="${REQUESTED_START_DATE:-}"
requested_end_date="${REQUESTED_END_DATE:-}"

cleanup() {
    rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

scratch_to_oak() {
    local scratch_path="$1"
    SCRATCH_ROOT="${scratch_root}" OAK_ROOT="${oak_root}" SCRATCH_PATH="${scratch_path}" python3 - <<'PY2'
import os
scratch_root = os.environ['SCRATCH_ROOT']
oak_root = os.environ['OAK_ROOT']
path = os.environ['SCRATCH_PATH']
if not path.startswith(scratch_root + '/') and path != scratch_root:
    print('')
    raise SystemExit(0)
rel = os.path.relpath(path, scratch_root)
print(os.path.join(oak_root, rel))
PY2
}

rsync_path() {
    local src_path="$1"
    local dst_path="$2"
    if [[ -d "${src_path}" ]]; then
        mkdir -p "${dst_path}"
        rsync -a "${src_path}/" "${dst_path}/"
    else
        mkdir -p "$(dirname "${dst_path}")"
        rsync -a "${src_path}" "${dst_path}"
    fi
}

ensure_on_scratch() {
    local scratch_path="$1"
    local label="$2"
    local allow_missing="$3"
    local oak_path
    if [[ -e "${scratch_path}" ]]; then
        echo "Using scratch ${label}: ${scratch_path}"
        return 0
    fi
    oak_path="$(scratch_to_oak "${scratch_path}")"
    if [[ -n "${oak_path}" && -e "${oak_path}" ]]; then
        echo "Copying ${label} from OAK to scratch: ${oak_path} -> ${scratch_path}"
        rsync_path "${oak_path}" "${scratch_path}"
        return 0
    fi
    if [[ "${allow_missing}" == "1" ]]; then
        echo "${label} is missing on scratch and OAK; continuing because allow_missing=1"
        return 0
    fi
    echo "Required ${label} is missing on scratch and OAK: ${scratch_path}"
    return 1
}

sync_to_oak() {
    local scratch_path="$1"
    local label="$2"
    local oak_path
    if [[ ! -e "${scratch_path}" ]]; then
        echo "Skipping OAK sync for missing ${label}: ${scratch_path}"
        return 0
    fi
    oak_path="$(scratch_to_oak "${scratch_path}")"
    if [[ -z "${oak_path}" ]]; then
        echo "Skipping OAK sync for ${label}; could not map scratch path: ${scratch_path}"
        return 0
    fi
    echo "Syncing ${label} back to OAK: ${scratch_path} -> ${oak_path}"
    rsync_path "${scratch_path}" "${oak_path}"
}

submit_sbatch_job() {
    local script_path="$1"
    local label="$2"
    shift 2
    local export_arg="ALL"
    local kv
    for kv in "$@"; do
        export_arg="${export_arg},${kv}"
    done
    local job_id
    job_id="$(sbatch --parsable --export="${export_arg}" "${script_path}")"
    echo "Submitted ${label} job ${job_id}" >&2
    printf '%s\n' "${job_id}"
}

job_is_active() {
    local job_id="$1"
    local states
    states="$(squeue -h -j "${job_id}" -o '%T')"
    [[ -n "${states}" ]]
}

final_job_state() {
    local job_id="$1"
    local states
    states="$(sacct -j "${job_id}" --format=State -n -P)"
    if [[ -z "${states}" ]]; then
        printf 'UNKNOWN\n'
        return 0
    fi
    while IFS= read -r line; do
        local state="${line%%|*}"
        [[ -z "${state}" ]] && continue
        if [[ "${state}" != "COMPLETED" ]]; then
            printf '%s\n' "${state}"
            return 0
        fi
    done <<< "${states}"
    printf 'COMPLETED\n'
}

wait_for_job() {
    local job_id="$1"
    local label="$2"
    local poll_seconds=30
    echo "Waiting for ${label} job ${job_id}"
    while job_is_active "${job_id}"; do
        echo "  ${label} job=${job_id} state=ACTIVE; sleeping ${poll_seconds}s"
        sleep "${poll_seconds}"
    done
    local state
    state="$(final_job_state "${job_id}")"
    echo "  ${label} job=${job_id} final_state=${state}"
    if [[ "${state}" != "COMPLETED" ]]; then
        echo "${label} job ${job_id} failed with state ${state}"
        return 1
    fi
}

write_status() {
    STATUS_PATH="${status_path}" python3 - <<'PY3'
import json
import os
from pathlib import Path

record = {
    'requested_start_date': os.environ['REQUESTED_START_DATE_VALUE'],
    'requested_end_date': os.environ['REQUESTED_END_DATE_VALUE'],
    'final_status': os.environ['FINAL_STATUS'],
    'message': os.environ['MESSAGE'],
    'bootstrap': {
        'modis_zarr': os.environ['BOOTSTRAP_MODIS'],
        'modis_regrid_root': os.environ['BOOTSTRAP_MODIS_REGRID'],
        'low_latency_climate': os.environ['BOOTSTRAP_LOW_LATENCY_CLIMATE'],
        'daymet_combined': os.environ['BOOTSTRAP_DAYMET_COMBINED'],
        'daymet_climatology': os.environ['BOOTSTRAP_DAYMET_CLIMATOLOGY'],
        'nlcd_annual': os.environ['BOOTSTRAP_NLCD'],
        'static': os.environ['BOOTSTRAP_STATIC'],
        'soils': os.environ['BOOTSTRAP_SOILS'],
        'canopy_height': os.environ['BOOTSTRAP_CANOPY_HEIGHT'],
        'grid': os.environ['BOOTSTRAP_GRID'],
        'ensemble_outputs': os.environ['BOOTSTRAP_MODELS'],
        'production_zarr': os.environ['BOOTSTRAP_PRODUCTION'],
    },
    'checks': {
        'modis': os.environ['MODIS_CHECK_STATUS'],
        'low_latency_climate': os.environ['CLIMATE_CHECK_STATUS'],
    },
    'updates': {
        'modis': os.environ['MODIS_UPDATE_STATUS'],
        'low_latency_climate': os.environ['CLIMATE_UPDATE_STATUS'],
        'combined_weather': os.environ['COMBINED_WEATHER_UPDATE_STATUS'],
    },
    'inference_status': os.environ['INFERENCE_STATUS'],
    'sync_back_status': os.environ['SYNC_BACK_STATUS'],
}
path = Path(os.environ['STATUS_PATH'])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, indent=2, sort_keys=True))
print(path)
PY3
}

update_status_env() {
    export REQUESTED_START_DATE_VALUE="${requested_start_date}"
    export REQUESTED_END_DATE_VALUE="${requested_end_date}"
    export BOOTSTRAP_MODIS="${bootstrap_modis}"
    export BOOTSTRAP_MODIS_REGRID="${bootstrap_modis_regrid}"
    export BOOTSTRAP_LOW_LATENCY_CLIMATE="${bootstrap_low_latency_climate}"
    export BOOTSTRAP_DAYMET_COMBINED="${bootstrap_daymet_combined}"
    export BOOTSTRAP_DAYMET_CLIMATOLOGY="${bootstrap_daymet_climatology}"
    export BOOTSTRAP_NLCD="${bootstrap_nlcd}"
    export BOOTSTRAP_STATIC="${bootstrap_static}"
    export BOOTSTRAP_SOILS="${bootstrap_soils}"
    export BOOTSTRAP_CANOPY_HEIGHT="${bootstrap_canopy_height}"
    export BOOTSTRAP_GRID="${bootstrap_grid}"
    export BOOTSTRAP_MODELS="${bootstrap_models}"
    export BOOTSTRAP_PRODUCTION="${bootstrap_production}"
    export MODIS_CHECK_STATUS="${modis_check_status}"
    export CLIMATE_CHECK_STATUS="${climate_check_status}"
    export MODIS_UPDATE_STATUS="${modis_update_status}"
    export CLIMATE_UPDATE_STATUS="${climate_update_status}"
    export COMBINED_WEATHER_UPDATE_STATUS="${combined_weather_update_status}"
    export INFERENCE_STATUS="${inference_status}"
    export SYNC_BACK_STATUS="${sync_back_status}"
    export FINAL_STATUS="${final_status}"
    export MESSAGE="${message}"
}

resolve_range() {
    TARGET_END_DATE="${requested_end_date}" TARGET_START_DATE="${requested_start_date}" DEFAULT_START_DATE="${default_start_date}" PRODUCTION_ZARR="${production_zarr}" PRISM_LATENCY_DAYS="${prism_latency_days}" TODAY_OVERRIDE_VALUE="${today_override}" SAFE_END_DATE_OVERRIDE_VALUE="${safe_end_date_override}" python3 - <<'PY4'
import os
import numpy as np
import pandas as pd
import zarr

requested_start = os.environ['TARGET_START_DATE']
requested_end = os.environ['TARGET_END_DATE']
default_start = pd.Timestamp(os.environ['DEFAULT_START_DATE']).normalize()
safe_end_override = os.environ.get('SAFE_END_DATE_OVERRIDE_VALUE', '')
today_override = os.environ.get('TODAY_OVERRIDE_VALUE', '')
if safe_end_override:
    safe_end = pd.Timestamp(safe_end_override).normalize()
else:
    today_value = pd.Timestamp(today_override).normalize() if today_override else pd.Timestamp.today().normalize()
    safe_end = today_value - pd.Timedelta(days=int(os.environ['PRISM_LATENCY_DAYS']))
if requested_end:
    end_date = pd.Timestamp(requested_end).normalize()
else:
    end_date = safe_end
if end_date > safe_end:
    end_date = safe_end
if requested_start:
    start_date = pd.Timestamp(requested_start).normalize()
else:
    start_date = None
    if end_date < default_start:
        start_date = end_date + pd.Timedelta(days=1)
    else:
        expected = pd.date_range(default_start, end_date, freq='D')
        existing = set()
        if os.path.exists(os.environ['PRODUCTION_ZARR']):
            root = zarr.open_group(os.environ['PRODUCTION_ZARR'], mode='r')
            if 'time' in root:
                time_array = root['time']
                time_size = int(time_array.shape[0]) if len(time_array.shape) > 0 else 0
                if time_size > 0:
                    times = pd.to_datetime(np.asarray(time_array[:], dtype=np.int64)).normalize()
                    existing = {pd.Timestamp(ts).normalize() for ts in times}
        missing = [ts for ts in expected if pd.Timestamp(ts).normalize() not in existing]
        if missing:
            start_date = pd.Timestamp(missing[0]).normalize()
        else:
            start_date = end_date + pd.Timedelta(days=1)
print(start_date.strftime('%Y-%m-%d'))
print(end_date.strftime('%Y-%m-%d'))
PY4
}

echo "Running low-latency daily orchestration"
echo "  active_model_label=${active_model_label}"
echo "  ensemble_outputs_root=${ensemble_outputs_root}"
echo "  input_data_name=${input_data_name}"

if ! mkdir "${lock_dir}" 2>/dev/null; then
    final_status="failed_lock_exists"
    message="Another low-latency daily update appears to be running: ${lock_dir}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

readarray -t range_values < <(resolve_range)
if [[ "${#range_values[@]}" -ne 2 ]]; then
    final_status="failed_range_resolution"
    message="Low-latency range resolution failed"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi
requested_start_date="${range_values[0]}"
requested_end_date="${range_values[1]}"

if [[ "${requested_start_date}" > "${requested_end_date}" ]]; then
    final_status="nothing_to_do"
    message="No low-latency dates need updating"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 0
fi

if ensure_on_scratch "${modis_path}" "canonical MODIS zarr" "0"; then bootstrap_modis="ready"; else bootstrap_modis="missing"; fi
if ensure_on_scratch "${modis_regrid_root}" "MODIS regrid root" "1"; then bootstrap_modis_regrid="ready_or_missing_allowed"; fi
if ensure_on_scratch "${low_latency_climate_path}" "low-latency climate zarr" "1"; then bootstrap_low_latency_climate="ready_or_missing_allowed"; fi
if ensure_on_scratch "${daymet_combined_path}" "Daymet clim20 zarr" "0"; then bootstrap_daymet_combined="ready"; fi
if ensure_on_scratch "${daymet_climatology_path}" "Daymet clim20 climatology zarr" "0"; then bootstrap_daymet_climatology="ready"; fi
if ensure_on_scratch "${nlcd_annual_path}" "annual NLCD zarr" "0"; then bootstrap_nlcd="ready"; else bootstrap_nlcd="missing"; fi
if ensure_on_scratch "${static_path}" "static dataset" "0"; then bootstrap_static="ready"; fi
if ensure_on_scratch "${soils_path}" "soils dataset" "0"; then bootstrap_soils="ready"; fi
if ensure_on_scratch "${canopy_height_path}" "canopy-height dataset" "0"; then bootstrap_canopy_height="ready"; fi
if ensure_on_scratch "${grid_path}" "grid" "0"; then bootstrap_grid="ready"; fi
if ensure_on_scratch "${ensemble_outputs_root}" "ensemble outputs" "0"; then bootstrap_models="ready"; fi
if ensure_on_scratch "${production_zarr}" "production LFMC zarr" "1"; then bootstrap_production="ready_or_missing_allowed"; fi

if python3 "${modis_update_script}" --start_date "${requested_start_date}" --end_date "${requested_end_date}" --tail_context_days "${modis_tail_context_days}" --registry_path "${registry_path}" --check_only; then
    modis_check_status="ready"
else
    modis_check_status="not_ready"
    final_status="not_ready"
    message="MODIS is not yet fully available for ${requested_start_date} -> ${requested_end_date}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 0
fi

if python3 "${climate_update_script}" --start_date "${requested_start_date}" --end_date "${requested_end_date}" --registry_path "${registry_path}" --check_only; then
    climate_check_status="ready"
else
    climate_check_status="not_ready"
    final_status="not_ready"
    message="Low-latency climate source is not yet fully available for ${requested_start_date} -> ${requested_end_date}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 0
fi

modis_update_status="submitted"
climate_update_status="submitted"
modis_job_id="$(submit_sbatch_job \
    "${modis_worker_script}" \
    "modis_range" \
    "START_DATE=${requested_start_date}" \
    "END_DATE=${requested_end_date}" \
    "TAIL_CONTEXT_DAYS=${modis_tail_context_days}" \
    "REGISTRY_PATH=${registry_path}")"
climate_job_id="$(submit_sbatch_job \
    "${climate_worker_script}" \
    "low_latency_climate" \
    "START_DATE=${requested_start_date}" \
    "END_DATE=${requested_end_date}" \
    "REGISTRY_PATH=${registry_path}")"

modis_wait_failed=0
climate_wait_failed=0

if wait_for_job "${modis_job_id}" "modis_range"; then
    modis_update_status="completed"
else
    modis_update_status="failed"
    modis_wait_failed=1
fi

if wait_for_job "${climate_job_id}" "low_latency_climate"; then
    climate_update_status="completed"
else
    climate_update_status="failed"
    climate_wait_failed=1
fi

if [[ "${modis_wait_failed}" == "1" || "${climate_wait_failed}" == "1" ]]; then
    final_status="failed_parallel_source_update"
    message="One or more low-latency source update jobs failed: modis=${modis_update_status} climate=${climate_update_status}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

python3 "${combined_weather_update_script}" \
    --standard_zarr "${low_latency_climate_path}" \
    --combined_zarr "${daymet_combined_path}" \
    --climatology_zarr "${daymet_climatology_path}" \
    --start_date "${requested_start_date}" \
    --end_date "${requested_end_date}"
combined_weather_update_status="completed"

source "${model_env}"
CONFIG_PATH="${config_path}" REGISTRY_PATH="${registry_path}" REQUESTED_START_DATE="${requested_start_date}" REQUESTED_END_DATE="${requested_end_date}" bash "${promotion_script}"
inference_status="completed"

if [[ "${skip_oak_sync}" == "1" ]]; then
    sync_back_status="skipped"
    echo "Skipping OAK sync because SKIP_OAK_SYNC=1"
else
    sync_to_oak "${modis_path}" "canonical MODIS zarr"
    sync_to_oak "${modis_regrid_root}" "MODIS regrid root"
    sync_to_oak "${low_latency_climate_path}" "low-latency climate zarr"
    sync_to_oak "${daymet_combined_path}" "Daymet clim20 zarr"
    sync_to_oak "${daymet_climatology_path}" "Daymet clim20 climatology zarr"
    sync_to_oak "${scientific_current_zarr}" "current scientific LFMC zarr"
    sync_to_oak "${production_zarr}" "production LFMC zarr"
    sync_to_oak "${metadata_dir}" "production metadata"
    sync_back_status="completed"
fi

final_status="completed"
message="Low-latency daily update completed for ${requested_start_date} -> ${requested_end_date}"
update_status_env
write_status >/dev/null

echo "${message}"
echo "  status_path=${status_path}"
