#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
repo_root="/home/users/trobinet/long_lfmc"
logs_dir="${script_dir}/logs"
mkdir -p "${logs_dir}"

update_year="${UPDATE_YEAR:-2025}"
update_years_csv="${UPDATE_YEARS_CSV:-}"
config_path="${CONFIG_PATH:-${script_dir}/map_configs_final_update.yaml}"
registry_path="${REGISTRY_PATH:-${script_dir}/source_registry.yaml}"
process_env="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
model_env="/home/users/trobinet/uv_activations/activate_lfmc_model_py312.sh"
current_model_script="${script_dir}/current_model_family_utils.py"

final_driver_script="${script_dir}/submit_final_year_promotion.sh"
daymet_update_script="${repo_root}/data_processing/daymet/update_daymet_archive_year.py"
daymet_combined_update_script="${repo_root}/data_processing/daymet/append_daymet_year_from_saved_climatology.py"
nlcd_update_script="${repo_root}/data_processing/nlcd/update_nlcd_year.py"
skip_oak_sync="${SKIP_OAK_SYNC:-0}"

source "${process_env}"

readarray -t model_values < <(python3 "${current_model_script}" --variant multitask --format lines)
active_model_label="${model_values[0]}"
ensemble_outputs_root="${model_values[1]}"
input_data_name="${model_values[2]}"

readarray -t registry_values < <(REGISTRY_PATH="${registry_path}" CONFIG_PATH="${config_path}" python3 - <<'PY2'
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
    registry['sources']['daymet']['combined_path'],
    registry['sources']['daymet']['climatology_path'],
    registry['sources']['daymet']['archive_path'],
    registry['processing']['daymet']['earthaccess_root'],
    registry['processing']['daymet']['daily_root'],
    registry['processing']['daymet']['regrid_root'],
    registry['sources']['nlcd']['raw_path'],
    registry['sources']['nlcd']['annual_path'],
    registry['processing']['nlcd']['raw_dir'],
    registry['sources']['static']['path'],
    registry['sources']['soils']['path'],
    registry['sources']['canopy_height']['path'],
    registry['sources']['scientific_current']['zarr_path'],
    registry['sources']['production']['zarr_path'],
    registry['sources']['production']['metadata_dir'],
    cfg['data']['grid_path'],
]
for value in values:
    print(value)
PY2
)

scratch_root="${registry_values[0]}"
oak_root="${registry_values[1]}"
modis_path="${registry_values[2]}"
daymet_combined_path="${registry_values[3]}"
daymet_climatology_path="${registry_values[4]}"
daymet_archive_path="${registry_values[5]}"
daymet_earthaccess_root="${registry_values[6]}"
daymet_daily_root="${registry_values[7]}"
daymet_regrid_root="${registry_values[8]}"
nlcd_raw_path="${registry_values[9]}"
nlcd_annual_path="${registry_values[10]}"
nlcd_raw_dir="${registry_values[11]}"
static_path="${registry_values[12]}"
soils_path="${registry_values[13]}"
canopy_height_path="${registry_values[14]}"
scientific_current_zarr="${registry_values[15]}"
production_zarr="${registry_values[16]}"
metadata_dir="${registry_values[17]}"
grid_path="${registry_values[18]}"

batch_stamp="$(date +%Y%m%d_%H%M%S)"
status_dir="${metadata_dir}/status_reports"
lock_dir="${metadata_dir}/locks/final_year_batch.lock"
mkdir -p "${status_dir}" "${metadata_dir}/locks"
status_path="${status_dir}/final_year_batch_${batch_stamp}.json"
probe_tsv="/tmp/final_year_probe_${batch_stamp}.tsv"
availability_tsv="/tmp/final_year_availability_${batch_stamp}.tsv"
completed_tsv="/tmp/final_year_completed_${batch_stamp}.tsv"
: > "${probe_tsv}"
: > "${availability_tsv}"
: > "${completed_tsv}"

bootstrap_modis="not_started"
bootstrap_daymet_archive="not_started"
bootstrap_daymet_combined="not_started"
bootstrap_daymet_climatology="not_started"
bootstrap_nlcd_raw="not_started"
bootstrap_nlcd_annual="not_started"
bootstrap_static="not_started"
bootstrap_soils="not_started"
bootstrap_canopy_height="not_started"
bootstrap_grid="not_started"
bootstrap_models="not_started"
bootstrap_production="not_started"
daymet_update_status="not_started"
nlcd_update_status="not_started"
inference_status="not_started"
sync_back_status="not_started"
final_status="started"
message="started"

cleanup() {
    rmdir "${lock_dir}" 2>/dev/null || true
    rm -f "${probe_tsv}" "${availability_tsv}" "${completed_tsv}"
}
trap cleanup EXIT INT TERM

write_status() {
    python3 - <<'PY3'
import json
import os
from pathlib import Path


def load_tsv(path_str):
    path = Path(path_str)
    rows = []
    if not path.exists():
        return rows
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split('\t')
        rows.append(parts)
    return rows

path = Path(os.environ['STATUS_PATH'])
path.parent.mkdir(parents=True, exist_ok=True)
probe_rows = load_tsv(os.environ['PROBE_TSV'])
availability_rows = load_tsv(os.environ['AVAILABILITY_TSV'])
completed_rows = load_tsv(os.environ['COMPLETED_TSV'])
record = {
    'final_status': os.environ['FINAL_STATUS'],
    'message': os.environ['MESSAGE'],
    'update_year_default': os.environ['UPDATE_YEAR_DEFAULT'],
    'update_years_csv': os.environ['UPDATE_YEARS_CSV'],
    'bootstrap': {
        'modis': os.environ['BOOTSTRAP_MODIS'],
        'daymet_archive': os.environ['BOOTSTRAP_DAYMET_ARCHIVE'],
        'daymet_combined': os.environ['BOOTSTRAP_DAYMET_COMBINED'],
        'daymet_climatology': os.environ['BOOTSTRAP_DAYMET_CLIMATOLOGY'],
        'nlcd_raw': os.environ['BOOTSTRAP_NLCD_RAW'],
        'nlcd_annual': os.environ['BOOTSTRAP_NLCD_ANNUAL'],
        'static': os.environ['BOOTSTRAP_STATIC'],
        'soils': os.environ['BOOTSTRAP_SOILS'],
        'canopy_height': os.environ['BOOTSTRAP_CANOPY_HEIGHT'],
        'grid': os.environ['BOOTSTRAP_GRID'],
        'ensemble_outputs': os.environ['BOOTSTRAP_MODELS'],
        'production_zarr': os.environ['BOOTSTRAP_PRODUCTION'],
    },
    'updates': {
        'daymet': os.environ['DAYMET_UPDATE_STATUS'],
        'nlcd': os.environ['NLCD_UPDATE_STATUS'],
    },
    'inference_status': os.environ['INFERENCE_STATUS'],
    'sync_back_status': os.environ['SYNC_BACK_STATUS'],
    'probe_results': [
        {'year': int(year), 'status': status, 'reason': reason}
        for year, status, reason in probe_rows
    ],
    'availability_results': [
        {'year': int(year), 'status': status, 'reason': reason}
        for year, status, reason in availability_rows
    ],
    'completed_years': [int(year) for year in completed_rows],
    'paths': {
        'modis_path': os.environ['MODIS_PATH'],
        'daymet_archive_path': os.environ['DAYMET_ARCHIVE_PATH'],
        'nlcd_raw_path': os.environ['NLCD_RAW_PATH'],
        'nlcd_annual_path': os.environ['NLCD_ANNUAL_PATH'],
        'production_zarr': os.environ['PRODUCTION_ZARR'],
        'metadata_dir': os.environ['METADATA_DIR'],
        'grid_path': os.environ['GRID_PATH'],
        'ensemble_outputs_root': os.environ['ENSEMBLE_OUTPUTS_ROOT'],
    },
}
path.write_text(json.dumps(record, indent=2, sort_keys=True))
print(path)
PY3
}

export STATUS_PATH="${status_path}"
export PROBE_TSV="${probe_tsv}"
export AVAILABILITY_TSV="${availability_tsv}"
export COMPLETED_TSV="${completed_tsv}"
export UPDATE_YEAR_DEFAULT="${update_year}"
export UPDATE_YEARS_CSV="${update_years_csv}"
export MODIS_PATH="${modis_path}"
export DAYMET_ARCHIVE_PATH="${daymet_archive_path}"
export NLCD_RAW_PATH="${nlcd_raw_path}"
export NLCD_ANNUAL_PATH="${nlcd_annual_path}"
export PRODUCTION_ZARR="${production_zarr}"
export METADATA_DIR="${metadata_dir}"
export GRID_PATH="${grid_path}"
export ENSEMBLE_OUTPUTS_ROOT="${ensemble_outputs_root}"

update_status_env() {
    export BOOTSTRAP_MODIS="${bootstrap_modis}"
    export BOOTSTRAP_DAYMET_ARCHIVE="${bootstrap_daymet_archive}"
    export BOOTSTRAP_DAYMET_COMBINED="${bootstrap_daymet_combined}"
    export BOOTSTRAP_DAYMET_CLIMATOLOGY="${bootstrap_daymet_climatology}"
    export BOOTSTRAP_NLCD_RAW="${bootstrap_nlcd_raw}"
    export BOOTSTRAP_NLCD_ANNUAL="${bootstrap_nlcd_annual}"
    export BOOTSTRAP_STATIC="${bootstrap_static}"
    export BOOTSTRAP_SOILS="${bootstrap_soils}"
    export BOOTSTRAP_CANOPY_HEIGHT="${bootstrap_canopy_height}"
    export BOOTSTRAP_GRID="${bootstrap_grid}"
    export BOOTSTRAP_MODELS="${bootstrap_models}"
    export BOOTSTRAP_PRODUCTION="${bootstrap_production}"
    export DAYMET_UPDATE_STATUS="${daymet_update_status}"
    export NLCD_UPDATE_STATUS="${nlcd_update_status}"
    export INFERENCE_STATUS="${inference_status}"
    export SYNC_BACK_STATUS="${sync_back_status}"
    export FINAL_STATUS="${final_status}"
    export MESSAGE="${message}"
}

if ! mkdir "${lock_dir}" 2>/dev/null; then
    final_status="failed_lock_exists"
    message="Another final-year batch update appears to be running: ${lock_dir}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

scratch_to_oak() {
    local scratch_path="$1"
    python3 - <<'PY4'
import os
scratch_root = os.environ['SCRATCH_ROOT']
oak_root = os.environ['OAK_ROOT']
path = os.environ['SCRATCH_PATH']
if not path.startswith(scratch_root + '/') and path != scratch_root:
    print('')
    raise SystemExit(0)
rel = os.path.relpath(path, scratch_root)
print(os.path.join(oak_root, rel))
PY4
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
    export SCRATCH_ROOT="${scratch_root}"
    export OAK_ROOT="${oak_root}"
    export SCRATCH_PATH="${scratch_path}"
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
    export SCRATCH_ROOT="${scratch_root}"
    export OAK_ROOT="${oak_root}"
    export SCRATCH_PATH="${scratch_path}"
    oak_path="$(scratch_to_oak "${scratch_path}")"
    if [[ -z "${oak_path}" ]]; then
        echo "Skipping OAK sync for ${label}; could not map scratch path: ${scratch_path}"
        return 0
    fi
    echo "Syncing ${label} back to OAK: ${scratch_path} -> ${oak_path}"
    rsync_path "${scratch_path}" "${oak_path}"
}

probe_candidate_years() {
    PRODUCTION_ZARR="${production_zarr}" UPDATE_YEARS_CSV="${update_years_csv}" python3 - <<'PY5' > "${probe_tsv}"
import calendar
import os

import numpy as np
import pandas as pd
import zarr

production_zarr = os.environ['PRODUCTION_ZARR']
manual_years = [s.strip() for s in os.environ.get('UPDATE_YEARS_CSV', '').split(',') if s.strip()]
if manual_years:
    for year_str in manual_years:
        print(f'{int(year_str)}\tcandidate\tmanual_override')
    raise SystemExit(0)
if not os.path.exists(production_zarr):
    raise SystemExit(0)
root = zarr.open_group(production_zarr, mode='r')
if 'time' not in root or 'quality_flag' not in root:
    raise SystemExit(0)
times = pd.to_datetime(np.asarray(root['time'][:], dtype=np.int64))
quality = np.asarray(root['quality_flag'][:], dtype=np.uint8)
current_year = pd.Timestamp.today().year
for year in sorted(set(times.year.tolist())):
    if year >= current_year:
        print(f'{year}\tskip\tcurrent_or_future_year')
        continue
    mask = times.year == year
    expected_days = 366 if calendar.isleap(int(year)) else 365
    observed_days = int(mask.sum())
    if observed_days < expected_days:
        print(f'{year}\tskip\tincomplete_year_{observed_days}_of_{expected_days}')
        continue
    qvals = sorted(set(quality[mask].tolist()))
    if qvals == [0]:
        print(f'{year}\tskip\talready_final')
        continue
    print(f'{year}\tcandidate\tquality_flags_{"-".join(str(v) for v in qvals)}')
PY5
}

echo "Running yearly final orchestration batch"
echo "  update_year_default=${update_year}"
echo "  update_years_csv=${update_years_csv}"
echo "  active_model_label=${active_model_label}"
echo "  ensemble_outputs_root=${ensemble_outputs_root}"
echo "  input_data_name=${input_data_name}"

if ensure_on_scratch "${production_zarr}" "production LFMC zarr" "1"; then
    bootstrap_production="ready_or_missing_allowed"
else
    bootstrap_production="missing"
fi

probe_candidate_years
candidate_years=()
echo "Year probe summary:"
while IFS=$'\t' read -r year status reason; do
    [[ -z "${year:-}" ]] && continue
    echo "  year=${year} status=${status} reason=${reason}"
    if [[ "${status}" == "candidate" ]]; then
        candidate_years+=("${year}")
    fi
done < "${probe_tsv}"

passed_years=()
echo "Availability check summary:"
for year in "${candidate_years[@]}"; do
    daymet_reason="daymet_confirmed"
    nlcd_reason="nlcd_confirmed"
    if python3 "${daymet_update_script}" \
        --year "${year}" \
        --earthaccess_root "${daymet_earthaccess_root}" \
        --daily_root "${daymet_daily_root}" \
        --regrid_root "${daymet_regrid_root}" \
        --archive_zarr "${daymet_archive_path}" \
        --check_only; then
        :
    else
        daymet_reason="daymet_unavailable"
    fi
    if python3 "${nlcd_update_script}" \
        --year "${year}" \
        --raw_dir "${nlcd_raw_dir}" \
        --raw_zarr "${nlcd_raw_path}" \
        --target_zarr "${nlcd_annual_path}" \
        --check_only; then
        :
    else
        nlcd_reason="nlcd_unavailable"
    fi
    if [[ "${daymet_reason}" == "daymet_confirmed" && "${nlcd_reason}" == "nlcd_confirmed" ]]; then
        echo -e "${year}\tpassed\tboth_sources_confirmed" >> "${availability_tsv}"
        passed_years+=("${year}")
        echo "  year=${year} status=passed reason=both_sources_confirmed"
    else
        reason="${daymet_reason};${nlcd_reason}"
        echo -e "${year}\tfailed\t${reason}" >> "${availability_tsv}"
        echo "  year=${year} status=failed reason=${reason}"
    fi
done

if [[ ${#candidate_years[@]} -eq 0 ]]; then
    final_status="completed_no_action"
    message="No fully completed low-quality past years were found in the production dataset"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    echo "  status_path=${status_path}"
    exit 0
fi

if [[ ${#passed_years[@]} -eq 0 ]]; then
    final_status="completed_no_action"
    message="No candidate years passed the Daymet/NLCD availability checks"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    echo "  status_path=${status_path}"
    exit 0
fi

if ensure_on_scratch "${modis_path}" "MODIS canonical zarr" "0"; then bootstrap_modis="ready"; else bootstrap_modis="missing"; fi
if ensure_on_scratch "${daymet_archive_path}" "Daymet archive zarr" "1"; then bootstrap_daymet_archive="ready_or_missing_allowed"; else bootstrap_daymet_archive="missing"; fi
if ensure_on_scratch "${daymet_combined_path}" "Daymet combined clim20 zarr" "0"; then bootstrap_daymet_combined="ready"; else bootstrap_daymet_combined="missing"; fi
if ensure_on_scratch "${daymet_climatology_path}" "Daymet climatology zarr" "0"; then bootstrap_daymet_climatology="ready"; else bootstrap_daymet_climatology="missing"; fi
if ensure_on_scratch "${nlcd_raw_path}" "NLCD raw zarr" "1"; then bootstrap_nlcd_raw="ready_or_missing_allowed"; else bootstrap_nlcd_raw="missing"; fi
if ensure_on_scratch "${nlcd_annual_path}" "NLCD annual target zarr" "0"; then bootstrap_nlcd_annual="ready"; else bootstrap_nlcd_annual="missing"; fi
if ensure_on_scratch "${static_path}" "static dataset" "0"; then bootstrap_static="ready"; else bootstrap_static="missing"; fi
if ensure_on_scratch "${soils_path}" "soils dataset" "0"; then bootstrap_soils="ready"; else bootstrap_soils="missing"; fi
if ensure_on_scratch "${canopy_height_path}" "canopy-height dataset" "0"; then bootstrap_canopy_height="ready"; else bootstrap_canopy_height="missing"; fi
if ensure_on_scratch "${grid_path}" "target grid" "0"; then bootstrap_grid="ready"; else bootstrap_grid="missing"; fi
if ensure_on_scratch "${ensemble_outputs_root}" "ensemble outputs root" "0"; then bootstrap_models="ready"; else bootstrap_models="missing"; fi

if [[ ! -e "${modis_path}" ]]; then
    final_status="failed_missing_modis"
    message="MODIS canonical zarr is missing on both scratch and OAK: ${modis_path}"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

if [[ "${bootstrap_daymet_combined}" != "ready" || "${bootstrap_daymet_climatology}" != "ready" || "${bootstrap_nlcd_annual}" != "ready" || "${bootstrap_static}" != "ready" || "${bootstrap_soils}" != "ready" || "${bootstrap_canopy_height}" != "ready" || "${bootstrap_grid}" != "ready" || "${bootstrap_models}" != "ready" ]]; then
    final_status="failed_missing_required_input"
    message="One or more required scratch inputs are missing and could not be copied from OAK"
    update_status_env
    write_status >/dev/null
    echo "${message}"
    exit 1
fi

source "${model_env}"
for year in "${passed_years[@]}"; do
    echo "Running final upgrade for confirmed year ${year}"
    source "${process_env}"
    if python3 "${daymet_update_script}" \
        --year "${year}" \
        --earthaccess_root "${daymet_earthaccess_root}" \
        --daily_root "${daymet_daily_root}" \
        --regrid_root "${daymet_regrid_root}" \
        --archive_zarr "${daymet_archive_path}"; then
        daymet_update_status="completed"
    else
        daymet_update_status="failed_or_unavailable"
        final_status="failed_daymet_update"
        message="Daymet archive update failed for year ${year}"
        update_status_env
        write_status >/dev/null
        echo "${message}"
        exit 1
    fi
    if python3 "${daymet_combined_update_script}" \
        --raw_archive_zarr "${daymet_archive_path}" \
        --combined_zarr "${daymet_combined_path}" \
        --climatology_zarr "${daymet_climatology_path}" \
        --year "${year}"; then
        daymet_update_status="completed_with_combined_append"
    else
        daymet_update_status="failed_combined_append"
        final_status="failed_daymet_combined_update"
        message="Daymet combined clim20 update failed for year ${year}"
        update_status_env
        write_status >/dev/null
        echo "${message}"
        exit 1
    fi
    if python3 "${nlcd_update_script}" \
        --year "${year}" \
        --raw_dir "${nlcd_raw_dir}" \
        --raw_zarr "${nlcd_raw_path}" \
        --target_zarr "${nlcd_annual_path}"; then
        nlcd_update_status="completed"
    else
        nlcd_update_status="failed_or_unavailable"
        final_status="failed_nlcd_update"
        message="NLCD update failed for year ${year}"
        update_status_env
        write_status >/dev/null
        echo "${message}"
        exit 1
    fi
    source "${model_env}"
    if UPDATE_YEAR="${year}" SKIP_SOURCE_UPDATE=1 CONFIG_PATH="${config_path}" REGISTRY_PATH="${registry_path}" ENSEMBLE_ROOT="${ensemble_outputs_root}" INPUT_DATA_NAME="${input_data_name}" bash "${final_driver_script}"; then
        inference_status="completed"
        echo "${year}" >> "${completed_tsv}"
    else
        inference_status="failed"
        final_status="failed_inference_or_promotion"
        message="Final-year inference or production promotion failed for year ${year}"
        update_status_env
        write_status >/dev/null
        echo "${message}"
        exit 1
    fi
done

source "${process_env}"
if [[ "${skip_oak_sync}" == "1" ]]; then
    sync_back_status="skipped"
else
    if sync_to_oak "${daymet_archive_path}" "Daymet archive zarr" && \
       sync_to_oak "${daymet_combined_path}" "Daymet clim20 zarr" && \
       sync_to_oak "${daymet_climatology_path}" "Daymet clim20 climatology zarr" && \
       sync_to_oak "${nlcd_raw_path}" "NLCD raw zarr" && \
       sync_to_oak "${nlcd_annual_path}" "NLCD annual target zarr" && \
       sync_to_oak "${scientific_current_zarr}" "current scientific LFMC zarr" && \
       sync_to_oak "${production_zarr}" "production LFMC zarr" && \
       sync_to_oak "${metadata_dir}" "production metadata directory"; then
        sync_back_status="completed"
    else
        sync_back_status="failed"
        final_status="failed_oak_sync"
        message="One or more OAK sync-back steps failed"
        update_status_env
        write_status >/dev/null
        echo "${message}"
        exit 1
    fi
fi

final_status="completed"
message="Yearly final upgrades completed successfully for confirmed years: ${passed_years[*]}"
update_status_env
write_status >/dev/null

echo "${message}"
echo "  status_path=${status_path}"
echo "  production_zarr=${production_zarr}"
