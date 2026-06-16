#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/rao_s1_source"
transfer_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out"
config_path="${script_dir}/rao_s1_source_config.yaml"
env_path="/home/users/trobinet/uv_activations/activate_lfmc_process_py312.sh"
logs_dir="${script_dir}/logs"

target_date=""
today_date=""
dry_run=0
skip_generation=0
skip_upload=0

usage() {
    cat <<'USAGE'
Usage: run_rao_s1_source_update.sh [--target-date YYYY-MM-DD] [--today YYYY-MM-DD] [--dry-run] [--skip-generation] [--skip-upload]

Build/publish the Rao S1-informed LFMC Source artifacts for the eligible 1st/15th
date. In scheduled use, run on the 11th and 26th; the wrapper applies 10-day
latency to choose the 1st or 15th target date.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-date)
            target_date="$2"
            shift 2
            ;;
        --today)
            today_date="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --skip-generation)
            skip_generation=1
            shift
            ;;
        --skip-upload)
            skip_upload=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "${logs_dir}"
cd "${script_dir}"
source "${env_path}"

if [[ -z "${target_date}" ]]; then
    target_date="$(python3 - "${today_date}" <<'PY'
import datetime as dt
import sys

today_arg = sys.argv[1].strip()
if today_arg:
    today = dt.datetime.strptime(today_arg, "%Y-%m-%d").date()
else:
    today = dt.date.today()
if today.day >= 26:
    target = today.replace(day=15)
elif today.day >= 11:
    target = today.replace(day=1)
else:
    first_this_month = today.replace(day=1)
    previous_month_last = first_this_month - dt.timedelta(days=1)
    target = previous_month_last.replace(day=15)
print(target.isoformat())
PY
)"
fi

if ! [[ "${target_date}" =~ ^[0-9]{4}-[0-9]{2}-(01|15)$ ]]; then
    echo "[ERROR] target date must be YYYY-MM-DD with day 01 or 15, got ${target_date}" >&2
    exit 1
fi

lfmc_maps_dir="$(python3 - "${config_path}" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as file_obj:
    cfg = yaml.safe_load(file_obj)
print(cfg["paths"]["lfmc_maps_dir"])
PY
)"
rao_pipeline_script="$(python3 - "${config_path}" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as file_obj:
    cfg = yaml.safe_load(file_obj)
print(cfg["paths"]["rao_pipeline_script"])
PY
)"
lfmc_map="${lfmc_maps_dir}/lfmc_map_${target_date}.tif"
echo "[INFO] Rao S1-informed LFMC Source update target: ${target_date}"
echo "[INFO] LFMC map path: ${lfmc_map}"

run_cmd() {
    if [[ "${dry_run}" -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

if [[ ! -f "${lfmc_map}" ]]; then
    if [[ "${skip_generation}" -eq 1 ]]; then
        echo "[ERROR] LFMC map is missing and --skip-generation was set: ${lfmc_map}" >&2
        exit 1
    fi
    echo "[INFO] LFMC map missing; Rao pipeline would generate it."
    run_cmd sbatch --wait "${rao_pipeline_script}" "${target_date}"
else
    echo "[INFO] LFMC map already exists; generation step is not needed."
fi

run_cmd python3 -u "${script_dir}/build_rao_s1_scientific_zarr.py" \
    --config "${config_path}" \
    --mode append \
    --target-dates "${target_date}"

run_cmd python3 -u "${script_dir}/build_rao_s1_viewer_zarr.py" \
    --config "${config_path}" \
    --mode append \
    --target-dates "${target_date}"

run_cmd python3 -u "${script_dir}/validate_rao_s1_source_products.py" \
    --config "${config_path}" \
    --target-date "${target_date}"

run_cmd python3 -u "${script_dir}/build_rao_s1_viewer_tiles.py" \
    --config "${config_path}" \
    --dates "${target_date}" \
    --layers lfmc anomaly

run_cmd python3 -u "${script_dir}/validate_rao_s1_source_products.py" \
    --config "${config_path}" \
    --target-date "${target_date}" \
    --check-assets

manifest_dir="/home/users/trobinet/long_lfmc/logs/rao_s1_source_manifests/${target_date}"
run_cmd python3 -u "${script_dir}/build_rao_s1_source_upload_manifests.py" \
    --config "${config_path}" \
    --target-date "${target_date}" \
    --output-dir "${manifest_dir}"

if [[ "${skip_upload}" -eq 1 ]]; then
    echo "[INFO] --skip-upload set; not uploading Source artifacts."
    exit 0
fi

if [[ "${dry_run}" -eq 1 ]]; then
    run_cmd bash "${transfer_dir}/run_upload_rao_s1_source_products.sh" \
        --dry-run \
        --target-date "${target_date}" \
        --manifest-dir "${manifest_dir}"
else
    run_cmd bash "${transfer_dir}/run_upload_rao_s1_source_products.sh" \
        --target-date "${target_date}" \
        --manifest-dir "${manifest_dir}"
fi

echo "[INFO] Rao S1-informed LFMC Source update complete for ${target_date}"
