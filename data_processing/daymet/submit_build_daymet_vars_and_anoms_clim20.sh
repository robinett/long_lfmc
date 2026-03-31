#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

mkdir -p ./logs

smoke_test=0
if [[ "${1:-}" == "--smoke-test" ]]; then
    smoke_test=1
    shift
fi

if [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--smoke-test]"
    exit 1
fi

config_path="${SCRIPT_DIR}/configs_clim20.yaml"
run_script="${SCRIPT_DIR}/run_build_daymet_derived_products.sh"
array_task_script="${SCRIPT_DIR}/run_build_daymet_derived_products_array_task.sh"

mapfile -t config_values < <(
python3 - <<'PY' "${config_path}"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

processing = cfg["processing"]
for var_name in processing["standard_variable_order"]:
    print(f"STD:{var_name}")
for var_name in processing["anomaly_variable_order"]:
    print(f"ANOM:{var_name}")
print(f"NUM_SHARDS:{int(processing.get('num_shards', 24))}")
print(f"MAX_JOBS:{int(processing.get('max_concurrent_anomaly_tasks', 50))}")
PY
)

standard_vars=()
anomaly_vars=()
num_shards=""
max_jobs=""
for entry in "${config_values[@]}"; do
    case "${entry}" in
        STD:*)
            standard_vars+=("${entry#STD:}")
            ;;
        ANOM:*)
            anomaly_vars+=("${entry#ANOM:}")
            ;;
        NUM_SHARDS:*)
            num_shards="${entry#NUM_SHARDS:}"
            ;;
        MAX_JOBS:*)
            max_jobs="${entry#MAX_JOBS:}"
            ;;
    esac
done

if [[ ${#standard_vars[@]} -eq 0 || ${#anomaly_vars[@]} -eq 0 || -z "${num_shards}" || -z "${max_jobs}" ]]; then
    echo "Failed to load workflow settings from ${config_path}" >&2
    exit 1
fi

sbatch_submit() {
    local cmd=(sbatch --parsable --chdir "$SCRIPT_DIR" "$@")
    "${cmd[@]}"
}

run_script_submit() {
    local slurm_args=()
    local script_extra_args=()
    local parsing_script_args=0
    local arg
    for arg in "$@"; do
        if [[ "${parsing_script_args}" -eq 0 ]]; then
            if [[ "${arg}" == "--" ]]; then
                parsing_script_args=1
                continue
            fi
            slurm_args+=("${arg}")
        else
            script_extra_args+=("${arg}")
        fi
    done
    local script_args=(
        --config "${config_path}"
    )
    if [[ "${smoke_test}" -eq 1 ]]; then
        script_args+=(--smoke-test)
    fi
    script_args+=("${script_extra_args[@]}")
    sbatch_submit "${slurm_args[@]}" "${run_script}" "${script_args[@]}"
}

echo "Submitting Daymet clim20 workflow"
echo "Config: ${config_path}"
echo "Smoke test: ${smoke_test}"
echo "Anomaly shards per variable: ${num_shards}"
echo "Max concurrent anomaly tasks: ${max_jobs}"

init_job_id=$(
    run_script_submit \
        --job-name=daymet_c20_init_store \
        --output=./logs/daymet_c20_init_store_%j.out \
        --error=./logs/daymet_c20_init_store_%j.err \
        -- \
        --mode init-store
)

echo "Submitted init-store job: ${init_job_id}"

standard_job_ids=()
for var_name in "${standard_vars[@]}"; do
    this_job_id=$(
        run_script_submit \
            --dependency=afterok:"${init_job_id}" \
            --job-name="daymet_c20_std_${var_name}" \
            --output="./logs/daymet_c20_std_${var_name}_%j.out" \
            --error="./logs/daymet_c20_std_${var_name}_%j.err" \
            -- \
            --mode build-standard-var \
            --var "${var_name}"
    )
    standard_job_ids+=("${this_job_id}")
    echo "Submitted standard worker ${var_name}: ${this_job_id}"
done

standard_dep=$(IFS=:; echo "${standard_job_ids[*]}")
finalize_standard_job_id=$(
    run_script_submit \
        --dependency=afterok:"${standard_dep}" \
        --job-name=daymet_c20_finalize_standard \
        --output=./logs/daymet_c20_finalize_standard_%j.out \
        --error=./logs/daymet_c20_finalize_standard_%j.err \
        -- \
        --mode finalize-standard
)

echo "Submitted finalize-standard job: ${finalize_standard_job_id}"

init_anomaly_job_id=$(
    run_script_submit \
        --dependency=afterok:"${finalize_standard_job_id}" \
        --job-name=daymet_c20_init_anomaly \
        --output=./logs/daymet_c20_init_anomaly_%j.out \
        --error=./logs/daymet_c20_init_anomaly_%j.err \
        -- \
        --mode init-anomaly
)

echo "Submitted init-anomaly job: ${init_anomaly_job_id}"

total_anomaly_tasks=$(( ${#anomaly_vars[@]} * num_shards ))
array_spec="0-$((total_anomaly_tasks - 1))%${max_jobs}"
anomaly_array_job_id=$(
    sbatch_submit \
        --dependency=afterok:"${init_anomaly_job_id}" \
        --array="${array_spec}" \
        --job-name=daymet_c20_anom \
        --output=./logs/daymet_c20_anom_%A_%a.out \
        --error=./logs/daymet_c20_anom_%A_%a.err \
        "${array_task_script}" \
        "${config_path}" \
        "${num_shards}" \
        "${smoke_test}" \
        "${anomaly_vars[@]}"
)

echo "Submitted anomaly array job: ${anomaly_array_job_id}"
echo "Array spec: ${array_spec} (${#anomaly_vars[@]} vars x ${num_shards} shards)"

finalize_anomaly_job_id=$(
    run_script_submit \
        --dependency=afterok:"${anomaly_array_job_id}" \
        --job-name=daymet_c20_finalize_anomaly \
        --output=./logs/daymet_c20_finalize_anomaly_%j.out \
        --error=./logs/daymet_c20_finalize_anomaly_%j.err \
        -- \
        --mode finalize-anomaly
)

echo "Submitted finalize-anomaly job: ${finalize_anomaly_job_id}"
echo "Daymet clim20 workflow submission complete."
