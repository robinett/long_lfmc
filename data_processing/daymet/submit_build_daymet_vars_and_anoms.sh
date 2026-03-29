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

run_script="${SCRIPT_DIR}/run_build_daymet_derived_products.sh"

standard_vars=(tmax vpd prcp srad swe)
anomaly_vars=(tmax_daily_anom vpd_daily_anom prcp_rolling30_anom srad_daily_anom swe_daily_anom)

sbatch_submit() {
    local cmd=(sbatch --parsable --chdir "$SCRIPT_DIR" "$@")
    if [[ "$smoke_test" -eq 1 ]]; then
        cmd+=(--smoke-test)
    fi
    "${cmd[@]}"
}

init_job_id=$(
    sbatch_submit \
        --job-name=daymet_init_store \
        --output=./logs/daymet_init_store_%j.out \
        --error=./logs/daymet_init_store_%j.err \
        "$run_script" \
        --mode init-store
)

echo "Submitted init-store job: ${init_job_id}"

standard_job_ids=()
for var_name in "${standard_vars[@]}"; do
    this_job_id=$(
        sbatch_submit \
            --dependency=afterok:"${init_job_id}" \
            --job-name="daymet_std_${var_name}" \
            --output="./logs/daymet_std_${var_name}_%j.out" \
            --error="./logs/daymet_std_${var_name}_%j.err" \
            "$run_script" \
            --mode build-standard-var \
            --var "${var_name}"
    )
    standard_job_ids+=("${this_job_id}")
    echo "Submitted standard worker ${var_name}: ${this_job_id}"
done

standard_dep=$(IFS=:; echo "${standard_job_ids[*]}")
finalize_standard_job_id=$(
    sbatch_submit \
        --dependency=afterok:"${standard_dep}" \
        --job-name=daymet_finalize_standard \
        --output=./logs/daymet_finalize_standard_%j.out \
        --error=./logs/daymet_finalize_standard_%j.err \
        "$run_script" \
        --mode finalize-standard
)

echo "Submitted finalize-standard job: ${finalize_standard_job_id}"

init_anomaly_job_id=$(
    sbatch_submit \
        --dependency=afterok:"${finalize_standard_job_id}" \
        --job-name=daymet_init_anomaly \
        --output=./logs/daymet_init_anomaly_%j.out \
        --error=./logs/daymet_init_anomaly_%j.err \
        "$run_script" \
        --mode init-anomaly
)

echo "Submitted init-anomaly job: ${init_anomaly_job_id}"

anomaly_job_ids=()
for var_name in "${anomaly_vars[@]}"; do
    this_job_id=$(
        sbatch_submit \
            --dependency=afterok:"${init_anomaly_job_id}" \
            --job-name="daymet_anom_${var_name}" \
            --output="./logs/daymet_anom_${var_name}_%j.out" \
            --error="./logs/daymet_anom_${var_name}_%j.err" \
            "$run_script" \
            --mode build-anomaly-var \
            --var "${var_name}"
    )
    anomaly_job_ids+=("${this_job_id}")
    echo "Submitted anomaly worker ${var_name}: ${this_job_id}"
done

anomaly_dep=$(IFS=:; echo "${anomaly_job_ids[*]}")
finalize_anomaly_job_id=$(
    sbatch_submit \
        --dependency=afterok:"${anomaly_dep}" \
        --job-name=daymet_finalize_anomaly \
        --output=./logs/daymet_finalize_anomaly_%j.out \
        --error=./logs/daymet_finalize_anomaly_%j.err \
        "$run_script" \
        --mode finalize-anomaly
)

echo "Submitted finalize-anomaly job: ${finalize_anomaly_job_id}"
echo "Workflow submission complete."
