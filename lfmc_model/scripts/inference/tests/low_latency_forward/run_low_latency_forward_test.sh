#!/bin/bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/tests/low_latency_forward"
log_dir="/home/users/trobinet/long_lfmc/logs/low_latency_forward_setup"
stage_export="ALL,RESET_PROCESSING_ROOTS=1,LFMC_BASELINE_START_DATE=2023-01-01,LFMC_BASELINE_END_DATE=2023-12-31,SOURCE_BASELINE_START_DATE=2022-01-01,SOURCE_BASELINE_END_DATE=2022-12-31,NLCD_BASELINE_START_DATE=2023-01-01,NLCD_BASELINE_END_DATE=2023-12-31,TEST_START_DATE=2024-01-01,TEST_END_DATE=2024-12-31,SOURCE_PREWARM_START_DATE=2023-01-01,SOURCE_PREWARM_END_DATE=2023-12-31,TODAY_OVERRIDE=2025-01-07"
prewarm_export="ALL,REQUESTED_START_DATE=2023-01-01,REQUESTED_END_DATE=2023-12-31,TODAY_OVERRIDE=2025-01-07"
coordinator_export="ALL,REQUESTED_START_DATE=2024-01-01,REQUESTED_END_DATE=2024-12-31,TODAY_OVERRIDE=2025-01-07"
mkdir -p "${log_dir}"
stage_job_id="$(
    sbatch \
        --parsable \
        --export="${stage_export}" \
        "${script_dir}/stage_low_latency_forward_setup.sbatch"
)"
prewarm_job_id="$(
    sbatch \
        --parsable \
        --export="${prewarm_export}" \
        --dependency=afterok:${stage_job_id} \
        "${script_dir}/run_low_latency_source_prewarm_test.sbatch"
)"
coordinator_job_id="$(
    sbatch \
        --parsable \
        --export="${coordinator_export}" \
        --dependency=afterok:${prewarm_job_id} \
        "${script_dir}/run_low_latency_forward_inference_test.sbatch"
)"

echo "Submitted low-latency forward test workflow"
echo "STAGE_JOB_ID=${stage_job_id}"
echo "SOURCE_PREWARM_JOB_ID=${prewarm_job_id}"
echo "COORDINATOR_JOB_ID=${coordinator_job_id}"
