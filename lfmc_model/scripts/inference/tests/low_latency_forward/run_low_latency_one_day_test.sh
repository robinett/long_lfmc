#!/bin/bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/tests/low_latency_forward"
coordinator_job_id="$(
    sbatch \
        --parsable \
        --export=NONE \
        "${script_dir}/run_low_latency_one_day_inference_test.sbatch"
)"

echo "Submitted low-latency one-day debug workflow"
echo "COORDINATOR_JOB_ID=${coordinator_job_id}"
