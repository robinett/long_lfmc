#!/bin/bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/tests/archive_forward"
stage_job_id="$(
    sbatch \
        --parsable \
        --export=NONE \
        "${script_dir}/stage_archive_forward_setup.sbatch"
)"
coordinator_job_id="$(
    sbatch \
        --parsable \
        --export=NONE \
        --dependency=afterok:${stage_job_id} \
        "${script_dir}/run_archive_forward_inference_test.sbatch"
)"

echo "Submitted archive-forward test workflow"
echo "STAGE_JOB_ID=${stage_job_id}"
echo "COORDINATOR_JOB_ID=${coordinator_job_id}"
