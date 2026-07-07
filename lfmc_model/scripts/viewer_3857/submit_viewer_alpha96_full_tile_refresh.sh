#!/usr/bin/env bash

set -euo pipefail

repo_dir="/home/users/trobinet/long_lfmc"
script_dir="${repo_dir}/lfmc_model/scripts/viewer_3857"
logs_dir="${repo_dir}/logs/viewer_alpha96_full_tiles"
coordinator="${script_dir}/run_viewer_alpha96_full_tile_refresh_coordinator.sh"

mkdir -p "${logs_dir}"

coordinator_job_id="$(sbatch --parsable \
    --partition=serc \
    --time=01:00:00 \
    --cpus-per-task=2 \
    --mem=16G \
    --output="${logs_dir}/viewer_alpha96_coordinator_%j.out" \
    --error="${logs_dir}/viewer_alpha96_coordinator_%j.err" \
    --wrap="bash ${coordinator}")"

echo "Submitted viewer alpha-96 full tile refresh coordinator ${coordinator_job_id}"
echo "Coordinator log: ${logs_dir}/viewer_alpha96_coordinator_${coordinator_job_id}.out"
