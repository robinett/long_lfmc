#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857"
pipeline_submit="${script_dir}/submit_viewer_init_pipeline.sh"

cd "${script_dir}"

bash "${pipeline_submit}"
