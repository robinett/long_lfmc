#!/bin/bash

set -euo pipefail

# Example aggregation command for a partially completed longweather ensemble.
# Incomplete members are skipped automatically.

ensemble_root='/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_ens_fullrandom'
out_dir="${ensemble_root}/analysis/aggregate_ensemble"

python3 -u aggregate_ensemble_cv.py \
    --member-glob "${ensemble_root}/transformer_*" \
    --out-dir "${out_dir}" \
    --split test \
    --tasks lfmc vh vv \
    --skip-incomplete-members
