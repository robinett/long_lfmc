#!/usr/bin/env bash

set -euo pipefail

trap 'echo "Caught Ctrl-C, exiting..."; exit 130' INT

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
mkdir -p logs

########################
# Constant directories
########################

input_data_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_vh_vv"
save_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_vh_vv_ens"

########################
# Ensemble settings
########################

ensemble_size=32
base_seed=1000
split_seed=42

########################
# Job throttling via lock files
########################

max_jobs=8
lock_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/gpu_locks"
mkdir -p "${lock_dir}"

########################
# Single hyperparameter setting
# Edit these values after selecting the winning config from submit_all_new.sh
########################

num_tasks=3
weighting_type='manual'
task_weights=(3.0 1.0 1.0)

batch_size=128
lr=5e-4
val_split=0.15
adam_wd=1e-4
dropout=0.15

d_model=32
nhead=1
num_layers=2
dim_feedforward=64

long_d_model=64
long_nhead=2
long_num_layers=3
long_dim_feedforward=128
long_out_dim=16

########################
# Submission loop (seeds)
########################

exp_name="$(basename "${save_root}")"
submitted=0

for (( member_idx=0; member_idx<ensemble_size; member_idx++ )); do
  seed=$(( base_seed + member_idx ))
  run_tag=$(printf "seed%03d" "${member_idx}")
  run_name=$(printf "ens%02d_%s" "${member_idx}" "${exp_name}")

  while [ "$(find "${lock_dir}" -type f | wc -l)" -ge "${max_jobs}" ]; do
    echo "Found $(find "${lock_dir}" -type f | wc -l) jobs in progress. Waiting."
    sleep $(( 30 + RANDOM % 31 ))
  done

  touch "${lock_dir}/lock_${run_name}.lock"

  echo "Submitting ensemble member ${member_idx}/${ensemble_size} (seed=${seed}, run_tag=${run_tag})"

  sbatch \
    --export=ALL,LOCK_FILE="${lock_dir}/lock_${run_name}.lock" \
    --job-name="${run_name}" \
    train_job_longweather.sbatch \
    --input_data_dir "${input_data_dir}" \
    --save_dir "${save_root}" \
    --batch_size "${batch_size}" \
    --lr "${lr}" \
    --val_split "${val_split}" \
    --adam_wd "${adam_wd}" \
    --d_model "${d_model}" \
    --nhead "${nhead}" \
    --num_layers "${num_layers}" \
    --dim_feedforward "${dim_feedforward}" \
    --dropout "${dropout}" \
    --long_d_model "${long_d_model}" \
    --long_nhead "${long_nhead}" \
    --long_num_layers "${long_num_layers}" \
    --long_dim_feedforward "${long_dim_feedforward}" \
    --long_out_dim "${long_out_dim}" \
    --num_tasks "${num_tasks}" \
    --task_weight_type "${weighting_type}" \
    --manual_task_weights "${task_weights[@]}" \
    --seed "${seed}" \
    --batch_seed "${seed}" \
    --split_seed "${split_seed}" \
    --run_tag "${run_tag}" \
    --overwrite

  submitted=$(( submitted + 1 ))
  sleep $(( 30 + RANDOM % 31 ))
done

echo "Submitted ${submitted} ensemble members."

