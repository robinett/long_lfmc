#!/usr/bin/env bash

set -euo pipefail

trap 'echo "Caught Ctrl-C, exiting..."; exit 130' INT

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
mkdir -p logs

########################
# Constant directories
########################

input_source_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_365_fullspatial"
input_data_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs/lfmc_365_shared"
save_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc_365_shared_ensemble"
shared_fold_root="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/shared_training"
fold_info_path="${shared_fold_root}/canonical_fold_info_lfmc_365_shared.json"

mkdir -p "$(dirname "${input_data_dir}")" "${save_root}" "${shared_fold_root}"

if [[ ! -d "${input_source_dir}" ]]; then
  echo "Missing source tensor directory: ${input_source_dir}" >&2
  exit 1
fi

if [[ ! -e "${input_data_dir}" ]]; then
  ln -s "${input_source_dir}" "${input_data_dir}"
  echo "Created shared input alias: ${input_data_dir} -> ${input_source_dir}"
elif [[ -L "${input_data_dir}" ]]; then
  current_target="$(readlink -f "${input_data_dir}")"
  source_target="$(readlink -f "${input_source_dir}")"
  if [[ "${current_target}" != "${source_target}" ]]; then
    rm -f "${input_data_dir}"
    ln -s "${input_source_dir}" "${input_data_dir}"
    echo "Updated shared input alias: ${input_data_dir} -> ${input_source_dir}"
  fi
elif [[ ! -d "${input_data_dir}" ]]; then
  echo "Shared input path exists but is not a directory/symlink: ${input_data_dir}" >&2
  exit 1
fi

if [[ ! -f "${fold_info_path}" ]]; then
  echo "Missing shared canonical fold file: ${fold_info_path}" >&2
  echo "Run the multitask shared submission first so the shared folds are generated." >&2
  exit 1
fi

########################
# Ensemble settings
########################

ensemble_size=32
base_seed=1000
split_seed=42
submission_tag="dm64"

########################
# Job throttling via lock files
########################

max_jobs=8
lock_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/gpu_locks"
mkdir -p "${lock_dir}"

########################
# Hyperparameter setting
########################

num_tasks=1
weighting_type="manual"
task_weights=(1.0 1.0 1.0)

batch_size=128
lr=1e-4
val_split=0.15
adam_wd=1e-4
dropout=0.15

d_model=64
nhead=2
num_layers=3
dim_feedforward=128

long_d_model=128
long_nhead=4
long_num_layers=3
long_dim_feedforward=256
long_out_dim=64

########################
# Submission loop
########################

exp_name="$(basename "${save_root}")"
submitted=0

for (( member_idx=0; member_idx<ensemble_size; member_idx++ )); do
  seed=$(( base_seed + member_idx ))
  run_tag=$(printf "seed%03d" "${member_idx}")
  run_name=$(printf "ens%02d_%s_%s" "${member_idx}" "${exp_name}" "${submission_tag}")

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
    --fold_info_in "${fold_info_path}" \
    --run_tag "${run_tag}" \
    --overwrite

  submitted=$(( submitted + 1 ))
  sleep $(( 30 + RANDOM % 31 ))
done

echo "Submitted ${submitted} ensemble members."
