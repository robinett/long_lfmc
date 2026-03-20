#!/usr/bin/env bash

set -euo pipefail

trap 'echo "Caught Ctrl-C, exiting..."; exit 130' INT

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${script_dir}"
mkdir -p logs

########################
# Constant directories
########################

scratch_root="/scratch/users/trobinet/long_lfmc/final_lfmc"
sar_root="${scratch_root}/sar/ensemble/lfmc_vh_vv_365_shared_ensemble"
sample_index_root="${scratch_root}/lfmc_model/indexes/ensemble/lfmc_vh_vv_365_shared_ensemble"
input_root="${scratch_root}/lfmc_model/inputs/ensemble/lfmc_vh_vv_365_shared_ensemble"
save_root="${scratch_root}/lfmc_model/outputs/lfmc_vh_vv_365_shared_ensemble"
shared_fold_root="${scratch_root}/lfmc_model/outputs/shared_training"
fold_info_path="${shared_fold_root}/canonical_fold_info_lfmc_365_shared.json"

mkdir -p "${sar_root}" "${sample_index_root}" "${input_root}" "${save_root}" "${shared_fold_root}"

########################
# Ensemble settings
########################

ensemble_size=32
base_data_seed=1000
base_model_seed=1000
split_seed=42
submission_tag="dm128"

resume_from_tensors=true
required_tensor_files=(X_short.pt X_long.pt X_static.pt Y.pt source.pt stratifier.npy info.csv)

########################
# Job throttling via lock files
########################

max_jobs=8
lock_dir="${scratch_root}/lfmc_model/gpu_locks"
poll_seconds=60
mkdir -p "${lock_dir}"

########################
# Hyperparameter setting
########################

num_tasks=3
weighting_type="manual"
task_weights=(3.0 1.0 1.0)

batch_size=128
lr=1e-4
val_split=0.15
adam_wd=1e-4
dropout=0.15

d_model=128
nhead=4
num_layers=3
dim_feedforward=256

long_d_model=256
long_nhead=8
long_num_layers=3
long_dim_feedforward=512
long_out_dim=128

########################
# Submission loop
########################

exp_name="$(basename "${save_root}")"
fold_job_id=""
fold_input_dir=""
preprocess_submitted=0
train_submitted=0

declare -a tensor_job_ids
declare -a input_dirs
declare -a model_seeds
declare -a run_tags
declare -a run_names
declare -a training_submitted_flags

job_state() {
  local job_id="$1"
  local state=""
  state="$(
    sacct -j "${job_id}" --format=State -n -P 2>/dev/null \
      | awk -F'|' 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1; exit}'
  )"
  if [[ -n "${state}" ]]; then
    printf '%s\n' "${state}"
    return 0
  fi
  state="$(
    squeue -h -j "${job_id}" -o "%T" 2>/dev/null \
      | awk 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1; exit}'
  )"
  if [[ -n "${state}" ]]; then
    printf '%s\n' "${state}"
    return 0
  fi
  printf 'UNKNOWN\n'
}

job_failed() {
  case "$1" in
    BOOT_FAIL|CANCELLED|CANCELLED+|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|TIMEOUT)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

validate_prebuilt_input_dir() {
  local member_idx="$1"
  local member_input_dir="$2"
  local missing=0
  local required_file=""

  if [[ ! -d "${member_input_dir}" ]]; then
    echo "Prebuilt tensor directory missing for member ${member_idx}: ${member_input_dir}" >&2
    return 1
  fi

  for required_file in "${required_tensor_files[@]}"; do
    if [[ ! -f "${member_input_dir}/${required_file}" ]]; then
      echo "Missing ${required_file} in prebuilt tensor directory for member ${member_idx}: ${member_input_dir}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    return 1
  fi
}

submit_training_job() {
  local member_idx="$1"
  local run_name="${run_names[$member_idx]}"
  local run_tag="${run_tags[$member_idx]}"
  local model_seed="${model_seeds[$member_idx]}"
  local member_input_dir="${input_dirs[$member_idx]}"
  local lock_file="${lock_dir}/lock_${run_name}.lock"
  local train_job_id=""

  while [ "$(find "${lock_dir}" -type f | wc -l)" -ge "${max_jobs}" ]; do
    echo "Found $(find "${lock_dir}" -type f | wc -l) jobs in progress. Waiting for a GPU slot before submitting ${run_name}."
    sleep $(( 30 + RANDOM % 31 ))
  done

  touch "${lock_file}"
  if ! train_job_id="$(
    sbatch \
      --parsable \
      --export=ALL,LOCK_FILE="${lock_file}" \
      --job-name="${run_name}" \
      train_job_longweather.sbatch \
      --input_data_dir "${member_input_dir}" \
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
      --seed "${model_seed}" \
      --batch_seed "${model_seed}" \
      --split_seed "${split_seed}" \
      --fold_info_in "${fold_info_path}" \
      --run_tag "${run_tag}" \
      --overwrite
  )"; then
    rm -f "${lock_file}"
    return 1
  fi

  training_submitted_flags[$member_idx]=1
  train_submitted=$(( train_submitted + 1 ))
  echo "  Training job submitted for member ${member_idx}: ${train_job_id} (${run_name})"
  return 0
}

echo "resume_from_tensors=${resume_from_tensors}"

for (( member_idx=0; member_idx<ensemble_size; member_idx++ )); do
  data_seed=$(( base_data_seed + member_idx ))
  model_seed=$(( base_model_seed + member_idx ))
  data_tag=$(printf "ds%04d" "${data_seed}")
  run_tag=$(printf "ds%04d_ms%04d" "${data_seed}" "${model_seed}")
  run_name=$(printf "ens%02d_%s_%s" "${member_idx}" "${exp_name}" "${submission_tag}")
  member_input_dir="${input_root}/lfmc_vh_vv_${data_tag}"

  input_dirs[$member_idx]="${member_input_dir}"
  model_seeds[$member_idx]="${model_seed}"
  run_tags[$member_idx]="${run_tag}"
  run_names[$member_idx]="${run_name}"
  training_submitted_flags[$member_idx]=0

  echo "Registering prebuilt tensors for member ${member_idx}/${ensemble_size} (data_seed=${data_seed}, model_seed=${model_seed})"
  validate_prebuilt_input_dir "${member_idx}" "${member_input_dir}"
  tensor_job_ids[$member_idx]="PREBUILT"
  if [[ -z "${fold_input_dir}" ]]; then
    fold_input_dir="${member_input_dir}"
  fi
  preprocess_submitted=$(( preprocess_submitted + 1 ))
done

if [[ -z "${fold_input_dir}" ]]; then
  echo "No prebuilt tensor directories were registered; cannot regenerate canonical fold info." >&2
  exit 1
fi

rm -f "${fold_info_path}"
fold_job_id="$(
  sbatch \
    --parsable \
    --job-name="fold_lfmc_365_shared" \
    "${repo_root}/lfmc_model/scripts/training/generate_longweather_fold_info.sbatch" \
    --input-data-dir "${fold_input_dir}" \
    --out-path "${fold_info_path}" \
    --split-seed "${split_seed}"
)"
echo "Canonical fold-info regeneration job: ${fold_job_id}"
echo "Registered ${preprocess_submitted} prebuilt tensor directories."

echo "Canonical fold info will be written to ${fold_info_path}"
echo "Waiting for canonical folds before submitting GPU training jobs."

while [[ "${train_submitted}" -lt "${ensemble_size}" ]]; do
  fold_state="$(job_state "${fold_job_id}")"
  if job_failed "${fold_state}"; then
    echo "Canonical fold job ${fold_job_id} failed with state=${fold_state}."
    exit 1
  fi
  echo "Fold job ${fold_job_id} state=${fold_state}. Training submitted ${train_submitted}/${ensemble_size}."

  progress_this_round=0
  for (( member_idx=0; member_idx<ensemble_size; member_idx++ )); do
    if [[ "${training_submitted_flags[$member_idx]}" == "1" ]]; then
      continue
    fi

    if [[ "${fold_state}" == "COMPLETED" ]]; then
      echo "Member ${member_idx} is ready for training submission."
      submit_training_job "${member_idx}"
      progress_this_round=1
    fi
  done

  if [[ "${train_submitted}" -lt "${ensemble_size}" && "${progress_this_round}" -eq 0 ]]; then
    echo "No new training submissions this round. Sleeping for ${poll_seconds}s."
    sleep "${poll_seconds}"
  fi
done

echo "Submitted all ${train_submitted} GPU training jobs."
