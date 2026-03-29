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
submission_tag="dm128_vpd_anoms_nozone"

multitask_sar_root="${scratch_root}/sar/ensemble/lfmc_vh_vv_365_${submission_tag}"
multitask_sample_index_root="${scratch_root}/lfmc_model/indexes/ensemble/lfmc_vh_vv_365_${submission_tag}"
multitask_input_root="${scratch_root}/lfmc_model/inputs/ensemble/lfmc_vh_vv_365_${submission_tag}"
multitask_save_root="${scratch_root}/lfmc_model/outputs/lfmc_vh_vv_365_${submission_tag}"

single_sample_index_path="${scratch_root}/lfmc_model/indexes/sample_index_longweather_2000_2024_lfmc_${submission_tag}.parquet"
single_input_dir="${scratch_root}/lfmc_model/inputs/lfmc_365_${submission_tag}"
single_save_root="${scratch_root}/lfmc_model/outputs/lfmc_365_${submission_tag}"

shared_fold_root="${scratch_root}/lfmc_model/outputs/shared_training"
fold_info_path="${shared_fold_root}/canonical_fold_info_${submission_tag}.json"

mkdir -p \
  "${multitask_sar_root}" \
  "${multitask_sample_index_root}" \
  "${multitask_input_root}" \
  "${multitask_save_root}" \
  "$(dirname "${single_sample_index_path}")" \
  "$(dirname "${single_input_dir}")" \
  "${single_save_root}" \
  "${shared_fold_root}"

########################
# Ensemble settings
########################

ensemble_size=32
base_data_seed=1000
base_model_seed=1000
split_seed=42
sample_vars=(vv vh)
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

multitask_num_tasks=3
multitask_weighting_type="manual"
multitask_task_weights=(3.0 1.0 1.0)

single_num_tasks=1
single_weighting_type="manual"
single_task_weights=(1.0 1.0 1.0)

########################
# Helpers
########################

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

validate_input_dir() {
  local input_dir="$1"
  local missing=0
  local required_file=""

  if [[ ! -d "${input_dir}" ]]; then
    echo "Tensor directory missing: ${input_dir}" >&2
    return 1
  fi

  for required_file in "${required_tensor_files[@]}"; do
    if [[ ! -f "${input_dir}/${required_file}" ]]; then
      echo "Missing ${required_file} in tensor directory: ${input_dir}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    return 1
  fi
}

submit_training_job() {
  local run_name="$1"
  local run_tag="$2"
  local input_dir="$3"
  local save_dir="$4"
  local seed="$5"
  local num_tasks="$6"
  local weighting_type="$7"
  shift 7
  local task_weights=("$@")
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
      --input_data_dir "${input_dir}" \
      --save_dir "${save_dir}" \
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
  )"; then
    rm -f "${lock_file}"
    return 1
  fi

  echo "  Training job submitted: ${train_job_id} (${run_name})"
}

########################
# Submit preprocessing
########################

rm -f "${fold_info_path}"
echo "Removed any existing canonical fold file: ${fold_info_path}"

declare -a multitask_tensor_job_ids
declare -a multitask_input_dirs
declare -a multitask_model_seeds
declare -a multitask_run_tags
declare -a multitask_run_names
declare -a multitask_training_submitted

declare -a single_run_names
declare -a single_run_tags
declare -a single_model_seeds
declare -a single_training_submitted

multitask_train_submitted=0
single_train_submitted=0
fold_job_id=""
fold_input_dir=""

echo "Submitting multitask end-to-end preprocessing."
for (( member_idx=0; member_idx<ensemble_size; member_idx++ )); do
  data_seed=$(( base_data_seed + member_idx ))
  model_seed=$(( base_model_seed + member_idx ))
  data_tag=$(printf "ds%04d" "${data_seed}")
  run_tag=$(printf "multi_%s_ms%04d" "${data_tag}" "${model_seed}")
  run_name=$(printf "multi_ens%02d_%s" "${member_idx}" "${submission_tag}")

  member_sar_dir="${multitask_sar_root}/${data_tag}"
  member_sample_index="${multitask_sample_index_root}/sample_index_longweather_2000_2024_lfmc_vh_vv_${data_tag}.parquet"
  member_input_dir="${multitask_input_root}/lfmc_vh_vv_${data_tag}"
  mkdir -p "${member_sar_dir}"

  select_job_id="$(
    sbatch \
      --parsable \
      --job-name="sar_${run_name}" \
      "${repo_root}/data_processing/sar/sbatch_select_sar_sample.sh" \
      --sample-at-sites \
      --sample-at-random \
      --random-seed "${data_seed}" \
      --vars-to-sample "${sample_vars[@]}" \
      --output-dir "${member_sar_dir}" \
      --output-tag "${data_tag}"
  )"
  echo "  SAR selection job for member ${member_idx}: ${select_job_id}"

  index_job_id="$(
    sbatch \
      --parsable \
      --dependency="afterok:${select_job_id}" \
      --job-name="idx_${run_name}" \
      "${repo_root}/lfmc_model/scripts/data/build_sample_index_longweather.sbatch" \
      --out-path "${member_sample_index}" \
      --target-cols lfmc vv vh \
      --random-seed "${data_seed}" \
      --label-source "nfmd=${scratch_root}/nfmd/nfmd_processed.csv" \
      --label-source "vv_at_sites=${member_sar_dir}/vv_samples_at_sites_matching_${data_tag}.csv" \
      --label-source "vv_at_random=${member_sar_dir}/vv_samples_random_matching_${data_tag}.csv" \
      --label-source "vh_at_sites=${member_sar_dir}/vh_samples_at_sites_matching_${data_tag}.csv" \
      --label-source "vh_at_random=${member_sar_dir}/vh_samples_random_matching_${data_tag}.csv" \
      --target-sample-n "lfmc=-1" \
      --target-sample-n "vv=-1" \
      --target-sample-n "vh=-1"
  )"
  echo "  Sample-index job for member ${member_idx}: ${index_job_id}"

  tensor_job_id="$(
    sbatch \
      --parsable \
      --dependency="afterok:${index_job_id}" \
      --job-name="tensor_${run_name}" \
      "${repo_root}/lfmc_model/scripts/data/build_dataset_longweather_direct_single.sbatch" \
      "${member_sample_index}" \
      "${member_input_dir}" \
      --overwrite
  )"
  echo "  Tensor-build job for member ${member_idx}: ${tensor_job_id}"

  if [[ -z "${fold_job_id}" ]]; then
    fold_job_id="$(
      sbatch \
        --parsable \
        --dependency="afterok:${tensor_job_id}" \
        --job-name="fold_${submission_tag}" \
        "${repo_root}/lfmc_model/scripts/training/generate_longweather_fold_info.sbatch" \
        --input-data-dir "${member_input_dir}" \
        --out-path "${fold_info_path}" \
        --split-seed "${split_seed}"
    )"
    fold_input_dir="${member_input_dir}"
    echo "  Canonical fold-info job: ${fold_job_id}"
  fi

  multitask_tensor_job_ids[$member_idx]="${tensor_job_id}"
  multitask_input_dirs[$member_idx]="${member_input_dir}"
  multitask_model_seeds[$member_idx]="${model_seed}"
  multitask_run_tags[$member_idx]="${run_tag}"
  multitask_run_names[$member_idx]="${run_name}"
  multitask_training_submitted[$member_idx]=0

  single_model_seeds[$member_idx]="${model_seed}"
  single_run_tags[$member_idx]=$(printf "single_ms%04d" "${model_seed}")
  single_run_names[$member_idx]=$(printf "single_ens%02d_%s" "${member_idx}" "${submission_tag}")
  single_training_submitted[$member_idx]=0
done

echo "Submitting single-task sample-index build."
single_index_job_id="$(
  sbatch \
    --parsable \
    --job-name="idx_single_${submission_tag}" \
    "${repo_root}/lfmc_model/scripts/data/build_sample_index_longweather.sbatch" \
    --out-path "${single_sample_index_path}" \
    --target-cols lfmc \
    --label-source "nfmd=${scratch_root}/nfmd/nfmd_processed.csv"
)"
echo "  Single-task sample-index job: ${single_index_job_id}"

echo "Submitting single-task tensor build."
single_tensor_job_id="$(
  sbatch \
    --parsable \
    --dependency="afterok:${single_index_job_id}" \
    --job-name="tensor_single_${submission_tag}" \
    "${repo_root}/lfmc_model/scripts/data/build_dataset_longweather_direct_single.sbatch" \
    "${single_sample_index_path}" \
    "${single_input_dir}" \
    --overwrite
)"
echo "  Single-task tensor-build job: ${single_tensor_job_id}"

echo "Waiting for canonical folds and tensor builds before training submission."

while [[ "${multitask_train_submitted}" -lt "${ensemble_size}" || "${single_train_submitted}" -lt "${ensemble_size}" ]]; do
  fold_state="$(job_state "${fold_job_id}")"
  if job_failed "${fold_state}"; then
    echo "Canonical fold job ${fold_job_id} failed with state=${fold_state}." >&2
    exit 1
  fi

  single_tensor_state="$(job_state "${single_tensor_job_id}")"
  if job_failed "${single_tensor_state}"; then
    echo "Single-task tensor build ${single_tensor_job_id} failed with state=${single_tensor_state}." >&2
    exit 1
  fi

  echo "Fold job ${fold_job_id} state=${fold_state}; single tensor state=${single_tensor_state}; multitask submitted ${multitask_train_submitted}/${ensemble_size}; single submitted ${single_train_submitted}/${ensemble_size}."

  progress_this_round=0

  for (( member_idx=0; member_idx<ensemble_size; member_idx++ )); do
    if [[ "${multitask_training_submitted[$member_idx]}" == "1" ]]; then
      continue
    fi

    tensor_job_id="${multitask_tensor_job_ids[$member_idx]}"
    tensor_state="$(job_state "${tensor_job_id}")"
    if job_failed "${tensor_state}"; then
      echo "Multitask tensor build ${tensor_job_id} failed with state=${tensor_state}." >&2
      exit 1
    fi

    if [[ "${fold_state}" == "COMPLETED" && "${tensor_state}" == "COMPLETED" ]]; then
      validate_input_dir "${multitask_input_dirs[$member_idx]}"
      submit_training_job \
        "${multitask_run_names[$member_idx]}" \
        "${multitask_run_tags[$member_idx]}" \
        "${multitask_input_dirs[$member_idx]}" \
        "${multitask_save_root}" \
        "${multitask_model_seeds[$member_idx]}" \
        "${multitask_num_tasks}" \
        "${multitask_weighting_type}" \
        "${multitask_task_weights[@]}"
      multitask_training_submitted[$member_idx]=1
      multitask_train_submitted=$(( multitask_train_submitted + 1 ))
      progress_this_round=1
      sleep $(( 30 + RANDOM % 31 ))
    fi
  done

  if [[ "${fold_state}" == "COMPLETED" && "${single_tensor_state}" == "COMPLETED" ]]; then
    validate_input_dir "${single_input_dir}"
    for (( member_idx=0; member_idx<ensemble_size; member_idx++ )); do
      if [[ "${single_training_submitted[$member_idx]}" == "1" ]]; then
        continue
      fi
      submit_training_job \
        "${single_run_names[$member_idx]}" \
        "${single_run_tags[$member_idx]}" \
        "${single_input_dir}" \
        "${single_save_root}" \
        "${single_model_seeds[$member_idx]}" \
        "${single_num_tasks}" \
        "${single_weighting_type}" \
        "${single_task_weights[@]}"
      single_training_submitted[$member_idx]=1
      single_train_submitted=$(( single_train_submitted + 1 ))
      progress_this_round=1
      sleep $(( 30 + RANDOM % 31 ))
    done
  fi

  if [[ "${multitask_train_submitted}" -lt "${ensemble_size}" || "${single_train_submitted}" -lt "${ensemble_size}" ]]; then
    if [[ "${progress_this_round}" -eq 0 ]]; then
      echo "No new training submissions this round. Sleeping for ${poll_seconds}s."
      sleep "${poll_seconds}"
    fi
  fi
done

echo "Submitted all ${ensemble_size} multitask and ${ensemble_size} single-task dm128 training jobs."
