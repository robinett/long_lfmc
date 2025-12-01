#!/usr/bin/env bash
set -euo pipefail

########################
# Constant directories
########################

input_data_dir="/scratch/users/trobinet/long_lfmc/\
trent_datasets/lfmc_model/data/inputs_sarstats"

save_root="/scratch/users/trobinet/long_lfmc/\
trent_datasets/lfmc_model/data/outputs/sarstatsnomonthlymeans"

########################
# Experiment definitions
# Each index = one experiment
########################

batch_sizes=(128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128)
lrs=(1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4)
val_splits=(0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2)
adam_wds=(1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4)

d_models=(32 64 128 256 32 64 128 256 32 64 128 256 32 64 128 256 32 64 128 256 32 64 128 256 32 64 128 256 32 64 128 256)
nheads=(1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8)
num_layers_list=(2 2 3 4 2 2 3 4 2 2 3 4 2 2 3 4 2 2 3 4 2 2 3 4 2 2 3 4 2 2 3 4)
dim_feedforwards=(64 128 256 512 64 128 256 512 64 128 256 512 64 128 256 512 64 128 256 512 64 128 256 512 64 128 256 512 64 128 256 512)
dropouts=(0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15 0.15)

long_d_models=(32 32 32 32 64 64 64 64 128 128 128 128 256 256 256 256 32 32 32 32 64 64 64 64 128 128 128 128 256 256 256 256)
long_nheads=(1 1 1 1 2 2 2 2 4 4 4 4 8 8 8 8 1 1 1 1 2 2 2 2 4 4 4 4 8 8 8 8)
long_num_layers_list=(2 2 2 2 2 2 2 2 3 3 3 3 4 4 4 4 2 2 2 2 2 2 2 2 3 3 3 3 4 4 4 4)
long_dim_feedforwards=(64 64 64 64 128 128 128 128 256 256 256 256 512 512 512 512 64 64 64 64 128 128 128 128 256 256 256 256 512 512 512 512)
long_out_dims=(32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64)

########################
# Sanity check: lengths
########################

num_exps=${#batch_sizes[@]}

check_len () {
  local name="$1"
  local len="$2"
  if [[ "${len}" -ne "${num_exps}" ]]; then
    echo "Error: array '${name}' length" \
         "(${len}) != num_exps (${num_exps})"
    exit 1
  fi
}

check_len "lrs"               "${#lrs[@]}"
check_len "val_splits"        "${#val_splits[@]}"
check_len "adam_wds"          "${#adam_wds[@]}"
check_len "d_models"          "${#d_models[@]}"
check_len "nheads"            "${#nheads[@]}"
check_len "num_layers_list"   "${#num_layers_list[@]}"
check_len "dim_feedforwards"  "${#dim_feedforwards[@]}"
check_len "dropouts"          "${#dropouts[@]}"
check_len "long_d_models"     "${#long_d_models[@]}"
check_len "long_nheads"       "${#long_nheads[@]}"
check_len "long_num_layers_list" \
           "${#long_num_layers_list[@]}"
check_len "long_dim_feedforwards" \
           "${#long_dim_feedforwards[@]}"
check_len "long_out_dims"     "${#long_out_dims[@]}"

echo "Submitting ${num_exps} experiments..."

########################
# Submission loop (zip)
########################

for ((i=0; i< num_exps; i++)); do
  batch_size=${batch_sizes[$i]}
  lr=${lrs[$i]}
  val_split=${val_splits[$i]}
  adam_wd=${adam_wds[$i]}

  d_model=${d_models[$i]}
  nhead=${nheads[$i]}
  num_layers=${num_layers_list[$i]}
  dim_feedforward=${dim_feedforwards[$i]}
  dropout=${dropouts[$i]}

  long_d_model=${long_d_models[$i]}
  long_nhead=${long_nheads[$i]}
  long_num_layers=${long_num_layers_list[$i]}
  long_dim_feedforward=${long_dim_feedforwards[$i]}
  long_out_dim=${long_out_dims[$i]}

  run_name="exp${i}_bs${batch_size}_lr${lr}_dm${d_model}_nh${nhead}_nl${num_layers}"

  echo "Submitting: ${run_name}"

  sbatch train_job.sbatch \
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
    --long_out_dim "${long_out_dim}"
done
