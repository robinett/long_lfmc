#!/usr/bin/env bash

trap 'echo "Caught Ctrl-C, exiting..."; exit 130' INT

set -euo pipefail

########################
# Constant directories
########################

input_data_dir="/scratch/users/trobinet/long_lfmc/\
trent_datasets/lfmc_model/data/inputs_base"

save_root="/scratch/users/trobinet/long_lfmc/\
trent_datasets/lfmc_model/data/outputs/base"

########################
# Job throttling config
########################

# Max number of jobs (PENDING + RUNNING)
# for this user, across the whole cluster.
max_jobs=8

# How often to re-check (seconds)
job_check_interval=30

########################
# Helper: count jobs
########################

get_job_count() {
  local user="${USER}"

  squeue -u "${user}" \
         -t PENDING,RUNNING \
         -h \
         -o "%b" \
    | awk '$0 !~ /\(null\)/ {c++} END {print c+0}'
}

wait_for_slot() {
  local limit="$1"

  while true; do
    local n
    n=$(get_job_count)
    echo "Current job count: ${n}" >&2
    if (( n < limit )); then
      break
    fi
    echo "Have ${n} jobs; waiting for a slot..." >&2
    sleep "${job_check_interval}"
  done
}

########################
# Hyperparameter grids
########################

# things that we won't specify a grid search for
num_tasks=1

# Usually you'll keep most of these small/singleton
batch_sizes=(128)
lrs=(5e-4 1e-4)
val_splits=(0.2)
adam_wds=(1e-4)
dropouts=(0.15)

# You only specify dimensions; everything else is derived
# SHORT encoder
d_models=(32 64 128)

# LONG encoder
long_d_models=(32 64 128 256)

# Still explicit: long_out_dim (no rule given)
long_out_dims=(32 64)

########################
# Submission loop (grid)
########################

exp_idx=0

for batch_size in "${batch_sizes[@]}"; do
  for lr in "${lrs[@]}"; do
    for val_split in "${val_splits[@]}"; do
      for adam_wd in "${adam_wds[@]}"; do
        for dropout in "${dropouts[@]}"; do

          # Loop over SHORT model dimensions
          for d_idx in "${!d_models[@]}"; do
            d_model=${d_models[$d_idx]}

            # Derived:
            #   nhead = d / 32
            #   num_layers = 2, 3, 4, ... per d_models index
            #   dim_ff = d * 2
            nhead=$(( d_model / 32 ))
            num_layers=$(( 2 + d_idx ))
            dim_feedforward=$(( d_model * 2 ))

            # Loop over LONG model dimensions
            for long_d_idx in "${!long_d_models[@]}"; do
              long_d_model=${long_d_models[$long_d_idx]}

              long_nhead=$(( long_d_model / 32 ))
              long_num_layers=$(( 2 + long_d_idx ))
              long_dim_feedforward=$(( long_d_model * 2 ))

              for long_out_dim in "${long_out_dims[@]}"; do

                run_name="exp${exp_idx}_bs${batch_size}_\
dm${d_model}_nh${nhead}_nl${num_layers}_\
ldm${long_d_model}_lnh${long_nhead}_\
lnl${long_num_layers}"

                echo "Submitting: ${run_name}"

                # Wait until we have fewer than max_jobs jobs
                wait_for_slot "${max_jobs}"

                sbatch \
                  --job-name="${run_name}" \
                  train_job.sbatch \
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
                  --long_dim_feedforward \
                    "${long_dim_feedforward}" \
                  --long_out_dim "${long_out_dim}" \
                  --num_gradnorm_tasks "${num_tasks}"

                exp_idx=$((exp_idx + 1))

                sleep 30

              done
            done
          done

        done
      done
    done
  done
done

echo "Submitted ${exp_idx} experiments."
