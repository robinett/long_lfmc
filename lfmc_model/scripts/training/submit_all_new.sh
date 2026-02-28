#!/usr/bin/env bash

set -euo pipefail

trap 'echo "Caught Ctrl-C, exiting..."; exit 130' INT

########################
# Constant directories
########################

input_data_dir="/scratch/users/trobinet/long_lfmc/\
final_lfmc/lfmc_model/inputs/lfmc"

save_root="/scratch/users/trobinet/long_lfmc/\
final_lfmc/lfmc_model/outputs/lfmc"

########################
# Job throttling via lock files
########################

max_jobs=8
lock_dir="/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/gpu_locks"
mkdir -p "${lock_dir}"

########################
# Hyperparameter grids
########################

num_tasks=1
weighting_type='manual'
task_weights=(1.0 1.0 1.0)

batch_sizes=(128)
lrs=(5e-4 1e-4)
val_splits=(0.15)
adam_wds=(1e-4)
dropouts=(0.15)

d_models=(32 64 128)
long_d_models=(32 64 128 256)
long_out_dims=(32 64)

########################
# Submission loop (grid)
########################

exp_idx=0
exp_name=$(basename "$save_root")


for batch_size in "${batch_sizes[@]}"; do
  for lr in "${lrs[@]}"; do
    for val_split in "${val_splits[@]}"; do
      for adam_wd in "${adam_wds[@]}"; do
        for dropout in "${dropouts[@]}"; do
          for d_idx in "${!d_models[@]}"; do
            d_model=${d_models[$d_idx]}
            nhead=$(( d_model / 32 ))
            num_layers=$(( 2 + d_idx ))
            dim_feedforward=$(( d_model * 2 ))

            for long_d_idx in "${!long_d_models[@]}"; do
              long_d_model=${long_d_models[$long_d_idx]}
              long_nhead=$(( long_d_model / 32 ))
              long_num_layers=$(( 2 + long_d_idx ))
              long_dim_feedforward=$(( long_d_model * 2 ))

              for long_out_dim in "${long_out_dims[@]}"; do
                run_name="exp${exp_idx}_${exp_name}"

                # wait until # of files in lock dir < max_jobs
                while [ "$(find "${lock_dir}" -type f | wc -l)" -ge "${max_jobs}" ]; do
                  echo "Found $(find "${lock_dir}" -type f | wc -l) jobs in progress. Waiting."
                  sleep $(( 30 + RANDOM % 31 ))
                done


                # add a file if you made it through
                touch "${lock_dir}/lock_${run_name}.lock"

                echo "Submitting: ${run_name}"
                
                sbatch \
                  --export=ALL,LOCK_FILE="${lock_dir}/lock_${run_name}.lock" \
                  --job-name="${run_name}" \
                  "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/train_job_longweather.sbatch" \
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
                  --overwrite

                exp_idx=$((exp_idx + 1))

                sleep $(( 30 + RANDOM % 31 ))


              done
            done
          done
        done
      done
    done
  done
 done

echo "Submitted ${exp_idx} experiments."
