#!/usr/bin/env bash
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --output=/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs/submit_train_with_lock_%j.out
#SBATCH --error=/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs/submit_train_with_lock_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trobinet@stanford.edu

set -euo pipefail

cd "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training"
mkdir -p "/home/users/trobinet/long_lfmc/lfmc_model/scripts/training/logs"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <lock_dir> <max_jobs> <run_name> <train_args...>" >&2
    exit 2
fi

lock_dir="$1"
max_jobs="$2"
run_name="$3"
shift 3

mkdir -p "${lock_dir}"
lock_file="${lock_dir}/lock_${run_name}.lock"
submitted_job_id=""

cleanup_on_failure() {
    if [[ -z "${submitted_job_id}" ]]; then
        rm -f "${lock_file}"
    fi
}
trap cleanup_on_failure EXIT INT TERM

while [ "$(find "${lock_dir}" -type f | wc -l)" -ge "${max_jobs}" ]; do
    echo "Found $(find "${lock_dir}" -type f | wc -l) jobs in progress. Waiting."
    sleep $(( 30 + RANDOM % 31 ))
done

touch "${lock_file}"

submitted_job_id="$(
    sbatch \
        --parsable \
        --export=ALL,LOCK_FILE="${lock_file}" \
        --job-name="${run_name}" \
        /home/users/trobinet/long_lfmc/lfmc_model/scripts/training/train_job_longweather.sbatch \
        "$@"
)"

echo "Submitted GPU training job ${submitted_job_id} for ${run_name}"
trap - EXIT INT TERM
