#!/usr/bin/env bash

set -euo pipefail

job_id=""
job_name=""
node_name=""
gpu_idx=""
duration_seconds=600
interval_seconds=30
output_dir="/home/users/trobinet/long_lfmc/logs/gpu_efficiency"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id)
            job_id="$2"
            shift 2
            ;;
        --job-name)
            job_name="$2"
            shift 2
            ;;
        --node-name)
            node_name="$2"
            shift 2
            ;;
        --gpu-idx)
            gpu_idx="$2"
            shift 2
            ;;
        --duration-seconds)
            duration_seconds="$2"
            shift 2
            ;;
        --interval-seconds)
            interval_seconds="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${job_id}" || -z "${job_name}" || -z "${node_name}" || -z "${gpu_idx}" ]]; then
    echo "Missing required arguments for ${BASH_SOURCE[0]}" >&2
    exit 1
fi

mkdir -p "${output_dir}"

state_prefix="${output_dir}/job_${job_id}"
raw_log="${state_prefix}_raw.csv"
log_file="${state_prefix}_utilization.csv"
summary_csv="${state_prefix}_summary.csv"
summary_txt="${state_prefix}_summary.txt"
meta_json="${state_prefix}_metadata.json"

start_epoch="$(date +%s)"
target_end_epoch="$((start_epoch + duration_seconds))"

get_slurm_field() {
    scontrol show job "${job_id}" | awk -v field="$1" '
    {
        for (i = 1; i <= NF; i++) {
            if ($i ~ "^" field "=") {
                split($i, a, "=");
                print a[2];
                exit;
            }
        }
    }'
}

job_user="$(get_slurm_field UserId | cut -d'(' -f1)"
job_group="$(get_slurm_field GroupId | cut -d'(' -f1)"
allocated_gpus="${SLURM_GPUS:-1}"
visible_gpu_indices="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)"
visible_gpu_indices="$(printf '%s\n' "${visible_gpu_indices}" | sed '/^\s*$/d')"
if [[ -z "${visible_gpu_indices}" ]]; then
    echo "Could not determine any visible GPUs for job_id=${job_id}" >&2
    exit 1
fi

if ! printf '%s\n' "${visible_gpu_indices}" | grep -Fxq "${gpu_idx}"; then
    gpu_idx="$(printf '%s\n' "${visible_gpu_indices}" | head -n 1 | xargs)"
fi

gpu_type="$(nvidia-smi --id="${gpu_idx}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '\r')"
if [[ -z "${gpu_type}" ]]; then
    echo "Could not query GPU name for visible gpu_idx=${gpu_idx} job_id=${job_id}" >&2
    exit 1
fi

echo "Profiling job_id=${job_id} job_name=${job_name} node=${node_name} gpu_idx=${gpu_idx}"
echo "duration_seconds=${duration_seconds} interval_seconds=${interval_seconds}"

: > "${raw_log}"

sample_count=0
while true; do
    now_epoch="$(date +%s)"
    if (( now_epoch >= target_end_epoch )); then
        break
    fi

    if ! nvidia-smi --id="${gpu_idx}" \
        --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used \
        --format=csv,noheader,nounits >> "${raw_log}" 2>/dev/null; then
        echo "nvidia-smi sampling failed for job_id=${job_id}; stopping early" >&2
        break
    fi

    sample_count="$((sample_count + 1))"

    remaining_seconds="$((target_end_epoch - $(date +%s)))"
    if (( remaining_seconds <= 0 )); then
        break
    fi
    sleep_seconds="${interval_seconds}"
    if (( remaining_seconds < interval_seconds )); then
        sleep_seconds="${remaining_seconds}"
    fi
    sleep "${sleep_seconds}"
done

end_epoch="$(date +%s)"
elapsed_seconds="$((end_epoch - start_epoch))"

python3 - <<'PY' "${raw_log}" "${log_file}" "${summary_csv}" "${summary_txt}" "${meta_json}" "${job_id}" "${job_name}" "${job_user}" "${job_group}" "${allocated_gpus}" "${gpu_type}" "${gpu_idx}" "${node_name}" "${elapsed_seconds}"
import csv
import json
import math
import os
import sys

(
    raw_log,
    log_file,
    summary_csv,
    summary_txt,
    meta_json,
    job_id,
    job_name,
    job_user,
    job_group,
    allocated_gpus,
    gpu_type,
    gpu_idx,
    node_name,
    elapsed_seconds,
) = sys.argv[1:]

elapsed_seconds = int(elapsed_seconds)
allocated_gpus = int(float(allocated_gpus))

rows = []
with open(raw_log, "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 5:
            continue
        index = row[0].strip()
        timestamp = row[1].strip()
        gpu_util = float(row[2].strip())
        mem_total = float(row[3].strip())
        mem_used = float(row[4].strip())
        mem_util = 0.0 if mem_total <= 0 else (mem_used / mem_total) * 100.0
        rows.append(
            {
                "index": index,
                "timestamp": timestamp,
                "gpu_util": gpu_util,
                "mem_total": mem_total,
                "mem_used": mem_used,
                "mem_util": mem_util,
            }
        )

if rows:
    avg_util = sum(r["gpu_util"] for r in rows) / len(rows)
    max_util = max(r["gpu_util"] for r in rows)
    avg_mem_util = sum(r["mem_util"] for r in rows) / len(rows)
    max_mem_util = max(r["mem_util"] for r in rows)
else:
    avg_util = 0.0
    max_util = 0.0
    avg_mem_util = 0.0
    max_mem_util = 0.0

gpu_alloc_hours = allocated_gpus * elapsed_seconds / 3600.0
gpu_used_hours = gpu_alloc_hours * avg_util / 100.0

hours = elapsed_seconds // 3600
minutes = (elapsed_seconds % 3600) // 60
seconds = elapsed_seconds % 60
walltime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "index",
            "job_id",
            "user",
            "group",
            "type.gpu",
            "timestamp",
            "utilization.gpu [%]",
            "memory.total [MiB]",
            "memory.used [MiB]",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r["index"],
                job_id,
                job_user,
                job_group,
                gpu_type,
                r["timestamp"],
                f"{r['gpu_util']:.0f}",
                f"{r['mem_total']:.0f}",
                f"{r['mem_used']:.0f}",
            ]
        )

with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "job_id",
            "job_name",
            "user",
            "group",
            "elapsed",
            "ngpus",
            "type.gpu",
            "hours.gpu",
            "avg_utilization.gpu",
            "max_utilization.gpu",
            "avg_utilization.gpu_mem",
            "max_utilization.gpu_mem",
            "hours_used.gpu",
            "node_name",
            "gpu_idx",
            "samples",
        ]
    )
    writer.writerow(
        [
            job_id,
            job_name,
            job_user,
            job_group,
            walltime,
            allocated_gpus,
            gpu_type,
            f"{gpu_alloc_hours:.4f}",
            f"{avg_util:.2f}",
            f"{max_util:.0f}",
            f"{avg_mem_util:.2f}",
            f"{max_mem_util:.0f}",
            f"{gpu_used_hours:.4f}",
            node_name,
            gpu_idx,
            len(rows),
        ]
    )

with open(summary_txt, "w") as f:
    f.write(f"Job ID: {job_id}\n")
    f.write(f"Job Name: {job_name}\n")
    f.write(f"User: {job_user}\n")
    f.write(f"Group: {job_group}\n")
    f.write(f"Elapsed Time: {walltime}\n")
    f.write(f"GPUs Allocated: {allocated_gpus}\n")
    f.write(f"GPU Type: {gpu_type}\n")
    f.write(f"GPU Allocation (GPU-hours): {gpu_alloc_hours:.4f}\n")
    f.write(f"Average GPU Utilization: {avg_util:.2f}%\n")
    f.write(f"Maximum GPU Utilization: {max_util:.0f}%\n")
    f.write(f"Average GPU Memory Utilization: {avg_mem_util:.2f}%\n")
    f.write(f"Maximum GPU Memory Utilization: {max_mem_util:.0f}%\n")
    f.write(f"Estimated Used GPU-hours: {gpu_used_hours:.4f}\n")
    f.write("\n")
    f.write("------ Per-GPU Breakdown ------\n")
    f.write(
        f"GPU {gpu_idx} - Avg Util: {avg_util:.2f}%, Max Util: {max_util:.0f}%, "
        f"Avg Mem: {avg_mem_util:.2f}%, Max Mem: {max_mem_util:.0f}%\n"
    )

with open(meta_json, "w") as f:
    json.dump(
        {
            "job_id": job_id,
            "job_name": job_name,
            "user": job_user,
            "group": job_group,
            "node_name": node_name,
            "gpu_idx": gpu_idx,
            "gpu_type": gpu_type,
            "elapsed_seconds": elapsed_seconds,
            "samples": len(rows),
            "hours_gpu": gpu_alloc_hours,
            "avg_utilization_gpu": avg_util,
            "max_utilization_gpu": max_util,
            "avg_utilization_gpu_mem": avg_mem_util,
            "max_utilization_gpu_mem": max_mem_util,
            "hours_used_gpu": gpu_used_hours,
        },
        f,
        indent=2,
        sort_keys=True,
    )
PY

rm -f "${raw_log}"

echo "Completed profiling job_id=${job_id}"
echo "summary_csv=${summary_csv}"
echo "summary_txt=${summary_txt}"
echo "utilization_csv=${log_file}"
