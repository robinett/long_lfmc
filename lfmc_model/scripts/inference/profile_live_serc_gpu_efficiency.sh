#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/inference"
worker_script="${script_dir}/gpu_efficiency_attach_worker.sh"
output_dir="/home/users/trobinet/long_lfmc/logs/gpu_efficiency"
max_jobs=4
duration_seconds=600
interval_seconds=30
aggregate_csv="${output_dir}/live_serc_gpu_efficiency_summary.csv"
aggregate_txt="${output_dir}/live_serc_gpu_efficiency_summary.txt"

mkdir -p "${output_dir}"

mapfile -t selected_jobs < <(
MAX_JOBS="${max_jobs}" python3 - <<'PY'
import os
import random
import re
import subprocess
import sys

user = os.environ["USER"]
max_jobs = int(os.environ["MAX_JOBS"])
out = subprocess.check_output(
    ["squeue", "-u", user, "-h", "-o", "%i|%j|%T"],
    universal_newlines=True,
)
jobs = []
for line in out.splitlines():
    parts = line.strip().split("|")
    if len(parts) != 3:
        continue
    job_id, job_name, state = parts
    if state != "RUNNING":
        continue
    if not job_name.startswith("map_gpu_serc_"):
        continue
    detail = subprocess.check_output(
        ["scontrol", "show", "job", "-d", job_id],
        universal_newlines=True,
    )
    node_match = re.search(r"\bNodeList=([^\s]+)", detail)
    gres_match = re.search(r"GRES=gpu:\d+\(IDX:([^)]+)\)", detail)
    if not node_match or not gres_match:
        continue
    jobs.append((job_id, job_name, node_match.group(1), gres_match.group(1)))

if not jobs:
    sys.exit(0)

random.shuffle(jobs)
for job_id, job_name, node_name, gpu_idx in jobs[:max_jobs]:
    print("|".join([job_id, job_name, node_name, gpu_idx]))
PY
)

selected_count="${#selected_jobs[@]}"
if (( selected_count == 0 )); then
    echo "No running map_gpu_serc_* jobs with identifiable GPU IDX were found."
    exit 0
fi

echo "Profiling ${selected_count} running serc GPU jobs"
echo "duration_seconds=${duration_seconds}"
echo "interval_seconds=${interval_seconds}"
echo "output_dir=${output_dir}"

launcher_pids=()
selected_job_ids=()
for entry in "${selected_jobs[@]}"; do
    IFS='|' read -r job_id job_name node_name gpu_idx <<< "${entry}"
    selected_job_ids+=("${job_id}")

    attach_log="${output_dir}/attach_${job_id}.out"
    attach_err="${output_dir}/attach_${job_id}.err"

    echo "Launching profiler for job_id=${job_id} job_name=${job_name} node=${node_name} gpu_idx=${gpu_idx}"

    srun \
        --jobid="${job_id}" \
        --overlap \
        --ntasks=1 \
        --nodes=1 \
        --cpus-per-task=1 \
        bash "${worker_script}" \
            --job-id "${job_id}" \
            --job-name "${job_name}" \
            --node-name "${node_name}" \
            --gpu-idx "${gpu_idx}" \
            --duration-seconds "${duration_seconds}" \
            --interval-seconds "${interval_seconds}" \
            --output-dir "${output_dir}" \
        > "${attach_log}" \
        2> "${attach_err}" &

    launcher_pids+=("$!")
done

failed_launches=0
for pid in "${launcher_pids[@]}"; do
    if ! wait "${pid}"; then
        failed_launches="$((failed_launches + 1))"
    fi
done

echo ""
echo "====== Live SERC GPU Efficiency Summary ======"
successful_summary_csvs=()
for job_id in "${selected_job_ids[@]}"; do
    summary_csv="${output_dir}/job_${job_id}_summary.csv"
    if [[ -f "${summary_csv}" ]]; then
        successful_summary_csvs+=("${summary_csv}")
        tail -n +2 "${summary_csv}"
    else
        echo "job_id=${job_id},status=missing_summary"
    fi
done
echo "=============================================="

echo "failed_launches=${failed_launches}"
echo "logs_dir=${output_dir}"

python3 - <<'PY' "${aggregate_csv}" "${aggregate_txt}" "${failed_launches}" "${selected_count}" "${duration_seconds}" "${interval_seconds}" "${output_dir}" "${selected_job_ids[@]}" -- "${successful_summary_csvs[@]}"
import csv
import sys
from pathlib import Path

aggregate_csv = Path(sys.argv[1])
aggregate_txt = Path(sys.argv[2])
failed_launches = int(sys.argv[3])
selected_count = int(sys.argv[4])
duration_seconds = int(sys.argv[5])
interval_seconds = int(sys.argv[6])
output_dir = sys.argv[7]

args = sys.argv[8:]
sep_index = args.index("--")
selected_job_ids = args[:sep_index]
summary_paths = [Path(p) for p in args[sep_index + 1 :]]

rows = []
header = None
for path in summary_paths:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        current_header = next(reader)
        if header is None:
            header = current_header
        for row in reader:
            rows.append(row)

if header is None:
    header = [
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

with aggregate_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

avg_utils = [float(r[8]) for r in rows] if rows else []
max_utils = [float(r[9]) for r in rows] if rows else []
avg_mem_utils = [float(r[10]) for r in rows] if rows else []
max_mem_utils = [float(r[11]) for r in rows] if rows else []
used_gpu_hours = [float(r[12]) for r in rows] if rows else []

rollup_lines = [
    "Live SERC GPU Efficiency Aggregate",
    f"selected_jobs={selected_count}",
    f"successful_profiles={len(rows)}",
    f"failed_launches={failed_launches}",
    f"duration_seconds={duration_seconds}",
    f"interval_seconds={interval_seconds}",
    f"output_dir={output_dir}",
    f"sampled_job_ids={','.join(selected_job_ids)}",
]

if rows:
    rollup_lines.extend(
        [
            f"mean_avg_utilization_gpu={sum(avg_utils) / len(avg_utils):.2f}",
            f"max_utilization_gpu={max(max_utils):.0f}",
            f"mean_avg_utilization_gpu_mem={sum(avg_mem_utils) / len(avg_mem_utils):.2f}",
            f"max_utilization_gpu_mem={max(max_mem_utils):.0f}",
            f"total_hours_used_gpu={sum(used_gpu_hours):.4f}",
            "",
            "Per-job rows:",
            ",".join(header),
        ]
    )
    rollup_lines.extend(",".join(r) for r in rows)
else:
    rollup_lines.extend(["", "No successful profile summaries were written."])

aggregate_txt.write_text("\n".join(rollup_lines) + "\n")
PY
