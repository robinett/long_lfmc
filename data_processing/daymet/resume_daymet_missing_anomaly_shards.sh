#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

mkdir -p ./logs

config_path="${SCRIPT_DIR}/configs_clim20.yaml"
exclude_nodes="sh04-08n13"

if [[ $# -gt 0 ]]; then
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                config_path="$2"
                shift 2
                ;;
            --exclude-nodes)
                exclude_nodes="$2"
                shift 2
                ;;
            *)
                echo "Usage: $0 [--config PATH] [--exclude-nodes NODELIST]"
                exit 1
                ;;
        esac
    done
fi

run_script="${SCRIPT_DIR}/run_build_daymet_derived_products.sh"

source ~/uv_activations/activate_lfmc_process_py312.sh

mapfile -t missing_specs < <(
python3 - <<'PY' "${config_path}"
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

coord_dir = Path(cfg["paths"]["coord_dir"]) / "anomaly"
processing_cfg = cfg["processing"]
if "anomaly_variable_order" in processing_cfg:
    anomaly_vars = list(processing_cfg["anomaly_variable_order"])
elif "anomaly_variables" in processing_cfg:
    anomaly_vars = list(processing_cfg["anomaly_variables"])
else:
    raise KeyError("Expected processing.anomaly_variable_order or processing.anomaly_variables in config")
num_shards = int(cfg["processing"].get("num_shards", 24))

for var_name in anomaly_vars:
    present = {
        int(p.stem.split("__shard_")[1].split("_of_")[0])
        for p in coord_dir.glob(f"{var_name}__shard_*_of_{num_shards}.json")
    }
    for shard_index in range(num_shards):
        if shard_index not in present:
            print(f"{var_name} {shard_index} {num_shards}")
PY
)

if [[ ${#missing_specs[@]} -eq 0 ]]; then
    echo "No missing anomaly shards detected for ${config_path}"
    exit 0
fi

echo "Resubmitting ${#missing_specs[@]} missing anomaly shards for ${config_path}"
for spec in "${missing_specs[@]}"; do
    echo "  ${spec}"
done

submit_job() {
    local var_name="$1"
    local shard_index="$2"
    local num_shards="$3"
    local shard_label
    shard_label=$(printf "%02d" "${shard_index}")

    local cmd=(
        sbatch
        --parsable
        --chdir "$SCRIPT_DIR"
        --job-name="daymet_recover_${var_name}_s${shard_label}"
        --output="./logs/daymet_recover_${var_name}_s${shard_label}_%j.out"
        --error="./logs/daymet_recover_${var_name}_s${shard_label}_%j.err"
    )
    if [[ -n "${exclude_nodes}" ]]; then
        cmd+=(--exclude="${exclude_nodes}")
    fi
    cmd+=(
        "${run_script}"
        --config "${config_path}"
        --mode build-anomaly-var
        --var "${var_name}"
        --shard-index "${shard_index}"
        --num-shards "${num_shards}"
    )
    "${cmd[@]}"
}

recovery_job_ids=()
for spec in "${missing_specs[@]}"; do
    read -r var_name shard_index num_shards <<<"${spec}"
    job_id=$(submit_job "${var_name}" "${shard_index}" "${num_shards}")
    recovery_job_ids+=("${job_id}")
    echo "Submitted recovery shard ${var_name} shard ${shard_index}/${num_shards} as ${job_id}"
done

dep_string=$(IFS=:; echo "${recovery_job_ids[*]}")
finalize_job_id=$(
    sbatch \
        --parsable \
        --chdir "$SCRIPT_DIR" \
        --dependency=afterok:"${dep_string}" \
        --job-name=daymet_finalize_anomaly_recover \
        --output=./logs/daymet_finalize_anomaly_recover_%j.out \
        --error=./logs/daymet_finalize_anomaly_recover_%j.err \
        "${run_script}" \
        --config "${config_path}" \
        --mode finalize-anomaly
)

echo "Submitted finalize-anomaly recovery job: ${finalize_job_id}"
