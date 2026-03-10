#!/usr/bin/env python3

import argparse
import json
import os

import pandas as pd

from map_runtime_utils import (
    DEFAULT_MODEL_TYPE,
    aggregate_member_predictions,
    build_reference_tensor_payload,
    densify_tile_predictions,
    get_inference_datasets,
    load_ensemble_runtimes,
    load_tile_payload,
    save_tile_shard,
    run_runtime_forward,
)
from compare_timeseries import _convert_tensor_payload_norm, _runtimes_share_feature_layout


def get_args():
    parser = argparse.ArgumentParser(
        description="Run one ensemble wall-to-wall map task from a manifest row."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE)
    return parser.parse_args()


def main():
    args = get_args()
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if task_id < 0:
        raise ValueError("task_id must be provided or available via SLURM_ARRAY_TASK_ID")

    manifest_df = pd.read_csv(args.manifest_path)
    manifest_df = manifest_df.sort_values("task_id").reset_index(drop=True)
    row_mask = manifest_df["task_id"].astype(int) == int(task_id)
    if not row_mask.any():
        raise KeyError(f"task_id={task_id} not found in manifest {args.manifest_path}")
    task_row = manifest_df.loc[row_mask].iloc[0]
    run_dir = os.path.dirname(args.manifest_path)
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)

    print(
        f"[run_map_task] task_id={task_id}, tile={task_row['tile_name']}, "
        f"window={task_row['start_date']} to {task_row['end_date']}"
    )
    member_dirs, runtimes = load_ensemble_runtimes(
        ensemble_root=run_config["ensemble_root"],
        input_data_name=run_config["input_data_name"],
    )
    if len(member_dirs) != len(run_config["member_dirs"]):
        raise ValueError(
            f"Run config member count {len(run_config['member_dirs'])} does not match "
            f"resolved member count {len(member_dirs)}"
        )
    dss = get_inference_datasets()
    tile_payload = load_tile_payload(task_row["tile_meta_path"])
    block_start = pd.Timestamp(task_row["start_date"]).normalize()
    block_end = pd.Timestamp(task_row["end_date"]).normalize()
    reference_runtime = runtimes[0]
    reference_payload = build_reference_tensor_payload(
        tile_payload=tile_payload,
        runtime=reference_runtime,
        dss=dss,
        start_date=block_start,
        end_date=block_end,
    )
    print(
        f"[run_map_task] built reference tensors for tile {task_row['tile_name']} "
        f"with {len(tile_payload['iy'])} pixels"
    )

    renorm_cache = {}
    member_dfs = []
    for member_idx, runtime in enumerate(runtimes, start=1):
        if member_idx == 1:
            tensor_payload = reference_payload
        elif _runtimes_share_feature_layout(reference_runtime, runtime):
            tensor_payload = _convert_tensor_payload_norm(
                reference_payload,
                reference_runtime,
                runtime,
                renorm_cache,
                site=f"tile_{task_row['tile_name']}_{task_row['start_date']}",
            )
        else:
            print(
                f"[run_map_task] member {member_idx}/{len(runtimes)} requires tensor rebuild "
                "because feature layout differs"
            )
            tensor_payload = build_reference_tensor_payload(
                tile_payload=tile_payload,
                runtime=runtime,
                dss=dss,
                start_date=block_start,
                end_date=block_end,
            )
        preds_df = run_runtime_forward(
            runtime=runtime,
            tensor_payload=tensor_payload,
            model_type=args.model_type,
        )
        member_dfs.append(preds_df)
        print(f"[run_map_task] completed member {member_idx}/{len(runtimes)}")

    agg_df = aggregate_member_predictions(member_dfs)
    dense_payload = densify_tile_predictions(agg_df, tile_payload)
    save_tile_shard(
        shard_path=task_row["shard_path"],
        dense_payload=dense_payload,
        tile_payload=tile_payload,
        task_row=task_row,
    )
    print(f"[run_map_task] wrote shard {task_row['shard_path']}")


if __name__ == "__main__":
    main()
