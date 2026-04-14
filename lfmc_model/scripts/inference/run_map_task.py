#!/usr/bin/env python3

import argparse
import json
import os
import time

import pandas as pd

from map_runtime_utils import (
    DEFAULT_MODEL_TYPE,
    aggregate_member_predictions,
    build_reference_tensor_payload,
    build_reference_raw_tensor_cache,
    build_static_superset_runtime,
    convert_tensor_payload_to_runtime_bounded,
    densify_tile_predictions,
    get_inference_datasets,
    load_ensemble_runtimes,
    load_tile_payload,
    save_tile_shard,
    run_runtime_forward,
    timestamped_message,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run one ensemble wall-to-wall map Slurm job from grouped manifest rows."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    return parser.parse_args()


def _format_seconds(seconds):
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def process_task_row(
    task_row,
    runtimes,
    dss,
    model_type,
    forward_batch_size,
    use_cuda_autocast,
    progress_label,
):
    task_t0 = time.perf_counter()
    shard_path = str(task_row["shard_path"])
    if os.path.exists(shard_path):
        print(timestamped_message(
            f"[run_map_task] {progress_label} shard already exists; skipping {shard_path}"
        ))
        return 0.0
    print(timestamped_message(
        f"[run_map_task] {progress_label} fine_task_id={int(task_row['task_id'])}, tile={task_row['tile_name']}, "
        f"window={task_row['start_date']} to {task_row['end_date']}"
    ))
    tile_payload = load_tile_payload(task_row["tile_meta_path"])
    block_start = pd.Timestamp(task_row["start_date"]).normalize()
    block_end = pd.Timestamp(task_row["end_date"]).normalize()
    reference_runtime = build_static_superset_runtime(runtimes[0], runtimes)
    reference_payload = build_reference_tensor_payload(
        tile_payload=tile_payload,
        runtime=reference_runtime,
        dss=dss,
        start_date=block_start,
        end_date=block_end,
    )
    print(timestamped_message(
        f"[run_map_task] built reference tensors for tile {task_row['tile_name']} "
        f"with {len(tile_payload['iy'])} pixels"
    ))

    raw_tensor_cache = build_reference_raw_tensor_cache(
        reference_payload,
        reference_runtime,
    )
    member_dfs = []
    for member_idx, runtime in enumerate(runtimes, start=1):
        try:
            tensor_payload = convert_tensor_payload_to_runtime_bounded(
                reference_payload,
                reference_runtime,
                runtime,
                raw_tensor_cache,
            )
        except ValueError:
            print(timestamped_message(
                f"[run_map_task] member {member_idx}/{len(runtimes)} requires tensor rebuild "
                "because short/long feature layout differs"
            ))
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
            model_type=model_type,
            batch_size=forward_batch_size,
            use_cuda_autocast=use_cuda_autocast,
        )
        member_dfs.append(preds_df)
    agg_df = aggregate_member_predictions(member_dfs)
    dense_payload = densify_tile_predictions(agg_df, tile_payload)
    save_tile_shard(
        shard_path=task_row["shard_path"],
        dense_payload=dense_payload,
        tile_payload=tile_payload,
        task_row=task_row,
    )
    elapsed = time.perf_counter() - task_t0
    print(timestamped_message(
        f"[run_map_task] {progress_label} wrote shard {task_row['shard_path']} "
        f"in {_format_seconds(elapsed)}"
    ))
    return elapsed


def main():
    args = get_args()
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if task_id < 0:
        raise ValueError("task_id must be provided or available via SLURM_ARRAY_TASK_ID")

    manifest_df = pd.read_csv(args.manifest_path)
    manifest_df = manifest_df.sort_values(["job_task_id", "task_id"]).reset_index(drop=True)
    row_mask = manifest_df["job_task_id"].astype(int) == int(task_id)
    if not row_mask.any():
        raise KeyError(f"job_task_id={task_id} not found in manifest {args.manifest_path}")
    task_rows = manifest_df.loc[row_mask].copy()
    run_dir = os.path.dirname(args.manifest_path)
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)

    print(timestamped_message(
        f"[run_map_task] job_task_id={task_id} processing {len(task_rows)} fine tasks"
    ))
    member_dirs, runtimes = load_ensemble_runtimes(
        ensemble_root=run_config["ensemble_root"],
        input_data_name=run_config["input_data_name"],
        inputs_root=run_config.get("inputs_root"),
        fold=int(run_config.get("fold", 9998)),
        fallback_num_tasks=int(run_config.get("fallback_num_tasks", 3)),
        member_name_prefix=run_config.get("ensemble_member_name_prefix"),
        selection_key=run_config.get("ensemble_selection_key"),
    )
    if member_dirs != run_config["member_dirs"]:
        raise ValueError(
            "[run_map_task] resolved member_dirs differ from persisted run_config member_dirs"
        )
    model_type = args.model_type if args.model_type is not None else run_config.get("model_type", DEFAULT_MODEL_TYPE)
    forward_batch_size = int(run_config.get("forward_batch_size", 512))
    use_cuda_autocast = bool(run_config.get("use_cuda_autocast", True))
    dss = get_inference_datasets()
    job_t0 = time.perf_counter()
    total_fine_tasks = len(manifest_df)
    completed_elapsed = []
    task_rows = task_rows.reset_index(drop=False).rename(columns={"index": "overall_zero_idx"})
    for row_idx in range(len(task_rows)):
        task_row = task_rows.iloc[row_idx]
        overall_rank = int(task_row["overall_zero_idx"]) + 1
        elapsed_so_far = time.perf_counter() - job_t0
        mean_task_elapsed = (
            sum(completed_elapsed) / len(completed_elapsed)
            if len(completed_elapsed) > 0
            else None
        )
        remaining_in_job = len(task_rows) - row_idx
        eta_job = (
            mean_task_elapsed * remaining_in_job
            if mean_task_elapsed is not None
            else None
        )
        progress_label = (
            f"job {task_id} fine {row_idx + 1}/{len(task_rows)} overall {overall_rank}/{total_fine_tasks}"
        )
        print(timestamped_message(
            f"[run_map_task] {progress_label}; elapsed={_format_seconds(elapsed_so_far)}"
            + (
                f"; est_job_remaining={_format_seconds(eta_job)}"
                if eta_job is not None
                else ""
            )
        ))
        task_elapsed = process_task_row(
            task_row,
            runtimes=runtimes,
            dss=dss,
            model_type=model_type,
            forward_batch_size=forward_batch_size,
            use_cuda_autocast=use_cuda_autocast,
            progress_label=progress_label,
        )
        completed_elapsed.append(task_elapsed)
    total_elapsed = time.perf_counter() - job_t0
    print(timestamped_message(
        f"[run_map_task] job_task_id={task_id} complete in {_format_seconds(total_elapsed)}"
    ))


if __name__ == "__main__":
    main()
