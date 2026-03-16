#!/usr/bin/env python3

import argparse
import json
import os
import time

import pandas as pd

from map_runtime_utils import (
    DEFAULT_MODEL_TYPE,
    aggregate_member_predictions,
    build_static_superset_runtime,
    convert_tensor_payload_to_runtime,
    densify_tile_predictions,
    load_ensemble_runtimes,
    load_prepared_tensor_payload,
    load_tile_payload,
    run_runtime_forward,
    save_tile_shard,
    timestamped_message,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run grouped GPU-only ensemble forward passes from prepared tensor payloads."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    return parser.parse_args()


def _prepared_paths(prepared_dir: str, fine_task_id: int, member_idx: int | None = None) -> str:
    base = os.path.join(prepared_dir, f"task_{fine_task_id:06d}")
    if member_idx is None:
        return f"{base}_reference.pt"
    return f"{base}_member_{member_idx:02d}.pt"


def _format_seconds(seconds):
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def main():
    args = get_args()
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("GPU_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    if task_id < 0:
        raise ValueError("task_id must be provided or available via GPU_TASK_ID/SLURM_ARRAY_TASK_ID")

    manifest_df = pd.read_csv(args.manifest_path)
    manifest_df = manifest_df.sort_values(["gpu_job_task_id", "task_id"]).reset_index(drop=True)
    row_mask = manifest_df["gpu_job_task_id"].astype(int) == int(task_id)
    if not row_mask.any():
        raise KeyError(f"gpu_job_task_id={task_id} not found in manifest {args.manifest_path}")
    task_rows = manifest_df.loc[row_mask].copy()
    run_dir = os.path.dirname(args.manifest_path)
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)

    print(timestamped_message(
        f"[run_map_task_gpu] gpu_job_task_id={task_id} processing {len(task_rows)} fine tasks"
    ))
    member_dirs, runtimes = load_ensemble_runtimes(
        ensemble_root=run_config["ensemble_root"],
        input_data_name=run_config["input_data_name"],
        inputs_root=run_config.get("inputs_root"),
        fold=int(run_config.get("fold", 9998)),
        fallback_num_tasks=int(run_config.get("fallback_num_tasks", 3)),
    )
    if len(member_dirs) != len(run_config["member_dirs"]):
        raise ValueError(
            f"Run config member count {len(run_config['member_dirs'])} does not match "
            f"resolved member count {len(member_dirs)}"
        )
    model_type = (
        args.model_type
        if args.model_type is not None
        else run_config.get("model_type", DEFAULT_MODEL_TYPE)
    )
    forward_batch_size = int(run_config.get("forward_batch_size", 512))
    prepared_dir = run_config["prepared_dir"]
    reference_runtime = build_static_superset_runtime(runtimes[0], runtimes)

    job_t0 = time.perf_counter()
    total_fine_tasks = len(manifest_df)
    completed_elapsed = []
    task_rows = task_rows.reset_index(drop=False).rename(columns={"index": "overall_zero_idx"})
    for row_idx in range(len(task_rows)):
        task_row = task_rows.iloc[row_idx]
        fine_task_id = int(task_row["task_id"])
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
            f"gpu job {task_id} fine {row_idx + 1}/{len(task_rows)} overall {overall_rank}/{total_fine_tasks}"
        )
        print(timestamped_message(
            f"[run_map_task_gpu] {progress_label}; elapsed={_format_seconds(elapsed_so_far)}"
            + (
                f"; est_job_remaining={_format_seconds(eta_job)}"
                if eta_job is not None
                else ""
            )
        ))
        task_t0 = time.perf_counter()
        print(timestamped_message(
            f"[run_map_task_gpu] {progress_label} "
            f"task_id={fine_task_id} tile={task_row['tile_name']} "
            f"window={task_row['start_date']} to {task_row['end_date']}"
        ))
        tile_payload = load_tile_payload(task_row["tile_meta_path"])
        reference_path = _prepared_paths(prepared_dir, fine_task_id)
        reference_payload = load_prepared_tensor_payload(reference_path)
        renorm_cache = {}
        member_dfs = []
        for member_idx, runtime in enumerate(runtimes, start=1):
            try:
                tensor_payload = convert_tensor_payload_to_runtime(
                    reference_payload,
                    reference_runtime,
                    runtime,
                    renorm_cache,
                    site=f"tile_{task_row['tile_name']}_{task_row['start_date']}",
                )
            except ValueError:
                member_path = _prepared_paths(prepared_dir, fine_task_id, member_idx=member_idx)
                print(timestamped_message(
                    f"[run_map_task_gpu] member {member_idx}/{len(runtimes)} "
                    f"loading member-specific tensors {member_path} because short/long "
                    "feature layout differs"
                ))
                tensor_payload = load_prepared_tensor_payload(member_path)
            preds_df = run_runtime_forward(
                runtime=runtime,
                tensor_payload=tensor_payload,
                model_type=model_type,
                batch_size=forward_batch_size,
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
        task_elapsed = time.perf_counter() - task_t0
        completed_elapsed.append(task_elapsed)
        print(timestamped_message(
            f"[run_map_task_gpu] {progress_label} wrote shard {task_row['shard_path']} "
            f"in {_format_seconds(task_elapsed)}"
        ))
    total_elapsed = time.perf_counter() - job_t0
    print(timestamped_message(
        f"[run_map_task_gpu] gpu_job_task_id={task_id} complete in {_format_seconds(total_elapsed)}"
    ))


if __name__ == "__main__":
    main()
