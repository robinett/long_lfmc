#!/usr/bin/env python3

import argparse
import json
import os
import time

import pandas as pd

from map_runtime_utils import (
    DEFAULT_MODEL_TYPE,
    build_static_superset_runtime,
    convert_tensor_payload_to_runtime,
    densify_tile_predictions,
    finalize_running_ensemble_predictions,
    initialize_running_ensemble_predictions,
    load_ensemble_runtimes,
    load_prepared_tensor_payload,
    load_runtime_forward_predictor,
    load_tile_payload,
    run_runtime_forward_loaded,
    save_tile_shard,
    timestamped_message,
    update_running_ensemble_predictions,
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
        max_members=run_config.get("max_ensemble_members"),
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
    use_cuda_autocast = bool(run_config.get("use_cuda_autocast", True))
    prepared_dir = run_config["prepared_dir"]
    reference_runtime = build_static_superset_runtime(runtimes[0], runtimes)
    total_fine_tasks = len(manifest_df)
    task_rows = task_rows.reset_index(drop=False).rename(columns={"index": "overall_zero_idx"})

    print(timestamped_message(
        f"[run_map_task_gpu] selected {len(runtimes)} ensemble members for GPU job {task_id}"
    ))

    preload_t0 = time.perf_counter()
    task_states = []
    for row_idx in range(len(task_rows)):
        task_row = task_rows.iloc[row_idx]
        fine_task_id = int(task_row["task_id"])
        overall_rank = int(task_row["overall_zero_idx"]) + 1
        progress_label = (
            f"gpu job {task_id} fine {row_idx + 1}/{len(task_rows)} overall {overall_rank}/{total_fine_tasks}"
        )
        print(timestamped_message(
            f"[run_map_task_gpu] preloading {progress_label} "
            f"task_id={fine_task_id} tile={task_row['tile_name']} "
            f"window={task_row['start_date']} to {task_row['end_date']}"
        ))
        tile_payload = load_tile_payload(task_row["tile_meta_path"])
        reference_path = _prepared_paths(prepared_dir, fine_task_id)
        reference_payload = load_prepared_tensor_payload(reference_path)
        task_states.append({
            "task_row": task_row,
            "fine_task_id": fine_task_id,
            "progress_label": progress_label,
            "site_key": f"tile_{task_row['tile_name']}_{task_row['start_date']}",
            "tile_payload": tile_payload,
            "reference_payload": reference_payload,
            "aggregator": initialize_running_ensemble_predictions(reference_payload["info_df"]),
            "started_at": None,
        })
    print(timestamped_message(
        f"[run_map_task_gpu] preloaded {len(task_states)} fine tasks in "
        f"{_format_seconds(time.perf_counter() - preload_t0)}"
    ))

    predictor_t0 = time.perf_counter()
    predictor_states = []
    for member_idx, runtime in enumerate(runtimes, start=1):
        print(timestamped_message(
            f"[run_map_task_gpu] loading member model {member_idx}/{len(runtimes)} "
            f"({os.path.basename(member_dirs[member_idx - 1])})"
        ))
        predictor_states.append({
            "member_idx": member_idx,
            "runtime": runtime,
            "predictor": load_runtime_forward_predictor(runtime, model_type=model_type),
        })
    print(timestamped_message(
        f"[run_map_task_gpu] loaded {len(predictor_states)} predictors in "
        f"{_format_seconds(time.perf_counter() - predictor_t0)}"
    ))

    job_t0 = time.perf_counter()
    completed_member_elapsed = []
    for member_pos, predictor_state in enumerate(predictor_states, start=1):
        member_idx = predictor_state["member_idx"]
        runtime = predictor_state["runtime"]
        mean_member_elapsed = (
            sum(completed_member_elapsed) / len(completed_member_elapsed)
            if completed_member_elapsed
            else None
        )
        remaining_members = len(predictor_states) - member_pos + 1
        eta_job = (
            mean_member_elapsed * remaining_members
            if mean_member_elapsed is not None
            else None
        )
        print(timestamped_message(
            f"[run_map_task_gpu] member sweep {member_pos}/{len(predictor_states)} "
            f"member_idx={member_idx}; elapsed={_format_seconds(time.perf_counter() - job_t0)}"
            + (
                f"; est_job_remaining={_format_seconds(eta_job)}"
                if eta_job is not None
                else ""
            )
        ))
        member_t0 = time.perf_counter()
        for task_state in task_states:
            task_row = task_state["task_row"]
            if task_state["started_at"] is None:
                task_state["started_at"] = time.perf_counter()
            print(timestamped_message(
                f"[run_map_task_gpu] member {member_pos}/{len(predictor_states)} "
                f"processing {task_state['progress_label']}"
            ))
            try:
                tensor_payload = convert_tensor_payload_to_runtime(
                    task_state["reference_payload"],
                    reference_runtime,
                    runtime,
                    {},
                    site=task_state["site_key"],
                )
            except ValueError:
                member_path = _prepared_paths(
                    prepared_dir,
                    task_state["fine_task_id"],
                    member_idx=member_idx,
                )
                print(timestamped_message(
                    f"[run_map_task_gpu] member {member_idx}/{len(runtimes)} "
                    f"loading member-specific tensors {member_path} because short/long "
                    "feature layout differs"
                ))
                tensor_payload = load_prepared_tensor_payload(member_path)
            pred_arrays = run_runtime_forward_loaded(
                predictor=predictor_state["predictor"],
                tensor_payload=tensor_payload,
                batch_size=forward_batch_size,
                return_info_df=False,
                use_cuda_autocast=use_cuda_autocast,
            )
            update_running_ensemble_predictions(
                task_state["aggregator"],
                pred_arrays["lfmc_pred"],
            )
            if member_pos == len(predictor_states):
                agg_df = finalize_running_ensemble_predictions(task_state["aggregator"])
                dense_payload = densify_tile_predictions(agg_df, task_state["tile_payload"])
                save_tile_shard(
                    shard_path=task_row["shard_path"],
                    dense_payload=dense_payload,
                    tile_payload=task_state["tile_payload"],
                    task_row=task_row,
                )
                task_elapsed = time.perf_counter() - task_state["started_at"]
                print(timestamped_message(
                    f"[run_map_task_gpu] {task_state['progress_label']} wrote shard {task_row['shard_path']} "
                    f"in {_format_seconds(task_elapsed)} using {len(predictor_states)} members"
                ))
                task_state["reference_payload"] = None
                task_state["aggregator"] = None
        member_elapsed = time.perf_counter() - member_t0
        completed_member_elapsed.append(member_elapsed)
        print(timestamped_message(
            f"[run_map_task_gpu] member sweep {member_pos}/{len(predictor_states)} complete in "
            f"{_format_seconds(member_elapsed)}"
        ))

    total_elapsed = time.perf_counter() - job_t0
    print(timestamped_message(
        f"[run_map_task_gpu] gpu_job_task_id={task_id} complete in {_format_seconds(total_elapsed)}"
    ))


if __name__ == "__main__":
    main()
