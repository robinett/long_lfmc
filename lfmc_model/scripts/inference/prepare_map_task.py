#!/usr/bin/env python3

import argparse
import json
import os

import pandas as pd

from map_runtime_utils import (
    build_reference_tensor_payload,
    build_static_superset_runtime,
    get_inference_datasets,
    load_ensemble_runtimes,
    load_tile_payload,
    save_prepared_tensor_payload,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare tensor payloads for grouped manifest rows without running model forward."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    return parser.parse_args()


def _prepared_paths(prepared_dir: str, fine_task_id: int, member_idx: int | None = None) -> str:
    base = os.path.join(prepared_dir, f"task_{fine_task_id:06d}")
    if member_idx is None:
        return f"{base}_reference.pt"
    return f"{base}_member_{member_idx:02d}.pt"


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
    task_rows = manifest_df.loc[row_mask].reset_index(drop=True)

    run_dir = os.path.dirname(args.manifest_path)
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)

    print(f"[prepare_map_task] job_task_id={task_id} preparing {len(task_rows)} fine tasks")
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
            "[prepare_map_task] resolved ensemble member dirs differ from persisted run_config member_dirs"
        )
    reference_runtime = build_static_superset_runtime(runtimes[0], runtimes)
    differing_member_indices = [
        member_idx
        for member_idx, runtime in enumerate(runtimes[1:], start=2)
        if (
            list(reference_runtime["var_names"]["short_vars"]) != list(runtime["var_names"]["short_vars"])
            or list(reference_runtime["var_names"]["long_vars"]) != list(runtime["var_names"]["long_vars"])
            or list(reference_runtime["short_lag_days"]) != list(runtime["short_lag_days"])
            or list(reference_runtime["long_lag_days"]) != list(runtime["long_lag_days"])
        )
    ]
    if len(differing_member_indices) > 0:
        print(
            f"[prepare_map_task] feature-layout rebuild members: {differing_member_indices}"
        )
    dss = get_inference_datasets()
    prepared_dir = run_config["prepared_dir"]

    for row_idx in range(len(task_rows)):
        task_row = task_rows.iloc[row_idx]
        fine_task_id = int(task_row["task_id"])
        print(
            f"[prepare_map_task] job_task_id={task_id} fine task {row_idx + 1}/{len(task_rows)} "
            f"task_id={fine_task_id} tile={task_row['tile_name']}"
        )
        tile_payload = load_tile_payload(task_row["tile_meta_path"])
        block_start = pd.Timestamp(task_row["start_date"]).normalize()
        block_end = pd.Timestamp(task_row["end_date"]).normalize()
        reference_payload = build_reference_tensor_payload(
            tile_payload=tile_payload,
            runtime=reference_runtime,
            dss=dss,
            start_date=block_start,
            end_date=block_end,
        )
        reference_path = _prepared_paths(prepared_dir, fine_task_id)
        save_prepared_tensor_payload(reference_path, reference_payload)
        print(f"[prepare_map_task] wrote reference payload {reference_path}")
        for member_idx in differing_member_indices:
            runtime = runtimes[member_idx - 1]
            member_payload = build_reference_tensor_payload(
                tile_payload=tile_payload,
                runtime=runtime,
                dss=dss,
                start_date=block_start,
                end_date=block_end,
            )
            member_path = _prepared_paths(prepared_dir, fine_task_id, member_idx=member_idx)
            save_prepared_tensor_payload(member_path, member_payload)
            print(
                f"[prepare_map_task] wrote member-specific payload member={member_idx} "
                f"{member_path}"
            )


if __name__ == "__main__":
    main()
