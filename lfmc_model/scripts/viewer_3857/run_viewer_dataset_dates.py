#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
import zarr

from viewer_pipeline_common import (
    dataset_status_dir,
    date_status_path,
    datestr_values,
    load_config,
    log,
    open_source_dataset,
    plan_path,
    read_json,
    read_lookup,
    selected_dates_from_plan,
    viewer_dataset_path,
    write_json,
)


def build_date_index(dates):
    return {date: idx for idx, date in enumerate(dates)}


def run_dates(args):
    cfg = load_config(args.config)
    plan = read_json(args.plan_path or plan_path(cfg))
    array_index = args.array_index
    if array_index is None and args.use_slurm_array:
        import os

        array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    dates = args.dates or selected_dates_from_plan(plan, array_index=array_index)
    if not dates:
        log("No dates selected for viewer dataset worker")
        return

    display_variable = str(cfg["dataset"]["display_variable"])
    uncertainty_variable = str(cfg["dataset"]["uncertainty_variable"])
    quality_variable = str(cfg["dataset"]["quality_variable"])
    ds = open_source_dataset(cfg)
    root = zarr.open_group(str(viewer_dataset_path(cfg)), mode="a")
    row_lookup, col_lookup = read_lookup(cfg)
    source_date_index = build_date_index(datestr_values(ds["time"].values))
    viewer_date_index = build_date_index(datestr_values(root["time"][:]))
    source_quality = np.asarray(ds[quality_variable].values, dtype=np.uint8)

    try:
        for date_str in dates:
            if date_str not in source_date_index:
                raise ValueError(f"Date {date_str} not found in source dataset")
            if date_str not in viewer_date_index:
                raise ValueError(f"Date {date_str} not found in viewer dataset; rerun init append first")
            source_idx = source_date_index[date_str]
            viewer_idx = viewer_date_index[date_str]
            log(f"Sampling viewer dataset date {date_str} source_idx={source_idx} viewer_idx={viewer_idx}")

            mean_source = np.asarray(ds[display_variable].isel(time=source_idx).values, dtype=np.float32)
            std_source = np.asarray(ds[uncertainty_variable].isel(time=source_idx).values, dtype=np.float32)
            root[display_variable][viewer_idx, :, :] = mean_source[row_lookup, col_lookup].astype(np.float32)
            root[uncertainty_variable][viewer_idx, :, :] = std_source[row_lookup, col_lookup].astype(np.float32)
            root[quality_variable][viewer_idx] = int(source_quality[source_idx])

            write_json(
                date_status_path(dataset_status_dir(cfg), date_str),
                {
                    "status": "completed",
                    "date": date_str,
                    "source_idx": int(source_idx),
                    "viewer_idx": int(viewer_idx),
                    "quality_flag": int(source_quality[source_idx]),
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            log(f"Completed viewer dataset date {date_str}")
    finally:
        ds.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Build viewer Zarr slices for selected dates.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "viewer_pipeline_config.yaml")
    parser.add_argument("--plan-path", type=Path, default=None)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--use-slurm-array", action="store_true")
    parser.add_argument("--dates", nargs="*", default=None)
    return parser.parse_args()


def main():
    run_dates(parse_args())


if __name__ == "__main__":
    main()
