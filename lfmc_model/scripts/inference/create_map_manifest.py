#!/usr/bin/env python3

import argparse
import datetime
import json
import os

import pandas as pd

from map_runtime_utils import (
    DEFAULT_ENSEMBLE_OUTPUT_ROOT,
    DEFAULT_INPUT_DATA_NAME,
    DEFAULT_MAP_RUN_ROOT,
    DEFAULT_MODEL_GRID_PATH,
    DEFAULT_SCRATCH_ROOT,
    CLIMATE_NC_PATH,
    OUTPUT_MEAN_NAME,
    OUTPUT_STD_NAME,
    STATIC_NC_PATH,
    build_tile_payloads,
    get_inference_datasets,
    load_ensemble_runtimes,
    month_blocks,
    open_model_grid,
    resolve_common_runtime_window,
    select_measurement_rich_month,
    select_validation_sites_for_month,
    locate_sites_to_tiles,
    runtime_temporal_source_lags,
    write_tile_payloads,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Create a blockwise manifest for ensemble wall-to-wall LFMC inference."
    )
    parser.add_argument("--ensemble_root", type=str, default=DEFAULT_ENSEMBLE_OUTPUT_ROOT)
    parser.add_argument("--input_data_name", type=str, default=DEFAULT_INPUT_DATA_NAME)
    parser.add_argument("--run_root", type=str, default=DEFAULT_MAP_RUN_ROOT)
    parser.add_argument("--grid_path", type=str, default=DEFAULT_MODEL_GRID_PATH)
    parser.add_argument("--requested_start_date", type=str, default=None)
    parser.add_argument("--requested_end_date", type=str, default="2024-12-31")
    parser.add_argument("--tile_size", type=int, default=10)
    parser.add_argument("--months_per_block", type=int, default=1)
    parser.add_argument("--time_chunk_days", type=int, default=31)
    parser.add_argument("--y_chunk", type=int, default=100)
    parser.add_argument("--x_chunk", type=int, default=100)
    parser.add_argument("--validation_test", action="store_true")
    parser.add_argument("--validation_site_n", type=int, default=5)
    parser.add_argument("--max_tiles", type=int, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    requested_start = (
        pd.Timestamp(args.requested_start_date).normalize()
        if args.requested_start_date is not None
        else pd.Timestamp("1900-01-01")
    )
    requested_end = pd.Timestamp(args.requested_end_date).normalize()

    print(f"[create_map_manifest] ensemble_root={args.ensemble_root}")
    print(f"[create_map_manifest] input_data_name={args.input_data_name}")
    member_dirs, runtimes = load_ensemble_runtimes(
        ensemble_root=args.ensemble_root,
        input_data_name=args.input_data_name,
    )
    print(f"[create_map_manifest] ensemble members={len(member_dirs)}")
    if len(runtimes) == 0:
        raise ValueError("No runtimes were resolved for manifest creation")
    for idx, runtime in enumerate(runtimes[:3], start=1):
        print(
            f"[create_map_manifest] runtime {idx}: "
            f"short_lags={len(runtime['short_lag_days'])}, "
            f"long_lags={len(runtime['long_lag_days'])}, "
            f"source_lags={runtime_temporal_source_lags(runtime)}"
        )

    dss = get_inference_datasets()
    safe_start, safe_end = resolve_common_runtime_window(
        dss,
        runtimes,
        requested_start,
        requested_end,
    )
    if safe_start > safe_end:
        raise ValueError(
            f"Requested window {requested_start.date()} to {requested_end.date()} "
            f"is outside the valid shared coverage across ensemble members"
        )
    print(
        f"[create_map_manifest] shared valid window: "
        f"{safe_start.date()} to {safe_end.date()}"
    )

    validation_month = None
    validation_sites = []
    model_grid = open_model_grid(args.grid_path)
    tile_payloads = build_tile_payloads(model_grid, tile_size=int(args.tile_size))
    if args.validation_test:
        month_start, month_end, site_error = select_measurement_rich_month(
            args.ensemble_root,
            safe_start,
            safe_end,
        )
        validation_month = {
            "start_date": str(month_start.date()),
            "end_date": str(month_end.date()),
        }
        validation_sites = select_validation_sites_for_month(
            site_error,
            month_start,
            month_end,
            n_sites=int(args.validation_site_n),
        )
        selected_tile_names = locate_sites_to_tiles(
            model_grid,
            validation_sites,
            tile_size=int(args.tile_size),
        )
        safe_start = month_start
        safe_end = month_end
        print(
            f"[create_map_manifest] validation_test selected month "
            f"{safe_start.date()} to {safe_end.date()} with "
            f"{len(validation_sites)} validation sites and {len(selected_tile_names)} tiles"
        )
    else:
        selected_tile_names = sorted(tile_payloads.keys())

    if args.max_tiles is not None:
        selected_tile_names = selected_tile_names[: int(args.max_tiles)]
        print(
            f"[create_map_manifest] restricting to first {len(selected_tile_names)} tiles "
            f"via --max_tiles"
        )

    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_stamp}"
    run_dir = os.path.join(args.run_root, run_name)
    shard_dir = os.path.join(run_dir, "shards")
    tile_meta_dir = os.path.join(run_dir, "tile_metadata")
    merged_dir = os.path.join(run_dir, "merged")
    validation_dir = os.path.join(run_dir, "validation")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(tile_meta_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    tile_payloads = {k: v for k, v in tile_payloads.items() if k in selected_tile_names}
    tile_meta_paths = write_tile_payloads(tile_payloads, tile_meta_dir)
    blocks = month_blocks(safe_start, safe_end, months_per_block=int(args.months_per_block))
    print(f"[create_map_manifest] time blocks={len(blocks)}")

    records = []
    task_id = 0
    for block_idx, (block_start, block_end) in enumerate(blocks):
        for tile_name in selected_tile_names:
            payload = tile_payloads[tile_name]
            tile_ix = int(payload["tile_ix"])
            tile_iy = int(payload["tile_iy"])
            shard_name = (
                f"block_{block_idx:04d}_"
                f"{block_start.strftime('%Y%m%d')}_{block_end.strftime('%Y%m%d')}_"
                f"tile_{tile_name}.npz"
            )
            shard_path = os.path.join(shard_dir, shard_name)
            records.append(
                {
                    "task_id": task_id,
                    "block_idx": block_idx,
                    "tile_name": tile_name,
                    "tile_ix": tile_ix,
                    "tile_iy": tile_iy,
                    "tile_meta_path": tile_meta_paths[tile_name],
                    "start_date": str(block_start.date()),
                    "end_date": str(block_end.date()),
                    "x0": int(payload["x0"]),
                    "x1": int(payload["x1"]),
                    "y0": int(payload["y0"]),
                    "y1": int(payload["y1"]),
                    "n_pixels": int(len(payload["iy"])),
                    "shard_path": shard_path,
                }
            )
            task_id += 1
    manifest_df = pd.DataFrame.from_records(records)
    manifest_path = os.path.join(run_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(
        f"[create_map_manifest] wrote manifest with {len(manifest_df):,} tasks to {manifest_path}"
    )

    out_zarr_path = os.path.join(merged_dir, "lfmc_ensemble_maps.zarr")
    run_config = {
        "run_name": run_name,
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "ensemble_root": args.ensemble_root,
        "input_data_name": args.input_data_name,
        "member_dirs": member_dirs,
        "requested_start_date": str(requested_start.date()),
        "requested_end_date": str(requested_end.date()),
        "safe_start_date": str(safe_start.date()),
        "safe_end_date": str(safe_end.date()),
        "tile_size": int(args.tile_size),
        "months_per_block": int(args.months_per_block),
        "time_chunk_days": int(args.time_chunk_days),
        "y_chunk": int(args.y_chunk),
        "x_chunk": int(args.x_chunk),
        "grid_path": args.grid_path,
        "modis_path": dss["modis"].encoding.get("source", DEFAULT_SCRATCH_ROOT),
        "daymet_path": dss["daymet"].encoding.get("source", DEFAULT_SCRATCH_ROOT),
        "static_path": STATIC_NC_PATH,
        "climate_path": CLIMATE_NC_PATH,
        "tile_metadata_dir": tile_meta_dir,
        "shard_dir": shard_dir,
        "merged_dir": merged_dir,
        "validation_dir": validation_dir,
        "out_zarr_path": out_zarr_path,
        "output_var_names": [OUTPUT_MEAN_NAME, OUTPUT_STD_NAME],
        "validation_test": bool(args.validation_test),
        "validation_month": validation_month,
        "validation_sites": [
            {
                "site_key": rec["site_key"],
                "fold": rec["fold"],
                "num_measurements_month": rec["num_measurements_month"],
            }
            for rec in validation_sites
        ],
    }
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)
    print(f"[create_map_manifest] wrote run config to {run_config_path}")


if __name__ == "__main__":
    main()
