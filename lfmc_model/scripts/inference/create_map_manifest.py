#!/usr/bin/env python3

import argparse
import datetime
import json
import os
from datetime import date, datetime as dt_datetime

import numpy as np
import pandas as pd

from input_source_resolver import (
    json_safe_source_resolution,
    open_inference_datasets_from_resolution,
    resolve_inference_sources,
)
from map_config import config_or_override, default_map_config_path, load_map_config
from map_runtime_utils import (
    ALLOWED_DOMINANT_LANDCOVER,
    DEFAULT_SCRATCH_ROOT,
    CLIMATE_NC_PATH,
    OUTPUT_DOMINANT_LANDCOVER_NAME,
    OUTPUT_MEAN_NAME,
    OUTPUT_QUALITY_FLAG_NAME,
    OUTPUT_STD_NAME,
    QUALITY_FLAG_VALUES,
    STATIC_NC_PATH,
    build_tile_payloads,
    filter_site_records_to_valid_tiles,
    get_inference_datasets,
    load_or_build_prediction_mask_for_year,
    load_ensemble_runtimes,
    month_blocks,
    open_model_grid,
    resolve_common_runtime_window,
    select_measurement_rich_month,
    select_validation_sites_for_month,
    validation_month_window_with_previous,
    locate_sites_to_tiles,
    runtime_temporal_source_lags,
    write_tile_payloads,
)


def _json_safe_obj(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe_obj(v) for v in obj]
    if isinstance(obj, (date, dt_datetime, pd.Timestamp)):
        return str(obj)
    return obj


def _runtime_climate_codes(runtime):
    return {
        int(var_name.split("_")[-1])
        for var_name in runtime["var_names"].get("static_vars", [])
        if var_name.startswith("climate_zone_")
    }


def get_args():
    parser = argparse.ArgumentParser(
        description="Create a blockwise manifest for ensemble wall-to-wall LFMC inference."
    )
    parser.add_argument("--config_path", type=str, default=default_map_config_path())
    parser.add_argument("--ensemble_root", type=str, default=None)
    parser.add_argument("--input_data_name", type=str, default=None)
    parser.add_argument("--inputs_root", type=str, default=None)
    parser.add_argument("--run_root", type=str, default=None)
    parser.add_argument("--source_registry_path", type=str, default=None)
    parser.add_argument("--grid_path", type=str, default=None)
    parser.add_argument("--requested_start_date", type=str, default=None)
    parser.add_argument("--requested_end_date", type=str, default=None)
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--months_per_block", type=int, default=None)
    parser.add_argument("--time_chunk_days", type=int, default=None)
    parser.add_argument("--y_chunk", type=int, default=None)
    parser.add_argument("--x_chunk", type=int, default=None)
    parser.add_argument("--validation_test", action="store_true")
    parser.add_argument("--no-validation_test", dest="validation_test", action="store_false")
    parser.set_defaults(validation_test=None)
    parser.add_argument("--validation_site_n", type=int, default=None)
    parser.add_argument("--max_tiles", type=int, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    cfg = load_map_config(args.config_path)
    ensemble_root = config_or_override(args.ensemble_root, cfg, "ensemble", "outputs_root")
    input_data_name = config_or_override(args.input_data_name, cfg, "ensemble", "input_data_name")
    inputs_root = config_or_override(args.inputs_root, cfg, "ensemble", "inputs_root", default=None)
    fold = int(config_or_override(None, cfg, "ensemble", "fold", default=9998))
    fallback_num_tasks = int(
        config_or_override(None, cfg, "ensemble", "fallback_num_tasks", default=3)
    )
    model_type = str(config_or_override(None, cfg, "ensemble", "model_type", default="standard"))
    max_ensemble_members = config_or_override(None, cfg, "ensemble", "max_members", default=None)
    if max_ensemble_members in {"", "None"}:
        max_ensemble_members = None
    if max_ensemble_members is not None:
        max_ensemble_members = int(max_ensemble_members)
        if max_ensemble_members <= 0:
            raise ValueError("ensemble.max_members must be >= 1 when provided")
    run_root = config_or_override(args.run_root, cfg, "paths", "run_root")
    source_registry_path = config_or_override(
        args.source_registry_path,
        cfg,
        "sources",
        "registry_path",
        default=None,
    )
    if source_registry_path in {"", "None"}:
        source_registry_path = None
    grid_path = config_or_override(args.grid_path, cfg, "data", "grid_path")
    requested_start_raw = config_or_override(
        args.requested_start_date,
        cfg,
        "data",
        "requested_start_date",
        default=None,
    )
    requested_end_raw = config_or_override(
        args.requested_end_date,
        cfg,
        "data",
        "requested_end_date",
        default="2024-12-31",
    )
    tile_size = int(config_or_override(args.tile_size, cfg, "chunking", "tile_size"))
    months_per_block = int(
        config_or_override(args.months_per_block, cfg, "chunking", "months_per_block")
    )
    time_chunk_days = int(
        config_or_override(args.time_chunk_days, cfg, "chunking", "time_chunk_days")
    )
    y_chunk = int(config_or_override(args.y_chunk, cfg, "chunking", "y_chunk"))
    x_chunk = int(config_or_override(args.x_chunk, cfg, "chunking", "x_chunk"))
    validation_test = bool(
        config_or_override(args.validation_test, cfg, "submission", "validation_test")
    )
    validation_site_n = int(
        config_or_override(args.validation_site_n, cfg, "submission", "validation_site_n")
    )
    validation_prediction_split = str(
        config_or_override(
            None,
            cfg,
            "submission",
            "validation_prediction_split",
            default="val",
        )
    ).strip().lower()
    if validation_prediction_split not in {"val", "test"}:
        raise ValueError(
            "submission.validation_prediction_split must be either 'val' or 'test'"
        )
    max_tiles = config_or_override(args.max_tiles, cfg, "submission", "max_tiles", default=None)
    if max_tiles in {"", "None"}:
        max_tiles = None
    if max_tiles is not None:
        max_tiles = int(max_tiles)
        if max_tiles <= 0:
            raise ValueError("submission.max_tiles must be >= 1 when provided")
    product_tier = str(
        config_or_override(None, cfg, "product", "tier", default="final")
    ).strip().lower()
    if product_tier not in QUALITY_FLAG_VALUES:
        raise ValueError(
            "product.tier must be one of "
            f"{sorted(QUALITY_FLAG_VALUES.keys())}; got {product_tier!r}"
        )
    forward_batch_size = int(
        config_or_override(None, cfg, "submission", "forward_batch_size", default=512)
    )
    if forward_batch_size <= 0:
        raise ValueError("submission.forward_batch_size must be >= 1")
    use_cuda_autocast = bool(
        config_or_override(None, cfg, "submission", "use_cuda_autocast", default=True)
    )
    tasks_per_job = int(
        config_or_override(None, cfg, "submission", "tasks_per_job", default=1)
    )
    if tasks_per_job <= 0:
        raise ValueError("submission.tasks_per_job must be >= 1")
    gpu_fine_tasks_per_job = int(
        config_or_override(None, cfg, "submission", "gpu_fine_tasks_per_job", default=1)
    )
    if gpu_fine_tasks_per_job <= 0:
        raise ValueError("submission.gpu_fine_tasks_per_job must be >= 1")
    use_gpu_forward = bool(
        config_or_override(None, cfg, "submission", "use_gpu_forward", default=False)
    )
    merge_blocks_per_job = int(
        config_or_override(None, cfg, "submission", "merge_blocks_per_job", default=1)
    )
    if merge_blocks_per_job <= 0:
        raise ValueError("submission.merge_blocks_per_job must be >= 1")
    requested_start = (
        pd.Timestamp(requested_start_raw).normalize()
        if requested_start_raw not in {None, "", "None"}
        else pd.Timestamp("1900-01-01")
    )
    requested_end = pd.Timestamp(requested_end_raw).normalize()

    print(f"[create_map_manifest] config_path={cfg['_config_path']}")
    print(f"[create_map_manifest] ensemble_root={ensemble_root}")
    print(f"[create_map_manifest] input_data_name={input_data_name}")
    member_dirs, runtimes = load_ensemble_runtimes(
        ensemble_root=ensemble_root,
        input_data_name=input_data_name,
        inputs_root=inputs_root,
        fold=fold,
        fallback_num_tasks=fallback_num_tasks,
        max_members=max_ensemble_members,
    )
    print(f"[create_map_manifest] ensemble members={len(member_dirs)}")
    if max_ensemble_members is not None:
        print(f"[create_map_manifest] member cap active: first {max_ensemble_members} members")
    if len(runtimes) == 0:
        raise ValueError("No runtimes were resolved for manifest creation")
    for idx, runtime in enumerate(runtimes[:3], start=1):
        print(
            f"[create_map_manifest] runtime {idx}: "
            f"short_lags={len(runtime['short_lag_days'])}, "
            f"long_lags={len(runtime['long_lag_days'])}, "
            f"source_lags={runtime_temporal_source_lags(runtime)}"
        )

    source_resolution = None
    if source_registry_path is not None:
        source_resolution = resolve_inference_sources(
            registry_path=source_registry_path,
            product_tier=product_tier,
            requested_start_date=requested_start,
            requested_end_date=requested_end,
            output_years=None,
        )
        print(
            f"[create_map_manifest] source registry={source_registry_path}; "
            f"tier={source_resolution['tier']}; daymet_mode={source_resolution['daymet_mode']}"
        )
        print(
            f"[create_map_manifest] resolved daymet paths={source_resolution['daymet_paths']}"
        )
        dss = open_inference_datasets_from_resolution(source_resolution)
    else:
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
    climate_layouts = [_runtime_climate_codes(runtime) for runtime in runtimes]
    allowed_climate_codes = sorted(set.intersection(*climate_layouts)) if climate_layouts else []
    if len(allowed_climate_codes) == 0:
        raise ValueError("No overlapping climate-zone channels were found across ensemble members")
    excluded_climate_codes = [
        code for code in range(1, 30) if code not in set(allowed_climate_codes)
    ]
    print(
        f"[create_map_manifest] climate-zone intersection across members="
        f"{allowed_climate_codes} (excluded={excluded_climate_codes})"
    )

    blocks = month_blocks(safe_start, safe_end, months_per_block=months_per_block)
    block_years = sorted({block_start.year for block_start, _ in blocks})
    if source_registry_path is not None:
        source_resolution = resolve_inference_sources(
            registry_path=source_registry_path,
            product_tier=product_tier,
            requested_start_date=safe_start,
            requested_end_date=safe_end,
            output_years=block_years,
        )
        print(
            f"[create_map_manifest] resolved NLCD source years="
            f"{source_resolution['nlcd_output_year_to_source_year']}"
        )
    print(
        f"[create_map_manifest] block years={block_years} "
        f"allowed_landcover={list(ALLOWED_DOMINANT_LANDCOVER)}"
    )

    validation_month = None
    validation_sites = []
    model_grid = open_model_grid(grid_path)
    year_tile_payloads = {}
    for block_year in block_years:
        prediction_mask = load_or_build_prediction_mask_for_year(
            model_grid=model_grid,
            landcover_ds=dss["landcover_frac"],
            year=block_year,
            grid_path=grid_path,
            climate_ds=dss["climate_zone"],
            allowed_climate_codes=allowed_climate_codes,
        )
        tile_payloads_for_year = build_tile_payloads(
            model_grid,
            tile_size=tile_size,
            valid_mask=prediction_mask,
        )
        if len(tile_payloads_for_year) == 0:
            raise ValueError(
                f"No valid prediction tiles remain after NLCD filtering for year {block_year}"
            )
        year_tile_payloads[block_year] = tile_payloads_for_year
        valid_pixel_n = int(np.asarray(prediction_mask.values, dtype=bool).sum())
        print(
            f"[create_map_manifest] year={block_year} valid tiles after "
            f"NLCD/random_vals/climate "
            f"filter: {len(tile_payloads_for_year):,}; valid pixels={valid_pixel_n:,}"
        )

    selected_tile_names_by_year = {}
    if validation_test:
        best_month_start, best_month_end, site_error = select_measurement_rich_month(
            ensemble_root,
            safe_start,
            safe_end,
            fold=fold,
            split=validation_prediction_split,
            max_members=max_ensemble_members,
        )
        month_start, month_end = validation_month_window_with_previous(
            best_month_start=best_month_start,
            start_date=safe_start,
            end_date=safe_end,
        )
        validation_year = best_month_start.year
        validation_month = {
            "start_date": str(month_start.date()),
            "end_date": str(month_end.date()),
            "anchor_month_start": str(best_month_start.date()),
            "anchor_month_end": str(best_month_end.date()),
        }
        validation_tile_payloads = year_tile_payloads[validation_year]
        validation_site_candidates = select_validation_sites_for_month(
            site_error,
            best_month_start,
            best_month_end,
            n_sites=max(validation_site_n * 5, validation_site_n),
        )
        print(
            f"[create_map_manifest] validation candidates before tile filter: "
            f"{len(validation_site_candidates)}"
        )
        runnable_validation_sites = filter_site_records_to_valid_tiles(
            model_grid,
            validation_site_candidates,
            tile_size=tile_size,
            valid_tile_names=validation_tile_payloads.keys(),
        )
        print(
            f"[create_map_manifest] validation candidates on runnable tiles: "
            f"{len(runnable_validation_sites)}"
        )
        validation_sites = runnable_validation_sites[:validation_site_n]
        print(
            f"[create_map_manifest] validation sites retained after validation_site_n="
            f"{validation_site_n}: {len(validation_sites)}"
        )
        if len(validation_sites) == 0:
            raise ValueError(
                "Validation-site selection found no sites that fall on tiles with valid "
                "prediction pixels. Check the grid mask and validation-site coverage."
            )
        selected_tile_names_by_year[validation_year] = locate_sites_to_tiles(
            model_grid,
            validation_sites,
            tile_size=tile_size,
        )
        safe_start = month_start
        safe_end = month_end
        blocks = month_blocks(safe_start, safe_end, months_per_block=months_per_block)
        print(
            f"[create_map_manifest] validation_test anchor month "
            f"{best_month_start.date()} to {best_month_end.date()} with "
            f"{len(validation_sites)} validation sites and "
            f"{len(selected_tile_names_by_year[validation_year])} tiles"
        )
        print(
            f"[create_map_manifest] validation_test run window "
            f"{safe_start.date()} to {safe_end.date()} with "
            f"{len(blocks)} monthly block(s)"
        )
    else:
        for block_year, tile_payloads_for_year in year_tile_payloads.items():
            selected_tile_names_by_year[block_year] = sorted(tile_payloads_for_year.keys())

    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_stamp}"
    run_dir = os.path.join(run_root, run_name)
    shard_dir = os.path.join(run_dir, config_or_override(None, cfg, "paths", "shard_subdir"))
    prepared_dir = os.path.join(
        run_dir,
        config_or_override(None, cfg, "paths", "prepared_subdir", default="prepared_tensors"),
    )
    tile_meta_dir = os.path.join(
        run_dir,
        config_or_override(None, cfg, "paths", "tile_metadata_subdir"),
    )
    merged_dir = os.path.join(run_dir, config_or_override(None, cfg, "paths", "merged_subdir"))
    validation_dir = os.path.join(
        run_dir,
        config_or_override(None, cfg, "paths", "validation_subdir"),
    )
    plots_dir = os.path.join(
        validation_dir,
        config_or_override(None, cfg, "paths", "plots_subdir"),
    )
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(prepared_dir, exist_ok=True)
    os.makedirs(tile_meta_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[create_map_manifest] time blocks={len(blocks)}")

    selected_tile_payloads_by_year = {}
    tile_meta_paths_by_year = {}
    active_years = sorted(selected_tile_names_by_year)
    for block_year in active_years:
        tile_payloads_for_year = year_tile_payloads[block_year]
        selected_tile_names = list(selected_tile_names_by_year.get(block_year, []))
        if max_tiles is not None:
            before_cap_n = len(selected_tile_names)
            selected_tile_names = selected_tile_names[:max_tiles]
            if before_cap_n > max_tiles:
                print(
                    f"[create_map_manifest] year={block_year} restricting tiles from "
                    f"{before_cap_n} to {len(selected_tile_names)} via --max_tiles={max_tiles}"
                )
            else:
                print(
                    f"[create_map_manifest] year={block_year} selected {before_cap_n} tiles; "
                    f"--max_tiles={max_tiles} not binding"
                )
        selected_tile_payloads = {
            k: v for k, v in tile_payloads_for_year.items() if k in selected_tile_names
        }
        missing_tile_names = [
            tile_name for tile_name in selected_tile_names if tile_name not in selected_tile_payloads
        ]
        if len(missing_tile_names) > 0:
            raise ValueError(
                "Selected validation tiles are not present in the valid prediction-tile payload "
                f"set for year {block_year}. Missing tiles: {missing_tile_names[:10]}"
            )
        if len(selected_tile_payloads) == 0:
            raise ValueError(
                f"No selected prediction tiles remain for year {block_year} after filtering"
            )
        selected_tile_payloads_by_year[block_year] = selected_tile_payloads
        tile_meta_paths_by_year[block_year] = write_tile_payloads(
            selected_tile_payloads,
            tile_meta_dir,
            file_prefix=f"year_{block_year}",
        )

    records = []
    task_id = 0
    for block_idx, (block_start, block_end) in enumerate(blocks):
        block_year = block_start.year
        tile_payloads = selected_tile_payloads_by_year[block_year]
        tile_meta_paths = tile_meta_paths_by_year[block_year]
        selected_tile_names = sorted(tile_payloads.keys())
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
                    "tile_year": block_year,
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
    if len(manifest_df) == 0:
        raise ValueError("Manifest has zero tile-time tasks after filtering")
    manifest_df = manifest_df.sort_values(
        ["block_idx", "tile_year", "tile_iy", "tile_ix", "task_id"]
    ).reset_index(drop=True)
    manifest_df["job_task_id"] = (
        np.arange(len(manifest_df), dtype=np.int64) // int(tasks_per_job)
    ).astype(np.int64)
    manifest_df["job_task_rank"] = (
        np.arange(len(manifest_df), dtype=np.int64) % int(tasks_per_job)
    ).astype(np.int64)
    manifest_df["gpu_job_task_id"] = (
        np.arange(len(manifest_df), dtype=np.int64) // int(gpu_fine_tasks_per_job)
    ).astype(np.int64)
    manifest_df["gpu_job_task_rank"] = (
        np.arange(len(manifest_df), dtype=np.int64) % int(gpu_fine_tasks_per_job)
    ).astype(np.int64)
    unique_block_ids = sorted(manifest_df["block_idx"].astype(int).unique().tolist())
    merge_task_lookup = {
        block_idx: merge_group_idx
        for merge_group_idx, block_idx in enumerate(unique_block_ids)
    }
    if merge_blocks_per_job > 1:
        merge_task_lookup = {
            block_idx: (block_pos // int(merge_blocks_per_job))
            for block_pos, block_idx in enumerate(unique_block_ids)
        }
    manifest_df["merge_task_id"] = (
        manifest_df["block_idx"].astype(int).map(merge_task_lookup).astype(np.int64)
    )
    manifest_path = os.path.join(run_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    job_task_n = int(manifest_df["job_task_id"].max()) + 1
    gpu_job_task_n = int(manifest_df["gpu_job_task_id"].max()) + 1
    merge_task_n = int(manifest_df["merge_task_id"].max()) + 1
    mean_pixels = float(manifest_df["n_pixels"].mean())
    median_pixels = float(manifest_df["n_pixels"].median())
    print(
        f"[create_map_manifest] wrote manifest with {len(manifest_df):,} fine tasks "
        f"across {job_task_n:,} Slurm jobs to {manifest_path}"
    )
    print(
        f"[create_map_manifest] tasks_per_job={tasks_per_job}; "
        f"mean_pixels_per_task={mean_pixels:.1f}; median_pixels_per_task={median_pixels:.1f}"
    )
    print(
        f"[create_map_manifest] gpu_fine_tasks_per_job={gpu_fine_tasks_per_job}; "
        f"num_gpu_jobs={gpu_job_task_n}"
    )
    print(
        f"[create_map_manifest] merge_blocks_per_job={merge_blocks_per_job}; "
        f"num_merge_tasks={merge_task_n}"
    )

    out_zarr_path = os.path.join(
        merged_dir,
        config_or_override(None, cfg, "paths", "merged_store_name"),
    )
    run_config = {
        "run_name": run_name,
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "config_path": cfg["_config_path"],
        "config": _json_safe_obj(cfg),
        "source_registry_path": source_registry_path,
        "source_resolution": json_safe_source_resolution(source_resolution),
        "ensemble_root": ensemble_root,
        "input_data_name": input_data_name,
        "member_dirs": member_dirs,
        "max_ensemble_members": max_ensemble_members,
        "inputs_root": inputs_root,
        "fold": fold,
        "fallback_num_tasks": fallback_num_tasks,
        "model_type": model_type,
        "tasks_per_job": tasks_per_job,
        "num_job_tasks": job_task_n,
        "forward_batch_size": forward_batch_size,
        "use_cuda_autocast": use_cuda_autocast,
        "use_gpu_forward": use_gpu_forward,
        "gpu_fine_tasks_per_job": gpu_fine_tasks_per_job,
        "num_gpu_job_tasks": gpu_job_task_n,
        "merge_blocks_per_job": merge_blocks_per_job,
        "num_merge_tasks": merge_task_n,
        "requested_start_date": str(requested_start.date()),
        "requested_end_date": str(requested_end.date()),
        "safe_start_date": str(safe_start.date()),
        "safe_end_date": str(safe_end.date()),
        "tile_size": tile_size,
        "months_per_block": months_per_block,
        "time_chunk_days": time_chunk_days,
        "y_chunk": y_chunk,
        "x_chunk": x_chunk,
        "grid_path": grid_path,
        "modis_path": (
            source_resolution["modis_path"]
            if source_resolution is not None
            else dss["modis"].encoding.get("source", DEFAULT_SCRATCH_ROOT)
        ),
        "daymet_path": (
            source_resolution["daymet_paths"][0]
            if source_resolution is not None
            else dss["daymet"].encoding.get("source", DEFAULT_SCRATCH_ROOT)
        ),
        "daymet_paths": (
            list(source_resolution["daymet_paths"])
            if source_resolution is not None
            else [dss["daymet"].encoding.get("source", DEFAULT_SCRATCH_ROOT)]
        ),
        "landcover_path": (
            source_resolution["landcover_path"]
            if source_resolution is not None
            else dss["landcover_frac"].encoding.get("source", DEFAULT_SCRATCH_ROOT)
        ),
        "landcover_output_years": [int(year) for year in active_years],
        "static_path": (
            source_resolution["static_path"]
            if source_resolution is not None
            else STATIC_NC_PATH
        ),
        "climate_path": (
            source_resolution["climate_path"]
            if source_resolution is not None
            else CLIMATE_NC_PATH
        ),
        "allowed_climate_codes": allowed_climate_codes,
        "excluded_climate_codes": excluded_climate_codes,
        "tile_metadata_dir": tile_meta_dir,
        "prepared_dir": prepared_dir,
        "shard_dir": shard_dir,
        "merged_dir": merged_dir,
        "validation_dir": validation_dir,
        "plots_dir": plots_dir,
        "out_zarr_path": out_zarr_path,
        "product_tier": product_tier,
        "quality_flag_value": int(QUALITY_FLAG_VALUES[product_tier]),
        "output_var_names": [
            OUTPUT_MEAN_NAME,
            OUTPUT_STD_NAME,
            OUTPUT_QUALITY_FLAG_NAME,
            OUTPUT_DOMINANT_LANDCOVER_NAME,
        ],
        "validation_test": validation_test,
        "validation_prediction_split": validation_prediction_split,
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
