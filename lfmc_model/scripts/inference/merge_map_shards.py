#!/usr/bin/env python3

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from map_runtime_utils import (
    DEFAULT_MODEL_GRID_PATH,
    OUTPUT_DOMINANT_LANDCOVER_NAME,
    OUTPUT_MEAN_NAME,
    OUTPUT_QUALITY_FLAG_NAME,
    OUTPUT_STD_NAME,
    build_dominant_landcover_metadata,
    initialize_output_store,
    merge_shard_into_store,
    open_model_grid,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge ensemble map shard files into the final grid zarr store."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--grid_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--initialize_only", action="store_true")
    parser.add_argument("--merge_task_id", type=int, default=None)
    return parser.parse_args()


def _load_run_context(manifest_path: str):
    run_dir = os.path.dirname(manifest_path)
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)
    manifest_df = pd.read_csv(manifest_path)
    return run_dir, run_config, manifest_df


def _build_time_index(run_config: dict):
    return pd.date_range(
        pd.Timestamp(run_config["safe_start_date"]).normalize(),
        pd.Timestamp(run_config["safe_end_date"]).normalize(),
        freq="D",
    )


def _initialize_store(
    out_zarr_path: str,
    run_config: dict,
    grid_path: str | None,
    overwrite: bool,
):
    if os.path.exists(out_zarr_path) and not overwrite:
        raise FileExistsError(
            f"Output zarr already exists: {out_zarr_path}. Use --overwrite to replace it."
        )
    if os.path.exists(out_zarr_path) and overwrite:
        print(f"[merge_map_shards] Removing existing store {out_zarr_path}")
        shutil.rmtree(out_zarr_path)

    resolved_grid_path = (
        grid_path if grid_path is not None else run_config.get("grid_path", DEFAULT_MODEL_GRID_PATH)
    )
    model_grid = open_model_grid(resolved_grid_path)
    time_index = _build_time_index(run_config)
    landcover_metadata = None
    landcover_path = run_config.get("landcover_path")
    landcover_output_years = run_config.get("landcover_output_years", [])
    if landcover_path and len(landcover_output_years) > 0:
        print(
            f"[merge_map_shards] Building dominant landcover metadata from {landcover_path} "
            f"for years {landcover_output_years}"
        )
        landcover_metadata = build_dominant_landcover_metadata(
            landcover_path=landcover_path,
            output_years=landcover_output_years,
        )
    print(
        f"[merge_map_shards] Initializing store {out_zarr_path} with "
        f"{len(time_index):,} daily steps"
    )
    initialize_output_store(
        out_path=out_zarr_path,
        model_grid=model_grid,
        time_index=time_index,
        time_chunk=int(run_config["time_chunk_days"]),
        y_chunk=int(run_config["y_chunk"]),
        x_chunk=int(run_config["x_chunk"]),
        landcover_metadata=landcover_metadata,
        product_tier=str(run_config.get("product_tier", "final")),
    )
    return time_index


def main():
    args = get_args()
    run_dir, run_config, manifest_df = _load_run_context(args.manifest_path)

    out_zarr_path = run_config["out_zarr_path"]
    if args.initialize_only:
        time_index = _initialize_store(
            out_zarr_path=out_zarr_path,
            run_config=run_config,
            grid_path=args.grid_path,
            overwrite=args.overwrite,
        )
        summary = {
            "manifest_path": args.manifest_path,
            "out_zarr_path": out_zarr_path,
            "mode": "initialize_only",
            "n_time_steps": int(len(time_index)),
            "safe_start_date": run_config["safe_start_date"],
            "safe_end_date": run_config["safe_end_date"],
        }
        summary_path = os.path.join(run_dir, "merge_summary_init.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"[merge_map_shards] Wrote initialization summary to {summary_path}")
        return

    if args.merge_task_id is not None:
        if "merge_task_id" not in manifest_df.columns:
            raise ValueError(
                "Manifest is missing merge_task_id. Rebuild the manifest with the current code."
            )
        manifest_df = manifest_df.loc[
            manifest_df["merge_task_id"].astype(int) == int(args.merge_task_id)
        ].copy()
        if len(manifest_df) == 0:
            raise ValueError(f"No manifest rows found for merge_task_id={args.merge_task_id}")
        print(
            f"[merge_map_shards] merge_task_id={args.merge_task_id} "
            f"covering {len(manifest_df):,} shard(s)"
        )
    else:
        if not os.path.exists(out_zarr_path):
            _initialize_store(
                out_zarr_path=out_zarr_path,
                run_config=run_config,
                grid_path=args.grid_path,
                overwrite=args.overwrite,
            )

    if not os.path.exists(out_zarr_path):
        raise FileNotFoundError(
            f"Output zarr store is missing before merge: {out_zarr_path}. "
            "Initialize the store before running merge tasks."
        )

    time_index = _build_time_index(run_config)
    time_lookup = {np.datetime64(ts.to_datetime64()): idx for idx, ts in enumerate(time_index)}

    missing_shards = [
        shard_path
        for shard_path in manifest_df["shard_path"].tolist()
        if not os.path.exists(shard_path)
    ]
    if len(missing_shards) > 0:
        raise FileNotFoundError(
            f"Missing {len(missing_shards)} shard(s); first missing shard: {missing_shards[0]}"
        )

    shard_iter = tqdm(
        manifest_df.sort_values(["block_idx", "tile_iy", "tile_ix"]).itertuples(index=False),
        total=len(manifest_df),
        desc=(
            "Merging shards"
            if args.merge_task_id is None
            else f"Merging merge_task_id={args.merge_task_id}"
        ),
        unit="shard",
    )
    for row in shard_iter:
        merge_shard_into_store(
            store_path=out_zarr_path,
            shard_path=row.shard_path,
            time_lookup=time_lookup,
        )

    summary = {
        "manifest_path": args.manifest_path,
        "out_zarr_path": out_zarr_path,
        "mode": "merge_task" if args.merge_task_id is not None else "full_merge",
        "merge_task_id": args.merge_task_id,
        "n_shards": int(len(manifest_df)),
        "vars": run_config.get(
            "output_var_names",
            [
                OUTPUT_MEAN_NAME,
                OUTPUT_STD_NAME,
                OUTPUT_QUALITY_FLAG_NAME,
                OUTPUT_DOMINANT_LANDCOVER_NAME,
            ],
        ),
        "safe_start_date": run_config["safe_start_date"],
        "safe_end_date": run_config["safe_end_date"],
        "product_tier": run_config.get("product_tier", "final"),
        "block_indices": sorted(manifest_df["block_idx"].astype(int).unique().tolist()),
    }
    if args.merge_task_id is None:
        summary_path = os.path.join(run_dir, "merge_summary.json")
    else:
        summary_path = os.path.join(
            run_dir,
            f"merge_summary_task_{int(args.merge_task_id):04d}.json",
        )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[merge_map_shards] Wrote merge summary to {summary_path}")


if __name__ == "__main__":
    main()
