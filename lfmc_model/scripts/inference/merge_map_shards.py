#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from map_runtime_utils import (
    DEFAULT_MODEL_GRID_PATH,
    OUTPUT_MEAN_NAME,
    OUTPUT_STD_NAME,
    initialize_output_store,
    merge_shard_into_store,
    open_model_grid,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge ensemble map shard files into the final grid zarr store."
    )
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--grid_path", type=str, default=DEFAULT_MODEL_GRID_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    manifest_df = pd.read_csv(args.manifest_path)
    run_dir = os.path.dirname(args.manifest_path)
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)

    out_zarr_path = run_config["out_zarr_path"]
    if os.path.exists(out_zarr_path) and not args.overwrite:
        raise FileExistsError(
            f"Output zarr already exists: {out_zarr_path}. Use --overwrite to replace it."
        )
    if os.path.exists(out_zarr_path) and args.overwrite:
        print(f"[merge_map_shards] Removing existing store {out_zarr_path}")
        import shutil

        shutil.rmtree(out_zarr_path)

    model_grid = open_model_grid(args.grid_path)
    time_index = pd.date_range(
        pd.Timestamp(run_config["safe_start_date"]).normalize(),
        pd.Timestamp(run_config["safe_end_date"]).normalize(),
        freq="D",
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
    )
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
        desc="Merging shards",
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
        "n_shards": int(len(manifest_df)),
        "vars": [OUTPUT_MEAN_NAME, OUTPUT_STD_NAME],
        "safe_start_date": run_config["safe_start_date"],
        "safe_end_date": run_config["safe_end_date"],
    }
    summary_path = os.path.join(run_dir, "merge_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[merge_map_shards] Wrote merge summary to {summary_path}")


if __name__ == "__main__":
    main()
