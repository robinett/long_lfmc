#!/usr/bin/env python3

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import zarr
from pyproj import Transformer

from viewer_pipeline_common import (
    asset_root,
    build_lookup,
    create_array,
    dataset_status_dir,
    ensure_state_dirs,
    load_config,
    log,
    open_source_dataset,
    source_grid,
    state_dir,
    tile_status_dir,
    viewer_dataset_path,
    write_json,
)


def dataset_attrs(ds, cfg, grid):
    attrs = dict(ds.attrs)
    attrs["viewer_dataset_label"] = str(cfg["dataset"]["dataset_label"])
    attrs["viewer_dataset_source_path"] = str(cfg["dataset"]["scientific_dataset_path"])
    attrs["viewer_dataset_source_crs"] = str(cfg["dataset"]["scientific_grid_crs"])
    attrs["viewer_dataset_target_crs"] = str(cfg["dataset"]["viewer_grid_crs"])
    attrs["viewer_dataset_target_grid_anchor_mode"] = str(cfg["dataset"].get("viewer_grid_anchor_mode", "source_extent_center"))
    attrs["viewer_dataset_sampling_method"] = str(cfg["dataset"].get("viewer_sampling_method", "nearest_source_cell"))
    attrs["viewer_dataset_target_bounds"] = [float(value) for value in grid["target_bounds"]]
    attrs["viewer_dataset_nominal_resolution_m"] = float(cfg["output"]["viewer_resolution_m"])
    attrs["viewer_dataset_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return attrs


def create_output(root, ds, cfg, grid):
    output_cfg = cfg["output"]
    display_variable = str(cfg["dataset"]["display_variable"])
    uncertainty_variable = str(cfg["dataset"]["uncertainty_variable"])
    quality_variable = str(cfg["dataset"]["quality_variable"])
    landcover_variable = str(cfg["dataset"]["landcover_variable"])
    target_height = int(grid["target_height"])
    target_width = int(grid["target_width"])
    spatial_chunk = int(output_cfg.get("spatial_chunk_size", 256))
    row_chunk = int(output_cfg.get("latlon_row_chunk_size", spatial_chunk))
    time_chunk = min(int(output_cfg.get("time_chunk_size", 64)), int(ds.sizes["time"]))

    create_array(root, "x", data=grid["target_x_values"], chunks=(min(spatial_chunk * 4, target_width),), dims=["x"])
    create_array(root, "y", data=grid["target_y_values"], chunks=(min(spatial_chunk * 4, target_height),), dims=["y"])
    create_array(root, "time", data=np.asarray(ds["time"].values), chunks=(time_chunk,), dims=["time"])
    create_array(
        root,
        "landcover_year",
        data=np.asarray(ds["landcover_year"].values),
        chunks=(max(1, int(ds.sizes["landcover_year"])),),
        dims=["landcover_year"],
    )
    create_array(
        root,
        "lat",
        shape=(target_height, target_width),
        dtype=np.float64,
        chunks=(row_chunk, spatial_chunk),
        dims=["y", "x"],
        attrs=dict(ds["lat"].attrs),
        fill_value=np.nan,
    )
    create_array(
        root,
        "lon",
        shape=(target_height, target_width),
        dtype=np.float64,
        chunks=(row_chunk, spatial_chunk),
        dims=["y", "x"],
        attrs=dict(ds["lon"].attrs),
        fill_value=np.nan,
    )
    create_array(
        root,
        display_variable,
        shape=(int(ds.sizes["time"]), target_height, target_width),
        dtype=np.float32,
        chunks=(time_chunk, spatial_chunk, spatial_chunk),
        dims=["time", "y", "x"],
        attrs=dict(ds[display_variable].attrs),
        fill_value=np.nan,
    )
    create_array(
        root,
        uncertainty_variable,
        shape=(int(ds.sizes["time"]), target_height, target_width),
        dtype=np.float32,
        chunks=(time_chunk, spatial_chunk, spatial_chunk),
        dims=["time", "y", "x"],
        attrs=dict(ds[uncertainty_variable].attrs),
        fill_value=np.nan,
    )
    landcover_attrs = dict(ds[landcover_variable].attrs)
    create_array(
        root,
        landcover_variable,
        shape=(int(ds.sizes["landcover_year"]), target_height, target_width),
        dtype=np.uint8,
        chunks=(1, spatial_chunk, spatial_chunk),
        dims=["landcover_year", "y", "x"],
        attrs=landcover_attrs,
        fill_value=int(landcover_attrs.get("nodata_code", 255)),
    )
    create_array(
        root,
        quality_variable,
        shape=(int(ds.sizes["time"]),),
        dtype=np.uint8,
        chunks=(time_chunk,),
        dims=["time"],
        attrs=dict(ds[quality_variable].attrs),
        fill_value=255,
    )


def resize_if_needed(root, ds, cfg):
    display_variable = str(cfg["dataset"]["display_variable"])
    uncertainty_variable = str(cfg["dataset"]["uncertainty_variable"])
    quality_variable = str(cfg["dataset"]["quality_variable"])
    landcover_variable = str(cfg["dataset"]["landcover_variable"])
    source_time_len = int(ds.sizes["time"])
    source_lc_len = int(ds.sizes["landcover_year"])

    if root["time"].shape[0] != source_time_len:
        old_len = root["time"].shape[0]
        log(f"Resizing viewer time arrays from {old_len} to {source_time_len}")
        root["time"].resize((source_time_len,))
        root[quality_variable].resize((source_time_len,))
        root[display_variable].resize((source_time_len, root[display_variable].shape[1], root[display_variable].shape[2]))
        root[uncertainty_variable].resize((source_time_len, root[uncertainty_variable].shape[1], root[uncertainty_variable].shape[2]))
    root["time"][:] = np.asarray(ds["time"].values)
    root[quality_variable][:] = np.asarray(ds[quality_variable].values, dtype=np.uint8)

    if root["landcover_year"].shape[0] != source_lc_len:
        old_len = root["landcover_year"].shape[0]
        log(f"Resizing landcover arrays from {old_len} to {source_lc_len}")
        root["landcover_year"].resize((source_lc_len,))
        root[landcover_variable].resize((source_lc_len, root[landcover_variable].shape[1], root[landcover_variable].shape[2]))
    root["landcover_year"][:] = np.asarray(ds["landcover_year"].values)


def write_lat_lon(root, cfg, grid):
    row_chunk = int(cfg["output"].get("latlon_row_chunk_size", 256))
    target_height = int(grid["target_height"])
    target_x_values = grid["target_x_values"]
    target_y_values = grid["target_y_values"]
    transformer = Transformer.from_crs(str(cfg["dataset"]["viewer_grid_crs"]), "EPSG:4326", always_xy=True)
    for row_start in range(0, target_height, row_chunk):
        row_end = min(row_start + row_chunk, target_height)
        y_block = target_y_values[row_start:row_end]
        x_grid, y_grid = np.meshgrid(target_x_values, y_block)
        lon_block, lat_block = transformer.transform(x_grid, y_grid)
        root["lat"][row_start:row_end, :] = lat_block.astype(np.float64)
        root["lon"][row_start:row_end, :] = lon_block.astype(np.float64)
        log(f"Computed lat/lon rows {row_start}-{row_end - 1} of {target_height - 1}")


def write_landcover(root, ds, cfg):
    from viewer_pipeline_common import read_lookup

    landcover_variable = str(cfg["dataset"]["landcover_variable"])
    row_lookup, col_lookup = read_lookup(cfg)
    total_years = int(ds.sizes["landcover_year"])
    for year_idx, year_value in enumerate(np.asarray(ds["landcover_year"].values), start=0):
        source_2d = np.asarray(ds[landcover_variable].isel(landcover_year=year_idx).values, dtype=np.uint8)
        root[landcover_variable][year_idx, :, :] = source_2d[row_lookup, col_lookup].astype(np.uint8)
        log(f"Sampled {landcover_variable} year {year_idx + 1}/{total_years} ({year_value})")


def initialize(args):
    cfg = load_config(args.config)
    ensure_state_dirs(cfg)
    ds = open_source_dataset(cfg)
    try:
        grid = source_grid(ds, cfg)
        out = viewer_dataset_path(cfg)
        rebuild = args.mode == "rebuild"
        if rebuild and out.exists():
            log(f"Removing existing viewer dataset {out}")
            shutil.rmtree(out)
        if rebuild:
            for stale_dir in (dataset_status_dir(cfg), tile_status_dir(cfg), asset_root(cfg) / "tiles"):
                if stale_dir.exists():
                    log(f"Removing stale rebuild path {stale_dir}")
                    shutil.rmtree(stale_dir)

        if not out.exists():
            log(f"Creating viewer dataset {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            root = zarr.open_group(
                str(out),
                mode="w",
                zarr_format=2,
                attributes=dataset_attrs(ds, cfg, grid),
            )
            create_output(root, ds, cfg, grid)
            write_lat_lon(root, cfg, grid)
        else:
            log(f"Opening existing viewer dataset {out}")
            root = zarr.open_group(str(out), mode="a")
            resize_if_needed(root, ds, cfg)

        build_lookup(ds, cfg, grid, overwrite=rebuild or args.rebuild_lookup)
        root = zarr.open_group(str(out), mode="a")
        root.attrs.update(dataset_attrs(ds, cfg, grid))
        resize_if_needed(root, ds, cfg)
        landcover_state = state_dir(cfg) / "viewer_landcover_completed.json"
        landcover_current = False
        if landcover_state.exists():
            import json

            state = json.loads(landcover_state.read_text(encoding="utf-8"))
            landcover_current = int(state.get("landcover_year_count", -1)) == int(ds.sizes["landcover_year"])
        if rebuild or not landcover_current:
            write_landcover(root, ds, cfg)
            write_json(
                landcover_state,
                {
                    "status": "completed",
                    "landcover_year_count": int(ds.sizes["landcover_year"]),
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        else:
            log(f"Using existing viewer landcover from {landcover_state}")

        write_json(
            state_dir(cfg) / "viewer_dataset_init.json",
            {
                "status": "completed",
                "mode": args.mode,
                "viewer_dataset_path": str(out),
                "source_dataset_path": str(cfg["dataset"]["scientific_dataset_path"]),
                "time_count": int(ds.sizes["time"]),
                "landcover_year_count": int(ds.sizes["landcover_year"]),
                "target_height": int(grid["target_height"]),
                "target_width": int(grid["target_width"]),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        log(f"Initialized viewer dataset {out}")
    finally:
        ds.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize or extend the viewer EPSG:3857 Zarr.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "viewer_pipeline_config.yaml")
    parser.add_argument("--mode", choices=["rebuild", "append"], default="append")
    parser.add_argument("--rebuild-lookup", action="store_true")
    return parser.parse_args()


def main():
    initialize(parse_args())


if __name__ == "__main__":
    main()
