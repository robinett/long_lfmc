#!/usr/bin/env python3

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from viewer_pipeline_common import (
    asset_root,
    build_color_ramp,
    date_status_path,
    datestr_values,
    load_config,
    log,
    plan_path,
    read_json,
    selected_dates_from_plan,
    tile_status_dir,
    viewer_dataset_path,
    write_json,
    write_png,
)


def landcover_mapping(ds: xr.Dataset, landcover_variable: str) -> Dict[int, str]:
    landcover_da = ds[landcover_variable]
    code_to_name = landcover_da.attrs.get("code_to_name", {})
    nodata_code = landcover_da.attrs.get("nodata_code")
    if isinstance(code_to_name, dict):
        mapping = {int(key): str(value) for key, value in code_to_name.items()}
        if nodata_code is not None:
            mapping[int(nodata_code)] = "nodata"
        return mapping
    dataset_key = ds.attrs.get("dominant_landcover_code_key")
    if isinstance(dataset_key, str):
        parsed = json.loads(dataset_key)
        mapping = {int(key): str(value) for key, value in parsed.items()}
        if nodata_code is not None:
            mapping[int(nodata_code)] = "nodata"
        return mapping
    return {}


def landcover_code_for_name(mapping: Dict[int, str], target_name: str):
    for code, name in mapping.items():
        if name == target_name:
            return code
    return None


def landcover_year_index(ds: xr.Dataset, landcover_variable: str, date_str: str) -> int:
    landcover_da = ds[landcover_variable]
    if "landcover_year" not in landcover_da.dims:
        return 0
    years = np.asarray(landcover_da["landcover_year"].values)
    year = int(date_str[:4])
    return int(np.argmin(np.abs(years.astype(np.int64) - year)))


def tile_relpath(layer_key: str, date_str: str, zoom: int, tile_x: int, tile_y: int) -> Path:
    return Path("tiles") / layer_key / date_str / str(zoom) / str(tile_x) / f"{tile_y}.png"


def tile_counts(height: int, width: int, tile_size: int) -> Tuple[int, int]:
    return int(math.ceil(width / tile_size)), int(math.ceil(height / tile_size))


def downsample_rgba(rgba: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return rgba
    return np.ascontiguousarray(rgba[::factor, ::factor, :])


def rgba_for_layer(ds, cfg, layer_key: str, time_idx: int, date_str: str, evergreen_code):
    layer_cfg = cfg["layers"][layer_key]
    variable_name = str(layer_cfg["variable"])
    landcover_variable = str(cfg["dataset"]["landcover_variable"])
    rendering_cfg = cfg.get("rendering", {})
    default_valid_alpha = int(rendering_cfg.get("default_valid_alpha", 235))
    evergreen_forest_alpha = int(rendering_cfg.get("evergreen_forest_alpha", 60))

    data = np.asarray(ds[variable_name].isel(time=time_idx).values, dtype=np.float32)
    valid_mask = np.isfinite(data)
    value_min = float(layer_cfg["min"])
    value_max = float(layer_cfg["max"])
    normalized = (data - value_min) / (value_max - value_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid_mask] = 0.0
    stops = np.asarray(layer_cfg["stops"], dtype=np.float32)
    colors = np.asarray(layer_cfg["palette"], dtype=np.float32)
    rgb = build_color_ramp(normalized, stops, colors)
    alpha = np.where(valid_mask, default_valid_alpha, 0).astype(np.uint8)

    if evergreen_code is not None:
        lc_idx = landcover_year_index(ds, landcover_variable, date_str)
        landcover_codes = np.asarray(ds[landcover_variable].isel(landcover_year=lc_idx).values)
        alpha[valid_mask & (landcover_codes == evergreen_code)] = evergreen_forest_alpha
    return np.dstack([rgb, alpha])


def write_tiles_for_image(root: Path, rgba: np.ndarray, cfg, layer_key: str, date_str: str) -> Dict[str, Dict[str, int]]:
    tiles_cfg = cfg["tiles"]
    tile_size = int(tiles_cfg["tile_size"])
    min_zoom = int(tiles_cfg["min_zoom"])
    max_zoom = int(tiles_cfg["max_zoom"])
    counts = {}
    for zoom in range(min_zoom, max_zoom + 1):
        factor = 2 ** (max_zoom - zoom)
        zoom_rgba = downsample_rgba(rgba, factor)
        array_height, array_width, _ = zoom_rgba.shape
        tiles_x, tiles_y = tile_counts(array_height, array_width, tile_size)
        counts[str(zoom)] = {"x": tiles_x, "y": tiles_y}
        log(f"Tiling {layer_key} {date_str} zoom {zoom}: {tiles_x}x{tiles_y} tiles")

        for tile_y in range(tiles_y):
            row_start = tile_y * tile_size
            row_end = min(row_start + tile_size, array_height)
            for tile_x in range(tiles_x):
                col_start = tile_x * tile_size
                col_end = min(col_start + tile_size, array_width)
                tile_rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
                tile_rgba[: row_end - row_start, : col_end - col_start, :] = zoom_rgba[row_start:row_end, col_start:col_end, :]
                write_png(root / tile_relpath(layer_key, date_str, zoom, tile_x, tile_y), tile_rgba)
    return counts


def run_dates(args):
    cfg = load_config(args.config)
    plan = read_json(args.plan_path or plan_path(cfg))
    array_index = args.array_index
    if array_index is None and args.use_slurm_array:
        import os

        array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    dates = args.dates or selected_dates_from_plan(plan, array_index=array_index)
    if not dates:
        log("No dates selected for tile worker")
        return

    ds = xr.open_zarr(viewer_dataset_path(cfg), consolidated=False)
    root = asset_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    available_dates = datestr_values(ds["time"].values)
    date_index = {date: idx for idx, date in enumerate(available_dates)}
    landcover_variable = str(cfg["dataset"]["landcover_variable"])
    evergreen_code = landcover_code_for_name(landcover_mapping(ds, landcover_variable), "evergreen_forest")

    try:
        for date_str in dates:
            if date_str not in date_index:
                raise ValueError(f"Date {date_str} not found in viewer dataset")
            time_idx = date_index[date_str]
            layer_counts = {}
            for layer_key in cfg["layers"]:
                date_layer_dir = root / "tiles" / layer_key / date_str
                if date_layer_dir.exists():
                    shutil.rmtree(date_layer_dir)
                log(f"Rendering tiles for {layer_key} {date_str}")
                rgba = rgba_for_layer(ds, cfg, layer_key, time_idx, date_str, evergreen_code)
                layer_counts[layer_key] = write_tiles_for_image(root, rgba, cfg, layer_key, date_str)

            write_json(
                date_status_path(tile_status_dir(cfg), date_str),
                {
                    "status": "completed",
                    "date": date_str,
                    "time_idx": int(time_idx),
                    "layers": layer_counts,
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            log(f"Completed tiles for {date_str}")
    finally:
        ds.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Build viewer PNG tiles for selected dates.")
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
