#!/usr/bin/env python3

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import xarray as xr
import yaml
import zarr
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "viewer_pipeline_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, object]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def datetime64_to_datestr(value) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


def datestr_values(values: Sequence[object]) -> List[str]:
    return [datetime64_to_datestr(value) for value in values]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def open_source_dataset(cfg: Dict[str, object]) -> xr.Dataset:
    path = Path(str(cfg["dataset"]["scientific_dataset_path"]))
    log(f"Opening scientific dataset {path}")
    return xr.open_zarr(path, consolidated=False)


def viewer_dataset_path(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["output"]["viewer_dataset_path"]))


def lookup_path(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["output"]["lookup_path"]))


def asset_root(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["output"]["asset_root"]))


def state_dir(cfg: Dict[str, object]) -> Path:
    return Path(str(cfg["output"]["state_dir"]))


def plan_path(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "viewer_update_plan.json"


def dataset_status_dir(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "dataset_dates"


def tile_status_dir(cfg: Dict[str, object]) -> Path:
    return state_dir(cfg) / "tile_dates"


def center_coordinates_from_transform(transform, width: int, height: int):
    x_values = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y_values = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x_values, y_values


def source_grid(ds: xr.Dataset, cfg: Dict[str, object]) -> Dict[str, object]:
    dataset_cfg = cfg["dataset"]
    x_values = np.asarray(ds["x"].values, dtype=np.float64)
    y_values = np.asarray(ds["y"].values, dtype=np.float64)
    dx = abs(float(np.median(np.diff(x_values))))
    dy = abs(float(np.median(np.diff(y_values))))
    source_extent = {
        "west": float(x_values[0] - dx / 2.0),
        "east": float(x_values[-1] + dx / 2.0),
        "north": float(y_values[0] + dy / 2.0),
        "south": float(y_values[-1] - dy / 2.0),
    }

    source_crs = str(dataset_cfg["scientific_grid_crs"])
    target_crs = str(dataset_cfg["viewer_grid_crs"])
    source_to_target = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    target_bounds = transform_bounds(
        source_crs,
        target_crs,
        source_extent["west"],
        source_extent["south"],
        source_extent["east"],
        source_extent["north"],
        densify_pts=21,
    )

    anchor_mode = str(dataset_cfg.get("viewer_grid_anchor_mode", "source_extent_center"))
    if anchor_mode != "source_extent_center":
        raise ValueError(f"Unsupported viewer_grid_anchor_mode {anchor_mode!r}")
    source_center_x = 0.5 * (source_extent["west"] + source_extent["east"])
    source_center_y = 0.5 * (source_extent["south"] + source_extent["north"])
    anchor_x, anchor_y = source_to_target.transform(source_center_x, source_center_y)
    resolution = float(cfg["output"]["viewer_resolution_m"])
    target_west, target_south, target_east, target_north = target_bounds

    col_min = int(np.floor(((target_west - anchor_x) / resolution) + 0.5))
    col_max = int(np.ceil(((target_east - anchor_x) / resolution) - 0.5))
    row_min = int(np.floor(((anchor_y - target_north) / resolution) + 0.5))
    row_max = int(np.ceil(((anchor_y - target_south) / resolution) - 0.5))

    target_width = col_max - col_min + 1
    target_height = row_max - row_min + 1
    target_west_edge = anchor_x + col_min * resolution - (0.5 * resolution)
    target_north_edge = anchor_y - row_min * resolution + (0.5 * resolution)
    target_transform = from_origin(target_west_edge, target_north_edge, resolution, resolution)
    target_x_values, target_y_values = center_coordinates_from_transform(
        target_transform,
        target_width,
        target_height,
    )

    return {
        "x_values": x_values,
        "y_values": y_values,
        "dx": dx,
        "dy": dy,
        "source_extent": source_extent,
        "source_crs": source_crs,
        "target_crs": target_crs,
        "target_bounds": target_bounds,
        "target_transform": target_transform,
        "target_width": target_width,
        "target_height": target_height,
        "target_x_values": target_x_values,
        "target_y_values": target_y_values,
    }


def nearest_index_for_sorted_axis(axis_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    insert_idx = np.searchsorted(axis_values, query_values, side="left")
    insert_idx = np.clip(insert_idx, 0, axis_values.size - 1)
    left_idx = np.clip(insert_idx - 1, 0, axis_values.size - 1)
    right_idx = insert_idx
    left_dist = np.abs(query_values - axis_values[left_idx])
    right_dist = np.abs(axis_values[right_idx] - query_values)
    return np.where(left_dist <= right_dist, left_idx, right_idx).astype(np.int32)


def create_array(root, name: str, *, shape=None, dtype=None, chunks=None, data=None, dims=None, attrs=None, fill_value=None):
    array_attrs = dict(attrs or {})
    array_attrs["_ARRAY_DIMENSIONS"] = list(dims or [])
    create_kwargs = {
        "name": name,
        "chunks": chunks,
        "attributes": array_attrs,
        "overwrite": True,
    }
    if data is not None:
        create_kwargs["data"] = data
    else:
        create_kwargs["shape"] = shape
        create_kwargs["dtype"] = dtype
        create_kwargs["fill_value"] = fill_value
    return root.create_array(**create_kwargs)


def build_lookup(ds: xr.Dataset, cfg: Dict[str, object], grid: Dict[str, object], overwrite: bool = False) -> None:
    out = lookup_path(cfg)
    if overwrite and out.exists():
        log(f"Removing existing lookup {out}")
        shutil.rmtree(out)
    if out.exists():
        log(f"Using existing lookup {out}")
        return

    log(f"Building source lookup {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(
        str(out),
        mode="w",
        zarr_format=2,
        attributes={
            "source_path": str(cfg["dataset"]["scientific_dataset_path"]),
            "source_crs": grid["source_crs"],
            "target_crs": grid["target_crs"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    target_height = int(grid["target_height"])
    target_width = int(grid["target_width"])
    spatial_chunk = int(cfg["output"].get("spatial_chunk_size", 256))
    row_chunk = int(cfg["output"].get("latlon_row_chunk_size", spatial_chunk))
    row_arr = create_array(
        root,
        "source_row",
        shape=(target_height, target_width),
        dtype=np.int32,
        chunks=(row_chunk, spatial_chunk),
        dims=["y", "x"],
        fill_value=-1,
    )
    col_arr = create_array(
        root,
        "source_col",
        shape=(target_height, target_width),
        dtype=np.int32,
        chunks=(row_chunk, spatial_chunk),
        dims=["y", "x"],
        fill_value=-1,
    )

    target_to_source = Transformer.from_crs(grid["target_crs"], grid["source_crs"], always_xy=True)
    source_x_sorted = np.asarray(ds["x"].values, dtype=np.float64)
    source_y_values = np.asarray(ds["y"].values, dtype=np.float64)
    source_y_sorted_asc = source_y_values[::-1].copy()
    target_x_values = grid["target_x_values"]
    target_y_values = grid["target_y_values"]

    for row_start in range(0, target_height, row_chunk):
        row_end = min(row_start + row_chunk, target_height)
        y_block = target_y_values[row_start:row_end]
        x_grid, y_grid = np.meshgrid(target_x_values, y_block)
        source_x_grid, source_y_grid = target_to_source.transform(x_grid, y_grid)
        col_block = nearest_index_for_sorted_axis(source_x_sorted, source_x_grid)
        row_block_asc = nearest_index_for_sorted_axis(source_y_sorted_asc, source_y_grid)
        row_block = (source_y_values.size - 1 - row_block_asc).astype(np.int32)
        row_arr[row_start:row_end, :] = row_block
        col_arr[row_start:row_end, :] = col_block
        log(f"Built lookup rows {row_start}-{row_end - 1} of {target_height - 1}")


def read_lookup(cfg: Dict[str, object]):
    root = zarr.open_group(str(lookup_path(cfg)), mode="r")
    return np.asarray(root["source_row"][:], dtype=np.int32), np.asarray(root["source_col"][:], dtype=np.int32)


def ensure_state_dirs(cfg: Dict[str, object]) -> None:
    state_dir(cfg).mkdir(parents=True, exist_ok=True)
    dataset_status_dir(cfg).mkdir(parents=True, exist_ok=True)
    tile_status_dir(cfg).mkdir(parents=True, exist_ok=True)


def chunk_dates(dates: Sequence[str], block_days: int) -> List[Dict[str, object]]:
    blocks = []
    block_days = max(1, int(block_days))
    for block_idx, start in enumerate(range(0, len(dates), block_days)):
        block_dates = list(dates[start:start + block_days])
        blocks.append(
            {
                "block_index": block_idx,
                "start_date": block_dates[0],
                "end_date": block_dates[-1],
                "dates": block_dates,
            }
        )
    return blocks


def selected_dates_from_plan(plan: Dict[str, object], array_index: int = None) -> List[str]:
    if array_index is None:
        return list(plan.get("dates", []))
    blocks = list(plan.get("blocks", []))
    if array_index < 0 or array_index >= len(blocks):
        raise IndexError(f"Array index {array_index} outside plan block count {len(blocks)}")
    return list(blocks[array_index]["dates"])


def build_color_ramp(normalized_values: np.ndarray, stops: np.ndarray, colors: np.ndarray) -> np.ndarray:
    out = np.empty(normalized_values.shape + (3,), dtype=np.uint8)
    flat = normalized_values.ravel()
    flat_rgb = out.reshape(-1, 3)
    for channel_idx in range(3):
        flat_rgb[:, channel_idx] = np.interp(flat, stops, colors[:, channel_idx]).astype(np.uint8)
    return out


def tile_counts(height: int, width: int, tile_size: int) -> Dict[str, int]:
    return {"x": int(math.ceil(width / tile_size)), "y": int(math.ceil(height / tile_size))}


def write_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG", optimize=True)


def date_status_path(status_root: Path, date_str: str) -> Path:
    return status_root / f"{date_str}.json"


def source_quality_by_date(ds: xr.Dataset, cfg: Dict[str, object]) -> Dict[str, int]:
    quality_variable = str(cfg["dataset"]["quality_variable"])
    dates = datestr_values(ds["time"].values)
    quality = np.asarray(ds[quality_variable].values, dtype=np.uint8)
    return {date: int(value) for date, value in zip(dates, quality)}


def viewer_quality_by_date(path: Path, quality_variable: str) -> Dict[str, int]:
    if not path.exists():
        return {}
    ds = xr.open_zarr(path, consolidated=False)
    try:
        dates = datestr_values(ds["time"].values)
        quality = np.asarray(ds[quality_variable].values, dtype=np.uint8)
        return {date: int(value) for date, value in zip(dates, quality)}
    finally:
        ds.close()


def atomic_replace_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(src, dst)
