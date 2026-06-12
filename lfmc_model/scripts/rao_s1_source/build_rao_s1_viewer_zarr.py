#!/usr/bin/env python3

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
import yaml
import zarr


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "rao_s1_source_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def create_array(root, name: str, *, shape=None, dtype=None, chunks=None, data=None, dims=None, attrs=None, fill_value=None):
    array_attrs = dict(attrs or {})
    array_attrs["_ARRAY_DIMENSIONS"] = list(dims or [])
    kwargs = {
        "name": name,
        "chunks": chunks,
        "attributes": array_attrs,
        "overwrite": True,
    }
    if data is None:
        kwargs.update({"shape": shape, "dtype": dtype, "fill_value": fill_value})
    else:
        kwargs["data"] = data
    return root.create_array(**kwargs)


def date_strings(root) -> List[str]:
    values = np.asarray(root["time"][:]).astype("datetime64[D]")
    return [np.datetime_as_string(value, unit="D") for value in values]


def center_transform(x_values: np.ndarray, y_values: np.ndarray) -> Affine:
    dx = float(np.median(np.diff(x_values)))
    dy = float(np.median(np.diff(y_values)))
    west = float(x_values[0] - dx / 2.0)
    north = float(y_values[0] - dy / 2.0)
    return Affine(dx, 0.0, west, 0.0, dy, north)


def center_coordinates_from_transform(transform: Affine, width: int, height: int):
    x_values = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y_values = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x_values, y_values


def native_webmercator_resolution(x_values: np.ndarray, y_values: np.ndarray, source_crs: str, target_crs: str) -> float:
    mid_col = x_values.size // 2
    mid_row = y_values.size // 2
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x0, y0 = transformer.transform(float(x_values[mid_col]), float(y_values[mid_row]))
    x1, _ = transformer.transform(float(x_values[min(mid_col + 1, x_values.size - 1)]), float(y_values[mid_row]))
    _, y1 = transformer.transform(float(x_values[mid_col]), float(y_values[min(mid_row + 1, y_values.size - 1)]))
    dx = abs(float(x1 - x0))
    dy = abs(float(y1 - y0))
    if dx <= 0 or dy <= 0:
        return max(dx, dy, 1.0)
    return float((dx + dy) / 2.0)


def output_grid(cfg: Dict[str, object], source_root) -> Tuple[Affine, int, int, np.ndarray, np.ndarray]:
    x_values = np.asarray(source_root["x"][:], dtype=np.float64)
    y_values = np.asarray(source_root["y"][:], dtype=np.float64)
    source_crs = str(cfg["dataset"]["source_crs"])
    target_crs = str(cfg["dataset"]["viewer_crs"])
    src_transform = center_transform(x_values, y_values)
    src_bounds = array_bounds(y_values.size, x_values.size, src_transform)
    resolution_cfg = cfg["viewer"].get("resolution_m", "auto")
    resolution = None
    if str(resolution_cfg).strip().lower() != "auto":
        resolution = float(resolution_cfg)
    else:
        resolution = native_webmercator_resolution(x_values, y_values, source_crs, target_crs)
    transform, width, height = calculate_default_transform(
        source_crs,
        target_crs,
        x_values.size,
        y_values.size,
        *src_bounds,
        resolution=(resolution, resolution),
    )
    target_x, target_y = center_coordinates_from_transform(transform, width, height)
    return transform, width, height, target_x, target_y


def write_lat_lon(root, target_x: np.ndarray, target_y: np.ndarray, target_crs: str, spatial_chunk: int, row_chunk: int) -> None:
    height = target_y.size
    width = target_x.size
    lat = create_array(
        root,
        "lat",
        shape=(height, width),
        dtype=np.float64,
        chunks=(row_chunk, spatial_chunk),
        dims=["y", "x"],
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )
    lon = create_array(
        root,
        "lon",
        shape=(height, width),
        dtype=np.float64,
        chunks=(row_chunk, spatial_chunk),
        dims=["y", "x"],
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    for row_start in range(0, height, row_chunk):
        row_end = min(row_start + row_chunk, height)
        x_grid, y_grid = np.meshgrid(target_x, target_y[row_start:row_end])
        lon_block, lat_block = transformer.transform(x_grid, y_grid)
        lat[row_start:row_end, :] = lat_block
        lon[row_start:row_end, :] = lon_block
        log(f"Wrote viewer lat/lon rows {row_start}-{row_end - 1} of {height - 1}")


def create_viewer_store(cfg: Dict[str, object], source_root, viewer_path: Path, dates: Sequence[str], overwrite: bool) -> None:
    if viewer_path.exists():
        if not overwrite:
            raise FileExistsError(f"Viewer zarr already exists: {viewer_path}. Use --overwrite to rebuild it.")
        log(f"Removing existing viewer zarr {viewer_path}")
        shutil.rmtree(viewer_path)

    chunks_cfg = cfg["chunks"]
    variable_name = str(cfg["dataset"]["variable_name"])
    spatial_chunk = int(chunks_cfg.get("spatial", 256))
    row_chunk = int(chunks_cfg.get("latlon_row", spatial_chunk))
    time_chunk = int(chunks_cfg.get("viewer_time", 32))
    target_transform, width, height, target_x, target_y = output_grid(cfg, source_root)
    bounds_3857 = array_bounds(height, width, target_transform)
    bounds_geo = transform_bounds(str(cfg["dataset"]["viewer_crs"]), "EPSG:4326", *bounds_3857, densify_pts=21)

    viewer_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(
        str(viewer_path),
        mode="w",
        zarr_format=2,
        attributes={
            "title": str(cfg["dataset"]["dataset_label"]) + " viewer 3857",
            "crs": str(cfg["dataset"]["viewer_crs"]),
            "source_crs": str(cfg["dataset"]["source_crs"]),
            "source_path": str(cfg["paths"]["scientific_zarr_path"]),
            "target_transform": tuple(float(value) for value in target_transform),
            "geospatial_bounds": {
                "west": float(bounds_geo[0]),
                "south": float(bounds_geo[1]),
                "east": float(bounds_geo[2]),
                "north": float(bounds_geo[3]),
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    create_array(root, "x", data=target_x, chunks=(min(width, spatial_chunk),), dims=["x"], attrs={"units": "m"})
    create_array(root, "y", data=target_y, chunks=(min(height, spatial_chunk),), dims=["y"], attrs={"units": "m"})
    create_array(root, "time", data=np.asarray(dates, dtype="datetime64[ns]"), chunks=(min(len(dates), time_chunk),), dims=["time"])
    create_array(
        root,
        variable_name,
        shape=(len(dates), height, width),
        dtype=np.float32,
        chunks=(min(len(dates), time_chunk), spatial_chunk, spatial_chunk),
        dims=["time", "y", "x"],
        attrs={"long_name": str(cfg["dataset"]["variable_label"]), "units": str(cfg["dataset"]["units"])},
        fill_value=np.nan,
    )
    create_array(
        root,
        "quality_flag",
        shape=(len(dates),),
        dtype=np.uint8,
        chunks=(min(len(dates), time_chunk),),
        dims=["time"],
        attrs={"flag_values": [0], "flag_meanings": "generated"},
        fill_value=0,
    )
    write_lat_lon(root, target_x, target_y, str(cfg["dataset"]["viewer_crs"]), spatial_chunk, row_chunk)


def write_viewer_date(cfg: Dict[str, object], source_root, viewer_root, source_time_idx: int, viewer_time_idx: int) -> None:
    variable_name = str(cfg["dataset"]["variable_name"])
    source_x = np.asarray(source_root["x"][:], dtype=np.float64)
    source_y = np.asarray(source_root["y"][:], dtype=np.float64)
    src_transform = center_transform(source_x, source_y)
    dst_transform = Affine(*viewer_root.attrs["target_transform"])
    source_data = np.asarray(source_root[variable_name][source_time_idx, :, :], dtype=np.float32)
    dest_data = np.full(viewer_root[variable_name].shape[1:], np.nan, dtype=np.float32)
    reproject(
        source=source_data,
        destination=dest_data,
        src_transform=src_transform,
        src_crs=str(cfg["dataset"]["source_crs"]),
        dst_transform=dst_transform,
        dst_crs=str(cfg["dataset"]["viewer_crs"]),
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    viewer_root[variable_name][viewer_time_idx, :, :] = dest_data
    log(f"Wrote viewer date index {viewer_time_idx}")


def build(args) -> None:
    cfg = load_config(args.config)
    source_path = Path(str(cfg["paths"]["scientific_zarr_path"]))
    viewer_path = Path(str(cfg["paths"]["viewer_zarr_path"]))

    if args.dry_run:
        log(f"Would read scientific zarr: {source_path}")
        log(f"Would write viewer zarr: {viewer_path}")
        log("Viewer chunks: lfmc=(32, 256, 256), lat/lon=(256, 256)")
        return

    source_root = zarr.open_group(str(source_path), mode="r")
    source_dates = date_strings(source_root)
    dates = args.target_dates or source_dates
    missing = [date for date in dates if date not in source_dates]
    if missing:
        raise ValueError(f"Dates missing from scientific zarr: {', '.join(missing)}")

    if args.mode == "rebuild" or not viewer_path.exists():
        create_viewer_store(cfg, source_root, viewer_path, dates, overwrite=args.overwrite)
        viewer_root = zarr.open_group(str(viewer_path), mode="a")
        viewer_dates = dates
    else:
        viewer_root = zarr.open_group(str(viewer_path), mode="a")
        current_dates = date_strings(viewer_root)
        dates = [date for date in dates if date not in current_dates]
        if not dates:
            log("No new viewer dates to append.")
            return
        old_count = len(current_dates)
        new_count = old_count + len(dates)
        viewer_root["time"].resize((new_count,))
        viewer_root[str(cfg["dataset"]["variable_name"])].resize(
            (new_count, viewer_root[str(cfg["dataset"]["variable_name"])].shape[1], viewer_root[str(cfg["dataset"]["variable_name"])].shape[2])
        )
        viewer_root["quality_flag"].resize((new_count,))
        viewer_root["time"][old_count:new_count] = np.asarray(dates, dtype="datetime64[ns]")
        viewer_root["quality_flag"][old_count:new_count] = np.zeros(len(dates), dtype=np.uint8)
        viewer_dates = current_dates + dates

    source_index = {date: idx for idx, date in enumerate(source_dates)}
    viewer_index = {date: idx for idx, date in enumerate(viewer_dates)}
    for date in dates:
        write_viewer_date(cfg, source_root, viewer_root, source_index[date], viewer_index[date])
    viewer_root.attrs["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    zarr.consolidate_metadata(str(viewer_path))
    log(f"Viewer zarr ready: {viewer_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build or append the Rao S1-informed EPSG:3857 viewer zarr.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=["append", "rebuild"], default="append")
    parser.add_argument("--target-dates", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
