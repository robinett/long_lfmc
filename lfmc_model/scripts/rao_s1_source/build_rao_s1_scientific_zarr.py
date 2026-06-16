#!/usr/bin/env python3

import argparse
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import rasterio
import yaml
import zarr


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "rao_s1_source_config.yaml"
DATE_RE = re.compile(r"^lfmc_map_(\d{4}-\d{2}-(?:01|15))\.tif$")


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


def discover_map_files(lfmc_maps_dir: Path) -> Dict[str, Path]:
    files = {}
    for path in sorted(lfmc_maps_dir.glob("lfmc_map_*.tif")):
        match = DATE_RE.match(path.name)
        if match:
            files[match.group(1)] = path
    return files


def validate_target_dates(dates: Sequence[str], available: Dict[str, Path]) -> List[str]:
    missing = [date for date in dates if date not in available]
    if missing:
        raise FileNotFoundError(f"Missing Rao LFMC GeoTIFFs for dates: {', '.join(missing)}")
    return sorted(dates)


def coordinates_from_transform(transform, width: int, height: int):
    x_values = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y_values = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x_values, y_values


def write_lat_lon(root, x_values: np.ndarray, y_values: np.ndarray, spatial_chunk: int, row_chunk: int) -> None:
    height = y_values.size
    width = x_values.size
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
    for row_start in range(0, height, row_chunk):
        row_end = min(row_start + row_chunk, height)
        block_rows = row_end - row_start
        lat[row_start:row_end, :] = np.broadcast_to(
            y_values[row_start:row_end, None],
            (block_rows, width),
        )
        lon[row_start:row_end, :] = np.broadcast_to(
            x_values[None, :],
            (block_rows, width),
        )
        log(f"Wrote lat/lon rows {row_start}-{row_end - 1} of {height - 1}")


def create_store(cfg: Dict[str, object], zarr_path: Path, dates: Sequence[str], first_tif: Path, overwrite: bool) -> None:
    if zarr_path.exists():
        if not overwrite:
            raise FileExistsError(f"Zarr already exists: {zarr_path}. Use --overwrite to rebuild it.")
        log(f"Removing existing zarr {zarr_path}")
        shutil.rmtree(zarr_path)

    dataset_cfg = cfg["dataset"]
    chunks_cfg = cfg["chunks"]
    nodata = float(dataset_cfg.get("source_nodata_value", -9999))
    spatial_chunk = int(chunks_cfg.get("spatial", 256))
    row_chunk = int(chunks_cfg.get("latlon_row", spatial_chunk))
    time_chunk = int(chunks_cfg.get("scientific_time", 128))

    with rasterio.open(first_tif) as src:
        height = int(src.height)
        width = int(src.width)
        transform = src.transform
        crs = str(src.crs)
        bounds = src.bounds
    x_values, y_values = coordinates_from_transform(transform, width, height)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(
        str(zarr_path),
        mode="w",
        zarr_format=2,
        attributes={
            "title": str(dataset_cfg["dataset_label"]),
            "crs": crs,
            "source_crs": str(dataset_cfg["source_crs"]),
            "source_nodata_value": nodata,
            "missing_value": "NaN",
            "source": "Rao 2020 S1-informed LFMC GeoTIFF maps",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "geospatial_bounds": {
                "west": float(bounds.left),
                "east": float(bounds.right),
                "south": float(bounds.bottom),
                "north": float(bounds.top),
            },
        },
    )
    create_array(root, "x", data=x_values, chunks=(min(width, spatial_chunk),), dims=["x"], attrs={"units": "degrees_east"})
    create_array(root, "y", data=y_values, chunks=(min(height, spatial_chunk),), dims=["y"], attrs={"units": "degrees_north"})
    create_array(
        root,
        "time",
        data=np.asarray(dates, dtype="datetime64[ns]"),
        chunks=(min(len(dates), time_chunk),),
        dims=["time"],
    )
    create_array(
        root,
        str(dataset_cfg["variable_name"]),
        shape=(len(dates), height, width),
        dtype=np.float32,
        chunks=(min(len(dates), time_chunk), spatial_chunk, spatial_chunk),
        dims=["time", "y", "x"],
        attrs={"long_name": str(dataset_cfg["variable_label"]), "units": str(dataset_cfg["units"])},
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
    write_lat_lon(root, x_values, y_values, spatial_chunk, row_chunk)


def existing_dates(root) -> List[str]:
    if "time" not in root:
        return []
    values = np.asarray(root["time"][:]).astype("datetime64[D]")
    return [np.datetime_as_string(value, unit="D") for value in values]


def append_dates(cfg: Dict[str, object], zarr_path: Path, dates: Sequence[str], map_files: Dict[str, Path]) -> None:
    dataset_cfg = cfg["dataset"]
    chunks_cfg = cfg["chunks"]
    variable_name = str(dataset_cfg["variable_name"])
    nodata = float(dataset_cfg.get("source_nodata_value", -9999))
    spatial_chunk = int(chunks_cfg.get("spatial", 256))

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    current_dates = existing_dates(root)
    new_dates = [date for date in dates if date not in current_dates]
    if not new_dates:
        log("No new dates to append.")
        return

    old_count = len(current_dates)
    new_count = old_count + len(new_dates)
    height = int(root[variable_name].shape[1])
    width = int(root[variable_name].shape[2])
    root["time"].resize((new_count,))
    root[variable_name].resize((new_count, height, width))
    root["quality_flag"].resize((new_count,))

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    if root["time"].shape[0] != new_count:
        raise RuntimeError(f"Time resize failed for {zarr_path}: {root['time'].shape[0]} != {new_count}")
    if root[variable_name].shape != (new_count, height, width):
        raise RuntimeError(f"{variable_name} resize failed for {zarr_path}: {root[variable_name].shape} != {(new_count, height, width)}")
    if root["quality_flag"].shape[0] != new_count:
        raise RuntimeError(f"quality_flag resize failed for {zarr_path}: {root['quality_flag'].shape[0]} != {new_count}")

    root["time"][old_count:new_count] = np.asarray(new_dates, dtype="datetime64[ns]")
    root["quality_flag"][old_count:new_count] = np.zeros(len(new_dates), dtype=np.uint8)

    for offset, date_str in enumerate(new_dates):
        write_date(root[variable_name], old_count + offset, map_files[date_str], nodata, spatial_chunk)
    root.attrs["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def write_date(out_array, time_index: int, tif_path: Path, nodata: float, spatial_chunk: int) -> None:
    log(f"Writing {tif_path.name} to time index {time_index}")
    with rasterio.open(tif_path) as src:
        for row_start in range(0, src.height, spatial_chunk):
            row_end = min(row_start + spatial_chunk, src.height)
            window = rasterio.windows.Window(
                col_off=0,
                row_off=row_start,
                width=src.width,
                height=row_end - row_start,
            )
            data = src.read(1, window=window).astype(np.float32)
            data[data == nodata] = np.nan
            out_array[time_index, row_start:row_end, :] = data
            if row_start == 0 or row_end == src.height or row_start % (spatial_chunk * 10) == 0:
                log(f"{tif_path.name}: wrote rows {row_start}-{row_end - 1} of {src.height - 1}")


def build(args) -> None:
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]
    lfmc_maps_dir = Path(str(paths_cfg["lfmc_maps_dir"]))
    zarr_path = Path(str(paths_cfg["scientific_zarr_path"]))
    available = discover_map_files(lfmc_maps_dir)
    if not available:
        raise FileNotFoundError(f"No Rao LFMC maps found in {lfmc_maps_dir}")
    dates = validate_target_dates(args.target_dates or list(available.keys()), available)

    if args.dry_run:
        log(f"Would write scientific zarr: {zarr_path}")
        log(f"Would process {len(dates)} date(s): {dates[0]} to {dates[-1]}")
        log("Scientific chunks: lfmc=(128, 256, 256), lat/lon=(256, 256)")
        return

    if args.mode == "rebuild" or not zarr_path.exists():
        create_store(cfg, zarr_path, dates, available[dates[0]], overwrite=args.overwrite)
        root = zarr.open_group(str(zarr_path), mode="a")
        for time_index, date_str in enumerate(dates):
            write_date(
                root[str(cfg["dataset"]["variable_name"])],
                time_index,
                available[date_str],
                float(cfg["dataset"].get("source_nodata_value", -9999)),
                int(cfg["chunks"].get("spatial", 256)),
            )
        root.attrs["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        append_dates(cfg, zarr_path, dates, available)
    zarr.consolidate_metadata(str(zarr_path))
    log(f"Scientific zarr ready: {zarr_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build or append the Rao S1-informed scientific LFMC zarr.")
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
