#!/usr/bin/env python3

import argparse
import gc
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SHARED_DIR = os.path.join(REPO_ROOT, "data_processing", "shared")
CONVERT_DIR = os.path.join(REPO_ROOT, "data_processing", "convert_to_zarr")

for extra_path in [SHARED_DIR, CONVERT_DIR]:
    if extra_path not in sys.path:
        sys.path.append(extra_path)

import plotting as plot
from zarr_build_utils import DEFAULT_COMP, consolidate


MODE_CHOICES = [
    "init-store",
    "build-standard-var",
    "finalize-standard",
    "init-anomaly",
    "build-anomaly-var",
    "finalize-anomaly",
]

STANDARD_CBAR_LABELS = {
    "tmax": "tmax",
    "vpd": "vpd",
    "prcp": "prcp",
    "srad": "srad",
    "swe": "swe",
}

ANOMALY_CBAR_LABELS = {
    "tmax_daily_anom": "tmax anomaly",
    "vpd_daily_anom": "vpd anomaly",
    "prcp_rolling30_anom": "prcp rolling30 anomaly",
    "srad_daily_anom": "srad anomaly",
    "swe_daily_anom": "swe anomaly",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a combined Daymet standard-variable and anomaly zarr store "
            "using init/worker/finalize modes."
        )
    )
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "configs.yaml"),
        help="Path to Daymet derived-products config YAML.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODE_CHOICES,
        help="Workflow mode to run.",
    )
    parser.add_argument(
        "--var",
        default=None,
        help="Variable name for worker modes.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on the configured smoke-test subset instead of the full archive.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based spatial shard index for sharded anomaly workers.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of spatial shards for sharded anomaly workers.",
    )
    args = parser.parse_args()
    if args.mode in {"build-standard-var", "build-anomaly-var"} and not args.var:
        parser.error("--var is required for worker modes")
    if args.mode not in {"build-standard-var", "build-anomaly-var"} and args.var is not None:
        parser.error("--var is only valid for worker modes")
    if (args.shard_index is None) != (args.num_shards is None):
        parser.error("--shard-index and --num-shards must be provided together")
    if args.shard_index is not None:
        if args.mode != "build-anomaly-var":
            parser.error("--shard-index/--num-shards are only valid for build-anomaly-var")
        if args.num_shards < 1:
            parser.error("--num-shards must be >= 1")
        if not 0 <= args.shard_index < args.num_shards:
            parser.error("--shard-index must be in [0, --num-shards)")
    return args


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def subset_extent_from_da(da_var: xr.DataArray) -> List[float]:
    x_vals = da_var["x"].values
    y_vals = da_var["y"].values
    x_res = float(np.abs(x_vals[1] - x_vals[0]))
    y_res = float(np.abs(y_vals[1] - y_vals[0]))
    return [
        float(np.min(x_vals) - x_res / 2.0),
        float(np.max(x_vals) + x_res / 2.0),
        float(np.min(y_vals) - y_res / 2.0),
        float(np.max(y_vals) + y_res / 2.0),
    ]


def open_daymet_store(zarr_path: str) -> xr.Dataset:
    print(f"Opening Daymet store: {zarr_path}")
    return xr.open_zarr(zarr_path, consolidated=False)


def select_store_var(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    da_var = ds["data"].sel(variable=var_name).drop_vars("variable", errors="ignore")
    da_var.name = var_name
    return da_var


def select_climatology_var(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    da_var = ds["data"].sel(variable=var_name).drop_vars("variable", errors="ignore")
    da_var.name = var_name
    return da_var


def month_day_coord(time_coord: xr.DataArray) -> xr.DataArray:
    month_day = time_coord.dt.month * 100 + time_coord.dt.day
    month_day.name = "month_day"
    return month_day


def month_day_values(time_values: np.ndarray) -> np.ndarray:
    dt_index = pd.DatetimeIndex(pd.to_datetime(time_values))
    return (dt_index.month * 100 + dt_index.day).to_numpy(dtype=np.int32)


def saturation_vapor_pressure_pa(temp_c: np.ndarray) -> np.ndarray:
    return 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def compute_vpd_numpy(
    tmax: np.ndarray,
    vp: np.ndarray,
    clip_min: float,
) -> np.ndarray:
    vpd = saturation_vapor_pressure_pa(tmax.astype(np.float32)) - vp.astype(np.float32)
    np.maximum(vpd, clip_min, out=vpd)
    return vpd.astype(np.float32, copy=False)


def stack_named_arrays(data_arrays: Sequence[xr.DataArray], write_chunks: Dict[str, int]) -> xr.DataArray:
    arr = xr.concat(
        [da_var.expand_dims(variable=[str(da_var.name)]) for da_var in data_arrays],
        dim="variable",
    ).transpose("time", "variable", "y", "x")
    target_chunks = {
        "time": int(write_chunks["time"]),
        "variable": min(int(write_chunks["variable"]), int(arr.sizes["variable"])),
        "y": int(write_chunks["y"]),
        "x": int(write_chunks["x"]),
    }
    return arr.chunk(target_chunks)


def write_stacked_store(
    arr: xr.DataArray,
    out_path: str,
    overwrite_existing: bool,
) -> None:
    if overwrite_existing and os.path.exists(out_path):
        print(f"Removing existing store before rewrite: {out_path}")
        shutil.rmtree(out_path)
    print(f"Writing stacked zarr store: {out_path}")
    arr.to_dataset(name="data").to_zarr(
        out_path,
        mode="w",
        consolidated=False,
        zarr_format=2,
        compute=True,
        safe_chunks=False,
    )
    consolidate(Path(out_path))


def get_run_paths(config: Dict, smoke_test: bool) -> Dict[str, str]:
    if smoke_test:
        return {
            "source_archive_zarr_path": config["smoke_test"]["source_archive_zarr_path"],
            "final_output_zarr_path": config["smoke_test"]["final_output_zarr_path"],
            "climatology_output_zarr_path": config["smoke_test"]["climatology_output_zarr_path"],
            "plots_dir": config["smoke_test"]["plots_dir"],
            "coord_dir": config["smoke_test"]["coord_dir"],
        }
    return {
        "source_archive_zarr_path": config["paths"]["source_archive_zarr_path"],
        "final_output_zarr_path": config["paths"]["final_output_zarr_path"],
        "climatology_output_zarr_path": config["paths"]["climatology_output_zarr_path"],
        "plots_dir": config["paths"]["plots_dir"],
        "coord_dir": config["paths"]["coord_dir"],
    }


def standard_var_names(config: Dict) -> List[str]:
    return list(config["processing"]["standard_variable_order"])


def anomaly_var_names(config: Dict) -> List[str]:
    return list(config["processing"]["anomaly_variable_order"])


def climatology_var_names(config: Dict) -> List[str]:
    return list(config["processing"]["climatology_variable_order"])


def output_var_names(config: Dict) -> List[str]:
    return standard_var_names(config) + anomaly_var_names(config)


def standard_source_lookup(var_name: str) -> str | None:
    if var_name == "vpd":
        return None
    return var_name


def anomaly_source_lookup(var_name: str) -> str:
    lookup = {
        "tmax_daily_anom": "tmax",
        "vpd_daily_anom": "vpd",
        "prcp_rolling30_anom": "prcp",
        "srad_daily_anom": "srad",
        "swe_daily_anom": "swe",
    }
    return lookup[var_name]


def anomaly_climatology_lookup(var_name: str) -> str:
    lookup = {
        "tmax_daily_anom": "tmax_daily_clim",
        "vpd_daily_anom": "vpd_daily_clim",
        "prcp_rolling30_anom": "prcp_rolling30_daily_clim",
        "srad_daily_anom": "srad_daily_clim",
        "swe_daily_anom": "swe_daily_clim",
    }
    return lookup[var_name]


def smoke_source_indexers(config: Dict) -> Dict[str, slice]:
    smoke = config["smoke_test"]
    return {
        "time": slice(smoke["time_start"], smoke["time_end"]),
        "x": slice(int(smoke["x_slice"]["start"]), int(smoke["x_slice"]["stop"])),
        "y": slice(int(smoke["y_slice"]["start"]), int(smoke["y_slice"]["stop"])),
    }


def iter_spatial_slices(ny: int, nx: int, chunk_y: int, chunk_x: int):
    total = int(np.ceil(ny / chunk_y) * np.ceil(nx / chunk_x))
    counter = 0
    for y_start in range(0, ny, chunk_y):
        y_stop = min(y_start + chunk_y, ny)
        for x_start in range(0, nx, chunk_x):
            x_stop = min(x_start + chunk_x, nx)
            counter += 1
            yield counter, total, slice(y_start, y_stop), slice(x_start, x_stop)


def iter_spatial_slices_for_assignment(
    ny: int,
    nx: int,
    chunk_y: int,
    chunk_x: int,
    shard_index: int | None,
    num_shards: int | None,
):
    for chunk_index, total_chunks, y_slice, x_slice in iter_spatial_slices(ny, nx, chunk_y, chunk_x):
        if shard_index is not None and ((chunk_index - 1) % num_shards != shard_index):
            continue
        yield chunk_index, total_chunks, y_slice, x_slice


def print_chunk_progress(prefix: str, chunk_index: int, total_chunks: int, y_slice: slice, x_slice: slice) -> None:
    pct = 100.0 * chunk_index / total_chunks
    print(
        f"{prefix}: chunk {chunk_index}/{total_chunks} ({pct:.1f}%) "
        f"y={y_slice.start}:{y_slice.stop} x={x_slice.start}:{x_slice.stop}"
    )


def marker_dir(coord_dir: str, phase_name: str) -> Path:
    return Path(coord_dir) / phase_name


def marker_path(coord_dir: str, phase_name: str, item_name: str) -> Path:
    return marker_dir(coord_dir, phase_name) / f"{item_name}.json"


def shard_marker_item_name(var_name: str, shard_index: int, num_shards: int) -> str:
    return f"{var_name}__shard_{shard_index:02d}_of_{num_shards:02d}"


def clear_path(path_str: str) -> None:
    path = Path(path_str)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def write_marker(coord_dir: str, phase_name: str, item_name: str, payload: Dict) -> None:
    phase_dir = marker_dir(coord_dir, phase_name)
    phase_dir.mkdir(parents=True, exist_ok=True)
    marker_file = marker_path(coord_dir, phase_name, item_name)
    with open(marker_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote marker: {marker_file}")


def require_markers(coord_dir: str, phase_name: str, expected_items: Sequence[str]) -> None:
    missing = [
        item_name
        for item_name in expected_items
        if not marker_path(coord_dir, phase_name, item_name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {phase_name} markers for {missing} under {marker_dir(coord_dir, phase_name)}"
        )


def require_anomaly_markers(coord_dir: str, expected_items: Sequence[str]) -> None:
    phase_dir = marker_dir(coord_dir, "anomaly")
    missing = []
    for item_name in expected_items:
        if marker_path(coord_dir, "anomaly", item_name).exists():
            continue
        shard_files = sorted(phase_dir.glob(f"{item_name}__shard_*_of_*.json"))
        if not shard_files:
            missing.append(item_name)
            continue
        shard_indices = set()
        shard_totals = set()
        for shard_file in shard_files:
            shard_stem = shard_file.stem
            try:
                shard_suffix = shard_stem.split("__shard_", 1)[1]
                shard_idx_str, shard_total_str = shard_suffix.split("_of_")
                shard_indices.add(int(shard_idx_str))
                shard_totals.add(int(shard_total_str))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Malformed anomaly shard marker: {shard_file.name}") from exc
        if len(shard_totals) != 1:
            raise ValueError(f"Inconsistent shard totals for {item_name}: {sorted(shard_totals)}")
        shard_total = next(iter(shard_totals))
        if shard_indices != set(range(shard_total)):
            missing.append(item_name)
    if missing:
        raise FileNotFoundError(
            f"Missing anomaly completion markers for {missing} under {phase_dir}"
        )


def initialize_empty_stacked_store(
    source_ds: xr.Dataset,
    out_path: str,
    variable_names: Sequence[str],
    write_chunks: Dict[str, int],
    overwrite_existing: bool,
) -> None:
    if overwrite_existing and os.path.exists(out_path):
        print(f"Removing existing store before rewrite: {out_path}")
        shutil.rmtree(out_path)

    ensure_dirs([os.path.dirname(out_path)])
    data_shape = (
        int(source_ds.sizes["time"]),
        len(variable_names),
        int(source_ds.sizes["y"]),
        int(source_ds.sizes["x"]),
    )
    chunk_shape = (
        int(write_chunks["time"]),
        min(int(write_chunks["variable"]), len(variable_names)),
        int(write_chunks["y"]),
        int(write_chunks["x"]),
    )
    template_data = da.empty(data_shape, chunks=chunk_shape, dtype=np.float32)
    template = xr.Dataset(
        {
            "data": (("time", "variable", "y", "x"), template_data),
        },
        coords={
            "time": source_ds["time"],
            "variable": np.asarray(variable_names, dtype=object),
            "y": source_ds["y"],
            "x": source_ds["x"],
            "lat": source_ds["lat"],
            "lon": source_ds["lon"],
            "lambert_conformal_conic": source_ds["lambert_conformal_conic"],
        },
        attrs=dict(source_ds.attrs),
    )
    print(f"Initializing zarr store metadata: {out_path}")
    delayed_write = template.to_zarr(
        out_path,
        mode="w",
        consolidated=False,
        zarr_format=2,
        compute=False,
        encoding={"data": {"compressor": DEFAULT_COMP, "chunks": chunk_shape}},
    )
    del delayed_write
    consolidate(Path(out_path))


def initialize_empty_climatology_store(
    reference_ds: xr.Dataset,
    out_path: str,
    variable_names: Sequence[str],
    month_day_vals: np.ndarray,
    write_chunks: Dict[str, int],
    overwrite_existing: bool,
) -> None:
    if overwrite_existing and os.path.exists(out_path):
        print(f"Removing existing climatology store before rewrite: {out_path}")
        shutil.rmtree(out_path)

    ensure_dirs([os.path.dirname(out_path)])
    data_shape = (
        int(len(month_day_vals)),
        len(variable_names),
        int(reference_ds.sizes["y"]),
        int(reference_ds.sizes["x"]),
    )
    chunk_shape = (
        min(int(write_chunks["time"]), int(len(month_day_vals))),
        min(int(write_chunks["variable"]), len(variable_names)),
        int(write_chunks["y"]),
        int(write_chunks["x"]),
    )
    template_data = da.empty(data_shape, chunks=chunk_shape, dtype=np.float32)
    template = xr.Dataset(
        {
            "data": (("month_day", "variable", "y", "x"), template_data),
        },
        coords={
            "month_day": np.asarray(month_day_vals, dtype=np.int32),
            "variable": np.asarray(variable_names, dtype=object),
            "y": reference_ds["y"],
            "x": reference_ds["x"],
            "lat": reference_ds["lat"],
            "lon": reference_ds["lon"],
            "lambert_conformal_conic": reference_ds["lambert_conformal_conic"],
        },
        attrs=dict(reference_ds.attrs),
    )
    print(f"Initializing climatology zarr store metadata: {out_path}")
    delayed_write = template.to_zarr(
        out_path,
        mode="w",
        consolidated=False,
        zarr_format=2,
        compute=False,
        encoding={"data": {"compressor": DEFAULT_COMP, "chunks": chunk_shape}},
    )
    del delayed_write
    consolidate(Path(out_path))


def trailing_window_mean_numpy(arr: np.ndarray, window_days: int) -> np.ndarray:
    rolling = np.full(arr.shape, np.nan, dtype=np.float32)
    if arr.shape[0] < window_days:
        return rolling

    valid = np.isfinite(arr)
    arr_filled = np.where(valid, arr, 0.0).astype(np.float32, copy=False)
    valid_counts = valid.astype(np.uint16)

    csum = np.cumsum(arr_filled, axis=0, dtype=np.float32)
    ccount = np.cumsum(valid_counts, axis=0, dtype=np.uint16)

    window_sum = csum[window_days - 1:].copy()
    window_count = ccount[window_days - 1:].copy()
    if window_days > 1:
        window_sum[1:] -= csum[:-window_days]
        window_count[1:] -= ccount[:-window_days]

    with np.errstate(invalid="ignore", divide="ignore"):
        window_mean = window_sum / window_count
    rolling[window_days - 1:] = np.where(window_count == window_days, window_mean, np.nan).astype(np.float32)
    return rolling


def circular_centered_window_mean_numpy(arr: np.ndarray, window_days: int) -> np.ndarray:
    if window_days <= 1:
        return arr.astype(np.float32, copy=True)
    if arr.shape[0] == 0:
        return arr.astype(np.float32, copy=True)
    if window_days > arr.shape[0]:
        raise ValueError(
            f"Centered climatology smoothing window ({window_days}) exceeds climatology axis "
            f"length ({arr.shape[0]})"
        )

    left_pad = (window_days - 1) // 2
    right_pad = window_days // 2
    padded = np.concatenate([arr[-left_pad:], arr, arr[:right_pad]], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smoothed = np.empty_like(arr, dtype=np.float32)
        for idx in range(arr.shape[0]):
            smoothed[idx] = np.nanmean(padded[idx : idx + window_days], axis=0).astype(np.float32)
    return smoothed


def calendar_day_climatology_and_anomaly_numpy(
    arr: np.ndarray,
    md_values: np.ndarray,
    unique_md_values: np.ndarray,
    climatology_smoothing_window_days: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    climatology = np.full((len(unique_md_values),) + arr.shape[1:], np.nan, dtype=np.float32)
    anomaly = np.full(arr.shape, np.nan, dtype=np.float32)
    for idx, month_day in enumerate(unique_md_values):
        mask = md_values == month_day
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            climatology_chunk = np.nanmean(arr[mask], axis=0).astype(np.float32)
        climatology[idx] = climatology_chunk
    climatology = circular_centered_window_mean_numpy(
        climatology,
        int(climatology_smoothing_window_days),
    )
    for idx, month_day in enumerate(unique_md_values):
        mask = md_values == month_day
        anomaly[mask] = arr[mask] - climatology[idx]
    return climatology, anomaly


def compute_anomaly_chunk(
    base_chunk: np.ndarray,
    var_name: str,
    config: Dict,
    md_values: np.ndarray,
    unique_md_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    climatology_smoothing_window_days = int(
        config["processing"].get("climatology_smoothing_window_days", 1)
    )
    if var_name == "prcp_rolling30_anom":
        rolling = trailing_window_mean_numpy(
            base_chunk.astype(np.float32, copy=False),
            int(config["processing"]["prcp_rolling_window_days"]),
        )
        return calendar_day_climatology_and_anomaly_numpy(
            rolling,
            md_values,
            unique_md_values,
            climatology_smoothing_window_days=climatology_smoothing_window_days,
        )
    return calendar_day_climatology_and_anomaly_numpy(
        base_chunk.astype(np.float32, copy=False),
        md_values,
        unique_md_values,
        climatology_smoothing_window_days=climatology_smoothing_window_days,
    )


def get_source_store_path(args, config: Dict, run_paths: Dict[str, str]) -> str:
    return config["paths"]["source_archive_zarr_path"]


def init_store(args, config: Dict) -> None:
    run_paths = get_run_paths(config, smoke_test=args.smoke_test)
    ensure_dirs([run_paths["plots_dir"], os.path.dirname(run_paths["final_output_zarr_path"])])

    if bool(config["processing"]["overwrite_coord_dir"]):
        clear_path(run_paths["coord_dir"])
    ensure_dirs([run_paths["coord_dir"]])

    source_archive = open_daymet_store(config["paths"]["source_archive_zarr_path"])
    if args.smoke_test:
        smoke_indexers = smoke_source_indexers(config)
        source_ds = source_archive.sel(time=smoke_indexers["time"]).isel(
            x=smoke_indexers["x"],
            y=smoke_indexers["y"],
        ).load()
        source_archive_path = config["paths"]["source_archive_zarr_path"]
    else:
        source_ds = source_archive
        source_archive_path = run_paths["source_archive_zarr_path"]
    initialize_empty_stacked_store(
        source_ds,
        run_paths["final_output_zarr_path"],
        output_var_names(config),
        config["processing"]["write_chunks"],
        overwrite_existing=bool(config["processing"]["overwrite_final_store"]),
    )
    source_ds.close()
    if source_archive is not source_ds:
        source_archive.close()

    write_marker(
        run_paths["coord_dir"],
        "workflow",
        "init_store",
        {
            "mode": "init-store",
            "smoke_test": args.smoke_test,
            "source_archive_zarr_path": source_archive_path,
            "final_output_zarr_path": run_paths["final_output_zarr_path"],
            "variables": output_var_names(config),
        },
    )
    print(f"Initialized final Daymet store: {run_paths['final_output_zarr_path']}")


def build_standard_var(args, config: Dict) -> None:
    var_name = args.var
    allowed = standard_var_names(config)
    if var_name not in allowed:
        raise ValueError(f"Unknown standard variable {var_name}; expected one of {allowed}")

    run_paths = get_run_paths(config, smoke_test=args.smoke_test)
    source_store_path = get_source_store_path(args, config, run_paths)
    source_root = zarr.open_group(source_store_path, mode="r")
    final_root = zarr.open_group(run_paths["final_output_zarr_path"], mode="a")
    source_data = source_root["data"]
    final_data = final_root["data"]
    source_var_names = [str(val) for val in source_root["variable"][:]]
    final_var_names = [str(val) for val in final_root["variable"][:]]
    dst_idx = final_var_names.index(var_name)
    chunk_y = int(final_data.chunks[2])
    chunk_x = int(final_data.chunks[3])
    ny = int(final_data.shape[2])
    nx = int(final_data.shape[3])
    source_x_offset = 0
    source_y_offset = 0
    if args.smoke_test:
        smoke_indexers = smoke_source_indexers(config)
        source_x_offset = smoke_indexers["x"].start
        source_y_offset = smoke_indexers["y"].start

    print(
        f"Building standard Daymet variable {var_name} from {source_store_path} "
        f"with tile size y={chunk_y}, x={chunk_x}"
    )
    if var_name == "vpd":
        tmax_idx = source_var_names.index("tmax")
        vp_idx = source_var_names.index("vp")
        clip_min = float(config["processing"]["vpd_clip_min_pa"])
    else:
        src_idx = source_var_names.index(standard_source_lookup(var_name))

    for chunk_index, total_chunks, y_slice, x_slice in iter_spatial_slices(ny, nx, chunk_y, chunk_x):
        print_chunk_progress(var_name, chunk_index, total_chunks, y_slice, x_slice)
        src_y_slice = slice(source_y_offset + y_slice.start, source_y_offset + y_slice.stop)
        src_x_slice = slice(source_x_offset + x_slice.start, source_x_offset + x_slice.stop)
        if var_name == "vpd":
            tmax_chunk = np.asarray(
                source_data[:, tmax_idx, src_y_slice, src_x_slice],
                dtype=np.float32,
            )
            vp_chunk = np.asarray(
                source_data[:, vp_idx, src_y_slice, src_x_slice],
                dtype=np.float32,
            )
            out_chunk = compute_vpd_numpy(tmax_chunk, vp_chunk, clip_min=clip_min)
            del tmax_chunk, vp_chunk
        else:
            out_chunk = np.asarray(
                source_data[:, src_idx, src_y_slice, src_x_slice],
                dtype=np.float32,
            )
        final_data[:, dst_idx, y_slice, x_slice] = out_chunk
        del out_chunk
        gc.collect()

    write_marker(
        run_paths["coord_dir"],
        "standard",
        var_name,
        {
            "mode": "build-standard-var",
            "var": var_name,
            "store_path": run_paths["final_output_zarr_path"],
        },
    )
    print(f"Finished standard variable {var_name}")


def finalize_standard(args, config: Dict) -> None:
    run_paths = get_run_paths(config, smoke_test=args.smoke_test)
    required = standard_var_names(config)
    require_markers(run_paths["coord_dir"], "standard", required)
    consolidate(Path(run_paths["final_output_zarr_path"]))

    final_ds = open_daymet_store(run_paths["final_output_zarr_path"])
    standard_plots_dir = os.path.join(run_paths["plots_dir"], "standard")
    plot_standard_maps(final_ds, standard_plots_dir, config)
    plot_standard_timeseries(final_ds, standard_plots_dir, config)
    final_ds.close()

    write_marker(
        run_paths["coord_dir"],
        "workflow",
        "finalize_standard",
        {
            "mode": "finalize-standard",
            "vars": required,
            "plots_dir": standard_plots_dir,
            "store_path": run_paths["final_output_zarr_path"],
        },
    )
    print(f"Finished standard finalize for {run_paths['final_output_zarr_path']}")


def init_anomaly(args, config: Dict) -> None:
    run_paths = get_run_paths(config, smoke_test=args.smoke_test)
    require_markers(run_paths["coord_dir"], "standard", standard_var_names(config))
    clear_path(str(marker_dir(run_paths["coord_dir"], "anomaly")))
    final_ds = open_daymet_store(run_paths["final_output_zarr_path"])
    unique_md_values = np.unique(month_day_values(final_ds["time"].values))
    initialize_empty_climatology_store(
        final_ds,
        run_paths["climatology_output_zarr_path"],
        climatology_var_names(config),
        unique_md_values,
        config["processing"]["write_chunks"],
        overwrite_existing=bool(config["processing"]["overwrite_climatology_store"]),
    )
    final_ds.close()
    write_marker(
        run_paths["coord_dir"],
        "workflow",
        "init_anomaly",
        {
            "mode": "init-anomaly",
            "vars": anomaly_var_names(config),
            "store_path": run_paths["final_output_zarr_path"],
            "climatology_store_path": run_paths["climatology_output_zarr_path"],
            "climatology_smoothing_window_days": int(
                config["processing"].get("climatology_smoothing_window_days", 1)
            ),
        },
    )
    print("Initialized anomaly phase markers")


def build_anomaly_var(args, config: Dict) -> None:
    var_name = args.var
    allowed = anomaly_var_names(config)
    if var_name not in allowed:
        raise ValueError(f"Unknown anomaly variable {var_name}; expected one of {allowed}")

    run_paths = get_run_paths(config, smoke_test=args.smoke_test)
    final_ds = open_daymet_store(run_paths["final_output_zarr_path"])
    md_values = month_day_values(final_ds["time"].values)
    final_ds.close()

    final_root = zarr.open_group(run_paths["final_output_zarr_path"], mode="a")
    climatology_root = zarr.open_group(run_paths["climatology_output_zarr_path"], mode="a")
    final_data = final_root["data"]
    climatology_data = climatology_root["data"]
    final_var_names = [str(val) for val in final_root["variable"][:]]
    climatology_var_list = [str(val) for val in climatology_root["variable"][:]]
    unique_md_values = np.asarray(climatology_root["month_day"][:], dtype=np.int32)
    src_idx = final_var_names.index(anomaly_source_lookup(var_name))
    dst_idx = final_var_names.index(var_name)
    climatology_idx = climatology_var_list.index(anomaly_climatology_lookup(var_name))
    chunk_y = int(final_data.chunks[2])
    chunk_x = int(final_data.chunks[3])
    ny = int(final_data.shape[2])
    nx = int(final_data.shape[3])

    print(
        f"Building anomaly variable {var_name} from {anomaly_source_lookup(var_name)} "
        f"with tile size y={chunk_y}, x={chunk_x}"
    )
    progress_name = var_name
    if args.shard_index is not None:
        progress_name = f"{var_name} shard {args.shard_index + 1}/{args.num_shards}"
    for chunk_index, total_chunks, y_slice, x_slice in iter_spatial_slices_for_assignment(
        ny,
        nx,
        chunk_y,
        chunk_x,
        args.shard_index,
        args.num_shards,
    ):
        print_chunk_progress(progress_name, chunk_index, total_chunks, y_slice, x_slice)
        base_chunk = np.asarray(
            final_data[:, src_idx, y_slice, x_slice],
            dtype=np.float32,
        )
        climatology_chunk, anomaly_chunk = compute_anomaly_chunk(
            base_chunk,
            var_name,
            config,
            md_values,
            unique_md_values,
        )
        climatology_data[:, climatology_idx, y_slice, x_slice] = climatology_chunk
        final_data[:, dst_idx, y_slice, x_slice] = anomaly_chunk
        del base_chunk, climatology_chunk, anomaly_chunk
        gc.collect()

    marker_name = var_name
    marker_payload = {
        "mode": "build-anomaly-var",
        "var": var_name,
        "store_path": run_paths["final_output_zarr_path"],
        "climatology_store_path": run_paths["climatology_output_zarr_path"],
    }
    if args.shard_index is not None:
        marker_name = shard_marker_item_name(var_name, args.shard_index, args.num_shards)
        marker_payload["shard_index"] = args.shard_index
        marker_payload["num_shards"] = args.num_shards
    write_marker(
        run_paths["coord_dir"],
        "anomaly",
        marker_name,
        marker_payload,
    )
    if args.shard_index is not None:
        print(f"Finished anomaly variable {var_name} shard {args.shard_index + 1}/{args.num_shards}")
    else:
        print(f"Finished anomaly variable {var_name}")


def finalize_anomaly(args, config: Dict) -> None:
    run_paths = get_run_paths(config, smoke_test=args.smoke_test)
    required = anomaly_var_names(config)
    require_anomaly_markers(run_paths["coord_dir"], required)
    consolidate(Path(run_paths["final_output_zarr_path"]))
    consolidate(Path(run_paths["climatology_output_zarr_path"]))

    final_ds = open_daymet_store(run_paths["final_output_zarr_path"])
    clim_ds = open_daymet_store(run_paths["climatology_output_zarr_path"])
    anomaly_plots_dir = os.path.join(run_paths["plots_dir"], "anomaly")
    plot_anomaly_maps(final_ds, anomaly_plots_dir, config)
    plot_anomaly_timeseries(final_ds, clim_ds, anomaly_plots_dir, config)
    final_ds.close()
    clim_ds.close()

    write_marker(
        run_paths["coord_dir"],
        "workflow",
        "finalize_anomaly",
        {
            "mode": "finalize-anomaly",
            "vars": required,
            "plots_dir": anomaly_plots_dir,
            "store_path": run_paths["final_output_zarr_path"],
            "climatology_store_path": run_paths["climatology_output_zarr_path"],
        },
    )
    print("Finished anomaly finalize")
    print(f"Final store: {run_paths['final_output_zarr_path']}")
    print(f"Plots: {run_paths['plots_dir']}")


def plot_standard_maps(
    final_ds: xr.Dataset,
    plots_dir: str,
    config: Dict,
) -> None:
    ensure_dirs([plots_dir])
    extent = subset_extent_from_da(select_store_var(final_ds, standard_var_names(config)[0]))
    map_dates = [pd.Timestamp(val) for val in config["qc"]["map_dates"]]

    for map_date in map_dates:
        for var_name in standard_var_names(config):
            da_var = select_store_var(final_ds, var_name).sel(time=map_date)
            out_path = os.path.join(
                plots_dir,
                f"{map_date.strftime('%Y%m%d')}_{var_name}.png",
            )
            plot.plot_from_xarray(
                "da",
                da_var,
                var_name,
                "EPSG:5070",
                "EPSG:5070",
                out_path,
                extent=extent,
                extent_crs="EPSG:5070",
                title=f"{var_name} on {map_date.date()}",
                cbar_label=STANDARD_CBAR_LABELS.get(var_name, var_name),
            )


def plot_standard_timeseries(
    final_ds: xr.Dataset,
    plots_dir: str,
    config: Dict,
) -> None:
    ensure_dirs([plots_dir])
    plot_year = int(config["qc"]["plot_year"])
    year_slice = slice(f"{plot_year}-01-01", f"{plot_year}-12-31")
    point_specs = config["qc"]["sample_points"]

    for point in point_specs:
        label = point["label"]
        x_idx = int(point["x_index"])
        y_idx = int(point["y_index"])
        for var_name in standard_var_names(config):
            da_var = (
                select_store_var(final_ds, var_name)
                .isel(x=x_idx, y=y_idx)
                .sel(time=year_slice)
                .compute()
            )
            plot.plot_timeseries_lines(
                [
                    {
                        "x": pd.to_datetime(da_var["time"].values),
                        "y": da_var.values,
                        "label": var_name,
                    }
                ],
                os.path.join(plots_dir, f"{label}_{var_name}_timeseries.png"),
                title=f"{label}: {var_name} ({plot_year})",
                ylabel=STANDARD_CBAR_LABELS.get(var_name, var_name),
            )


def plot_anomaly_maps(
    final_ds: xr.Dataset,
    plots_dir: str,
    config: Dict,
) -> None:
    ensure_dirs([plots_dir])
    extent = subset_extent_from_da(select_store_var(final_ds, standard_var_names(config)[0]))
    map_dates = [pd.Timestamp(val) for val in config["qc"]["map_dates"]]

    for map_date in map_dates:
        for var_name in anomaly_var_names(config):
            da_var = select_store_var(final_ds, var_name).sel(time=map_date)
            finite_vals = np.asarray(da_var.values, dtype=np.float32)
            max_abs = float(np.nanmax(np.abs(finite_vals))) if np.isfinite(finite_vals).any() else 1.0
            out_path = os.path.join(
                plots_dir,
                f"{map_date.strftime('%Y%m%d')}_{var_name}.png",
            )
            plot.plot_from_xarray(
                "da",
                da_var,
                var_name,
                "EPSG:5070",
                "EPSG:5070",
                out_path,
                cmap="RdBu_r",
                extent=extent,
                extent_crs="EPSG:5070",
                title=f"{var_name} on {map_date.date()}",
                cbar_label=ANOMALY_CBAR_LABELS.get(var_name, var_name),
                vmin=-max_abs,
                vmax=max_abs,
            )


def plot_anomaly_timeseries(
    final_ds: xr.Dataset,
    clim_ds: xr.Dataset,
    plots_dir: str,
    config: Dict,
) -> None:
    ensure_dirs([plots_dir])
    plot_year = int(config["qc"]["plot_year"])
    year_slice = slice(f"{plot_year}-01-01", f"{plot_year}-12-31")
    point_specs = config["qc"]["sample_points"]
    window_days = int(config["processing"]["prcp_rolling_window_days"])

    for point in point_specs:
        label = point["label"]
        x_idx = int(point["x_index"])
        y_idx = int(point["y_index"])
        for var_name in anomaly_var_names(config):
            source_name = anomaly_source_lookup(var_name)
            clim_name = anomaly_climatology_lookup(var_name)

            if var_name == "prcp_rolling30_anom":
                source_full = select_store_var(final_ds, source_name).isel(x=x_idx, y=y_idx).compute()
                source_full = source_full.rolling(
                    time=window_days,
                    min_periods=window_days,
                ).mean()
                source_full.name = "prcp_rolling30"
            else:
                source_full = select_store_var(final_ds, source_name).isel(x=x_idx, y=y_idx).compute()

            source_here = source_full.sel(time=year_slice).compute()
            md_here = month_day_coord(source_here["time"])
            clim_here = (
                select_climatology_var(clim_ds, clim_name)
                .isel(x=x_idx, y=y_idx)
                .sel(month_day=md_here)
                .compute()
            )
            anom_here = (
                select_store_var(final_ds, var_name)
                .isel(x=x_idx, y=y_idx)
                .sel(time=year_slice)
                .compute()
            )

            plot.plot_timeseries_lines(
                [
                    {
                        "x": pd.to_datetime(source_here["time"].values),
                        "y": source_here.values,
                        "label": source_full.name,
                    },
                    {
                        "x": pd.to_datetime(source_here["time"].values),
                        "y": clim_here.values,
                        "label": clim_name,
                    },
                    {
                        "x": pd.to_datetime(anom_here["time"].values),
                        "y": anom_here.values,
                        "label": var_name,
                    },
                ],
                os.path.join(plots_dir, f"{label}_{var_name}_timeseries.png"),
                title=f"{label}: {var_name} ({plot_year})",
                ylabel=ANOMALY_CBAR_LABELS.get(var_name, var_name),
            )


def main():
    args = parse_args()
    config = load_config(args.config)

    mode_lookup = {
        "init-store": init_store,
        "build-standard-var": build_standard_var,
        "finalize-standard": finalize_standard,
        "init-anomaly": init_anomaly,
        "build-anomaly-var": build_anomaly_var,
        "finalize-anomaly": finalize_anomaly,
    }
    mode_lookup[args.mode](args, config)


if __name__ == "__main__":
    main()
