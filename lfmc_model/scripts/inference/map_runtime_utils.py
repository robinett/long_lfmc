import calendar
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from pyproj import Transformer

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, "..", "..", "..")
sys.path.append(os.path.join(project_root, "lfmc_model", "scripts", "eval"))
sys.path.append(os.path.join(project_root, "lfmc_model", "scripts", "inference"))

from compare_models_at_sites import get_site_error
from compare_timeseries import (
    CLIMATE_NC_PATH,
    DEFAULT_INPUTS_ROOT,
    DEFAULT_OAK_ROOT,
    DEFAULT_SCRATCH_ROOT,
    MODIS_ZARR_PATH,
    STATIC_NC_PATH,
    VAR_LOCS,
    _clamp_inference_window,
    _convert_tensor_payload_norm,
    _load_model_runtime,
    _runtimes_share_feature_layout,
    aggregate_site_errors,
    get_inference_datasets,
    select_ensemble_member_dirs,
)
from point_tool_new import build_tensors, run_model_forward


DEFAULT_ENSEMBLE_OUTPUT_ROOT = os.path.join(
    DEFAULT_SCRATCH_ROOT,
    "lfmc_model",
    "outputs",
    "lfmc_vh_vv_ens_fullrandom",
)
DEFAULT_INPUT_DATA_NAME = "ensemble/lfmc_vh_vv_ens_fullrandom"
DEFAULT_MAP_RUN_ROOT = os.path.join(
    DEFAULT_SCRATCH_ROOT,
    "lfmc_model",
    "inference",
    "map_runs",
    "lfmc_vh_vv_ens_fullrandom",
)
DEFAULT_MODEL_GRID_PATH = os.path.join(
    DEFAULT_OAK_ROOT,
    "grid",
    "epsg5070_500m_westUS_grid.nc4",
)
OUTPUT_MEAN_NAME = "lfmc_ens_mean"
OUTPUT_STD_NAME = "lfmc_ens_std"
DEFAULT_FALLBACK_NUM_TASKS = 3
DEFAULT_MODEL_TYPE = "standard"


def _var_to_source(var_locs: Dict[str, Sequence[str]]) -> Dict[str, str]:
    out = {}
    for source_name, vars_here in var_locs.items():
        for var_name in vars_here:
            out[var_name] = source_name
    return out


def _nearest_index(coords: np.ndarray, value: float) -> int:
    coords = np.asarray(coords)
    if coords.ndim != 1:
        raise ValueError("Expected a 1D coordinate array")
    if coords.size == 1:
        return 0
    idx = int(np.searchsorted(coords, value))
    if idx <= 0:
        return 0
    if idx >= coords.size:
        return coords.size - 1
    left = coords[idx - 1]
    right = coords[idx]
    return idx - 1 if abs(value - left) <= abs(right - value) else idx


def get_time_coord_bounds(ds: xr.Dataset, coord_name: str = "time") -> Tuple[pd.Timestamp, pd.Timestamp]:
    vals = pd.to_datetime(ds[coord_name].values)
    return pd.Timestamp(vals.min()).normalize(), pd.Timestamp(vals.max()).normalize()


def get_year_coord_bounds(ds: xr.Dataset, coord_name: str = "year") -> Tuple[pd.Timestamp, pd.Timestamp]:
    vals = pd.to_datetime(ds[coord_name].values)
    min_ts = pd.Timestamp(vals.min()).normalize()
    max_ts = pd.Timestamp(vals.max()).normalize()
    max_ts = pd.Timestamp(max_ts.year, 12, 31)
    return min_ts, max_ts


def runtime_temporal_source_lags(
    runtime: Dict[str, object],
    var_locs: Dict[str, Sequence[str]] = VAR_LOCS,
) -> Dict[str, int]:
    var_to_source = _var_to_source(var_locs)
    out: Dict[str, int] = {}
    short_lag_days = [int(v) for v in runtime.get("short_lag_days", [])]
    long_lag_days = [int(v) for v in runtime.get("long_lag_days", [])]
    short_max = max(short_lag_days) if len(short_lag_days) > 0 else 0
    long_max = max(long_lag_days) if len(long_lag_days) > 0 else 0
    for var_name in runtime["var_names"].get("short_vars", []):
        if var_name == "lfrac":
            continue
        source_name = var_to_source.get(var_name)
        if source_name is None:
            continue
        out[source_name] = max(out.get(source_name, 0), short_max)
    for var_name in runtime["var_names"].get("long_vars", []):
        if var_name == "lfrac":
            continue
        source_name = var_to_source.get(var_name)
        if source_name is None:
            continue
        out[source_name] = max(out.get(source_name, 0), long_max)
    return out


def runtime_static_sources(
    runtime: Dict[str, object],
    var_locs: Dict[str, Sequence[str]] = VAR_LOCS,
) -> List[str]:
    var_to_source = _var_to_source(var_locs)
    sources = set()
    for var_name in runtime["var_names"].get("static_vars", []):
        if var_name in {"latitude", "longitude"}:
            continue
        source_name = var_to_source.get(var_name)
        if source_name is not None:
            sources.add(source_name)
    return sorted(sources)


def clamp_runtime_window_for_sources(
    dss: Dict[str, xr.Dataset],
    runtime: Dict[str, object],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    var_locs: Dict[str, Sequence[str]] = VAR_LOCS,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    safe_start, safe_end = _clamp_inference_window(
        dss,
        start_date,
        end_date,
        runtime["short_lag_days"],
        runtime["long_lag_days"],
    )
    static_sources = runtime_static_sources(runtime, var_locs=var_locs)
    if "landcover_frac" in static_sources:
        lc_start, lc_end = get_year_coord_bounds(dss["landcover_frac"], coord_name="year")
        safe_start = max(safe_start, lc_start)
        safe_end = min(safe_end, lc_end)
    return safe_start, safe_end


def resolve_common_runtime_window(
    dss: Dict[str, xr.Dataset],
    runtimes: Sequence[Dict[str, object]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    var_locs: Dict[str, Sequence[str]] = VAR_LOCS,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    starts = []
    ends = []
    for runtime in runtimes:
        safe_start, safe_end = clamp_runtime_window_for_sources(
            dss,
            runtime,
            start_date,
            end_date,
            var_locs=var_locs,
        )
        starts.append(safe_start)
        ends.append(safe_end)
    return max(starts), min(ends)


def load_ensemble_runtimes(
    ensemble_root: str = DEFAULT_ENSEMBLE_OUTPUT_ROOT,
    input_data_name: str = DEFAULT_INPUT_DATA_NAME,
    inputs_root: str = DEFAULT_INPUTS_ROOT,
    fold: int = 9998,
    fallback_num_tasks: int = DEFAULT_FALLBACK_NUM_TASKS,
) -> Tuple[List[str], List[Dict[str, object]]]:
    member_dirs = select_ensemble_member_dirs(ensemble_root)
    runtime_cache: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    runtimes = [
        _load_model_runtime(
            model_dir=member_dir,
            fold=str(fold),
            inputs_root=inputs_root,
            input_data_name=input_data_name,
            fallback_num_tasks=fallback_num_tasks,
            runtime_cache=runtime_cache,
        )
        for member_dir in member_dirs
    ]
    return member_dirs, runtimes


def month_blocks(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_per_block: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if months_per_block <= 0:
        raise ValueError("months_per_block must be >= 1")
    out = []
    current = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    while current <= end_date:
        block_start = current
        year = current.year
        month = current.month + months_per_block - 1
        year += (month - 1) // 12
        month = ((month - 1) % 12) + 1
        last_day = calendar.monthrange(year, month)[1]
        block_end = pd.Timestamp(year, month, last_day)
        block_end = min(block_end.normalize(), end_date)
        out.append((block_start, block_end))
        current = block_end + pd.Timedelta(days=1)
    return out


def open_model_grid(grid_path: str = DEFAULT_MODEL_GRID_PATH) -> xr.Dataset:
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Missing model grid: {grid_path}")
    return xr.open_dataset(grid_path)


def build_tile_payloads(model_grid: xr.Dataset, tile_size: int) -> Dict[str, Dict[str, np.ndarray]]:
    vals = model_grid["random_vals"]
    lats = model_grid["lat"].values
    lons = model_grid["lon"].values
    mask = vals.notnull()
    iy_all, ix_all = np.where(mask.data)
    height, width = mask.shape
    tile_df = pd.DataFrame(
        {
            "iy": iy_all,
            "ix": ix_all,
            "tile_iy": iy_all // tile_size,
            "tile_ix": ix_all // tile_size,
        }
    )
    tile_payloads: Dict[str, Dict[str, np.ndarray]] = {}
    grouped = tile_df.groupby(["tile_ix", "tile_iy"], sort=False)
    for (tile_ix, tile_iy), group in grouped:
        iy = group["iy"].to_numpy(dtype=np.int32)
        ix = group["ix"].to_numpy(dtype=np.int32)
        y0 = int(tile_iy * tile_size)
        y1 = int(min((tile_iy + 1) * tile_size, height))
        x0 = int(tile_ix * tile_size)
        x1 = int(min((tile_ix + 1) * tile_size, width))
        tile_name = f"{int(tile_ix)}_{int(tile_iy)}"
        tile_payloads[tile_name] = {
            "tile_name": np.asarray(tile_name),
            "tile_ix": np.asarray(int(tile_ix), dtype=np.int32),
            "tile_iy": np.asarray(int(tile_iy), dtype=np.int32),
            "x0": np.asarray(x0, dtype=np.int32),
            "x1": np.asarray(x1, dtype=np.int32),
            "y0": np.asarray(y0, dtype=np.int32),
            "y1": np.asarray(y1, dtype=np.int32),
            "iy": iy,
            "ix": ix,
            "lat": lats[iy, ix].astype(np.float64),
            "lon": lons[iy, ix].astype(np.float64),
        }
    return tile_payloads


def write_tile_payloads(tile_payloads: Dict[str, Dict[str, np.ndarray]], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tile_meta_paths = {}
    for tile_name, payload in tile_payloads.items():
        tile_path = os.path.join(out_dir, f"tile_{tile_name}.npz")
        np.savez_compressed(tile_path, **payload)
        tile_meta_paths[tile_name] = tile_path
    return tile_meta_paths


def load_tile_payload(tile_meta_path: str) -> Dict[str, np.ndarray]:
    with np.load(tile_meta_path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def build_reference_tensor_payload(
    tile_payload: Dict[str, np.ndarray],
    runtime: Dict[str, object],
    dss: Dict[str, xr.Dataset],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Dict[str, object]:
    locs = np.column_stack((tile_payload["lon"], tile_payload["lat"])).tolist()
    short_tensor, long_tensor, static_tensor, info_df = build_tensors(
        locs=locs,
        start_dates=[start_date for _ in locs],
        end_dates=[end_date for _ in locs],
        var_names=runtime["var_names"],
        var_locs=VAR_LOCS,
        dss=dss,
        short_lag_days=runtime["short_lag_days"],
        long_lag_days=runtime["long_lag_days"],
        norm_params=runtime["norm_params"],
        all_nearby=True,
    )
    return {
        "empty": False,
        "safe_start": start_date,
        "safe_end": end_date,
        "short_tensor": short_tensor,
        "long_tensor": long_tensor,
        "static_tensor": static_tensor,
        "info_df": info_df,
    }


def run_runtime_forward(
    runtime: Dict[str, object],
    tensor_payload: Dict[str, object],
    model_type: str = DEFAULT_MODEL_TYPE,
) -> pd.DataFrame:
    preds_df = run_model_forward(
        tensor_payload["short_tensor"],
        tensor_payload["long_tensor"],
        tensor_payload["static_tensor"],
        tensor_payload["info_df"].copy(),
        runtime["checkpoint_path"],
        runtime["norm_params"],
        model_task_weights=runtime["model_num_tasks"],
        model_type=model_type,
    )
    preds_df = preds_df.copy().reset_index(drop=True)
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    return preds_df


def aggregate_member_predictions(
    member_dfs: Sequence[pd.DataFrame],
    mean_col_name: str = OUTPUT_MEAN_NAME,
    std_col_name: str = OUTPUT_STD_NAME,
) -> pd.DataFrame:
    if len(member_dfs) == 0:
        raise ValueError("No ensemble member prediction frames provided")
    base = member_dfs[0][["lat", "lon", "date"]].copy().reset_index(drop=True)
    for member_idx, member_df in enumerate(member_dfs[1:], start=1):
        same_lat = np.allclose(
            member_df["lat"].to_numpy(dtype=np.float64),
            base["lat"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-8,
            equal_nan=True,
        )
        same_lon = np.allclose(
            member_df["lon"].to_numpy(dtype=np.float64),
            base["lon"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-8,
            equal_nan=True,
        )
        same_date = np.array_equal(
            pd.to_datetime(member_df["date"]).values,
            pd.to_datetime(base["date"]).values,
        )
        if not (same_lat and same_lon and same_date):
            raise ValueError(
                f"Ensemble member {member_idx + 1} prediction rows do not align"
            )
    pred_stack = np.stack(
        [df["lfmc_pred"].to_numpy(dtype=np.float64) for df in member_dfs],
        axis=1,
    )
    out = base
    out["ensemble_n"] = int(len(member_dfs))
    out[mean_col_name] = np.nanmean(pred_stack, axis=1).astype(np.float32)
    out[std_col_name] = np.nanstd(pred_stack, axis=1, ddof=0).astype(np.float32)
    return out


def densify_tile_predictions(
    agg_df: pd.DataFrame,
    tile_payload: Dict[str, np.ndarray],
    mean_col_name: str = OUTPUT_MEAN_NAME,
    std_col_name: str = OUTPUT_STD_NAME,
) -> Dict[str, np.ndarray]:
    start_date = pd.Timestamp(pd.to_datetime(agg_df["date"]).min()).normalize()
    end_date = pd.Timestamp(pd.to_datetime(agg_df["date"]).max()).normalize()
    date_index = pd.date_range(start_date, end_date, freq="D")
    n_time = len(date_index)
    n_pixels = len(tile_payload["iy"])
    expected_rows = n_time * n_pixels
    if len(agg_df) != expected_rows:
        raise ValueError(
            f"Tile row count mismatch: expected {expected_rows:,}, got {len(agg_df):,}"
        )
    mean_vals = agg_df[mean_col_name].to_numpy(dtype=np.float32).reshape(n_pixels, n_time).T
    std_vals = agg_df[std_col_name].to_numpy(dtype=np.float32).reshape(n_pixels, n_time).T
    y0 = int(tile_payload["y0"])
    y1 = int(tile_payload["y1"])
    x0 = int(tile_payload["x0"])
    x1 = int(tile_payload["x1"])
    tile_h = int(y1 - y0)
    tile_w = int(x1 - x0)
    dense_mean = np.full((n_time, tile_h, tile_w), np.nan, dtype=np.float32)
    dense_std = np.full((n_time, tile_h, tile_w), np.nan, dtype=np.float32)
    local_y = tile_payload["iy"].astype(np.int64) - y0
    local_x = tile_payload["ix"].astype(np.int64) - x0
    for pix_idx in range(n_pixels):
        dense_mean[:, local_y[pix_idx], local_x[pix_idx]] = mean_vals[:, pix_idx]
        dense_std[:, local_y[pix_idx], local_x[pix_idx]] = std_vals[:, pix_idx]
    return {
        "dates": np.asarray(date_index.values, dtype="datetime64[ns]"),
        OUTPUT_MEAN_NAME: dense_mean,
        OUTPUT_STD_NAME: dense_std,
        "x0": np.asarray(x0, dtype=np.int32),
        "x1": np.asarray(x1, dtype=np.int32),
        "y0": np.asarray(y0, dtype=np.int32),
        "y1": np.asarray(y1, dtype=np.int32),
    }


def save_tile_shard(
    shard_path: str,
    dense_payload: Dict[str, np.ndarray],
    tile_payload: Dict[str, np.ndarray],
    task_row: pd.Series,
) -> None:
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    np.savez_compressed(
        shard_path,
        tile_name=np.asarray(str(task_row["tile_name"])),
        start_date=np.asarray(str(task_row["start_date"])),
        end_date=np.asarray(str(task_row["end_date"])),
        dates=dense_payload["dates"],
        x0=dense_payload["x0"],
        x1=dense_payload["x1"],
        y0=dense_payload["y0"],
        y1=dense_payload["y1"],
        iy=tile_payload["iy"].astype(np.int32),
        ix=tile_payload["ix"].astype(np.int32),
        lat=tile_payload["lat"].astype(np.float64),
        lon=tile_payload["lon"].astype(np.float64),
        lfmc_ens_mean=dense_payload[OUTPUT_MEAN_NAME],
        lfmc_ens_std=dense_payload[OUTPUT_STD_NAME],
    )


def initialize_output_store(
    out_path: str,
    model_grid: xr.Dataset,
    time_index: pd.DatetimeIndex,
    time_chunk: int,
    y_chunk: int,
    x_chunk: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open_group(out_path, mode="w")
    y_size = int(model_grid.sizes["y"])
    x_size = int(model_grid.sizes["x"])
    root.create_dataset(
        OUTPUT_MEAN_NAME,
        shape=(len(time_index), y_size, x_size),
        chunks=(time_chunk, y_chunk, x_chunk),
        dtype="f4",
        fill_value=np.nan,
        overwrite=True,
    )
    root[OUTPUT_MEAN_NAME].attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
    root.create_dataset(
        OUTPUT_STD_NAME,
        shape=(len(time_index), y_size, x_size),
        chunks=(time_chunk, y_chunk, x_chunk),
        dtype="f4",
        fill_value=np.nan,
        overwrite=True,
    )
    root[OUTPUT_STD_NAME].attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
    time_vals = np.asarray(time_index.values, dtype="datetime64[ns]").astype("int64")
    root.create_dataset(
        "time",
        data=time_vals,
        chunks=(time_chunk,),
        dtype="i8",
        overwrite=True,
    )
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["time"].attrs["units"] = "nanoseconds since 1970-01-01 00:00:00"
    root["time"].attrs["calendar"] = "proleptic_gregorian"
    for coord_name in ["y", "x"]:
        vals = np.asarray(model_grid[coord_name].values)
        root.create_dataset(
            coord_name,
            data=vals,
            chunks=(min(len(vals), y_chunk if coord_name == "y" else x_chunk),),
            dtype=str(vals.dtype),
            overwrite=True,
        )
        root[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]
    for coord_name in ["lat", "lon", "random_vals"]:
        if coord_name not in model_grid:
            continue
        vals = np.asarray(model_grid[coord_name].values)
        root.create_dataset(
            coord_name,
            data=vals,
            chunks=(y_chunk, x_chunk),
            dtype=str(vals.dtype),
            overwrite=True,
        )
        root[coord_name].attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]


def merge_shard_into_store(
    store_path: str,
    shard_path: str,
    time_lookup: Dict[np.datetime64, int],
) -> None:
    root = zarr.open_group(store_path, mode="a")
    with np.load(shard_path, allow_pickle=False) as npz:
        dates = np.asarray(npz["dates"], dtype="datetime64[ns]")
        t0 = time_lookup[dates[0]]
        t1 = time_lookup[dates[-1]] + 1
        y0 = int(npz["y0"])
        y1 = int(npz["y1"])
        x0 = int(npz["x0"])
        x1 = int(npz["x1"])
        root[OUTPUT_MEAN_NAME][t0:t1, y0:y1, x0:x1] = np.asarray(
            npz[OUTPUT_MEAN_NAME],
            dtype=np.float32,
        )
        root[OUTPUT_STD_NAME][t0:t1, y0:y1, x0:x1] = np.asarray(
            npz[OUTPUT_STD_NAME],
            dtype=np.float32,
        )


def select_measurement_rich_month(
    ensemble_root: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.Timestamp, pd.Timestamp, Dict[str, object]]:
    member_dirs = select_ensemble_member_dirs(ensemble_root)
    member_site_errors = [get_site_error(member_dir) for member_dir in member_dirs]
    site_error = aggregate_site_errors(member_site_errors)
    month_counts: Dict[pd.Timestamp, int] = {}
    for site_key, site_data in site_error.items():
        dates = pd.to_datetime(site_data["dates"])
        keep = (dates >= start_date) & (dates <= end_date)
        dates = dates[keep]
        if len(dates) == 0:
            continue
        month_starts = dates.to_period("M").to_timestamp()
        counts = month_starts.value_counts()
        for month_start, count in counts.items():
            month_counts[month_start] = month_counts.get(month_start, 0) + int(count)
    if len(month_counts) == 0:
        raise ValueError(
            f"No site-error measurements found between {start_date.date()} and {end_date.date()}"
        )
    best_month = max(month_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    month_end = (best_month + pd.offsets.MonthEnd(0)).normalize()
    if month_end > end_date:
        month_end = pd.Timestamp(end_date).normalize()
    return best_month.normalize(), month_end, site_error


def select_validation_sites_for_month(
    site_error: Dict[str, Dict[str, object]],
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    n_sites: int = 3,
) -> List[Dict[str, object]]:
    candidates = []
    for site_key, site_data in site_error.items():
        dates = pd.to_datetime(site_data["dates"])
        keep = (dates >= month_start) & (dates <= month_end)
        if not np.any(keep):
            continue
        dates_here = dates[keep]
        true_here = np.asarray(site_data["true_values"], dtype=float)[keep]
        pred_here = np.asarray(site_data["predictions"], dtype=float)[keep]
        candidates.append(
            {
                "site_key": site_key,
                "fold": str(site_data["fold"]),
                "num_measurements_month": int(np.sum(keep)),
                "dates": dates_here,
                "true_values": true_here,
                "predictions": pred_here,
            }
        )
    candidates = sorted(
        candidates,
        key=lambda x: (-x["num_measurements_month"], x["site_key"]),
    )
    return candidates[:n_sites]


def locate_sites_to_tiles(
    model_grid: xr.Dataset,
    site_records: Sequence[Dict[str, object]],
    tile_size: int,
) -> List[str]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x_coords = np.asarray(model_grid["x"].values, dtype=np.float64)
    y_coords = np.asarray(model_grid["y"].values, dtype=np.float64)
    tile_names = set()
    for site_record in site_records:
        lat_str, lon_str = site_record["site_key"].split("_")
        lat = float(lat_str)
        lon = float(lon_str)
        site_x, site_y = transformer.transform(lon, lat)
        x_idx = _nearest_index(x_coords, site_x)
        y_idx = _nearest_index(y_coords, site_y)
        tile_name = f"{x_idx // tile_size}_{y_idx // tile_size}"
        tile_names.add(tile_name)
    return sorted(tile_names)


def latest_run_dir(base_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No run_* directories found under {base_dir}")
    return candidates[-1]
