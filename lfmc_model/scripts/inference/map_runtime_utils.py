import calendar
import copy
import datetime as dt
import glob
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
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
    _effective_static_norm_arrays,
    _input_norm_signature,
    _load_model_runtime,
    _renormalize_tensor,
    _runtimes_share_feature_layout,
    aggregate_site_errors,
    get_inference_datasets,
    select_ensemble_member_dirs,
)
from point_tool_new import (
    build_tensors,
    load_model_for_inference,
    predict_with_loaded_model,
    run_model_forward,
)


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
    DEFAULT_SCRATCH_ROOT,
    "grid",
    "epsg5070_500m_westUS_grid.nc4",
)
OUTPUT_MEAN_NAME = "lfmc_ens_mean"
OUTPUT_STD_NAME = "lfmc_ens_std"
OUTPUT_QUALITY_FLAG_NAME = "quality_flag"
OUTPUT_DOMINANT_LANDCOVER_NAME = "dominant_landcover_code"
OUTPUT_LANDCOVER_YEAR_NAME = "landcover_year"
DEFAULT_FALLBACK_NUM_TASKS = 3
DEFAULT_MODEL_TYPE = "standard"
DEFAULT_LANDCOVER_MASK_CACHE_DIR = os.path.join(
    DEFAULT_SCRATCH_ROOT,
    "lfmc_model",
    "inference",
    "cache",
    "landcover_masks",
)
ALLOWED_DOMINANT_LANDCOVER = (
    "deciduous_forest",
    "evergreen_forest",
    "mixed_forest",
    "shrub",
    "grass",
)
QUALITY_FLAG_VALUES = {"final": 0, "low_latency": 1}
LANDCOVER_NODATA_CODE = np.uint8(255)


def timestamped_message(message: str) -> str:
    return f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"


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
    coord_diffs = np.diff(coords)
    if np.all(coord_diffs > 0):
        idx = int(np.searchsorted(coords, value))
    elif np.all(coord_diffs < 0):
        idx = int(coords.size - np.searchsorted(coords[::-1], value))
    else:
        raise ValueError("Coordinate array must be monotonic")
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
        var_names=runtime.get("var_names"),
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


def _select_member_subset(member_dirs: Sequence[str], max_members: Optional[int] = None) -> List[str]:
    if max_members in {None, "", "None"}:
        return list(member_dirs)
    max_members = int(max_members)
    if max_members <= 0:
        raise ValueError("max_members must be >= 1 when provided")
    return list(member_dirs[:max_members])


def load_ensemble_runtimes(
    ensemble_root: str = DEFAULT_ENSEMBLE_OUTPUT_ROOT,
    input_data_name: str = DEFAULT_INPUT_DATA_NAME,
    inputs_root: str = DEFAULT_INPUTS_ROOT,
    fold: int = 9998,
    fallback_num_tasks: int = DEFAULT_FALLBACK_NUM_TASKS,
    max_members: Optional[int] = None,
    member_name_prefix: Optional[str] = None,
    selection_key: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, object]]]:
    member_dirs = _select_member_subset(
        select_ensemble_member_dirs(
            ensemble_root,
            member_name_prefix=member_name_prefix,
            selection_key=selection_key,
        ),
        max_members=max_members,
    )
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
        year_end = pd.Timestamp(block_start.year, 12, 31)
        block_end = min(block_end.normalize(), year_end, end_date)
        out.append((block_start, block_end))
        current = block_end + pd.Timedelta(days=1)
    return out


def open_model_grid(grid_path: str = DEFAULT_MODEL_GRID_PATH) -> xr.Dataset:
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Missing model grid: {grid_path}")
    return xr.open_dataset(grid_path)


def build_landcover_allowed_mask_for_year(
    landcover_ds: xr.Dataset,
    year: int,
    allowed_classes: Sequence[str] = ALLOWED_DOMINANT_LANDCOVER,
) -> xr.DataArray:
    landcover_vars = list(landcover_ds.data_vars)
    missing_classes = [name for name in allowed_classes if name not in landcover_vars]
    if len(missing_classes) > 0:
        raise KeyError(
            f"Missing allowed landcover class(es) in landcover dataset: {missing_classes}"
        )
    year_key = pd.Timestamp(year, 1, 1)
    year_ds = landcover_ds.sel(year=year_key)
    lc_array = year_ds.to_array(dim="landcover")
    lc_filled = lc_array.fillna(-np.inf)
    dominant_idx = lc_filled.argmax(dim="landcover")
    any_valid = lc_array.notnull().any(dim="landcover")
    allowed_indices = [landcover_vars.index(name) for name in allowed_classes]
    allowed_mask = xr.zeros_like(any_valid, dtype=bool)
    for idx in allowed_indices:
        allowed_mask = allowed_mask | (dominant_idx == idx)
    return any_valid & allowed_mask


def _mask_cache_signature(
    model_grid: xr.Dataset,
    landcover_ds: xr.Dataset,
    grid_path: str,
    allowed_classes: Sequence[str],
) -> str:
    def _source_signature(path: str) -> Dict[str, object]:
        out: Dict[str, object] = {"path": str(path)}
        if path and os.path.exists(path):
            stat = os.stat(path)
            out["mtime_ns"] = int(stat.st_mtime_ns)
            out["size"] = int(stat.st_size)
        return out

    landcover_var = next(iter(landcover_ds.data_vars.values()))
    payload = {
        "grid_path": grid_path,
        "grid_shape": {k: int(v) for k, v in model_grid.sizes.items()},
        "grid_x_bounds": [
            float(model_grid["x"].values[0]),
            float(model_grid["x"].values[-1]),
        ],
        "grid_y_bounds": [
            float(model_grid["y"].values[0]),
            float(model_grid["y"].values[-1]),
        ],
        "landcover_source": _source_signature(str(landcover_var.encoding.get("source", ""))),
        "landcover_shape": {k: int(v) for k, v in landcover_ds.sizes.items()},
        "landcover_vars": sorted(landcover_ds.data_vars),
        "allowed_classes": list(allowed_classes),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def load_or_build_prediction_mask_for_year(
    model_grid: xr.Dataset,
    landcover_ds: xr.Dataset,
    year: int,
    grid_path: str = DEFAULT_MODEL_GRID_PATH,
    allowed_classes: Sequence[str] = ALLOWED_DOMINANT_LANDCOVER,
    cache_root: str = DEFAULT_LANDCOVER_MASK_CACHE_DIR,
) -> xr.DataArray:
    cache_sig = _mask_cache_signature(
        model_grid=model_grid,
        landcover_ds=landcover_ds,
        grid_path=grid_path,
        allowed_classes=allowed_classes,
    )
    cache_dir = os.path.join(cache_root, cache_sig)
    cache_path = os.path.join(cache_dir, f"prediction_mask_{year}.npz")
    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as npz:
            mask = np.asarray(npz["mask"], dtype=bool)
            y_vals = np.asarray(npz["y"])
            x_vals = np.asarray(npz["x"])
        print(f"[prediction_mask] year={year} cache hit: {cache_path}")
        return xr.DataArray(mask, coords={"y": y_vals, "x": x_vals}, dims=("y", "x"))

    landcover_mask = build_landcover_allowed_mask_for_year(
        landcover_ds=landcover_ds,
        year=year,
        allowed_classes=allowed_classes,
    )
    model_random_mask = model_grid["random_vals"].notnull()
    prediction_mask = (model_random_mask & landcover_mask).transpose(*model_random_mask.dims)
    mask = np.asarray(prediction_mask.values, dtype=bool)
    y_vals = np.asarray(model_grid["y"].values)
    x_vals = np.asarray(model_grid["x"].values)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        cache_path,
        mask=mask,
        y=y_vals,
        x=x_vals,
    )
    print(f"[prediction_mask] year={year} wrote cache: {cache_path}")
    return xr.DataArray(mask, coords={"y": y_vals, "x": x_vals}, dims=("y", "x"))


def build_tile_payloads(
    model_grid: xr.Dataset,
    tile_size: int,
    valid_mask: Optional[xr.DataArray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    vals = model_grid["random_vals"]
    lats = model_grid["lat"].values
    lons = model_grid["lon"].values
    mask = vals.notnull()
    if valid_mask is not None:
        aligned_mask = valid_mask.transpose(*mask.dims)
        mask = mask & aligned_mask
    mask_data = np.asarray(mask.values, dtype=bool)
    iy_all, ix_all = np.where(mask_data)
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


def write_tile_payloads(
    tile_payloads: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
    file_prefix: Optional[str] = None,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tile_meta_paths = {}
    for tile_name, payload in tile_payloads.items():
        filename = f"tile_{tile_name}.npz"
        if file_prefix is not None and file_prefix != "":
            filename = f"{file_prefix}_{filename}"
        tile_path = os.path.join(out_dir, filename)
        np.savez_compressed(tile_path, **payload)
        tile_meta_paths[tile_name] = tile_path
    return tile_meta_paths


def load_tile_payload(tile_meta_path: str) -> Dict[str, np.ndarray]:
    with np.load(tile_meta_path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def save_prepared_tensor_payload(prepared_path: str, tensor_payload: Dict[str, object]) -> None:
    prepared_dir = os.path.dirname(prepared_path)
    os.makedirs(prepared_dir, exist_ok=True)
    info_df = tensor_payload["info_df"].copy().reset_index(drop=True)
    payload_to_save = {
        "safe_start": str(pd.Timestamp(tensor_payload["safe_start"]).date()),
        "safe_end": str(pd.Timestamp(tensor_payload["safe_end"]).date()),
        "short_tensor": tensor_payload["short_tensor"].detach().cpu(),
        "long_tensor": tensor_payload["long_tensor"].detach().cpu(),
        "static_tensor": tensor_payload["static_tensor"].detach().cpu(),
        "lat": info_df["lat"].to_numpy(dtype=np.float64),
        "lon": info_df["lon"].to_numpy(dtype=np.float64),
        "date": pd.to_datetime(info_df["date"]).to_numpy(dtype="datetime64[ns]"),
    }
    temp_path = (
        f"{prepared_path}.tmp_{os.getpid()}_{pd.Timestamp.utcnow().value}"
    )
    try:
        torch.save(payload_to_save, temp_path)
        os.replace(temp_path, prepared_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_prepared_tensor_payload(prepared_path: str) -> Dict[str, object]:
    raw = torch.load(prepared_path, map_location="cpu", weights_only=False)
    info_df = pd.DataFrame(
        {
            "lat": np.asarray(raw["lat"], dtype=np.float64),
            "lon": np.asarray(raw["lon"], dtype=np.float64),
            "date": pd.to_datetime(np.asarray(raw["date"])),
        }
    )
    return {
        "empty": False,
        "safe_start": pd.Timestamp(raw["safe_start"]).normalize(),
        "safe_end": pd.Timestamp(raw["safe_end"]).normalize(),
        "short_tensor": raw["short_tensor"],
        "long_tensor": raw["long_tensor"],
        "static_tensor": raw["static_tensor"],
        "info_df": info_df,
    }


def runtimes_share_short_long_layout(
    reference_runtime: Dict[str, object],
    runtime: Dict[str, object],
) -> bool:
    return (
        list(reference_runtime["var_names"]["short_vars"]) == list(runtime["var_names"]["short_vars"])
        and list(reference_runtime["var_names"]["long_vars"]) == list(runtime["var_names"]["long_vars"])
        and list(reference_runtime["short_lag_days"]) == list(runtime["short_lag_days"])
        and list(reference_runtime["long_lag_days"]) == list(runtime["long_lag_days"])
    )


def build_static_superset_runtime(
    reference_runtime: Dict[str, object],
    runtimes: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    superset_static_vars: List[str] = []
    for runtime in runtimes:
        for var_name in runtime["var_names"]["static_vars"]:
            if var_name not in superset_static_vars:
                superset_static_vars.append(var_name)

    out = copy.deepcopy(reference_runtime)
    out["var_names"]["static_vars"] = superset_static_vars

    ref_static_vars = list(reference_runtime["var_names"]["static_vars"])
    ref_static_mean, ref_static_std = _effective_static_norm_arrays(
        ref_static_vars,
        reference_runtime["norm_params"],
    )
    ref_idx_lookup = {var_name: idx for idx, var_name in enumerate(ref_static_vars)}
    sup_mean = np.empty(len(superset_static_vars), dtype=np.float32)
    sup_std = np.empty(len(superset_static_vars), dtype=np.float32)
    for idx, var_name in enumerate(superset_static_vars):
        if var_name in ref_idx_lookup:
            ref_idx = ref_idx_lookup[var_name]
            sup_mean[idx] = ref_static_mean[ref_idx]
            sup_std[idx] = ref_static_std[ref_idx]
        elif var_name.startswith("climate_zone_") or _static_var_is_landcover(var_name):
            sup_mean[idx] = 0.0
            sup_std[idx] = 1.0
        else:
            raise ValueError(
                f"Cannot synthesize superset static normalization for normalized feature {var_name}"
            )
    out["norm_params"]["train_static_mean"] = sup_mean.tolist()
    out["norm_params"]["train_static_std"] = sup_std.tolist()
    return out


def convert_tensor_payload_to_runtime(
    reference_payload: Dict[str, object],
    reference_runtime: Dict[str, object],
    runtime: Dict[str, object],
    tensor_cache: Dict[Tuple[str, ...], Dict[str, object]],
    site: str,
) -> Dict[str, object]:
    if _runtimes_share_feature_layout(reference_runtime, runtime):
        return _convert_tensor_payload_norm(
            reference_payload,
            reference_runtime,
            runtime,
            tensor_cache,
            site,
        )
    if not runtimes_share_short_long_layout(reference_runtime, runtime):
        raise ValueError("Short/long feature layout differs; full tensor rebuild required")

    cache_key = (
        "static_project_renorm",
        site,
        str(reference_payload["safe_start"].date()),
        str(reference_payload["safe_end"].date()),
        reference_runtime["input_data_dir"],
        runtime["input_data_dir"],
        _input_norm_signature(reference_runtime["norm_params"]),
        _input_norm_signature(runtime["norm_params"]),
        "|".join(reference_runtime["var_names"]["static_vars"]),
        "|".join(runtime["var_names"]["static_vars"]),
    )
    if cache_key in tensor_cache:
        return tensor_cache[cache_key]

    short_ref_mean = np.asarray(reference_runtime["norm_params"]["train_short_mean"], dtype=np.float32)
    short_ref_std = np.asarray(reference_runtime["norm_params"]["train_short_std"], dtype=np.float32)
    short_new_mean = np.asarray(runtime["norm_params"]["train_short_mean"], dtype=np.float32)
    short_new_std = np.asarray(runtime["norm_params"]["train_short_std"], dtype=np.float32)
    long_ref_mean = np.asarray(reference_runtime["norm_params"]["train_long_mean"], dtype=np.float32)
    long_ref_std = np.asarray(reference_runtime["norm_params"]["train_long_std"], dtype=np.float32)
    long_new_mean = np.asarray(runtime["norm_params"]["train_long_mean"], dtype=np.float32)
    long_new_std = np.asarray(runtime["norm_params"]["train_long_std"], dtype=np.float32)

    ref_static_vars = list(reference_runtime["var_names"]["static_vars"])
    new_static_vars = list(runtime["var_names"]["static_vars"])
    ref_static_idx = {var_name: idx for idx, var_name in enumerate(ref_static_vars)}
    missing_static = [var_name for var_name in new_static_vars if var_name not in ref_static_idx]
    if len(missing_static) > 0:
        raise ValueError(
            f"Reference static superset is missing runtime static vars: {missing_static}"
        )
    static_indices = torch.tensor(
        [ref_static_idx[var_name] for var_name in new_static_vars],
        dtype=torch.long,
    )
    selected_static = reference_payload["static_tensor"].index_select(2, static_indices)
    static_ref_mean, static_ref_std = _effective_static_norm_arrays(
        ref_static_vars,
        reference_runtime["norm_params"],
    )
    static_new_mean, static_new_std = _effective_static_norm_arrays(
        new_static_vars,
        runtime["norm_params"],
    )
    static_ref_mean = static_ref_mean[static_indices.numpy()]
    static_ref_std = static_ref_std[static_indices.numpy()]

    out = {
        "empty": False,
        "safe_start": reference_payload["safe_start"],
        "safe_end": reference_payload["safe_end"],
        "short_tensor": _renormalize_tensor(
            reference_payload["short_tensor"],
            short_ref_mean,
            short_ref_std,
            short_new_mean,
            short_new_std,
        ),
        "long_tensor": _renormalize_tensor(
            reference_payload["long_tensor"],
            long_ref_mean,
            long_ref_std,
            long_new_mean,
            long_new_std,
        ),
        "static_tensor": _renormalize_tensor(
            selected_static,
            static_ref_mean,
            static_ref_std,
            static_new_mean,
            static_new_std,
        ),
        "info_df": reference_payload["info_df"].copy(),
    }
    tensor_cache[cache_key] = out
    return out


def _model_grid_coords_for_tile(
    tile_payload: Dict[str, np.ndarray],
    dss: Dict[str, xr.Dataset],
) -> Tuple[np.ndarray, np.ndarray]:
    ix = tile_payload["ix"].astype(np.int64)
    iy = tile_payload["iy"].astype(np.int64)
    x_coords = np.asarray(dss["modis"]["x"].values, dtype=np.float64)[ix]
    # Full-grid inference datasets use the reverse y ordering of the model grid.
    y_model_order = np.asarray(dss["modis"]["y"].values, dtype=np.float64)[::-1]
    y_coords = y_model_order[iy]
    return x_coords, y_coords


def _map_coords_to_source_indices(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    coord_name: str,
    source_name: str,
    atol: float = 1e-6,
) -> np.ndarray:
    mapped = np.empty(len(target_coords), dtype=np.int64)
    unique_targets = np.unique(target_coords)
    lookup: Dict[float, int] = {}
    source_coords = np.asarray(source_coords, dtype=np.float64)
    for coord_val in unique_targets:
        idx = _nearest_index(source_coords, float(coord_val))
        if not np.isclose(source_coords[idx], coord_val, atol=atol, rtol=0.0):
            raise ValueError(
                f"{source_name} {coord_name}-coordinate mismatch for tile-native inference: "
                f"requested {coord_val}, matched {source_coords[idx]}"
            )
        lookup[float(coord_val)] = int(idx)
    for i, coord_val in enumerate(target_coords):
        mapped[i] = lookup[float(coord_val)]
    return mapped


def _resolve_tile_source_indices(
    tile_payload: Dict[str, np.ndarray],
    source_ds: xr.Dataset,
    source_name: str,
    dss: Dict[str, xr.Dataset],
) -> Dict[str, np.ndarray]:
    model_x_coords, model_y_coords = _model_grid_coords_for_tile(tile_payload, dss)
    source_x = np.asarray(source_ds["x"].values, dtype=np.float64)
    source_y = np.asarray(source_ds["y"].values, dtype=np.float64)
    source_ix = _map_coords_to_source_indices(source_x, model_x_coords, "x", source_name)
    source_iy = _map_coords_to_source_indices(source_y, model_y_coords, "y", source_name)
    x_min = int(source_ix.min())
    x_max = int(source_ix.max())
    y_min = int(source_iy.min())
    y_max = int(source_iy.max())
    return {
        "source_ix": source_ix,
        "source_iy": source_iy,
        "x_min": np.asarray(x_min, dtype=np.int64),
        "x_max": np.asarray(x_max, dtype=np.int64),
        "y_min": np.asarray(y_min, dtype=np.int64),
        "y_max": np.asarray(y_max, dtype=np.int64),
        "local_ix": source_ix - x_min,
        "local_iy": source_iy - y_min,
    }


def _stack_source_vars(
    ds: xr.Dataset,
    var_names: Sequence[str],
    time_slice: Optional[slice],
    x_slice: slice,
    y_slice: slice,
    source_name: str,
) -> xr.DataArray:
    base = ds.isel(x=x_slice, y=y_slice)
    if time_slice is not None:
        base = base.sel(time=time_slice)
    if source_name == "daymet":
        out = base["data"].sel(variable=list(var_names)).transpose("time", "variable", "y", "x")
    elif source_name == "modis" and "data" in base:
        out = base["data"].sel(variable=list(var_names)).transpose("time", "variable", "y", "x")
    else:
        missing = [var_name for var_name in var_names if var_name not in base.data_vars]
        if len(missing) > 0:
            raise KeyError(
                f"{source_name} is missing variables required for tile-native inference: {missing}"
            )
        out = (
            base[list(var_names)]
            .to_array(dim="variable")
            .transpose("time", "variable", "y", "x")
        )
    return out.compute()


def _validate_daily_time_axis(time_vals: np.ndarray, source_name: str) -> pd.DatetimeIndex:
    time_index = pd.to_datetime(time_vals)
    expected = pd.date_range(time_index.min(), time_index.max(), freq="D")
    if len(time_index) != len(expected) or not np.array_equal(time_index.values, expected.values):
        raise ValueError(
            f"{source_name} time axis is not complete daily coverage for tile-native inference"
        )
    return time_index


def _build_lag_fraction_values(lag_days: Sequence[int], mean: float, std: float) -> np.ndarray:
    lag_days = np.asarray(lag_days, dtype=np.float32)
    max_lag = float(np.max(lag_days)) if len(lag_days) > 0 else 0.0
    if max_lag <= 0:
        raw = np.ones_like(lag_days, dtype=np.float32)
    else:
        raw = lag_days / max_lag
    safe_std = std if abs(std) > 0 else 1.0
    return ((raw - mean) / safe_std).astype(np.float32)


def _static_var_is_landcover(var_name: str) -> bool:
    return any(
        token in var_name
        for token in (
            "barren",
            "crops",
            "forest",
            "developed",
            "grass",
            "other",
            "shrub",
            "water",
            "wetlands",
        )
    )


def _tile_native_tensor_payload(
    tile_payload: Dict[str, np.ndarray],
    runtime: Dict[str, object],
    dss: Dict[str, xr.Dataset],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    pixel_batch_size: int = 2048,
    day_batch_size: int = 31,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    pred_dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(pred_dates)
    n_pixels = int(len(tile_payload["iy"]))
    if n_pixels <= 0 or n_days <= 0:
        raise ValueError("Tile-native tensor payload requires at least one pixel and one day")

    short_vars = list(runtime["var_names"]["short_vars"])
    long_vars = list(runtime["var_names"]["long_vars"])
    static_vars = list(runtime["var_names"]["static_vars"])
    short_lag_days = np.asarray(runtime["short_lag_days"], dtype=np.int64)
    long_lag_days = np.asarray(runtime["long_lag_days"], dtype=np.int64)
    norm_params = runtime["norm_params"]
    var_to_source = _var_to_source(VAR_LOCS)

    short_source_vars = [var_name for var_name in short_vars if var_name != "lfrac"]
    long_source_vars = [var_name for var_name in long_vars if var_name != "lfrac"]
    static_source_vars = [
        var_name
        for var_name in static_vars
        if var_name not in {"latitude", "longitude"}
        and not _static_var_is_landcover(var_name)
    ]
    landcover_vars = [var_name for var_name in static_vars if _static_var_is_landcover(var_name)]

    short_input = np.full(
        (n_pixels, n_days, len(short_lag_days), len(short_vars)),
        np.nan,
        dtype=np.float32,
    )
    long_input = np.full(
        (n_pixels, n_days, len(long_lag_days), len(long_vars)),
        np.nan,
        dtype=np.float32,
    )
    static_input = np.full((n_pixels, n_days, 1, len(static_vars)), np.nan, dtype=np.float32)

    full_grid_sources = {
        "modis": dss["modis"],
        "landcover_frac": dss["landcover_frac"],
    }
    if len(long_source_vars) > 0:
        full_grid_sources["daymet"] = dss["daymet"]
    subset_sources = {
        "static": dss["static"],
        "soils": dss["soils"],
        "canopy_height": dss["canopy_height"],
    }
    if any(var_name.startswith("climate_zone_") for var_name in static_vars):
        raise KeyError(
            "Inference no longer supports climate_zone_* static vars. "
            "Rebuild or retire runtimes that still request them."
        )
    source_indexers = {
        source_name: _resolve_tile_source_indices(tile_payload, ds, source_name, dss)
        for source_name, ds in {**full_grid_sources, **subset_sources}.items()
    }

    short_max_lag = int(short_lag_days.max()) if len(short_lag_days) > 0 else 0
    long_max_lag = int(long_lag_days.max()) if len(long_lag_days) > 0 else 0
    short_time_start = start_date - pd.Timedelta(days=short_max_lag)
    long_time_start = start_date - pd.Timedelta(days=long_max_lag)

    modis_idx = source_indexers["modis"]
    daymet_idx = source_indexers.get("daymet")
    landcover_idx = source_indexers["landcover_frac"]
    modis_cube = _stack_source_vars(
        dss["modis"],
        short_source_vars,
        slice(short_time_start, end_date),
        slice(int(modis_idx["x_min"]), int(modis_idx["x_max"]) + 1),
        slice(int(modis_idx["y_min"]), int(modis_idx["y_max"]) + 1),
        "modis",
    ) if len(short_source_vars) > 0 else None
    daymet_cube = _stack_source_vars(
        dss["daymet"],
        long_source_vars,
        slice(long_time_start, end_date),
        slice(int(daymet_idx["x_min"]), int(daymet_idx["x_max"]) + 1),
        slice(int(daymet_idx["y_min"]), int(daymet_idx["y_max"]) + 1),
        "daymet",
    ) if len(long_source_vars) > 0 else None
    static_source_vars_by_source = {
        source_name: [
            var_name
            for var_name in static_source_vars
            if var_to_source.get(var_name) == source_name
        ]
        for source_name in ("static", "soils", "canopy_height")
    }
    static_cubes = {}
    for source_name, vars_here in static_source_vars_by_source.items():
        source_idx = source_indexers[source_name]
        if len(vars_here) == 0:
            static_cubes[source_name] = None
            continue
        static_cubes[source_name] = (
            dss[source_name]
            .isel(
                x=slice(int(source_idx["x_min"]), int(source_idx["x_max"]) + 1),
                y=slice(int(source_idx["y_min"]), int(source_idx["y_max"]) + 1),
            )[vars_here]
            .to_array(dim="variable")
            .transpose("variable", "y", "x")
            .compute()
        )
    landcover_cube = (
        dss["landcover_frac"]
        .isel(
            x=slice(int(landcover_idx["x_min"]), int(landcover_idx["x_max"]) + 1),
            y=slice(int(landcover_idx["y_min"]), int(landcover_idx["y_max"]) + 1),
        )[landcover_vars]
        .to_array(dim="variable")
        .transpose("year", "variable", "y", "x")
        .load()
        if len(landcover_vars) > 0
        else None
    )

    if modis_cube is not None:
        modis_time_index = _validate_daily_time_axis(modis_cube["time"].values, "modis")
        if modis_time_index[0] != short_time_start or modis_time_index[-1] != end_date:
            raise ValueError("MODIS time window does not match expected tile-native lag coverage")
        modis_arr = np.asarray(modis_cube.values, dtype=np.float32)
    else:
        modis_arr = None
    if daymet_cube is not None:
        daymet_time_index = _validate_daily_time_axis(daymet_cube["time"].values, "daymet")
        if daymet_time_index[0] != long_time_start or daymet_time_index[-1] != end_date:
            raise ValueError("daymet time window does not match expected tile-native lag coverage")
        daymet_arr = np.asarray(daymet_cube.values, dtype=np.float32)
    else:
        daymet_arr = None
    static_arr_by_source = {
        source_name: (
            np.asarray(cube.values, dtype=np.float32)
            if cube is not None
            else None
        )
        for source_name, cube in static_cubes.items()
    }
    landcover_arr = np.asarray(landcover_cube.values, dtype=np.float32) if landcover_cube is not None else None
    landcover_years = (
        pd.to_datetime(landcover_cube["year"].values).year.astype(int)
        if landcover_cube is not None
        else np.asarray([], dtype=np.int64)
    )
    landcover_year_lookup = {int(year): idx for idx, year in enumerate(landcover_years)}

    pred_years = pred_dates.year.to_numpy(dtype=np.int64)
    for year in np.unique(pred_years):
        if len(landcover_vars) > 0 and int(year) not in landcover_year_lookup:
            raise KeyError(f"Missing landcover year {year} for tile-native inference")

    short_pred_offsets = np.arange(short_max_lag, short_max_lag + n_days, dtype=np.int64)
    long_pred_offsets = np.arange(long_max_lag, long_max_lag + n_days, dtype=np.int64)

    modis_var_lookup = {var_name: idx for idx, var_name in enumerate(short_source_vars)}
    daymet_var_lookup = {var_name: idx for idx, var_name in enumerate(long_source_vars)}
    static_var_lookup_by_source = {
        source_name: {var_name: idx for idx, var_name in enumerate(vars_here)}
        for source_name, vars_here in static_source_vars_by_source.items()
    }
    landcover_var_lookup = {var_name: idx for idx, var_name in enumerate(landcover_vars)}

    lat_vals = tile_payload["lat"].astype(np.float32)
    lon_vals = tile_payload["lon"].astype(np.float32)

    for pix_start in range(0, n_pixels, pixel_batch_size):
        pix_end = min(pix_start + pixel_batch_size, n_pixels)
        pix_slice = slice(pix_start, pix_end)
        modis_y_local = modis_idx["local_iy"][pix_slice].astype(np.int64)
        modis_x_local = modis_idx["local_ix"][pix_slice].astype(np.int64)
        if daymet_idx is not None:
            daymet_y_local = daymet_idx["local_iy"][pix_slice].astype(np.int64)
            daymet_x_local = daymet_idx["local_ix"][pix_slice].astype(np.int64)
        else:
            daymet_y_local = None
            daymet_x_local = None
        landcover_y_local = landcover_idx["local_iy"][pix_slice].astype(np.int64)
        landcover_x_local = landcover_idx["local_ix"][pix_slice].astype(np.int64)
        batch_lat = lat_vals[pix_slice]
        batch_lon = lon_vals[pix_slice]

        modis_series = (
            modis_arr[:, :, modis_y_local, modis_x_local]
            if modis_arr is not None
            else None
        )
        daymet_series = (
            daymet_arr[:, :, daymet_y_local, daymet_x_local]
            if daymet_arr is not None
            else None
        )
        static_series_by_source = {}
        for source_name, source_arr in static_arr_by_source.items():
            if source_arr is None:
                static_series_by_source[source_name] = None
                continue
            source_idx = source_indexers[source_name]
            source_y_local = source_idx["local_iy"][pix_slice].astype(np.int64)
            source_x_local = source_idx["local_ix"][pix_slice].astype(np.int64)
            static_series_by_source[source_name] = (
                source_arr[:, source_y_local, source_x_local]
            )
        landcover_series = (
            landcover_arr[:, :, landcover_y_local, landcover_x_local]
            if landcover_arr is not None
            else None
        )

        for day_start in range(0, n_days, day_batch_size):
            day_end = min(day_start + day_batch_size, n_days)
            day_slice = slice(day_start, day_end)
            day_years = pred_years[day_slice]

            for s_idx, s_var in enumerate(short_vars):
                this_norm_mean = float(norm_params["train_short_mean"][s_idx])
                this_norm_std = float(norm_params["train_short_std"][s_idx])
                safe_std = this_norm_std if abs(this_norm_std) > 0 else 1.0
                if s_var == "lfrac":
                    lag_vals = _build_lag_fraction_values(short_lag_days, this_norm_mean, safe_std)
                    short_input[pix_slice, day_slice, :, s_idx] = lag_vals.reshape(1, 1, -1)
                    continue
                if var_to_source[s_var] != "modis":
                    raise NotImplementedError(f"Short variable source not implemented: {s_var}")
                var_idx = modis_var_lookup[s_var]
                gather_idx = short_pred_offsets[day_slice][:, None] - short_lag_days[None, :]
                vals = modis_series[gather_idx, var_idx, :].transpose(2, 0, 1)
                short_input[pix_slice, day_slice, :, s_idx] = (vals - this_norm_mean) / safe_std

            for l_idx, l_var in enumerate(long_vars):
                this_norm_mean = float(norm_params["train_long_mean"][l_idx])
                this_norm_std = float(norm_params["train_long_std"][l_idx])
                safe_std = this_norm_std if abs(this_norm_std) > 0 else 1.0
                if l_var == "lfrac":
                    lag_vals = _build_lag_fraction_values(long_lag_days, this_norm_mean, safe_std)
                    long_input[pix_slice, day_slice, :, l_idx] = lag_vals.reshape(1, 1, -1)
                    continue
                if var_to_source[l_var] != "daymet":
                    raise NotImplementedError(f"Long variable source not implemented: {l_var}")
                var_idx = daymet_var_lookup[l_var]
                gather_idx = long_pred_offsets[day_slice][:, None] - long_lag_days[None, :]
                vals = daymet_series[gather_idx, var_idx, :].transpose(2, 0, 1)
                long_input[pix_slice, day_slice, :, l_idx] = (vals - this_norm_mean) / safe_std

            for st_idx, st_var in enumerate(static_vars):
                this_norm_mean = float(norm_params["train_static_mean"][st_idx])
                this_norm_std = float(norm_params["train_static_std"][st_idx])
                safe_std = this_norm_std if abs(this_norm_std) > 0 else 1.0
                if st_var == "latitude":
                    vals = ((batch_lat - this_norm_mean) / safe_std).reshape(-1, 1)
                elif st_var == "longitude":
                    vals = ((batch_lon - this_norm_mean) / safe_std).reshape(-1, 1)
                elif _static_var_is_landcover(st_var):
                    var_idx = landcover_var_lookup[st_var]
                    out = np.empty((pix_end - pix_start, day_end - day_start), dtype=np.float32)
                    for offset, year in enumerate(day_years):
                        year_idx = landcover_year_lookup[int(year)]
                        out[:, offset] = landcover_series[year_idx, var_idx, :]
                    vals = out
                else:
                    source_name = var_to_source[st_var]
                    var_idx = static_var_lookup_by_source[source_name][st_var]
                    source_series = static_series_by_source[source_name]
                    vals = ((source_series[var_idx, :] - this_norm_mean) / safe_std).reshape(-1, 1)
                static_input[pix_slice, day_slice, 0, st_idx] = vals

    short_tensor = torch.tensor(
        short_input.reshape(n_pixels * n_days, len(short_lag_days), len(short_vars))
    )
    long_tensor = torch.tensor(
        long_input.reshape(n_pixels * n_days, len(long_lag_days), len(long_vars))
    )
    static_tensor = torch.tensor(
        static_input.reshape(n_pixels * n_days, 1, len(static_vars))
    )
    info_df = pd.DataFrame(
        {
            "lat": np.repeat(tile_payload["lat"].astype(np.float64), n_days),
            "lon": np.repeat(tile_payload["lon"].astype(np.float64), n_days),
            "date": np.tile(pred_dates.values, n_pixels),
        }
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[tile_tensor] built tile-native tensors for {n_pixels:,} pixels x {n_days} days "
        f"in {elapsed:.1f}s"
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


def build_reference_tensor_payload(
    tile_payload: Dict[str, np.ndarray],
    runtime: Dict[str, object],
    dss: Dict[str, xr.Dataset],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Dict[str, object]:
    return _tile_native_tensor_payload(
        tile_payload=tile_payload,
        runtime=runtime,
        dss=dss,
        start_date=start_date,
        end_date=end_date,
    )


def load_runtime_forward_predictor(
    runtime: Dict[str, object],
    model_type: str = DEFAULT_MODEL_TYPE,
    device: Optional[str] = None,
) -> Dict[str, object]:
    model, resolved_device = load_model_for_inference(
        short_input_dim=len(runtime["var_names"]["short_vars"]),
        long_input_dim=len(runtime["var_names"]["long_vars"]),
        static_input_dim=len(runtime["var_names"]["static_vars"]),
        model_path=runtime["checkpoint_path"],
        model_task_weights=runtime["model_num_tasks"],
        model_type=model_type,
        device=device,
    )
    return {
        "model": model,
        "device": resolved_device,
        "norm_params": runtime["norm_params"],
    }


def run_runtime_forward_loaded(
    predictor: Dict[str, object],
    tensor_payload: Dict[str, object],
    batch_size: int = 512,
    return_info_df: bool = False,
    use_cuda_autocast: bool = True,
) -> Dict[str, np.ndarray] | pd.DataFrame:
    preds = predict_with_loaded_model(
        tensor_payload["short_tensor"],
        tensor_payload["long_tensor"],
        tensor_payload["static_tensor"],
        model=predictor["model"],
        device=predictor["device"],
        norm_params=predictor["norm_params"],
        batch_size=batch_size,
        use_cuda_autocast=use_cuda_autocast,
    )
    if not return_info_df:
        return preds
    out = tensor_payload["info_df"].copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    out["lfmc_pred"] = np.asarray(preds["lfmc_pred"], dtype=np.float32)
    out["lfmc_pred_std"] = np.asarray(preds["lfmc_pred_std"], dtype=np.float32)
    out["vv_pred"] = np.asarray(preds["vv_pred"], dtype=np.float32)
    out["vv_pred_std"] = np.asarray(preds["vv_pred_std"], dtype=np.float32)
    out["vh_pred"] = np.asarray(preds["vh_pred"], dtype=np.float32)
    out["vh_pred_std"] = np.asarray(preds["vh_pred_std"], dtype=np.float32)
    return out


def initialize_running_ensemble_predictions(info_df: pd.DataFrame) -> Dict[str, object]:
    base = info_df[["lat", "lon", "date"]].copy().reset_index(drop=True)
    base["date"] = pd.to_datetime(base["date"])
    n = len(base)
    return {
        "base": base,
        "count": 0,
        "mean": np.zeros(n, dtype=np.float64),
        "m2": np.zeros(n, dtype=np.float64),
    }


def update_running_ensemble_predictions(
    accumulator: Dict[str, object],
    preds: np.ndarray,
) -> None:
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    mean = accumulator["mean"]
    if preds.shape[0] != mean.shape[0]:
        raise ValueError(
            f"Prediction length mismatch: expected {mean.shape[0]:,}, got {preds.shape[0]:,}"
        )
    accumulator["count"] += 1
    count = float(accumulator["count"])
    delta = preds - mean
    mean += delta / count
    accumulator["m2"] += delta * (preds - mean)


def finalize_running_ensemble_predictions(
    accumulator: Dict[str, object],
    mean_col_name: str = OUTPUT_MEAN_NAME,
    std_col_name: str = OUTPUT_STD_NAME,
) -> pd.DataFrame:
    count = int(accumulator["count"])
    if count <= 0:
        raise ValueError("Cannot finalize ensemble predictions without member predictions")
    variance = np.maximum(accumulator["m2"] / float(count), 0.0)
    out = accumulator["base"].copy()
    out["ensemble_n"] = count
    out[mean_col_name] = np.asarray(accumulator["mean"], dtype=np.float32)
    out[std_col_name] = np.sqrt(variance).astype(np.float32)
    return out


def run_runtime_forward(
    runtime: Dict[str, object],
    tensor_payload: Dict[str, object],
    model_type: str = DEFAULT_MODEL_TYPE,
    batch_size: int = 512,
    use_cuda_autocast: bool = True,
) -> pd.DataFrame:
    predictor = load_runtime_forward_predictor(runtime, model_type=model_type)
    return run_runtime_forward_loaded(
        predictor=predictor,
        tensor_payload=tensor_payload,
        batch_size=batch_size,
        return_info_df=True,
        use_cuda_autocast=use_cuda_autocast,
    )


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


def normalize_product_tier(product_tier: str) -> str:
    tier = str(product_tier).strip().lower()
    if tier not in QUALITY_FLAG_VALUES:
        raise ValueError(
            "product tier must be one of "
            f"{sorted(QUALITY_FLAG_VALUES.keys())}; got {product_tier!r}"
        )
    return tier


def _year_coord_to_int(value: object) -> int:
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return int(pd.Timestamp(value).year)
    return int(value)


def _resolve_landcover_source_year(
    available_years: Sequence[int],
    output_year: int,
) -> int:
    available_years = sorted(int(year) for year in available_years)
    if output_year in available_years:
        return int(output_year)
    prior_years = [year for year in available_years if year <= output_year]
    if len(prior_years) > 0:
        return int(prior_years[-1])
    return int(available_years[0])


def build_dominant_landcover_metadata(
    landcover_path: str,
    output_years: Sequence[int],
    target_x: Optional[Sequence[float]] = None,
    target_y: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    landcover_ds = xr.open_zarr(landcover_path)
    if "year" not in landcover_ds.coords:
        raise ValueError(f"Landcover store is missing 'year' coordinate: {landcover_path}")
    if (target_x is None) != (target_y is None):
        raise ValueError("target_x and target_y must both be provided together")
    if target_x is not None and target_y is not None:
        target_x = np.asarray(target_x, dtype=np.float64)
        target_y = np.asarray(target_y, dtype=np.float64)
        landcover_ds = landcover_ds.sel(
            x=xr.DataArray(target_x, dims=("x",)),
            y=xr.DataArray(target_y, dims=("y",)),
        )
    landcover_var_names = [
        var_name
        for var_name in landcover_ds.data_vars
        if {"year", "y", "x"}.issubset(set(landcover_ds[var_name].dims))
    ]
    if len(landcover_var_names) == 0:
        raise ValueError(f"No yearly landcover variables found in {landcover_path}")
    available_year_lookup = {
        _year_coord_to_int(coord_value): coord_value
        for coord_value in landcover_ds["year"].values
    }
    available_years = sorted(available_year_lookup.keys())
    requested_output_years = sorted({int(year) for year in output_years})
    y_size = int(landcover_ds.sizes["y"])
    x_size = int(landcover_ds.sizes["x"])
    dominant_codes = np.full(
        (len(requested_output_years), y_size, x_size),
        LANDCOVER_NODATA_CODE,
        dtype=np.uint8,
    )
    output_year_to_source_year: Dict[str, int] = {}
    code_to_name = {str(code): var_name for code, var_name in enumerate(landcover_var_names)}
    for year_idx, output_year in enumerate(requested_output_years):
        source_year = _resolve_landcover_source_year(available_years, output_year)
        output_year_to_source_year[str(output_year)] = int(source_year)
        if source_year != output_year:
            print(
                f"[map_runtime_utils] landcover year {output_year} not available; "
                f"using source year {source_year}"
            )
        best_fraction = np.full((y_size, x_size), -np.inf, dtype=np.float32)
        best_code = np.full((y_size, x_size), LANDCOVER_NODATA_CODE, dtype=np.uint8)
        year_coord_value = available_year_lookup[source_year]
        for code, var_name in enumerate(landcover_var_names):
            frac_vals = np.asarray(
                landcover_ds[var_name].sel(year=year_coord_value).values,
                dtype=np.float32,
            )
            valid = np.isfinite(frac_vals)
            better = valid & (frac_vals > best_fraction)
            best_fraction[better] = frac_vals[better]
            best_code[better] = np.uint8(code)
        dominant_codes[year_idx] = best_code
    return {
        "output_years": np.asarray(requested_output_years, dtype=np.int32),
        "dominant_landcover_code": dominant_codes,
        "code_to_name": code_to_name,
        "output_year_to_source_year": output_year_to_source_year,
        "landcover_var_names": list(landcover_var_names),
    }


def initialize_output_store(
    out_path: str,
    model_grid: xr.Dataset,
    time_index: pd.DatetimeIndex,
    time_chunk: int,
    y_chunk: int,
    x_chunk: int,
    landcover_metadata: Optional[Dict[str, object]] = None,
    product_tier: str = "final",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open_group(out_path, mode="w")
    product_tier = normalize_product_tier(product_tier)
    y_size = int(model_grid.sizes["y"])
    x_size = int(model_grid.sizes["x"])
    root.create_dataset(
        OUTPUT_MEAN_NAME,
        shape=(len(time_index), y_size, x_size),
        chunks=(time_chunk, y_chunk, x_chunk),
        dtype="f4",
        fill_value=np.nan,
        dimension_names=("time", "y", "x"),
        overwrite=True,
    )
    root[OUTPUT_MEAN_NAME].attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
    root[OUTPUT_MEAN_NAME].attrs["dimension_names"] = ["time", "y", "x"]
    root.create_dataset(
        OUTPUT_STD_NAME,
        shape=(len(time_index), y_size, x_size),
        chunks=(time_chunk, y_chunk, x_chunk),
        dtype="f4",
        fill_value=np.nan,
        dimension_names=("time", "y", "x"),
        overwrite=True,
    )
    root[OUTPUT_STD_NAME].attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
    root[OUTPUT_STD_NAME].attrs["dimension_names"] = ["time", "y", "x"]
    time_vals = np.asarray(time_index.values, dtype="datetime64[ns]").astype("int64")
    root.create_dataset(
        "time",
        data=time_vals,
        shape=time_vals.shape,
        chunks=(time_chunk,),
        dtype="i8",
        dimension_names=("time",),
        overwrite=True,
    )
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root["time"].attrs["dimension_names"] = ["time"]
    root["time"].attrs["units"] = "nanoseconds since 1970-01-01 00:00:00"
    root["time"].attrs["calendar"] = "proleptic_gregorian"
    quality_vals = np.full(
        len(time_index),
        np.uint8(QUALITY_FLAG_VALUES[product_tier]),
        dtype=np.uint8,
    )
    root.create_dataset(
        OUTPUT_QUALITY_FLAG_NAME,
        data=quality_vals,
        shape=quality_vals.shape,
        chunks=(time_chunk,),
        dtype="u1",
        fill_value=int(LANDCOVER_NODATA_CODE),
        dimension_names=("time",),
        overwrite=True,
    )
    root[OUTPUT_QUALITY_FLAG_NAME].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    root[OUTPUT_QUALITY_FLAG_NAME].attrs["dimension_names"] = ["time"]
    root[OUTPUT_QUALITY_FLAG_NAME].attrs["flag_values"] = [
        int(QUALITY_FLAG_VALUES["final"]),
        int(QUALITY_FLAG_VALUES["low_latency"]),
    ]
    root[OUTPUT_QUALITY_FLAG_NAME].attrs["flag_meanings"] = (
        "final_high_quality low_latency_preliminary"
    )
    root.attrs["quality_flag_key"] = json.dumps(
        {
            str(int(QUALITY_FLAG_VALUES["final"])): "high-quality, final",
            str(int(QUALITY_FLAG_VALUES["low_latency"])): "low-latency, preliminary",
            "2": "forced prediction on unapproved land cover",
        },
        sort_keys=True,
    )
    for coord_name in ["y", "x"]:
        vals = np.asarray(model_grid[coord_name].values)
        root.create_dataset(
            coord_name,
            data=vals,
            shape=vals.shape,
            chunks=(min(len(vals), y_chunk if coord_name == "y" else x_chunk),),
            dtype=str(vals.dtype),
            dimension_names=(coord_name,),
            overwrite=True,
        )
        root[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]
        root[coord_name].attrs["dimension_names"] = [coord_name]
    for coord_name in ["lat", "lon"]:
        if coord_name not in model_grid:
            continue
        vals = np.asarray(model_grid[coord_name].values)
        root.create_dataset(
            coord_name,
            data=vals,
            shape=vals.shape,
            chunks=(y_chunk, x_chunk),
            dtype=str(vals.dtype),
            dimension_names=("y", "x"),
            overwrite=True,
        )
        root[coord_name].attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
        root[coord_name].attrs["dimension_names"] = ["y", "x"]
    if landcover_metadata is not None:
        landcover_years = np.asarray(landcover_metadata["output_years"], dtype=np.int32)
        dominant_codes = np.asarray(
            landcover_metadata["dominant_landcover_code"],
            dtype=np.uint8,
        )
        root.create_dataset(
            OUTPUT_LANDCOVER_YEAR_NAME,
            data=landcover_years,
            shape=landcover_years.shape,
            chunks=(max(1, min(len(landcover_years), 32)),),
            dtype="i4",
            dimension_names=(OUTPUT_LANDCOVER_YEAR_NAME,),
            overwrite=True,
        )
        root[OUTPUT_LANDCOVER_YEAR_NAME].attrs["_ARRAY_DIMENSIONS"] = [
            OUTPUT_LANDCOVER_YEAR_NAME
        ]
        root[OUTPUT_LANDCOVER_YEAR_NAME].attrs["dimension_names"] = [
            OUTPUT_LANDCOVER_YEAR_NAME
        ]
        root.create_dataset(
            OUTPUT_DOMINANT_LANDCOVER_NAME,
            data=dominant_codes,
            shape=dominant_codes.shape,
            chunks=(1, y_chunk, x_chunk),
            dtype="u1",
            fill_value=int(LANDCOVER_NODATA_CODE),
            dimension_names=(OUTPUT_LANDCOVER_YEAR_NAME, "y", "x"),
            overwrite=True,
        )
        root[OUTPUT_DOMINANT_LANDCOVER_NAME].attrs["_ARRAY_DIMENSIONS"] = [
            OUTPUT_LANDCOVER_YEAR_NAME,
            "y",
            "x",
        ]
        root[OUTPUT_DOMINANT_LANDCOVER_NAME].attrs["dimension_names"] = [
            OUTPUT_LANDCOVER_YEAR_NAME,
            "y",
            "x",
        ]
        root[OUTPUT_DOMINANT_LANDCOVER_NAME].attrs["code_to_name"] = dict(
            landcover_metadata["code_to_name"]
        )
        root[OUTPUT_DOMINANT_LANDCOVER_NAME].attrs["nodata_code"] = int(
            LANDCOVER_NODATA_CODE
        )
        root[OUTPUT_DOMINANT_LANDCOVER_NAME].attrs["output_year_to_source_year"] = dict(
            landcover_metadata["output_year_to_source_year"]
        )
        root.attrs["dominant_landcover_code_key"] = json.dumps(
            dict(landcover_metadata["code_to_name"]),
            sort_keys=True,
        )
    root.attrs["quality_flag_values"] = {
        "final_high_quality": int(QUALITY_FLAG_VALUES["final"]),
        "low_latency_preliminary": int(QUALITY_FLAG_VALUES["low_latency"]),
        "forced_unapproved_landcover": 2,
    }


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
    fold: Optional[int] = None,
    split: str = "test",
    max_members: Optional[int] = None,
    member_name_prefix: Optional[str] = None,
    selection_key: Optional[str] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp, Dict[str, object]]:
    member_dirs = _select_member_subset(
        select_ensemble_member_dirs(
            ensemble_root,
            member_name_prefix=member_name_prefix,
            selection_key=selection_key,
        ),
        max_members=max_members,
    )
    member_site_errors = []
    for member_idx, member_dir in enumerate(member_dirs, start=1):
        member_site_errors.append(
            get_site_error(
                member_dir,
                progress_label=(
                    f"validation-month member {member_idx}/{len(member_dirs)} "
                    f"({os.path.basename(member_dir)})"
                ),
                fold=fold,
                split=split,
            )
        )
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


def validation_month_window_with_previous(
    best_month_start: pd.Timestamp,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    best_month_start = pd.Timestamp(best_month_start).normalize()
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    prev_month_start = (best_month_start - pd.offsets.MonthBegin(1)).normalize()
    window_start = max(prev_month_start, start_date)
    window_end = min((best_month_start + pd.offsets.MonthEnd(0)).normalize(), end_date)
    return window_start, window_end


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
        pred_std_here = None
        if "prediction_std" in site_data:
            pred_std_here = np.asarray(site_data["prediction_std"], dtype=float)[keep]
        candidates.append(
            {
                "site_key": site_key,
                "fold": str(site_data["fold"]),
                "num_measurements_month": int(np.sum(keep)),
                "dates": dates_here,
                "true_values": true_here,
                "predictions": pred_here,
                "prediction_std": pred_std_here,
            }
        )
    candidates = sorted(
        candidates,
        key=lambda x: (-x["num_measurements_month"], x["site_key"]),
    )
    return candidates[:n_sites]


def filter_site_records_to_valid_tiles(
    model_grid: xr.Dataset,
    site_records: Sequence[Dict[str, object]],
    tile_size: int,
    valid_tile_names: Sequence[str],
) -> List[Dict[str, object]]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x_coords = np.asarray(model_grid["x"].values, dtype=np.float64)
    y_coords = np.asarray(model_grid["y"].values, dtype=np.float64)
    valid_tile_name_set = set(valid_tile_names)
    filtered_records = []
    for site_record in site_records:
        lat_str, lon_str = site_record["site_key"].split("_")
        lat = float(lat_str)
        lon = float(lon_str)
        site_x, site_y = transformer.transform(lon, lat)
        x_idx = _nearest_index(x_coords, site_x)
        y_idx = _nearest_index(y_coords, site_y)
        tile_name = f"{x_idx // tile_size}_{y_idx // tile_size}"
        if tile_name in valid_tile_name_set:
            filtered_records.append(site_record)
    return filtered_records


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
