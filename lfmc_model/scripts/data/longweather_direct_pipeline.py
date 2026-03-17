import builtins
import concurrent.futures as cf
import datetime
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pyproj import CRS, Transformer
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _print_with_timestamp(*args, **kwargs):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{ts}]", *args, **kwargs)


print = _print_with_timestamp


def _nearest_index(coords: np.ndarray, val: float) -> int:
    coords = np.asarray(coords)
    if coords.ndim != 1:
        raise ValueError("Expected 1D coordinate array")
    if coords.size == 0:
        raise ValueError("Empty coordinate array")
    if coords.size == 1:
        return 0
    if coords[0] <= coords[-1]:
        idx = int(np.searchsorted(coords, val))
        if idx <= 0:
            return 0
        if idx >= coords.size:
            return coords.size - 1
        left = coords[idx - 1]
        right = coords[idx]
        return idx - 1 if abs(val - left) <= abs(right - val) else idx
    coords_rev = coords[::-1]
    idx_rev = int(np.searchsorted(coords_rev, val))
    if idx_rev <= 0:
        return coords.size - 1
    if idx_rev >= coords_rev.size:
        return 0
    left = coords_rev[idx_rev - 1]
    right = coords_rev[idx_rev]
    nearest_rev = idx_rev - 1 if abs(val - left) <= abs(right - val) else idx_rev
    return coords.size - 1 - nearest_rev


def _get_chunk_size(data_array: xr.DataArray, dim: str, fallback: int = 64) -> int:
    if hasattr(data_array, "chunksizes") and data_array.chunksizes and dim in data_array.chunksizes:
        return int(data_array.chunksizes[dim][0])
    if hasattr(data_array, "chunks") and data_array.chunks:
        dim_to_axis = {d: i for i, d in enumerate(data_array.dims)}
        if dim in dim_to_axis:
            axis = dim_to_axis[dim]
            return int(data_array.chunks[axis][0])
    return int(fallback)


def _normalize_time_to_day(ds: xr.Dataset) -> xr.Dataset:
    if "time" in ds.coords:
        ds = ds.copy()
        try:
            ds["time"] = pd.DatetimeIndex(ds.indexes["time"]).normalize()
        except Exception:
            ds["time"] = pd.to_datetime(ds["time"].values).normalize()
    return ds


def _safe_scalar(da_or_val) -> float:
    if hasattr(da_or_val, "values"):
        arr = np.asarray(da_or_val.values)
    else:
        arr = np.asarray(da_or_val)
    if arr.size == 0:
        return np.nan
    return float(np.ravel(arr)[0])


def _to_datetime_utc_naive(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, utc=True, errors="coerce")
    return out.dt.tz_convert(None)


def _extract_dataset_crs(ds: xr.Dataset) -> Optional[CRS]:
    candidates: List[object] = []
    for key in ["crs", "crs_wkt", "spatial_ref", "proj4", "proj4_params"]:
        if key in ds.attrs:
            candidates.append(ds.attrs[key])

    if "spatial_ref" in ds:
        sr = ds["spatial_ref"]
        for key in ["crs_wkt", "spatial_ref", "crs", "proj4", "proj4_params"]:
            if key in sr.attrs:
                candidates.append(sr.attrs[key])

    for cand in candidates:
        if cand is None:
            continue
        try:
            if isinstance(cand, bytes):
                cand = cand.decode("utf-8", errors="ignore")
            cand_str = str(cand).strip()
            if not cand_str:
                continue
            return CRS.from_user_input(cand_str)
        except Exception:
            continue
    return None


def _write_table(df: pd.DataFrame, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError("Output path must end with .parquet or .csv")


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Input path must end with .parquet or .csv")


def default_paths() -> Dict[str, str]:
    final_dir = "/scratch/users/trobinet/long_lfmc/final_lfmc"
    oak_dir = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets"
    return {"final_dir": final_dir, "oak_dir": oak_dir}


def default_dataset_paths() -> Dict[str, str]:
    paths = default_paths()
    final_dir = paths["final_dir"]
    scratch_dir = final_dir
    return {
        "daymet": os.path.join(final_dir, "daymet", "daymet_all_vars.zarr"),
        "modis": os.path.join(
            final_dir,
            "modis",
            "modis_regrid_interpolated",
            "modis_interp_5d.zarr",
        ),
        "static": os.path.join(final_dir, "static", "static_features_500m_epsg5070_float32.nc"),
        "climate_zone": os.path.join(final_dir, "climate_zones", "climate_zone_per_pixel_fullgrid.nc4"),
        "landcover_frac": os.path.join(final_dir, "nlcd", "nlcd_target_grid_2000_2024.zarr"),
        "nlcd": os.path.join(final_dir, "nlcd", "nlcd_2000_2024.zarr"),
        "sar_vh": os.path.join(scratch_dir, "sar", "sar_500m_filled.zarr"),
        "sar_vv": os.path.join(scratch_dir, "sar", "sar_500m_filled_vv.zarr"),
        "sar_ratios": os.path.join(scratch_dir, "sar", "sar_500m_filled_ratios.zarr"),
    }


def default_label_sources() -> Dict[str, str]:
    paths = default_paths()
    return {
        "nfmd": os.path.join(
            paths["oak_dir"],
            "nfmd",
            "nfmd_processed_landcovermatches.csv",
        )
    }


def default_short_features() -> List[str]:
    return [
        "Nadir_Reflectance_Band1_filled",
        "Nadir_Reflectance_Band2_filled",
        "Nadir_Reflectance_Band3_filled",
        "Nadir_Reflectance_Band4_filled",
        "Nadir_Reflectance_Band5_filled",
        "Nadir_Reflectance_Band6_filled",
        "Nadir_Reflectance_Band7_filled",
    ]


def default_long_features() -> List[str]:
    return ["srad", "prcp", "swe", "tmax", "vp"]


def default_static_features() -> List[str]:
    return [
        "slope",
        "elevation",
        "clay",
        "sand",
        "latitude",
        "longitude",
        "climate_zone_1",
        "climate_zone_2",
        "climate_zone_3",
        "climate_zone_4",
        "climate_zone_5",
        "climate_zone_6",
        "climate_zone_7",
        "climate_zone_8",
        "climate_zone_9",
        "climate_zone_10",
        "climate_zone_11",
        "climate_zone_12",
        "climate_zone_13",
        "climate_zone_14",
        "climate_zone_15",
        "climate_zone_16",
        "climate_zone_17",
        "climate_zone_18",
        "climate_zone_19",
        "climate_zone_20",
        "climate_zone_21",
        "climate_zone_22",
        "climate_zone_23",
        "climate_zone_24",
        "climate_zone_25",
        "climate_zone_26",
        "climate_zone_27",
        "climate_zone_28",
        "climate_zone_29",
        "barren",
        "crops",
        "deciduous_forest",
        "developed",
        "evergreen_forest",
        "grass",
        "mixed_forest",
        "other",
        "shrub",
        "water",
        "wetlands",
    ]


def default_info_features() -> List[str]:
    return ["date", "latitude", "longitude", "source", "source_legible"]


def default_short_lag_days() -> List[int]:
    return list(range(31))


def default_long_lag_days() -> List[int]:
    return list(range(365))


def default_var_locs() -> Dict[str, List[str]]:
    return {
        "modis": default_short_features(),
        "daymet": default_long_features(),
        "sar_vh": ["vh_backscatter"],
        "sar_vv": ["vv_backscatter"],
        "sar_ratios": [
            "vv_minus_vh",
            "vv_over_1",
            "vh_over_1",
            "vv_over_2",
            "vh_over_2",
            "vv_over_3",
            "vh_over_3",
        ],
        "static": ["slope", "elevation", "canopy_height", "clay", "sand"],
        "climate_zone": [f"climate_zone_{i}" for i in range(1, 30)],
        "landcover_frac": [
            "barren",
            "crops",
            "deciduous_forest",
            "developed",
            "evergreen_forest",
            "grass",
            "mixed_forest",
            "other",
            "shrub",
            "water",
            "wetlands",
        ],
    }


def default_var_names(
    static_features: Optional[Sequence[str]] = None,
    short_features: Optional[Sequence[str]] = None,
    long_features: Optional[Sequence[str]] = None,
    info_features: Optional[Sequence[str]] = None,
    target_cols: Optional[Sequence[str]] = None,
    include_lag_feature: bool = True,
) -> Dict[str, List[str]]:
    static_features = list(static_features or default_static_features())
    short_features = list(short_features or default_short_features())
    long_features = list(long_features or default_long_features())
    info_features = list(info_features or default_info_features())
    target_cols = list(target_cols or ["lfmc", "vh_backscatter"])
    return {
        "short_vars": short_features + (["lfrac"] if include_lag_feature else []),
        "long_vars": long_features + (["lfrac"] if include_lag_feature else []),
        "static_vars": static_features,
        "info_vars": info_features,
        "lfmc_vars": target_cols,
    }


def open_source_datasets(dataset_paths: Optional[Dict[str, str]] = None) -> Dict[str, xr.Dataset]:
    dataset_paths = dataset_paths or default_dataset_paths()
    print(f"[open_source_datasets] Requested sources: {list(dataset_paths.keys())}")

    def _open_any(path: str) -> xr.Dataset:
        if path.endswith(".zarr"):
            try:
                print(f"[open_source_datasets] Opening zarr (consolidated): {path}")
                return xr.open_zarr(path)
            except Exception:
                print(f"[open_source_datasets] Consolidated open failed, retry unconsolidated: {path}")
                return xr.open_zarr(path, consolidated=False)
        if path.endswith(".nc") or path.endswith(".nc4"):
            print(f"[open_source_datasets] Opening netcdf: {path}")
            return xr.open_dataset(path)
        raise ValueError(f"Unsupported dataset path type: {path}")

    opened = {}
    for ds_name, ds_path in dataset_paths.items():
        if ds_path is None or len(str(ds_path)) == 0:
            print(f"[open_source_datasets] Skipping '{ds_name}' (empty path)")
            continue
        if not os.path.exists(ds_path):
            print(f"[open_source_datasets] Skipping '{ds_name}' (missing): {ds_path}")
            continue
        opened[ds_name] = _open_any(ds_path)
        print(f"[open_source_datasets] Loaded '{ds_name}'")
    print(f"[open_source_datasets] Final loaded sources: {list(opened.keys())}")
    return opened


def build_sample_index_from_label_sources(
    label_sources: Dict[str, str],
    start_date: str,
    end_date: str,
    out_path: Optional[str] = None,
    sort_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    print(
        f"[build_sample_index_from_label_sources] Building sample index for "
        f"{start_date} to {end_date} from {len(label_sources)} source(s)"
    )
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)
    frames = []
    for source_name, label_path in label_sources.items():
        print(f"[build_sample_index_from_label_sources] Reading source '{source_name}': {label_path}")
        df = pd.read_csv(label_path)
        before = len(df)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        if "date" not in df.columns:
            raise ValueError(f"{label_path} is missing required 'date' column")
        df = df.copy()
        df["source"] = source_name
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        print(
            f"[build_sample_index_from_label_sources] Source '{source_name}' "
            f"rows: {before:,} -> {len(df):,} after date filter"
        )
        frames.append(df)
    if not frames:
        raise ValueError("No label sources provided")
    out = pd.concat(frames, ignore_index=True)
    print(f"[build_sample_index_from_label_sources] Concatenated rows: {len(out):,}")
    if sort_by:
        out = out.sort_values(list(sort_by)).reset_index(drop=True)
        print(f"[build_sample_index_from_label_sources] Applied sort by: {list(sort_by)}")
    if out_path is not None:
        _write_table(out, out_path)
        print(f"[build_sample_index_from_label_sources] Wrote sample index to {out_path}")
    return out


def load_sample_index(path_or_paths: Sequence[str]) -> pd.DataFrame:
    print(f"[load_sample_index] Loading {len(path_or_paths)} file(s)")
    frames = []
    for path in path_or_paths:
        print(f"[load_sample_index] Reading: {path}")
        frame = _read_table(path)
        print(f"[load_sample_index] Rows read: {len(frame):,}")
        frames.append(frame)
    if not frames:
        raise ValueError("No sample index paths provided")
    df = pd.concat(frames, ignore_index=True)
    print(f"[load_sample_index] Concatenated rows: {len(df):,}")
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    if "date" not in df.columns:
        raise ValueError("Sample index must contain a 'date' column")
    df = df.copy()
    df["date"] = _to_datetime_utc_naive(df["date"])
    print("[load_sample_index] Converted date column to UTC-naive pandas timestamps")
    return df


def _choose_target_per_row(df: pd.DataFrame, target_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    present_target_cols = [c for c in target_cols if c in out.columns]
    if not present_target_cols:
        raise ValueError(f"None of target columns found: {list(target_cols)}")
    counts = out[present_target_cols].notna().sum(axis=1)
    if (counts > 1).any():
        bad_n = int((counts > 1).sum())
        raise ValueError(f"{bad_n} rows have more than one target populated; expected one target per row")
    out = out[counts == 1].copy()
    out["target_name"] = ""
    out["target_value"] = np.nan
    for tcol in present_target_cols:
        mask = out[tcol].notna()
        out.loc[mask, "target_name"] = tcol
        out.loc[mask, "target_value"] = pd.to_numeric(out.loc[mask, tcol], errors="coerce")
    return out


def _source_encoding_for_target(target_name: str) -> Tuple[int, str]:
    if target_name == "lfmc":
        return 0, "nfmd"
    if target_name == "vv":
        return 1, "vv"
    if target_name == "vh":
        return 2, "vh"
    if target_name == 'vv_over_vh':
        return 1, "vv_over_vh"
    if target_name == 'vv_minus_vh':
        return 1, "vv_minus_vh"
    return 99, str(target_name)


def prepare_training_rows(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    acceptable_lfmc_range: Tuple[float, float] = (30.0, 500.0),
    num_rs_samples: int = 100_000_000,
    vh_locations: str = "all",
    target_sample_n: Optional[Dict[str, int]] = None,
    target_sample_fraction: Optional[Dict[str, float]] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    print(f"[prepare_training_rows] Starting with {len(df):,} raw row(s)")
    out = _choose_target_per_row(df, target_cols).copy()
    print(f"[prepare_training_rows] Rows after one-target enforcement: {len(out):,}")
    if "latitude" not in out.columns or "longitude" not in out.columns:
        raise ValueError("Sample index must contain 'latitude' and 'longitude' columns")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    before_drop = len(out)
    out = out.dropna(subset=["date", "latitude", "longitude", "target_value"])
    print(f"[prepare_training_rows] Rows after required-field dropna: {before_drop:,} -> {len(out):,}")
    lfmc_mask = out["target_name"] == "lfmc"
    before_lfmc = len(out)
    out = out[
        (~lfmc_mask) |
        (
            (out["target_value"] >= acceptable_lfmc_range[0]) &
            (out["target_value"] <= acceptable_lfmc_range[1])
        )
    ].copy()
    print(
        f"[prepare_training_rows] Rows after LFMC range filter {acceptable_lfmc_range}: "
        f"{before_lfmc:,} -> {len(out):,}"
    )

    source_codes = []
    source_legible = []
    for tgt in out["target_name"].tolist():
        code, legible = _source_encoding_for_target(tgt)
        source_codes.append(code)
        source_legible.append(legible)
    out["source_code"] = np.asarray(source_codes, dtype=np.int64)
    out["source_legible"] = source_legible

    if target_sample_n is None:
        target_sample_n = {
            "lfmc": -1,  # keep all by default
            "VV": int(num_rs_samples),
            "vh_backscatter": int(num_rs_samples),
        }
    if target_sample_fraction is None:
        target_sample_fraction = {}

    kept = []
    for target_name in out["target_name"].dropna().unique().tolist():
        this_target = out[out["target_name"] == target_name].copy()
        initial_target_rows = len(this_target)
        if target_name == "vh_backscatter":
            if vh_locations == "at_sites":
                this_target = this_target[this_target["source"].astype(str) == "vh_at_sites"].copy()
            elif vh_locations == "at_random":
                this_target = this_target[this_target["source"].astype(str) == "vh_at_random"].copy()
            elif vh_locations != "all":
                raise ValueError(f"Unknown vh_locations={vh_locations}")

        if this_target.empty:
            print(f"[prepare_training_rows] Target '{target_name}' has no rows after source filters")
            continue

        n_cfg = target_sample_n.get(target_name, -1)
        if n_cfg is not None and int(n_cfg) >= 0:
            n_keep = min(int(n_cfg), len(this_target))
            if n_keep < len(this_target):
                this_target = this_target.sample(n=n_keep, random_state=random_seed)
            print(
                f"[prepare_training_rows] Target '{target_name}': "
                f"{initial_target_rows:,} -> {len(this_target):,} via n cap ({n_cfg})"
            )
            kept.append(this_target)
            continue

        frac_cfg = target_sample_fraction.get(target_name, None)
        if frac_cfg is not None:
            frac = float(frac_cfg)
            if frac <= 0.0:
                continue
            if frac < 1.0:
                n_keep = int(round(frac * len(this_target)))
                n_keep = max(1, min(n_keep, len(this_target)))
                this_target = this_target.sample(n=n_keep, random_state=random_seed)
            print(
                f"[prepare_training_rows] Target '{target_name}': "
                f"{initial_target_rows:,} -> {len(this_target):,} via fraction ({frac_cfg})"
            )
            kept.append(this_target)
            continue

        print(
            f"[prepare_training_rows] Target '{target_name}': "
            f"{initial_target_rows:,} -> {len(this_target):,} (kept all)"
        )
        kept.append(this_target)

    if not kept:
        return out.iloc[0:0].copy()
    out = pd.concat(kept, ignore_index=True)
    out = out.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)
    print(f"[prepare_training_rows] Final prepared rows: {len(out):,}")
    return out


def build_training_sample_index_from_label_sources(
    label_sources: Dict[str, str],
    start_date: str,
    end_date: str,
    out_path: str,
    target_cols: Sequence[str],
    acceptable_lfmc_range: Tuple[float, float] = (30.0, 500.0),
    target_sample_n: Optional[Dict[str, int]] = None,
    target_sample_fraction: Optional[Dict[str, float]] = None,
    vh_locations: str = "all",
    random_seed: int = 42,
    sort_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    print("[build_training_sample_index_from_label_sources] Starting training sample index build")
    out = build_sample_index_from_label_sources(
        label_sources=label_sources,
        start_date=start_date,
        end_date=end_date,
        out_path=None,
        sort_by=sort_by,
    )
    out = prepare_training_rows(
        out,
        target_cols=target_cols,
        acceptable_lfmc_range=acceptable_lfmc_range,
        vh_locations=vh_locations,
        target_sample_n=target_sample_n,
        target_sample_fraction=target_sample_fraction,
        random_seed=random_seed,
    )
    print(f"[build_training_sample_index_from_label_sources] Writing {len(out):,} row(s) to {out_path}")
    _write_table(out, out_path)
    if "target_name" in out.columns:
        print(
            "[build_training_sample_index_from_label_sources] Target counts: "
            f"{out['target_name'].value_counts(dropna=False).to_dict()}"
        )
    return out


def _extract_series_for_lags(
    series: xr.DataArray,
    date: pd.Timestamp,
    lag_days: Sequence[int],
) -> np.ndarray:
    vals = np.full(len(lag_days), np.nan, dtype=np.float32)
    if "time" not in series.coords:
        return vals
    for i, lag in enumerate(lag_days):
        target_time = pd.Timestamp(date) - pd.Timedelta(days=int(lag))
        try:
            vals[i] = _safe_scalar(series.sel(time=target_time))
        except Exception:
            vals[i] = np.nan
    return vals


def _chunk_bounds_for_indices(
    x_idx: int,
    y_idx: int,
    x_chunk: int,
    y_chunk: int,
    x_size: int,
    y_size: int,
) -> Tuple[int, int, int, int, Tuple[int, int]]:
    chunk_key = (x_idx // x_chunk, y_idx // y_chunk)
    x0 = chunk_key[0] * x_chunk
    x1 = min((chunk_key[0] + 1) * x_chunk, x_size)
    y0 = chunk_key[1] * y_chunk
    y1 = min((chunk_key[1] + 1) * y_chunk, y_size)
    return x0, x1, y0, y1, chunk_key


def _invert_var_locs(var_locs: Dict[str, Sequence[str]]) -> Dict[str, str]:
    out = {}
    for source_name, vars_here in var_locs.items():
        for var_name in vars_here:
            if var_name in out and out[var_name] != source_name:
                raise ValueError(
                    f"Variable '{var_name}' appears in multiple sources: "
                    f"{out[var_name]} and {source_name}"
                )
            out[var_name] = source_name
    return out


def _has_prepared_target_columns(df: pd.DataFrame) -> bool:
    req = {"target_name", "target_value", "source_code", "source_legible"}
    return req.issubset(set(df.columns))


def _dataarray_for_chunking(ds: xr.Dataset) -> xr.DataArray:
    if "data" in ds.data_vars:
        return ds["data"]
    if not ds.data_vars:
        raise ValueError("Dataset has no data variables for chunk-size lookup")
    first_var = list(ds.data_vars)[0]
    return ds[first_var]


def _extract_temporal_series_from_point(point_ds: xr.Dataset, var_name: str) -> xr.DataArray:
    if "data" in point_ds.data_vars:
        data_da = point_ds["data"]
        if "variable" in data_da.coords:
            return data_da.sel(variable=var_name)
    if var_name in point_ds.data_vars:
        return point_ds[var_name]
    raise KeyError(f"Variable '{var_name}' not found in dataset variables: {list(point_ds.data_vars)}")


def _extract_temporal_dataarray(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    if "data" in ds.data_vars:
        data_da = ds["data"]
        if "variable" in data_da.coords:
            return data_da.sel(variable=var_name)
    if var_name in ds.data_vars:
        return ds[var_name]
    raise KeyError(f"Variable '{var_name}' not found in dataset variables: {list(ds.data_vars)}")


def _build_lag_tasks(
    row_indices: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    sample_dates: pd.Series,
    lag_days: Sequence[int],
    desc: Optional[str] = None,
) -> List[Dict[str, np.ndarray]]:
    tasks: List[Dict[str, np.ndarray]] = []
    if len(row_indices) == 0 or len(lag_days) == 0:
        return tasks
    sample_days = sample_dates.to_numpy(dtype="datetime64[D]")
    lag_iter = tqdm(
        list(enumerate(lag_days)),
        total=len(lag_days),
        desc=desc or "lag graph",
        unit="lag",
        leave=False,
    )
    for lag_idx, lag in lag_iter:
        need_days = sample_days - np.timedelta64(int(lag), "D")
        tasks.append(
            {
                "row_indices": row_indices,
                "x_idx": x_idx,
                "y_idx": y_idx,
                "need_dates": need_days.astype("datetime64[ns]"),
                "lag_idx": int(lag_idx),
            }
        )
    return tasks


def _map_need_dates_to_time_index(
    need_dates_ns: np.ndarray,
    time_index: Optional[pd.DatetimeIndex],
    time_lookup: Optional[Dict[str, object]],
) -> np.ndarray:
    if time_index is None:
        return np.full(need_dates_ns.shape[0], -1, dtype=np.int64)
    if time_lookup is not None and time_lookup.get("mode") == "daily_contiguous":
        time0_day = time_lookup["time0_day"]
        size = int(time_lookup["size"])
        need_days = need_dates_ns.astype("datetime64[D]")
        day_offsets = (need_days - time0_day).astype("timedelta64[D]").astype(np.int64)
        valid = (day_offsets >= 0) & (day_offsets < size)
        out = np.full(need_dates_ns.shape[0], -1, dtype=np.int64)
        out[valid] = day_offsets[valid]
        return out

    if time_lookup is not None and time_lookup.get("mode") == "searchsorted":
        time_vals_ns = time_lookup["time_vals_ns"]
    else:
        time_vals_ns = np.asarray(time_index.values, dtype="datetime64[ns]")
    idx = np.searchsorted(time_vals_ns, need_dates_ns)
    valid = (idx >= 0) & (idx < time_vals_ns.size)
    matched = np.zeros(idx.shape[0], dtype=bool)
    if np.any(valid):
        matched[valid] = time_vals_ns[idx[valid]] == need_dates_ns[valid]
    out = np.full(idx.shape[0], -1, dtype=np.int64)
    out[matched] = idx[matched].astype(np.int64)
    return out


def _split_tasks_to_native_chunk_keys(
    tasks: List[Dict[str, np.ndarray]],
    x_chunk_size: int,
    y_chunk_size: int,
    time_index: Optional[pd.DatetimeIndex],
    time_chunk_size: int,
    desc: Optional[str] = None,
    time_lookup: Optional[Dict[str, object]] = None,
) -> Dict[Tuple[int, int, int], List[Dict[str, np.ndarray]]]:
    out: Dict[Tuple[int, int, int], List[Dict[str, np.ndarray]]] = {}
    if not tasks:
        return out
    base_x_chunk_ids = (tasks[0]["x_idx"] // max(1, int(x_chunk_size))).astype(np.int64)
    base_y_chunk_ids = (tasks[0]["y_idx"] // max(1, int(y_chunk_size))).astype(np.int64)
    task_iter = tqdm(
        tasks,
        total=len(tasks),
        desc=desc or "split native keys",
        unit="lag_task",
        leave=False,
    )
    for task in task_iter:
        row_indices = task["row_indices"]
        x_idx = task["x_idx"]
        y_idx = task["y_idx"]
        need_dates = task["need_dates"]
        lag_idx = int(task["lag_idx"])
        if time_index is None:
            valid_mask = np.ones(row_indices.shape[0], dtype=bool)
            time_chunk_ids = np.zeros(row_indices.shape[0], dtype=np.int64)
        else:
            date_idx = _map_need_dates_to_time_index(
                need_dates_ns=need_dates,
                time_index=time_index,
                time_lookup=time_lookup,
            )
            valid_mask = date_idx >= 0
            if not np.any(valid_mask):
                continue
            time_chunk_ids = (date_idx[valid_mask] // max(1, int(time_chunk_size))).astype(np.int64)
            time_idx_global = date_idx[valid_mask].astype(np.int64)

        row_indices = row_indices[valid_mask]
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        if time_index is None:
            time_idx_global = np.zeros(row_indices.shape[0], dtype=np.int64)
        x_chunk_ids = base_x_chunk_ids[valid_mask]
        y_chunk_ids = base_y_chunk_ids[valid_mask]
        key_triplets = np.stack([time_chunk_ids, x_chunk_ids, y_chunk_ids], axis=1)
        unique_keys, inverse = np.unique(key_triplets, axis=0, return_inverse=True)
        for key_i, key_vals in enumerate(unique_keys):
            keep = inverse == key_i
            if not np.any(keep):
                continue
            key = (int(key_vals[0]), int(key_vals[1]), int(key_vals[2]))
            out.setdefault(key, []).append(
                {
                    "row_indices": row_indices[keep],
                    "x_idx": x_idx[keep],
                    "y_idx": y_idx[keep],
                    "t_idx_global": time_idx_global[keep],
                    "lag_idx": int(lag_idx),
                }
            )
    return out


def _split_points_by_xy_chunks(
    row_indices: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    x_chunk_size: int,
    y_chunk_size: int,
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    out: Dict[Tuple[int, int, int], Dict[str, np.ndarray]] = {}
    if row_indices.size == 0:
        return out
    x_chunk_ids = (x_idx // max(1, int(x_chunk_size))).astype(np.int64)
    y_chunk_ids = (y_idx // max(1, int(y_chunk_size))).astype(np.int64)
    key_triplets = np.stack(
        [np.zeros(row_indices.shape[0], dtype=np.int64), x_chunk_ids, y_chunk_ids], axis=1
    )
    unique_keys, inverse = np.unique(key_triplets, axis=0, return_inverse=True)
    for key_i, key_vals in enumerate(unique_keys):
        keep = inverse == key_i
        if not np.any(keep):
            continue
        key = (int(key_vals[0]), int(key_vals[1]), int(key_vals[2]))
        out[key] = {
            "row_indices": row_indices[keep],
            "x_idx": x_idx[keep],
            "y_idx": y_idx[keep],
        }
    return out


def _split_points_by_txy_chunks(
    row_indices: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    t_idx_global: np.ndarray,
    x_chunk_size: int,
    y_chunk_size: int,
    t_chunk_size: int,
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    out: Dict[Tuple[int, int, int], Dict[str, np.ndarray]] = {}
    if row_indices.size == 0:
        return out
    valid = t_idx_global >= 0
    if not np.any(valid):
        return out
    row_indices = row_indices[valid]
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    t_idx_global = t_idx_global[valid]
    t_chunk_ids = (t_idx_global // max(1, int(t_chunk_size))).astype(np.int64)
    x_chunk_ids = (x_idx // max(1, int(x_chunk_size))).astype(np.int64)
    y_chunk_ids = (y_idx // max(1, int(y_chunk_size))).astype(np.int64)
    key_triplets = np.stack([t_chunk_ids, x_chunk_ids, y_chunk_ids], axis=1)
    unique_keys, inverse = np.unique(key_triplets, axis=0, return_inverse=True)
    for key_i, key_vals in enumerate(unique_keys):
        keep = inverse == key_i
        if not np.any(keep):
            continue
        key = (int(key_vals[0]), int(key_vals[1]), int(key_vals[2]))
        out[key] = {
            "row_indices": row_indices[keep],
            "x_idx": x_idx[keep],
            "y_idx": y_idx[keep],
            "t_idx_global": t_idx_global[keep],
        }
    return out


def _assign_loaded_values_from_tasks(
    target_np: np.ndarray,
    loaded_txy: np.ndarray,
    loaded_t0: int,
    loaded_x0: int,
    loaded_y0: int,
    var_slot: int,
    tasks: List[Dict[str, np.ndarray]],
    global_to_local_t: Optional[np.ndarray] = None,
) -> None:
    if not tasks:
        return
    arr_txy = np.asarray(loaded_txy)
    if arr_txy.ndim != 3:
        return
    t_size, x_size_local, y_size_local = arr_txy.shape
    for task in tasks:
        row_idxs = task["row_indices"]
        t_idx_global = task["t_idx_global"].astype(np.int64)
        if global_to_local_t is not None:
            valid_global = (t_idx_global >= 0) & (t_idx_global < global_to_local_t.shape[0])
            local_t = np.full(t_idx_global.shape[0], -1, dtype=np.int64)
            if np.any(valid_global):
                local_t[valid_global] = global_to_local_t[t_idx_global[valid_global]]
        else:
            local_t = (t_idx_global - int(loaded_t0)).astype(np.int64)
        local_x = (task["x_idx"] - int(loaded_x0)).astype(np.int64)
        local_y = (task["y_idx"] - int(loaded_y0)).astype(np.int64)
        lag_idx = int(task["lag_idx"])
        valid = (
            (local_t >= 0) & (local_t < t_size) &
            (local_x >= 0) & (local_x < x_size_local) &
            (local_y >= 0) & (local_y < y_size_local)
        )
        if not np.any(valid):
            continue
        vals = arr_txy[local_t[valid], local_x[valid], local_y[valid]]
        target_np[row_idxs[valid], var_slot, lag_idx] = np.asarray(vals, dtype=np.float32)


def _assign_loaded_xy_feature(
    target_np: np.ndarray,
    var_slot: Optional[int],
    loaded_ds: xr.Dataset,
    loaded_x0: int,
    loaded_y0: int,
    var_name: str,
    task: Dict[str, np.ndarray],
    onehot_value: Optional[int] = None,
) -> None:
    try:
        da = _extract_temporal_dataarray(loaded_ds, var_name)
    except Exception:
        return
    if "x" not in da.dims or "y" not in da.dims:
        return
    extra_dims = [d for d in da.dims if d not in {"x", "y"}]
    for d in extra_dims:
        if int(da.sizes.get(d, 0)) == 1:
            da = da.isel({d: 0})
        else:
            print(
                f"Warning: static var '{var_name}' has unsupported non-singleton dim '{d}' "
                f"with size={int(da.sizes.get(d, 0))}; skipping this var for this chunk."
            )
            return
    arr_xy = np.asarray(da.transpose("x", "y").values)
    row_idxs = task["row_indices"]
    local_x = (task["x_idx"] - int(loaded_x0)).astype(np.int64)
    local_y = (task["y_idx"] - int(loaded_y0)).astype(np.int64)
    valid = (
        (local_x >= 0) & (local_x < arr_xy.shape[0]) &
        (local_y >= 0) & (local_y < arr_xy.shape[1])
    )
    if not np.any(valid):
        return
    vals = arr_xy[local_x[valid], local_y[valid]]
    if onehot_value is not None:
        vals_num = np.asarray(vals, dtype=np.float32)
        out_vals = np.zeros(vals_num.shape, dtype=np.float32)
        finite = np.isfinite(vals_num)
        if np.any(finite):
            out_vals[finite] = (
                vals_num[finite].astype(np.int64) == int(onehot_value)
            ).astype(np.float32)
        vals = out_vals
    else:
        vals = np.asarray(vals, dtype=np.float32)
    if target_np.ndim == 2 and var_slot is not None:
        target_np[row_idxs[valid], int(var_slot)] = vals
    elif target_np.ndim == 1:
        target_np[row_idxs[valid]] = vals


def _assign_loaded_txy_feature(
    target_np: np.ndarray,
    var_slot: Optional[int],
    loaded_ds: xr.Dataset,
    t_dim: str,
    loaded_t0: int,
    loaded_x0: int,
    loaded_y0: int,
    var_name: str,
    task: Dict[str, np.ndarray],
) -> None:
    try:
        da = _extract_temporal_dataarray(loaded_ds, var_name)
    except Exception:
        return
    if t_dim not in da.dims or "x" not in da.dims or "y" not in da.dims:
        return
    extra_dims = [d for d in da.dims if d not in {t_dim, "x", "y"}]
    for d in extra_dims:
        if int(da.sizes.get(d, 0)) == 1:
            da = da.isel({d: 0})
        else:
            print(
                f"Warning: indexed var '{var_name}' has unsupported non-singleton dim '{d}' "
                f"with size={int(da.sizes.get(d, 0))}; skipping this var for this chunk."
            )
            return
    arr_txy = np.asarray(da.transpose(t_dim, "x", "y").values)
    row_idxs = task["row_indices"]
    local_t = (task["t_idx_global"] - int(loaded_t0)).astype(np.int64)
    local_x = (task["x_idx"] - int(loaded_x0)).astype(np.int64)
    local_y = (task["y_idx"] - int(loaded_y0)).astype(np.int64)
    valid = (
        (local_t >= 0) & (local_t < arr_txy.shape[0]) &
        (local_x >= 0) & (local_x < arr_txy.shape[1]) &
        (local_y >= 0) & (local_y < arr_txy.shape[2])
    )
    if not np.any(valid):
        return
    vals = np.asarray(arr_txy[local_t[valid], local_x[valid], local_y[valid]], dtype=np.float32)
    if target_np.ndim == 2 and var_slot is not None:
        target_np[row_idxs[valid], int(var_slot)] = vals
    elif target_np.ndim == 1:
        target_np[row_idxs[valid]] = vals


def _extract_chunk_subsets(
    dss: Dict[str, xr.Dataset],
    temporal_source_max_lag: Dict[str, int],
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> Dict[str, xr.Dataset]:
    out = {}

    for source_name, max_lag in temporal_source_max_lag.items():
        if source_name not in dss:
            raise KeyError(f"Requested source '{source_name}' is not loaded in dss")
        ds = dss[source_name].isel(x=slice(x0, x1), y=slice(y0, y1))
        if "time" in ds.coords:
            ds = ds.sel(
                time=slice(
                    min_date - pd.Timedelta(days=int(max_lag)),
                    max_date,
                )
            ).compute()
            ds = _normalize_time_to_day(ds)
        else:
            ds = ds.load()
        out[source_name] = ds

    for static_source in ["landcover_frac", "nlcd", "static", "climate_zone"]:
        if static_source in dss and static_source not in out:
            out[static_source] = dss[static_source].isel(
                x=slice(x0, x1), y=slice(y0, y1)
            ).load()

    return out


def _extract_static_feature_value(
    feature_name: str,
    row: pd.Series,
    chunk_dss: Dict[str, xr.Dataset],
    local_x: int,
    local_y: int,
    var_to_source: Optional[Dict[str, str]] = None,
) -> float:
    if feature_name == "latitude":
        return float(row["latitude"])
    if feature_name == "longitude":
        return float(row["longitude"])
    if feature_name.startswith("climate_zone_"):
        if "climate_zone" not in chunk_dss:
            return np.nan
        want = int(feature_name.split("_")[-1])
        here = _safe_scalar(chunk_dss["climate_zone"]["climate_zone"].isel(x=local_x, y=local_y))
        if np.isnan(here):
            return np.nan
        return 1.0 if int(here) == want else 0.0
    if "landcover_frac" in chunk_dss and feature_name in chunk_dss["landcover_frac"].data_vars:
        year_key = pd.Timestamp(int(pd.Timestamp(row["date"]).year), 1, 1)
        try:
            return _safe_scalar(
                chunk_dss["landcover_frac"][feature_name]
                .sel(year=year_key)
                .isel(x=local_x, y=local_y)
            )
        except Exception:
            return np.nan
    if "static" in chunk_dss and feature_name in chunk_dss["static"].data_vars:
        return _safe_scalar(chunk_dss["static"][feature_name].isel(x=local_x, y=local_y))
    if var_to_source and feature_name in var_to_source:
        source_name = var_to_source[feature_name]
        if source_name in chunk_dss:
            ds = chunk_dss[source_name]
            if feature_name in ds.data_vars:
                da = ds[feature_name]
                if "year" in da.dims:
                    year_key = pd.Timestamp(int(pd.Timestamp(row["date"]).year), 1, 1)
                    da = da.sel(year=year_key)
                elif "time" in da.dims:
                    da = da.sel(time=pd.Timestamp(row["date"]).normalize(), method="nearest")
                return _safe_scalar(da.isel(x=local_x, y=local_y))
    return np.nan


def _extract_stratifier_value(
    stratifier: str,
    row: pd.Series,
    chunk_dss: Dict[str, xr.Dataset],
    local_x: int,
    local_y: int,
) -> float:
    if stratifier == "nlcd":
        if "nlcd" not in chunk_dss:
            return np.nan
        year_key = np.datetime64(f"{pd.Timestamp(row['date']).year}-01-01")
        try:
            return _safe_scalar(
                chunk_dss["nlcd"]["nlcd"].sel(time=year_key).isel(x=local_x, y=local_y)
            )
        except Exception:
            return np.nan
    if stratifier in row.index:
        try:
            return float(row[stratifier])
        except Exception:
            return np.nan
    return np.nan


def _drop_zero_variance_channels(
    arr: np.ndarray,
    feature_names: Sequence[str],
    group_name: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    feature_names = list(feature_names)
    if arr.ndim != 3:
        raise ValueError(f"{group_name}: expected 3D array, got shape {arr.shape}")
    if arr.shape[2] != len(feature_names):
        raise ValueError(
            f"{group_name}: feature name length ({len(feature_names)}) does not match "
            f"array channel size ({arr.shape[2]})"
        )
    if arr.shape[0] == 0:
        return arr, feature_names, []

    variances = np.nanvar(arr, axis=(0, 1))
    keep_mask = np.isfinite(variances) & (variances > 0.0)
    kept_features = [f for f, keep in zip(feature_names, keep_mask.tolist()) if keep]
    dropped_features = [f for f, keep in zip(feature_names, keep_mask.tolist()) if not keep]
    if not kept_features:
        print(
            f"Warning: all {group_name} features have zero variance after filtering. "
            f"Keeping all {len(feature_names)} feature(s) unchanged."
        )
        return arr, feature_names, []
    filtered = arr[:, :, keep_mask]
    return filtered, kept_features, dropped_features


def _format_seconds_hms(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class DirectBuildResult:
    X_short: torch.Tensor
    X_long: torch.Tensor
    X_static: torch.Tensor
    Y: torch.Tensor
    source: torch.Tensor
    info: pd.DataFrame
    stratifier: np.ndarray
    var_names: Dict[str, List[str]]
    keep_mask: np.ndarray
    build_metadata: Optional[Dict[str, object]] = None


def _extract_row_climate_zone_codes_from_static(
    static_arr: np.ndarray,
    static_feature_names: Sequence[str],
) -> np.ndarray:
    climate_slots: List[int] = []
    climate_codes: List[int] = []
    for idx, feat in enumerate(static_feature_names):
        if feat.startswith("climate_zone_"):
            climate_slots.append(idx)
            climate_codes.append(int(feat.split("_")[-1]))
    if not climate_slots:
        raise ValueError("No climate zone features found in static feature list")

    static_np = np.asarray(static_arr)
    if static_np.ndim == 3:
        static_np = static_np[:, 0, :]
    if static_np.ndim != 2:
        raise ValueError(f"Expected static array with 2 or 3 dims, got shape {static_np.shape}")

    climate_block = static_np[:, climate_slots]
    active_mask = climate_block > 0.5
    active_counts = active_mask.sum(axis=1)
    if np.any(active_counts == 0):
        raise ValueError(
            "Found rows with no active climate zone one-hot channel in static tensor"
        )
    if np.any(active_counts > 1):
        raise ValueError(
            "Found rows with multiple active climate zone one-hot channels in static tensor"
        )
    return np.asarray(climate_codes, dtype=np.int16)[np.argmax(active_mask, axis=1)]


def _filter_rows_by_min_sites_per_climate_zone(
    rows: pd.DataFrame,
    X_short: np.ndarray,
    X_long: np.ndarray,
    X_static: np.ndarray,
    Y: np.ndarray,
    source: np.ndarray,
    stratifier: np.ndarray,
    static_feature_names: Sequence[str],
    min_sites_per_zone: int = 2,
    round_decimals: int = 10,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, object],
]:
    climate_codes = _extract_row_climate_zone_codes_from_static(
        X_static, static_feature_names
    )
    rows_with_climate = rows.copy()
    rows_with_climate["climate_zone_code"] = climate_codes
    rows_with_climate["site_lat"] = pd.to_numeric(
        rows_with_climate["latitude"], errors="coerce"
    ).round(round_decimals)
    rows_with_climate["site_lon"] = pd.to_numeric(
        rows_with_climate["longitude"], errors="coerce"
    ).round(round_decimals)

    site_counts_by_zone = (
        rows_with_climate[["climate_zone_code", "site_lat", "site_lon"]]
        .drop_duplicates()
        .groupby("climate_zone_code")
        .size()
        .astype(int)
    )
    kept_codes = sorted(
        int(code) for code, count in site_counts_by_zone.items() if int(count) >= min_sites_per_zone
    )
    dropped_codes = sorted(
        int(code) for code, count in site_counts_by_zone.items() if int(count) < min_sites_per_zone
    )
    keep_rows = rows_with_climate["climate_zone_code"].isin(kept_codes).to_numpy(dtype=bool)

    kept_rows = rows.loc[keep_rows].copy().reset_index(drop=True)
    kept_climate_codes = climate_codes[keep_rows]
    metadata = {
        "min_sites_per_climate_zone": int(min_sites_per_zone),
        "kept_climate_zone_codes": kept_codes,
        "dropped_climate_zone_codes": dropped_codes,
        "dropped_rows_due_to_rare_climate_zone": int((~keep_rows).sum()),
        "dropped_sites_due_to_rare_climate_zone": int(
            site_counts_by_zone.loc[dropped_codes].sum() if dropped_codes else 0
        ),
        "site_counts_by_climate_zone": {
            str(int(code)): int(count) for code, count in site_counts_by_zone.items()
        },
    }
    print(
        "[build_direct_tensors] Climate-zone site filter: "
        f"kept_codes={kept_codes}, dropped_codes={dropped_codes}, "
        f"dropped_rows={metadata['dropped_rows_due_to_rare_climate_zone']:,}, "
        f"dropped_sites={metadata['dropped_sites_due_to_rare_climate_zone']:,}"
    )

    return (
        kept_rows,
        X_short[keep_rows],
        X_long[keep_rows],
        X_static[keep_rows],
        Y[keep_rows],
        source[keep_rows],
        stratifier[keep_rows],
        kept_climate_codes,
        keep_rows,
        metadata,
    )


def build_direct_tensors_from_sample_index(
    sample_df: pd.DataFrame,
    dss: Dict[str, xr.Dataset],
    short_features: Sequence[str],
    long_features: Sequence[str],
    static_features: Sequence[str],
    short_lag_days: Sequence[int],
    long_lag_days: Sequence[int],
    target_cols: Optional[Sequence[str]] = None,
    stratifier: str = "nlcd",
    include_lag_feature: bool = True,
    acceptable_lfmc_range: Tuple[float, float] = (30.0, 500.0),
    num_rs_samples: int = 100_000_000,
    vh_locations: str = "all",
    target_sample_n: Optional[Dict[str, int]] = None,
    target_sample_fraction: Optional[Dict[str, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    var_locs: Optional[Dict[str, Sequence[str]]] = None,
    grid_source: str = "modis",
    assume_prepared_index: Optional[bool] = None,
    temporal_batch_mode: str = "all_needed_dates_per_xy_chunk",
    temporal_month_block_size: int = 12,
    temporal_num_workers: int = 1,
    temporal_max_inflight: int = 2,
) -> DirectBuildResult:
    print(f"[build_direct_tensors] Input sample rows: {len(sample_df):,}")
    rows = sample_df.copy()
    rows["date"] = _to_datetime_utc_naive(rows["date"])
    before_date_filter = len(rows)
    if start_date is not None:
        rows = rows[rows["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        rows = rows[rows["date"] <= pd.to_datetime(end_date)]
    print(
        f"[build_direct_tensors] Rows after date filter "
        f"(start={start_date}, end={end_date}): {before_date_filter:,} -> {len(rows):,}"
    )

    if assume_prepared_index is None:
        assume_prepared_index = _has_prepared_target_columns(rows)
    print(f"[build_direct_tensors] assume_prepared_index={assume_prepared_index}")
    if temporal_batch_mode == "needed_days_in_month_block":
        temporal_batch_mode = "needed_days_in_month_blocks"
    if temporal_batch_mode not in {
        "all_needed_dates_per_xy_chunk",
        "year",
        "native_chunk",
        "needed_days_in_month_blocks",
    }:
        raise ValueError(
            f"Unsupported temporal_batch_mode={temporal_batch_mode}. "
            "Use one of: all_needed_dates_per_xy_chunk, year, native_chunk, "
            "needed_days_in_month_blocks (alias: needed_days_in_month_block)"
        )
    temporal_month_block_size = int(temporal_month_block_size)
    if temporal_month_block_size <= 0:
        raise ValueError("temporal_month_block_size must be >= 1")
    temporal_num_workers = max(1, int(temporal_num_workers))
    temporal_max_inflight = max(1, int(temporal_max_inflight))

    if assume_prepared_index:
        req = ["target_name", "target_value", "source_code", "source_legible"]
        missing = [c for c in req if c not in rows.columns]
        if missing:
            raise ValueError(
                f"Sample index is marked as prepared but missing columns: {missing}"
            )
        rows = rows.copy()
        rows["latitude"] = pd.to_numeric(rows["latitude"], errors="coerce")
        rows["longitude"] = pd.to_numeric(rows["longitude"], errors="coerce")
        rows["target_name"] = rows["target_name"].astype(str)
        rows["target_value"] = pd.to_numeric(rows["target_value"], errors="coerce")
        rows["source_code"] = pd.to_numeric(rows["source_code"], errors="coerce")
        rows = rows.dropna(
            subset=["date", "latitude", "longitude", "target_name", "target_value", "source_code"]
        )
        rows["source_code"] = rows["source_code"].astype(np.int64)
        rows["source_legible"] = rows["source_legible"].astype(str)
        print(f"[build_direct_tensors] Prepared index rows after validation/dropna: {len(rows):,}")
    else:
        if target_cols is None or len(target_cols) == 0:
            raise ValueError(
                "target_cols must be provided when assume_prepared_index=False"
            )
        rows = prepare_training_rows(
            rows,
            target_cols=target_cols,
            acceptable_lfmc_range=acceptable_lfmc_range,
            num_rs_samples=num_rs_samples,
            vh_locations=vh_locations,
            target_sample_n=target_sample_n,
            target_sample_fraction=target_sample_fraction,
        )
        print(f"[build_direct_tensors] Rows after on-the-fly preparation: {len(rows):,}")
    rows = rows.reset_index(drop=True)

    short_vars_out = list(short_features) + (["lfrac"] if include_lag_feature else [])
    long_vars_out = list(long_features) + (["lfrac"] if include_lag_feature else [])
    static_features = list(static_features)
    short_features = list(short_features)
    long_features = list(long_features)
    var_locs = dict(var_locs or default_var_locs())
    var_to_source = _invert_var_locs(var_locs)
    temporal_features = list(dict.fromkeys(short_features + long_features))
    for var_name in temporal_features:
        if var_name not in var_to_source:
            raise ValueError(
                f"Feature '{var_name}' is not mapped in var_locs. Add it to your config."
            )
        source_name = var_to_source[var_name]
        if source_name not in dss:
            raise ValueError(
                f"Feature '{var_name}' maps to source '{source_name}', "
                f"but '{source_name}' is not loaded in dss."
            )

    n = len(rows)
    if n == 0:
        raise ValueError("No rows remaining after filtering/sampling")
    print(
        f"[build_direct_tensors] Tensor dimensions target: N={n:,}, "
        f"short=({len(short_lag_days)} lags x {len(short_vars_out)} vars), "
        f"long=({len(long_lag_days)} lags x {len(long_vars_out)} vars), "
        f"static={len(static_features)} vars"
    )

    if grid_source not in dss:
        raise ValueError(f"grid_source '{grid_source}' not found in loaded datasets")

    lon_vals = rows["longitude"].to_numpy(dtype=np.float64)
    lat_vals = rows["latitude"].to_numpy(dtype=np.float64)
    source_point_index: Dict[str, Dict[str, np.ndarray]] = {}
    for source_name, src_ds in dss.items():
        if "x" not in src_ds.coords or "y" not in src_ds.coords:
            continue
        src_crs = _extract_dataset_crs(src_ds)
        if src_crs is None:
            print(
                f"Warning: source '{source_name}' missing CRS metadata; "
                "defaulting to EPSG:5070 for sample-point projection."
            )
            src_crs = CRS.from_epsg(5070)
        trns_src = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        xs_src, ys_src = trns_src.transform(lon_vals, lat_vals)
        src_x_coords = np.asarray(src_ds["x"].values)
        src_y_coords = np.asarray(src_ds["y"].values)
        src_x_idx = np.array([_nearest_index(src_x_coords, v) for v in xs_src], dtype=np.int64)
        src_y_idx = np.array([_nearest_index(src_y_coords, v) for v in ys_src], dtype=np.int64)
        source_point_index[source_name] = {
            "x": np.asarray(xs_src, dtype=np.float64),
            "y": np.asarray(ys_src, dtype=np.float64),
            "x_idx": src_x_idx,
            "y_idx": src_y_idx,
        }
        print(
            f"[build_direct_tensors] Source indexing '{source_name}': "
            f"x_idx[{int(src_x_idx.min())}, {int(src_x_idx.max())}] / {src_x_coords.size - 1}, "
            f"y_idx[{int(src_y_idx.min())}, {int(src_y_idx.max())}] / {src_y_coords.size - 1}"
        )

    if grid_source not in source_point_index:
        raise ValueError(
            f"grid_source '{grid_source}' has no x/y coordinates for sample-point indexing"
        )

    grid_ds = dss[grid_source]
    x_coords = grid_ds["x"].values
    y_coords = grid_ds["y"].values
    chunk_da = _dataarray_for_chunking(grid_ds)
    x_chunk = _get_chunk_size(chunk_da, "x", fallback=64)
    y_chunk = _get_chunk_size(chunk_da, "y", fallback=64)
    rows["x"] = source_point_index[grid_source]["x"]
    rows["y"] = source_point_index[grid_source]["y"]
    rows["x_idx"] = source_point_index[grid_source]["x_idx"]
    rows["y_idx"] = source_point_index[grid_source]["y_idx"]

    x_size = x_coords.size
    y_size = y_coords.size
    print(
        f"[build_direct_tensors] Grid/chunk info: x_size={x_size}, y_size={y_size}, "
        f"x_chunk={x_chunk}, y_chunk={y_chunk}"
    )
    short_lag_days = [int(v) for v in short_lag_days]
    long_lag_days = [int(v) for v in long_lag_days]
    short_max_lag = max(short_lag_days) if short_lag_days else 0
    long_max_lag = max(long_lag_days) if long_lag_days else 0
    temporal_source_max_lag = {}
    for var_name in short_features:
        source_name = var_to_source[var_name]
        temporal_source_max_lag[source_name] = max(
            short_max_lag,
            temporal_source_max_lag.get(source_name, 0),
        )
    for var_name in long_features:
        source_name = var_to_source[var_name]
        temporal_source_max_lag[source_name] = max(
            long_max_lag,
            temporal_source_max_lag.get(source_name, 0),
        )
    temporal_sources = sorted(temporal_source_max_lag.keys())
    source_chunk_meta: Dict[str, Dict[str, object]] = {}
    for source_name, src_ds in dss.items():
        src_ds = dss[source_name]
        src_da = _dataarray_for_chunking(src_ds)
        src_x_chunk = _get_chunk_size(src_da, "x", fallback=64)
        src_y_chunk = _get_chunk_size(src_da, "y", fallback=64)
        src_x_size = int(src_ds.sizes.get("x", x_size))
        src_y_size = int(src_ds.sizes.get("y", y_size))
        if "time" in src_ds.coords:
            src_t_dim = "time"
            src_t_chunk = _get_chunk_size(src_da, "time", fallback=31)
            src_time_index = pd.DatetimeIndex(pd.to_datetime(src_ds["time"].values)).normalize()
            src_t_size = int(src_time_index.size)
            time_vals_ns = np.asarray(src_time_index.values, dtype="datetime64[ns]")
            time_vals_day = time_vals_ns.astype("datetime64[D]")
            if time_vals_day.size >= 2:
                daily_diffs = np.diff(time_vals_day).astype("timedelta64[D]").astype(np.int64)
                is_daily_contiguous = bool(np.all(daily_diffs == 1))
            else:
                is_daily_contiguous = True
            if is_daily_contiguous and time_vals_day.size > 0:
                time_lookup = {
                    "mode": "daily_contiguous",
                    "time0_day": time_vals_day[0],
                    "size": int(time_vals_day.size),
                }
            else:
                time_lookup = {
                    "mode": "searchsorted",
                    "time_vals_ns": time_vals_ns,
                }
        elif "year" in src_ds.coords:
            src_t_dim = "year"
            src_t_chunk = _get_chunk_size(src_da, "year", fallback=1)
            src_time_index = pd.DatetimeIndex(pd.to_datetime(src_ds["year"].values)).normalize()
            src_t_size = int(src_time_index.size)
            time_lookup = {
                "mode": "searchsorted",
                "time_vals_ns": np.asarray(src_time_index.values, dtype="datetime64[ns]"),
            }
        else:
            src_t_dim = None
            src_t_chunk = 1
            src_time_index = None
            src_t_size = 0
            time_lookup = None
        source_chunk_meta[source_name] = {
            "x_chunk": int(src_x_chunk),
            "y_chunk": int(src_y_chunk),
            "t_chunk": int(src_t_chunk),
            "t_dim": src_t_dim,
            "x_size": int(src_x_size),
            "y_size": int(src_y_size),
            "t_size": int(src_t_size),
            "time_index": src_time_index,
            "time_lookup": time_lookup,
        }

    X_short_np = np.full((n, len(short_vars_out), len(short_lag_days)), np.nan, dtype=np.float32)
    X_long_np = np.full((n, len(long_vars_out), len(long_lag_days)), np.nan, dtype=np.float32)
    X_static_np = np.full((n, len(static_features)), np.nan, dtype=np.float32)
    Y_np = rows["target_value"].to_numpy(dtype=np.float32)
    source_np = rows["source_code"].to_numpy(dtype=np.int64)
    stratifier_np = np.full(n, np.nan, dtype=np.float32)

    lag_frac_short = None
    lag_frac_long = None
    if include_lag_feature and short_lag_days:
        denom = float(max(short_lag_days) if max(short_lag_days) > 0 else 1.0)
        lag_frac_short = np.asarray([d / denom for d in short_lag_days], dtype=np.float32)
    if include_lag_feature and long_lag_days:
        denom = float(max(long_lag_days) if max(long_lag_days) > 0 else 1.0)
        lag_frac_long = np.asarray([d / denom for d in long_lag_days], dtype=np.float32)

    row_indices_all = np.arange(n, dtype=np.int64)
    x_idx_all = rows["x_idx"].to_numpy(dtype=np.int64)
    y_idx_all = rows["y_idx"].to_numpy(dtype=np.int64)
    date_series = pd.to_datetime(rows["date"]).dt.normalize().reset_index(drop=True)

    if include_lag_feature and lag_frac_short is not None:
        X_short_np[:, len(short_features), :] = lag_frac_short[None, :]
    if include_lag_feature and lag_frac_long is not None:
        X_long_np[:, len(long_features), :] = lag_frac_long[None, :]

    overall_start_time = datetime.datetime.now()
    print("[build_direct_tensors] Stage 1/4: building temporal lag graph")
    short_tasks = _build_lag_tasks(
        row_indices=row_indices_all,
        x_idx=x_idx_all,
        y_idx=y_idx_all,
        sample_dates=date_series,
        lag_days=short_lag_days,
        desc="build short lag graph",
    )
    long_tasks = _build_lag_tasks(
        row_indices=row_indices_all,
        x_idx=x_idx_all,
        y_idx=y_idx_all,
        sample_dates=date_series,
        lag_days=long_lag_days,
        desc="build long lag graph",
    )

    print("[build_direct_tensors] Stage 2/4: building per-source temporal chunk plans")
    temporal_plans: Dict[str, Dict[str, object]] = {}
    plan_iter = tqdm(
        temporal_sources,
        total=len(temporal_sources),
        desc="temporal chunk plans",
        unit="source",
        leave=False,
    )
    for source_name in plan_iter:
        long_vars_here = [
            (j, var) for j, var in enumerate(long_features)
            if var_to_source[var] == source_name
        ]
        short_vars_here = [
            (j, var) for j, var in enumerate(short_features)
            if var_to_source[var] == source_name
        ]
        src_meta = source_chunk_meta[source_name]
        src_x_chunk = int(src_meta["x_chunk"])
        src_y_chunk = int(src_meta["y_chunk"])
        src_t_chunk = int(src_meta["t_chunk"])
        src_time_index = src_meta["time_index"]
        src_time_lookup = src_meta.get("time_lookup")
        src_t_dim = src_meta["t_dim"]
        source_idx_payload = source_point_index.get(source_name)
        if source_idx_payload is None:
            raise ValueError(
                f"Source '{source_name}' is missing sample-point indices. "
                "Ensure it has x/y coordinates and CRS metadata."
            )
        source_x_idx_all = source_idx_payload["x_idx"]
        source_y_idx_all = source_idx_payload["y_idx"]
        if not long_vars_here and not short_vars_here:
            temporal_plans[source_name] = {
                "long_vars": long_vars_here,
                "short_vars": short_vars_here,
                "src_meta": src_meta,
                "all_native_keys": [],
                "long_tasks_by_key": {},
                "short_tasks_by_key": {},
            }
            continue
        if src_t_dim is None or src_time_index is None:
            temporal_plans[source_name] = {
                "long_vars": long_vars_here,
                "short_vars": short_vars_here,
                "src_meta": src_meta,
                "all_native_keys": [],
                "long_tasks_by_key": {},
                "short_tasks_by_key": {},
            }
            continue
        long_tasks_for_source = []
        if long_vars_here:
            long_tasks_for_source = [
                {
                    "row_indices": task["row_indices"],
                    "x_idx": source_x_idx_all,
                    "y_idx": source_y_idx_all,
                    "need_dates": task["need_dates"],
                    "lag_idx": task["lag_idx"],
                }
                for task in long_tasks
            ]
        short_tasks_for_source = []
        if short_vars_here:
            short_tasks_for_source = [
                {
                    "row_indices": task["row_indices"],
                    "x_idx": source_x_idx_all,
                    "y_idx": source_y_idx_all,
                    "need_dates": task["need_dates"],
                    "lag_idx": task["lag_idx"],
                }
                for task in short_tasks
            ]
        long_tasks_by_key = _split_tasks_to_native_chunk_keys(
            long_tasks_for_source,
            x_chunk_size=src_x_chunk,
            y_chunk_size=src_y_chunk,
            time_index=src_time_index,
            time_chunk_size=src_t_chunk,
            desc=f"plan {source_name} long",
            time_lookup=src_time_lookup,
        )
        short_tasks_by_key = _split_tasks_to_native_chunk_keys(
            short_tasks_for_source,
            x_chunk_size=src_x_chunk,
            y_chunk_size=src_y_chunk,
            time_index=src_time_index,
            time_chunk_size=src_t_chunk,
            desc=f"plan {source_name} short",
            time_lookup=src_time_lookup,
        )
        all_native_keys = sorted(set(long_tasks_by_key.keys()) | set(short_tasks_by_key.keys()))
        temporal_plans[source_name] = {
            "long_vars": long_vars_here,
            "short_vars": short_vars_here,
            "src_meta": src_meta,
            "all_native_keys": all_native_keys,
            "long_tasks_by_key": long_tasks_by_key,
            "short_tasks_by_key": short_tasks_by_key,
        }

    print("[build_direct_tensors] Stage 3/4: executing temporal native chunks")
    total_temporal_sources = len(temporal_sources)
    for source_i, source_name in enumerate(temporal_sources, start=1):
        now = datetime.datetime.now()
        elapsed_s = (now - overall_start_time).total_seconds()
        if source_i > 1:
            avg_source_s = elapsed_s / float(source_i - 1)
            eta_s = avg_source_s * float(total_temporal_sources - (source_i - 1))
        else:
            eta_s = 0.0
        print(
            f"[build_direct_tensors] processing temporal source {source_i} / {total_temporal_sources}: {source_name} "
            f"(elapsed={_format_seconds_hms(elapsed_s)}, "
            f"eta={_format_seconds_hms(eta_s)})"
        )
        plan = temporal_plans[source_name]
        long_vars_here = plan["long_vars"]
        short_vars_here = plan["short_vars"]
        src_meta = plan["src_meta"]
        all_native_keys = plan["all_native_keys"]
        long_tasks_by_key = plan["long_tasks_by_key"]
        short_tasks_by_key = plan["short_tasks_by_key"]
        if not long_vars_here and not short_vars_here:
            continue

        src_x_chunk = int(src_meta["x_chunk"])
        src_y_chunk = int(src_meta["y_chunk"])
        src_t_chunk = int(src_meta["t_chunk"])
        src_t_size = int(src_meta["t_size"])
        src_t_dim = src_meta["t_dim"]
        if src_t_dim is None:
            continue
        print(f"[build_direct_tensors] {source_name}: needed temporal native chunks={len(all_native_keys):,}")
        print(
            f"[build_direct_tensors] {source_name}: temporal workers={temporal_num_workers}, "
            f"max_inflight={temporal_max_inflight}"
        )
        source_vars_needed = sorted(
            set([v for _, v in long_vars_here] + [v for _, v in short_vars_here])
        )
        loaded_temporal_chunks = 0
        src_x_size = int(src_meta["x_size"])
        src_y_size = int(src_meta["y_size"])

        def _assign_for_loaded_block(
            loaded_ds: xr.Dataset,
            load_t0_block: int,
            load_x0_block: int,
            load_y0_block: int,
            long_tasks_block: List[Dict[str, np.ndarray]],
            short_tasks_block: List[Dict[str, np.ndarray]],
            global_to_local_t: Optional[np.ndarray] = None,
        ) -> None:
            stacked_data = None
            variable_index: Dict[str, int] = {}
            if "data" in loaded_ds.data_vars:
                data_da = loaded_ds["data"]
                if "variable" in data_da.dims and "variable" in data_da.coords:
                    try:
                        stacked_data = np.asarray(
                            data_da.transpose(src_t_dim, "x", "y", "variable").values
                        )
                        variable_index = {
                            str(v): i for i, v in enumerate(data_da["variable"].values.tolist())
                        }
                    except Exception:
                        stacked_data = None
                        variable_index = {}

            per_var_cache: Dict[str, np.ndarray] = {}
            if stacked_data is None:
                for var_name in source_vars_needed:
                    try:
                        da = _extract_temporal_dataarray(loaded_ds, var_name)
                    except Exception:
                        continue
                    if src_t_dim not in da.dims or "x" not in da.dims or "y" not in da.dims:
                        continue
                    try:
                        per_var_cache[var_name] = np.asarray(da.transpose(src_t_dim, "x", "y").values)
                    except Exception:
                        continue

            if long_tasks_block:
                for var_slot, var_name in long_vars_here:
                    if stacked_data is not None:
                        if var_name not in variable_index:
                            continue
                        arr_txy = stacked_data[:, :, :, variable_index[var_name]]
                    else:
                        arr_txy = per_var_cache.get(var_name)
                        if arr_txy is None:
                            continue
                    _assign_loaded_values_from_tasks(
                        target_np=X_long_np,
                        loaded_txy=arr_txy,
                        loaded_t0=load_t0_block,
                        loaded_x0=load_x0_block,
                        loaded_y0=load_y0_block,
                        var_slot=var_slot,
                        tasks=long_tasks_block,
                        global_to_local_t=global_to_local_t,
                    )
            if short_tasks_block:
                for var_slot, var_name in short_vars_here:
                    if stacked_data is not None:
                        if var_name not in variable_index:
                            continue
                        arr_txy = stacked_data[:, :, :, variable_index[var_name]]
                    else:
                        arr_txy = per_var_cache.get(var_name)
                        if arr_txy is None:
                            continue
                    _assign_loaded_values_from_tasks(
                        target_np=X_short_np,
                        loaded_txy=arr_txy,
                        loaded_t0=load_t0_block,
                        loaded_x0=load_x0_block,
                        loaded_y0=load_y0_block,
                        var_slot=var_slot,
                        tasks=short_tasks_block,
                        global_to_local_t=global_to_local_t,
                    )

        def _load_temporal_job(job: Dict[str, object]) -> Tuple[xr.Dataset, pd.Timedelta]:
            before = pd.Timestamp.now()
            selector = dss[source_name].isel(
                x=slice(int(job["load_x0"]), int(job["load_x1"])),
                y=slice(int(job["load_y0"]), int(job["load_y1"])),
            )
            if job["kind"] == "needed":
                loaded_ds = selector.isel(
                    **{src_t_dim: np.asarray(job["needed_t_idx"], dtype=np.int64).tolist()}
                ).compute()
            else:
                loaded_ds = selector.isel(
                    **{src_t_dim: slice(int(job["load_t0"]), int(job["load_t1"]))}
                ).compute()
            after = pd.Timestamp.now()
            if src_t_dim == "time":
                loaded_ds = _normalize_time_to_day(loaded_ds)
            return loaded_ds, (after - before)

        def _apply_temporal_job(loaded_ds: xr.Dataset, job: Dict[str, object]) -> None:
            if job["kind"] == "needed":
                needed_t_idx = np.asarray(job["needed_t_idx"], dtype=np.int64)
                global_to_local_t = np.full(src_t_size, -1, dtype=np.int64)
                global_to_local_t[needed_t_idx] = np.arange(needed_t_idx.size, dtype=np.int64)
                _assign_for_loaded_block(
                    loaded_ds=loaded_ds,
                    load_t0_block=0,
                    load_x0_block=int(job["load_x0"]),
                    load_y0_block=int(job["load_y0"]),
                    long_tasks_block=job["long_tasks"],
                    short_tasks_block=job["short_tasks"],
                    global_to_local_t=global_to_local_t,
                )
            else:
                _assign_for_loaded_block(
                    loaded_ds=loaded_ds,
                    load_t0_block=int(job["load_t0"]),
                    load_x0_block=int(job["load_x0"]),
                    load_y0_block=int(job["load_y0"]),
                    long_tasks_block=job["long_tasks"],
                    short_tasks_block=job["short_tasks"],
                )

        def _execute_temporal_jobs(jobs: List[Dict[str, object]], desc: str) -> int:
            if not jobs:
                return 0
            completed = 0
            if temporal_num_workers <= 1:
                job_iter = tqdm(
                    jobs,
                    desc=desc,
                    unit="temporal_job",
                    leave=False,
                )
                for job in job_iter:
                    loaded_ds, elapsed_td = _load_temporal_job(job)
                    if bool(job.get("log_elapsed", True)):
                        print(f"Elapsed: {elapsed_td}")
                    _apply_temporal_job(loaded_ds, job)
                    completed += 1
                return completed

            max_inflight = max(1, min(temporal_max_inflight, len(jobs)))
            pbar = tqdm(
                total=len(jobs),
                desc=desc,
                unit="temporal_job",
                leave=False,
            )
            with cf.ThreadPoolExecutor(max_workers=temporal_num_workers) as executor:
                pending: Dict[cf.Future, Dict[str, object]] = {}
                jobs_iter = iter(jobs)

                def _submit_next() -> bool:
                    try:
                        next_job = next(jobs_iter)
                    except StopIteration:
                        return False
                    fut = executor.submit(_load_temporal_job, next_job)
                    pending[fut] = next_job
                    return True

                while len(pending) < max_inflight and _submit_next():
                    pass

                while pending:
                    done, _ = cf.wait(list(pending.keys()), return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        job = pending.pop(fut)
                        loaded_ds, elapsed_td = fut.result()
                        if bool(job.get("log_elapsed", True)):
                            print(f"Elapsed: {elapsed_td}")
                        _apply_temporal_job(loaded_ds, job)
                        completed += 1
                        pbar.update(1)
                    while len(pending) < max_inflight and _submit_next():
                        pass
            pbar.close()
            return completed

        if (
            src_t_dim == "time"
            and src_meta.get("time_index") is not None
            and temporal_batch_mode == "all_needed_dates_per_xy_chunk"
        ):
            src_time_index = src_meta["time_index"]
            xy_groups: Dict[Tuple[int, int], Dict[str, List[Dict[str, np.ndarray]]]] = {}
            for t_chunk_id, x_chunk_id, y_chunk_id in all_native_keys:
                group_key = (int(x_chunk_id), int(y_chunk_id))
                if group_key not in xy_groups:
                    xy_groups[group_key] = {"long": [], "short": []}
                xy_groups[group_key]["long"].extend(
                    long_tasks_by_key.get((t_chunk_id, x_chunk_id, y_chunk_id), [])
                )
                xy_groups[group_key]["short"].extend(
                    short_tasks_by_key.get((t_chunk_id, x_chunk_id, y_chunk_id), [])
                )

            xy_keys = sorted(xy_groups.keys())
            print(
                f"[build_direct_tensors] {source_name}: needed full-time spatial chunks={len(xy_keys):,}"
            )
            jobs: List[Dict[str, object]] = []
            for x_chunk_id, y_chunk_id in xy_keys:
                load_x0 = int(x_chunk_id * src_x_chunk)
                load_x1 = int(min((x_chunk_id + 1) * src_x_chunk, src_x_size))
                load_y0 = int(y_chunk_id * src_y_chunk)
                load_y1 = int(min((y_chunk_id + 1) * src_y_chunk, src_y_size))
                if load_x0 >= load_x1 or load_y0 >= load_y1:
                    continue
                group_payload = xy_groups[(x_chunk_id, y_chunk_id)]
                t_arrays = []
                for task in group_payload["long"]:
                    if task["t_idx_global"].size > 0:
                        t_arrays.append(task["t_idx_global"])
                for task in group_payload["short"]:
                    if task["t_idx_global"].size > 0:
                        t_arrays.append(task["t_idx_global"])
                if not t_arrays:
                    continue
                needed_t_idx = np.unique(np.concatenate(t_arrays)).astype(np.int64)
                needed_t_idx = needed_t_idx[(needed_t_idx >= 0) & (needed_t_idx < src_t_size)]
                if needed_t_idx.size == 0:
                    continue
                jobs.append(
                    {
                        "kind": "needed",
                        "load_x0": load_x0,
                        "load_x1": load_x1,
                        "load_y0": load_y0,
                        "load_y1": load_y1,
                        "needed_t_idx": needed_t_idx,
                        "long_tasks": group_payload["long"],
                        "short_tasks": group_payload["short"],
                        "log_elapsed": False,
                    }
                )
            loaded_temporal_chunks += _execute_temporal_jobs(
                jobs,
                desc=f"{source_name} temporal-all-needed-time",
            )
        elif (
            src_t_dim == "time"
            and src_meta.get("time_index") is not None
            and temporal_batch_mode in {"year", "needed_days_in_month_blocks"}
        ):
            src_time_index = src_meta["time_index"]
            use_needed_days_within_blocks = temporal_batch_mode == "needed_days_in_month_blocks"
            month_codes = (
                src_time_index.year.to_numpy(dtype=np.int64) * 12
                + (src_time_index.month.to_numpy(dtype=np.int64) - 1)
            )
            unique_month_codes = np.unique(month_codes)
            month_bounds: Dict[int, Tuple[int, int]] = {}
            for mm in unique_month_codes:
                month_pos = np.where(month_codes == mm)[0]
                if month_pos.size == 0:
                    continue
                month_bounds[int(mm)] = (int(month_pos[0]), int(month_pos[-1] + 1))

            block_bounds: Dict[int, Tuple[int, int]] = {}
            for mm in sorted(int(v) for v in unique_month_codes.tolist()):
                block_start = (int(mm) // temporal_month_block_size) * temporal_month_block_size
                y0, y1 = month_bounds[int(mm)]
                if block_start not in block_bounds:
                    block_bounds[block_start] = (int(y0), int(y1))
                else:
                    prev0, prev1 = block_bounds[block_start]
                    block_bounds[block_start] = (min(prev0, int(y0)), max(prev1, int(y1)))

            block_groups: Dict[Tuple[int, int, int], Dict[str, List[Dict[str, np.ndarray]]]] = {}
            for t_chunk_id, x_chunk_id, y_chunk_id in all_native_keys:
                key = (t_chunk_id, x_chunk_id, y_chunk_id)
                for task_group_name, tasks_by_key in [("long", long_tasks_by_key), ("short", short_tasks_by_key)]:
                    tasks_here = tasks_by_key.get(key, [])
                    for task in tasks_here:
                        t_idx = task["t_idx_global"].astype(np.int64)
                        if t_idx.size == 0:
                            continue
                        valid_t = (t_idx >= 0) & (t_idx < src_t_size)
                        if not np.any(valid_t):
                            continue
                        t_idx_valid = t_idx[valid_t]
                        task_block_starts = (
                            (month_codes[t_idx_valid] // temporal_month_block_size) * temporal_month_block_size
                        ).astype(np.int64)
                        unique_blocks, inverse = np.unique(task_block_starts, return_inverse=True)
                        for block_i, block_start in enumerate(unique_blocks.tolist()):
                            keep = valid_t.copy()
                            keep[np.where(valid_t)[0]] = (inverse == block_i)
                            if not np.any(keep):
                                continue
                            group_key = (int(block_start), int(x_chunk_id), int(y_chunk_id))
                            if group_key not in block_groups:
                                block_groups[group_key] = {"long": [], "short": []}
                            block_groups[group_key][task_group_name].append(
                                {
                                    "row_indices": task["row_indices"][keep],
                                    "x_idx": task["x_idx"][keep],
                                    "y_idx": task["y_idx"][keep],
                                    "t_idx_global": task["t_idx_global"][keep],
                                    "lag_idx": int(task["lag_idx"]),
                                }
                            )

            block_keys = sorted(block_groups.keys())
            print(
                f"[build_direct_tensors] {source_name}: needed {temporal_month_block_size}-month spatial chunks={len(block_keys):,}"
            )
            mode_suffix = (
                f"needed-days-{temporal_month_block_size}m"
                if use_needed_days_within_blocks
                else f"temporal-{temporal_month_block_size}m"
            )
            jobs: List[Dict[str, object]] = []
            for block_start, x_chunk_id, y_chunk_id in block_keys:
                if int(block_start) not in block_bounds:
                    continue
                load_x0 = int(x_chunk_id * src_x_chunk)
                load_x1 = int(min((x_chunk_id + 1) * src_x_chunk, src_x_size))
                load_y0 = int(y_chunk_id * src_y_chunk)
                load_y1 = int(min((y_chunk_id + 1) * src_y_chunk, src_y_size))
                if load_x0 >= load_x1 or load_y0 >= load_y1:
                    continue
                group_payload = block_groups[(block_start, x_chunk_id, y_chunk_id)]
                load_t0, load_t1 = block_bounds[int(block_start)]
                if load_t0 >= load_t1:
                    continue
                if use_needed_days_within_blocks:
                    t_arrays = []
                    for task in group_payload["long"]:
                        if task["t_idx_global"].size > 0:
                            t_arrays.append(task["t_idx_global"])
                    for task in group_payload["short"]:
                        if task["t_idx_global"].size > 0:
                            t_arrays.append(task["t_idx_global"])
                    if not t_arrays:
                        continue
                    needed_t_idx = np.unique(np.concatenate(t_arrays)).astype(np.int64)
                    needed_t_idx = needed_t_idx[
                        (needed_t_idx >= 0)
                        & (needed_t_idx < src_t_size)
                        & (needed_t_idx >= load_t0)
                        & (needed_t_idx < load_t1)
                    ]
                    if needed_t_idx.size == 0:
                        continue
                    jobs.append(
                        {
                            "kind": "needed",
                            "load_x0": load_x0,
                            "load_x1": load_x1,
                            "load_y0": load_y0,
                            "load_y1": load_y1,
                            "needed_t_idx": needed_t_idx,
                            "long_tasks": group_payload["long"],
                            "short_tasks": group_payload["short"],
                            "log_elapsed": True,
                        }
                    )
                else:
                    jobs.append(
                        {
                            "kind": "slice",
                            "load_x0": load_x0,
                            "load_x1": load_x1,
                            "load_y0": load_y0,
                            "load_y1": load_y1,
                            "load_t0": load_t0,
                            "load_t1": load_t1,
                            "long_tasks": group_payload["long"],
                            "short_tasks": group_payload["short"],
                            "log_elapsed": True,
                        }
                    )
            loaded_temporal_chunks += _execute_temporal_jobs(
                jobs,
                desc=f"{source_name} {mode_suffix}",
            )
        else:
            jobs: List[Dict[str, object]] = []
            for t_chunk_id, x_chunk_id, y_chunk_id in all_native_keys:
                load_x0 = int(x_chunk_id * src_x_chunk)
                load_x1 = int(min((x_chunk_id + 1) * src_x_chunk, src_x_size))
                load_y0 = int(y_chunk_id * src_y_chunk)
                load_y1 = int(min((y_chunk_id + 1) * src_y_chunk, src_y_size))
                if load_x0 >= load_x1 or load_y0 >= load_y1:
                    continue
                load_t0 = int(t_chunk_id * src_t_chunk)
                load_t1 = int(min((t_chunk_id + 1) * src_t_chunk, src_t_size))
                if load_t0 >= load_t1:
                    continue
                long_tasks_here = long_tasks_by_key.get((t_chunk_id, x_chunk_id, y_chunk_id), [])
                short_tasks_here = short_tasks_by_key.get((t_chunk_id, x_chunk_id, y_chunk_id), [])
                jobs.append(
                    {
                        "kind": "slice",
                        "load_x0": load_x0,
                        "load_x1": load_x1,
                        "load_y0": load_y0,
                        "load_y1": load_y1,
                        "load_t0": load_t0,
                        "load_t1": load_t1,
                        "long_tasks": long_tasks_here,
                        "short_tasks": short_tasks_here,
                        "log_elapsed": True,
                    }
                )
            loaded_temporal_chunks += _execute_temporal_jobs(
                jobs,
                desc=f"{source_name} temporal",
            )
        print(
            f"[build_direct_tensors] {source_name}: loaded temporal chunks once={loaded_temporal_chunks:,}"
        )

        done_now = datetime.datetime.now()
        elapsed_s = (done_now - overall_start_time).total_seconds()
        avg_source_s = elapsed_s / float(source_i)
        eta_s = avg_source_s * float(total_temporal_sources - source_i)
        print(
            f"[build_direct_tensors] Completed temporal source {source_i} / {total_temporal_sources}: {source_name} "
            f"(elapsed={_format_seconds_hms(elapsed_s)}, "
            f"eta={_format_seconds_hms(eta_s)})"
        )

    print("[build_direct_tensors] Stage 4/4: building and executing static/stratifier chunk plans")
    static_specs_by_source: Dict[str, List[Dict[str, object]]] = {}
    static_feature_iter = tqdm(
        list(enumerate(static_features)),
        total=len(static_features),
        desc="static graph",
        unit="feature",
        leave=False,
    )
    for j, feat in static_feature_iter:
        if feat == "latitude":
            X_static_np[:, j] = rows["latitude"].to_numpy(dtype=np.float32)
            continue
        if feat == "longitude":
            X_static_np[:, j] = rows["longitude"].to_numpy(dtype=np.float32)
            continue
        if feat.startswith("climate_zone_") and "climate_zone" in dss:
            try:
                want_class = int(feat.split("_")[-1])
            except Exception:
                continue
            static_specs_by_source.setdefault("climate_zone", []).append(
                {
                    "target": "static",
                    "slot": int(j),
                    "mode": "direct_xy",
                    "kind": "climate_zone_onehot",
                    "var_name": "climate_zone",
                    "onehot_value": int(want_class),
                }
            )
            continue

        candidate_source = None
        if "landcover_frac" in dss and feat in dss["landcover_frac"].data_vars:
            candidate_source = "landcover_frac"
        elif "static" in dss and feat in dss["static"].data_vars:
            candidate_source = "static"
        elif feat in var_to_source and var_to_source[feat] in dss and feat in dss[var_to_source[feat]].data_vars:
            candidate_source = var_to_source[feat]

        if candidate_source is None:
            print(f"Warning: static feature '{feat}' not found in loaded datasets; leaving as NaN")
            continue
        try:
            da = _extract_temporal_dataarray(dss[candidate_source], feat)
        except Exception:
            print(f"Warning: static feature '{feat}' unavailable in source '{candidate_source}'; leaving as NaN")
            continue
        if "year" in da.dims:
            mode = "indexed_year"
        elif "time" in da.dims:
            mode = "indexed_time"
        else:
            mode = "direct_xy"
        static_specs_by_source.setdefault(candidate_source, []).append(
            {
                "target": "static",
                "slot": int(j),
                "mode": mode,
                "kind": "numeric",
                "var_name": feat,
            }
        )

    if stratifier == "nlcd" and "nlcd" in dss:
        static_specs_by_source.setdefault("nlcd", []).append(
            {
                "target": "stratifier",
                "slot": None,
                "mode": "indexed_year",
                "kind": "numeric",
                "var_name": "nlcd",
            }
        )

    static_sources = sorted(static_specs_by_source.keys())
    for source_name in static_sources:
        src_specs = static_specs_by_source[source_name]
        source_idx_payload = source_point_index.get(source_name)
        if source_idx_payload is None:
            print(
                f"Warning: source '{source_name}' missing sample-point indices; "
                "skipping static/stratifier assignment for this source."
            )
            continue
        source_x_idx_all = source_idx_payload["x_idx"]
        source_y_idx_all = source_idx_payload["y_idx"]
        src_meta = source_chunk_meta[source_name]
        src_x_chunk = int(src_meta["x_chunk"])
        src_y_chunk = int(src_meta["y_chunk"])
        src_t_chunk = int(src_meta["t_chunk"])
        src_t_dim = src_meta["t_dim"]
        src_time_index = src_meta["time_index"]
        src_x_size = int(src_meta["x_size"])
        src_y_size = int(src_meta["y_size"])
        src_t_size = int(src_meta["t_size"])

        direct_specs = [s for s in src_specs if s["mode"] == "direct_xy"]
        indexed_year_specs = [s for s in src_specs if s["mode"] == "indexed_year"]
        indexed_time_specs = [s for s in src_specs if s["mode"] == "indexed_time"]

        if direct_specs:
            direct_tasks_by_key = _split_points_by_xy_chunks(
                row_indices=row_indices_all,
                x_idx=source_x_idx_all,
                y_idx=source_y_idx_all,
                x_chunk_size=src_x_chunk,
                y_chunk_size=src_y_chunk,
            )
            print(f"[build_direct_tensors] {source_name}: needed direct native chunks={len(direct_tasks_by_key):,}")
            direct_iter = tqdm(
                sorted(direct_tasks_by_key.keys()),
                desc=f"{source_name} static-direct",
                unit="native_chunk",
                leave=False,
            )
            for _, x_chunk_id, y_chunk_id in direct_iter:
                task = direct_tasks_by_key[(0, x_chunk_id, y_chunk_id)]
                load_x0 = int(x_chunk_id * src_x_chunk)
                load_x1 = int(min((x_chunk_id + 1) * src_x_chunk, src_x_size))
                load_y0 = int(y_chunk_id * src_y_chunk)
                load_y1 = int(min((y_chunk_id + 1) * src_y_chunk, src_y_size))
                if load_x0 >= load_x1 or load_y0 >= load_y1:
                    continue
                loaded_ds = dss[source_name].isel(
                    x=slice(load_x0, load_x1),
                    y=slice(load_y0, load_y1),
                ).load()
                for spec in direct_specs:
                    target_name = spec["target"]
                    target_np = X_static_np if target_name == "static" else stratifier_np
                    _assign_loaded_xy_feature(
                        target_np=target_np,
                        var_slot=spec["slot"],
                        loaded_ds=loaded_ds,
                        loaded_x0=load_x0,
                        loaded_y0=load_y0,
                        var_name=spec["var_name"],
                        task=task,
                        onehot_value=spec.get("onehot_value"),
                    )

        for indexed_specs, mode_name in [
            (indexed_year_specs, "indexed_year"),
            (indexed_time_specs, "indexed_time"),
        ]:
            if not indexed_specs:
                continue
            if src_t_dim is None or src_time_index is None:
                continue
            if mode_name == "indexed_year":
                need_dates = pd.to_datetime(rows["date"]).dt.year.astype(int).astype(str) + "-01-01"
                need_dates = pd.to_datetime(need_dates)
            else:
                need_dates = date_series
            t_idx_global = src_time_index.get_indexer(pd.to_datetime(need_dates)).astype(np.int64)
            indexed_tasks_by_key = _split_points_by_txy_chunks(
                row_indices=row_indices_all,
                x_idx=source_x_idx_all,
                y_idx=source_y_idx_all,
                t_idx_global=t_idx_global,
                x_chunk_size=src_x_chunk,
                y_chunk_size=src_y_chunk,
                t_chunk_size=src_t_chunk,
            )
            print(
                f"[build_direct_tensors] {source_name}: needed {mode_name} native chunks="
                f"{len(indexed_tasks_by_key):,}"
            )
            indexed_iter = tqdm(
                sorted(indexed_tasks_by_key.keys()),
                desc=f"{source_name} static-{mode_name}",
                unit="native_chunk",
                leave=False,
            )
            for t_chunk_id, x_chunk_id, y_chunk_id in indexed_iter:
                task = indexed_tasks_by_key[(t_chunk_id, x_chunk_id, y_chunk_id)]
                load_x0 = int(x_chunk_id * src_x_chunk)
                load_x1 = int(min((x_chunk_id + 1) * src_x_chunk, src_x_size))
                load_y0 = int(y_chunk_id * src_y_chunk)
                load_y1 = int(min((y_chunk_id + 1) * src_y_chunk, src_y_size))
                load_t0 = int(t_chunk_id * src_t_chunk)
                load_t1 = int(min((t_chunk_id + 1) * src_t_chunk, src_t_size))
                if load_x0 >= load_x1 or load_y0 >= load_y1 or load_t0 >= load_t1:
                    continue
                loaded_ds = dss[source_name].isel(
                    x=slice(load_x0, load_x1),
                    y=slice(load_y0, load_y1),
                ).isel(**{src_t_dim: slice(load_t0, load_t1)}).compute()
                if src_t_dim == "time":
                    loaded_ds = _normalize_time_to_day(loaded_ds)
                for spec in indexed_specs:
                    target_name = spec["target"]
                    target_np = X_static_np if target_name == "static" else stratifier_np
                    _assign_loaded_txy_feature(
                        target_np=target_np,
                        var_slot=spec["slot"],
                        loaded_ds=loaded_ds,
                        t_dim=src_t_dim,
                        loaded_t0=load_t0,
                        loaded_x0=load_x0,
                        loaded_y0=load_y0,
                        var_name=spec["var_name"],
                        task=task,
                    )

    # Reorder dynamic dims to match existing saved tensors: (N, T, V)
    X_short_np = np.transpose(X_short_np, (0, 2, 1))
    X_long_np = np.transpose(X_long_np, (0, 2, 1))
    X_static_np = X_static_np[:, None, :]

    finite_y = np.isfinite(Y_np)
    finite_strat = np.isfinite(stratifier_np)
    finite_short = np.isfinite(X_short_np).all(axis=(1, 2))
    finite_long = np.isfinite(X_long_np).all(axis=(1, 2))
    finite_static = np.isfinite(X_static_np).all(axis=(1, 2))
    print(
        "[build_direct_tensors] Finite component counts: "
        f"Y={int(finite_y.sum()):,}/{n:,}, "
        f"stratifier={int(finite_strat.sum()):,}/{n:,}, "
        f"short={int(finite_short.sum()):,}/{n:,}, "
        f"long={int(finite_long.sum()):,}/{n:,}, "
        f"static={int(finite_static.sum()):,}/{n:,}"
    )

    finite_mask = finite_y & finite_strat & finite_short & finite_long & finite_static
    kept_n = int(finite_mask.sum())
    print(f"[build_direct_tensors] Finite-mask keep: {kept_n:,}/{n:,} rows")

    kept_rows = rows.loc[finite_mask].copy().reset_index(drop=True)
    kept_rows["source_legible"] = kept_rows["source_legible"].astype(str)

    X_short_kept = X_short_np[finite_mask]
    X_long_kept = X_long_np[finite_mask]
    X_static_kept = X_static_np[finite_mask]
    Y_kept_np = Y_np[finite_mask][:, None, None]
    source_kept_np = source_np[finite_mask]
    stratifier_kept = stratifier_np[finite_mask]

    climate_filter_keep_mask = np.ones(kept_rows.shape[0], dtype=bool)
    climate_filter_metadata = None
    (
        kept_rows,
        X_short_kept,
        X_long_kept,
        X_static_kept,
        Y_kept_np,
        source_kept_np,
        stratifier_kept,
        _kept_climate_codes,
        climate_filter_keep_mask,
        climate_filter_metadata,
    ) = _filter_rows_by_min_sites_per_climate_zone(
        rows=kept_rows,
        X_short=X_short_kept,
        X_long=X_long_kept,
        X_static=X_static_kept,
        Y=Y_kept_np,
        source=source_kept_np,
        stratifier=stratifier_kept,
        static_feature_names=static_features,
        min_sites_per_zone=2,
    )
    final_keep_mask = finite_mask.copy()
    if kept_n > 0:
        final_keep_mask[finite_mask] = climate_filter_keep_mask

    X_short_kept, short_vars_kept, short_vars_dropped = _drop_zero_variance_channels(
        X_short_kept, short_vars_out, "short"
    )
    X_long_kept, long_vars_kept, long_vars_dropped = _drop_zero_variance_channels(
        X_long_kept, long_vars_out, "long"
    )
    X_static_kept, static_vars_kept, static_vars_dropped = _drop_zero_variance_channels(
        X_static_kept, static_features, "static"
    )
    print(
        f"[build_direct_tensors] Short vars kept/dropped: "
        f"{len(short_vars_kept)}/{len(short_vars_dropped)}"
    )
    if short_vars_dropped:
        print(f"[build_direct_tensors] Short vars dropped (zero variance): {short_vars_dropped}")
    print(
        f"[build_direct_tensors] Long vars kept/dropped: "
        f"{len(long_vars_kept)}/{len(long_vars_dropped)}"
    )
    if long_vars_dropped:
        print(f"[build_direct_tensors] Long vars dropped (zero variance): {long_vars_dropped}")
    print(
        f"[build_direct_tensors] Static vars kept/dropped: "
        f"{len(static_vars_kept)}/{len(static_vars_dropped)}"
    )
    if static_vars_dropped:
        print(f"[build_direct_tensors] Static vars dropped (zero variance): {static_vars_dropped}")

    X_short = torch.from_numpy(X_short_kept).float()
    X_long = torch.from_numpy(X_long_kept).float()
    X_static = torch.from_numpy(X_static_kept).float()
    Y = torch.from_numpy(Y_kept_np).float()
    source = torch.from_numpy(source_kept_np)
    target_cols_out = list(target_cols) if target_cols else sorted(rows["target_name"].astype(str).unique().tolist())

    var_names = default_var_names(
        static_features=static_vars_kept,
        short_features=short_vars_kept,
        long_features=long_vars_kept,
        info_features=default_info_features(),
        target_cols=target_cols_out,
        include_lag_feature=False,
    )

    return DirectBuildResult(
        X_short=X_short,
        X_long=X_long,
        X_static=X_static,
        Y=Y,
        source=source,
        info=kept_rows,
        stratifier=stratifier_kept,
        var_names=var_names,
        keep_mask=final_keep_mask,
        build_metadata=climate_filter_metadata,
    )


def save_direct_build_result(result: DirectBuildResult, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    print(f"[save_direct_build_result] Writing tensors/dataframe to {save_dir}")
    torch.save(result.X_short, os.path.join(save_dir, "X_short.pt"))
    print("[save_direct_build_result] Wrote X_short.pt")
    torch.save(result.X_long, os.path.join(save_dir, "X_long.pt"))
    print("[save_direct_build_result] Wrote X_long.pt")
    torch.save(result.X_static, os.path.join(save_dir, "X_static.pt"))
    print("[save_direct_build_result] Wrote X_static.pt")
    torch.save(result.Y, os.path.join(save_dir, "Y.pt"))
    print("[save_direct_build_result] Wrote Y.pt")
    torch.save(result.source, os.path.join(save_dir, "source.pt"))
    print("[save_direct_build_result] Wrote source.pt")
    np.save(os.path.join(save_dir, "stratifier.npy"), result.stratifier)
    print("[save_direct_build_result] Wrote stratifier.npy")
    result.info.to_csv(os.path.join(save_dir, "info.csv"), index=False)
    print("[save_direct_build_result] Wrote info.csv")
    try:
        result.info.to_parquet(os.path.join(save_dir, "info.parquet"), index=False)
        print("[save_direct_build_result] Wrote info.parquet")
    except Exception as exc:
        print(f"Warning: failed to write info.parquet ({exc})")
    with open(os.path.join(save_dir, "var_names.json"), "w") as f:
        json.dump(result.var_names, f)
    print("[save_direct_build_result] Wrote var_names.json")
