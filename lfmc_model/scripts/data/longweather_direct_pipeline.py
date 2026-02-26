import datetime
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pyproj import Transformer


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


def _fix_daymet_leap_days(daymet_point_ds: xr.Dataset) -> xr.Dataset:
    if "time" not in daymet_point_ds.coords:
        return daymet_point_ds
    ds = daymet_point_ds
    years = np.unique(ds.time.dt.year.values)
    new_slices = []
    for y in years:
        if not pd.Timestamp(f"{int(y)}-01-01").is_leap_year:
            continue
        dec30 = pd.Timestamp(f"{int(y)}-12-30")
        dec31 = pd.Timestamp(f"{int(y)}-12-31")
        times_here = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
        if dec30 in times_here and dec31 not in times_here:
            s31 = ds.sel(time=dec30).copy(deep=True)
            s31 = s31.assign_coords(time=dec31)
            new_slices.append(s31)
    if new_slices:
        ds = xr.concat([ds] + new_slices, dim="time").sortby("time")
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
    scratch_dir = "/scratch/users/trobinet/long_lfmc/trent_datasets"
    oak_dir = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets"
    return {"scratch_dir": scratch_dir, "oak_dir": oak_dir}


def default_dataset_paths() -> Dict[str, str]:
    paths = default_paths()
    scratch_dir = paths["scratch_dir"]
    oak_dir = paths["oak_dir"]
    return {
        "daymet": os.path.join(oak_dir, "daymet", "daymet_all_vars.zarr"),
        "modis": os.path.join(
            oak_dir,
            "modis",
            "modis_regridded_gapfilled",
            "quality_1",
            "interpolated",
            "modis_all_vars.zarr",
        ),
        "static": os.path.join(oak_dir, "static", "static_features_500m_epsg5070_float32.nc"),
        "climate_zone": os.path.join(oak_dir, "climate_zones", "climate_zone_per_pixel_westUS.nc4"),
        "landcover_frac": os.path.join(oak_dir, "nlcd", "nlcd_target_grid_2003_2023.zarr"),
        "nlcd": os.path.join(oak_dir, "nlcd", "nlcd_2003_2023.zarr"),
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
    return list(range(181))


def default_var_locs() -> Dict[str, List[str]]:
    return {
        "modis": default_short_features(),
        "daymet": default_long_features(),
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
    return {
        "daymet": xr.open_zarr(dataset_paths["daymet"], consolidated=False),
        "modis": xr.open_zarr(dataset_paths["modis"]),
        "static": xr.open_dataset(dataset_paths["static"]),
        "climate_zone": xr.open_dataset(dataset_paths["climate_zone"]),
        "landcover_frac": xr.open_zarr(dataset_paths["landcover_frac"]),
        "nlcd": xr.open_zarr(dataset_paths["nlcd"]),
    }


def build_sample_index_from_label_sources(
    label_sources: Dict[str, str],
    start_date: str,
    end_date: str,
    out_path: str,
    sort_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)
    frames = []
    for source_name, label_path in label_sources.items():
        df = pd.read_csv(label_path)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        if "date" not in df.columns:
            raise ValueError(f"{label_path} is missing required 'date' column")
        df = df.copy()
        df["source"] = source_name
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        frames.append(df)
    if not frames:
        raise ValueError("No label sources provided")
    out = pd.concat(frames, ignore_index=True)
    if sort_by:
        out = out.sort_values(list(sort_by)).reset_index(drop=True)
    _write_table(out, out_path)
    return out


def load_sample_index(path_or_paths: Sequence[str]) -> pd.DataFrame:
    frames = []
    for path in path_or_paths:
        frames.append(_read_table(path))
    if not frames:
        raise ValueError("No sample index paths provided")
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    if "date" not in df.columns:
        raise ValueError("Sample index must contain a 'date' column")
    df = df.copy()
    df["date"] = _to_datetime_utc_naive(df["date"])
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
    if target_name == "VV":
        return 1, "vv"
    if target_name == "vh_backscatter":
        return 2, "vh_backscatter"
    return 99, str(target_name)


def prepare_training_rows(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    acceptable_lfmc_range: Tuple[float, float] = (30.0, 500.0),
    num_rs_samples: int = 100_000_000,
    vh_locations: str = "all",
) -> pd.DataFrame:
    out = _choose_target_per_row(df, target_cols).copy()
    if "latitude" not in out.columns or "longitude" not in out.columns:
        raise ValueError("Sample index must contain 'latitude' and 'longitude' columns")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["date", "latitude", "longitude", "target_value"])
    lfmc_mask = out["target_name"] == "lfmc"
    out = out[
        (~lfmc_mask) |
        (
            (out["target_value"] >= acceptable_lfmc_range[0]) &
            (out["target_value"] <= acceptable_lfmc_range[1])
        )
    ].copy()

    source_codes = []
    source_legible = []
    for tgt in out["target_name"].tolist():
        code, legible = _source_encoding_for_target(tgt)
        source_codes.append(code)
        source_legible.append(legible)
    out["source_code"] = np.asarray(source_codes, dtype=np.int64)
    out["source_legible"] = source_legible

    insitu = out[out["source_code"] == 0].copy()
    kept = [insitu]

    vv = out[out["source_code"] == 1].copy()
    if not vv.empty:
        n = min(int(num_rs_samples), len(vv))
        kept.append(vv.sample(n=n, random_state=42) if n < len(vv) else vv)

    vh = out[out["source_code"] == 2].copy()
    if not vh.empty:
        if vh_locations == "at_sites":
            vh = vh[vh["source"].astype(str) == "vh_at_sites"].copy()
        elif vh_locations == "at_random":
            vh = vh[vh["source"].astype(str) == "vh_at_random"].copy()
        elif vh_locations != "all":
            raise ValueError(f"Unknown vh_locations={vh_locations}")
        if not vh.empty:
            n = min(int(num_rs_samples), len(vh))
            kept.append(vh.sample(n=n, random_state=42) if n < len(vh) else vh)

    out = pd.concat(kept, ignore_index=True)
    out = out.sort_values(["date", "latitude", "longitude"]).reset_index(drop=True)
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


def _extract_chunk_subsets(
    dss: Dict[str, xr.Dataset],
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    short_max_lag: int,
    long_max_lag: int,
) -> Dict[str, xr.Dataset]:
    modis_chunk = (
        dss["modis"]
        .isel(x=slice(x0, x1), y=slice(y0, y1))
        .sel(time=slice(min_date - pd.Timedelta(days=short_max_lag), max_date))
        .compute()
    )
    daymet_chunk = (
        dss["daymet"]
        .isel(x=slice(x0, x1), y=slice(y0, y1))
        .sel(time=slice(min_date - pd.Timedelta(days=long_max_lag), max_date + pd.Timedelta(days=1)))
        .compute()
    )
    daymet_chunk = _normalize_time_to_day(daymet_chunk)

    landcover_chunk = dss["landcover_frac"].isel(x=slice(x0, x1), y=slice(y0, y1)).load()
    nlcd_chunk = dss["nlcd"].isel(x=slice(x0, x1), y=slice(y0, y1)).load()
    static_chunk = dss["static"].isel(x=slice(x0, x1), y=slice(y0, y1)).load()
    climate_chunk = dss["climate_zone"].isel(x=slice(x0, x1), y=slice(y0, y1)).load()

    return {
        "modis": modis_chunk,
        "daymet": daymet_chunk,
        "landcover_frac": landcover_chunk,
        "nlcd": nlcd_chunk,
        "static": static_chunk,
        "climate_zone": climate_chunk,
    }


def _extract_static_feature_value(
    feature_name: str,
    row: pd.Series,
    chunk_dss: Dict[str, xr.Dataset],
    local_x: int,
    local_y: int,
) -> float:
    if feature_name == "latitude":
        return float(row["latitude"])
    if feature_name == "longitude":
        return float(row["longitude"])
    if feature_name.startswith("climate_zone_"):
        want = int(feature_name.split("_")[-1])
        here = _safe_scalar(chunk_dss["climate_zone"]["climate_zone"].isel(x=local_x, y=local_y))
        if np.isnan(here):
            return np.nan
        return 1.0 if int(here) == want else 0.0
    if feature_name in chunk_dss["landcover_frac"].data_vars:
        year_key = pd.Timestamp(int(pd.Timestamp(row["date"]).year), 1, 1)
        try:
            return _safe_scalar(
                chunk_dss["landcover_frac"][feature_name]
                .sel(year=year_key)
                .isel(x=local_x, y=local_y)
            )
        except Exception:
            return np.nan
    if feature_name in chunk_dss["static"].data_vars:
        return _safe_scalar(chunk_dss["static"][feature_name].isel(x=local_x, y=local_y))
    return np.nan


def _extract_stratifier_value(
    stratifier: str,
    row: pd.Series,
    chunk_dss: Dict[str, xr.Dataset],
    local_x: int,
    local_y: int,
) -> float:
    if stratifier == "nlcd":
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


def build_direct_tensors_from_sample_index(
    sample_df: pd.DataFrame,
    dss: Dict[str, xr.Dataset],
    short_features: Sequence[str],
    long_features: Sequence[str],
    static_features: Sequence[str],
    target_cols: Sequence[str],
    short_lag_days: Sequence[int],
    long_lag_days: Sequence[int],
    stratifier: str = "nlcd",
    include_lag_feature: bool = True,
    acceptable_lfmc_range: Tuple[float, float] = (30.0, 500.0),
    num_rs_samples: int = 100_000_000,
    vh_locations: str = "all",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> DirectBuildResult:
    rows = sample_df.copy()
    rows["date"] = _to_datetime_utc_naive(rows["date"])
    if start_date is not None:
        rows = rows[rows["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        rows = rows[rows["date"] <= pd.to_datetime(end_date)]
    rows = prepare_training_rows(
        rows,
        target_cols=target_cols,
        acceptable_lfmc_range=acceptable_lfmc_range,
        num_rs_samples=num_rs_samples,
        vh_locations=vh_locations,
    )
    rows = rows.reset_index(drop=True)

    short_vars_out = list(short_features) + (["lfrac"] if include_lag_feature else [])
    long_vars_out = list(long_features) + (["lfrac"] if include_lag_feature else [])
    static_features = list(static_features)

    n = len(rows)
    if n == 0:
        raise ValueError("No rows remaining after filtering/sampling")

    x_coords = dss["modis"]["x"].values
    y_coords = dss["modis"]["y"].values
    x_chunk = _get_chunk_size(dss["modis"]["data"], "x", fallback=64)
    y_chunk = _get_chunk_size(dss["modis"]["data"], "y", fallback=64)
    trns = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = trns.transform(rows["longitude"].to_numpy(), rows["latitude"].to_numpy())
    rows["x"] = xs
    rows["y"] = ys

    x_idx = np.array([_nearest_index(x_coords, v) for v in rows["x"].to_numpy()], dtype=np.int64)
    y_idx = np.array([_nearest_index(y_coords, v) for v in rows["y"].to_numpy()], dtype=np.int64)
    rows["x_idx"] = x_idx
    rows["y_idx"] = y_idx
    rows["chunk_x"] = rows["x_idx"] // x_chunk
    rows["chunk_y"] = rows["y_idx"] // y_chunk

    x_size = x_coords.size
    y_size = y_coords.size
    short_lag_days = [int(v) for v in short_lag_days]
    long_lag_days = [int(v) for v in long_lag_days]
    short_max_lag = max(short_lag_days) if short_lag_days else 0
    long_max_lag = max(long_lag_days) if long_lag_days else 0

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

    group_cols = ["chunk_x", "chunk_y"]
    grouped = rows.groupby(group_cols, sort=False)
    for (_, _), grp in grouped:
        idxs = grp.index.to_numpy()
        gx = int(grp["chunk_x"].iloc[0])
        gy = int(grp["chunk_y"].iloc[0])
        x0 = gx * x_chunk
        x1 = min((gx + 1) * x_chunk, x_size)
        y0 = gy * y_chunk
        y1 = min((gy + 1) * y_chunk, y_size)
        min_date = pd.to_datetime(grp["date"]).min().normalize()
        max_date = pd.to_datetime(grp["date"]).max().normalize()
        chunk_dss = _extract_chunk_subsets(
            dss=dss,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            min_date=min_date,
            max_date=max_date,
            short_max_lag=short_max_lag,
            long_max_lag=long_max_lag,
        )

        modis_var_series = {}
        for var in short_features:
            modis_var_series[var] = chunk_dss["modis"]["data"].sel(variable=var)
        daymet_var_series = {}
        for var in long_features:
            daymet_var_series[var] = chunk_dss["daymet"]["data"].sel(variable=var)

        for idx in idxs:
            row = rows.loc[idx]
            local_x = int(row["x_idx"] - x0)
            local_y = int(row["y_idx"] - y0)
            date = pd.Timestamp(row["date"]).normalize()

            point_daymet = chunk_dss["daymet"].isel(x=local_x, y=local_y)
            point_daymet = _fix_daymet_leap_days(point_daymet)
            for j, var in enumerate(long_features):
                point_series = point_daymet["data"].sel(variable=var)
                X_long_np[idx, j, :] = _extract_series_for_lags(point_series, date, long_lag_days)

            point_modis = chunk_dss["modis"].isel(x=local_x, y=local_y)
            point_modis = _normalize_time_to_day(point_modis)
            for j, var in enumerate(short_features):
                point_series = point_modis["data"].sel(variable=var)
                X_short_np[idx, j, :] = _extract_series_for_lags(point_series, date, short_lag_days)

            if include_lag_feature:
                X_short_np[idx, len(short_features), :] = lag_frac_short
                X_long_np[idx, len(long_features), :] = lag_frac_long

            for j, feat in enumerate(static_features):
                X_static_np[idx, j] = _extract_static_feature_value(
                    feat, row, chunk_dss, local_x, local_y
                )

            stratifier_np[idx] = _extract_stratifier_value(
                stratifier, row, chunk_dss, local_x, local_y
            )

    # Reorder dynamic dims to match existing saved tensors: (N, T, V)
    X_short_np = np.transpose(X_short_np, (0, 2, 1))
    X_long_np = np.transpose(X_long_np, (0, 2, 1))
    X_static_np = X_static_np[:, None, :]

    finite_mask = np.isfinite(Y_np)
    finite_mask &= np.isfinite(stratifier_np)
    finite_mask &= np.isfinite(X_short_np).all(axis=(1, 2))
    finite_mask &= np.isfinite(X_long_np).all(axis=(1, 2))
    finite_mask &= np.isfinite(X_static_np).all(axis=(1, 2))

    kept_rows = rows.loc[finite_mask].copy().reset_index(drop=True)
    kept_rows["source_legible"] = kept_rows["source_legible"].astype(str)

    X_short = torch.from_numpy(X_short_np[finite_mask]).float()
    X_long = torch.from_numpy(X_long_np[finite_mask]).float()
    X_static = torch.from_numpy(X_static_np[finite_mask]).float()
    Y = torch.from_numpy(Y_np[finite_mask][:, None, None]).float()
    source = torch.from_numpy(source_np[finite_mask])
    stratifier_kept = stratifier_np[finite_mask]

    var_names = default_var_names(
        static_features=static_features,
        short_features=short_features,
        long_features=long_features,
        info_features=default_info_features(),
        target_cols=target_cols,
        include_lag_feature=include_lag_feature,
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
        keep_mask=finite_mask,
    )


def save_direct_build_result(result: DirectBuildResult, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(result.X_short, os.path.join(save_dir, "X_short.pt"))
    torch.save(result.X_long, os.path.join(save_dir, "X_long.pt"))
    torch.save(result.X_static, os.path.join(save_dir, "X_static.pt"))
    torch.save(result.Y, os.path.join(save_dir, "Y.pt"))
    torch.save(result.source, os.path.join(save_dir, "source.pt"))
    np.save(os.path.join(save_dir, "stratifier.npy"), result.stratifier)
    result.info.to_csv(os.path.join(save_dir, "info.csv"), index=False)
    try:
        result.info.to_parquet(os.path.join(save_dir, "info.parquet"), index=False)
    except Exception as exc:
        print(f"Warning: failed to write info.parquet ({exc})")
    with open(os.path.join(save_dir, "var_names.json"), "w") as f:
        json.dump(result.var_names, f)
