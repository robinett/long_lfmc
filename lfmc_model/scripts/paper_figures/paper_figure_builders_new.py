#!/usr/bin/env python3

import contextlib
import gc
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr
from shapely.geometry import Point

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
eval_dir = os.path.join(project_root, "lfmc_model", "scripts", "eval")
if eval_dir not in sys.path:
    sys.path.append(eval_dir)

from compare_models_at_sites import get_site_error  # noqa: E402
from compare_timeseries import (  # noqa: E402
    _to_naive_datetime,
    aggregate_site_errors,
    build_site_df,
    get_vv_vh_site_series,
    get_model_inference_series,
    get_site_landcover_annotation,
    get_site_state_annotation,
    select_ensemble_member_dirs,
)
from eval_deep import (  # noqa: E402
    _WGS84_TO_5070,
    _build_ensemble_eval_df,
    _build_row_keys,
    _extract_target_frame,
    build_lfmc_space_time_tables,
    build_lfmc_y2y_df,
    build_site_month_anomaly_eval_df,
    build_site_landcover_lookup,
    compute_landcover_decomposition_metrics,
    compute_basic_metrics,
    get_nlcd_frac_ds,
    load_fold_predictions,
)
from paper_figure_plotting import (  # noqa: E402
    plot_lfmc_snapshot_quadrants,
    plot_landcover_comparison_panels,
    plot_landcover_metric_grouped,
    plot_placeholder_figure,
    plot_scatter_triptych,
    plot_site_r2_landcover_distribution,
    plot_stacked_timeseries_panels,
    plot_training_location_maps,
    plot_training_sample_landcover_comparison,
)


CANONICAL_YEAR_START = 2000
STATE_NAME_LOOKUP = {
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "ID": "Idaho",
    "MT": "Montana",
    "NM": "New Mexico",
    "NV": "Nevada",
    "OR": "Oregon",
    "UT": "Utah",
    "WA": "Washington",
    "WY": "Wyoming",
}
_GMBA_BASIC_GDF = None
_GMBA_SITE_LABEL_CACHE = {}
_GMBA_SITE_WITHIN_CACHE = {}


def _figure_output_path(runtime: Dict[str, object], filename: str) -> str:
    return os.path.join(runtime["figures_dir"], filename)


def _table_output_path(runtime: Dict[str, object], stem: str) -> str:
    return os.path.join(runtime["tables_dir"], f"{stem}.csv")


def _model_cfg(cfg: Dict[str, object], model_key: str) -> Dict[str, object]:
    return cfg["models"][model_key]


def _metric_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.std(arr, ddof=0))


def _sample_index_point_key(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    lat_text = latitude.map(lambda value: f"{float(value):.6f}")
    lon_text = longitude.map(lambda value: f"{float(value):.6f}")
    return lat_text + "_" + lon_text


def _build_point_landcover_lookup_fast(point_df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    required_cols = ["point_key", "latitude", "longitude", "year"]
    missing_cols = [col for col in required_cols if col not in point_df.columns]
    if len(missing_cols) > 0:
        raise KeyError(
            f"Point dataframe is missing required columns for land-cover lookup: {missing_cols}"
        )
    requested_points = (
        point_df[required_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    requested_points["point_key"] = requested_points["point_key"].astype(str)
    requested_points["year"] = pd.to_numeric(requested_points["year"], errors="coerce")
    requested_points["latitude"] = pd.to_numeric(requested_points["latitude"], errors="coerce")
    requested_points["longitude"] = pd.to_numeric(requested_points["longitude"], errors="coerce")
    requested_points = requested_points.dropna(subset=["point_key", "year", "latitude", "longitude"]).copy()
    requested_points["year"] = requested_points["year"].astype(int)
    required_lookup_cols = [
        "point_key",
        "year",
        "latitude",
        "longitude",
        "dominant_landcover",
        "dominant_landcover_frac",
    ]
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, dtype={"point_key": str})
        if all(col in existing_df.columns for col in required_lookup_cols):
            existing_df["point_key"] = existing_df["point_key"].astype(str)
            existing_df["year"] = pd.to_numeric(existing_df["year"], errors="coerce")
            existing_df = existing_df[existing_df["year"].notna()].copy()
            existing_df["year"] = existing_df["year"].astype(int)
            existing_df = (
                existing_df[required_lookup_cols]
                .drop_duplicates(subset=["point_key", "year"])
                .reset_index(drop=True)
            )
        else:
            existing_df = pd.DataFrame(columns=required_lookup_cols)
    else:
        existing_df = pd.DataFrame(columns=required_lookup_cols)
    if len(requested_points) == 0:
        return existing_df
    cached_keys = set(
        zip(
            existing_df["point_key"].tolist(),
            existing_df["year"].tolist(),
        )
    )
    requested_keys = list(
        zip(
            requested_points["point_key"].tolist(),
            requested_points["year"].tolist(),
        )
    )
    missing_mask = np.asarray([key not in cached_keys for key in requested_keys], dtype=bool)
    if not missing_mask.any():
        print(f"Using cached point land-cover lookup: {save_path}")
        return existing_df
    missing_points = requested_points.loc[missing_mask].reset_index(drop=True)
    print(
        f"Computing vectorized land-cover lookup for {len(missing_points):,} missing point-years: "
        f"{os.path.basename(save_path)}"
    )
    x_vals, y_vals = _WGS84_TO_5070.transform(
        missing_points["longitude"].to_numpy(),
        missing_points["latitude"].to_numpy(),
    )
    nlcd_ds = get_nlcd_frac_ds()
    landcover_vars = list(nlcd_ds.data_vars)
    point_cube = nlcd_ds.sel(
        x=xr.DataArray(x_vals, dims="point"),
        y=xr.DataArray(y_vals, dims="point"),
        method="nearest",
    )
    point_years = pd.to_datetime(missing_points["year"].astype(str) + "-01-01")
    point_cube = point_cube.sel(
        year=xr.DataArray(point_years.to_numpy(), dims="point"),
        method="nearest",
    ).load()
    frac_arr = point_cube.to_array("landcover").transpose("point", "landcover").values
    valid_mask = np.isfinite(frac_arr).any(axis=1)
    dominant_idx = np.full(len(missing_points), -1, dtype=int)
    if bool(valid_mask.any()):
        dominant_idx[valid_mask] = np.nanargmax(frac_arr[valid_mask], axis=1)
    dominant_landcover = np.full(len(missing_points), "unknown", dtype=object)
    dominant_frac = np.full(len(missing_points), np.nan, dtype=float)
    if bool(valid_mask.any()):
        dominant_landcover[valid_mask] = np.asarray(landcover_vars, dtype=object)[dominant_idx[valid_mask]]
        dominant_frac[valid_mask] = frac_arr[valid_mask, dominant_idx[valid_mask]]
    lookup_df = missing_points.copy()
    lookup_df["dominant_landcover"] = dominant_landcover
    lookup_df["dominant_landcover_frac"] = dominant_frac
    lookup_df = lookup_df[required_lookup_cols]
    lookup_df = pd.concat([existing_df, lookup_df], ignore_index=True)
    lookup_df = lookup_df.drop_duplicates(subset=["point_key", "year"], keep="last").reset_index(drop=True)
    lookup_df.to_csv(save_path, index=False)
    print(f"Wrote point land-cover lookup: {save_path}")
    return lookup_df


def _filtered_landcover_df(df: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    order = list(cfg["filters"]["landcover_order"])
    work = df.copy()
    work = work[work["dominant_landcover"].isin(order)].copy()
    work["dominant_landcover"] = pd.Categorical(
        work["dominant_landcover"],
        categories=order,
        ordered=True,
    )
    return work.sort_values("dominant_landcover").reset_index(drop=True)


def _parse_site_lat_lon(site_key: str) -> Tuple[float, float]:
    lat_str, lon_str = str(site_key).split("_", 1)
    return float(lat_str), float(lon_str)


def _state_display_name(state_value: Optional[str]) -> Optional[str]:
    if state_value is None:
        return None
    state_str = str(state_value).strip()
    if state_str == "":
        return None
    return STATE_NAME_LOOKUP.get(state_str, state_str)


def _north_south_state_label(site_key: str, state_value: Optional[str]) -> Optional[str]:
    state_name = _state_display_name(state_value)
    if state_name is None:
        return None
    lat, _ = _parse_site_lat_lon(site_key)
    state_code = str(state_value).strip()
    split_lookup = {
        "AZ": 34.0,
        "CA": 37.5,
        "CO": 39.0,
        "ID": 45.0,
        "MT": 46.8,
        "NM": 34.5,
        "NV": 38.6,
        "OR": 44.0,
        "UT": 39.3,
        "WA": 47.2,
        "WY": 43.2,
    }
    split_lat = split_lookup.get(state_code)
    if split_lat is None:
        return state_name
    prefix = "Northern" if lat >= split_lat else "Southern"
    return f"{prefix} {state_name}"


def _gmba_display_name(row: pd.Series, state_value: Optional[str]) -> Optional[str]:
    for column in ["Name_EN", "AsciiName", "MapName"]:
        if column not in row.index:
            continue
        value = row[column]
        if pd.isna(value) or str(value).strip() == "":
            continue
        out = str(value).strip()
        state_name = _state_display_name(state_value)
        if state_name is not None and state_name not in out:
            out = f"{out}, {state_name}"
        return out
    return None


def _load_gmba_basic_gdf(shapefile_path: str):
    global _GMBA_BASIC_GDF
    if _GMBA_BASIC_GDF is not None:
        return _GMBA_BASIC_GDF
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Missing GMBA Basic shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        raise ValueError(f"GMBA Basic shapefile is missing CRS: {shapefile_path}")
    _GMBA_BASIC_GDF = gdf.to_crs("EPSG:4326")
    return _GMBA_BASIC_GDF


def _gmba_region_label(site_key: str, state_value: Optional[str], shapefile_path: str) -> Optional[str]:
    cache_key = (site_key, str(state_value), shapefile_path)
    if cache_key in _GMBA_SITE_LABEL_CACHE:
        return _GMBA_SITE_LABEL_CACHE[cache_key]
    gdf = _load_gmba_basic_gdf(shapefile_path)
    lat, lon = _parse_site_lat_lon(site_key)
    point = Point(lon, lat)
    matches = gdf[gdf.geometry.contains(point)]
    if len(matches) == 0:
        matches = gdf[gdf.geometry.touches(point)]
    label = None
    if len(matches) > 0:
        matches = matches.copy()
        if "Area" in matches.columns:
            matches["Area"] = pd.to_numeric(matches["Area"], errors="coerce")
            matches = matches.sort_values("Area", ascending=True)
        label = _gmba_display_name(matches.iloc[0], state_value)
    if label is None:
        label = _north_south_state_label(site_key, state_value)
    _GMBA_SITE_LABEL_CACHE[cache_key] = label
    return label


def _site_within_gmba_polygon(site_key: str, shapefile_path: str) -> bool:
    cache_key = (site_key, shapefile_path)
    if cache_key in _GMBA_SITE_WITHIN_CACHE:
        return _GMBA_SITE_WITHIN_CACHE[cache_key]
    gdf = _load_gmba_basic_gdf(shapefile_path)
    lat, lon = _parse_site_lat_lon(site_key)
    point = Point(lon, lat)
    matches = gdf[gdf.geometry.contains(point)]
    if len(matches) == 0:
        matches = gdf[gdf.geometry.touches(point)]
    out = len(matches) > 0
    _GMBA_SITE_WITHIN_CACHE[cache_key] = out
    return out


def _site_panel_title(cfg: Dict[str, object], site_key: str) -> str:
    state_text = get_site_state_annotation(site_key)
    landcover_text = get_site_landcover_annotation(site_key)
    vegetation = (
        str(landcover_text).replace("Land cover: ", "")
        if landcover_text is not None else "Unknown vegetation"
    )
    region_label = _gmba_region_label(
        site_key,
        state_text,
        cfg["paths"]["gmba_basic_shapefile"],
    )
    if region_label is None:
        region_label = _state_display_name(state_text) or "Unknown location"
    return f"{vegetation} | {region_label}"


def _timeseries_panel_title(
    cfg: Dict[str, object],
    site_key: str,
    criterion_label: str,
    selected_years: Sequence[int],
    site_r2: float,
    extra_r2_parts: Optional[Sequence[str]] = None,
) -> str:
    criterion_key = str(criterion_label).strip().lower()
    percentile_lookup = cfg.get("timeseries_selection", {}).get("r2_percentiles", {})
    pct_value = percentile_lookup.get(criterion_key, np.nan)
    if np.isfinite(pd.to_numeric(pct_value, errors="coerce")):
        criterion_text = f"{int(round(float(pct_value)))}th percentile Site by R²"
    else:
        criterion_text = "Selected Site by R²"
    line1_parts = [_site_panel_title(cfg, site_key)]
    if selected_years:
        line1_parts.append(f"Years shown: {selected_years[0]} - {selected_years[-1]}")
    line2_parts = [criterion_text]
    line2_parts.append(f"Site R²: {site_r2:.2f}" if np.isfinite(site_r2) else "Site R²: nan")
    if extra_r2_parts is not None:
        line2_parts.extend([str(part) for part in extra_r2_parts if str(part).strip() != ""])
    return " | ".join(line1_parts) + "\n" + " | ".join(line2_parts)


def _date_matched_r2(obs_dates, obs_values, pred_dates, pred_values) -> float:
    obs_dt = pd.to_datetime(obs_dates, errors="coerce")
    pred_dt = pd.to_datetime(pred_dates, errors="coerce")
    obs_arr = np.asarray(obs_values, dtype=float)
    pred_arr = np.asarray(pred_values, dtype=float)
    if len(obs_dt) == 0 or len(pred_dt) == 0 or len(obs_arr) != len(obs_dt) or len(pred_arr) != len(pred_dt):
        return np.nan
    obs_df = pd.DataFrame({"date": obs_dt.normalize(), "obs": obs_arr})
    pred_df = pd.DataFrame({"date": pred_dt.normalize(), "pred": pred_arr})
    obs_df = obs_df[np.isfinite(obs_df["obs"])].dropna(subset=["date"]).copy()
    pred_df = pred_df[np.isfinite(pred_df["pred"])].dropna(subset=["date"]).copy()
    if len(obs_df) == 0 or len(pred_df) == 0:
        return np.nan
    pred_df = pred_df.groupby("date", as_index=False)["pred"].mean()
    matched = obs_df.merge(pred_df, on="date", how="inner")
    if len(matched) < 2:
        return np.nan
    metrics = compute_basic_metrics(matched["obs"].values, matched["pred"].values)
    return float(metrics.get("r2", np.nan))


def _top_consecutive_observation_years(dates, n_years: int) -> List[int]:
    dt = pd.to_datetime(dates, errors="coerce")
    dt = dt[dt.notna()]
    if len(dt) == 0:
        return []
    n_years = int(max(n_years, 1))
    year_counts = pd.Series(dt.year).value_counts().to_dict()
    min_year = int(dt.year.min())
    max_year = int(dt.year.max())
    span_years = max_year - min_year + 1
    if span_years < n_years:
        return list(range(min_year, min_year + n_years))
    best_start = min_year
    best_score = -1
    for start_year in range(min_year, max_year - n_years + 2):
        years_here = list(range(start_year, start_year + n_years))
        score = int(sum(year_counts.get(year, 0) for year in years_here))
        if score > best_score:
            best_score = score
            best_start = start_year
    return list(range(best_start, best_start + n_years))


def _count_observations_in_years(dates, selected_years: Sequence[int]) -> int:
    dt = pd.to_datetime(dates, errors="coerce")
    dt = dt[dt.notna()]
    if len(dt) == 0 or len(selected_years) == 0:
        return 0
    year_mask = dt.year.isin([int(year) for year in selected_years])
    return int(np.sum(year_mask))


def _sar_observation_count_for_years(
    vv_vh_obs: Optional[Dict[str, object]],
    selected_years: Sequence[int],
) -> int:
    if vv_vh_obs is None or len(selected_years) == 0:
        return 0
    return (
        _count_observations_in_years(vv_vh_obs.get("vv_dates", []), selected_years)
        + _count_observations_in_years(vv_vh_obs.get("vh_dates", []), selected_years)
    )


def _select_consecutive_timeseries_years(
    lfmc_dates,
    n_years: int,
    vv_vh_obs: Optional[Dict[str, object]] = None,
    prefer_sar_observation_density: bool = False,
) -> Dict[str, object]:
    dt = pd.to_datetime(lfmc_dates, errors="coerce")
    dt = dt[dt.notna()]
    if len(dt) == 0:
        return {
            "selected_years": [],
            "lfmc_obs_count": 0,
            "lfmc_years_present": 0,
            "sar_obs_count": 0,
        }
    n_years = int(max(n_years, 1))
    year_counts = pd.Series(dt.year).value_counts().to_dict()
    min_year = int(dt.year.min())
    max_year = int(dt.year.max())
    span_years = max_year - min_year + 1
    if span_years < n_years:
        start_years = [min_year]
    else:
        start_years = list(range(min_year, max_year - n_years + 2))

    best_payload = None
    best_score = None
    for start_year in start_years:
        selected_years = list(range(int(start_year), int(start_year) + int(n_years)))
        lfmc_obs_count = int(sum(year_counts.get(int(year), 0) for year in selected_years))
        lfmc_years_present = int(sum(1 for year in selected_years if year_counts.get(int(year), 0) > 0))
        sar_obs_count = _sar_observation_count_for_years(vv_vh_obs, selected_years)
        if prefer_sar_observation_density:
            score = (sar_obs_count, lfmc_obs_count, lfmc_years_present, -int(start_year))
        else:
            score = (lfmc_obs_count, sar_obs_count, lfmc_years_present, -int(start_year))
        if best_score is None or score > best_score:
            best_score = score
            best_payload = {
                "selected_years": selected_years,
                "lfmc_obs_count": lfmc_obs_count,
                "lfmc_years_present": lfmc_years_present,
                "sar_obs_count": sar_obs_count,
            }
    return best_payload


def _filter_series_to_years(dates, values, selected_years: Sequence[int]):
    if dates is None or values is None or len(dates) == 0:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=float)
    dt = pd.to_datetime(dates, errors="coerce")
    mask = dt.year.isin(list(selected_years))
    return dt[mask].to_numpy(dtype="datetime64[ns]"), np.asarray(values, dtype=float)[mask]


def _canonicalize_dates_to_year_slots(dates, selected_years: Sequence[int]):
    if dates is None or len(dates) == 0:
        return np.array([], dtype="datetime64[ns]")
    slot_lookup = {
        int(year): CANONICAL_YEAR_START + idx
        for idx, year in enumerate(selected_years)
    }
    out = []
    for ts in pd.to_datetime(dates, errors="coerce"):
        if pd.isna(ts):
            continue
        target_year = slot_lookup.get(int(ts.year))
        if target_year is None:
            continue
        month = int(ts.month)
        max_day = int(pd.Period(year=target_year, month=month, freq="M").days_in_month)
        day = min(int(ts.day), max_day)
        out.append(pd.Timestamp(year=target_year, month=month, day=day))
    return np.asarray(out, dtype="datetime64[ns]")


def _reindex_series_to_daily(dates, values, lower=None, upper=None):
    if dates is None or len(dates) == 0:
        empty = np.array([], dtype="datetime64[ns]")
        return empty, np.array([], dtype=float), lower, upper
    dt = pd.to_datetime(dates, errors="coerce")
    valid_mask = dt.notna()
    dt = dt[valid_mask]
    work_dict = {
        "date": dt,
        "value": np.asarray(values, dtype=float)[valid_mask],
    }
    if lower is not None:
        work_dict["lower"] = np.asarray(lower, dtype=float)[valid_mask]
    if upper is not None:
        work_dict["upper"] = np.asarray(upper, dtype=float)[valid_mask]
    work = (
        pd.DataFrame(work_dict)
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
        .sort_index()
    )
    full_index = pd.date_range(work.index.min(), work.index.max(), freq="D")
    work = work.reindex(full_index)
    out_dates = work.index.to_numpy(dtype="datetime64[ns]")
    out_vals = work["value"].to_numpy(dtype=float)
    out_lower = work["lower"].to_numpy(dtype=float) if "lower" in work.columns else None
    out_upper = work["upper"].to_numpy(dtype=float) if "upper" in work.columns else None
    return out_dates, out_vals, out_lower, out_upper


def _pick_percentile_sites(
    ranked: pd.DataFrame,
    metric_col: str,
    target_percentile: float,
    n_sites: int,
    used_sites: set,
) -> List[str]:
    if len(ranked) == 0:
        return []
    target_value = float(np.percentile(ranked[metric_col].to_numpy(dtype=float), target_percentile))
    return _pick_sites_closest_to_target_value(
        ranked=ranked,
        metric_col=metric_col,
        target_value=target_value,
        n_sites=n_sites,
        used_sites=used_sites,
    )


def _pick_sites_closest_to_target_value(
    ranked: pd.DataFrame,
    metric_col: str,
    target_value: float,
    n_sites: int,
    used_sites: set,
) -> List[str]:
    if len(ranked) == 0 or not np.isfinite(target_value):
        return []
    work = ranked.copy()
    work["percentile_dist"] = np.abs(work[metric_col] - target_value)
    sort_cols = ["percentile_dist"]
    ascending = [True]
    if "sar_window_obs_count" in work.columns:
        sort_cols.append("sar_window_obs_count")
        ascending.append(False)
    sort_cols.append(metric_col)
    ascending.append(False)
    work = work.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    picked = []
    for site in work["site"].tolist():
        if site in used_sites:
            continue
        used_sites.add(site)
        picked.append(site)
        if len(picked) >= n_sites:
            break
    return picked


def _select_model_member_dirs(model_cfg: Dict[str, object]) -> List[str]:
    member_dirs = select_ensemble_member_dirs(
        str(model_cfg["outputs_root"]),
        member_name_prefix=model_cfg.get("ensemble_member_name_prefix"),
        selection_key=model_cfg.get("ensemble_selection_key"),
        member_name_allowlist=model_cfg.get("ensemble_member_name_allowlist"),
        member_name_suffix_allowlist=model_cfg.get("ensemble_member_name_suffix_allowlist"),
        member_training_id_allowlist=model_cfg.get("ensemble_member_training_id_allowlist"),
    )
    subset_size = model_cfg.get("ensemble_random_subset_size")
    if subset_size in {None, "", "None"}:
        return member_dirs
    subset_size = int(subset_size)
    subset_seed = int(model_cfg.get("ensemble_random_subset_seed", 0))
    rng = np.random.default_rng(subset_seed)
    selected_idx = np.sort(rng.choice(len(member_dirs), size=subset_size, replace=False))
    return [member_dirs[int(idx)] for idx in selected_idx]


def _model_cache_token(runtime: Dict[str, object], model_key: str) -> str:
    cache = runtime.setdefault("model_cache_tokens", {})
    if model_key in cache:
        return cache[model_key]
    model_cfg = _model_cfg(runtime["cfg"], model_key)
    raw = "|".join(
        [
            str(model_key),
            str(model_cfg.get("outputs_root", "")),
            str(model_cfg.get("input_data_name", "")),
            str(model_cfg.get("model_num_tasks", "")),
            str(model_cfg.get("ensemble_member_name_prefix", "")),
            str(model_cfg.get("ensemble_selection_key", "")),
            str(model_cfg.get("ensemble_member_name_allowlist", "")),
            str(model_cfg.get("ensemble_member_name_suffix_allowlist", "")),
            str(model_cfg.get("ensemble_member_training_id_allowlist", "")),
            str(model_cfg.get("ensemble_random_subset_size", "")),
            str(model_cfg.get("ensemble_random_subset_seed", "")),
        ]
    )
    token = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    cache[model_key] = token
    return token


def _load_fold_predictions_for_targets(model_dir: str, target_names: Sequence[str]) -> pd.DataFrame:
    requested_targets = {str(target_name).strip().lower() for target_name in target_names}
    fold_info_path = os.path.join(model_dir, "fold_info.json")
    with open(fold_info_path, "r") as file_obj:
        fold_info = json.load(file_obj)
    fold_frames = []
    for fold in fold_info.keys():
        print(f"Loading test outputs for fold {fold}")
        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        test_info_path = os.path.join(fold_dir, "test_info.csv")
        test_outputs_path = os.path.join(fold_dir, "test_outputs.pth")
        test_info = pd.read_csv(test_info_path, low_memory=False)
        test_outputs = torch.load(test_outputs_path, map_location="cpu", weights_only=False)
        source = test_info["source"].astype(str)
        frame_builders = [
            (
                "lfmc",
                source == "nfmd",
                test_outputs.get("lfmc_preds", []),
                test_outputs.get("lfmc_true", []),
            ),
            (
                "vv",
                source.str.startswith("vv"),
                test_outputs.get("vv_preds", []),
                test_outputs.get("vv_true", []),
            ),
            (
                "vh",
                source.str.startswith("vh"),
                test_outputs.get("vh_preds", []),
                test_outputs.get("vh_true", []),
            ),
        ]
        for target_name, mask, preds, true_vals in frame_builders:
            if target_name not in requested_targets:
                continue
            target_frame = _extract_target_frame(
                test_info=test_info,
                preds=preds,
                true_vals=true_vals,
                mask=mask,
                target_name=target_name,
                fold=fold,
            )
            if len(target_frame) > 0:
                fold_frames.append(target_frame)
                print(
                    f"  Added {len(target_frame)} rows for target {target_name} "
                    f"from fold {fold}"
                )
        del test_outputs
        gc.collect()
    if len(fold_frames) == 0:
        raise ValueError(
            f"No evaluation rows found for targets {sorted(requested_targets)} in model dir: {model_dir}"
        )
    eval_df = pd.concat(fold_frames, ignore_index=True)
    eval_df["date"] = pd.to_datetime(eval_df["date"], errors="coerce")
    return eval_df


def _write_compact_target_prediction_cache(
    model_dir: str,
    target_names: Sequence[str],
    legacy_cache_path: str,
    row_key_path: str,
    pred_path: str,
    template_path: Optional[str],
) -> None:
    try:
        if os.path.exists(legacy_cache_path):
            df = pd.read_pickle(legacy_cache_path)
        else:
            df = _load_fold_predictions_for_targets(model_dir, target_names)
            df.to_pickle(legacy_cache_path)
        df = df.reset_index(drop=True)
        _build_row_keys(df).to_pickle(row_key_path)
        np.save(pred_path, df["pred"].to_numpy(dtype=float))
        if template_path is not None:
            df.to_pickle(template_path)
    except Exception:
        traceback.print_exc()
        raise


def _ensure_compact_target_prediction_cache(
    model_dir: str,
    target_names: Sequence[str],
    legacy_cache_path: str,
    row_key_path: str,
    pred_path: str,
    template_path: Optional[str],
) -> None:
    required_paths = [row_key_path, pred_path]
    if template_path is not None:
        required_paths.append(template_path)
    if all(os.path.exists(path) for path in required_paths):
        return
    os.makedirs(os.path.dirname(legacy_cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(row_key_path), exist_ok=True)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    if template_path is not None:
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
    ctx = mp.get_context("fork")
    proc = ctx.Process(
        target=_write_compact_target_prediction_cache,
        args=(model_dir, target_names, legacy_cache_path, row_key_path, pred_path, template_path),
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"Failed to write compact target prediction cache for {model_dir}; "
            f"child exit code {proc.exitcode}"
        )


def _load_ensemble_eval_df_for_targets_streaming(
    runtime: Dict[str, object],
    model_key: str,
    target_names: Sequence[str],
) -> pd.DataFrame:
    cache = runtime.setdefault("target_eval_df_streaming_cache", {})
    target_key = tuple(sorted(str(target_name).strip().lower() for target_name in target_names))
    cache_key = (model_key, target_key)
    if cache_key in cache:
        return cache[cache_key].copy()
    model_cfg = _model_cfg(runtime["cfg"], model_key)
    member_dirs = _select_model_member_dirs(model_cfg)
    if len(member_dirs) == 0:
        raise ValueError(f"No ensemble members were found for model '{model_key}'")
    template = None
    template_row_keys = None
    pred_arrays = []
    legacy_member_cache_dir = os.path.join(
        _model_cache_dir(runtime, model_key),
        "target_prediction_cache",
        "_".join(target_key),
    )
    compact_member_cache_dir = os.path.join(
        _model_cache_dir(runtime, model_key),
        "target_prediction_compact_cache",
        "_".join(target_key),
    )
    for member_idx, member_dir in enumerate(member_dirs):
        member_cache_name = hashlib.sha1(str(member_dir).encode("utf-8")).hexdigest()[:12]
        legacy_member_cache_path = os.path.join(
            legacy_member_cache_dir,
            f"member_{member_idx:02d}_{member_cache_name}.pkl",
        )
        row_key_path = os.path.join(
            compact_member_cache_dir,
            f"member_{member_idx:02d}_{member_cache_name}_row_keys.pkl",
        )
        pred_path = os.path.join(
            compact_member_cache_dir,
            f"member_{member_idx:02d}_{member_cache_name}_pred.npy",
        )
        template_path = None
        if member_idx == 0:
            template_path = os.path.join(
                compact_member_cache_dir,
                f"member_{member_idx:02d}_{member_cache_name}_template.pkl",
            )
        print(f"Preparing compact target prediction cache for member {member_idx + 1}/{len(member_dirs)}")
        _ensure_compact_target_prediction_cache(
            member_dir,
            target_names,
            legacy_member_cache_path,
            row_key_path,
            pred_path,
            template_path,
        )
        member_row_keys = pd.read_pickle(row_key_path)
        member_pred = np.load(pred_path)
        if template is None:
            template = pd.read_pickle(template_path).reset_index(drop=True)
            template_row_keys = member_row_keys.copy()
        else:
            if len(member_pred) != len(template):
                raise ValueError(
                    f"Member {member_idx} row count mismatch: {len(member_pred)} vs {len(template)}"
                )
            if not template_row_keys.equals(member_row_keys):
                raise ValueError(
                    f"Member {member_idx} row alignment mismatch in streaming ensemble evaluation"
                )
        pred_arrays.append(np.asarray(member_pred, dtype=float).copy())
        print(f"  Added cached predictions for member {member_idx + 1}/{len(member_dirs)}")
        del member_row_keys, member_pred
        gc.collect()
    print("Combining cached member predictions")
    pred_stack = np.stack(pred_arrays, axis=1)
    out = template.copy()
    out["pred"] = pred_stack.mean(axis=1)
    out["pred_std_ensemble"] = pred_stack.std(axis=1, ddof=0)
    print(f"Built streaming ensemble dataframe with {len(out)} rows")
    cache[cache_key] = out
    return out.copy()


def _load_eval_context(
    runtime: Dict[str, object],
    model_key: str,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    cache = runtime.setdefault("eval_contexts", {})
    target_key = None
    if target_names is not None:
        target_key = tuple(sorted(str(target_name).strip().lower() for target_name in target_names))
    cache_key = (model_key, target_key)
    if cache_key in cache:
        return cache[cache_key]
    model_cfg = _model_cfg(runtime["cfg"], model_key)
    member_dirs = _select_model_member_dirs(model_cfg)
    if target_names is None:
        member_eval_dfs = [load_fold_predictions(member_dir) for member_dir in member_dirs]
    else:
        member_eval_dfs = [
            _load_fold_predictions_for_targets(member_dir, target_names)
            for member_dir in member_dirs
        ]
    context = {
        "model_key": model_key,
        "member_dirs": member_dirs,
        "member_eval_dfs": member_eval_dfs,
        "eval_df": _build_ensemble_eval_df(member_eval_dfs),
    }
    cache[cache_key] = context
    return context


def _select_representative_member_dir(
    cfg: Dict[str, object],
    model_key: str,
    representative_member_index: int = 0,
) -> str:
    member_dirs = _select_model_member_dirs(_model_cfg(cfg, model_key))
    if len(member_dirs) == 0:
        raise ValueError(f"No ensemble members were found for model '{model_key}'")
    member_idx = int(representative_member_index)
    if member_idx < 0 or member_idx >= len(member_dirs):
        raise IndexError(
            f"Representative member index {member_idx} is out of range for model '{model_key}' "
            f"with {len(member_dirs)} members"
        )
    return member_dirs[member_idx]


def _load_operational_lfmc_map_dataset(runtime: Dict[str, object]) -> xr.Dataset:
    zarr_path = str(runtime["cfg"]["paths"]["operational_lfmc_map_zarr"])
    cache = runtime.setdefault("map_dataset_cache", {})
    if zarr_path in cache:
        return cache[zarr_path]
    ds = xr.open_zarr(zarr_path, consolidated=False)
    cache[zarr_path] = ds
    return ds


def _model_cache_dir(runtime: Dict[str, object], model_key: str) -> str:
    cache_dir = os.path.join(
        runtime["cache_dir"],
        f"{model_key}_{_model_cache_token(runtime, model_key)}",
    )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _site_group_cols(df: pd.DataFrame) -> List[str]:
    if "site_key" in df.columns:
        return ["site_key"]
    if "latitude" in df.columns and "longitude" in df.columns:
        return ["latitude", "longitude"]
    raise KeyError("Could not determine site-group columns for site-level filtering")


def _filter_site_level_rows(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    work = df.copy()
    if len(work) == 0 or int(min_obs) <= 1:
        return work
    site_cols = _site_group_cols(work)
    site_counts = (
        work.groupby(site_cols, dropna=False)
        .size()
        .rename("_site_obs_n")
        .reset_index()
    )
    work = work.merge(site_counts, on=site_cols, how="left")
    work = work[work["_site_obs_n"] >= int(min_obs)].copy()
    return work.drop(columns="_site_obs_n")


def _build_filtered_site_space_time_tables(
    lfmc_df: pd.DataFrame,
    min_obs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered_df = _filter_site_level_rows(lfmc_df, min_obs=min_obs)
    site_summary_df, anomaly_df = build_lfmc_space_time_tables(filtered_df)
    return filtered_df, site_summary_df, anomaly_df


def _compute_landcover_site_level_metrics(
    lfmc_lc_df: pd.DataFrame,
    min_obs: int,
) -> pd.DataFrame:
    filtered_df = _filter_site_level_rows(lfmc_lc_df, min_obs=min_obs)
    if len(filtered_df) == 0:
        return pd.DataFrame(
            columns=[
                "dominant_landcover",
                "site_mean_r2",
                "site_anom_r2",
                "n_sites",
                "site_anom_n",
                "site_mean_n",
            ]
        )
    site_mean_df = (
        filtered_df.groupby(["site_key", "dominant_landcover"], dropna=False)
        .agg(
            obs=("site_obs_mean", "first"),
            pred=("site_pred_mean", "first"),
        )
        .reset_index()
    )
    records = []
    for lc in filtered_df["dominant_landcover"].dropna().astype(str).drop_duplicates().tolist():
        class_obs_df = filtered_df[filtered_df["dominant_landcover"] == lc].copy()
        class_site_mean_df = site_mean_df[site_mean_df["dominant_landcover"] == lc].copy()
        records.append(
            {
                "dominant_landcover": lc,
                "site_mean_r2": compute_basic_metrics(
                    class_site_mean_df["obs"].values,
                    class_site_mean_df["pred"].values,
                ).get("r2", np.nan),
                "site_anom_r2": compute_basic_metrics(
                    class_obs_df["site_obs_anom"].values,
                    class_obs_df["site_pred_anom"].values,
                ).get("r2", np.nan),
                "n_sites": int(class_obs_df["site_key"].nunique()),
                "site_anom_n": int(len(class_obs_df)),
                "site_mean_n": int(class_site_mean_df["site_key"].nunique()),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_filtered_site_month_anomaly_tables(
    lfmc_df: pd.DataFrame,
    min_obs: int,
    min_years: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_df, valid_groups = build_site_month_anomaly_eval_df(
        lfmc_df=lfmc_df,
        min_obs=min_obs,
        min_years=min_years,
    )
    return lfmc_df, eval_df, valid_groups


def _overall_metric_std(context: Dict[str, object]) -> Dict[str, float]:
    rmse_vals = []
    r2_vals = []
    for member_eval_df in context["member_eval_dfs"]:
        lfmc_df = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
        metrics = compute_basic_metrics(lfmc_df["obs"].values, lfmc_df["pred"].values)
        rmse_vals.append(metrics["rmse"])
        r2_vals.append(metrics["r2"])
    return {"rmse": _metric_std(rmse_vals), "r2": _metric_std(r2_vals)}


def _space_time_metric_stds(
    context: Dict[str, object],
    min_obs: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    space_metrics = []
    time_metrics = []
    for member_eval_df in context["member_eval_dfs"]:
        member_lfmc_df = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
        _, site_summary_df, anomaly_df = _build_filtered_site_space_time_tables(
            member_lfmc_df,
            min_obs=min_obs,
        )
        if len(site_summary_df) > 0:
            space_metrics.append(
                compute_basic_metrics(
                    site_summary_df["obs_mean"].values,
                    site_summary_df["pred_mean"].values,
                )
            )
        if len(anomaly_df) > 0:
            time_metrics.append(
                compute_basic_metrics(
                    anomaly_df["obs_anom"].values,
                    anomaly_df["pred_anom"].values,
                )
            )
    return (
        {
            "rmse": _metric_std([metric["rmse"] for metric in space_metrics]),
            "r2": _metric_std([metric["r2"] for metric in space_metrics]),
        },
        {
            "rmse": _metric_std([metric["rmse"] for metric in time_metrics]),
            "r2": _metric_std([metric["r2"] for metric in time_metrics]),
        },
    )


def _monthly_source_centered_metric_std(
    context: Dict[str, object],
    min_obs: int,
    min_years: int,
) -> Dict[str, float]:
    metrics = []
    for member_eval_df in context["member_eval_dfs"]:
        member_lfmc_df = build_lfmc_y2y_df(member_eval_df)
        _, month_anom_df, valid_groups = _build_filtered_site_month_anomaly_tables(
            lfmc_df=member_lfmc_df,
            min_obs=min_obs,
            min_years=min_years,
        )
        if len(month_anom_df) == 0 or len(valid_groups) == 0:
            continue
        metrics.append(
            compute_basic_metrics(
                month_anom_df["obs_dev"].values,
                month_anom_df["pred_dev"].values,
            )
        )
    return {
        "rmse": _metric_std([metric["rmse"] for metric in metrics]),
        "r2": _metric_std([metric["r2"] for metric in metrics]),
    }


def _attach_landcover_lookup(eval_df: pd.DataFrame, site_lookup_df: pd.DataFrame) -> pd.DataFrame:
    lfmc_df = build_lfmc_y2y_df(eval_df)
    if len(lfmc_df) == 0:
        return pd.DataFrame()
    lfmc_df = lfmc_df.copy()
    lfmc_df["site_key"] = lfmc_df["site_key"].astype(str)
    lfmc_df = lfmc_df.merge(
        site_lookup_df[["site_key", "dominant_landcover", "dominant_landcover_frac"]],
        on="site_key",
        how="left",
    )
    lfmc_df = lfmc_df[lfmc_df["dominant_landcover"].notna()].copy()
    if len(lfmc_df) == 0:
        return pd.DataFrame()
    lfmc_df["site_obs_mean"] = lfmc_df.groupby("site_key")["obs"].transform("mean")
    lfmc_df["site_pred_mean"] = lfmc_df.groupby("site_key")["pred"].transform("mean")
    lfmc_df["site_obs_anom"] = lfmc_df["obs"] - lfmc_df["site_obs_mean"]
    lfmc_df["site_pred_anom"] = lfmc_df["pred"] - lfmc_df["site_pred_mean"]
    grp_cols = ["site_key", "month"]
    lfmc_df["seasonal_obs_mean"] = lfmc_df.groupby(grp_cols)["obs"].transform("mean")
    lfmc_df["seasonal_pred_mean"] = lfmc_df.groupby(grp_cols)["pred"].transform("mean")
    lfmc_df["seasonal_obs_anom"] = lfmc_df["obs"] - lfmc_df["seasonal_obs_mean"]
    lfmc_df["seasonal_pred_anom"] = lfmc_df["pred"] - lfmc_df["seasonal_pred_mean"]
    return lfmc_df


def _landcover_metric_table(
    runtime: Dict[str, object],
    model_key: str,
    site_min_obs: int,
    monthly_min_obs: int,
    monthly_min_years: int,
) -> pd.DataFrame:
    cache_key = (model_key, site_min_obs, monthly_min_obs, monthly_min_years)
    if cache_key in runtime["landcover_metric_tables"]:
        return runtime["landcover_metric_tables"][cache_key]
    context = _load_eval_context(runtime, model_key)
    cache_dir = _model_cache_dir(runtime, model_key)
    site_lookup_path = os.path.join(cache_dir, "lfmc_site_landcover_lookup.csv")
    site_lookup_df = build_site_landcover_lookup(
        build_lfmc_y2y_df(context["eval_df"]),
        site_lookup_path,
    )
    site_lookup_df = site_lookup_df.copy()
    site_lookup_df["site_key"] = site_lookup_df["site_key"].astype(str)
    lfmc_lc_df = _attach_landcover_lookup(context["eval_df"], site_lookup_df)
    metric_df = compute_landcover_decomposition_metrics(lfmc_lc_df)
    site_metric_df = _compute_landcover_site_level_metrics(lfmc_lc_df, min_obs=site_min_obs)
    if len(site_metric_df) > 0:
        site_metric_lookup = site_metric_df.set_index("dominant_landcover").to_dict("index")
        metric_df["site_mean_r2"] = metric_df["dominant_landcover"].map(
            lambda lc: site_metric_lookup.get(lc, {}).get("site_mean_r2", np.nan)
        )
        metric_df["site_anom_r2"] = metric_df["dominant_landcover"].map(
            lambda lc: site_metric_lookup.get(lc, {}).get("site_anom_r2", np.nan)
        )
        metric_df["n_sites"] = metric_df["dominant_landcover"].map(
            lambda lc: site_metric_lookup.get(lc, {}).get("n_sites", 0)
        )
        metric_df["site_anom_n"] = metric_df["dominant_landcover"].map(
            lambda lc: site_metric_lookup.get(lc, {}).get("site_anom_n", np.nan)
        )
        metric_df["site_mean_n"] = metric_df["dominant_landcover"].map(
            lambda lc: site_metric_lookup.get(lc, {}).get("site_mean_n", np.nan)
        )
    else:
        metric_df["site_anom_n"] = np.nan
        metric_df["site_mean_n"] = np.nan

    member_metric_frames = []
    member_site_metric_frames = []
    for member_eval_df in context["member_eval_dfs"]:
        member_lfmc_lc_df = _attach_landcover_lookup(member_eval_df, site_lookup_df)
        if len(member_lfmc_lc_df) == 0:
            continue
        member_metric_frames.append(compute_landcover_decomposition_metrics(member_lfmc_lc_df))
        member_site_metric_frames.append(
            _compute_landcover_site_level_metrics(member_lfmc_lc_df, min_obs=site_min_obs)
        )

    if len(member_metric_frames) > 0:
        std_lookup = {}
        for lc in metric_df["dominant_landcover"].tolist():
            vals = []
            for member_metric_df in member_metric_frames:
                row = member_metric_df[member_metric_df["dominant_landcover"] == lc]
                if len(row) == 0:
                    continue
                vals.append(float(row.iloc[0]["overall_r2"]))
            std_lookup[lc] = _metric_std(vals)
        metric_df["overall_r2_std"] = metric_df["dominant_landcover"].map(std_lookup)

    if len(member_site_metric_frames) > 0:
        for metric_name in ["site_mean_r2", "site_anom_r2"]:
            std_lookup = {}
            for lc in metric_df["dominant_landcover"].tolist():
                vals = []
                for member_metric_df in member_site_metric_frames:
                    row = member_metric_df[member_metric_df["dominant_landcover"] == lc]
                    if len(row) == 0:
                        continue
                    vals.append(float(row.iloc[0][metric_name]))
                std_lookup[lc] = _metric_std(vals)
            metric_df[f"{metric_name}_std"] = metric_df["dominant_landcover"].map(std_lookup)

    _, month_anom_df, valid_month_groups = _build_filtered_site_month_anomaly_tables(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=monthly_min_obs,
        min_years=monthly_min_years,
    )
    if len(month_anom_df) > 0 and len(valid_month_groups) > 0:
        month_anom_df = month_anom_df.merge(
            site_lookup_df[["site_key", "dominant_landcover"]],
            on="site_key",
            how="left",
        )
        month_anom_df = month_anom_df[month_anom_df["dominant_landcover"].notna()].copy()
        month_r2_df = (
            month_anom_df.groupby("dominant_landcover", dropna=False)
            .apply(
                lambda df: pd.Series(
                    {
                        "monthly_dev_r2": compute_basic_metrics(
                            df["obs_dev"].values,
                            df["pred_dev"].values,
                        )["r2"],
                        "total_obs": int(len(df)),
                    }
                )
            )
            .reset_index()
        )
        metric_df = metric_df.merge(
            month_r2_df,
            on="dominant_landcover",
            how="left",
        )
        metric_df["monthly_dev_n"] = metric_df["total_obs"]
        std_lookup = {}
        for lc in metric_df["dominant_landcover"].tolist():
            vals = []
            for member_eval_df in context["member_eval_dfs"]:
                _, member_month_anom_df, member_valid_groups = _build_filtered_site_month_anomaly_tables(
                    lfmc_df=build_lfmc_y2y_df(member_eval_df),
                    min_obs=monthly_min_obs,
                    min_years=monthly_min_years,
                )
                if len(member_month_anom_df) == 0 or len(member_valid_groups) == 0:
                    continue
                member_month_anom_df = member_month_anom_df.merge(
                    site_lookup_df[["site_key", "dominant_landcover"]],
                    on="site_key",
                    how="left",
                )
                member_month_anom_df = member_month_anom_df[
                    member_month_anom_df["dominant_landcover"] == lc
                ].copy()
                if len(member_month_anom_df) == 0:
                    continue
                vals.append(
                    compute_basic_metrics(
                        member_month_anom_df["obs_dev"].values,
                        member_month_anom_df["pred_dev"].values,
                    )["r2"]
                )
            std_lookup[lc] = _metric_std(vals)
        metric_df["monthly_dev_r2_std"] = metric_df["dominant_landcover"].map(std_lookup)
    else:
        metric_df["monthly_dev_r2"] = np.nan
        metric_df["monthly_dev_r2_std"] = np.nan
        metric_df["total_obs"] = np.nan
        metric_df["monthly_dev_n"] = np.nan

    metric_df["overall_n"] = metric_df["n_points"]
    if "site_mean_n" not in metric_df.columns:
        metric_df["site_mean_n"] = metric_df["n_sites"]

    metric_df = _filtered_landcover_df(metric_df, runtime["cfg"])
    runtime["landcover_metric_tables"][cache_key] = metric_df
    return metric_df


def _prepend_overall_landcover_metrics(
    runtime: Dict[str, object],
    model_key: str,
    metric_df: pd.DataFrame,
    site_min_obs: int,
    monthly_min_obs: int,
    monthly_min_years: int,
) -> pd.DataFrame:
    context = _load_eval_context(runtime, model_key)
    lfmc_df = context["eval_df"][context["eval_df"]["target"] == "lfmc"].reset_index(drop=True)
    _, site_summary_df, anomaly_df = _build_filtered_site_space_time_tables(
        lfmc_df,
        min_obs=site_min_obs,
    )
    _, month_anom_df, _ = _build_filtered_site_month_anomaly_tables(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=monthly_min_obs,
        min_years=monthly_min_years,
    )
    overall_std = _overall_metric_std(context)
    mean_std, anomaly_std = _space_time_metric_stds(context, min_obs=site_min_obs)
    monthly_std = _monthly_source_centered_metric_std(
        context,
        min_obs=monthly_min_obs,
        min_years=monthly_min_years,
    )
    overall_metric = compute_basic_metrics(lfmc_df["obs"].values, lfmc_df["pred"].values)
    anomaly_metric = compute_basic_metrics(anomaly_df["obs_anom"].values, anomaly_df["pred_anom"].values)
    mean_metric = compute_basic_metrics(site_summary_df["obs_mean"].values, site_summary_df["pred_mean"].values)
    monthly_metric = compute_basic_metrics(month_anom_df["obs_dev"].values, month_anom_df["pred_dev"].values)
    overall_row = {column: np.nan for column in metric_df.columns}
    overall_row["dominant_landcover"] = "overall"
    overall_row["overall_r2"] = overall_metric.get("r2", np.nan)
    overall_row["site_anom_r2"] = anomaly_metric.get("r2", np.nan)
    overall_row["site_mean_r2"] = mean_metric.get("r2", np.nan)
    overall_row["monthly_dev_r2"] = monthly_metric.get("r2", np.nan)
    overall_row["overall_r2_std"] = overall_std.get("r2", np.nan)
    overall_row["site_anom_r2_std"] = anomaly_std.get("r2", np.nan)
    overall_row["site_mean_r2_std"] = mean_std.get("r2", np.nan)
    overall_row["monthly_dev_r2_std"] = monthly_std.get("r2", np.nan)
    overall_row["n_points"] = overall_metric.get("n", np.nan)
    overall_row["n_sites"] = mean_metric.get("n", np.nan)
    overall_row["total_obs"] = monthly_metric.get("n", np.nan)
    overall_row["overall_n"] = overall_metric.get("n", np.nan)
    overall_row["site_anom_n"] = anomaly_metric.get("n", np.nan)
    overall_row["site_mean_n"] = mean_metric.get("n", np.nan)
    overall_row["monthly_dev_n"] = monthly_metric.get("n", np.nan)
    return pd.concat([pd.DataFrame([overall_row]), metric_df], ignore_index=True)


def _build_site_r2_landcover_df(
    runtime: Dict[str, object],
    model_key: str,
    min_obs: int,
) -> pd.DataFrame:
    cache_key = (model_key, int(min_obs))
    if cache_key in runtime["site_r2_landcover_tables"]:
        return runtime["site_r2_landcover_tables"][cache_key].copy()
    context = _load_eval_context(runtime, model_key)
    lfmc_df = context["eval_df"][context["eval_df"]["target"] == "lfmc"].reset_index(drop=True)
    if len(lfmc_df) == 0:
        out = pd.DataFrame(
            columns=[
                "site_key",
                "latitude",
                "longitude",
                "year",
                "n_points",
                "site_r2",
                "dominant_landcover",
                "dominant_landcover_frac",
            ]
        )
        runtime["site_r2_landcover_tables"][cache_key] = out
        return out.copy()
    lfmc_df = lfmc_df.copy()
    if "site_key" not in lfmc_df.columns:
        if "latitude" not in lfmc_df.columns or "longitude" not in lfmc_df.columns:
            raise KeyError("LFMC evaluation rows are missing site_key and latitude/longitude columns")
        lfmc_df["site_key"] = (
            pd.to_numeric(lfmc_df["latitude"], errors="coerce").astype(str)
            + "_"
            + pd.to_numeric(lfmc_df["longitude"], errors="coerce").astype(str)
        )
    if "year" not in lfmc_df.columns:
        if "date" not in lfmc_df.columns:
            raise KeyError("LFMC evaluation rows are missing year and date columns")
        lfmc_df["year"] = pd.to_datetime(lfmc_df["date"], errors="coerce").dt.year
    site_df = (
        lfmc_df.groupby("site_key", dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "latitude": float(pd.to_numeric(group["latitude"], errors="coerce").iloc[0]),
                    "longitude": float(pd.to_numeric(group["longitude"], errors="coerce").iloc[0]),
                    "year": int(pd.to_numeric(group["year"], errors="coerce").dropna().min()),
                    "n_points": int(len(group)),
                    "obs": np.asarray(group["obs"], dtype=float),
                    "pred": np.asarray(group["pred"], dtype=float),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    site_df = site_df[site_df["n_points"] >= int(min_obs)].copy()
    if len(site_df) == 0:
        out = pd.DataFrame(
            columns=[
                "site_key",
                "latitude",
                "longitude",
                "year",
                "n_points",
                "site_r2",
                "dominant_landcover",
                "dominant_landcover_frac",
            ]
        )
        runtime["site_r2_landcover_tables"][cache_key] = out
        return out.copy()
    site_df["site_r2"] = site_df.apply(
        lambda row: float(compute_basic_metrics(row["obs"], row["pred"]).get("r2", np.nan)),
        axis=1,
    )
    lookup_path = os.path.join(
        _model_cache_dir(runtime, model_key),
        "site_r2_landcover_lookup.csv",
    )
    lookup_df = build_site_landcover_lookup(
        site_df[["site_key", "latitude", "longitude", "year"]].copy(),
        lookup_path,
    )
    lookup_df["site_key"] = lookup_df["site_key"].astype(str)
    site_df["site_key"] = site_df["site_key"].astype(str)
    site_df = site_df.merge(
        lookup_df[["site_key", "dominant_landcover", "dominant_landcover_frac"]],
        on="site_key",
        how="left",
    )
    site_df = site_df[site_df["dominant_landcover"].notna()].copy()
    site_df = site_df.drop(columns=["obs", "pred"])
    site_df = _filtered_landcover_df(site_df, runtime["cfg"])
    site_df = site_df.sort_values(
        ["dominant_landcover", "site_r2", "site_key"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    runtime["site_r2_landcover_tables"][cache_key] = site_df
    return site_df.copy()


def _load_train_info_fold(
    runtime: Dict[str, object],
    member_dir: str,
    fold_num: int,
) -> pd.DataFrame:
    cache_key = (member_dir, int(fold_num))
    cache = runtime.setdefault("train_info_fold_cache", {})
    if cache_key in cache:
        return cache[cache_key].copy()
    train_info_path = os.path.join(member_dir, f"fold_{int(fold_num)}", "train_info.csv")
    if not os.path.exists(train_info_path):
        raise FileNotFoundError(f"Missing training info: {train_info_path}")
    df = pd.read_csv(train_info_path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cache[cache_key] = df
    return df.copy()


def _load_train_info_union(
    runtime: Dict[str, object],
    member_dir: str,
) -> pd.DataFrame:
    cache = runtime.setdefault("train_info_union_cache", {})
    if member_dir in cache:
        return cache[member_dir].copy()
    fold_frames = [
        _load_train_info_fold(runtime, member_dir, fold_num)
        for fold_num in range(1, 7)
    ]
    df = pd.concat(fold_frames, ignore_index=True, sort=False)
    df = df.drop_duplicates().reset_index(drop=True)
    cache[member_dir] = df
    return df.copy()


def _normalize_train_landcover_column(
    runtime: Dict[str, object],
    df: pd.DataFrame,
    cache_stem: str,
) -> pd.DataFrame:
    work = df.copy()
    if "landcover" in work.columns:
        work["dominant_landcover"] = (
            work["landcover"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
    else:
        work["dominant_landcover"] = pd.NA
    missing_lc_mask = work["dominant_landcover"].isna() | (work["dominant_landcover"] == "")
    if bool(missing_lc_mask.any()):
        lookup_df = work.loc[missing_lc_mask, ["date", "latitude", "longitude"]].copy()
        lookup_df["date"] = pd.to_datetime(lookup_df["date"], errors="coerce")
        lookup_df["year"] = lookup_df["date"].dt.year
        lookup_df["latitude"] = pd.to_numeric(lookup_df["latitude"], errors="coerce")
        lookup_df["longitude"] = pd.to_numeric(lookup_df["longitude"], errors="coerce")
        lookup_df = lookup_df.dropna(subset=["year", "latitude", "longitude"]).copy()
        if len(lookup_df) > 0:
            lookup_df["point_key"] = _sample_index_point_key(
                lookup_df["latitude"],
                lookup_df["longitude"],
            )
            save_path = os.path.join(
                runtime["cache_dir"],
                f"{cache_stem}_point_landcover_lookup.csv",
            )
            point_lookup_df = _build_point_landcover_lookup_fast(
                lookup_df[["point_key", "latitude", "longitude", "year"]]
                .drop_duplicates()
                .reset_index(drop=True),
                save_path,
            )
            work["date"] = pd.to_datetime(work["date"], errors="coerce")
            work["year"] = work["date"].dt.year
            work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
            work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
            work["point_key"] = _sample_index_point_key(
                work["latitude"],
                work["longitude"],
            )
            point_lookup_df = point_lookup_df.rename(columns={"dominant_landcover": "lookup_landcover"})
            work = work.merge(
                point_lookup_df[["point_key", "year", "lookup_landcover"]],
                on=["point_key", "year"],
                how="left",
            )
            lookup_landcover = (
                work["lookup_landcover"]
                .astype("string")
                .str.strip()
                .str.lower()
            )
            work["dominant_landcover"] = work["dominant_landcover"].fillna(lookup_landcover)
            work = work.drop(columns=["lookup_landcover"])
    work = work[work["dominant_landcover"].notna() & (work["dominant_landcover"] != "")].copy()
    return work


def _count_landcover_rows_from_train_info(
    runtime: Dict[str, object],
    train_info_df: pd.DataFrame,
    cache_stem: str,
    target_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    work = train_info_df.copy()
    if target_names is not None:
        requested_targets = {str(target_name).strip().lower() for target_name in target_names}
        work = work[
            work["target_name"].astype(str).str.strip().str.lower().isin(requested_targets)
        ].copy()
    work = _normalize_train_landcover_column(
        runtime=runtime,
        df=work,
        cache_stem=cache_stem,
    )
    counts = (
        work.groupby("dominant_landcover", as_index=False)
        .size()
        .rename(columns={"size": "n_samples"})
    )
    return counts


def _build_training_sample_landcover_tables(
    runtime: Dict[str, object],
    fig_cfg: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_cfgs = dict(fig_cfg.get("datasets", {}))
    dataset_order = list(fig_cfg.get("dataset_order", dataset_cfgs.keys()))
    if len(dataset_cfgs) == 0:
        raise ValueError("Supplementary Figure 4 requires at least one configured dataset")
    member_records = []
    for dataset_key in dataset_order:
        dataset_cfg = dataset_cfgs[dataset_key]
        label = str(dataset_cfg["label"])
        color = str(dataset_cfg["color"])
        model_key = str(dataset_cfg["model_key"])
        model_cfg = _model_cfg(runtime["cfg"], model_key)
        member_dirs = _select_model_member_dirs(model_cfg)
        if len(member_dirs) == 0:
            raise ValueError(f"No ensemble members were found for dataset '{dataset_key}'")
        target_names = dataset_cfg.get("target_names")
        for member_dir in member_dirs:
            train_info_df = _load_train_info_union(
                runtime=runtime,
                member_dir=member_dir,
            )
            counts_df = _count_landcover_rows_from_train_info(
                runtime=runtime,
                train_info_df=train_info_df,
                cache_stem=f"figure_08_{dataset_key}_{os.path.basename(member_dir)}",
                target_names=target_names,
            )
            for row in counts_df.itertuples(index=False):
                member_records.append(
                    {
                        "dataset_key": dataset_key,
                        "label": label,
                        "color": color,
                        "member_id": os.path.basename(member_dir),
                        "dominant_landcover": str(row.dominant_landcover),
                        "n_samples": int(row.n_samples),
                    }
                )
    member_df = pd.DataFrame.from_records(member_records)
    if len(member_df) == 0:
        empty = pd.DataFrame(
            columns=[
                "dataset_key",
                "label",
                "color",
                "dominant_landcover",
                "mean_n_samples",
                "std_n_samples",
                "fraction",
                "fraction_std",
                "total_n",
                "n_members",
            ]
        )
        return empty, empty.copy()
    member_df = _filtered_landcover_df(member_df, runtime["cfg"])
    summary_df = (
        member_df.groupby(
            ["dataset_key", "label", "color", "dominant_landcover"],
            as_index=False,
            dropna=False,
            observed=True,
        )
        .agg(
            mean_n_samples=("n_samples", "mean"),
            std_n_samples=("n_samples", "std"),
            n_members=("member_id", "nunique"),
        )
    )
    summary_df["std_n_samples"] = pd.to_numeric(
        summary_df["std_n_samples"],
        errors="coerce",
    ).fillna(0.0)
    summary_df["total_n"] = summary_df.groupby(
        "dataset_key",
        dropna=False,
        observed=True,
    )["mean_n_samples"].transform("sum")
    summary_df["fraction"] = np.where(
        summary_df["total_n"] > 0,
        summary_df["mean_n_samples"] / summary_df["total_n"],
        np.nan,
    )
    summary_df["fraction_std"] = np.where(
        summary_df["total_n"] > 0,
        summary_df["std_n_samples"] / summary_df["total_n"],
        np.nan,
    )
    summary_df = _filtered_landcover_df(summary_df, runtime["cfg"])
    return summary_df, member_df


def _load_test_row_union(
    runtime: Dict[str, object],
    member_dir: str,
) -> pd.DataFrame:
    cache = runtime.setdefault("test_row_union_cache", {})
    if member_dir in cache:
        return cache[member_dir].copy()
    fold_info_path = os.path.join(member_dir, "fold_info.json")
    if not os.path.exists(fold_info_path):
        raise FileNotFoundError(f"Missing fold_info.json in {member_dir}")
    with open(fold_info_path, "r") as file_obj:
        fold_info = json.load(file_obj)
    fold_frames = []
    for fold in fold_info.keys():
        fold_dir = os.path.join(member_dir, f"fold_{fold}")
        test_info_path = os.path.join(fold_dir, "test_info.csv")
        if not os.path.exists(test_info_path):
            raise FileNotFoundError(f"Missing test info: {test_info_path}")
        test_info = pd.read_csv(test_info_path, low_memory=False)
        source = test_info["source"].astype(str)
        target_masks = [
            ("lfmc", source == "nfmd"),
            ("vv", source.str.startswith("vv")),
            ("vh", source.str.startswith("vh")),
        ]
        for target_name, mask in target_masks:
            target_df = test_info.loc[mask].reset_index(drop=True).copy()
            if len(target_df) == 0:
                continue
            target_df["target"] = target_name
            target_df["fold"] = str(fold)
            fold_frames.append(target_df)
    if len(fold_frames) == 0:
        raise ValueError(f"No test rows found in model dir: {member_dir}")
    df = pd.concat(fold_frames, ignore_index=True, sort=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cache[member_dir] = df
    return df.copy()


def _build_test_obs_row_key(frame: pd.DataFrame) -> pd.Series:
    key_cols = [
        col for col in [
            "target",
            "date",
            "sample_id",
            "site_id",
            "latitude",
            "longitude",
            "source",
            "source_legible",
            "site_name",
            "fuel_type",
            "target_value",
        ]
        if col in frame.columns
    ]
    if len(key_cols) == 0:
        return pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index)
    work = frame[key_cols].copy()
    for col in work.columns:
        if np.issubdtype(np.asarray(work[col]).dtype, np.datetime64):
            work[col] = pd.to_datetime(work[col], errors="coerce").astype(str)
        else:
            work[col] = work[col].fillna("__nan__").astype(str)
    return work.agg("|".join, axis=1)


def _build_test_location_count_table(
    runtime: Dict[str, object],
    model_key: str,
    target_names: Sequence[str],
    representative_member_index: int = 0,
    include_reference_member_flag: bool = False,
    lfmc_location_keys: Optional[set] = None,
) -> pd.DataFrame:
    model_cfg = _model_cfg(runtime["cfg"], model_key)
    member_dirs = _select_model_member_dirs(model_cfg)
    if len(member_dirs) == 0:
        raise ValueError(f"No ensemble members were found for model '{model_key}'")
    member_idx = int(representative_member_index)
    if member_idx < 0 or member_idx >= len(member_dirs):
        raise ValueError(
            f"Representative member index {member_idx} is out of range for model '{model_key}' "
            f"with {len(member_dirs)} members"
        )

    target_name_lookup = {str(target_name).strip().lower() for target_name in target_names}
    reference_member_dir = member_dirs[member_idx]
    reference_location_keys = set()
    seen_row_keys = set()
    location_counts = {}
    member_location_counts = {}
    for member_dir in member_dirs:
        member_df = _load_test_row_union(runtime, member_dir)
        member_df = member_df[
            member_df["target"].astype(str).str.strip().str.lower().isin(target_name_lookup)
        ].copy()
        if len(member_df) == 0:
            continue
        member_df["_row_key"] = _build_test_obs_row_key(member_df)
        member_df["latitude"] = pd.to_numeric(member_df["latitude"], errors="coerce")
        member_df["longitude"] = pd.to_numeric(member_df["longitude"], errors="coerce")
        member_df = member_df.dropna(subset=["latitude", "longitude", "_row_key"]).copy()
        if len(member_df) == 0:
            continue
        member_location_keys = {
            (float(row.latitude), float(row.longitude))
            for row in member_df[["latitude", "longitude"]].drop_duplicates().itertuples(index=False)
        }
        for location_key in member_location_keys:
            member_location_counts[location_key] = member_location_counts.get(location_key, 0) + 1
        if member_dir == reference_member_dir:
            reference_location_keys = member_location_keys
        member_df = member_df.drop_duplicates(subset=["_row_key"]).reset_index(drop=True)
        for row in member_df[["_row_key", "latitude", "longitude"]].itertuples(index=False):
            row_key = str(row[0])
            if row_key in seen_row_keys:
                continue
            seen_row_keys.add(row_key)
            location_key = (float(row.latitude), float(row.longitude))
            location_counts[location_key] = location_counts.get(location_key, 0) + 1
    if len(location_counts) == 0:
        raise ValueError(
            f"No test rows matched targets {list(target_names)} for model '{model_key}'"
        )
    grouped = pd.DataFrame.from_records(
        [
            {
                "latitude": location_key[0],
                "longitude": location_key[1],
                "n_points": int(n_points),
                "n_members": int(member_location_counts.get(location_key, 0)),
            }
            for location_key, n_points in location_counts.items()
        ]
    )
    grouped["marker_group"] = "all_points"
    if include_reference_member_flag:
        if lfmc_location_keys is not None:
            def _classify_sar_location(row):
                key = (float(row["latitude"]), float(row["longitude"]))
                if key in lfmc_location_keys:
                    return "lfmc_coincident"
                if key in reference_location_keys:
                    return "member_1_non_coincident"
                return "other_members_non_coincident"
            grouped["marker_group"] = grouped.apply(_classify_sar_location, axis=1)
        else:
            grouped["marker_group"] = grouped.apply(
                lambda row: (
                    "member_1"
                    if (float(row["latitude"]), float(row["longitude"])) in reference_location_keys
                    else "other_members"
                ),
                axis=1,
            )
    grouped["model_key"] = model_key
    grouped["reference_member_dir"] = os.path.basename(reference_member_dir)
    grouped["target_group"] = "+".join(str(target_name) for target_name in target_names)
    return grouped.sort_values(
        ["n_points", "n_members", "latitude", "longitude"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _split_species_names(values: pd.Series) -> List[str]:
    species_names = set()
    for value in values.dropna().astype(str):
        for part in value.split(";"):
            name = part.strip()
            if name != "":
                species_names.add(name)
    return sorted(species_names)


def _target_member_unique_rows(
    runtime: Dict[str, object],
    member_dir: str,
    target_names: Sequence[str],
) -> pd.DataFrame:
    target_name_lookup = {str(target_name).strip().lower() for target_name in target_names}
    df = _load_test_row_union(runtime, member_dir)
    df = df[
        df["target"].astype(str).str.strip().str.lower().isin(target_name_lookup)
    ].copy()
    if len(df) == 0:
        return df
    df["_row_key"] = _build_test_obs_row_key(df)
    df = df.drop_duplicates(subset=["_row_key"]).reset_index(drop=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["measurement_day"] = df["date"].dt.normalize()
    else:
        df["measurement_day"] = pd.NaT
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return df


def _union_unique_target_rows(
    runtime: Dict[str, object],
    member_dirs: Sequence[str],
    target_names: Sequence[str],
) -> pd.DataFrame:
    seen_row_keys = set()
    frames = []
    for member_dir in member_dirs:
        member_df = _target_member_unique_rows(runtime, member_dir, target_names)
        if len(member_df) == 0:
            continue
        keep_mask = ~member_df["_row_key"].astype(str).isin(seen_row_keys)
        new_df = member_df.loc[keep_mask].copy()
        seen_row_keys.update(new_df["_row_key"].astype(str).tolist())
        if len(new_df) > 0:
            frames.append(new_df)
    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric)]


def _add_distribution_stats(row: Dict[str, object], prefix: str, values: pd.Series) -> None:
    finite = _finite_series(values)
    row[f"mean_{prefix}"] = float(finite.mean()) if len(finite) > 0 else np.nan
    row[f"sd_{prefix}"] = float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
    row[f"median_{prefix}"] = float(finite.median()) if len(finite) > 0 else np.nan
    row[f"min_{prefix}"] = float(finite.min()) if len(finite) > 0 else np.nan
    row[f"max_{prefix}"] = float(finite.max()) if len(finite) > 0 else np.nan


def _lfmc_site_detail_table(lfmc_rows: pd.DataFrame) -> pd.DataFrame:
    detail_rows = []
    if len(lfmc_rows) == 0:
        return pd.DataFrame(
            columns=[
                "latitude",
                "longitude",
                "n_observations",
                "n_measurement_days",
                "n_species",
                "species_site_class",
                "species_names",
            ]
        )
    work = lfmc_rows.copy()
    work = work.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if "measurement_day" not in work.columns:
        work["measurement_day"] = pd.NaT
    for (latitude, longitude), group in work.groupby(["latitude", "longitude"], dropna=False):
        species = _split_species_names(group["fuel_type"]) if "fuel_type" in group.columns else []
        detail_rows.append(
            {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "n_observations": int(len(group)),
                "n_measurement_days": int(group["measurement_day"].dropna().nunique()),
                "n_species": int(len(species)),
                "species_site_class": "multi_species" if len(species) > 1 else "single_species",
                "species_names": "; ".join(species),
            }
        )
    return pd.DataFrame.from_records(detail_rows).sort_values(
        ["n_observations", "latitude", "longitude"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _summarize_lfmc_site_detail(site_detail_df: pd.DataFrame) -> Dict[str, object]:
    row: Dict[str, object] = {
        "n_sites": int(len(site_detail_df)),
        "single_species_sites": int((site_detail_df["species_site_class"] == "single_species").sum()),
        "multi_species_sites": int((site_detail_df["species_site_class"] == "multi_species").sum()),
        "total_lfmc_observations": int(site_detail_df["n_observations"].sum()),
        "total_lfmc_measurement_days": int(site_detail_df["n_measurement_days"].sum()),
    }
    _add_distribution_stats(row, "observations_per_site", site_detail_df["n_observations"])
    _add_distribution_stats(row, "measurement_days_per_site", site_detail_df["n_measurement_days"])
    return row


def _build_lfmc_sampling_statistics(
    runtime: Dict[str, object],
    model_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_cfg = _model_cfg(runtime["cfg"], model_key)
    member_dirs = _select_model_member_dirs(model_cfg)
    member_rows = []
    for member_idx, member_dir in enumerate(member_dirs, start=1):
        member_df = _target_member_unique_rows(runtime, member_dir, ["lfmc"])
        site_detail_df = _lfmc_site_detail_table(member_df)
        row = _summarize_lfmc_site_detail(site_detail_df)
        row.update(
            {
                "model_key": model_key,
                "member_index": int(member_idx),
                "member_dir": os.path.basename(member_dir),
            }
        )
        member_rows.append(row)
    by_member_df = pd.DataFrame.from_records(member_rows)
    union_df = _union_unique_target_rows(runtime, member_dirs, ["lfmc"])
    site_detail_df = _lfmc_site_detail_table(union_df)
    overall_row = _summarize_lfmc_site_detail(site_detail_df)
    overall_row.update(
        {
            "model_key": model_key,
            "aggregation": "unique_rows_across_members",
            "n_members": int(len(member_dirs)),
        }
    )
    overall_df = pd.DataFrame.from_records([overall_row])
    return by_member_df, overall_df, site_detail_df


def _summarize_s1_counts(
    location_counts: pd.Series,
    day_counts: pd.Series,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "n_locations": int(len(location_counts)),
        "total_s1_observations": int(location_counts.sum()) if len(location_counts) > 0 else 0,
    }
    _add_distribution_stats(row, "s1_observations_per_location", location_counts)
    _add_distribution_stats(row, "s1_observation_days_per_location", day_counts)
    return row


def _build_s1_sampling_statistics(
    runtime: Dict[str, object],
    model_key: str,
    lfmc_location_keys: set,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_cfg = _model_cfg(runtime["cfg"], model_key)
    member_dirs = _select_model_member_dirs(model_cfg)
    member_rows = []
    coincident_rows = []
    seen_row_keys = set()
    union_location_counts = {}
    union_location_days = {}
    union_location_keys = set()
    for member_idx, member_dir in enumerate(member_dirs, start=1):
        member_df = _target_member_unique_rows(runtime, member_dir, ["vv", "vh"])
        if len(member_df) == 0:
            continue
        member_location_counts = member_df.groupby(["latitude", "longitude"], dropna=False).size()
        member_day_counts = (
            member_df.dropna(subset=["measurement_day"])
            .groupby(["latitude", "longitude"], dropna=False)["measurement_day"]
            .nunique()
        )
        row = _summarize_s1_counts(member_location_counts, member_day_counts)
        row.update(
            {
                "model_key": model_key,
                "member_index": int(member_idx),
                "member_dir": os.path.basename(member_dir),
            }
        )
        member_rows.append(row)
        member_location_keys = {
            (float(latitude), float(longitude))
            for latitude, longitude in member_location_counts.index.to_list()
        }
        coincident_count = len(member_location_keys & lfmc_location_keys)
        noncoincident_count = len(member_location_keys - lfmc_location_keys)
        coincident_rows.append(
            {
                "model_key": model_key,
                "member_index": int(member_idx),
                "member_dir": os.path.basename(member_dir),
                "s1_locations": int(len(member_location_keys)),
                "coincident_with_lfmc": int(coincident_count),
                "not_coincident_with_lfmc": int(noncoincident_count),
                "pct_coincident_with_lfmc": (
                    100.0 * float(coincident_count) / float(len(member_location_keys))
                    if len(member_location_keys) > 0 else np.nan
                ),
            }
        )
        for row_key, latitude, longitude, measurement_day in member_df[
            ["_row_key", "latitude", "longitude", "measurement_day"]
        ].itertuples(index=False, name=None):
            row_key = str(row_key)
            if row_key in seen_row_keys:
                continue
            seen_row_keys.add(row_key)
            location_key = (float(latitude), float(longitude))
            union_location_keys.add(location_key)
            union_location_counts[location_key] = union_location_counts.get(location_key, 0) + 1
            if pd.notna(measurement_day):
                union_location_days.setdefault(location_key, set()).add(pd.Timestamp(measurement_day).normalize())
    union_counts = pd.Series(union_location_counts, dtype=float)
    union_day_counts = pd.Series(
        {location_key: len(days) for location_key, days in union_location_days.items()},
        dtype=float,
    )
    overall_row = _summarize_s1_counts(union_counts, union_day_counts)
    overall_row.update(
        {
            "model_key": model_key,
            "aggregation": "unique_rows_across_members",
            "n_members": int(len(member_dirs)),
        }
    )
    union_coincident_count = len(union_location_keys & lfmc_location_keys)
    union_noncoincident_count = len(union_location_keys - lfmc_location_keys)
    coincident_overall_df = pd.DataFrame.from_records(
        [
            {
                "model_key": model_key,
                "aggregation": "unique_locations_across_members",
                "s1_locations": int(len(union_location_keys)),
                "coincident_with_lfmc": int(union_coincident_count),
                "not_coincident_with_lfmc": int(union_noncoincident_count),
                "pct_coincident_with_lfmc": (
                    100.0 * float(union_coincident_count) / float(len(union_location_keys))
                    if len(union_location_keys) > 0 else np.nan
                ),
            }
        ]
    )
    return (
        pd.DataFrame.from_records(member_rows),
        pd.DataFrame.from_records([overall_row]),
        pd.DataFrame.from_records(coincident_rows),
        coincident_overall_df,
    )


def _dry_lfmc_performance_for_eval_df(
    eval_df: pd.DataFrame,
    model_key: str,
    member_label: str,
    min_obs: int,
    dry_quantile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lfmc_df = build_lfmc_y2y_df(eval_df)
    if len(lfmc_df) == 0:
        empty_summary = pd.DataFrame()
        empty_site = pd.DataFrame()
        return empty_summary, empty_site
    lfmc_df = lfmc_df.copy()
    lfmc_df["site_key"] = lfmc_df["site_key"].astype(str)
    eligible_site_counts = lfmc_df.groupby("site_key", dropna=False).size()
    eligible_site_keys = set(eligible_site_counts[eligible_site_counts >= int(min_obs)].index.astype(str))
    eligible_df = lfmc_df[lfmc_df["site_key"].isin(eligible_site_keys)].copy()
    dry_indices = []
    site_rows = []
    for site_key, site_df in eligible_df.groupby("site_key", dropna=False):
        site_df = site_df.sort_values(["obs", "date"], ascending=[True, True], kind="mergesort")
        dry_n = max(1, int(np.ceil(float(dry_quantile) * len(site_df))))
        dry_site_df = site_df.iloc[:dry_n].copy()
        dry_indices.extend(dry_site_df.index.tolist())
        all_metrics = compute_basic_metrics(site_df["obs"].values, site_df["pred"].values)
        dry_metrics = compute_basic_metrics(dry_site_df["obs"].values, dry_site_df["pred"].values)
        site_rows.append(
            {
                "model_key": model_key,
                "member": member_label,
                "site_key": str(site_key),
                "latitude": float(pd.to_numeric(site_df["latitude"], errors="coerce").iloc[0]),
                "longitude": float(pd.to_numeric(site_df["longitude"], errors="coerce").iloc[0]),
                "n_observations": int(len(site_df)),
                "n_dry_observations": int(len(dry_site_df)),
                "dry_quantile": float(dry_quantile),
                "dry_lfmc_threshold": float(dry_site_df["obs"].max()) if len(dry_site_df) > 0 else np.nan,
                "all_site_r2": all_metrics.get("r2", np.nan),
                "all_site_rmse": all_metrics.get("rmse", np.nan),
                "dry_site_r2": dry_metrics.get("r2", np.nan),
                "dry_site_rmse": dry_metrics.get("rmse", np.nan),
            }
        )
    dry_df = eligible_df.loc[dry_indices].copy() if len(dry_indices) > 0 else eligible_df.iloc[0:0].copy()
    site_metric_df = pd.DataFrame.from_records(site_rows)
    all_metrics = compute_basic_metrics(eligible_df["obs"].values, eligible_df["pred"].values)
    dry_metrics = compute_basic_metrics(dry_df["obs"].values, dry_df["pred"].values)
    summary_rows = []
    for subset_name, subset_df, metrics, site_r2_col, site_rmse_col in [
        ("all_lfmc_at_sites_with_min_obs", eligible_df, all_metrics, "all_site_r2", "all_site_rmse"),
        ("driest_within_site_lfmc", dry_df, dry_metrics, "dry_site_r2", "dry_site_rmse"),
    ]:
        finite_site_r2 = _finite_series(site_metric_df[site_r2_col])
        finite_site_rmse = _finite_series(site_metric_df[site_rmse_col])
        summary_rows.append(
            {
                "model_key": model_key,
                "member": member_label,
                "subset": subset_name,
                "min_obs_per_site": int(min_obs),
                "dry_quantile": float(dry_quantile),
                "n_sites": int(subset_df["site_key"].nunique()) if len(subset_df) > 0 else 0,
                "n_observations": int(len(subset_df)),
                "r2": metrics.get("r2", np.nan),
                "rmse": metrics.get("rmse", np.nan),
                "n_sites_with_finite_site_r2": int(len(finite_site_r2)),
                "mean_site_r2": float(finite_site_r2.mean()) if len(finite_site_r2) > 0 else np.nan,
                "median_site_r2": float(finite_site_r2.median()) if len(finite_site_r2) > 0 else np.nan,
                "mean_site_rmse": float(finite_site_rmse.mean()) if len(finite_site_rmse) > 0 else np.nan,
                "median_site_rmse": float(finite_site_rmse.median()) if len(finite_site_rmse) > 0 else np.nan,
            }
        )
    return pd.DataFrame.from_records(summary_rows), site_metric_df


def _build_dry_lfmc_performance_statistics(
    runtime: Dict[str, object],
    model_key: str,
    min_obs: int,
    dry_quantile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = _load_eval_context(runtime, model_key)
    overall_df, site_df = _dry_lfmc_performance_for_eval_df(
        context["eval_df"],
        model_key=model_key,
        member_label="ensemble_mean",
        min_obs=min_obs,
        dry_quantile=dry_quantile,
    )
    member_rows = []
    for member_idx, member_eval_df in enumerate(context["member_eval_dfs"], start=1):
        member_summary_df, _ = _dry_lfmc_performance_for_eval_df(
            member_eval_df,
            model_key=model_key,
            member_label=f"member_{member_idx}",
            min_obs=min_obs,
            dry_quantile=dry_quantile,
        )
        if len(member_summary_df) == 0:
            continue
        member_summary_df["member_index"] = int(member_idx)
        member_rows.append(member_summary_df)
    by_member_df = (
        pd.concat(member_rows, ignore_index=True, sort=False)
        if len(member_rows) > 0 else pd.DataFrame()
    )
    return overall_df, by_member_df, site_df


def _load_member_site_error_subset_worker(member_dir: str, site_keys: Sequence[str], queue) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            site_error = get_site_error(member_dir)
        site_key_set = {str(site_key) for site_key in site_keys}
        queue.put({site_key: site_error[site_key] for site_key in site_key_set if site_key in site_error})
    except Exception:
        traceback.print_exc()
        raise


def _load_ensemble_site_entry(
    cfg: Dict[str, object],
    model_key: str,
    site_keys: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    model_cfg = _model_cfg(cfg, model_key)
    member_dirs = _select_model_member_dirs(model_cfg)
    member_site_errors = {}
    member_site_error_list = []
    site_key_list = None if site_keys is None else [str(site_key) for site_key in site_keys]
    for member_idx, member_dir in enumerate(member_dirs, start=1):
        print(
            f"Loading site errors for {model_cfg['display_name']} member "
            f"{member_idx}/{len(member_dirs)}: {os.path.basename(member_dir)}"
        )
        if site_key_list is None:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                this_site_error = get_site_error(member_dir)
        else:
            ctx = mp.get_context("fork")
            queue = ctx.Queue(maxsize=1)
            proc = ctx.Process(
                target=_load_member_site_error_subset_worker,
                args=(member_dir, site_key_list, queue),
            )
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"Failed to load selected site errors for {os.path.basename(member_dir)}; "
                    f"child exit code {proc.exitcode}"
                )
            this_site_error = queue.get()
        member_site_errors[member_dir] = this_site_error
        member_site_error_list.append(this_site_error)
    site_error = aggregate_site_errors(member_site_error_list)
    return {
        "paper_model_key": model_key,
        "name": model_cfg["display_name"],
        "paper_color": model_cfg["color"],
        "outputs_root": model_cfg["outputs_root"],
        "input_data_name": model_cfg["input_data_name"],
        "model_type": "standard",
        "model_num_tasks": int(model_cfg.get("model_num_tasks", 3)),
        "model_dir": model_cfg["outputs_root"],
        "is_ensemble": True,
        "ensemble_member_name_prefix": model_cfg.get("ensemble_member_name_prefix"),
        "member_dirs": member_dirs,
        "member_site_errors": member_site_errors,
        "site_error": site_error,
    }


def _select_timeseries_sites(
    runtime: Dict[str, object],
    entry: Dict[str, object],
    fig_cfg: Dict[str, object],
    require_sar_overlap_same_year: bool = False,
    prefer_sar_observation_density: bool = False,
) -> Dict[str, List[str]]:
    site_df = build_site_df(entry["site_error"], entry["site_error"].keys())
    site_df["r2"] = site_df["site"].map(
        lambda site: float(entry["site_error"][site].get("r2", np.nan))
    )
    percentile_cfg = runtime["cfg"].get("timeseries_selection", {}).get("r2_percentiles", {})
    base_ranked = site_df[np.isfinite(site_df["r2"])].copy().reset_index(drop=True)
    if len(base_ranked) == 0:
        raise ValueError("No finite site R2 values were available for timeseries percentile selection")
    target_lookup = {
        "good": float(np.percentile(base_ranked["r2"].to_numpy(dtype=float), float(percentile_cfg.get("good", 95)))),
        "average": float(np.percentile(base_ranked["r2"].to_numpy(dtype=float), float(percentile_cfg.get("average", 50)))),
        "poor": float(np.percentile(base_ranked["r2"].to_numpy(dtype=float), float(percentile_cfg.get("poor", 5)))),
    }

    ranked = base_ranked.copy().reset_index(drop=True)
    if bool(fig_cfg.get("require_mountain_polygon", False)):
        ranked = ranked[
            ranked["site"].map(
                lambda site: _site_within_gmba_polygon(
                    site,
                    runtime["cfg"]["paths"]["gmba_basic_shapefile"],
                )
            )
        ].reset_index(drop=True)
    years_to_plot = int(fig_cfg.get("years_to_plot", 3))
    ranked["year_window_info"] = ranked["site"].map(
        lambda site: _get_site_year_window_info(
            runtime=runtime,
            model_entry=entry,
            site_key=site,
            years_to_plot=years_to_plot,
            prefer_sar_observation_density=prefer_sar_observation_density,
        )
    )
    ranked["lfmc_window_years_present"] = ranked["year_window_info"].map(
        lambda info: int(info.get("lfmc_years_present", 0))
    )
    ranked["sar_window_obs_count"] = ranked["year_window_info"].map(
        lambda info: int(info.get("sar_obs_count", 0))
    )
    ranked = ranked[ranked["lfmc_window_years_present"] >= 2].reset_index(drop=True)
    if require_sar_overlap_same_year:
        ranked = ranked[ranked["sar_window_obs_count"] > 0].reset_index(drop=True)
    ranked = ranked[ranked["num_measurements"] >= int(fig_cfg["min_measurements"])]
    if len(ranked) == 0:
        if require_sar_overlap_same_year:
            raise ValueError("No eligible sites with same-year LFMC/SAR overlap were found")
        raise ValueError("No eligible sites were found for timeseries selection")
    ranked = ranked.sort_values("r2", ascending=False).reset_index(drop=True)
    used_sites = set()
    selected = {
        "good": _pick_sites_closest_to_target_value(
            ranked,
            metric_col="r2",
            target_value=target_lookup["good"],
            n_sites=int(fig_cfg["num_sites_per_criterion"]),
            used_sites=used_sites,
        ),
        "average": _pick_sites_closest_to_target_value(
            ranked,
            metric_col="r2",
            target_value=target_lookup["average"],
            n_sites=int(fig_cfg["num_sites_per_criterion"]) + 1,
            used_sites=used_sites,
        ),
        "poor": _pick_sites_closest_to_target_value(
            ranked,
            metric_col="r2",
            target_value=target_lookup["poor"],
            n_sites=int(fig_cfg["num_sites_per_criterion"]),
            used_sites=used_sites,
        ),
    }
    if len(selected["average"]) > int(fig_cfg["num_sites_per_criterion"]):
        selected["average"] = selected["average"][1 : 1 + int(fig_cfg["num_sites_per_criterion"])]
    if sum(len(site_list) for site_list in selected.values()) == 0:
        raise ValueError("No sites were selected after applying timeseries filters")
    return selected


def _select_medium_sites_by_landcover(
    runtime: Dict[str, object],
    site_df: pd.DataFrame,
    n_sites: int,
    min_measurements: int,
    metric_col: str = "r2",
    landcover_order: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    categories = list(landcover_order or runtime["cfg"]["filters"]["landcover_order"])
    out = {category: [] for category in categories}
    ranked = site_df.copy()
    ranked = ranked[ranked["num_measurements"] >= min_measurements]
    ranked = ranked[np.isfinite(ranked[metric_col])]
    if len(ranked) == 0:
        return out
    def _normalize_landcover(site: str):
        annotation = get_site_landcover_annotation(site)
        if annotation is None:
            return np.nan
        normalized = str(annotation).replace("Land cover: ", "").strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        return normalized

    ranked["dominant_landcover"] = ranked["site"].map(_normalize_landcover)
    ranked = ranked[ranked["dominant_landcover"].isin(categories)].reset_index(drop=True)
    used_sites = set()
    for category in categories:
        category_ranked = ranked[ranked["dominant_landcover"] == category].copy()
        out[category] = _pick_percentile_sites(
            category_ranked,
            metric_col=metric_col,
            target_percentile=50.0,
            n_sites=n_sites,
            used_sites=used_sites,
        )
    return out


def _select_timeseries_sites_by_landcover(
    runtime: Dict[str, object],
    model_entries: Sequence[Dict[str, object]],
    anchor_model_key: str,
    num_sites_per_landcover: int,
    min_measurements: int,
    years_to_plot: int = 3,
    landcover_order: Optional[Sequence[str]] = None,
) -> tuple[Dict[str, object], Dict[str, List[str]]]:
    site_sets = [set(entry["site_error"].keys()) for entry in model_entries]
    common_sites = set.intersection(*site_sets) if len(site_sets) > 1 else site_sets[0]
    anchor_entry = next(
        (entry for entry in model_entries if entry["paper_model_key"] == anchor_model_key),
        None,
    )
    if anchor_entry is None:
        raise KeyError(f"Anchor model '{anchor_model_key}' not found in timeseries entries")
    site_df = build_site_df(anchor_entry["site_error"], common_sites)
    site_df["r2"] = site_df["site"].map(
        lambda site: float(anchor_entry["site_error"][site].get("r2", np.nan))
    )
    keep_sites = []
    for site in site_df["site"].tolist():
        lfmc_dates = pd.to_datetime(anchor_entry["site_error"][site]["dates"], errors="coerce")
        lfmc_dates = lfmc_dates[lfmc_dates.notna()]
        selected_years = _top_consecutive_observation_years(lfmc_dates, years_to_plot)
        observed_years = set(lfmc_dates.year.tolist())
        n_years_present = sum(1 for year in selected_years if year in observed_years)
        if n_years_present >= 2:
            keep_sites.append(site)
    site_df = site_df[site_df["site"].isin(keep_sites)].reset_index(drop=True)
    if len(site_df) == 0:
        raise ValueError("No eligible sites were found for landcover-based timeseries selection")
    selected = _select_medium_sites_by_landcover(
        runtime=runtime,
        site_df=site_df,
        n_sites=num_sites_per_landcover,
        min_measurements=min_measurements,
        metric_col="r2",
        landcover_order=landcover_order,
    )
    if sum(len(site_list) for site_list in selected.values()) == 0:
        raise ValueError("No sites were selected after applying landcover and measurement filters")
    return anchor_entry, selected


def _get_site_vv_vh_obs(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
) -> Optional[Dict[str, object]]:
    cache = runtime.setdefault("site_vv_vh_obs_cache", {})
    model_cache_key = str(
        model_entry.get("paper_model_key", model_entry.get("outputs_root", model_entry.get("name", "model")))
    )
    cache_key = (model_cache_key, site_key)
    if cache_key not in cache:
        cache[cache_key] = get_vv_vh_site_series(
            model_entry,
            site_key,
            runtime["vhvv_fold_cache"],
            start_date=None,
            end_date=None,
        )
    return cache[cache_key]


def _get_site_year_window_info(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    years_to_plot: int,
    prefer_sar_observation_density: bool = False,
) -> Dict[str, object]:
    cache = runtime.setdefault("timeseries_year_window_cache", {})
    model_cache_key = str(
        model_entry.get("paper_model_key", model_entry.get("outputs_root", model_entry.get("name", "model")))
    )
    cache_key = (model_cache_key, site_key, int(years_to_plot), bool(prefer_sar_observation_density))
    if cache_key in cache:
        return dict(cache[cache_key])
    anchor_site = model_entry["site_error"][site_key]
    lfmc_dates = _to_naive_datetime(anchor_site["dates"])
    vv_vh_obs = None
    if bool(prefer_sar_observation_density) and int(model_entry.get("model_num_tasks", 0)) >= 3:
        vv_vh_obs = _get_site_vv_vh_obs(runtime, model_entry, site_key)
    out = _select_consecutive_timeseries_years(
        lfmc_dates=lfmc_dates,
        n_years=int(years_to_plot),
        vv_vh_obs=vv_vh_obs,
        prefer_sar_observation_density=bool(prefer_sar_observation_density),
    )
    cache[cache_key] = dict(out)
    return dict(out)


def _site_has_same_year_lfmc_sar_overlap(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    years_to_plot: int,
) -> bool:
    if int(model_entry.get("model_num_tasks", 0)) < 3:
        return False
    year_info = _get_site_year_window_info(
        runtime=runtime,
        model_entry=model_entry,
        site_key=site_key,
        years_to_plot=int(years_to_plot),
        prefer_sar_observation_density=True,
    )
    return int(year_info.get("sar_obs_count", 0)) > 0


def _normalize_inference_output(out: Dict[str, object]) -> Dict[str, np.ndarray]:
    return {
        "dates": np.asarray(pd.to_datetime(out.get("dates", []), errors="coerce"), dtype="datetime64[ns]"),
        "lfmc_pred": np.asarray(out.get("lfmc_pred", []), dtype=float),
        "lfmc_pred_std": np.asarray(out.get("lfmc_pred_std", []), dtype=float),
        "vv_pred": np.asarray(out.get("vv_pred", []), dtype=float),
        "vv_pred_std": np.asarray(out.get("vv_pred_std", []), dtype=float),
        "vh_pred": np.asarray(out.get("vh_pred", []), dtype=float),
        "vh_pred_std": np.asarray(out.get("vh_pred_std", []), dtype=float),
    }


def _timeseries_inference_cache_path(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> str:
    model_key = model_entry.get("paper_model_key")
    if model_key not in {None, ""}:
        base_cache_dir = _model_cache_dir(runtime, str(model_key))
    else:
        base_cache_dir = runtime["cache_dir"]
    cache_dir = os.path.join(base_cache_dir, "timeseries_inference")
    os.makedirs(cache_dir, exist_ok=True)
    cache_meta = {
        "site_key": str(site_key),
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "outputs_root": str(model_entry.get("outputs_root", model_entry.get("model_dir", ""))),
        "paper_model_key": str(model_entry.get("paper_model_key", "")),
        "model_num_tasks": int(model_entry.get("model_num_tasks", 0)),
        "is_ensemble": bool(model_entry.get("is_ensemble", False)),
    }
    cache_hash = hashlib.md5(
        json.dumps(cache_meta, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return os.path.join(cache_dir, f"{cache_hash}.npz")


def _load_cached_timeseries_inference(cache_path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(cache_path):
        return None
    with np.load(cache_path, allow_pickle=False) as npz:
        return {
            "dates": np.asarray(npz["dates"], dtype="datetime64[ns]"),
            "lfmc_pred": np.asarray(npz["lfmc_pred"], dtype=float),
            "lfmc_pred_std": np.asarray(npz["lfmc_pred_std"], dtype=float),
            "vv_pred": np.asarray(npz["vv_pred"], dtype=float),
            "vv_pred_std": np.asarray(npz["vv_pred_std"], dtype=float),
            "vh_pred": np.asarray(npz["vh_pred"], dtype=float),
            "vh_pred_std": np.asarray(npz["vh_pred_std"], dtype=float),
        }


def _write_cached_timeseries_inference(
    cache_path: str,
    normalized_out: Dict[str, np.ndarray],
) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        dates=np.asarray(normalized_out["dates"], dtype="datetime64[ns]"),
        lfmc_pred=np.asarray(normalized_out["lfmc_pred"], dtype=float),
        lfmc_pred_std=np.asarray(normalized_out["lfmc_pred_std"], dtype=float),
        vv_pred=np.asarray(normalized_out["vv_pred"], dtype=float),
        vv_pred_std=np.asarray(normalized_out["vv_pred_std"], dtype=float),
        vh_pred=np.asarray(normalized_out["vh_pred"], dtype=float),
        vh_pred_std=np.asarray(normalized_out["vh_pred_std"], dtype=float),
    )


def _concat_inference_parts(parts: Sequence[Dict[str, np.ndarray]], key: str, dtype) -> np.ndarray:
    arrays = [part[key] for part in parts if len(part[key]) > 0]
    if len(arrays) == 0:
        return np.array([], dtype=dtype)
    return np.concatenate(arrays)


def _collect_year_window_inference(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    selected_years: Sequence[int],
) -> Dict[str, np.ndarray]:
    parts = []
    inputs_root = runtime["cfg"]["paths"]["inputs_root"]
    forward_batch_size = int(runtime["cfg"]["plotting"].get("forward_batch_size", 4096))
    for year in selected_years:
        start_date = pd.Timestamp(int(year), 1, 1)
        end_date = pd.Timestamp(int(year), 12, 31)
        cache_path = _timeseries_inference_cache_path(
            runtime=runtime,
            model_entry=model_entry,
            site_key=site_key,
            start_date=start_date,
            end_date=end_date,
        )
        normalized_out = _load_cached_timeseries_inference(cache_path)
        if normalized_out is None:
            out = get_model_inference_series(
                model_entry,
                site_key,
                start_date,
                end_date,
                runtime["inference_cache"],
                runtime["tensor_cache"],
                runtime["runtime_cache"],
                inputs_root,
                forward_batch_size,
            )
            normalized_out = _normalize_inference_output(out)
            _write_cached_timeseries_inference(cache_path, normalized_out)
        parts.append(normalized_out)
    if len(parts) == 0:
        return {
            "dates": np.array([], dtype="datetime64[ns]"),
            "lfmc_pred": np.array([], dtype=float),
            "lfmc_pred_std": np.array([], dtype=float),
            "vv_pred": np.array([], dtype=float),
            "vv_pred_std": np.array([], dtype=float),
            "vh_pred": np.array([], dtype=float),
            "vh_pred_std": np.array([], dtype=float),
        }
    return {
        "dates": _concat_inference_parts(parts, "dates", "datetime64[ns]"),
        "lfmc_pred": _concat_inference_parts(parts, "lfmc_pred", float),
        "lfmc_pred_std": _concat_inference_parts(parts, "lfmc_pred_std", float),
        "vv_pred": _concat_inference_parts(parts, "vv_pred", float),
        "vv_pred_std": _concat_inference_parts(parts, "vv_pred_std", float),
        "vh_pred": _concat_inference_parts(parts, "vh_pred", float),
        "vh_pred_std": _concat_inference_parts(parts, "vh_pred_std", float),
    }


def _build_timeseries_panel(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    criterion_label: str,
    years_to_plot: int,
    plot_vv: bool,
    plot_vh: bool,
    prefer_sar_observation_density: bool = False,
    prediction_label: Optional[str] = None,
    include_sar_r2_in_title: bool = False,
) -> Dict[str, object]:
    cfg = runtime["cfg"]
    anchor_site = model_entry["site_error"][site_key]
    lfmc_dates = _to_naive_datetime(anchor_site["dates"])
    lfmc_values = np.asarray(anchor_site["true_values"], dtype=float)
    year_window_info = _get_site_year_window_info(
        runtime=runtime,
        model_entry=model_entry,
        site_key=site_key,
        years_to_plot=int(years_to_plot),
        prefer_sar_observation_density=prefer_sar_observation_density,
    )
    selected_years = list(year_window_info.get("selected_years", []))
    lfmc_dates, lfmc_values = _filter_series_to_years(lfmc_dates, lfmc_values, selected_years)
    lfmc_dates = _canonicalize_dates_to_year_slots(lfmc_dates, selected_years)
    lfmc_dates, lfmc_values, _, _ = _reindex_series_to_daily(lfmc_dates, lfmc_values)

    infer_out = _collect_year_window_inference(runtime, model_entry, site_key, selected_years)
    infer_dates = _canonicalize_dates_to_year_slots(infer_out["dates"], selected_years)
    infer_lower = None
    infer_upper = None
    if len(infer_out["lfmc_pred_std"]) == len(infer_out["lfmc_pred"]):
        infer_lower = infer_out["lfmc_pred"] - infer_out["lfmc_pred_std"]
        infer_upper = infer_out["lfmc_pred"] + infer_out["lfmc_pred_std"]
    infer_dates, infer_values, infer_lower, infer_upper = _reindex_series_to_daily(
        infer_dates,
        infer_out["lfmc_pred"],
        lower=infer_lower,
        upper=infer_upper,
    )

    site_r2 = float(anchor_site.get("r2", np.nan))
    site_lat, site_lon = _parse_site_lat_lon(site_key)
    extra_r2_parts = []
    panel = {
        "title": _timeseries_panel_title(
            cfg=cfg,
            site_key=site_key,
            criterion_label=criterion_label,
            selected_years=selected_years,
            site_r2=site_r2,
            extra_r2_parts=extra_r2_parts,
        ),
        "site_latitude": float(site_lat),
        "site_longitude": float(site_lon),
        "series": [
            {
                "label": prediction_label or model_entry["name"],
                "dates": infer_dates,
                "values": infer_values,
                "lower": infer_lower,
                "upper": infer_upper,
                "color": model_entry["paper_color"],
                "linewidth": 2.2,
                "linestyle": "-",
                "alpha": 0.95,
                "legend_group": "predictions",
                "axis_group": "lfmc",
            },
            {
                "label": "Observed LFMC",
                "dates": lfmc_dates,
                "values": lfmc_values,
                "color": "#111111",
                "linestyle": "",
                "marker": "o",
                "markersize": 5,
                "alpha": 0.9,
                "linewidth": 0.0,
                "zorder": 4,
                "legend_group": "observations",
                "axis_group": "lfmc",
            },
        ],
        "right_series": [],
        "ylabel": "LFMC (%)",
        "use_month_aligned_axis": True,
        "timeseries_mode": "lfmc_only",
    }

    if int(model_entry.get("model_num_tasks", 0)) < 3 or (not plot_vv and not plot_vh):
        return panel

    vv_vh_obs = get_vv_vh_site_series(
        model_entry,
        site_key,
        runtime["vhvv_fold_cache"],
        start_date=None,
        end_date=None,
    )
    if vv_vh_obs is None:
        return panel

    if plot_vv:
        vv_dates, vv_obs_vals = _filter_series_to_years(
            vv_vh_obs["vv_dates"],
            vv_vh_obs["vv_true"],
            selected_years,
        )
        vv_dates = _canonicalize_dates_to_year_slots(vv_dates, selected_years)
        vv_r2 = np.nan
        if include_sar_r2_in_title and len(infer_out["vv_pred"]) > 0:
            vv_infer_dates = _canonicalize_dates_to_year_slots(infer_out["dates"], selected_years)
            vv_r2 = _date_matched_r2(vv_dates, vv_obs_vals, vv_infer_dates, infer_out["vv_pred"])
        if include_sar_r2_in_title:
            extra_r2_parts.append(f"VV R²: {vv_r2:.2f}" if np.isfinite(vv_r2) else "VV R²: nan")
        panel["right_series"].append(
            {
                "label": "Observed VV",
                "dates": vv_dates,
                "values": vv_obs_vals,
                "color": "#f4a259",
                "linestyle": "",
                "marker": "s",
                "markersize": 3.5,
                "alpha": 0.85,
                "linewidth": 0.0,
                "markerfacecolor": "#f4a259",
                "markeredgecolor": "#9c4f15",
                "legend_group": "observations",
                "axis_group": "vv",
            }
        )
        vv_dates_daily, vv_vals_daily, _, _ = _reindex_series_to_daily(
            panel["right_series"][-1]["dates"],
            panel["right_series"][-1]["values"],
        )
        panel["right_series"][-1]["dates"] = vv_dates_daily
        panel["right_series"][-1]["values"] = vv_vals_daily
        if len(infer_out["vv_pred"]) > 0:
            vv_infer_dates = _canonicalize_dates_to_year_slots(infer_out["dates"], selected_years)
            panel["right_series"].append(
                {
                    "label": "Predicted VV",
                    "dates": vv_infer_dates,
                    "values": infer_out["vv_pred"],
                    "lower": (
                        infer_out["vv_pred"] - infer_out["vv_pred_std"]
                        if len(infer_out["vv_pred_std"]) == len(infer_out["vv_pred"])
                        else None
                    ),
                    "upper": (
                        infer_out["vv_pred"] + infer_out["vv_pred_std"]
                        if len(infer_out["vv_pred_std"]) == len(infer_out["vv_pred"])
                        else None
                    ),
                    "color": "#d95f02",
                    "linestyle": "-",
                    "linewidth": 1.45,
                    "alpha": 0.95,
                    "legend_group": "predictions",
                    "axis_group": "vv",
                }
            )
            vv_dates_daily, vv_vals_daily, vv_lower_daily, vv_upper_daily = _reindex_series_to_daily(
                panel["right_series"][-1]["dates"],
                panel["right_series"][-1]["values"],
                lower=panel["right_series"][-1]["lower"],
                upper=panel["right_series"][-1]["upper"],
            )
            panel["right_series"][-1]["dates"] = vv_dates_daily
            panel["right_series"][-1]["values"] = vv_vals_daily
            panel["right_series"][-1]["lower"] = vv_lower_daily
            panel["right_series"][-1]["upper"] = vv_upper_daily

    if plot_vh:
        vh_dates, vh_obs_vals = _filter_series_to_years(
            vv_vh_obs["vh_dates"],
            vv_vh_obs["vh_true"],
            selected_years,
        )
        vh_dates = _canonicalize_dates_to_year_slots(vh_dates, selected_years)
        vh_r2 = np.nan
        if include_sar_r2_in_title and len(infer_out["vh_pred"]) > 0:
            vh_infer_dates = _canonicalize_dates_to_year_slots(infer_out["dates"], selected_years)
            vh_r2 = _date_matched_r2(vh_dates, vh_obs_vals, vh_infer_dates, infer_out["vh_pred"])
        if include_sar_r2_in_title:
            extra_r2_parts.append(f"VH R²: {vh_r2:.2f}" if np.isfinite(vh_r2) else "VH R²: nan")
        panel["right_series"].append(
            {
                "label": "Observed VH",
                "dates": vh_dates,
                "values": vh_obs_vals,
                "color": "#ffd166",
                "linestyle": "",
                "marker": "D",
                "markersize": 3.3,
                "alpha": 0.85,
                "linewidth": 0.0,
                "markerfacecolor": "#ffd166",
                "markeredgecolor": "#9b7a00",
                "legend_group": "observations",
                "axis_group": "vh",
            }
        )
        vh_dates_daily, vh_vals_daily, _, _ = _reindex_series_to_daily(
            panel["right_series"][-1]["dates"],
            panel["right_series"][-1]["values"],
        )
        panel["right_series"][-1]["dates"] = vh_dates_daily
        panel["right_series"][-1]["values"] = vh_vals_daily
        if len(infer_out["vh_pred"]) > 0:
            vh_infer_dates = _canonicalize_dates_to_year_slots(infer_out["dates"], selected_years)
            panel["right_series"].append(
                {
                    "label": "Predicted VH",
                    "dates": vh_infer_dates,
                    "values": infer_out["vh_pred"],
                    "lower": (
                        infer_out["vh_pred"] - infer_out["vh_pred_std"]
                        if len(infer_out["vh_pred_std"]) == len(infer_out["vh_pred"])
                        else None
                    ),
                    "upper": (
                        infer_out["vh_pred"] + infer_out["vh_pred_std"]
                        if len(infer_out["vh_pred_std"]) == len(infer_out["vh_pred"])
                        else None
                    ),
                    "color": "#e6ab02",
                    "linestyle": "-",
                    "linewidth": 1.45,
                    "alpha": 0.95,
                    "legend_group": "predictions",
                    "axis_group": "vh",
                }
            )
            vh_dates_daily, vh_vals_daily, vh_lower_daily, vh_upper_daily = _reindex_series_to_daily(
                panel["right_series"][-1]["dates"],
                panel["right_series"][-1]["values"],
                lower=panel["right_series"][-1]["lower"],
                upper=panel["right_series"][-1]["upper"],
            )
            panel["right_series"][-1]["dates"] = vh_dates_daily
            panel["right_series"][-1]["values"] = vh_vals_daily
            panel["right_series"][-1]["lower"] = vh_lower_daily
            panel["right_series"][-1]["upper"] = vh_upper_daily

    if len(panel["right_series"]) > 0:
        panel["right_ylabel"] = "VV / VH (dB)"
        panel["timeseries_mode"] = "banded_sar"
    if include_sar_r2_in_title and len(extra_r2_parts) > 0:
        panel["title"] = _timeseries_panel_title(
            cfg=cfg,
            site_key=site_key,
            criterion_label=criterion_label,
            selected_years=selected_years,
            site_r2=site_r2,
            extra_r2_parts=extra_r2_parts,
        )
    return panel


def _build_timeseries_comparison_panel(
    runtime: Dict[str, object],
    model_entries: Sequence[Dict[str, object]],
    anchor_entry: Dict[str, object],
    site_key: str,
    criterion_label: str,
    years_to_plot: int,
) -> Dict[str, object]:
    cfg = runtime["cfg"]
    anchor_site = anchor_entry["site_error"][site_key]
    lfmc_dates = _to_naive_datetime(anchor_site["dates"])
    lfmc_values = np.asarray(anchor_site["true_values"], dtype=float)
    selected_years = _top_consecutive_observation_years(lfmc_dates, years_to_plot)
    lfmc_dates, lfmc_values = _filter_series_to_years(lfmc_dates, lfmc_values, selected_years)
    lfmc_dates = _canonicalize_dates_to_year_slots(lfmc_dates, selected_years)
    lfmc_dates, lfmc_values, _, _ = _reindex_series_to_daily(lfmc_dates, lfmc_values)

    prediction_series = []
    for model_entry in model_entries:
        infer_out = _collect_year_window_inference(runtime, model_entry, site_key, selected_years)
        infer_dates = _canonicalize_dates_to_year_slots(infer_out["dates"], selected_years)
        infer_lower = None
        infer_upper = None
        if len(infer_out["lfmc_pred_std"]) == len(infer_out["lfmc_pred"]):
            infer_lower = infer_out["lfmc_pred"] - infer_out["lfmc_pred_std"]
            infer_upper = infer_out["lfmc_pred"] + infer_out["lfmc_pred_std"]
        infer_dates, infer_values, infer_lower, infer_upper = _reindex_series_to_daily(
            infer_dates,
            infer_out["lfmc_pred"],
            lower=infer_lower,
            upper=infer_upper,
        )
        prediction_series.append(
            {
                "label": model_entry["name"],
                "dates": infer_dates,
                "values": infer_values,
                "lower": infer_lower,
                "upper": infer_upper,
                "color": model_entry["paper_color"],
                "linewidth": 2.2,
                "linestyle": "-",
                "alpha": 0.95,
                "legend_group": "predictions",
                "axis_group": "lfmc",
            }
        )

    site_r2 = float(anchor_site.get("r2", np.nan))
    return {
        "title": _timeseries_panel_title(
            cfg=cfg,
            site_key=site_key,
            criterion_label=criterion_label,
            selected_years=selected_years,
            site_r2=site_r2,
        ),
        "series": prediction_series
        + [
            {
                "label": "Observed LFMC",
                "dates": lfmc_dates,
                "values": lfmc_values,
                "color": "#111111",
                "linestyle": "",
                "marker": "o",
                "markersize": 5,
                "alpha": 0.9,
                "linewidth": 0.0,
                "zorder": 4,
                "legend_group": "observations",
                "axis_group": "lfmc",
            }
        ],
        "right_series": [],
        "ylabel": "LFMC (%)",
        "use_month_aligned_axis": True,
        "timeseries_mode": "lfmc_only",
    }


def init_runtime(cfg: Dict[str, object]) -> Dict[str, object]:
    output_root = str(cfg["paths"]["output_root"])
    figures_dir = os.path.join(output_root, "figures")
    tables_dir = os.path.join(output_root, "tables")
    cache_dir = os.path.join(output_root, "cache")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    return {
        "cfg": cfg,
        "output_root": output_root,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
        "cache_dir": cache_dir,
        "inference_cache": {},
        "tensor_cache": {},
        "runtime_cache": {},
        "vhvv_fold_cache": {},
        "site_vv_vh_obs_cache": {},
        "timeseries_year_window_cache": {},
        "eval_contexts": {},
        "landcover_metric_tables": {},
        "site_r2_landcover_tables": {},
        "train_info_fold_cache": {},
        "train_info_union_cache": {},
        "map_dataset_cache": {},
    }


def _copy_manual_asset_figure(runtime: Dict[str, object], figure_key: str) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"][figure_key]
    source_path = fig_cfg.get("source_path")
    if source_path in {None, ""}:
        raise ValueError(
            f"{figure_key} is configured as a manual asset figure. "
            f"Set figures.{figure_key}.source_path in paper_figure_configs_new.yaml."
        )
    source_path = os.path.abspath(str(source_path))
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"{figure_key} source asset does not exist: {source_path}")
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.abspath(source_path) != os.path.abspath(save_path):
        shutil.copy2(source_path, save_path)
    return save_path


def build_figure_1(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_1"]
    if fig_cfg.get("source_path") not in {None, ""}:
        return _copy_manual_asset_figure(runtime, "figure_1")
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    plot_placeholder_figure(
        title=str(fig_cfg.get("title", "Figure 1 Placeholder")),
        description=str(fig_cfg.get("description", "")),
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
    )
    return save_path


def build_figure_2(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_2"]
    ds = _load_operational_lfmc_map_dataset(runtime)
    map_var = str(fig_cfg.get("map_variable", "lfmc_ens_mean"))
    requested_dates = list(fig_cfg["map_dates"])
    panel_labels = list(fig_cfg.get("panel_labels", ["a", "b", "c", "d"]))
    panels = []
    table_rows = []
    for panel_idx, requested_date in enumerate(requested_dates):
        requested_ts = pd.Timestamp(requested_date).normalize()
        da = ds[map_var].sel(time=requested_ts).load()
        actual_ts = pd.Timestamp(da["time"].values).normalize()
        values = np.asarray(da.values, dtype=float)
        panels.append(
            {
                "panel_label": panel_labels[panel_idx],
                "title": (
                    f"{panel_labels[panel_idx]}) "
                    f"{actual_ts.strftime('%B %d, %Y').replace(' 0', ' ')}"
                ),
                "values": values,
                "x": np.asarray(da["x"].values, dtype=float),
                "y": np.asarray(da["y"].values, dtype=float),
            }
        )
        table_rows.append(
            {
                "panel": panel_labels[panel_idx],
                "requested_date": requested_ts.date().isoformat(),
                "selected_date": actual_ts.date().isoformat(),
                "n_finite": int(np.isfinite(values).sum()),
                "min_value": float(np.nanmin(values)),
                "max_value": float(np.nanmax(values)),
            }
        )
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "figure_02_lfmc_snapshot_dates")
    pd.DataFrame.from_records(table_rows).to_csv(table_path, index=False)
    plot_lfmc_snapshot_quadrants(
        panels=panels,
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        vmin=fig_cfg.get("vmin"),
        vmax=fig_cfg.get("vmax"),
        col_labels=fig_cfg.get("col_labels"),
        row_labels=fig_cfg.get("row_labels"),
        state_lines_only=bool(fig_cfg.get("state_lines_only", False)),
        subplot_wspace=float(fig_cfg.get("subplot_wspace", -0.15)),
        subplot_hspace=float(fig_cfg.get("subplot_hspace", 0.04)),
    )
    return save_path


def build_figure_3(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_3"]
    figure_fontsize = int(round(
        int(cfg["plotting"].get("fontsize", 14)) * float(fig_cfg.get("text_scale", 1.0))
    ))
    lfmc_df = _build_test_location_count_table(
        runtime,
        model_key=str(fig_cfg["lfmc_model_key"]),
        target_names=["lfmc"],
        representative_member_index=int(fig_cfg.get("representative_member_index", 0)),
    )
    lfmc_location_keys = {
        (float(row["latitude"]), float(row["longitude"]))
        for _, row in lfmc_df[["latitude", "longitude"]].iterrows()
    }
    print(f"  LFMC location keys for cross-reference: {len(lfmc_location_keys)}")
    sar_df = _build_test_location_count_table(
        runtime,
        model_key=str(fig_cfg["sar_model_key"]),
        target_names=["vv", "vh"],
        representative_member_index=int(fig_cfg.get("representative_member_index", 0)),
        include_reference_member_flag=True,
    )
    for grp, cnt in sar_df["marker_group"].value_counts().items():
        print(f"  SAR marker group '{grp}': {cnt} locations")
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    lfmc_table_path = _table_output_path(runtime, "figure_03_lfmc_training_locations")
    sar_table_path = _table_output_path(runtime, "figure_03_vv_vh_training_locations")
    lfmc_df.to_csv(lfmc_table_path, index=False)
    sar_df.to_csv(sar_table_path, index=False)
    plot_training_location_maps(
        panels=[
            {
                "panel_label": "a",
                "title": "LFMC sampling sites",
                "title_fontweight": "normal",
                "map_df": lfmc_df,
                "marker_defs": {
                    "all_points": {"marker": "o", "label": "LFMC training locations"},
                },
                "cmap": str(fig_cfg.get("lfmc_cmap", "viridis")),
                "cbar_label": "LFMC measurements at site",
                "cbar_pad": 0.06,
                "stats_total_label": "Total measurements",
                "stats_mean_label": "Mean measurements/site",
                "stats_x": 0.50,
                "stats_y": -0.045,
                "stats_ha": "center",
                "stats_va": "top",
                "marker_size": float(fig_cfg.get("marker_size", 34.0)),
            },
            {
                "panel_label": "b",
                "title": "Sentinel-1 sampling sites",
                "title_fontweight": "normal",
                "map_df": sar_df,
                "marker_defs": {
                    "member_1": {
                        "marker": "o",
                        "label": "ensemble member 1 sites",
                    },
                    "other_members": {
                        "marker": "o",
                        "label": "all other ensemble member sites",
                        "size": 10.0,
                        "alpha": 0.18,
                        "edgecolor": "none",
                        "linewidth": 0.0,
                        "zorder_offset": -0.4,
                    },
                },
                "cmap": str(fig_cfg.get("sar_cmap", "cividis")),
                "cbar_label": "SAR observations at site",
                "cbar_pad": 0.06,
                "stats_total_label": "Total observations",
                "stats_mean_label": "Mean observations/site",
                "stats_x": 0.50,
                "stats_y": -0.045,
                "stats_ha": "center",
                "stats_va": "top",
                "marker_size": float(fig_cfg.get("marker_size", 34.0)),
                "legend_loc": "upper center",
                "legend_bbox_to_anchor": (0.50, -0.31),
            },
        ],
        save_path=save_path,
        fontsize=figure_fontsize,
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        state_lines_only=True,
    )
    return save_path


def build_figure_4(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_4"]
    site_min_obs = int(cfg["variability"]["site_min_obs"])
    monthly_min_obs = int(cfg["variability"]["monthly_min_obs"])
    monthly_min_years = int(cfg["variability"]["monthly_min_years"])
    lfmc_df = _load_ensemble_eval_df_for_targets_streaming(
        runtime,
        str(fig_cfg["model_key"]),
        target_names=["lfmc"],
    ).reset_index(drop=True)
    lfmc_y2y_df = build_lfmc_y2y_df(lfmc_df)
    _, site_summary_df, anomaly_df = _build_filtered_site_space_time_tables(
        lfmc_df,
        min_obs=site_min_obs,
    )
    _, month_anom_df, valid_month_groups = _build_filtered_site_month_anomaly_tables(
        lfmc_df=lfmc_y2y_df,
        min_obs=monthly_min_obs,
        min_years=monthly_min_years,
    )
    if len(month_anom_df) == 0 or len(valid_month_groups) == 0:
        raise ValueError("No source-centered monthly anomaly rows available for Figure 4")
    def mean_bias(obs_values, pred_values) -> float:
        obs_arr = np.asarray(obs_values, dtype=float)
        pred_arr = np.asarray(pred_values, dtype=float)
        mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
        if not np.any(mask):
            return float("nan")
        return float(np.mean(pred_arr[mask] - obs_arr[mask]))

    def dry_diagnostics(obs_values, pred_values, worst_error_fraction: float) -> Dict[str, float]:
        obs_arr = np.asarray(obs_values, dtype=float)
        pred_arr = np.asarray(pred_values, dtype=float)
        mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
        obs_arr = obs_arr[mask]
        pred_arr = pred_arr[mask]
        if obs_arr.size == 0:
            return {
                "n_observations": 0,
                "pearson_r": np.nan,
                "full_r2": np.nan,
                "bias": np.nan,
                "r2_bias_removed": np.nan,
                "worst_error_fraction": float(worst_error_fraction),
                "worst_error_n_removed": 0,
                "worst_error_sse_share": np.nan,
                "r2_without_worst_error_fraction": np.nan,
            }
        residual = pred_arr - obs_arr
        bias = float(np.mean(residual))
        full_metrics = compute_basic_metrics(obs_arr, pred_arr)
        bias_removed_metrics = compute_basic_metrics(obs_arr, pred_arr - bias)
        if obs_arr.size > 1 and np.std(obs_arr) > 0.0 and np.std(pred_arr) > 0.0:
            pearson_r = float(np.corrcoef(obs_arr, pred_arr)[0, 1])
        else:
            pearson_r = np.nan
        squared_error = residual ** 2
        worst_n = max(1, int(round(obs_arr.size * float(worst_error_fraction))))
        worst_n = min(worst_n, int(obs_arr.size))
        worst_order = np.argsort(squared_error)[::-1]
        worst_idx = worst_order[:worst_n]
        keep_mask = np.ones(obs_arr.size, dtype=bool)
        keep_mask[worst_idx] = False
        if np.any(keep_mask):
            trimmed_metrics = compute_basic_metrics(obs_arr[keep_mask], pred_arr[keep_mask])
        else:
            trimmed_metrics = {"r2": np.nan}
        total_sse = float(np.sum(squared_error))
        if total_sse > 0.0:
            worst_sse_share = float(np.sum(squared_error[worst_idx]) / total_sse)
        else:
            worst_sse_share = np.nan
        return {
            "n_observations": int(obs_arr.size),
            "pearson_r": pearson_r,
            "full_r2": full_metrics.get("r2", np.nan),
            "bias": bias,
            "r2_bias_removed": bias_removed_metrics.get("r2", np.nan),
            "worst_error_fraction": float(worst_error_fraction),
            "worst_error_n_removed": int(worst_n),
            "worst_error_sse_share": worst_sse_share,
            "r2_without_worst_error_fraction": trimmed_metrics.get("r2", np.nan),
        }

    dry_site_min_obs = int(fig_cfg.get("dry_site_min_obs", 10))
    dry_quantile = float(fig_cfg.get("dry_site_quantile", 0.20))
    dry_worst_error_fraction = float(fig_cfg.get("dry_worst_error_fraction", 0.10))
    dry_eligible_df = lfmc_y2y_df.copy()
    dry_eligible_df["site_key"] = dry_eligible_df["site_key"].astype(str)
    site_counts = dry_eligible_df.groupby("site_key", dropna=False).size()
    eligible_site_keys = set(site_counts[site_counts >= dry_site_min_obs].index.astype(str))
    dry_eligible_df = dry_eligible_df[dry_eligible_df["site_key"].isin(eligible_site_keys)].copy()
    dry_chunks = []
    for site_key, site_df in dry_eligible_df.groupby("site_key", dropna=False):
        site_df = site_df.sort_values(["obs", "date"], ascending=[True, True], kind="mergesort")
        dry_n = max(1, int(np.ceil(dry_quantile * len(site_df))))
        dry_site_df = site_df.iloc[:dry_n].copy()
        dry_site_df["dry_rank_within_site"] = np.arange(1, len(dry_site_df) + 1)
        dry_site_df["dry_n_within_site"] = dry_n
        dry_site_df["site_n_observations"] = len(site_df)
        dry_site_df["dry_lfmc_threshold"] = float(dry_site_df["obs"].max())
        dry_chunks.append(dry_site_df)
    if len(dry_chunks) == 0:
        raise ValueError("No LFMC rows were available for the Figure 4 driest-observation panel")
    dry_lfmc_df = pd.concat(dry_chunks, ignore_index=True)
    overall_metrics = compute_basic_metrics(lfmc_df["obs"].values, lfmc_df["pred"].values)
    site_mean_metrics = compute_basic_metrics(
        site_summary_df["obs_mean"].values,
        site_summary_df["pred_mean"].values,
    )
    anomaly_metrics = compute_basic_metrics(
        anomaly_df["obs_anom"].values,
        anomaly_df["pred_anom"].values,
    )
    month_anom_metrics = compute_basic_metrics(
        month_anom_df["obs_dev"].values,
        month_anom_df["pred_dev"].values,
    )
    dry_lfmc_metrics = compute_basic_metrics(
        dry_lfmc_df["obs"].values,
        dry_lfmc_df["pred"].values,
    )
    dry_diagnostic_row = {
        "model_key": str(fig_cfg["model_key"]),
        "dry_site_min_obs": int(dry_site_min_obs),
        "dry_site_quantile": float(dry_quantile),
    }
    dry_diagnostic_row.update(
        dry_diagnostics(
            dry_lfmc_df["obs"].values,
            dry_lfmc_df["pred"].values,
            worst_error_fraction=dry_worst_error_fraction,
        )
    )
    print("Prepared Figure 4 metrics and bias annotations")
    panels = [
        {
            "title": "Overall",
            "panel_label": "a",
            "kind": "hexbin",
            "x": lfmc_df["obs"].values,
            "y": lfmc_df["pred"].values,
            "xlabel": "Observed LFMC (%)",
            "ylabel": "Predicted LFMC (%)",
            "metrics": {
                "n": overall_metrics["n"],
                "rmse": overall_metrics["rmse"],
                "r2": overall_metrics["r2"],
                "bias": mean_bias(lfmc_df["obs"].values, lfmc_df["pred"].values),
                "rmse_std": np.nan,
                "r2_std": np.nan,
            },
            "cbar_label": "Count",
            "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
        },
        {
            "title": "Site Anomalies",
            "panel_label": "b",
            "kind": "hexbin",
            "x": anomaly_df["obs_anom"].values,
            "y": anomaly_df["pred_anom"].values,
            "xlabel": "Observed anomaly (%)",
            "ylabel": "Predicted anomaly (%)",
            "metrics": {
                "n": anomaly_metrics["n"],
                "rmse": anomaly_metrics["rmse"],
                "r2": anomaly_metrics["r2"],
                "bias": mean_bias(anomaly_df["obs_anom"].values, anomaly_df["pred_anom"].values),
                "rmse_std": np.nan,
                "r2_std": np.nan,
            },
            "cbar_label": "Count",
            "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
        },
        {
            "title": "Site Means",
            "panel_label": "c",
            "kind": "scatter",
            "x": site_summary_df["obs_mean"].values,
            "y": site_summary_df["pred_mean"].values,
            "xlabel": "Observed site mean (%)",
            "ylabel": "Predicted site mean (%)",
            "metrics": {
                "n": site_mean_metrics["n"],
                "rmse": site_mean_metrics["rmse"],
                "r2": site_mean_metrics["r2"],
                "bias": mean_bias(site_summary_df["obs_mean"].values, site_summary_df["pred_mean"].values),
                "rmse_std": np.nan,
                "r2_std": np.nan,
            },
            "color_array": site_summary_df["n_obs"].values,
            "cbar_label": "Observations at site",
            "cmap": "viridis",
            "cbar_vmax": 200,
            "cbar_extend": "max",
        },
        {
            "title": "Deviation from\nmonthly average",
            "panel_label": "d",
            "kind": "hexbin",
            "x": month_anom_df["obs_dev"].values,
            "y": month_anom_df["pred_dev"].values,
            "xlabel": "Observed deviation from monthly mean (%)",
            "ylabel": "Predicted deviation\nfrom monthly mean (%)",
            "metrics": {
                "n": month_anom_metrics["n"],
                "rmse": month_anom_metrics["rmse"],
                "r2": month_anom_metrics["r2"],
                "bias": mean_bias(month_anom_df["obs_dev"].values, month_anom_df["pred_dev"].values),
                "rmse_std": np.nan,
                "r2_std": np.nan,
            },
            "cbar_label": "Count",
            "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
            "cbar_vmax": 600,
            "cbar_extend": "max",
            "xlim": (-100, 100),
            "ylim": (-100, 100),
        },
        {
            "title": f"Driest {dry_quantile * 100:.0f}%\nwithin site",
            "panel_label": "e",
            "kind": "hexbin",
            "x": dry_lfmc_df["obs"].values,
            "y": dry_lfmc_df["pred"].values,
            "xlabel": "Observed LFMC (%)",
            "ylabel": "Predicted LFMC (%)",
            "metrics": {
                "n": dry_lfmc_metrics["n"],
                "rmse": dry_lfmc_metrics["rmse"],
                "r2": dry_lfmc_metrics["r2"],
                "bias": mean_bias(dry_lfmc_df["obs"].values, dry_lfmc_df["pred"].values),
                "rmse_std": np.nan,
                "r2_std": np.nan,
            },
            "cbar_label": "Count",
            "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
        },
    ]
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "figure_04_site_tables")
    dry_table_path = _table_output_path(runtime, "figure_04_driest_lfmc_points")
    dry_diagnostics_path = _table_output_path(runtime, "figure_04_driest_lfmc_diagnostics")
    combined_table = site_summary_df.merge(
        anomaly_df.groupby(["latitude", "longitude"], as_index=False).agg(
            n_anomaly_rows=("obs_anom", "size")
        ),
        on=["latitude", "longitude"],
        how="left",
    )
    month_summary = (
        month_anom_df.groupby(["latitude", "longitude"], as_index=False)
        .agg(n_month_dev_rows=("obs_dev", "size"))
    )
    combined_table = combined_table.merge(
        month_summary,
        on=["latitude", "longitude"],
        how="left",
    )
    combined_table.to_csv(table_path, index=False)
    dry_lfmc_df.to_csv(dry_table_path, index=False)
    pd.DataFrame.from_records([dry_diagnostic_row]).to_csv(dry_diagnostics_path, index=False)
    print(f"Wrote Figure 4 site table: {table_path}")
    print(f"Wrote Figure 4 driest LFMC point table: {dry_table_path}")
    print(f"Wrote Figure 4 driest LFMC diagnostics table: {dry_diagnostics_path}")
    print(f"Plotting Figure 4 to {save_path}")
    plot_scatter_triptych(
        panels=panels,
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        title_fontsize=int(fig_cfg.get("title_fontsize", 24)),
        axis_label_fontsize=int(fig_cfg.get("axis_label_fontsize", 28)),
        tick_label_fontsize=int(fig_cfg.get("tick_label_fontsize", 26)),
        colorbar_label_fontsize=int(fig_cfg.get("colorbar_label_fontsize", 28)),
        colorbar_tick_fontsize=int(fig_cfg.get("colorbar_tick_fontsize", 26)),
        stats_fontsize=int(fig_cfg.get("stats_fontsize", 20)),
        panel_label_fontsize=int(fig_cfg.get("panel_label_fontsize", 30)),
    )
    return save_path


def build_figure_5(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_5"]
    model_entry = _load_ensemble_site_entry(cfg, str(fig_cfg["model_key"]))
    selected = _select_timeseries_sites(runtime, model_entry, fig_cfg)
    panels = []
    for criterion in fig_cfg["criteria_order"]:
        site_list = selected.get(criterion, [])
        if len(site_list) == 0:
            continue
        panels.append(
            _build_timeseries_panel(
                runtime=runtime,
                model_entry=model_entry,
                site_key=site_list[0],
                criterion_label=str(criterion).capitalize(),
                years_to_plot=int(fig_cfg.get("years_to_plot", 3)),
                plot_vv=bool(fig_cfg.get("plot_vv", False)),
                plot_vh=bool(fig_cfg.get("plot_vh", False)),
                prediction_label=fig_cfg.get("prediction_legend_label"),
            )
        )
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "figure_05_sites")
    site_rows = []
    for criterion, site_list in selected.items():
        for rank_idx, site_key in enumerate(site_list, start=1):
            site_rows.append(
                {
                    "criterion": criterion,
                    "rank": rank_idx,
                    "site_key": site_key,
                }
            )
    pd.DataFrame.from_records(site_rows).to_csv(table_path, index=False)
    plot_stacked_timeseries_panels(
        panels=panels,
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        locator_inset_bounds=fig_cfg.get("locator_inset_bounds"),
        locator_marker_size=float(fig_cfg.get("locator_marker_size", 24.0)),
        uncertainty_before_observations=bool(fig_cfg.get("uncertainty_before_observations", False)),
        legend_fontsize=fig_cfg.get("legend_fontsize"),
        legend_ncol=fig_cfg.get("legend_ncol"),
    )
    return save_path


def build_figure_6(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_6"]
    site_min_obs = int(cfg["variability"]["site_min_obs"])
    monthly_min_obs = int(cfg["variability"]["monthly_min_obs"])
    monthly_min_years = int(cfg["variability"]["monthly_min_years"])
    metric_df = _landcover_metric_table(
        runtime=runtime,
        model_key=str(fig_cfg["model_key"]),
        site_min_obs=site_min_obs,
        monthly_min_obs=monthly_min_obs,
        monthly_min_years=monthly_min_years,
    )
    metric_df = _prepend_overall_landcover_metrics(
        runtime=runtime,
        model_key=str(fig_cfg["model_key"]),
        metric_df=metric_df,
        site_min_obs=site_min_obs,
        monthly_min_obs=monthly_min_obs,
        monthly_min_years=monthly_min_years,
    )
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "figure_06_landcover_metrics")
    metric_df.to_csv(table_path, index=False)
    categories = metric_df["dominant_landcover"].astype(str).tolist()
    values = np.column_stack(
        [
            metric_df["overall_r2"].to_numpy(dtype=float),
            metric_df["site_anom_r2"].to_numpy(dtype=float),
            metric_df["site_mean_r2"].to_numpy(dtype=float),
            metric_df["monthly_dev_r2"].to_numpy(dtype=float),
        ]
    )
    counts = np.column_stack(
        [
            metric_df["overall_n"].to_numpy(dtype=float),
            metric_df["site_anom_n"].to_numpy(dtype=float),
            metric_df["site_mean_n"].to_numpy(dtype=float),
            metric_df["monthly_dev_n"].to_numpy(dtype=float),
        ]
    )
    errors = np.column_stack(
        [
            metric_df["overall_r2_std"].to_numpy(dtype=float),
            metric_df["site_anom_r2_std"].to_numpy(dtype=float),
            metric_df["site_mean_r2_std"].to_numpy(dtype=float),
            metric_df["monthly_dev_r2_std"].to_numpy(dtype=float),
        ]
    )
    plot_landcover_metric_grouped(
        categories=categories,
        metric_labels=["Overall", "Anomaly", "Site Mean", "Deviation from monthly mean"],
        values=values,
        counts=counts,
        errors=errors,
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        colors=[
            fig_cfg["metric_colors"][0],
            fig_cfg["metric_colors"][1],
            fig_cfg["metric_colors"][2],
            fig_cfg["metric_colors"][3],
        ],
        legend_below=True,
        group_gap_scale=float(fig_cfg.get("group_gap_scale", 1.0)),
        count_label_rotation=float(fig_cfg.get("count_label_rotation", 0.0)),
        count_values_only=bool(fig_cfg.get("count_values_only", False)),
        n_label=str(fig_cfg["n_label"]) if fig_cfg.get("n_label") is not None else None,
        count_label_y=float(fig_cfg.get("count_label_y", -0.035)),
        x_tick_pad=float(fig_cfg.get("x_tick_pad", 36.0)),
        y_tick_fontsize=fig_cfg.get("y_tick_fontsize"),
        category_label_fontsize=fig_cfg.get("category_label_fontsize"),
        annotation_fontsize=fig_cfg.get("annotation_fontsize"),
        value_label_fontsize=fig_cfg.get("value_label_fontsize"),
        count_label_fontsize=fig_cfg.get("count_label_fontsize"),
        n_label_fontsize=fig_cfg.get("n_label_fontsize"),
        y_label_fontsize=fig_cfg.get("y_label_fontsize"),
        legend_fontsize=fig_cfg.get("legend_fontsize"),
        legend_ncol=fig_cfg.get("legend_ncol"),
        legend_bbox_y=float(fig_cfg.get("legend_bbox_y", 0.015)),
        legend_bottom=float(fig_cfg.get("legend_bottom", 0.31)),
        value_label_rotation=float(fig_cfg.get("value_label_rotation", 0.0)),
        wrap_landcover_labels=bool(fig_cfg.get("wrap_landcover_labels", False)),
    )
    return save_path


def build_figure_7(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_7"]
    site_min_obs = int(cfg["variability"]["site_min_obs"])
    monthly_min_obs = int(cfg["variability"]["monthly_min_obs"])
    monthly_min_years = int(cfg["variability"]["monthly_min_years"])
    categories = ["overall"] + list(cfg["filters"]["landcover_order"])
    model_labels = []
    colors = []
    overall_values = []
    overall_errors = []
    overall_counts = []
    merged_rows = []
    for model_key in fig_cfg["model_keys"]:
        model_cfg = _model_cfg(cfg, str(model_key))
        model_labels.append(str(model_cfg["display_name"]))
        configured_colors = fig_cfg.get("model_colors")
        if configured_colors is not None:
            colors.append(str(configured_colors[len(colors)]))
        else:
            colors.append(str(model_cfg["color"]))
        metric_df = _landcover_metric_table(
            runtime=runtime,
            model_key=str(model_key),
            site_min_obs=site_min_obs,
            monthly_min_obs=monthly_min_obs,
            monthly_min_years=monthly_min_years,
        )
        metric_df = _prepend_overall_landcover_metrics(
            runtime=runtime,
            model_key=str(model_key),
            metric_df=metric_df,
            site_min_obs=site_min_obs,
            monthly_min_obs=monthly_min_obs,
            monthly_min_years=monthly_min_years,
        )
        metric_lookup = metric_df.set_index("dominant_landcover").to_dict("index")
        overall_row = []
        overall_err_row = []
        overall_count_row = []
        for category in categories:
            row = metric_lookup.get(category, {})
            overall_row.append(float(row.get("overall_r2", np.nan)))
            overall_err_row.append(float(row.get("overall_r2_std", np.nan)))
            overall_count_row.append(float(row.get("n_points", np.nan)))
            merged_rows.append(
                {
                    "model_key": model_key,
                    "display_name": model_cfg["display_name"],
                    "dominant_landcover": category,
                    "overall_r2": row.get("overall_r2", np.nan),
                    "overall_r2_std": row.get("overall_r2_std", np.nan),
                    "overall_n": row.get("n_points", np.nan),
                }
            )
        overall_values.append(overall_row)
        overall_errors.append(overall_err_row)
        overall_counts.append(overall_count_row)
    legend_labels = list(fig_cfg.get("legend_labels", model_labels))
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "figure_08_ablation_overall")
    pd.DataFrame.from_records(merged_rows).to_csv(table_path, index=False)
    plot_landcover_comparison_panels(
        categories=categories,
        model_labels=legend_labels,
        colors=colors,
        panels=[
            {
                "title": str(fig_cfg.get("panel_title", "")),
                "ylabel": "R²",
                "values": np.asarray(overall_values, dtype=float).T,
                "errors": np.asarray(overall_errors, dtype=float).T,
                "counts": np.asarray(overall_counts, dtype=float).T,
            }
        ],
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        group_gap_scale=float(fig_cfg.get("group_gap_scale", 1.0)),
        count_label_y=float(fig_cfg.get("count_label_y", -0.06)),
        count_values_only=bool(fig_cfg.get("count_values_only", False)),
        n_label=str(fig_cfg["n_label"]) if fig_cfg.get("n_label") is not None else None,
        x_tick_pad=float(fig_cfg.get("x_tick_pad", 28.0)),
        value_label_fontsize=fig_cfg.get("value_label_fontsize"),
        count_label_fontsize=fig_cfg.get("count_label_fontsize"),
        n_label_fontsize=fig_cfg.get("n_label_fontsize"),
        y_tick_fontsize=fig_cfg.get("y_tick_fontsize"),
        category_label_fontsize=fig_cfg.get("category_label_fontsize"),
        y_label_fontsize=fig_cfg.get("y_label_fontsize"),
        legend_fontsize=fig_cfg.get("legend_fontsize"),
        legend_bbox_y=float(fig_cfg.get("legend_bbox_y", 0.005)),
        legend_bottom=float(fig_cfg.get("legend_bottom", 0.27)),
        value_label_rotation=float(fig_cfg.get("value_label_rotation", 0.0)),
        count_label_rotation=float(fig_cfg.get("count_label_rotation", 0.0)),
    )
    return save_path


def build_figure_8(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_8"]
    summary_df, member_df = _build_training_sample_landcover_tables(runtime, fig_cfg)
    if len(summary_df) == 0:
        raise ValueError("No training-sample landcover rows were available for Figure 8")
    summary_table_path = _table_output_path(runtime, "supplementary_figure_02_training_sample_landcover_counts")
    member_table_path = _table_output_path(runtime, "supplementary_figure_02_training_sample_landcover_member_counts")
    summary_df.to_csv(summary_table_path, index=False)
    member_df.to_csv(member_table_path, index=False)
    categories = [
        category for category in cfg["filters"]["landcover_order"]
        if category in summary_df["dominant_landcover"].astype(str).tolist()
    ]
    dataset_order = list(fig_cfg.get("dataset_order", fig_cfg["datasets"].keys()))
    dataset_lookup = {
        row["dataset_key"]: row
        for _, row in summary_df[["dataset_key", "label", "color"]].drop_duplicates().iterrows()
    }
    dataset_labels = [
        str(dataset_lookup[key]["label"])
        for key in dataset_order
    ]
    colors = [str(dataset_lookup[key]["color"]) for key in dataset_order]
    values = []
    count_values = []
    for category in categories:
        value_row = []
        count_row = []
        for dataset_key in dataset_order:
            row = summary_df[
                (summary_df["dataset_key"] == dataset_key)
                & (summary_df["dominant_landcover"] == category)
            ]
            if len(row) == 0:
                value_row.append(np.nan)
                count_row.append(np.nan)
                continue
            value_row.append(float(row.iloc[0]["fraction"]))
            count_row.append(float(row.iloc[0]["mean_n_samples"]))
        values.append(value_row)
        count_values.append(count_row)
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    plot_training_sample_landcover_comparison(
        categories=categories,
        dataset_labels=dataset_labels,
        colors=colors,
        values=np.asarray(values, dtype=float),
        errors=None,
        count_values=np.asarray(count_values, dtype=float),
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        note_text=None,
        legend_below=bool(fig_cfg.get("legend_below", True)),
        text_scale=float(fig_cfg.get("text_scale", 1.0)),
        x_label_rotation=float(fig_cfg.get("x_label_rotation", 25.0)),
        counts_below_axis=bool(fig_cfg.get("counts_below_axis", False)),
        count_label_y=float(fig_cfg.get("count_label_y", -0.12)),
        value_label_fontsize=fig_cfg.get("value_label_fontsize"),
        count_label_fontsize=fig_cfg.get("count_label_fontsize"),
        value_label_rotation=float(fig_cfg.get("value_label_rotation", 0.0)),
        y_tick_fontsize=fig_cfg.get("y_tick_fontsize"),
        category_label_fontsize=fig_cfg.get("category_label_fontsize"),
        y_label_fontsize=fig_cfg.get("y_label_fontsize"),
        legend_fontsize=fig_cfg.get("legend_fontsize"),
        legend_bbox_y=float(fig_cfg.get("legend_bbox_y", 0.02)),
        legend_bottom=fig_cfg.get("legend_bottom"),
        x_tick_pad=fig_cfg.get("x_tick_pad"),
    )
    return save_path


def build_supplementary_figure_1(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_1"]
    model_key = str(fig_cfg["model_key"])
    table_path = _table_output_path(runtime, "supplementary_figure_01_sites")
    selected = None
    if bool(fig_cfg.get("reuse_existing_sites_table", False)) and os.path.exists(table_path):
        site_table = pd.read_csv(table_path)
        required_columns = {"criterion", "site_key"}
        if required_columns.issubset(site_table.columns):
            if "rank" in site_table.columns:
                site_table = site_table.sort_values(["criterion", "rank"])
            selected = {
                str(criterion): group["site_key"].astype(str).tolist()
                for criterion, group in site_table.groupby("criterion", sort=False)
            }
            print(f"Reusing Supplementary Figure 1 site table: {table_path}")
    selected_site_keys = None
    if selected is not None:
        selected_site_keys = [
            site_key
            for criterion in fig_cfg["criteria_order"]
            for site_key in selected.get(criterion, [])
        ]
    model_entry = _load_ensemble_site_entry(cfg, model_key, site_keys=selected_site_keys)
    if selected is None:
        print(
            "Supplementary Figure 1 SAR test R2 across members "
            "(each member uses all held-out test rows across folds):"
        )
        sar_member_r2_vals = {"vv": [], "vh": []}
        sar_member_n_vals = {"vv": [], "vh": []}
        for member_idx, member_dir in enumerate(model_entry["member_dirs"], start=1):
            print(
                f"Loading SAR test outputs for Supplementary Figure 1 member "
                f"{member_idx}/{len(model_entry['member_dirs'])}: {os.path.basename(member_dir)}"
            )
            member_eval_df = _load_fold_predictions_for_targets(member_dir, ["vv", "vh"])
            for target_name in ["vv", "vh"]:
                target_df = member_eval_df[
                    member_eval_df["target"].astype(str).str.strip().str.lower() == target_name
                ].reset_index(drop=True)
                member_metrics = compute_basic_metrics(
                    target_df["obs"].values,
                    target_df["pred"].values,
                )
                sar_member_r2_vals[target_name].append(member_metrics.get("r2", np.nan))
                sar_member_n_vals[target_name].append(member_metrics.get("n", 0))
            del member_eval_df
            gc.collect()
        for target_name, display_name in [("vv", "VV"), ("vh", "VH")]:
            member_r2_vals = sar_member_r2_vals[target_name]
            member_n_vals = sar_member_n_vals[target_name]
            finite_r2 = np.asarray(member_r2_vals, dtype=float)
            finite_r2 = finite_r2[np.isfinite(finite_r2)]
            mean_r2 = float(finite_r2.mean()) if finite_r2.size > 0 else np.nan
            std_r2 = _metric_std(member_r2_vals)
            total_n = int(np.sum(np.asarray(member_n_vals, dtype=int)))
            if np.isfinite(mean_r2):
                r2_summary = f"{mean_r2:.2f}"
                if np.isfinite(std_r2):
                    r2_summary = f"{r2_summary} +/- {std_r2:.2f}"
            else:
                r2_summary = "nan"
            print(
                f"  {display_name}: R2 = {r2_summary} "
                f"across {finite_r2.size}/{len(member_r2_vals)} members; "
                f"total held-out rows across members = {total_n}"
            )
        selected = _select_timeseries_sites(
            runtime,
            model_entry,
            fig_cfg,
            require_sar_overlap_same_year=True,
            prefer_sar_observation_density=True,
        )
    panels = []
    for criterion in fig_cfg["criteria_order"]:
        site_list = selected.get(criterion, [])
        if len(site_list) == 0:
            continue
        panels.append(
            _build_timeseries_panel(
                runtime=runtime,
                model_entry=model_entry,
                site_key=site_list[0],
                criterion_label=str(criterion).capitalize(),
                years_to_plot=int(fig_cfg.get("years_to_plot", 3)),
                plot_vv=bool(fig_cfg.get("plot_vv", False)),
                plot_vh=bool(fig_cfg.get("plot_vh", False)),
                prefer_sar_observation_density=True,
                prediction_label=fig_cfg.get("prediction_legend_label"),
                include_sar_r2_in_title=bool(fig_cfg.get("include_sar_r2_in_title", False)),
            )
        )
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    site_rows = []
    for criterion, site_list in selected.items():
        for rank_idx, site_key in enumerate(site_list, start=1):
            site_rows.append(
                {
                    "criterion": criterion,
                    "rank": rank_idx,
                    "site_key": site_key,
                }
            )
    pd.DataFrame.from_records(site_rows).to_csv(table_path, index=False)
    plot_stacked_timeseries_panels(
        panels=panels,
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        locator_inset_bounds=fig_cfg.get("locator_inset_bounds"),
        locator_marker_size=float(fig_cfg.get("locator_marker_size", 24.0)),
        uncertainty_before_observations=bool(fig_cfg.get("uncertainty_before_observations", False)),
        legend_fontsize=fig_cfg.get("legend_fontsize"),
        legend_ncol=fig_cfg.get("legend_ncol"),
    )
    return save_path


def build_supplementary_figure_2(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_2"]
    site_min_obs = int(fig_cfg.get("site_min_obs", cfg["variability"]["site_min_obs"]))
    site_r2_df = _build_site_r2_landcover_df(
        runtime=runtime,
        model_key=str(fig_cfg["model_key"]),
        min_obs=site_min_obs,
    )
    if len(site_r2_df) == 0:
        raise ValueError("No site-level R2 rows were available for Supplementary Figure 2")
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "supplementary_figure_02_site_r2_by_landcover")
    site_r2_df.to_csv(table_path, index=False)
    categories = [
        category for category in cfg["filters"]["landcover_order"]
        if category in site_r2_df["dominant_landcover"].astype(str).tolist()
    ]
    plot_site_r2_landcover_distribution(
        site_r2_df=site_r2_df,
        categories=categories,
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        x_limits=fig_cfg.get("x_limits", [-1.0, 1.0]),
        show_summary_text=False,
    )
    return save_path


def build_supplementary_figure_3(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_3"]
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    plot_placeholder_figure(
        title=str(fig_cfg.get("title", "Supplementary Figure Placeholder")),
        description=str(fig_cfg.get("description", "")),
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
        group_gap_scale=float(fig_cfg.get("group_gap_scale", 1.0)),
        count_label_y=float(fig_cfg.get("count_label_y", -0.06)),
        count_values_only=bool(fig_cfg.get("count_values_only", False)),
        n_label=str(fig_cfg["n_label"]) if fig_cfg.get("n_label") is not None else None,
        x_tick_pad=float(fig_cfg.get("x_tick_pad", 28.0)),
        value_label_fontsize=fig_cfg.get("value_label_fontsize"),
        count_label_fontsize=fig_cfg.get("count_label_fontsize"),
        n_label_fontsize=fig_cfg.get("n_label_fontsize"),
        y_tick_fontsize=fig_cfg.get("y_tick_fontsize"),
        category_label_fontsize=fig_cfg.get("category_label_fontsize"),
        y_label_fontsize=fig_cfg.get("y_label_fontsize"),
        legend_fontsize=fig_cfg.get("legend_fontsize"),
        legend_bbox_y=float(fig_cfg.get("legend_bbox_y", 0.005)),
        legend_bottom=float(fig_cfg.get("legend_bottom", 0.27)),
        value_label_rotation=float(fig_cfg.get("value_label_rotation", 0.0)),
        count_label_rotation=float(fig_cfg.get("count_label_rotation", 0.0)),
    )
    return save_path


def build_supplementary_figure_4(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_4"]
    site_min_obs = int(cfg["variability"]["site_min_obs"])
    monthly_min_obs = int(cfg["variability"]["monthly_min_obs"])
    monthly_min_years = int(cfg["variability"]["monthly_min_years"])
    categories = ["overall"] + list(cfg["filters"]["landcover_order"])
    model_labels = []
    colors = []
    overall_values = []
    anomaly_values = []
    mean_values = []
    monthly_values = []
    overall_errors = []
    anomaly_errors = []
    mean_errors = []
    monthly_errors = []
    overall_counts = []
    anomaly_counts = []
    mean_counts = []
    monthly_counts = []
    merged_rows = []
    for model_key in fig_cfg["model_keys"]:
        model_cfg = _model_cfg(cfg, str(model_key))
        model_labels.append(str(model_cfg["display_name"]))
        configured_colors = fig_cfg.get("model_colors")
        if configured_colors is not None:
            colors.append(str(configured_colors[len(colors)]))
        else:
            colors.append(str(model_cfg["color"]))
        metric_df = _landcover_metric_table(
            runtime=runtime,
            model_key=str(model_key),
            site_min_obs=site_min_obs,
            monthly_min_obs=monthly_min_obs,
            monthly_min_years=monthly_min_years,
        )
        metric_df = _prepend_overall_landcover_metrics(
            runtime=runtime,
            model_key=str(model_key),
            metric_df=metric_df,
            site_min_obs=site_min_obs,
            monthly_min_obs=monthly_min_obs,
            monthly_min_years=monthly_min_years,
        )
        metric_lookup = metric_df.set_index("dominant_landcover").to_dict("index")
        overall_row = []
        anomaly_row = []
        mean_row = []
        monthly_row = []
        overall_err_row = []
        anomaly_err_row = []
        mean_err_row = []
        monthly_err_row = []
        overall_count_row = []
        anomaly_count_row = []
        mean_count_row = []
        monthly_count_row = []
        for category in categories:
            row = metric_lookup.get(category, {})
            overall_row.append(float(row.get("overall_r2", np.nan)))
            anomaly_row.append(float(row.get("site_anom_r2", np.nan)))
            mean_row.append(float(row.get("site_mean_r2", np.nan)))
            monthly_row.append(float(row.get("monthly_dev_r2", np.nan)))
            overall_err_row.append(float(row.get("overall_r2_std", np.nan)))
            anomaly_err_row.append(float(row.get("site_anom_r2_std", np.nan)))
            mean_err_row.append(float(row.get("site_mean_r2_std", np.nan)))
            monthly_err_row.append(float(row.get("monthly_dev_r2_std", np.nan)))
            overall_count_row.append(float(row.get("overall_n", np.nan)))
            anomaly_count_row.append(float(row.get("site_anom_n", np.nan)))
            mean_count_row.append(float(row.get("site_mean_n", np.nan)))
            monthly_count_row.append(float(row.get("monthly_dev_n", np.nan)))
            merged_rows.append(
                {
                    "model_key": model_key,
                    "display_name": model_cfg["display_name"],
                    "dominant_landcover": category,
                    "overall_r2": row.get("overall_r2", np.nan),
                    "site_anom_r2": row.get("site_anom_r2", np.nan),
                    "site_mean_r2": row.get("site_mean_r2", np.nan),
                    "monthly_dev_r2": row.get("monthly_dev_r2", np.nan),
                    "overall_r2_std": row.get("overall_r2_std", np.nan),
                    "site_anom_r2_std": row.get("site_anom_r2_std", np.nan),
                    "site_mean_r2_std": row.get("site_mean_r2_std", np.nan),
                    "monthly_dev_r2_std": row.get("monthly_dev_r2_std", np.nan),
                    "overall_n": row.get("overall_n", np.nan),
                    "site_anom_n": row.get("site_anom_n", np.nan),
                    "site_mean_n": row.get("site_mean_n", np.nan),
                    "monthly_dev_n": row.get("monthly_dev_n", np.nan),
                }
            )
        overall_values.append(overall_row)
        anomaly_values.append(anomaly_row)
        mean_values.append(mean_row)
        monthly_values.append(monthly_row)
        overall_errors.append(overall_err_row)
        anomaly_errors.append(anomaly_err_row)
        mean_errors.append(mean_err_row)
        monthly_errors.append(monthly_err_row)
        overall_counts.append(overall_count_row)
        anomaly_counts.append(anomaly_count_row)
        mean_counts.append(mean_count_row)
        monthly_counts.append(monthly_count_row)
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "supplementary_figure_03_ablation_all_metrics")
    pd.DataFrame.from_records(merged_rows).to_csv(table_path, index=False)
    legend_labels = list(fig_cfg.get("legend_labels", model_labels))
    plot_landcover_comparison_panels(
        categories=categories,
        model_labels=legend_labels,
        colors=colors,
        panels=[
            {
                "title": "Overall",
                "ylabel": "R²",
                "values": np.asarray(overall_values, dtype=float).T,
                "errors": np.asarray(overall_errors, dtype=float).T,
                "counts": np.asarray(overall_counts, dtype=float).T,
            },
            {
                "title": "Anomalies",
                "ylabel": "R²",
                "values": np.asarray(anomaly_values, dtype=float).T,
                "errors": np.asarray(anomaly_errors, dtype=float).T,
                "counts": np.asarray(anomaly_counts, dtype=float).T,
            },
            {
                "title": "Site Means",
                "ylabel": "R²",
                "values": np.asarray(mean_values, dtype=float).T,
                "errors": np.asarray(mean_errors, dtype=float).T,
                "counts": np.asarray(mean_counts, dtype=float).T,
            },
            {
                "title": "Monthly variability",
                "ylabel": "R²",
                "values": np.asarray(monthly_values, dtype=float).T,
                "errors": np.asarray(monthly_errors, dtype=float).T,
                "counts": np.asarray(monthly_counts, dtype=float).T,
            },
        ],
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
    )
    return save_path


def build_supplementary_figure_5(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_5"]
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    plot_placeholder_figure(
        title=str(fig_cfg.get("title", "Supplementary Figure Placeholder")),
        description=str(fig_cfg.get("description", "")),
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
    )
    return save_path


def _member_distribution_table(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    group_cols = list(group_cols or [])
    rows = []
    if len(df) == 0:
        return pd.DataFrame(columns=group_cols + ["metric", "mean", "sd", "min", "max"])
    if len(group_cols) == 0:
        grouped = [((), df)]
    else:
        grouped = df.groupby(group_cols, dropna=False)
    for group_key, group_df in grouped:
        if len(group_cols) == 0:
            group_values = {}
        else:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = {
                group_col: group_value
                for group_col, group_value in zip(group_cols, group_key)
            }
        for metric_col in metric_cols:
            if metric_col not in group_df.columns:
                continue
            finite = _finite_series(group_df[metric_col])
            row = dict(group_values)
            row.update(
                {
                    "metric": metric_col,
                    "mean": float(finite.mean()) if len(finite) > 0 else np.nan,
                    "sd": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                    "min": float(finite.min()) if len(finite) > 0 else np.nan,
                    "max": float(finite.max()) if len(finite) > 0 else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_result_statistics(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    stats_cfg = cfg["figures"]["result_statistics"]
    lfmc_model_key = str(stats_cfg.get("lfmc_model_key", "single_task_16"))
    sar_model_key = str(stats_cfg.get("sar_model_key", "multitask_16"))
    performance_model_key = str(stats_cfg.get("performance_model_key", "multitask_16"))
    dry_site_min_obs = int(stats_cfg.get("dry_site_min_obs", 10))
    dry_site_quantile = float(stats_cfg.get("dry_site_quantile", 0.20))

    print("  Building reproducible LFMC sampling statistics...")
    lfmc_by_member_df, lfmc_overall_df, lfmc_site_detail_df = _build_lfmc_sampling_statistics(
        runtime,
        model_key=lfmc_model_key,
    )
    lfmc_location_keys = {
        (float(row.latitude), float(row.longitude))
        for row in lfmc_site_detail_df[["latitude", "longitude"]].itertuples(index=False)
    }

    print("  Building reproducible Sentinel-1 sampling statistics...")
    (
        s1_by_member_df,
        s1_overall_df,
        s1_coincident_by_member_df,
        s1_coincident_overall_df,
    ) = _build_s1_sampling_statistics(
        runtime,
        model_key=sar_model_key,
        lfmc_location_keys=lfmc_location_keys,
    )

    print("  Building reproducible dry-LFMC performance statistics...")
    (
        dry_performance_df,
        dry_performance_by_member_df,
        dry_performance_by_site_df,
    ) = _build_dry_lfmc_performance_statistics(
        runtime,
        model_key=performance_model_key,
        min_obs=dry_site_min_obs,
        dry_quantile=dry_site_quantile,
    )
    lfmc_member_distribution_df = _member_distribution_table(
        lfmc_by_member_df,
        metric_cols=[
            "n_sites",
            "single_species_sites",
            "multi_species_sites",
            "total_lfmc_observations",
            "total_lfmc_measurement_days",
            "mean_observations_per_site",
            "sd_observations_per_site",
            "median_observations_per_site",
            "min_observations_per_site",
            "max_observations_per_site",
            "mean_measurement_days_per_site",
            "sd_measurement_days_per_site",
            "median_measurement_days_per_site",
            "min_measurement_days_per_site",
            "max_measurement_days_per_site",
        ],
    )
    s1_member_distribution_df = _member_distribution_table(
        s1_by_member_df,
        metric_cols=[
            "n_locations",
            "total_s1_observations",
            "mean_s1_observations_per_location",
            "sd_s1_observations_per_location",
            "median_s1_observations_per_location",
            "min_s1_observations_per_location",
            "max_s1_observations_per_location",
            "mean_s1_observation_days_per_location",
            "sd_s1_observation_days_per_location",
            "median_s1_observation_days_per_location",
            "min_s1_observation_days_per_location",
            "max_s1_observation_days_per_location",
        ],
    )
    s1_coincident_member_distribution_df = _member_distribution_table(
        s1_coincident_by_member_df,
        metric_cols=[
            "s1_locations",
            "coincident_with_lfmc",
            "not_coincident_with_lfmc",
            "pct_coincident_with_lfmc",
        ],
    )
    dry_performance_member_distribution_df = _member_distribution_table(
        dry_performance_by_member_df,
        metric_cols=[
            "n_sites",
            "n_observations",
            "r2",
            "rmse",
            "n_sites_with_finite_site_r2",
            "mean_site_r2",
            "median_site_r2",
            "mean_site_rmse",
            "median_site_rmse",
        ],
        group_cols=["subset"],
    )

    outputs = [
        (
            "result_statistics_lfmc_site_summary_by_member.csv",
            lfmc_by_member_df,
            "LFMC sampling-site species and observation summaries for each ensemble member.",
        ),
        (
            "result_statistics_lfmc_site_summary_member_distribution.csv",
            lfmc_member_distribution_df,
            "Mean, standard deviation, minimum, and maximum of LFMC sampling summaries across ensemble members.",
        ),
        (
            "result_statistics_lfmc_site_summary_overall.csv",
            lfmc_overall_df,
            "LFMC sampling-site species and observation summaries after de-duplicating across members.",
        ),
        (
            "result_statistics_lfmc_site_details_overall.csv",
            lfmc_site_detail_df,
            "Per-site LFMC species counts, observation counts, and measurement-day counts.",
        ),
        (
            "result_statistics_s1_site_summary_by_member.csv",
            s1_by_member_df,
            "Sentinel-1 observation and observation-day summaries for each ensemble member.",
        ),
        (
            "result_statistics_s1_site_summary_member_distribution.csv",
            s1_member_distribution_df,
            "Mean, standard deviation, minimum, and maximum of Sentinel-1 summaries across ensemble members.",
        ),
        (
            "result_statistics_s1_site_summary_overall.csv",
            s1_overall_df,
            "Sentinel-1 observation and observation-day summaries after de-duplicating across members.",
        ),
        (
            "result_statistics_s1_lfmc_coincident_locations_by_member.csv",
            s1_coincident_by_member_df,
            "Sentinel-1 locations coincident and not coincident with LFMC sampling sites for each member.",
        ),
        (
            "result_statistics_s1_lfmc_coincident_locations_member_distribution.csv",
            s1_coincident_member_distribution_df,
            "Mean, standard deviation, minimum, and maximum of Sentinel-1/LFMC coincidence counts across members.",
        ),
        (
            "result_statistics_s1_lfmc_coincident_locations_overall.csv",
            s1_coincident_overall_df,
            "Sentinel-1 locations coincident and not coincident with LFMC sampling sites after aggregating members.",
        ),
        (
            "result_statistics_lfmc_dry_performance.csv",
            dry_performance_df,
            "Overall LFMC R2/RMSE for all eligible observations and the driest within-site observations.",
        ),
        (
            "result_statistics_lfmc_dry_performance_by_member.csv",
            dry_performance_by_member_df,
            "Per-member LFMC R2/RMSE for all eligible observations and the driest within-site observations.",
        ),
        (
            "result_statistics_lfmc_dry_performance_member_distribution.csv",
            dry_performance_member_distribution_df,
            "Mean, standard deviation, minimum, and maximum of dry-LFMC performance metrics across ensemble members.",
        ),
        (
            "result_statistics_lfmc_dry_performance_by_site.csv",
            dry_performance_by_site_df,
            "Per-site all-observation and dry-observation LFMC performance for eligible sites.",
        ),
    ]
    manifest_rows = []
    for filename, df, description in outputs:
        path = _table_output_path(runtime, os.path.splitext(filename)[0])
        df.to_csv(path, index=False)
        manifest_rows.append(
            {
                "filename": filename,
                "path": path,
                "n_rows": int(len(df)),
                "description": description,
            }
        )
        print(f"  Wrote {filename}: {len(df):,} rows")
    manifest_path = _table_output_path(runtime, os.path.splitext(str(stats_cfg["filename"]))[0])
    pd.DataFrame.from_records(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def build_enabled_figures(
    cfg: Dict[str, object],
    only_figures: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    runtime = init_runtime(cfg)
    figure_builders = {
        "result_statistics": build_result_statistics,
        "figure_1": build_figure_1,
        "figure_2": build_figure_2,
        "figure_3": build_figure_3,
        "figure_4": build_figure_4,
        "figure_5": build_figure_5,
        "figure_6": build_figure_6,
        "figure_7": build_figure_7,
        "figure_8": build_figure_8,
        "supplementary_figure_1": build_supplementary_figure_1,
        "supplementary_figure_2": build_supplementary_figure_2,
        "supplementary_figure_3": build_supplementary_figure_3,
        "supplementary_figure_4": build_supplementary_figure_4,
        "supplementary_figure_5": build_supplementary_figure_5,
    }
    outputs = {}
    for fig_key, fig_cfg in cfg["figures"].items():
        if not bool(fig_cfg.get("enabled", False)):
            continue
        if only_figures is not None and fig_key not in only_figures:
            continue
        builder = figure_builders.get(fig_key)
        if builder is None:
            raise NotImplementedError(
                f"No builder has been added yet for new workflow figure '{fig_key}'"
            )
        print(f"Building {fig_key} ...")
        outputs[fig_key] = builder(runtime)
        print(f"Wrote {fig_key}: {outputs[fig_key]}")
    return outputs
