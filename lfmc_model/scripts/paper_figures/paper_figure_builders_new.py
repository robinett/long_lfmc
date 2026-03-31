#!/usr/bin/env python3

import contextlib
import hashlib
import io
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
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
    _build_ensemble_eval_df,
    build_lfmc_space_time_tables,
    build_lfmc_y2y_df,
    build_site_month_anomaly_eval_df,
    build_site_landcover_lookup,
    compute_landcover_decomposition_metrics,
    compute_basic_metrics,
    load_fold_predictions,
)
from paper_figure_plotting import (  # noqa: E402
    plot_landcover_comparison_panels,
    plot_landcover_metric_grouped,
    plot_scatter_triptych,
    plot_stacked_timeseries_panels,
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
    return " | ".join(line1_parts) + "\n" + " | ".join(line2_parts)


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
    work = ranked.copy()
    work["percentile_dist"] = np.abs(work[metric_col] - target_value)
    work = work.sort_values(["percentile_dist", metric_col], ascending=[True, False]).reset_index(drop=True)
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


def _load_eval_context(runtime: Dict[str, object], model_key: str) -> Dict[str, object]:
    cache = runtime.setdefault("eval_contexts", {})
    if model_key in cache:
        return cache[model_key]
    model_cfg = _model_cfg(runtime["cfg"], model_key)
    member_dirs = _select_model_member_dirs(model_cfg)
    member_eval_dfs = [load_fold_predictions(member_dir) for member_dir in member_dirs]
    context = {
        "model_key": model_key,
        "member_dirs": member_dirs,
        "member_eval_dfs": member_eval_dfs,
        "eval_df": _build_ensemble_eval_df(member_eval_dfs),
    }
    cache[model_key] = context
    return context


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
            columns=["dominant_landcover", "site_mean_r2", "site_anom_r2", "n_sites"]
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
    return pd.concat([pd.DataFrame([overall_row]), metric_df], ignore_index=True)


def _load_ensemble_site_entry(cfg: Dict[str, object], model_key: str) -> Dict[str, object]:
    model_cfg = _model_cfg(cfg, model_key)
    member_dirs = _select_model_member_dirs(model_cfg)
    member_site_errors = {}
    member_site_error_list = []
    for member_idx, member_dir in enumerate(member_dirs, start=1):
        print(
            f"Loading site errors for {model_cfg['display_name']} member "
            f"{member_idx}/{len(member_dirs)}: {os.path.basename(member_dir)}"
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            this_site_error = get_site_error(member_dir)
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
) -> Dict[str, List[str]]:
    site_df = build_site_df(entry["site_error"], entry["site_error"].keys())
    site_df["r2"] = site_df["site"].map(
        lambda site: float(entry["site_error"][site].get("r2", np.nan))
    )
    keep_sites = []
    for site in site_df["site"].tolist():
        lfmc_dates = pd.to_datetime(entry["site_error"][site]["dates"], errors="coerce")
        lfmc_dates = lfmc_dates[lfmc_dates.notna()]
        selected_years = _top_consecutive_observation_years(lfmc_dates, int(fig_cfg.get("years_to_plot", 3)))
        observed_years = set(lfmc_dates.year.tolist())
        n_years_present = sum(1 for year in selected_years if year in observed_years)
        if n_years_present >= 2:
            keep_sites.append(site)
    site_df = site_df[site_df["site"].isin(keep_sites)].reset_index(drop=True)
    if require_sar_overlap_same_year:
        keep_sites = [
            site for site in site_df["site"].tolist()
            if _site_has_same_year_lfmc_sar_overlap(
                runtime=runtime,
                model_entry=entry,
                site_key=site,
                years_to_plot=int(fig_cfg.get("years_to_plot", 3)),
            )
        ]
        site_df = site_df[site_df["site"].isin(keep_sites)].reset_index(drop=True)
    ranked = site_df.copy()
    ranked = ranked[ranked["num_measurements"] >= int(fig_cfg["min_measurements"])]
    ranked = ranked[np.isfinite(ranked["r2"])]
    if len(ranked) == 0:
        if require_sar_overlap_same_year:
            raise ValueError("No eligible sites with same-year LFMC/SAR overlap were found")
        raise ValueError("No eligible sites were found for timeseries selection")
    ranked = ranked.sort_values("r2", ascending=False).reset_index(drop=True)
    percentile_cfg = runtime["cfg"].get("timeseries_selection", {}).get("r2_percentiles", {})
    used_sites = set()
    selected = {
        "good": _pick_percentile_sites(
            ranked,
            metric_col="r2",
            target_percentile=float(percentile_cfg.get("good", 95)),
            n_sites=int(fig_cfg["num_sites_per_criterion"]),
            used_sites=used_sites,
        ),
        "average": _pick_percentile_sites(
            ranked,
            metric_col="r2",
            target_percentile=float(percentile_cfg.get("average", 50)),
            n_sites=int(fig_cfg["num_sites_per_criterion"]),
            used_sites=used_sites,
        ),
        "poor": _pick_percentile_sites(
            ranked,
            metric_col="r2",
            target_percentile=float(percentile_cfg.get("poor", 5)),
            n_sites=int(fig_cfg["num_sites_per_criterion"]),
            used_sites=used_sites,
        ),
    }
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


def _site_has_same_year_lfmc_sar_overlap(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    years_to_plot: int,
) -> bool:
    if int(model_entry.get("model_num_tasks", 0)) < 3:
        return False
    lfmc_dates = pd.to_datetime(model_entry["site_error"][site_key]["dates"], errors="coerce")
    lfmc_dates = lfmc_dates[lfmc_dates.notna()]
    selected_years = _top_consecutive_observation_years(lfmc_dates, years_to_plot)
    if len(selected_years) == 0:
        return False
    vv_vh_obs = get_vv_vh_site_series(
        model_entry,
        site_key,
        runtime["vhvv_fold_cache"],
        start_date=None,
        end_date=None,
    )
    if vv_vh_obs is None:
        return False
    selected_year_set = set(int(year) for year in selected_years)
    sar_years = set()
    for key in ["vv_dates", "vh_dates"]:
        dates = pd.to_datetime(vv_vh_obs.get(key, []), errors="coerce")
        dates = dates[dates.notna()]
        if len(dates) == 0:
            continue
        sar_years.update(dates.year.tolist())
    return len(selected_year_set.intersection(sar_years)) > 0


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
        parts.append(_normalize_inference_output(out))
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
) -> Dict[str, object]:
    cfg = runtime["cfg"]
    anchor_site = model_entry["site_error"][site_key]
    lfmc_dates = _to_naive_datetime(anchor_site["dates"])
    lfmc_values = np.asarray(anchor_site["true_values"], dtype=float)
    selected_years = _top_consecutive_observation_years(lfmc_dates, years_to_plot)
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
    panel = {
        "title": _timeseries_panel_title(
            cfg=cfg,
            site_key=site_key,
            criterion_label=criterion_label,
            selected_years=selected_years,
            site_r2=site_r2,
        ),
        "series": [
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
        "eval_contexts": {},
        "landcover_metric_tables": {},
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
    return _copy_manual_asset_figure(runtime, "figure_1")


def build_figure_6(runtime: Dict[str, object]) -> str:
    return _copy_manual_asset_figure(runtime, "figure_6")


def build_figure_2(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_2"]
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
            )
        )
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "figure_02_sites")
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
    )
    return save_path


def build_supplementary_figure_1(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_1"]
    model_entry = _load_ensemble_site_entry(cfg, str(fig_cfg["model_key"]))
    selected = _select_timeseries_sites(
        runtime,
        model_entry,
        fig_cfg,
        require_sar_overlap_same_year=True,
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
            )
        )
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "supplementary_figure_01_sites")
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
    )
    return save_path


def build_supplementary_figure_2(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_2"]
    model_entries = [
        _load_ensemble_site_entry(cfg, str(model_key))
        for model_key in fig_cfg["model_keys"]
    ]
    anchor_entry, selected = _select_timeseries_sites_by_landcover(
        runtime=runtime,
        model_entries=model_entries,
        anchor_model_key=str(fig_cfg["anchor_model_key"]),
        num_sites_per_landcover=int(fig_cfg["num_sites_per_criterion"]),
        min_measurements=int(fig_cfg["min_measurements"]),
        years_to_plot=int(fig_cfg.get("years_to_plot", 3)),
        landcover_order=list(fig_cfg.get("landcover_order", cfg["filters"]["landcover_order"])),
    )
    panels = []
    panel_order = list(fig_cfg.get("landcover_order", cfg["filters"]["landcover_order"]))
    for criterion in panel_order:
        site_list = selected.get(criterion, [])
        if len(site_list) == 0:
            continue
        panels.append(
            _build_timeseries_comparison_panel(
                runtime=runtime,
                model_entries=model_entries,
                anchor_entry=anchor_entry,
                site_key=site_list[0],
                criterion_label=str(criterion).replace("_", " ").title(),
                years_to_plot=int(fig_cfg.get("years_to_plot", 3)),
            )
        )
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "supplementary_figure_02_sites")
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
    )
    return save_path


def build_figure_3(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_3"]
    site_min_obs = int(cfg["variability"]["site_min_obs"])
    monthly_min_obs = int(cfg["variability"]["monthly_min_obs"])
    monthly_min_years = int(cfg["variability"]["monthly_min_years"])
    context = _load_eval_context(runtime, str(fig_cfg["model_key"]))
    lfmc_df = context["eval_df"][context["eval_df"]["target"] == "lfmc"].reset_index(drop=True)
    _, site_summary_df, anomaly_df = _build_filtered_site_space_time_tables(
        lfmc_df,
        min_obs=site_min_obs,
    )
    _, month_anom_df, valid_month_groups = _build_filtered_site_month_anomaly_tables(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=monthly_min_obs,
        min_years=monthly_min_years,
    )
    if len(month_anom_df) == 0 or len(valid_month_groups) == 0:
        raise ValueError("No source-centered monthly anomaly rows available for Figure 3")
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
    overall_std = _overall_metric_std(context)
    space_std, time_std = _space_time_metric_stds(context, min_obs=site_min_obs)
    month_anom_std = _monthly_source_centered_metric_std(
        context,
        min_obs=monthly_min_obs,
        min_years=monthly_min_years,
    )
    panels = [
        {
            "title": "Overall",
            "kind": "hexbin",
            "x": lfmc_df["obs"].values,
            "y": lfmc_df["pred"].values,
            "xlabel": "Observed LFMC (%)",
            "ylabel": "Predicted LFMC (%)",
            "metrics": {
                "n": overall_metrics["n"],
                "rmse": overall_metrics["rmse"],
                "r2": overall_metrics["r2"],
                "rmse_std": overall_std["rmse"],
                "r2_std": overall_std["r2"],
            },
            "cbar_label": "Count",
            "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
        },
        {
            "title": "Site Anomalies",
            "kind": "hexbin",
            "x": anomaly_df["obs_anom"].values,
            "y": anomaly_df["pred_anom"].values,
            "xlabel": "Observed anomaly (%)",
            "ylabel": "Predicted anomaly (%)",
            "metrics": {
                "n": anomaly_metrics["n"],
                "rmse": anomaly_metrics["rmse"],
                "r2": anomaly_metrics["r2"],
                "rmse_std": time_std["rmse"],
                "r2_std": time_std["r2"],
            },
            "cbar_label": "Count",
            "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
        },
        {
            "title": "Site Means",
            "kind": "scatter",
            "x": site_summary_df["obs_mean"].values,
            "y": site_summary_df["pred_mean"].values,
            "xlabel": "Observed site mean (%)",
            "ylabel": "Predicted site mean (%)",
            "metrics": {
                "n": site_mean_metrics["n"],
                "rmse": site_mean_metrics["rmse"],
                "r2": site_mean_metrics["r2"],
                "rmse_std": space_std["rmse"],
                "r2_std": space_std["r2"],
            },
            "color_array": site_summary_df["n_obs"].values,
            "cbar_label": "Total observations",
            "cmap": "viridis",
            "cbar_vmax": 200,
            "cbar_extend": "max",
        },
        {
            "title": "Deviation from monthly average",
            "kind": "hexbin",
            "x": month_anom_df["obs_dev"].values,
            "y": month_anom_df["pred_dev"].values,
            "xlabel": "Observed deviation from monthly mean (%)",
            "ylabel": "Predicted deviation from monthly mean (%)",
            "metrics": {
                "n": month_anom_metrics["n"],
                "rmse": month_anom_metrics["rmse"],
                "r2": month_anom_metrics["r2"],
                "rmse_std": month_anom_std["rmse"],
                "r2_std": month_anom_std["r2"],
            },
            "cbar_label": "Count",
            "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
            "cbar_vmax": 600,
            "cbar_extend": "max",
            "xlim": (-100, 100),
            "ylim": (-100, 100),
        },
    ]
    save_path = _figure_output_path(runtime, str(fig_cfg["filename"]))
    table_path = _table_output_path(runtime, "figure_03_site_tables")
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
    plot_scatter_triptych(
        panels=panels,
        save_path=save_path,
        fontsize=int(cfg["plotting"].get("fontsize", 14)),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"].get("dpi", 350)),
    )
    return save_path


def build_figure_4(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_4"]
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
    table_path = _table_output_path(runtime, "figure_04_landcover_metrics")
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
            metric_df["n_points"].to_numpy(dtype=float),
            metric_df["n_points"].to_numpy(dtype=float),
            metric_df["n_sites"].to_numpy(dtype=float),
            metric_df["total_obs"].to_numpy(dtype=float),
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
    )
    return save_path


def build_figure_5(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_5"]
    site_min_obs = int(cfg["variability"]["site_min_obs"])
    monthly_min_obs = int(cfg["variability"]["monthly_min_obs"])
    monthly_min_years = int(cfg["variability"]["monthly_min_years"])
    landcover_categories = list(cfg["filters"]["landcover_order"])
    categories = ["overall"] + landcover_categories
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
            overall_count_row.append(float(row.get("n_points", np.nan)))
            anomaly_count_row.append(float(row.get("n_points", np.nan)))
            mean_count_row.append(float(row.get("n_sites", np.nan)))
            monthly_count_row.append(float(row.get("total_obs", np.nan)))
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
                    "overall_n": row.get("n_points", np.nan),
                    "site_anom_n": row.get("n_points", np.nan),
                    "site_mean_n": row.get("n_sites", np.nan),
                    "monthly_dev_n": row.get("total_obs", np.nan),
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
    table_path = _table_output_path(runtime, "figure_05_landcover_model_comparison")
    pd.DataFrame.from_records(merged_rows).to_csv(table_path, index=False)
    plot_landcover_comparison_panels(
        categories=categories,
        model_labels=model_labels,
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
                "title": "Site Anomalies",
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
                "title": "Deviation from monthly mean",
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


def build_enabled_figures(
    cfg: Dict[str, object],
    only_figures: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    runtime = init_runtime(cfg)
    figure_builders = {
        "figure_1": build_figure_1,
        "figure_2": build_figure_2,
        "figure_3": build_figure_3,
        "figure_4": build_figure_4,
        "figure_5": build_figure_5,
        "figure_6": build_figure_6,
        "supplementary_figure_1": build_supplementary_figure_1,
        "supplementary_figure_2": build_supplementary_figure_2,
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
