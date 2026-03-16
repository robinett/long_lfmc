#!/usr/bin/env python3

import contextlib
import hashlib
import io
import os
import sys
import time
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

from compare_timeseries import (  # noqa: E402
    _combined_obs_window,
    _slice_series_to_window,
    _to_naive_datetime,
    build_model_entries,
    build_site_df,
    find_vhvv_entry,
    get_model_inference_series,
    get_site_landcover_annotation,
    get_site_state_annotation,
    get_vv_vh_site_series,
    select_sites_for_anchor,
)
from eval_compare_models import _prepare_train_landcover_fraction_summary  # noqa: E402
from eval_deep import (  # noqa: E402
    build_site_month_anomaly_eval_df,
    build_lfmc_space_time_tables,
    build_lfmc_y2y_df,
    compute_basic_metrics,
    compute_landcover_decomposition_metrics,
    compute_landcover_y2y_metrics,
    compute_monthly_y2y_metrics,
    load_eval_context,
    prepare_lfmc_landcover_eval_df,
)

from paper_figure_plotting import (  # noqa: E402
    plot_landcover_comparison_panels,
    plot_landcover_metric_grouped,
    plot_monthly_variability_bars,
    plot_scatter_triptych,
    plot_stacked_timeseries_panels,
)


MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
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
CANONICAL_YEAR_START = 2000
_GMBA_BASIC_GDF = None
_GMBA_SITE_LABEL_CACHE = {}


def _metric_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.std(arr, ddof=0))


def _figure_output_paths(runtime: Dict[str, object], filename: str, stem: str) -> Tuple[str, str]:
    fig_path = os.path.join(runtime["figures_dir"], filename)
    table_path = os.path.join(runtime["tables_dir"], f"{stem}.csv")
    return fig_path, table_path


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _timeseries_cache_cfg(runtime: Dict[str, object]) -> Dict[str, object]:
    return runtime["cfg"].get("timeseries_cache", {})


def _model_cfg(cfg: Dict[str, object], model_key: str) -> Dict[str, object]:
    return cfg["models"][model_key]


def _model_outputs_root(model_cfg: Dict[str, object]) -> str:
    return str(model_cfg.get("outputs_root", ""))


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
        ]
    )
    token = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    cache[model_key] = token
    return token


def _model_ensemble_root(model_cfg: Dict[str, object]) -> Optional[str]:
    if bool(model_cfg.get("is_ensemble", False)):
        return str(model_cfg["outputs_root"])
    return None


def _load_named_eval_context(runtime: Dict[str, object], model_key: str) -> Dict[str, object]:
    if model_key in runtime["eval_contexts"]:
        return runtime["eval_contexts"][model_key]
    cfg = runtime["cfg"]
    model_cfg = _model_cfg(cfg, model_key)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        context = load_eval_context(
            model_dir=None if bool(model_cfg.get("is_ensemble", False)) else _model_outputs_root(model_cfg),
            ensemble_outputs_root=_model_ensemble_root(model_cfg),
            outputs_root=None if bool(model_cfg.get("is_ensemble", False)) else _model_outputs_root(model_cfg),
            ascending=True,
        )
    context["name"] = model_cfg["display_name"]
    context["paper_model_key"] = model_key
    runtime["eval_contexts"][model_key] = context
    return context


def _timeseries_model_entries(runtime: Dict[str, object], model_keys: Sequence[str]) -> List[Dict[str, object]]:
    cache_key = tuple(model_keys)
    if cache_key in runtime["timeseries_entries"]:
        return runtime["timeseries_entries"][cache_key]
    cfg = runtime["cfg"]
    model_configs = []
    for model_key in model_keys:
        model_cfg = _model_cfg(cfg, model_key)
        model_configs.append(
            {
                "paper_model_key": model_key,
                "name": model_cfg["display_name"],
                "outputs_root": model_cfg["outputs_root"],
                "input_data_name": model_cfg["input_data_name"],
                "model_type": "standard",
                "model_num_tasks": int(model_cfg.get("model_num_tasks", 3)),
                "is_ensemble": bool(model_cfg.get("is_ensemble", False)),
                "paper_color": model_cfg["color"],
            }
        )
    entries = build_model_entries(model_configs)
    for cfg_row, entry in zip(model_configs, entries):
        entry["paper_model_key"] = cfg_row["paper_model_key"]
        entry["paper_color"] = cfg_row["paper_color"]
    runtime["timeseries_entries"][cache_key] = entries
    return entries


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
    work = work.sort_values("dominant_landcover").reset_index(drop=True)
    return work


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
    split_lat = split_lookup.get(state_code, None)
    if split_lat is None:
        return state_name
    prefix = "Northern" if lat >= split_lat else "Southern"
    return f"{prefix} {state_name}"


def _coincident_obs_cache_path(runtime: Dict[str, object], model_key: str) -> str:
    token = _model_cache_token(runtime, model_key)
    return os.path.join(runtime["cache_dir"], "coincident_obs", f"{model_key}_{token}.csv.gz")


def _compute_pearson_metrics(x_vals: Sequence[float], y_vals: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return {"n": 0, "pearson_r": np.nan}
    if len(x) < 2:
        return {"n": int(len(x)), "pearson_r": np.nan}
    if np.allclose(np.std(x, ddof=0), 0.0) or np.allclose(np.std(y, ddof=0), 0.0):
        pearson_r = np.nan
    else:
        pearson_r = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": int(len(x)),
        "pearson_r": pearson_r,
    }


def _pearson_stats_text(metrics: Dict[str, float]) -> str:
    pearson_r = metrics.get("pearson_r", np.nan)
    n = metrics.get("n", 0)
    parts = [
        f"Pearson r = {pearson_r:.2f}" if np.isfinite(pearson_r) else "Pearson r = nan",
        f"N = {int(n)}",
    ]
    return "\n".join(parts)


def _gmba_display_name(row: pd.Series, state_value: Optional[str]) -> Optional[str]:
    for column in ["Name_EN", "AsciiName", "MapName"]:
        if column in row.index:
            value = row[column]
            if pd.notna(value) and str(value).strip() != "":
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
    gdf = gdf.to_crs("EPSG:4326")
    _GMBA_BASIC_GDF = gdf
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
        return np.array([]), np.array([])
    dt = pd.to_datetime(dates, errors="coerce")
    mask = dt.year.isin(list(selected_years))
    return dt[mask].to_numpy(), np.asarray(values)[mask]


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
        out.append(
            pd.Timestamp(
                year=target_year,
                month=month,
                day=day,
            )
        )
    return np.asarray(out, dtype="datetime64[ns]")


def _reindex_series_to_daily(
    dates,
    values,
    lower=None,
    upper=None,
):
    if dates is None or len(dates) == 0:
        empty = np.array([], dtype="datetime64[ns]")
        return empty, np.array([]), lower, upper
    dt = pd.to_datetime(dates, errors="coerce")
    valid_mask = dt.notna()
    dt = dt[valid_mask]
    work_dict = {
        "date": dt,
        "value": np.asarray(values, dtype=float)[valid_mask],
    }
    if len(dt) == 0:
        empty = np.array([], dtype="datetime64[ns]")
        return empty, np.array([]), lower, upper
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
    out_lower = None
    out_upper = None
    if "lower" in work.columns:
        out_lower = work["lower"].to_numpy(dtype=float)
    if "upper" in work.columns:
        out_upper = work["upper"].to_numpy(dtype=float)
    return out_dates, out_vals, out_lower, out_upper


def _normalize_inference_output(out: Dict[str, object]) -> Dict[str, np.ndarray]:
    normalized = {}
    for key in [
        "dates",
        "lfmc_pred",
        "lfmc_pred_std",
        "vv_pred",
        "vv_pred_std",
        "vh_pred",
        "vh_pred_std",
    ]:
        if key == "dates":
            normalized[key] = np.asarray(pd.to_datetime(out.get(key, []), errors="coerce"), dtype="datetime64[ns]")
        else:
            normalized[key] = np.asarray(out.get(key, []), dtype=float)
    return normalized


def _empty_inference_output() -> Dict[str, np.ndarray]:
    return {
        "dates": np.array([], dtype="datetime64[ns]"),
        "lfmc_pred": np.array([], dtype=float),
        "lfmc_pred_std": np.array([], dtype=float),
        "vv_pred": np.array([], dtype=float),
        "vv_pred_std": np.array([], dtype=float),
        "vh_pred": np.array([], dtype=float),
        "vh_pred_std": np.array([], dtype=float),
    }


def _timeseries_cache_site_token(site_key: str) -> str:
    return hashlib.sha1(str(site_key).encode("utf-8")).hexdigest()[:12]


def _timeseries_year_cache_path(
    runtime: Dict[str, object],
    model_key: str,
    site_key: str,
    year: int,
) -> str:
    model_dir = os.path.join(
        runtime["timeseries_cache_dir"],
        f"{model_key}_{_model_cache_token(runtime, model_key)}",
    )
    os.makedirs(model_dir, exist_ok=True)
    site_token = _timeseries_cache_site_token(site_key)
    return os.path.join(
        model_dir,
        f"{site_token}_{int(year)}.npz",
    )


def _save_timeseries_year_cache(
    cache_path: str,
    model_key: str,
    site_key: str,
    year: int,
    out: Dict[str, np.ndarray],
) -> None:
    payload = _normalize_inference_output(out)
    np.savez_compressed(
        cache_path,
        cache_version=np.asarray([1], dtype=int),
        model_key=np.asarray([str(model_key)]),
        site_key=np.asarray([str(site_key)]),
        year=np.asarray([int(year)], dtype=int),
        dates=payload["dates"],
        lfmc_pred=payload["lfmc_pred"],
        lfmc_pred_std=payload["lfmc_pred_std"],
        vv_pred=payload["vv_pred"],
        vv_pred_std=payload["vv_pred_std"],
        vh_pred=payload["vh_pred"],
        vh_pred_std=payload["vh_pred_std"],
    )


def _load_timeseries_year_cache(
    cache_path: str,
    model_key: str,
    site_key: str,
    year: int,
) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            cached_model_key = str(npz["model_key"][0])
            cached_site_key = str(npz["site_key"][0])
            cached_year = int(npz["year"][0])
            if cached_model_key != str(model_key):
                return None
            if cached_site_key != str(site_key):
                return None
            if cached_year != int(year):
                return None
            return {
                "dates": np.asarray(npz["dates"], dtype="datetime64[ns]"),
                "lfmc_pred": np.asarray(npz["lfmc_pred"], dtype=float),
                "lfmc_pred_std": np.asarray(npz["lfmc_pred_std"], dtype=float),
                "vv_pred": np.asarray(npz["vv_pred"], dtype=float),
                "vv_pred_std": np.asarray(npz["vv_pred_std"], dtype=float),
                "vh_pred": np.asarray(npz["vh_pred"], dtype=float),
                "vh_pred_std": np.asarray(npz["vh_pred_std"], dtype=float),
            }
    except Exception:
        return None


def _complete_figure_1_task(
    progress_state: Dict[str, object],
    site_key: str,
    task_key,
    elapsed: float,
    from_cache: bool,
) -> None:
    progress_state["completed_task_keys"].add(task_key)
    progress_state["durations"].append(float(max(elapsed, 0.0)))
    site_idx = progress_state["site_order"].get(site_key, 0)
    n_sites = progress_state["n_sites"]
    year = int(task_key[2])
    eta_str = _progress_eta_string(progress_state)
    suffix = " | cache hit" if from_cache else ""
    print(
        f"Figure 1 | site {site_idx}/{n_sites} | year {year} done in "
        f"{elapsed:.1f}s | ETA {eta_str}{suffix}"
    )


def _load_or_build_year_window_inference(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    year: int,
    figure_key: str,
) -> Dict[str, np.ndarray]:
    model_key = model_entry["paper_model_key"]
    task_key = (str(figure_key), str(site_key), int(year))
    progress_state = runtime.get("figure_1_progress")
    should_track = (
        figure_key == "figure_1"
        and progress_state is not None
        and bool(progress_state.get("enabled", False))
        and task_key in progress_state.get("task_keys", [])
    )
    already_done = should_track and task_key in progress_state["completed_task_keys"]
    cache_cfg = _timeseries_cache_cfg(runtime)
    use_cache = bool(cache_cfg.get("enabled", True))
    rebuild = bool(cache_cfg.get("rebuild", False))
    cache_path = _timeseries_year_cache_path(runtime, model_key, site_key, year)
    if use_cache and not rebuild:
        cached = _load_timeseries_year_cache(cache_path, model_key, site_key, year)
        if cached is not None:
            if should_track and not already_done:
                _complete_figure_1_task(
                    progress_state=progress_state,
                    site_key=site_key,
                    task_key=task_key,
                    elapsed=0.0,
                    from_cache=True,
                )
            return cached
    inputs_root = runtime["cfg"]["paths"]["inputs_root"]
    forward_batch_size = int(runtime["cfg"]["plotting"]["forward_batch_size"])
    out = _run_quiet_inference_with_progress(
        runtime=runtime,
        model_entry=model_entry,
        site_key=site_key,
        start_date=pd.Timestamp(int(year), 1, 1),
        end_date=pd.Timestamp(int(year), 12, 31),
        inputs_root=inputs_root,
        forward_batch_size=forward_batch_size,
        task_key=task_key,
    )
    out = _normalize_inference_output(out)
    if use_cache:
        _save_timeseries_year_cache(cache_path, model_key, site_key, year, out)
    return out


def _concat_inference_chunks(chunks: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    merged = {}
    empty = _empty_inference_output()
    for key in empty.keys():
        valid_parts = []
        for chunk in chunks:
            part = np.asarray(chunk.get(key, []))
            if len(part) > 0:
                valid_parts.append(part)
        if len(valid_parts) == 0:
            merged[key] = empty[key].copy()
        else:
            merged[key] = np.concatenate(valid_parts)
    return merged


def _slice_inference_output_to_window(
    out: Dict[str, np.ndarray],
    start_date,
    end_date,
) -> Dict[str, np.ndarray]:
    if len(out.get("dates", [])) == 0:
        return _empty_inference_output()
    dt = pd.to_datetime(out["dates"], errors="coerce")
    mask = dt.notna()
    if start_date is not None:
        mask = mask & (dt >= pd.Timestamp(start_date))
    if end_date is not None:
        mask = mask & (dt <= pd.Timestamp(end_date))
    sliced = {}
    for key, values in out.items():
        arr = np.asarray(values)
        sliced[key] = arr[mask]
    return sliced


def _collect_year_window_inference(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    selected_years: Sequence[int],
) -> Dict[str, np.ndarray]:
    chunk_outputs = []
    for year in selected_years:
        chunk_outputs.append(
            _load_or_build_year_window_inference(
                runtime=runtime,
                model_entry=model_entry,
                site_key=site_key,
                year=int(year),
                figure_key="figure_1",
            )
        )
    return _concat_inference_chunks(chunk_outputs)


def _collect_window_inference(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    start_date,
    end_date,
    figure_key: str,
) -> Dict[str, np.ndarray]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    years = list(range(int(start_ts.year), int(end_ts.year) + 1))
    chunk_outputs = [
        _load_or_build_year_window_inference(
            runtime=runtime,
            model_entry=model_entry,
            site_key=site_key,
            year=year,
            figure_key=figure_key,
        )
        for year in years
    ]
    merged = _concat_inference_chunks(chunk_outputs)
    return _slice_inference_output_to_window(merged, start_ts, end_ts)


def _init_figure_1_progress(
    runtime: Dict[str, object],
    anchor_entry: Dict[str, object],
    selected: Dict[str, List[str]],
    years_to_plot: int,
) -> None:
    site_rows = []
    criteria_order = ["good", "average", "poor"]
    for criterion in criteria_order:
        for site_key in selected.get(criterion, []):
            site_rows.append(site_key)
    task_keys = []
    site_order = {}
    site_years = {}
    for site_idx, site_key in enumerate(site_rows, start=1):
        site_order[site_key] = site_idx
        lfmc_dates = _to_naive_datetime(anchor_entry["site_error"][site_key]["dates"])
        selected_years = _top_consecutive_observation_years(lfmc_dates, years_to_plot)
        site_years[site_key] = selected_years
        for year in selected_years:
            task_keys.append(("figure_1", str(site_key), int(year)))
    runtime["figure_1_progress"] = {
        "enabled": True,
        "site_order": site_order,
        "site_years": site_years,
        "n_sites": len(site_rows),
        "task_keys": task_keys,
        "completed_task_keys": set(),
        "durations": [],
    }


def _progress_eta_string(progress_state: Dict[str, object]) -> str:
    completed = len(progress_state["completed_task_keys"])
    total = len(progress_state["task_keys"])
    remaining = max(total - completed, 0)
    durations = progress_state["durations"]
    if remaining == 0:
        return "0s"
    if len(durations) == 0:
        return "estimating"
    mean_seconds = float(np.mean(np.asarray(durations, dtype=float)))
    return _format_seconds(mean_seconds * remaining)


def _run_quiet_inference_with_progress(
    runtime: Dict[str, object],
    model_entry: Dict[str, object],
    site_key: str,
    start_date,
    end_date,
    inputs_root: str,
    forward_batch_size: int,
    task_key,
):
    progress_state = runtime.get("figure_1_progress")
    should_track = (
        progress_state is not None and
        bool(progress_state.get("enabled", False)) and
        task_key in progress_state.get("task_keys", [])
    )
    already_done = should_track and task_key in progress_state["completed_task_keys"]
    if should_track and not already_done:
        site_idx = progress_state["site_order"].get(site_key, 0)
        n_sites = progress_state["n_sites"]
        year = int(task_key[2])
        print(
            f"Figure 1 | site {site_idx}/{n_sites} | year {year} start"
        )
        t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
    if should_track and not already_done:
        elapsed = time.perf_counter() - t0
        _complete_figure_1_task(
            progress_state=progress_state,
            site_key=site_key,
            task_key=task_key,
            elapsed=elapsed,
            from_cache=False,
        )
    return out


def _overall_metric_std(context: Dict[str, object]) -> Dict[str, float]:
    member_eval_dfs = context["member_eval_dfs"]
    if member_eval_dfs is None or len(member_eval_dfs) == 0:
        return {"rmse": np.nan, "r2": np.nan}
    rmse_vals = []
    r2_vals = []
    for member_eval_df in member_eval_dfs:
        lfmc_df = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
        metrics = compute_basic_metrics(lfmc_df["obs"].values, lfmc_df["pred"].values)
        rmse_vals.append(metrics["rmse"])
        r2_vals.append(metrics["r2"])
    return {
        "rmse": _metric_std(rmse_vals),
        "r2": _metric_std(r2_vals),
    }


def _space_time_metric_stds(context: Dict[str, object]) -> Tuple[Dict[str, float], Dict[str, float]]:
    member_eval_dfs = context["member_eval_dfs"]
    if member_eval_dfs is None or len(member_eval_dfs) == 0:
        return {"rmse": np.nan, "r2": np.nan}, {"rmse": np.nan, "r2": np.nan}
    space_metrics = []
    time_metrics = []
    for member_eval_df in member_eval_dfs:
        member_lfmc_df = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
        site_summary_df, anomaly_df = build_lfmc_space_time_tables(member_lfmc_df)
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
    space_std = {
        "rmse": _metric_std([metric["rmse"] for metric in space_metrics]),
        "r2": _metric_std([metric["r2"] for metric in space_metrics]),
    }
    time_std = {
        "rmse": _metric_std([metric["rmse"] for metric in time_metrics]),
        "r2": _metric_std([metric["r2"] for metric in time_metrics]),
    }
    return space_std, time_std


def _monthly_source_centered_metric_std(
    context: Dict[str, object],
    min_obs: int,
    min_years: int,
) -> Dict[str, float]:
    member_eval_dfs = context["member_eval_dfs"]
    if member_eval_dfs is None or len(member_eval_dfs) == 0:
        return {"rmse": np.nan, "r2": np.nan}
    metric_rows = []
    for member_eval_df in member_eval_dfs:
        member_lfmc_df = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
        eval_df, valid_groups = build_site_month_anomaly_eval_df(
            lfmc_df=build_lfmc_y2y_df(member_lfmc_df),
            min_obs=min_obs,
            min_years=min_years,
        )
        if len(eval_df) == 0 or len(valid_groups) == 0:
            continue
        metric_rows.append(
            compute_basic_metrics(
                eval_df["obs_dev"].values,
                eval_df["pred_dev"].values,
            )
        )
    if len(metric_rows) == 0:
        return {"rmse": np.nan, "r2": np.nan}
    return {
        "rmse": _metric_std([metric["rmse"] for metric in metric_rows]),
        "r2": _metric_std([metric["r2"] for metric in metric_rows]),
    }


def _landcover_metric_table(
    runtime: Dict[str, object],
    model_key: str,
    min_obs: int,
    min_years: int,
) -> pd.DataFrame:
    cache_key = (model_key, min_obs, min_years)
    if cache_key in runtime["landcover_metric_tables"]:
        return runtime["landcover_metric_tables"][cache_key]
    context = _load_named_eval_context(runtime, model_key)
    cache_dir = os.path.join(runtime["cache_dir"], model_key)
    os.makedirs(cache_dir, exist_ok=True)
    lfmc_lc_df = prepare_lfmc_landcover_eval_df(context["eval_df"], cache_dir)
    site_lookup_path = os.path.join(cache_dir, "lfmc_site_landcover_lookup.csv")
    from eval_deep import build_site_landcover_lookup  # noqa: E402
    site_lookup_df = build_site_landcover_lookup(build_lfmc_y2y_df(context["eval_df"]), site_lookup_path)
    site_lookup_df = site_lookup_df.copy()
    site_lookup_df["site_key"] = site_lookup_df["site_key"].astype(str)

    def _attach_shared_landcover_lookup(member_eval_df: pd.DataFrame) -> pd.DataFrame:
        member_lfmc_df = build_lfmc_y2y_df(member_eval_df)
        if len(member_lfmc_df) == 0:
            return pd.DataFrame()
        member_lfmc_df = member_lfmc_df.copy()
        member_lfmc_df["site_key"] = member_lfmc_df["site_key"].astype(str)
        member_lfmc_df = member_lfmc_df.merge(
            site_lookup_df[["site_key", "dominant_landcover", "dominant_landcover_frac"]],
            on="site_key",
            how="left",
        )
        member_lfmc_df = member_lfmc_df[member_lfmc_df["dominant_landcover"].notna()].copy()
        if len(member_lfmc_df) == 0:
            return pd.DataFrame()
        member_lfmc_df["site_obs_mean"] = member_lfmc_df.groupby("site_key")["obs"].transform("mean")
        member_lfmc_df["site_pred_mean"] = member_lfmc_df.groupby("site_key")["pred"].transform("mean")
        member_lfmc_df["site_obs_anom"] = member_lfmc_df["obs"] - member_lfmc_df["site_obs_mean"]
        member_lfmc_df["site_pred_anom"] = member_lfmc_df["pred"] - member_lfmc_df["site_pred_mean"]
        grp_cols = ["site_key", "month"]
        member_lfmc_df["seasonal_obs_mean"] = member_lfmc_df.groupby(grp_cols)["obs"].transform("mean")
        member_lfmc_df["seasonal_pred_mean"] = member_lfmc_df.groupby(grp_cols)["pred"].transform("mean")
        member_lfmc_df["seasonal_obs_anom"] = member_lfmc_df["obs"] - member_lfmc_df["seasonal_obs_mean"]
        member_lfmc_df["seasonal_pred_anom"] = member_lfmc_df["pred"] - member_lfmc_df["seasonal_pred_mean"]
        return member_lfmc_df

    metric_df = compute_landcover_decomposition_metrics(lfmc_lc_df)
    member_eval_dfs = context.get("member_eval_dfs")
    if member_eval_dfs is not None and len(member_eval_dfs) > 0:
        member_metric_frames = []
        for member_eval_df in member_eval_dfs:
            member_lfmc_lc_df = _attach_shared_landcover_lookup(member_eval_df)
            if len(member_lfmc_lc_df) > 0:
                member_metric_frames.append(compute_landcover_decomposition_metrics(member_lfmc_lc_df))
        if len(member_metric_frames) > 0:
            for metric_name in ["overall_r2", "site_mean_r2", "site_anom_r2"]:
                std_lookup = {}
                for lc in metric_df["dominant_landcover"].tolist():
                    vals = []
                    for member_metric_df in member_metric_frames:
                        row = member_metric_df[member_metric_df["dominant_landcover"] == lc]
                        if len(row) == 0:
                            continue
                        vals.append(float(row.iloc[0][metric_name]))
                    std_lookup[lc] = _metric_std(vals)
                metric_df[f"{metric_name}_std"] = metric_df["dominant_landcover"].map(std_lookup)
    y2y_df, _, _ = compute_landcover_y2y_metrics(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=min_obs,
        min_years=min_years,
        plot_dir=cache_dir,
    )
    if len(y2y_df) > 0:
        metric_df = metric_df.merge(
            y2y_df[
                [
                    "dominant_landcover",
                    "n_groups",
                    "total_obs",
                ]
            ],
            on="dominant_landcover",
            how="left",
        )
    else:
        metric_df["n_groups"] = np.nan
        metric_df["total_obs"] = np.nan
    month_anom_df, valid_month_groups = build_site_month_anomaly_eval_df(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=min_obs,
        min_years=min_years,
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
        if member_eval_dfs is not None and len(member_eval_dfs) > 0:
            std_lookup = {}
            for lc in metric_df["dominant_landcover"].tolist():
                vals = []
                for member_eval_df in member_eval_dfs:
                    member_month_anom_df, member_valid_groups = build_site_month_anomaly_eval_df(
                        lfmc_df=build_lfmc_y2y_df(member_eval_df),
                        min_obs=min_obs,
                        min_years=min_years,
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
    metric_df = _filtered_landcover_df(metric_df, runtime["cfg"])
    runtime["landcover_metric_tables"][cache_key] = metric_df
    return metric_df


def _training_fraction_table(runtime: Dict[str, object], model_key: str) -> pd.DataFrame:
    if model_key in runtime["training_fraction_tables"]:
        return runtime["training_fraction_tables"][model_key]
    context = _load_named_eval_context(runtime, model_key)
    cache_dir = os.path.join(runtime["cache_dir"], model_key)
    os.makedirs(cache_dir, exist_ok=True)
    summary_df = _prepare_train_landcover_fraction_summary(context, cache_dir)
    summary_df = _filtered_landcover_df(summary_df, runtime["cfg"])
    runtime["training_fraction_tables"][model_key] = summary_df
    return summary_df


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


def _site_has_same_year_lfmc_sar_overlap(
    runtime: Dict[str, object],
    anchor_entry: Dict[str, object],
    model_entries: Sequence[Dict[str, object]],
    site_key: str,
    years_to_plot: int,
) -> bool:
    vhvv_entry = find_vhvv_entry(model_entries)
    if vhvv_entry is None:
        return False
    lfmc_dates = pd.to_datetime(anchor_entry["site_error"][site_key]["dates"], errors="coerce")
    lfmc_dates = lfmc_dates[lfmc_dates.notna()]
    selected_years = _top_consecutive_observation_years(lfmc_dates, years_to_plot)
    if len(selected_years) == 0:
        return False
    selected_year_set = set(int(year) for year in selected_years)
    vv_vh_obs = get_vv_vh_site_series(
        vhvv_entry,
        site_key,
        runtime["vhvv_fold_cache"],
        start_date=None,
        end_date=None,
    )
    sar_years = set()
    for key in ["vv_dates", "vh_dates"]:
        dates = pd.to_datetime(vv_vh_obs.get(key, []), errors="coerce")
        if len(dates) == 0:
            continue
        sar_years.update(dates[dates.notna()].year.tolist())
    return len(selected_year_set.intersection(sar_years)) > 0


def _site_has_min_years_in_plot_window(
    anchor_entry: Dict[str, object],
    site_key: str,
    years_to_plot: int,
    min_years_with_obs: int = 2,
) -> bool:
    lfmc_dates = pd.to_datetime(anchor_entry["site_error"][site_key]["dates"], errors="coerce")
    lfmc_dates = lfmc_dates[lfmc_dates.notna()]
    if len(lfmc_dates) == 0:
        return False
    selected_years = _top_consecutive_observation_years(lfmc_dates, years_to_plot)
    if len(selected_years) == 0:
        return False
    observed_years = set(lfmc_dates.year.tolist())
    n_years_present = sum(1 for year in selected_years if year in observed_years)
    return n_years_present >= int(min_years_with_obs)


def _paper_select_sites_for_anchor(
    runtime: Dict[str, object],
    site_df: pd.DataFrame,
    n_sites: int,
    min_measurements: int,
    metric_col: str = "r2",
) -> Dict[str, List[str]]:
    out = {"good": [], "average": [], "poor": []}
    ranked = site_df.copy()
    ranked = ranked[ranked["num_measurements"] >= min_measurements]
    ranked = ranked[np.isfinite(ranked[metric_col])]
    if len(ranked) == 0:
        return out
    ranked = ranked.sort_values(metric_col, ascending=False).reset_index(drop=True)
    percentile_cfg = runtime["cfg"].get("timeseries_selection", {}).get("r2_percentiles", {})
    used_sites = set()
    out["good"] = _pick_percentile_sites(
        ranked,
        metric_col=metric_col,
        target_percentile=float(percentile_cfg.get("good", 90)),
        n_sites=n_sites,
        used_sites=used_sites,
    )
    out["average"] = _pick_percentile_sites(
        ranked,
        metric_col=metric_col,
        target_percentile=float(percentile_cfg.get("average", 50)),
        n_sites=n_sites,
        used_sites=used_sites,
    )
    out["poor"] = _pick_percentile_sites(
        ranked,
        metric_col=metric_col,
        target_percentile=float(percentile_cfg.get("poor", 10)),
        n_sites=n_sites,
        used_sites=used_sites,
    )
    return out


def _select_timeseries_sites(
    runtime: Dict[str, object],
    model_keys: Sequence[str],
    anchor_model_key: str,
    num_sites_per_criterion: int,
    min_measurements: int,
    require_sar_overlap_same_year: bool = False,
    years_to_plot: int = 3,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, List[str]]]:
    entries = _timeseries_model_entries(runtime, model_keys)
    site_sets = [set(entry["site_error"].keys()) for entry in entries]
    common_sites = set.intersection(*site_sets) if len(site_sets) > 1 else site_sets[0]
    anchor_entry = None
    for entry in entries:
        if entry["paper_model_key"] == anchor_model_key:
            anchor_entry = entry
            break
    if anchor_entry is None:
        raise KeyError(f"Anchor model '{anchor_model_key}' not found in timeseries entries")
    site_df = build_site_df(anchor_entry["site_error"], common_sites)
    site_df["r2"] = site_df["site"].map(
        lambda site: float(anchor_entry["site_error"][site].get("r2", np.nan))
    )
    keep_sites = [
        site for site in site_df["site"].tolist()
        if _site_has_min_years_in_plot_window(
            anchor_entry=anchor_entry,
            site_key=site,
            years_to_plot=years_to_plot,
            min_years_with_obs=2,
        )
    ]
    site_df = site_df[site_df["site"].isin(keep_sites)].reset_index(drop=True)
    if require_sar_overlap_same_year:
        keep_sites = [
            site for site in site_df["site"].tolist()
            if _site_has_same_year_lfmc_sar_overlap(
                runtime=runtime,
                anchor_entry=anchor_entry,
                model_entries=entries,
                site_key=site,
                years_to_plot=years_to_plot,
            )
        ]
        site_df = site_df[site_df["site"].isin(keep_sites)].reset_index(drop=True)
    selected = _paper_select_sites_for_anchor(
        runtime=runtime,
        site_df=site_df,
        n_sites=num_sites_per_criterion,
        min_measurements=min_measurements,
        metric_col="r2",
    )
    return entries, anchor_entry, selected


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


def _build_coincident_lfmc_sar_obs_df(
    runtime: Dict[str, object],
    model_key: str,
) -> pd.DataFrame:
    if model_key in runtime["coincident_obs_tables"]:
        return runtime["coincident_obs_tables"][model_key].copy()
    cache_path = _coincident_obs_cache_path(runtime, model_key)
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        runtime["coincident_obs_tables"][model_key] = cached
        return cached.copy()
    context = _load_named_eval_context(runtime, model_key)
    eval_df = context["eval_df"].copy()
    if len(eval_df) == 0:
        empty = pd.DataFrame(
            columns=["site_key", "latitude", "longitude", "date", "lfmc_obs", "vv_obs", "vh_obs"]
        )
        runtime["coincident_obs_tables"][model_key] = empty
        return empty.copy()
    eval_df["date"] = pd.to_datetime(eval_df["date"], errors="coerce").dt.normalize()
    eval_df = eval_df[eval_df["date"].notna()].copy()
    eval_df["site_key"] = eval_df.apply(
        lambda row: f"{float(row['latitude'])}_{float(row['longitude'])}",
        axis=1,
    )

    def _daily_obs(target_name: str, value_name: str) -> pd.DataFrame:
        target_df = eval_df[eval_df["target"] == target_name].copy()
        if len(target_df) == 0:
            return pd.DataFrame(columns=["site_key", "latitude", "longitude", "date", value_name])
        return (
            target_df.groupby(["site_key", "latitude", "longitude", "date"], as_index=False)
            .agg(**{value_name: ("obs", "mean")})
        )

    lfmc_daily = _daily_obs("lfmc", "lfmc_obs")
    vv_daily = _daily_obs("vv", "vv_obs")
    vh_daily = _daily_obs("vh", "vh_obs")
    coincident_df = lfmc_daily.merge(
        vv_daily[["site_key", "date", "vv_obs"]],
        on=["site_key", "date"],
        how="left",
    ).merge(
        vh_daily[["site_key", "date", "vh_obs"]],
        on=["site_key", "date"],
        how="left",
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    coincident_df.to_csv(cache_path, index=False, compression="gzip")
    runtime["coincident_obs_tables"][model_key] = coincident_df
    return coincident_df.copy()


def _build_timeseries_panel(
    runtime: Dict[str, object],
    site_key: str,
    criterion_label: str,
    model_entries: Sequence[Dict[str, object]],
    anchor_entry: Dict[str, object],
    plot_vv: bool,
    plot_vh: bool,
    include_model_keys: Optional[Sequence[str]] = None,
    top_n_obs_years: Optional[int] = None,
) -> Dict[str, object]:
    cfg = runtime["cfg"]
    figure_cfg = cfg["plotting"]
    inputs_root = cfg["paths"]["inputs_root"]
    anchor_site = anchor_entry["site_error"][site_key]
    lfmc_dates = _to_naive_datetime(anchor_site["dates"])
    lfmc_values = np.asarray(anchor_site["true_values"], dtype=float)
    vhvv_entry = find_vhvv_entry(model_entries) if (plot_vv or plot_vh) else None
    selected_years = None
    start_date = None
    end_date = None
    vv_vh_obs = None
    if top_n_obs_years is not None:
        selected_years = _top_consecutive_observation_years(
            lfmc_dates,
            int(top_n_obs_years),
        )
        lfmc_dates, lfmc_values = _filter_series_to_years(
            lfmc_dates,
            lfmc_values,
            selected_years,
        )
        lfmc_dates = _canonicalize_dates_to_year_slots(lfmc_dates, selected_years)
        lfmc_dates, lfmc_values, _, _ = _reindex_series_to_daily(
            lfmc_dates,
            lfmc_values,
        )
    else:
        if vhvv_entry is not None and (plot_vv or plot_vh):
            vv_vh_obs = get_vv_vh_site_series(
                vhvv_entry,
                site_key,
                runtime["vhvv_fold_cache"],
                start_date=None,
                end_date=None,
            )
        start_date, end_date = _combined_obs_window(
            lfmc_dates,
            vv_vh_obs,
            padding_days=int(runtime["current_timeseries_padding_days"]),
        )
        lfmc_dates, lfmc_values = _slice_series_to_window(
            lfmc_dates,
            lfmc_values,
            start_date,
            end_date,
        )
    panel_series = []
    right_series = []
    if include_model_keys is None:
        filtered_entries = list(model_entries)
    else:
        filtered_entries = [
            entry for entry in model_entries
            if entry["paper_model_key"] in include_model_keys
        ]
    for entry in filtered_entries:
        if selected_years is not None:
            infer_out = _collect_year_window_inference(
                runtime=runtime,
                model_entry=entry,
                site_key=site_key,
                selected_years=selected_years,
            )
            infer_dates = _canonicalize_dates_to_year_slots(
                infer_out["dates"],
                selected_years,
            )
        else:
            infer_out = _collect_window_inference(
                runtime=runtime,
                model_entry=entry,
                site_key=site_key,
                start_date=start_date,
                end_date=end_date,
                figure_key="figure_5",
            )
            infer_dates = infer_out["dates"]
        panel_series.append(
            {
                "label": entry["name"],
                "dates": infer_dates,
                "values": infer_out["lfmc_pred"],
                "lower": (
                    infer_out["lfmc_pred"] - infer_out["lfmc_pred_std"]
                    if len(infer_out.get("lfmc_pred_std", [])) == len(infer_out["lfmc_pred"])
                    else None
                ),
                "upper": (
                    infer_out["lfmc_pred"] + infer_out["lfmc_pred_std"]
                    if len(infer_out.get("lfmc_pred_std", [])) == len(infer_out["lfmc_pred"])
                    else None
                ),
                "color": entry["paper_color"],
                "linewidth": 2.2,
                "linestyle": "-",
                "alpha": 0.95,
            }
        )
        panel_dates, panel_values, panel_lower, panel_upper = _reindex_series_to_daily(
            panel_series[-1]["dates"],
            panel_series[-1]["values"],
            lower=panel_series[-1]["lower"],
            upper=panel_series[-1]["upper"],
        )
        panel_series[-1]["dates"] = panel_dates
        panel_series[-1]["values"] = panel_values
        panel_series[-1]["lower"] = panel_lower
        panel_series[-1]["upper"] = panel_upper
    panel_series.append(
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
    )
    if vhvv_entry is not None and (plot_vv or plot_vh):
        if vv_vh_obs is None:
            vv_vh_obs = get_vv_vh_site_series(
                vhvv_entry,
                site_key,
                runtime["vhvv_fold_cache"],
                start_date=None,
                end_date=None,
            )
        if selected_years is not None:
            vhvv_infer = _collect_year_window_inference(
                runtime=runtime,
                model_entry=vhvv_entry,
                site_key=site_key,
                selected_years=selected_years,
            )
            vv_infer_dates = _canonicalize_dates_to_year_slots(
                vhvv_infer["dates"],
                selected_years,
            )
            vh_infer_dates = vv_infer_dates
        else:
            vhvv_infer = _collect_window_inference(
                runtime=runtime,
                model_entry=vhvv_entry,
                site_key=site_key,
                start_date=start_date,
                end_date=end_date,
                figure_key="figure_1",
            )
            vv_infer_dates = vhvv_infer["dates"]
            vh_infer_dates = vhvv_infer["dates"]
        if plot_vv and vv_vh_obs is not None:
            if selected_years is not None:
                vv_dates, vv_obs_vals = _filter_series_to_years(
                    vv_vh_obs["vv_dates"],
                    vv_vh_obs["vv_true"],
                    selected_years,
                )
                vv_dates = _canonicalize_dates_to_year_slots(vv_dates, selected_years)
            else:
                vv_dates, vv_obs_vals = _slice_series_to_window(
                    vv_vh_obs["vv_dates"],
                    vv_vh_obs["vv_true"],
                    start_date,
                    end_date,
                )
            right_series.append(
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
                right_series[-1]["dates"],
                right_series[-1]["values"],
            )
            right_series[-1]["dates"] = vv_dates_daily
            right_series[-1]["values"] = vv_vals_daily
            if len(vhvv_infer.get("vv_pred", [])) > 0:
                right_series.append(
                    {
                        "label": "Predicted VV",
                        "dates": vv_infer_dates,
                        "values": vhvv_infer["vv_pred"],
                        "lower": (
                            vhvv_infer["vv_pred"] - vhvv_infer["vv_pred_std"]
                            if len(vhvv_infer.get("vv_pred_std", [])) == len(vhvv_infer["vv_pred"])
                            else None
                        ),
                        "upper": (
                            vhvv_infer["vv_pred"] + vhvv_infer["vv_pred_std"]
                            if len(vhvv_infer.get("vv_pred_std", [])) == len(vhvv_infer["vv_pred"])
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
                    right_series[-1]["dates"],
                    right_series[-1]["values"],
                    lower=right_series[-1]["lower"],
                    upper=right_series[-1]["upper"],
                )
                right_series[-1]["dates"] = vv_dates_daily
                right_series[-1]["values"] = vv_vals_daily
                right_series[-1]["lower"] = vv_lower_daily
                right_series[-1]["upper"] = vv_upper_daily
        if plot_vh and vv_vh_obs is not None:
            if selected_years is not None:
                vh_dates, vh_obs_vals = _filter_series_to_years(
                    vv_vh_obs["vh_dates"],
                    vv_vh_obs["vh_true"],
                    selected_years,
                )
                vh_dates = _canonicalize_dates_to_year_slots(vh_dates, selected_years)
            else:
                vh_dates, vh_obs_vals = _slice_series_to_window(
                    vv_vh_obs["vh_dates"],
                    vv_vh_obs["vh_true"],
                    start_date,
                    end_date,
                )
            right_series.append(
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
                right_series[-1]["dates"],
                right_series[-1]["values"],
            )
            right_series[-1]["dates"] = vh_dates_daily
            right_series[-1]["values"] = vh_vals_daily
            if len(vhvv_infer.get("vh_pred", [])) > 0:
                right_series.append(
                    {
                        "label": "Predicted VH",
                        "dates": vh_infer_dates,
                        "values": vhvv_infer["vh_pred"],
                        "lower": (
                            vhvv_infer["vh_pred"] - vhvv_infer["vh_pred_std"]
                            if len(vhvv_infer.get("vh_pred_std", [])) == len(vhvv_infer["vh_pred"])
                            else None
                        ),
                        "upper": (
                            vhvv_infer["vh_pred"] + vhvv_infer["vh_pred_std"]
                            if len(vhvv_infer.get("vh_pred_std", [])) == len(vhvv_infer["vh_pred"])
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
                    right_series[-1]["dates"],
                    right_series[-1]["values"],
                    lower=right_series[-1]["lower"],
                    upper=right_series[-1]["upper"],
                )
                right_series[-1]["dates"] = vh_dates_daily
                right_series[-1]["values"] = vh_vals_daily
                right_series[-1]["lower"] = vh_lower_daily
                right_series[-1]["upper"] = vh_upper_daily
    criterion_key = str(criterion_label).strip().lower()
    percentile_lookup = cfg.get("timeseries_selection", {}).get("r2_percentiles", {})
    pct_value = percentile_lookup.get(criterion_key, np.nan)
    site_r2 = float(anchor_site.get("r2", np.nan))
    annotation_lines = []
    if np.isfinite(pd.to_numeric(pct_value, errors="coerce")):
        annotation_lines.append(f"{int(round(float(pct_value)))}th percentile site by R²")
    if np.isfinite(site_r2):
        annotation_lines.append(f"Site R²: {site_r2:.2f}")
    else:
        annotation_lines.append("Site R²: nan")
    if selected_years is not None and len(selected_years) > 0:
        annotation_lines.append(f"Years shown: {selected_years[0]}-{selected_years[-1]}")
    for series in panel_series:
        series.setdefault("legend_group", "predictions" if series.get("linewidth", 0.0) > 0 else "observations")
        series.setdefault("axis_group", "lfmc")
    return {
        "title": _site_panel_title(cfg, site_key),
        "annotation_text": "\n".join(annotation_lines),
        "series": panel_series,
        "right_series": right_series,
        "ylabel": "LFMC (%)",
        "right_ylabel": "VV / VH (dB)",
        "use_month_aligned_axis": bool(selected_years is not None),
        "timeseries_mode": "banded_sar" if len(right_series) > 0 else "lfmc_only",
    }


def _lfmc_only_panel(panel: Dict[str, object]) -> Dict[str, object]:
    out = dict(panel)
    out["series"] = list(panel.get("series", []))
    out["right_series"] = []
    out["ylabel"] = "LFMC (%)"
    out["timeseries_mode"] = "lfmc_only"
    return out


def _sar_only_panel(panel: Dict[str, object]) -> Dict[str, object]:
    out = dict(panel)
    out["series"] = list(panel.get("right_series", []) or [])
    out["right_series"] = []
    out["ylabel"] = "VV / VH (dB)"
    out["timeseries_mode"] = "sar_only"
    return out


def _select_shared_figure_1_sites(
    runtime: Dict[str, object],
    model_key: str,
    fig_cfg: Dict[str, object],
    require_sar_overlap_same_year: bool = False,
):
    model_entries, anchor_entry, selected = _select_timeseries_sites(
        runtime,
        model_keys=[model_key],
        anchor_model_key=model_key,
        num_sites_per_criterion=int(fig_cfg["num_sites_per_criterion"]),
        min_measurements=int(fig_cfg["min_measurements"]),
        require_sar_overlap_same_year=require_sar_overlap_same_year,
        years_to_plot=int(fig_cfg.get("years_to_plot", 3)),
    )
    _init_figure_1_progress(
        runtime=runtime,
        anchor_entry=anchor_entry,
        selected=selected,
        years_to_plot=int(fig_cfg.get("years_to_plot", 3)),
    )
    runtime["current_timeseries_padding_days"] = int(fig_cfg["padding_days"])
    return model_entries, anchor_entry, selected


def _build_shared_figure_1_panels(
    runtime: Dict[str, object],
    fig_cfg: Dict[str, object],
    model_entries: Sequence[Dict[str, object]],
    anchor_entry: Dict[str, object],
    selected: Dict[str, List[str]],
):
    panels = []
    for criterion in fig_cfg["criteria_order"]:
        site_list = selected.get(criterion, [])
        if len(site_list) == 0:
            continue
        panels.append(
            _build_timeseries_panel(
                runtime=runtime,
                site_key=site_list[0],
                criterion_label=str(criterion).capitalize(),
                model_entries=model_entries,
                anchor_entry=anchor_entry,
                plot_vv=bool(fig_cfg.get("plot_vv", False)),
                plot_vh=bool(fig_cfg.get("plot_vh", False)),
                include_model_keys=[fig_cfg["model_key"]],
                top_n_obs_years=int(fig_cfg.get("years_to_plot", 3)),
            )
        )
    return panels


def build_figure_1(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_1"]
    model_key = fig_cfg["model_key"]
    model_entries, anchor_entry, selected = _select_shared_figure_1_sites(
        runtime=runtime,
        model_key=model_key,
        fig_cfg=fig_cfg,
    )
    runtime["_figure_1_selected_sites"] = selected
    shared_panels = _build_shared_figure_1_panels(
        runtime=runtime,
        fig_cfg=fig_cfg,
        model_entries=model_entries,
        anchor_entry=anchor_entry,
        selected=selected,
    )
    runtime["_figure_1_shared_panels"] = shared_panels
    panels = [_lfmc_only_panel(panel) for panel in shared_panels]
    save_path, table_path = _figure_output_paths(
        runtime,
        filename=fig_cfg["filename"],
        stem="figure_02_sites",
    )
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
        fontsize=int(cfg["plotting"]["fontsize"]),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"]["dpi"]),
    )
    return save_path


def build_supplementary_figure_2(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_2"]
    model_key = fig_cfg["model_key"]
    model_entries, anchor_entry, selected = _select_shared_figure_1_sites(
        runtime=runtime,
        model_key=model_key,
        fig_cfg=fig_cfg,
        require_sar_overlap_same_year=True,
    )
    shared_panels = _build_shared_figure_1_panels(
        runtime=runtime,
        fig_cfg=fig_cfg,
        model_entries=model_entries,
        anchor_entry=anchor_entry,
        selected=selected,
    )
    panels = list(shared_panels)
    save_path, table_path = _figure_output_paths(
        runtime,
        filename=fig_cfg["filename"],
        stem="supplementary_figure_02_sites",
    )
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
        fontsize=int(cfg["plotting"]["fontsize"]),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"]["dpi"]),
    )
    return save_path


def build_figure_2(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_2"]
    min_obs = int(cfg["variability"]["min_obs"])
    min_years = int(cfg["variability"]["min_years"])
    context = _load_named_eval_context(runtime, fig_cfg["model_key"])
    lfmc_df = context["eval_df"][context["eval_df"]["target"] == "lfmc"].reset_index(drop=True)
    site_summary_df, anomaly_df = build_lfmc_space_time_tables(lfmc_df)
    month_anom_df, valid_month_groups = build_site_month_anomaly_eval_df(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=min_obs,
        min_years=min_years,
    )
    if len(month_anom_df) == 0 or len(valid_month_groups) == 0:
        raise ValueError("No source-centered monthly anomaly rows available for Figure 2")
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
    space_std, time_std = _space_time_metric_stds(context)
    month_anom_std = _monthly_source_centered_metric_std(context, min_obs=min_obs, min_years=min_years)
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
            "title": "Deviation from\nmonthly average",
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
    save_path, table_path = _figure_output_paths(
        runtime,
        filename=fig_cfg["filename"],
        stem="figure_03_site_tables",
    )
    combined_table = site_summary_df.merge(
        anomaly_df.groupby(["latitude", "longitude"], as_index=False).agg(
            n_anomaly_rows=("obs_anom", "size")
        ),
        on=["latitude", "longitude"],
        how="left",
    )
    if len(month_anom_df) > 0 and len(valid_month_groups) > 0:
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
        fontsize=int(cfg["plotting"]["fontsize"]),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"]["dpi"]),
    )
    return save_path


def build_supplementary_figure_1(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_1"]
    coincident_df = _build_coincident_lfmc_sar_obs_df(runtime, fig_cfg["model_key"])
    vv_df = coincident_df[np.isfinite(coincident_df["lfmc_obs"]) & np.isfinite(coincident_df["vv_obs"])].copy()
    vh_df = coincident_df[np.isfinite(coincident_df["lfmc_obs"]) & np.isfinite(coincident_df["vh_obs"])].copy()
    if len(vv_df) == 0 and len(vh_df) == 0:
        raise ValueError("No coincident LFMC/VV/VH observation rows available for Supplementary Figure 1")
    panels = []
    if len(vv_df) > 0:
        vv_metrics = _compute_pearson_metrics(vv_df["lfmc_obs"].values, vv_df["vv_obs"].values)
        panels.append(
            {
                "title": "LFMC vs VV",
                "kind": "hexbin",
                "x": vv_df["lfmc_obs"].values,
                "y": vv_df["vv_obs"].values,
                "xlabel": "Observed LFMC (%)",
                "ylabel": "Observed VV (dB)",
                "metrics": vv_metrics,
                "stats_text": _pearson_stats_text(vv_metrics),
                "draw_identity": False,
                "cmap": "YlOrBr",
                "cbar_label": "Count",
                "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
            }
        )
    if len(vh_df) > 0:
        vh_metrics = _compute_pearson_metrics(vh_df["lfmc_obs"].values, vh_df["vh_obs"].values)
        panels.append(
            {
                "title": "LFMC vs VH",
                "kind": "hexbin",
                "x": vh_df["lfmc_obs"].values,
                "y": vh_df["vh_obs"].values,
                "xlabel": "Observed LFMC (%)",
                "ylabel": "Observed VH (dB)",
                "metrics": vh_metrics,
                "stats_text": _pearson_stats_text(vh_metrics),
                "draw_identity": False,
                "cmap": "YlGn",
                "cbar_label": "Count",
                "gridsize": int(fig_cfg.get("hexbin_gridsize", 60)),
            }
        )
    save_path, table_path = _figure_output_paths(
        runtime,
        filename=fig_cfg["filename"],
        stem="supplementary_figure_01_coincident_obs",
    )
    coincident_df.to_csv(table_path, index=False)
    plot_scatter_triptych(
        panels=panels,
        save_path=save_path,
        fontsize=int(cfg["plotting"]["fontsize"]),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"]["dpi"]),
    )
    return save_path


def build_figure_3(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_3"]
    min_obs = int(cfg["variability"]["min_obs"])
    min_years = int(cfg["variability"]["min_years"])
    context = _load_named_eval_context(runtime, fig_cfg["model_key"])
    month_df = compute_monthly_y2y_metrics(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=min_obs,
        min_years=min_years,
    )
    if len(month_df) == 0:
        raise ValueError("No monthly variability rows available for Figure 3")
    month_df["month_label"] = [
        MONTH_LABELS[int(month) - 1] for month in month_df["month"].to_numpy(dtype=int)
    ]
    save_path, table_path = _figure_output_paths(
        runtime,
        filename=fig_cfg["filename"],
        stem="figure_04_monthly_variability",
    )
    month_df.to_csv(table_path, index=False)
    plot_monthly_variability_bars(
        month_df=month_df,
        save_path=save_path,
        fontsize=int(cfg["plotting"]["fontsize"]),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"]["dpi"]),
        bar_color=str(fig_cfg.get("bar_color", _model_cfg(cfg, fig_cfg["model_key"])["color"])),
    )
    return save_path


def build_figure_4(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_4"]
    min_obs = int(cfg["variability"]["min_obs"])
    min_years = int(cfg["variability"]["min_years"])
    metric_df = _landcover_metric_table(
        runtime=runtime,
        model_key=fig_cfg["model_key"],
        min_obs=min_obs,
        min_years=min_years,
    )
    if len(metric_df) == 0:
        raise ValueError("No landcover rows available for Figure 4")
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
            metric_df["overall_r2_std"].to_numpy(dtype=float)
            if "overall_r2_std" in metric_df.columns else np.full(len(metric_df), np.nan),
            metric_df["site_anom_r2_std"].to_numpy(dtype=float)
            if "site_anom_r2_std" in metric_df.columns else np.full(len(metric_df), np.nan),
            metric_df["site_mean_r2_std"].to_numpy(dtype=float)
            if "site_mean_r2_std" in metric_df.columns else np.full(len(metric_df), np.nan),
            metric_df["monthly_dev_r2_std"].to_numpy(dtype=float)
            if "monthly_dev_r2_std" in metric_df.columns else np.full(len(metric_df), np.nan),
        ]
    )
    save_path, table_path = _figure_output_paths(
        runtime,
        filename=fig_cfg["filename"],
        stem="figure_04_landcover_metrics",
    )
    metric_df.to_csv(table_path, index=False)
    plot_landcover_metric_grouped(
        categories=categories,
        metric_labels=["Overall", "Anomaly", "Site Mean", "Deviation from monthly mean"],
        values=values,
        counts=counts,
        errors=errors,
        save_path=save_path,
        fontsize=int(cfg["plotting"]["fontsize"]),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"]["dpi"]),
        colors=[
            fig_cfg["metric_colors"][0],
            fig_cfg["metric_colors"][1],
            fig_cfg["metric_colors"][2],
            "#8d99ae",
        ],
    )
    return save_path


def build_supplementary_figure_3(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["supplementary_figure_3"]
    model_keys = fig_cfg["model_keys"]
    selected = runtime.get("_figure_1_selected_sites")
    if selected is None:
        figure_1_cfg = cfg["figures"]["figure_1"]
        _, _, selected = _select_shared_figure_1_sites(
            runtime=runtime,
            model_key=figure_1_cfg["model_key"],
            fig_cfg=figure_1_cfg,
            require_sar_overlap_same_year=False,
        )
        runtime["_figure_1_selected_sites"] = selected
    entries = _timeseries_model_entries(runtime, model_keys)
    anchor_entry = next(
        entry for entry in entries if entry["paper_model_key"] == fig_cfg["anchor_model_key"]
    )
    runtime["current_timeseries_padding_days"] = int(fig_cfg["padding_days"])
    panels = []
    for criterion in fig_cfg["criteria_order"]:
        site_list = selected.get(criterion, [])
        if len(site_list) == 0:
            continue
        panels.append(
            _build_timeseries_panel(
                runtime=runtime,
                site_key=site_list[0],
                criterion_label=str(criterion).capitalize(),
                model_entries=entries,
                anchor_entry=anchor_entry,
                plot_vv=False,
                plot_vh=False,
                include_model_keys=model_keys,
                top_n_obs_years=int(cfg["figures"]["figure_1"].get("years_to_plot", 3)),
            )
        )
    save_path, table_path = _figure_output_paths(
        runtime,
        filename=fig_cfg["filename"],
        stem="supplementary_figure_03_sites",
    )
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
        fontsize=int(cfg["plotting"]["fontsize"]),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"]["dpi"]),
    )
    return save_path


def build_figure_6(runtime: Dict[str, object]) -> str:
    cfg = runtime["cfg"]
    fig_cfg = cfg["figures"]["figure_6"]
    min_obs = int(cfg["variability"]["min_obs"])
    min_years = int(cfg["variability"]["min_years"])
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
        model_cfg = _model_cfg(cfg, model_key)
        model_labels.append(model_cfg["display_name"])
        if str(model_key) == "single_task":
            colors.append("#7ca596")
        else:
            colors.append(model_cfg["color"])
        metric_df = _landcover_metric_table(runtime, model_key, min_obs, min_years)
        context = _load_named_eval_context(runtime, model_key)
        lfmc_df = context["eval_df"][context["eval_df"]["target"] == "lfmc"].reset_index(drop=True)
        site_summary_df, anomaly_df = build_lfmc_space_time_tables(lfmc_df)
        month_anom_df, valid_month_groups = build_site_month_anomaly_eval_df(
            lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
            min_obs=min_obs,
            min_years=min_years,
        )
        overall_std = _overall_metric_std(context)
        mean_std, anomaly_std = _space_time_metric_stds(context)
        monthly_std = _monthly_source_centered_metric_std(context, min_obs=min_obs, min_years=min_years)
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

        overall_metric = compute_basic_metrics(lfmc_df["obs"].values, lfmc_df["pred"].values)
        anomaly_metric = compute_basic_metrics(anomaly_df["obs_anom"].values, anomaly_df["pred_anom"].values)
        mean_metric = compute_basic_metrics(site_summary_df["obs_mean"].values, site_summary_df["pred_mean"].values)
        monthly_metric = compute_basic_metrics(month_anom_df["obs_dev"].values, month_anom_df["pred_dev"].values)
        overall_row.append(float(overall_metric.get("r2", np.nan)))
        anomaly_row.append(float(anomaly_metric.get("r2", np.nan)))
        mean_row.append(float(mean_metric.get("r2", np.nan)))
        monthly_row.append(float(monthly_metric.get("r2", np.nan)))
        overall_err_row.append(float(overall_std.get("r2", np.nan)))
        anomaly_err_row.append(float(anomaly_std.get("r2", np.nan)))
        mean_err_row.append(float(mean_std.get("r2", np.nan)))
        monthly_err_row.append(float(monthly_std.get("r2", np.nan)))
        overall_count_row.append(float(overall_metric.get("n", np.nan)))
        anomaly_count_row.append(float(anomaly_metric.get("n", np.nan)))
        mean_count_row.append(float(mean_metric.get("n", np.nan)))
        monthly_count_row.append(float(monthly_metric.get("n", np.nan)))
        merged_rows.append(
            {
                "model_key": model_key,
                "display_name": model_cfg["display_name"],
                "dominant_landcover": "overall",
                "overall_r2": overall_metric.get("r2", np.nan),
                "site_anom_r2": anomaly_metric.get("r2", np.nan),
                "site_mean_r2": mean_metric.get("r2", np.nan),
                "monthly_dev_r2": monthly_metric.get("r2", np.nan),
                "overall_r2_std": overall_std.get("r2", np.nan),
                "site_anom_r2_std": anomaly_std.get("r2", np.nan),
                "site_mean_r2_std": mean_std.get("r2", np.nan),
                "monthly_dev_r2_std": monthly_std.get("r2", np.nan),
                "overall_n": overall_metric.get("n", np.nan),
                "site_anom_n": anomaly_metric.get("n", np.nan),
                "site_mean_n": mean_metric.get("n", np.nan),
                "monthly_dev_n": monthly_metric.get("n", np.nan),
            }
        )

        for category in landcover_categories:
            metric_row = metric_lookup.get(category, {})
            overall_row.append(float(metric_row.get("overall_r2", np.nan)))
            anomaly_row.append(float(metric_row.get("site_anom_r2", np.nan)))
            mean_row.append(float(metric_row.get("site_mean_r2", np.nan)))
            monthly_row.append(float(metric_row.get("monthly_dev_r2", np.nan)))
            overall_err_row.append(float(metric_row.get("overall_r2_std", np.nan)))
            anomaly_err_row.append(float(metric_row.get("site_anom_r2_std", np.nan)))
            mean_err_row.append(float(metric_row.get("site_mean_r2_std", np.nan)))
            monthly_err_row.append(float(metric_row.get("monthly_dev_r2_std", np.nan)))
            overall_count_row.append(float(metric_row.get("n_points", np.nan)))
            anomaly_count_row.append(float(metric_row.get("n_points", np.nan)))
            mean_count_row.append(float(metric_row.get("n_sites", np.nan)))
            monthly_count_row.append(float(metric_row.get("total_obs", np.nan)))
            merged_rows.append(
                {
                    "model_key": model_key,
                    "display_name": model_cfg["display_name"],
                    "dominant_landcover": category,
                    "overall_r2": metric_row.get("overall_r2", np.nan),
                    "site_anom_r2": metric_row.get("site_anom_r2", np.nan),
                    "site_mean_r2": metric_row.get("site_mean_r2", np.nan),
                    "monthly_dev_r2": metric_row.get("monthly_dev_r2", np.nan),
                    "overall_r2_std": metric_row.get("overall_r2_std", np.nan),
                    "site_anom_r2_std": metric_row.get("site_anom_r2_std", np.nan),
                    "site_mean_r2_std": metric_row.get("site_mean_r2_std", np.nan),
                    "monthly_dev_r2_std": metric_row.get("monthly_dev_r2_std", np.nan),
                    "overall_n": metric_row.get("n_points", np.nan),
                    "site_anom_n": metric_row.get("n_points", np.nan),
                    "site_mean_n": metric_row.get("n_sites", np.nan),
                    "monthly_dev_n": metric_row.get("total_obs", np.nan),
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
    save_path, table_path = _figure_output_paths(
        runtime,
        filename=fig_cfg["filename"],
        stem="figure_05_landcover_comparison",
    )
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
                "title": "Deviation from monthly mean",
                "ylabel": "R²",
                "values": np.asarray(monthly_values, dtype=float).T,
                "errors": np.asarray(monthly_errors, dtype=float).T,
                "counts": np.asarray(monthly_counts, dtype=float).T,
            },
        ],
        save_path=save_path,
        fontsize=int(cfg["plotting"]["fontsize"]),
        figsize=fig_cfg["figsize"],
        dpi=int(cfg["plotting"]["dpi"]),
    )
    return save_path


def init_runtime(cfg: Dict[str, object]) -> Dict[str, object]:
    output_root = cfg["paths"]["output_root"]
    figures_dir = os.path.join(output_root, "figures")
    tables_dir = os.path.join(output_root, "tables")
    cache_dir = os.path.join(output_root, "cache")
    timeseries_cache_cfg = cfg.get("timeseries_cache", {})
    timeseries_cache_dir = str(
        timeseries_cache_cfg.get("cache_root") or os.path.join(cache_dir, "timeseries")
    )
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(timeseries_cache_dir, exist_ok=True)
    return {
        "cfg": cfg,
        "output_root": output_root,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
        "cache_dir": cache_dir,
        "timeseries_cache_dir": timeseries_cache_dir,
        "eval_contexts": {},
        "timeseries_entries": {},
        "coincident_obs_tables": {},
        "landcover_metric_tables": {},
        "training_fraction_tables": {},
        "inference_cache": {},
        "tensor_cache": {},
        "runtime_cache": {},
        "vhvv_fold_cache": {},
        "current_timeseries_padding_days": 60,
        "model_cache_tokens": {},
    }


def build_enabled_figures(cfg: Dict[str, object], only_figures: Optional[Sequence[str]] = None) -> Dict[str, str]:
    runtime = init_runtime(cfg)
    figure_builders = {
        "figure_1": build_figure_1,
        "supplementary_figure_1": build_supplementary_figure_1,
        "supplementary_figure_2": build_supplementary_figure_2,
        "figure_2": build_figure_2,
        "figure_4": build_figure_4,
        "supplementary_figure_3": build_supplementary_figure_3,
        "figure_6": build_figure_6,
    }
    outputs = {}
    for fig_key, builder in figure_builders.items():
        fig_cfg = cfg["figures"].get(fig_key, {})
        if not bool(fig_cfg.get("enabled", False)):
            continue
        if only_figures is not None and fig_key not in only_figures:
            continue
        print(f"Building {fig_key} ...")
        outputs[fig_key] = builder(runtime)
        print(f"Wrote {fig_key}: {outputs[fig_key]}")
    return outputs
