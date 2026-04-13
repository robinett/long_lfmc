#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from pyproj import Transformer

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, "..", "..", "..")
sys.path.append(project_root)

from input_source_resolver import (
    default_source_registry_path,
    open_inference_datasets_from_resolution,
    resolve_inference_sources,
)
from lfmc_model.utils.plotting import plot_pixel_grid_timeseries
from map_runtime_utils import (
    ALLOWED_DOMINANT_LANDCOVER,
    DEFAULT_MODEL_TYPE,
    LANDCOVER_NODATA_CODE,
    OUTPUT_DOMINANT_LANDCOVER_NAME,
    OUTPUT_LANDCOVER_YEAR_NAME,
    OUTPUT_MEAN_NAME,
    OUTPUT_QUALITY_FLAG_NAME,
    OUTPUT_STD_NAME,
    QUALITY_FLAG_VALUES,
    _nearest_index,
    build_reference_tensor_payload,
    build_static_superset_runtime,
    convert_tensor_payload_to_runtime,
    densify_tile_predictions,
    finalize_running_ensemble_predictions,
    initialize_running_ensemble_predictions,
    load_ensemble_runtimes,
    load_runtime_forward_predictor,
    open_model_grid,
    resolve_common_runtime_window,
    run_runtime_forward_loaded,
    timestamped_message,
    update_running_ensemble_predictions,
)


DEFAULT_SITE_CSV = (
    "/scratch/users/trobinet/long_lfmc/final_lfmc/inference/"
    "sites_for_mitch/valid_psinet_sites.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/scratch/users/trobinet/long_lfmc/final_lfmc/inference/"
    "sites_for_mitch/pixel_grids_multisource_fusion_clim20"
)
DEFAULT_INPUTS_ROOT = "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs"
DEFAULT_GRID_PATH = (
    "/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4"
)
DEFAULT_MEMBER_NAME_PREFIX = "multisource_fusion_"
DEFAULT_OVERRIDE_START_DATE = "2001-01-01"
DEFAULT_OVERRIDE_END_DATE = "2024-12-31"
DEFAULT_MAX_PLOT_YEARS = 2


def _load_current_multitask_defaults() -> Dict[str, str]:
    config_path = os.path.join(project_root, "lfmc_model", "scripts", "current_model_family.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    multitask_cfg = cfg.get("current_model_family", {}).get("multitask", {})
    outputs_root = str(multitask_cfg.get("outputs_root", "")).strip()
    input_data_name = str(multitask_cfg.get("input_data_name", "")).strip()
    if outputs_root == "" or input_data_name == "":
        raise ValueError(
            f"Missing current multitask defaults in {config_path}"
        )
    return {
        "config_path": config_path,
        "outputs_root": outputs_root,
        "input_data_name": input_data_name,
    }


CURRENT_MULTITASK_DEFAULTS = _load_current_multitask_defaults()
DEFAULT_ENSEMBLE_ROOT = CURRENT_MULTITASK_DEFAULTS["outputs_root"]
DEFAULT_INPUT_DATA_NAME = CURRENT_MULTITASK_DEFAULTS["input_data_name"]


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run site-based 3x3 pixel-grid LFMC ensemble inference and write one "
            "NetCDF file per requested site/time window."
        )
    )
    parser.add_argument("--site_csv", type=str, default=DEFAULT_SITE_CSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ensemble_root", type=str, default=DEFAULT_ENSEMBLE_ROOT)
    parser.add_argument("--input_data_name", type=str, default=DEFAULT_INPUT_DATA_NAME)
    parser.add_argument("--inputs_root", type=str, default=DEFAULT_INPUTS_ROOT)
    parser.add_argument("--grid_path", type=str, default=DEFAULT_GRID_PATH)
    parser.add_argument(
        "--source_registry_path",
        type=str,
        default=default_source_registry_path(),
    )
    parser.add_argument("--product_tier", type=str, default="final")
    parser.add_argument("--fold", type=int, default=9998)
    parser.add_argument("--fallback_num_tasks", type=int, default=3)
    parser.add_argument("--max_members", type=int, default=None)
    parser.add_argument(
        "--member_name_prefix",
        type=str,
        default=DEFAULT_MEMBER_NAME_PREFIX,
        help="Restrict ensemble members to model directories with this prefix.",
    )
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--forward_batch_size", type=int, default=8192)
    parser.add_argument(
        "--override_start_date",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD start date applied to every site row before runtime clamping.",
    )
    parser.add_argument(
        "--override_end_date",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD end date applied to every site row before runtime clamping.",
    )
    parser.add_argument(
        "--max_period_2001_2024",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Override every site row to the broad 2001-01-01 through 2024-12-31 window "
            "before source/runtime clamping."
        ),
    )
    parser.add_argument(
        "--max_plot_years",
        type=float,
        default=DEFAULT_MAX_PLOT_YEARS,
        help="If the output span exceeds this many years, only plot the most recent slice.",
    )
    return parser.parse_args()


def _sanitize_token(value: object) -> str:
    token = str(value).strip()
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = token.strip("._-")
    return token or "site"


def _year_coord_to_int(value: object) -> int:
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return int(pd.Timestamp(value).year)
    return int(value)


def _normalize_site_dataframe(site_df: pd.DataFrame) -> pd.DataFrame:
    out = site_df.copy()
    unnamed_cols = [col for col in out.columns if str(col).startswith("Unnamed:")]
    if len(unnamed_cols) > 0:
        out = out.drop(columns=unnamed_cols)
    rename_map = {}
    if "latitude_wgs84" in out.columns and "lat" not in out.columns:
        rename_map["latitude_wgs84"] = "lat"
    if "longitude_wgs84" in out.columns and "lon" not in out.columns:
        rename_map["longitude_wgs84"] = "lon"
    if "min_date" in out.columns and "start_date" not in out.columns:
        rename_map["min_date"] = "start_date"
    if "max_date" in out.columns and "end_date" not in out.columns:
        rename_map["max_date"] = "end_date"
    if len(rename_map) > 0:
        out = out.rename(columns=rename_map)
    required_cols = {"lat", "lon", "start_date", "end_date"}
    missing_cols = sorted(required_cols - set(out.columns))
    if len(missing_cols) > 0:
        raise KeyError(f"site_csv is missing required columns after normalization: {missing_cols}")
    return out


def _resolve_site_date_overrides(args) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if args.max_period_2001_2024:
        return (
            pd.Timestamp(DEFAULT_OVERRIDE_START_DATE).normalize(),
            pd.Timestamp(DEFAULT_OVERRIDE_END_DATE).normalize(),
        )
    override_start = None
    override_end = None
    if args.override_start_date is not None:
        override_start = pd.Timestamp(args.override_start_date).normalize()
    if args.override_end_date is not None:
        override_end = pd.Timestamp(args.override_end_date).normalize()
    if (override_start is None) != (override_end is None):
        raise ValueError("override_start_date and override_end_date must be provided together")
    return override_start, override_end


def _slice_plot_payload(
    dates,
    mean_cube: np.ndarray,
    std_cube: np.ndarray,
    max_plot_years: float,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(dates) == 0 or max_plot_years <= 0:
        return dates, mean_cube, std_cube
    span_days = (dates[-1] - dates[0]).days
    if span_days <= int(round(max_plot_years * 365.25)):
        return dates, mean_cube, std_cube
    plot_start = dates[-1] - pd.Timedelta(days=int(round(max_plot_years * 365.25)))
    keep_mask = dates >= plot_start
    if not np.any(keep_mask):
        return dates, mean_cube, std_cube
    return dates[keep_mask], mean_cube[keep_mask, :, :], std_cube[keep_mask, :, :]


def _build_site_payload(
    model_grid: xr.Dataset,
    site_lat: float,
    site_lon: float,
) -> Dict[str, object]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x_coords = np.asarray(model_grid["x"].values, dtype=np.float64)
    y_coords = np.asarray(model_grid["y"].values, dtype=np.float64)
    lat_grid = np.asarray(model_grid["lat"].values, dtype=np.float64)
    lon_grid = np.asarray(model_grid["lon"].values, dtype=np.float64)
    site_x, site_y = transformer.transform(float(site_lon), float(site_lat))
    center_ix = _nearest_index(x_coords, site_x)
    center_iy = _nearest_index(y_coords, site_y)
    if (
        center_ix <= 0
        or center_ix >= x_coords.size - 1
        or center_iy <= 0
        or center_iy >= y_coords.size - 1
    ):
        raise ValueError(
            "Requested site is too close to the model-grid boundary for a full 3x3 "
            f"neighborhood: lat={site_lat}, lon={site_lon}"
        )

    x_idx_vals = np.arange(center_ix - 1, center_ix + 2, dtype=np.int32)
    y_idx_vals = np.arange(center_iy - 1, center_iy + 2, dtype=np.int32)
    yy, xx = np.meshgrid(y_idx_vals, x_idx_vals, indexing="ij")
    tile_payload = {
        "tile_name": np.asarray(f"site3x3_{center_ix}_{center_iy}"),
        "tile_ix": np.asarray(int(center_ix), dtype=np.int32),
        "tile_iy": np.asarray(int(center_iy), dtype=np.int32),
        "x0": np.asarray(int(center_ix - 1), dtype=np.int32),
        "x1": np.asarray(int(center_ix + 2), dtype=np.int32),
        "y0": np.asarray(int(center_iy - 1), dtype=np.int32),
        "y1": np.asarray(int(center_iy + 2), dtype=np.int32),
        "ix": xx.reshape(-1).astype(np.int32),
        "iy": yy.reshape(-1).astype(np.int32),
        "lat": lat_grid[yy, xx].reshape(-1).astype(np.float64),
        "lon": lon_grid[yy, xx].reshape(-1).astype(np.float64),
    }
    return {
        "tile_payload": tile_payload,
        "x_idx_vals": x_idx_vals,
        "y_idx_vals": y_idx_vals,
        "x_vals": x_coords[x_idx_vals].astype(np.float64),
        "y_vals": y_coords[y_idx_vals].astype(np.float64),
        "lat_vals": lat_grid[np.ix_(y_idx_vals, x_idx_vals)].astype(np.float64),
        "lon_vals": lon_grid[np.ix_(y_idx_vals, x_idx_vals)].astype(np.float64),
        "center_ix": int(center_ix),
        "center_iy": int(center_iy),
        "center_lat": float(lat_grid[center_iy, center_ix]),
        "center_lon": float(lon_grid[center_iy, center_ix]),
    }


def _subset_dominant_landcover(
    landcover_ds: xr.Dataset,
    x_idx_vals: np.ndarray,
    y_idx_vals: np.ndarray,
    output_years: Sequence[int],
) -> Dict[str, object]:
    landcover_var_names = [
        var_name
        for var_name in landcover_ds.data_vars
        if {"year", "y", "x"}.issubset(set(landcover_ds[var_name].dims))
    ]
    if len(landcover_var_names) == 0:
        raise ValueError("Landcover dataset is missing yearly landcover variables")

    year_coord_lookup = {
        _year_coord_to_int(coord_val): coord_val
        for coord_val in landcover_ds["year"].values
    }
    available_years = sorted(year_coord_lookup)
    code_to_name = {str(code): var_name for code, var_name in enumerate(landcover_var_names)}
    allowed_codes = {
        int(code)
        for code, var_name in enumerate(landcover_var_names)
        if var_name in ALLOWED_DOMINANT_LANDCOVER
    }

    dominant_codes = []
    output_year_to_source_year = {}
    for output_year in output_years:
        if output_year in year_coord_lookup:
            source_year = int(output_year)
        else:
            prior_years = [year for year in available_years if year <= output_year]
            if len(prior_years) > 0:
                source_year = int(prior_years[-1])
            else:
                source_year = int(available_years[0])
        output_year_to_source_year[str(int(output_year))] = int(source_year)
        year_coord_value = year_coord_lookup[source_year]
        y_slice = slice(int(y_idx_vals[0]), int(y_idx_vals[-1]) + 1)
        x_slice = slice(int(x_idx_vals[0]), int(x_idx_vals[-1]) + 1)
        best_code = np.full((len(y_idx_vals), len(x_idx_vals)), LANDCOVER_NODATA_CODE, dtype=np.uint8)
        best_score = np.full((len(y_idx_vals), len(x_idx_vals)), -np.inf, dtype=np.float32)
        any_valid = np.zeros((len(y_idx_vals), len(x_idx_vals)), dtype=bool)
        for code, var_name in enumerate(landcover_var_names):
            vals = np.asarray(
                landcover_ds[var_name]
                .sel(year=year_coord_value)
                .isel(y=y_slice, x=x_slice)
                .values,
                dtype=np.float32,
            )
            valid = np.isfinite(vals)
            better = valid & ((vals > best_score) | ~any_valid)
            best_code[better] = np.uint8(code)
            best_score[better] = vals[better]
            any_valid |= valid
        best_code[~any_valid] = LANDCOVER_NODATA_CODE
        dominant_codes.append(best_code)

    return {
        "output_years": np.asarray(output_years, dtype=np.int32),
        "dominant_landcover_code": np.stack(dominant_codes, axis=0),
        "code_to_name": code_to_name,
        "allowed_codes": sorted(allowed_codes),
        "output_year_to_source_year": output_year_to_source_year,
    }


def _build_site_dataset(
    site_row: pd.Series,
    site_payload: Dict[str, object],
    dense_payload: Dict[str, np.ndarray],
    landcover_meta: Dict[str, object],
    quality_flag_value: int,
    ensemble_member_count: int,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> xr.Dataset:
    output_years = np.asarray(landcover_meta["output_years"], dtype=np.int32)
    dense_mean = np.asarray(dense_payload[OUTPUT_MEAN_NAME], dtype=np.float32).copy()
    dense_std = np.asarray(dense_payload[OUTPUT_STD_NAME], dtype=np.float32).copy()
    dates = pd.DatetimeIndex(pd.to_datetime(dense_payload["dates"]))
    date_years = dates.year.to_numpy(dtype=np.int32)
    approved_codes = set(int(code) for code in landcover_meta["allowed_codes"])
    dominant_codes = np.asarray(landcover_meta["dominant_landcover_code"], dtype=np.uint8)
    quality_flag = np.full(
        (len(dates), len(site_payload["y_vals"]), len(site_payload["x_vals"])),
        np.uint8(quality_flag_value),
        dtype=np.uint8,
    )
    for year_idx, year_val in enumerate(output_years):
        time_mask = date_years == int(year_val)
        if not np.any(time_mask):
            continue
        forced_mask = (
            dominant_codes[year_idx] != LANDCOVER_NODATA_CODE
        ) & ~np.isin(dominant_codes[year_idx], np.asarray(sorted(approved_codes), dtype=np.uint8))
        quality_flag[time_mask, :, :] = np.where(
            forced_mask[None, :, :],
            np.uint8(2),
            quality_flag[time_mask, :, :],
        )

    ds = xr.Dataset(
        data_vars={
            OUTPUT_MEAN_NAME: (("time", "y", "x"), dense_mean),
            OUTPUT_STD_NAME: (("time", "y", "x"), dense_std),
            OUTPUT_QUALITY_FLAG_NAME: (("time", "y", "x"), quality_flag),
            OUTPUT_DOMINANT_LANDCOVER_NAME: (
                (OUTPUT_LANDCOVER_YEAR_NAME, "y", "x"),
                dominant_codes,
            ),
        },
        coords={
            "time": dates,
            "x": np.asarray(site_payload["x_vals"], dtype=np.float64),
            "y": np.asarray(site_payload["y_vals"], dtype=np.float64),
            OUTPUT_LANDCOVER_YEAR_NAME: output_years,
            "lat": (("y", "x"), np.asarray(site_payload["lat_vals"], dtype=np.float64)),
            "lon": (("y", "x"), np.asarray(site_payload["lon_vals"], dtype=np.float64)),
        },
        attrs={
            "description": (
                "3x3 pixel-grid LFMC ensemble inference centered on the nearest 500 m "
                "model-grid pixel to the requested site."
            ),
            "site_label": str(site_row.get("dataset_name", f"row_{int(site_row.name):03d}")),
            "requested_site_lat": float(site_row["lat"]),
            "requested_site_lon": float(site_row["lon"]),
            "requested_start_date": str(requested_start.date()),
            "requested_end_date": str(requested_end.date()),
            "ensemble_member_count": int(ensemble_member_count),
            "quality_flag_key": json.dumps(
                {
                    str(int(QUALITY_FLAG_VALUES["final"])): "high-quality, final",
                    str(int(QUALITY_FLAG_VALUES["low_latency"])): "low-latency, preliminary",
                    "2": "forced prediction on unapproved land cover",
                },
                sort_keys=True,
            ),
            "dominant_landcover_code_key": json.dumps(
                dict(landcover_meta["code_to_name"]),
                sort_keys=True,
            ),
            "crs": "EPSG:5070",
        },
    )
    ds["x"].attrs["units"] = "m"
    ds["y"].attrs["units"] = "m"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds[OUTPUT_MEAN_NAME].attrs["long_name"] = "LFMC ensemble mean"
    ds[OUTPUT_MEAN_NAME].attrs["units"] = "percent"
    ds[OUTPUT_STD_NAME].attrs["long_name"] = "LFMC ensemble standard deviation"
    ds[OUTPUT_STD_NAME].attrs["units"] = "percent"
    ds[OUTPUT_QUALITY_FLAG_NAME].attrs["flag_values"] = [
        int(QUALITY_FLAG_VALUES["final"]),
        int(QUALITY_FLAG_VALUES["low_latency"]),
        2,
    ]
    ds[OUTPUT_QUALITY_FLAG_NAME].attrs["flag_meanings"] = (
        "final_high_quality low_latency_preliminary forced_unapproved_landcover"
    )
    ds[OUTPUT_DOMINANT_LANDCOVER_NAME].attrs["code_to_name"] = json.dumps(
        dict(landcover_meta["code_to_name"]),
        sort_keys=True,
    )
    ds[OUTPUT_DOMINANT_LANDCOVER_NAME].attrs["nodata_code"] = int(LANDCOVER_NODATA_CODE)
    ds[OUTPUT_DOMINANT_LANDCOVER_NAME].attrs["output_year_to_source_year"] = json.dumps(
        dict(landcover_meta["output_year_to_source_year"]),
        sort_keys=True,
    )
    return ds


def _site_output_path(output_dir: str, site_row: pd.Series) -> str:
    dataset_name = _sanitize_token(site_row.get("dataset_name", f"row_{int(site_row.name):03d}"))
    return os.path.join(output_dir, f"site_{int(site_row.name):03d}_{dataset_name}.nc4")


def _site_plot_path(output_dir: str, site_row: pd.Series) -> str:
    dataset_name = _sanitize_token(site_row.get("dataset_name", f"row_{int(site_row.name):03d}"))
    return os.path.join(output_dir, "plots", f"site_{int(site_row.name):03d}_{dataset_name}_timeseries.png")


def main():
    args = get_args()
    if args.product_tier not in QUALITY_FLAG_VALUES:
        raise ValueError(
            f"product_tier must be one of {sorted(QUALITY_FLAG_VALUES)}; got {args.product_tier!r}"
        )
    if args.forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be >= 1")
    if args.max_plot_years <= 0:
        raise ValueError("max_plot_years must be > 0")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    site_df = _normalize_site_dataframe(pd.read_csv(args.site_csv))
    site_df["start_date"] = pd.to_datetime(site_df["start_date"]).dt.normalize()
    site_df["end_date"] = pd.to_datetime(site_df["end_date"]).dt.normalize()
    override_start, override_end = _resolve_site_date_overrides(args)
    if override_start is not None and override_end is not None:
        site_df["start_date"] = override_start
        site_df["end_date"] = override_end
    if (site_df["end_date"] < site_df["start_date"]).any():
        raise ValueError("Each site row must have end_date >= start_date")

    global_start = pd.Timestamp(site_df["start_date"].min()).normalize()
    global_end = pd.Timestamp(site_df["end_date"].max()).normalize()
    global_years = sorted(
        {
            int(year)
            for year in range(global_start.year, global_end.year + 1)
        }
    )

    print(timestamped_message(f"[site_pixel_grids] reading site csv {args.site_csv}"))
    print(timestamped_message(f"[site_pixel_grids] rows={len(site_df)} output_dir={args.output_dir}"))
    print(
        timestamped_message(
            "[site_pixel_grids] using current multitask family from "
            f"{CURRENT_MULTITASK_DEFAULTS['config_path']}"
        )
    )
    print(
        timestamped_message(
            f"[site_pixel_grids] ensemble_root={args.ensemble_root} "
            f"input_data_name={args.input_data_name}"
        )
    )
    if override_start is not None and override_end is not None:
        print(
            timestamped_message(
                f"[site_pixel_grids] overriding all site windows to "
                f"{override_start.date()} through {override_end.date()}"
            )
        )
    source_resolution = resolve_inference_sources(
        registry_path=args.source_registry_path,
        product_tier=args.product_tier,
        requested_start_date=global_start,
        requested_end_date=global_end,
        output_years=global_years,
    )
    print(
        timestamped_message(
            "[site_pixel_grids] opening inference datasets via source registry "
            f"{source_resolution['registry_path']}"
        )
    )
    dss = open_inference_datasets_from_resolution(source_resolution)
    model_grid = open_model_grid(args.grid_path)

    member_dirs, runtimes = load_ensemble_runtimes(
        ensemble_root=args.ensemble_root,
        input_data_name=args.input_data_name,
        inputs_root=args.inputs_root,
        fold=args.fold,
        fallback_num_tasks=args.fallback_num_tasks,
        max_members=args.max_members,
        member_name_prefix=args.member_name_prefix,
    )
    print(
        timestamped_message(
            f"[site_pixel_grids] selected {len(member_dirs)} ensemble members "
            f"with prefix={args.member_name_prefix!r}"
        )
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(timestamped_message("[site_pixel_grids] detected GPU; running model forward on CUDA"))
    else:
        print(
            timestamped_message(
                "[site_pixel_grids] warning: no GPU detected; running model forward on CPU"
            )
        )
    reference_runtime = build_static_superset_runtime(runtimes[0], runtimes)
    total_sites = len(site_df)
    total_members = len(runtimes)
    for row_idx in range(total_sites):
        site_row = site_df.iloc[row_idx]
        site_label = str(site_row.get("dataset_name", f"row_{row_idx:03d}"))
        requested_start = pd.Timestamp(site_row["start_date"]).normalize()
        requested_end = pd.Timestamp(site_row["end_date"]).normalize()
        safe_start, safe_end = resolve_common_runtime_window(
            dss,
            runtimes,
            requested_start,
            requested_end,
        )
        if safe_start > safe_end:
            raise ValueError(
                f"Site row {row_idx} has no runnable overlap after source/runtime clamping: "
                f"{requested_start.date()} to {requested_end.date()}"
            )
        site_payload = _build_site_payload(
            model_grid=model_grid,
            site_lat=float(site_row["lat"]),
            site_lon=float(site_row["lon"]),
        )
        tile_payload = site_payload["tile_payload"]
        reference_payload = build_reference_tensor_payload(
            tile_payload=tile_payload,
            runtime=reference_runtime,
            dss=dss,
            start_date=safe_start,
            end_date=safe_end,
        )
        output_years = sorted(pd.date_range(safe_start, safe_end, freq="D").year.unique().tolist())
        landcover_meta = _subset_dominant_landcover(
            dss["landcover_frac"],
            x_idx_vals=np.asarray(site_payload["x_idx_vals"], dtype=np.int32),
            y_idx_vals=np.asarray(site_payload["y_idx_vals"], dtype=np.int32),
            output_years=output_years,
        )
        out_path = _site_output_path(args.output_dir, site_row)
        plot_path = _site_plot_path(args.output_dir, site_row)
        aggregator = initialize_running_ensemble_predictions(reference_payload["info_df"])
        print(
            timestamped_message(
                f"[site_pixel_grids] prepared row={row_idx} "
                f"site={site_label} "
                f"window={safe_start.date()} to {safe_end.date()} "
                f"output={out_path}"
            )
        )

        tensor_cache: Dict[Tuple[str, ...], Dict[str, object]] = {}
        for member_idx, runtime in enumerate(runtimes, start=1):
            member_name = os.path.basename(member_dirs[member_idx - 1])
            print(
                timestamped_message(
                    f"[site_pixel_grids] site {row_idx + 1}/{total_sites} {site_label} "
                    f"loading member {member_idx}/{total_members}: {member_name}"
                )
            )
            member_t0 = time.perf_counter()
            predictor = load_runtime_forward_predictor(
                runtime,
                model_type=args.model_type,
                device=device,
            )
            try:
                print(
                    timestamped_message(
                        f"[site_pixel_grids] site {row_idx + 1}/{total_sites} {site_label} "
                        f"member {member_idx}/{total_members}"
                    )
                )
                try:
                    tensor_payload = convert_tensor_payload_to_runtime(
                        reference_payload=reference_payload,
                        reference_runtime=reference_runtime,
                        runtime=runtime,
                        tensor_cache=tensor_cache,
                        site=f"{site_label}_{int(site_row.name):03d}",
                    )
                except ValueError:
                    tensor_payload = build_reference_tensor_payload(
                        tile_payload=site_payload["tile_payload"],
                        runtime=runtime,
                        dss=dss,
                        start_date=safe_start,
                        end_date=safe_end,
                    )
                pred_arrays = run_runtime_forward_loaded(
                    predictor=predictor,
                    tensor_payload=tensor_payload,
                    batch_size=args.forward_batch_size,
                    return_info_df=False,
                    use_cuda_autocast=True,
                )
                update_running_ensemble_predictions(
                    aggregator,
                    pred_arrays["lfmc_pred"],
                )
            finally:
                del predictor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(
                timestamped_message(
                    f"[site_pixel_grids] site {row_idx + 1}/{total_sites} {site_label} "
                    f"member {member_idx}/{total_members} complete in "
                    f"{time.perf_counter() - member_t0:.1f}s"
                )
            )

        agg_df = finalize_running_ensemble_predictions(aggregator)
        dense_payload = densify_tile_predictions(
            agg_df,
            site_payload["tile_payload"],
        )
        ds = _build_site_dataset(
            site_row=site_row,
            site_payload=site_payload,
            dense_payload=dense_payload,
            landcover_meta=landcover_meta,
            quality_flag_value=int(source_resolution["quality_flag_value"]),
            ensemble_member_count=len(member_dirs),
            requested_start=requested_start,
            requested_end=requested_end,
        )
        ds.to_netcdf(out_path)
        print(timestamped_message(f"[site_pixel_grids] wrote {out_path}"))
        plot_dates, plot_mean, plot_std = _slice_plot_payload(
            dates=dense_payload["dates"],
            mean_cube=ds[OUTPUT_MEAN_NAME].values,
            std_cube=ds[OUTPUT_STD_NAME].values,
            max_plot_years=float(args.max_plot_years),
        )
        plot_pixel_grid_timeseries(
            dates=plot_dates,
            mean_cube=plot_mean,
            std_cube=plot_std,
            save_path=plot_path,
            site_label=site_label,
            center_lat=site_payload["center_lat"],
            center_lon=site_payload["center_lon"],
            y_label="LFMC ensemble mean (%)",
        )
        print(timestamped_message(f"[site_pixel_grids] wrote {plot_path}"))


if __name__ == "__main__":
    main()
