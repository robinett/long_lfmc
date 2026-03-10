#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, "..", "..", "..")
sys.path.append(os.path.join(project_root, "lfmc_model", "utils"))

import plotting
from map_runtime_utils import (
    DEFAULT_MODEL_GRID_PATH,
    OUTPUT_MEAN_NAME,
    OUTPUT_STD_NAME,
    open_model_grid,
    select_measurement_rich_month,
    select_validation_sites_for_month,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Validate merged ensemble map outputs with map and timeseries plots."
    )
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--grid_path", type=str, default=DEFAULT_MODEL_GRID_PATH)
    parser.add_argument("--n_map_sites", type=int, default=3)
    parser.add_argument("--n_validation_sites", type=int, default=3)
    return parser.parse_args()


def _nearest_index(coords: np.ndarray, value: float) -> int:
    coords = np.asarray(coords)
    idx = int(np.searchsorted(coords, value))
    if idx <= 0:
        return 0
    if idx >= coords.size:
        return coords.size - 1
    left = coords[idx - 1]
    right = coords[idx]
    return idx - 1 if abs(value - left) <= abs(right - value) else idx


def _site_key_to_lat_lon(site_key: str):
    lat_str, lon_str = site_key.split("_")
    return float(lat_str), float(lon_str)


def _pick_map_day(ds: xr.Dataset, month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.Timestamp:
    date_index = pd.to_datetime(ds["time"].values)
    keep = (date_index >= month_start) & (date_index <= month_end)
    candidate_dates = date_index[keep]
    if len(candidate_dates) == 0:
        candidate_dates = date_index
    return pd.Timestamp(candidate_dates[len(candidate_dates) // 2]).normalize()


def _pick_sample_pixels(day_vals: np.ndarray, n_sites: int) -> np.ndarray:
    valid_idx = np.flatnonzero(np.isfinite(day_vals))
    if len(valid_idx) == 0:
        raise ValueError("No finite map values available to sample")
    if len(valid_idx) <= n_sites:
        return valid_idx
    positions = np.linspace(0, len(valid_idx) - 1, n_sites).astype(int)
    return valid_idx[positions]


def _extract_point_series(
    ds: xr.Dataset,
    y_idx: int,
    x_idx: int,
    model_grid: xr.Dataset,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(ds["time"].values),
            OUTPUT_MEAN_NAME: ds[OUTPUT_MEAN_NAME].isel(y=y_idx, x=x_idx).values.astype(np.float32),
            OUTPUT_STD_NAME: ds[OUTPUT_STD_NAME].isel(y=y_idx, x=x_idx).values.astype(np.float32),
            "lat": np.repeat(float(model_grid["lat"].values[y_idx, x_idx]), ds.sizes["time"]),
            "lon": np.repeat(float(model_grid["lon"].values[y_idx, x_idx]), ds.sizes["time"]),
        }
    )
    return out


def _load_site_error_for_validation(run_config, safe_start, safe_end):
    month_start, month_end, site_error = select_measurement_rich_month(
        run_config["ensemble_root"],
        safe_start,
        safe_end,
    )
    if run_config.get("validation_month") is not None:
        month_start = pd.Timestamp(run_config["validation_month"]["start_date"]).normalize()
        month_end = pd.Timestamp(run_config["validation_month"]["end_date"]).normalize()
    return month_start, month_end, site_error


def _write_validation_site_plots(
    ds: xr.Dataset,
    model_grid: xr.Dataset,
    site_error: dict,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    out_dir: str,
    n_sites: int,
):
    os.makedirs(out_dir, exist_ok=True)
    selected_sites = select_validation_sites_for_month(
        site_error,
        month_start,
        month_end,
        n_sites=n_sites,
    )
    x_coords = np.asarray(model_grid["x"].values, dtype=np.float64)
    y_coords = np.asarray(model_grid["y"].values, dtype=np.float64)
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    summary_rows = []
    for site_rec in selected_sites:
        site_key = site_rec["site_key"]
        lat, lon = _site_key_to_lat_lon(site_key)
        site_x, site_y = transformer.transform(lon, lat)
        x_idx = _nearest_index(x_coords, site_x)
        y_idx = _nearest_index(y_coords, site_y)
        infer_df = _extract_point_series(ds, y_idx=y_idx, x_idx=x_idx, model_grid=model_grid)
        infer_mask = (
            (infer_df["date"] >= month_start) &
            (infer_df["date"] <= month_end)
        )
        infer_dates = pd.to_datetime(infer_df.loc[infer_mask, "date"]).values
        infer_vals = infer_df.loc[infer_mask, OUTPUT_MEAN_NAME].to_numpy(dtype=float)
        measure_dates = np.asarray(site_rec["dates"])
        measure_vals = np.asarray(site_rec["true_values"], dtype=float)
        train_pred_vals = np.asarray(site_rec["predictions"], dtype=float)
        save_path = os.path.join(
            out_dir,
            f"site_compare_{lat:.4f}_{lon:.4f}.png",
        )
        plotting.plot_multiple_timeseries(
            dates=[
                measure_dates,
                measure_dates,
                infer_dates,
            ],
            vals=[
                measure_vals,
                train_pred_vals,
                infer_vals,
            ],
            labels=[
                "measurement",
                "saved_test_prediction",
                "inference_prediction",
            ],
            linestyles=["", "-", "-"],
            markers=["o", ".", None],
            save_path=save_path,
        )
        infer_on_obs = []
        obs_dates_ts = pd.to_datetime(measure_dates)
        infer_series = pd.Series(infer_vals, index=pd.to_datetime(infer_dates))
        for obs_date in obs_dates_ts:
            infer_on_obs.append(float(infer_series.get(obs_date, np.nan)))
        infer_on_obs = np.asarray(infer_on_obs, dtype=float)
        mae = float(np.nanmean(np.abs(infer_on_obs - train_pred_vals)))
        summary_rows.append(
            {
                "site_key": site_key,
                "fold": site_rec["fold"],
                "num_measurements_month": site_rec["num_measurements_month"],
                "mean_abs_diff_saved_vs_infer": mae,
                "y_idx": int(y_idx),
                "x_idx": int(x_idx),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "site_compare_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[validate_map_outputs] wrote measured-site summary to {summary_path}")


def main():
    args = get_args()
    run_config_path = os.path.join(args.run_root, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)
    out_zarr_path = run_config["out_zarr_path"]
    ds = xr.open_zarr(out_zarr_path, consolidated=False)
    model_grid = open_model_grid(args.grid_path)

    safe_start = pd.Timestamp(run_config["safe_start_date"]).normalize()
    safe_end = pd.Timestamp(run_config["safe_end_date"]).normalize()
    month_start, month_end, site_error = _load_site_error_for_validation(
        run_config,
        safe_start,
        safe_end,
    )
    print(
        f"[validate_map_outputs] validation window {month_start.date()} to {month_end.date()}"
    )

    plots_root = os.path.join(run_config["validation_dir"], "plots")
    os.makedirs(plots_root, exist_ok=True)

    map_day = _pick_map_day(ds, month_start, month_end)
    day_field = ds[OUTPUT_MEAN_NAME].sel(time=map_day).values
    valid_mask = np.isfinite(day_field)
    lons = model_grid["lon"].values[valid_mask]
    lats = model_grid["lat"].values[valid_mask]
    vals = day_field[valid_mask]
    map_path = os.path.join(plots_root, f"map_{map_day.date().isoformat()}.png")
    plotting.map_points(
        lons=lons,
        lats=lats,
        counts_per_point=np.ones(len(vals), dtype=float),
        colors=vals,
        cmap="viridis",
        colorbar_label=f"{OUTPUT_MEAN_NAME} (%)",
        save_path=map_path,
        cbar_lim=(float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))),
        stats_text=f"date={map_day.date()} n={len(vals):,}",
    )
    print(f"[validate_map_outputs] wrote map plot to {map_path}")

    sample_idx = _pick_sample_pixels(day_field.reshape(-1), n_sites=int(args.n_map_sites))
    y_idx, x_idx = np.unravel_index(sample_idx, day_field.shape)
    sample_frames = []
    for yi, xi in zip(y_idx.tolist(), x_idx.tolist()):
        sample_frames.append(_extract_point_series(ds, y_idx=yi, x_idx=xi, model_grid=model_grid))
    sample_df = pd.concat(sample_frames, ignore_index=True)
    sample_ts_dir = os.path.join(plots_root, "sample_locations")
    plotting.plot_timeseries_by_site(
        sample_df,
        sample_ts_dir,
        OUTPUT_MEAN_NAME,
        "LFMC Ensemble Mean (%)",
    )
    print(f"[validate_map_outputs] wrote sample-location timeseries to {sample_ts_dir}")

    measured_site_dir = os.path.join(plots_root, "measured_sites")
    _write_validation_site_plots(
        ds=ds,
        model_grid=model_grid,
        site_error=site_error,
        month_start=month_start,
        month_end=month_end,
        out_dir=measured_site_dir,
        n_sites=int(args.n_validation_sites),
    )


if __name__ == "__main__":
    main()
