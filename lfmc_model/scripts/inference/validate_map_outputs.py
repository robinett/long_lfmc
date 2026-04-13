#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import sys

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import xarray as xr
import zarr

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, "..", "..", "..")
sys.path.append(os.path.join(project_root, "lfmc_model", "utils"))

import plotting
from map_runtime_utils import (
    DEFAULT_MODEL_GRID_PATH,
    OUTPUT_MEAN_NAME,
    OUTPUT_STD_NAME,
    _nearest_index,
    open_model_grid,
    select_measurement_rich_month,
    select_validation_sites_for_month,
)

shared_plotting_path = os.path.join(project_root, "data_processing", "shared", "plotting.py")
shared_plotting_spec = importlib.util.spec_from_file_location(
    "shared_plotting",
    shared_plotting_path,
)
shared_plotting = importlib.util.module_from_spec(shared_plotting_spec)
assert shared_plotting_spec.loader is not None
shared_plotting_spec.loader.exec_module(shared_plotting)


MAP_EXTENT_WUS_LONLAT = [-125.5, -92.0, 24.0, 50.0]
LFMC_BROWN_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "lfmc_brown_green",
    ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"],
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Validate merged ensemble map outputs with map and timeseries plots."
    )
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--grid_path", type=str, default=None)
    parser.add_argument("--n_map_sites", type=int, default=3)
    parser.add_argument("--n_validation_sites", type=int, default=3)
    return parser.parse_args()

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


def _write_map_plot(
    da: xr.DataArray,
    save_path: str,
    title: str,
    cmap,
    cbar_label: str,
):
    shared_plotting.plot_from_xarray(
        load_type="da",
        type_obj=da,
        var=str(da.name),
        proj_in="EPSG:5070",
        proj_out="EPSG:5070",
        fname=save_path,
        cmap=cmap,
        extent=MAP_EXTENT_WUS_LONLAT,
        extent_crs="EPSG:4326",
        title=title,
        cbar_label=cbar_label,
    )
    print(f"[validate_map_outputs] wrote map plot to {save_path}")


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


def _normalize_dates_to_midnight(dates) -> np.ndarray:
    return pd.to_datetime(dates).normalize().values


def _write_prediction_timeseries_plot(
    infer_df: pd.DataFrame,
    save_path: str,
    measurement_dates=None,
    measurement_vals=None,
    train_pred_dates=None,
    train_pred_vals=None,
    train_pred_std=None,
):
    infer_dates = pd.to_datetime(infer_df["date"]).values
    infer_vals = infer_df[OUTPUT_MEAN_NAME].to_numpy(dtype=float)
    infer_std = infer_df[OUTPUT_STD_NAME].to_numpy(dtype=float)
    infer_lower = infer_vals - infer_std
    infer_upper = infer_vals + infer_std

    lfmc_dates = [infer_dates]
    lfmc_vals = [infer_vals]
    lfmc_labels = ["lfmc_infer"]
    lfmc_linestyles = ["-"]
    lfmc_markers = [None]
    lfmc_colors = ["tab:blue"]
    lfmc_lower_vals = [infer_lower]
    lfmc_upper_vals = [infer_upper]

    if train_pred_dates is not None and train_pred_vals is not None and len(train_pred_vals) > 0:
        train_pred_dates = _normalize_dates_to_midnight(train_pred_dates)
        lfmc_dates.append(train_pred_dates)
        lfmc_vals.append(np.asarray(train_pred_vals, dtype=float))
        lfmc_labels.append("lfmc_train_pred")
        lfmc_linestyles.append("")
        lfmc_markers.append(".")
        lfmc_colors.append("tab:orange")
        if train_pred_std is not None and len(train_pred_std) == len(train_pred_vals):
            train_pred_std = np.asarray(train_pred_std, dtype=float)
            train_pred_vals = np.asarray(train_pred_vals, dtype=float)
            lfmc_lower_vals.append(train_pred_vals - train_pred_std)
            lfmc_upper_vals.append(train_pred_vals + train_pred_std)
        else:
            lfmc_lower_vals.append(None)
            lfmc_upper_vals.append(None)

    if measurement_dates is not None and measurement_vals is not None and len(measurement_vals) > 0:
        lfmc_dates.append(_normalize_dates_to_midnight(measurement_dates))
        lfmc_vals.append(np.asarray(measurement_vals, dtype=float))
        lfmc_labels.append("lfmc_true")
        lfmc_linestyles.append("")
        lfmc_markers.append("o")
        lfmc_colors.append("black")
        lfmc_lower_vals.append(None)
        lfmc_upper_vals.append(None)

    plotting.plot_lfmc_with_vv_vh(
        lfmc_dates=lfmc_dates,
        lfmc_vals=lfmc_vals,
        lfmc_labels=lfmc_labels,
        lfmc_linestyles=lfmc_linestyles,
        lfmc_markers=lfmc_markers,
        lfmc_colors=lfmc_colors,
        save_path=save_path,
        lfmc_lower_vals=lfmc_lower_vals,
        lfmc_upper_vals=lfmc_upper_vals,
    )


def _load_site_error_for_validation(run_config, safe_start, safe_end):
    validation_split = str(
        run_config.get("validation_prediction_split") or "val"
    ).strip().lower()
    if validation_split not in {"val", "test"}:
        raise ValueError(
            "run_config validation_prediction_split must be either 'val' or 'test'"
        )
    month_start, month_end, site_error = select_measurement_rich_month(
        run_config["ensemble_root"],
        safe_start,
        safe_end,
        fold=int(run_config.get("fold", 9998)),
        split=validation_split,
        member_name_prefix=run_config.get("ensemble_member_name_prefix"),
        selection_key=run_config.get("ensemble_selection_key"),
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
    selected_site_keys=None,
):
    os.makedirs(out_dir, exist_ok=True)
    if selected_site_keys is not None and len(selected_site_keys) > 0:
        selected_sites = []
        missing_site_keys = []
        for site_key in selected_site_keys:
            if site_key not in site_error:
                missing_site_keys.append(site_key)
                continue
            site_data = site_error[site_key]
            dates = pd.to_datetime(site_data["dates"]).normalize()
            keep = (dates >= month_start) & (dates <= month_end)
            if not np.any(keep):
                continue
            pred_std_here = None
            if "prediction_std" in site_data:
                pred_std_here = np.asarray(site_data["prediction_std"], dtype=float)[keep]
            selected_sites.append(
                {
                    "site_key": site_key,
                    "fold": str(site_data["fold"]),
                    "num_measurements_month": int(np.sum(keep)),
                    "dates": dates[keep],
                    "true_values": np.asarray(site_data["true_values"], dtype=float)[keep],
                    "predictions": np.asarray(site_data["predictions"], dtype=float)[keep],
                    "prediction_std": pred_std_here,
                }
            )
        if len(missing_site_keys) > 0:
            print(
                f"[validate_map_outputs] warning: {len(missing_site_keys)} stored validation "
                f"sites were not found in the current site-error payload"
            )
    else:
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
        infer_std = infer_df.loc[infer_mask, OUTPUT_STD_NAME].to_numpy(dtype=float)
        measure_dates = np.asarray(site_rec["dates"])
        measure_vals = np.asarray(site_rec["true_values"], dtype=float)
        train_pred_vals = np.asarray(site_rec["predictions"], dtype=float)
        train_pred_std = site_rec.get("prediction_std")
        save_path = os.path.join(
            out_dir,
            f"site_compare_{lat:.4f}_{lon:.4f}.png",
        )
        _write_prediction_timeseries_plot(
            infer_df=infer_df.loc[infer_mask].reset_index(drop=True),
            save_path=save_path,
            measurement_dates=measure_dates,
            measurement_vals=measure_vals,
            train_pred_dates=measure_dates,
            train_pred_vals=train_pred_vals,
            train_pred_std=train_pred_std,
        )
        infer_on_obs = []
        obs_dates_ts = pd.to_datetime(measure_dates).normalize()
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


def _open_merged_dataset(out_zarr_path: str) -> xr.Dataset:
    try:
        return xr.open_zarr(out_zarr_path, consolidated=False)
    except KeyError as exc:
        if "dimension_names" not in str(exc):
            raise
        print(
            "[validate_map_outputs] xarray.open_zarr could not infer dimensions; "
            "falling back to direct zarr reads"
        )
        root = zarr.open_group(out_zarr_path, mode="r")
        time_vals = np.asarray(root["time"][:], dtype=np.int64).astype("datetime64[ns]")
        y_vals = np.asarray(root["y"][:])
        x_vals = np.asarray(root["x"][:])
        ds = xr.Dataset(
            data_vars={
                OUTPUT_MEAN_NAME: (("time", "y", "x"), root[OUTPUT_MEAN_NAME]),
                OUTPUT_STD_NAME: (("time", "y", "x"), root[OUTPUT_STD_NAME]),
            },
            coords={
                "time": pd.to_datetime(time_vals),
                "y": y_vals,
                "x": x_vals,
            },
        )
        return ds


def main():
    args = get_args()
    run_config_path = os.path.join(args.run_root, "run_config.json")
    with open(run_config_path, "r") as f:
        run_config = json.load(f)
    out_zarr_path = run_config["out_zarr_path"]
    ds = _open_merged_dataset(out_zarr_path)
    grid_path = args.grid_path if args.grid_path is not None else run_config.get("grid_path", DEFAULT_MODEL_GRID_PATH)
    model_grid = open_model_grid(grid_path)

    safe_start = pd.Timestamp(run_config["safe_start_date"]).normalize()
    safe_end = pd.Timestamp(run_config["safe_end_date"]).normalize()
    month_start, month_end, site_error = _load_site_error_for_validation(
        run_config,
        safe_start,
        safe_end,
    )
    selected_site_keys = [
        rec["site_key"]
        for rec in run_config.get("validation_sites", [])
        if "site_key" in rec
    ]
    print(
        f"[validate_map_outputs] validation window {month_start.date()} to {month_end.date()}"
    )

    plots_root = run_config.get("plots_dir", os.path.join(run_config["validation_dir"], "plots"))
    os.makedirs(plots_root, exist_ok=True)

    map_day = _pick_map_day(ds, month_start, month_end)
    day_mean_da = ds[OUTPUT_MEAN_NAME].sel(time=map_day)
    day_field = day_mean_da.values
    valid_mask = np.isfinite(day_field)
    map_path = os.path.join(plots_root, f"map_{map_day.date().isoformat()}.png")
    _write_map_plot(
        da=day_mean_da.rename(OUTPUT_MEAN_NAME),
        save_path=map_path,
        title=f"{OUTPUT_MEAN_NAME} on {map_day.date()}",
        cmap=LFMC_BROWN_GREEN_CMAP,
        cbar_label="LFMC (%)",
    )
    std_map_path = os.path.join(plots_root, f"map_std_{map_day.date().isoformat()}.png")
    _write_map_plot(
        da=ds[OUTPUT_STD_NAME].sel(time=map_day).rename(OUTPUT_STD_NAME),
        save_path=std_map_path,
        title=f"{OUTPUT_STD_NAME} on {map_day.date()}",
        cmap="magma",
        cbar_label="LFMC uncertainty (std, %)",
    )

    sample_idx = _pick_sample_pixels(day_field.reshape(-1), n_sites=int(args.n_map_sites))
    y_idx, x_idx = np.unravel_index(sample_idx, day_field.shape)
    sample_ts_dir = os.path.join(plots_root, "sample_locations")
    os.makedirs(sample_ts_dir, exist_ok=True)
    for site_idx, (yi, xi) in enumerate(zip(y_idx.tolist(), x_idx.tolist()), start=1):
        infer_df = _extract_point_series(ds, y_idx=yi, x_idx=xi, model_grid=model_grid)
        save_path = os.path.join(
            sample_ts_dir,
            f"sample_location_{site_idx:02d}_{float(infer_df['lat'].iloc[0]):.4f}_{float(infer_df['lon'].iloc[0]):.4f}.png",
        )
        _write_prediction_timeseries_plot(
            infer_df=infer_df,
            save_path=save_path,
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
        selected_site_keys=selected_site_keys,
    )


if __name__ == "__main__":
    main()
