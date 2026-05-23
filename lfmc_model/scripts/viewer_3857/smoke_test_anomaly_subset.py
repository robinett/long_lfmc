#!/usr/bin/env python3

import datetime as dt
import json
import math
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from pyproj import Transformer


SOURCE_ZARR = Path(
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inference/final_products/"
    "lfmc_vh_vv_365_multisource_fusion_clim20_2001_2024.zarr"
)
SMOKE_ROOT = Path(
    os.environ.get(
        "SCRATCH",
        "/scratch/users/trobinet",
    )
) / "long_lfmc/final_lfmc/lfmc_model/viewer_3857/anomaly_smoke_test"
PLOT_ROOT = Path(
    os.environ.get(
        "SCRATCH",
        "/scratch/users/trobinet",
    )
) / "long_lfmc/final_lfmc/lfmc_model/plots/viewer_3857_anomaly_smoke"

DISPLAY_VARIABLE = "lfmc_ens_mean"
UNCERTAINTY_VARIABLE = "lfmc_ens_std"
QUALITY_VARIABLE = "quality_flag"
CLIMATOLOGY_VARIABLE = "lfmc_climatology_mean"
ANOMALY_VARIABLE = "lfmc_anomaly"
SOURCE_CRS = "EPSG:5070"
SUBSET_SIZE = 96
TIME_BLOCK = 128
SMOKE_DATE = "2024-12-31"
FEB29_DATE = "2020-02-29"
WINDOW_OFFSETS = tuple(range(-10, 11))
TILE_SIZE = 128


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def parse_date(value) -> dt.date:
    return dt.date.fromisoformat(np.datetime_as_string(np.datetime64(value), unit="D"))


def calendar_index_365(date_value: dt.date) -> int:
    if date_value.month == 2 and date_value.day == 29:
        date_value = dt.date(date_value.year, 2, 28)
    template_date = dt.date(2001, date_value.month, date_value.day)
    return int(template_date.timetuple().tm_yday - 1)


def calendar_window_indices(center_idx: int) -> np.ndarray:
    return np.asarray([(center_idx + offset) % 365 for offset in WINDOW_OFFSETS], dtype=np.int16)


def nearest_index(sorted_values: np.ndarray, target_value: float) -> int:
    insert_idx = int(np.searchsorted(sorted_values, target_value, side="left"))
    if insert_idx <= 0:
        return 0
    if insert_idx >= sorted_values.size:
        return sorted_values.size - 1
    left_idx = insert_idx - 1
    right_idx = insert_idx
    if abs(target_value - sorted_values[left_idx]) <= abs(sorted_values[right_idx] - target_value):
        return left_idx
    return right_idx


def source_cell_for_lat_lon(ds: xr.Dataset, lat: float, lon: float) -> tuple[int, int]:
    transformer = Transformer.from_crs("EPSG:4326", SOURCE_CRS, always_xy=True)
    x_value, y_value = transformer.transform(lon, lat)
    x_values = np.asarray(ds["x"].values, dtype=np.float64)
    y_values = np.asarray(ds["y"].values, dtype=np.float64)
    x_idx = nearest_index(x_values, x_value)
    y_idx_asc = nearest_index(y_values[::-1], y_value)
    y_idx = y_values.size - 1 - y_idx_asc
    return int(y_idx), int(x_idx)


def bounded_slice(center_idx: int, size: int, max_size: int) -> slice:
    start = max(0, min(center_idx - size // 2, max_size - size))
    return slice(start, start + size)


def choose_subset(ds: xr.Dataset) -> tuple[slice, slice, dict]:
    candidate_centers = [
        {"lat": 38.50, "lon": -120.00, "label": "sierra_nevada"},
        {"lat": 34.22, "lon": -119.05, "label": "southern_california"},
        {"lat": 40.00, "lon": -121.00, "label": "northern_california"},
        {"lat": 39.00, "lon": -105.50, "label": "colorado_front_range"},
        {"lat": 40.80, "lon": -111.70, "label": "wasatch"},
    ]
    dates = [np.datetime_as_string(np.datetime64(value), unit="D") for value in ds["time"].values]
    date_idx = dates.index(SMOKE_DATE)
    best = None

    for candidate in candidate_centers:
        y_center, x_center = source_cell_for_lat_lon(ds, candidate["lat"], candidate["lon"])
        y_slice = bounded_slice(y_center, SUBSET_SIZE, int(ds.sizes["y"]))
        x_slice = bounded_slice(x_center, SUBSET_SIZE, int(ds.sizes["x"]))
        block = np.asarray(
            ds[DISPLAY_VARIABLE].isel(time=date_idx, y=y_slice, x=x_slice).values,
            dtype=np.float32,
        )
        finite_fraction = float(np.isfinite(block).mean())
        log(
            "Candidate subset "
            f"{candidate['label']} lat={candidate['lat']:.3f} lon={candidate['lon']:.3f} "
            f"finite_fraction={finite_fraction:.3f}"
        )
        if best is None or finite_fraction > best["finite_fraction"]:
            best = {
                **candidate,
                "finite_fraction": finite_fraction,
                "y_slice": y_slice,
                "x_slice": x_slice,
            }

    if best is None or best["finite_fraction"] <= 0.0:
        raise RuntimeError("Could not find a smoke-test subset with finite LFMC values")

    metadata = {
        key: value
        for key, value in best.items()
        if key not in {"y_slice", "x_slice"}
    }
    metadata.update(
        {
            "y_start": int(best["y_slice"].start),
            "y_stop": int(best["y_slice"].stop),
            "x_start": int(best["x_slice"].start),
            "x_stop": int(best["x_slice"].stop),
        }
    )
    return best["y_slice"], best["x_slice"], metadata


def compute_climatology(ds: xr.Dataset, y_slice: slice, x_slice: slice) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dates = [parse_date(value) for value in ds["time"].values]
    day_indices = np.asarray([calendar_index_365(value) for value in dates], dtype=np.int16)
    sum_by_day = np.zeros((365, SUBSET_SIZE, SUBSET_SIZE), dtype=np.float64)
    count_by_day = np.zeros((365, SUBSET_SIZE, SUBSET_SIZE), dtype=np.uint16)
    total_times = len(dates)

    for start in range(0, total_times, TIME_BLOCK):
        end = min(start + TIME_BLOCK, total_times)
        block = np.asarray(
            ds[DISPLAY_VARIABLE].isel(time=slice(start, end), y=y_slice, x=x_slice).values,
            dtype=np.float32,
        )
        for day_idx in np.unique(day_indices[start:end]):
            local_mask = day_indices[start:end] == day_idx
            local_values = block[local_mask, :, :]
            finite = np.isfinite(local_values)
            sum_by_day[day_idx, :, :] += np.where(finite, local_values, 0.0).sum(axis=0)
            count_by_day[day_idx, :, :] += finite.sum(axis=0).astype(np.uint16)
        log(f"Accumulated climatology source times {start}-{end - 1} of {total_times - 1}")

    rolling_sum = np.zeros_like(sum_by_day, dtype=np.float64)
    rolling_count = np.zeros_like(count_by_day, dtype=np.uint16)
    for day_idx in range(365):
        window = calendar_window_indices(day_idx)
        rolling_sum[day_idx, :, :] = sum_by_day[window, :, :].sum(axis=0)
        rolling_count[day_idx, :, :] = count_by_day[window, :, :].sum(axis=0)

    climatology = np.full((365, SUBSET_SIZE, SUBSET_SIZE), np.nan, dtype=np.float32)
    np.divide(
        rolling_sum,
        rolling_count,
        out=climatology,
        where=rolling_count > 0,
        casting="unsafe",
    )
    climatology_day_labels = [dt.date(2001, 1, 1) + dt.timedelta(days=idx) for idx in range(365)]
    labels = [value.isoformat()[5:] for value in climatology_day_labels]
    return climatology, rolling_count, labels


def write_smoke_zarr(
    ds: xr.Dataset,
    y_slice: slice,
    x_slice: slice,
    climatology: np.ndarray,
    smoke_date_idx: int,
    smoke_day_idx: int,
) -> Path:
    out = SMOKE_ROOT / "lfmc_anomaly_smoke_subset.zarr"
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out), mode="w", zarr_format=2)
    x_values = np.asarray(ds["x"].isel(x=x_slice).values, dtype=np.float64)
    y_values = np.asarray(ds["y"].isel(y=y_slice).values, dtype=np.float64)
    time_value = np.asarray(ds["time"].isel(time=smoke_date_idx).values).reshape(1)
    lfmc = np.asarray(
        ds[DISPLAY_VARIABLE].isel(time=smoke_date_idx, y=y_slice, x=x_slice).values,
        dtype=np.float32,
    )
    clim = climatology[smoke_day_idx, :, :]
    anomaly = lfmc - clim

    root.create_array("x", data=x_values, chunks=(SUBSET_SIZE,), attributes={"_ARRAY_DIMENSIONS": ["x"]})
    root.create_array("y", data=y_values, chunks=(SUBSET_SIZE,), attributes={"_ARRAY_DIMENSIONS": ["y"]})
    root.create_array("time", data=time_value, chunks=(1,), attributes={"_ARRAY_DIMENSIONS": ["time"]})
    root.create_array(
        "climatology_day",
        data=np.arange(1, 366, dtype=np.int16),
        chunks=(365,),
        attributes={"_ARRAY_DIMENSIONS": ["climatology_day"]},
    )
    root.create_array(
        DISPLAY_VARIABLE,
        data=lfmc.reshape(1, SUBSET_SIZE, SUBSET_SIZE),
        chunks=(1, 64, 64),
        attributes={"_ARRAY_DIMENSIONS": ["time", "y", "x"], "units": "percent"},
    )
    root.create_array(
        CLIMATOLOGY_VARIABLE,
        data=climatology,
        chunks=(365, 32, 32),
        attributes={
            "_ARRAY_DIMENSIONS": ["climatology_day", "y", "x"],
            "units": "percent",
            "calendar": "365_day",
            "window_offsets": json.dumps(list(WINDOW_OFFSETS)),
            "feb29_policy": "map_to_feb28",
        },
    )
    root.create_array(
        ANOMALY_VARIABLE,
        data=anomaly.reshape(1, SUBSET_SIZE, SUBSET_SIZE),
        chunks=(1, 64, 64),
        attributes={"_ARRAY_DIMENSIONS": ["time", "y", "x"], "units": "percent"},
    )
    log(f"Wrote smoke zarr {out}")
    return out


def finite_percentiles(values: np.ndarray, percentiles: tuple[float, ...]) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [math.nan for _ in percentiles]
    return [float(np.nanpercentile(finite, pct)) for pct in percentiles]


def save_map_plot(data: np.ndarray, title: str, path: Path, *, cmap: str, norm=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log(f"Wrote plot {path}")


def save_histogram(data: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = data[np.isfinite(data)]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.hist(finite, bins=80, color="#536b7d")
    ax.axvline(0.0, color="#202020", linewidth=1.2)
    ax.set_title(f"Smoke-test LFMC anomaly distribution for {SMOKE_DATE}")
    ax.set_xlabel("LFMC anomaly (%)")
    ax.set_ylabel("Pixel count")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log(f"Wrote plot {path}")


def color_ramp(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    palette = np.asarray(
        [
            [65, 84, 139],
            [181, 209, 224],
            [247, 247, 241],
            [230, 178, 122],
            [132, 57, 43],
        ],
        dtype=np.float32,
    )
    stops = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    valid = np.isfinite(values)
    normalized = np.clip((values - value_min) / (value_max - value_min), 0.0, 1.0)
    normalized[~valid] = 0.5
    out = np.empty(values.shape + (3,), dtype=np.uint8)
    flat = normalized.ravel()
    flat_rgb = out.reshape(-1, 3)
    for channel_idx in range(3):
        flat_rgb[:, channel_idx] = np.interp(flat, stops, palette[:, channel_idx]).astype(np.uint8)
    alpha = np.where(valid, 220, 0).astype(np.uint8)
    return np.dstack([out, alpha])


def write_smoke_tile(anomaly: np.ndarray, max_abs: float) -> Path:
    tile_root = SMOKE_ROOT / "tiles/anomaly" / SMOKE_DATE / "0" / "0"
    tile_root.mkdir(parents=True, exist_ok=True)
    tile_path = tile_root / "0.png"
    rgba = color_ramp(anomaly, -max_abs, max_abs)
    tile_rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    tile_rgba[: anomaly.shape[0], : anomaly.shape[1], :] = rgba
    Image.fromarray(tile_rgba, mode="RGBA").save(tile_path, format="PNG", optimize=True)
    log(f"Wrote smoke anomaly tile {tile_path}")
    return tile_path


def verify_point(
    ds: xr.Dataset,
    y_slice: slice,
    x_slice: slice,
    climatology: np.ndarray,
    smoke_date_idx: int,
) -> dict:
    smoke_date = dt.date.fromisoformat(SMOKE_DATE)
    smoke_day_idx = calendar_index_365(smoke_date)
    smoke_block = np.asarray(
        ds[DISPLAY_VARIABLE].isel(time=smoke_date_idx, y=y_slice, x=x_slice).values,
        dtype=np.float32,
    )
    valid_point_mask = np.isfinite(smoke_block) & np.isfinite(climatology[smoke_day_idx, :, :])
    if not np.any(valid_point_mask):
        raise RuntimeError("No finite smoke-date pixels are available for point verification")

    valid_rows, valid_cols = np.where(valid_point_mask)
    center_row = SUBSET_SIZE // 2
    center_col = SUBSET_SIZE // 2
    distance2 = (valid_rows - center_row) ** 2 + (valid_cols - center_col) ** 2
    chosen_idx = int(np.argmin(distance2))
    y_local = int(valid_rows[chosen_idx])
    x_local = int(valid_cols[chosen_idx])
    y_idx = y_slice.start + y_local
    x_idx = x_slice.start + x_local
    dates = [parse_date(value) for value in ds["time"].values]
    day_indices = np.asarray([calendar_index_365(value) for value in dates], dtype=np.int16)
    window = calendar_window_indices(smoke_day_idx)
    window_mask = np.isin(day_indices, window)
    point_series = np.asarray(ds[DISPLAY_VARIABLE].isel(y=y_idx, x=x_idx).values, dtype=np.float32)
    manual_clim = float(np.nanmean(point_series[window_mask]))
    computed_clim = float(climatology[smoke_day_idx, y_local, x_local])
    lfmc_value = float(point_series[smoke_date_idx])
    std_value = float(
        np.asarray(ds[UNCERTAINTY_VARIABLE].isel(time=smoke_date_idx, y=y_idx, x=x_idx).values).item()
    )
    anomaly = lfmc_value - computed_clim

    start_idx = max(0, smoke_date_idx - 89)
    end_idx = smoke_date_idx + 1
    timeseries_dates = [value.isoformat() for value in dates[start_idx:end_idx]]
    timeseries_mean = point_series[start_idx:end_idx].astype(np.float32)
    timeseries_clim = np.asarray(
        [climatology[calendar_index_365(value), y_local, x_local] for value in dates[start_idx:end_idx]],
        dtype=np.float32,
    )
    timeseries_anomaly = timeseries_mean - timeseries_clim

    feb29_idx = dates.index(dt.date.fromisoformat(FEB29_DATE))
    feb28_idx = dates.index(dt.date(2020, 2, 28))
    feb29_day_idx = calendar_index_365(dates[feb29_idx])
    feb28_day_idx = calendar_index_365(dates[feb28_idx])

    return {
        "point_indices": {
            "source_y": int(y_idx),
            "source_x": int(x_idx),
            "local_y": int(y_local),
            "local_x": int(x_local),
        },
        "smoke_date": SMOKE_DATE,
        "smoke_climatology_day_index_zero_based": int(smoke_day_idx),
        "manual_climatology": manual_clim,
        "computed_climatology": computed_clim,
        "absolute_difference": abs(manual_clim - computed_clim),
        "lfmc_ens_mean": lfmc_value,
        "lfmc_ens_std": std_value,
        "lfmc_climatology_mean": computed_clim,
        "lfmc_anomaly": anomaly,
        "api_style_timeseries": {
            "dates": timeseries_dates,
            "lfmc_ens_mean": [None if not np.isfinite(value) else float(value) for value in timeseries_mean],
            "lfmc_climatology_mean": [
                None if not np.isfinite(value) else float(value) for value in timeseries_clim
            ],
            "lfmc_anomaly": [
                None if not np.isfinite(value) else float(value) for value in timeseries_anomaly
            ],
            "window_days": 90,
        },
        "feb29_policy_check": {
            "feb28_index_zero_based": int(feb28_day_idx),
            "feb29_index_zero_based": int(feb29_day_idx),
            "same_index": bool(feb28_day_idx == feb29_day_idx),
        },
    }


def main() -> None:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Opening source zarr {SOURCE_ZARR}")
    ds = xr.open_zarr(SOURCE_ZARR, consolidated=False)
    try:
        dates = [np.datetime_as_string(np.datetime64(value), unit="D") for value in ds["time"].values]
        smoke_date_idx = dates.index(SMOKE_DATE)
        smoke_day_idx = calendar_index_365(dt.date.fromisoformat(SMOKE_DATE))
        y_slice, x_slice, subset_metadata = choose_subset(ds)
        log(f"Selected subset metadata: {subset_metadata}")

        climatology, rolling_count, labels = compute_climatology(ds, y_slice, x_slice)
        lfmc = np.asarray(
            ds[DISPLAY_VARIABLE].isel(time=smoke_date_idx, y=y_slice, x=x_slice).values,
            dtype=np.float32,
        )
        clim = climatology[smoke_day_idx, :, :]
        anomaly = lfmc - clim
        smoke_zarr = write_smoke_zarr(ds, y_slice, x_slice, climatology, smoke_date_idx, smoke_day_idx)

        anomaly_percentiles = finite_percentiles(anomaly, (1, 2, 5, 25, 50, 75, 95, 98, 99))
        max_abs = max(abs(anomaly_percentiles[1]), abs(anomaly_percentiles[-2]), 1.0)
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        plot_paths = {
            "lfmc": PLOT_ROOT / f"lfmc_{SMOKE_DATE}.png",
            "climatology": PLOT_ROOT / f"climatology_{SMOKE_DATE}.png",
            "anomaly": PLOT_ROOT / f"anomaly_{SMOKE_DATE}.png",
            "histogram": PLOT_ROOT / f"anomaly_histogram_{SMOKE_DATE}.png",
        }
        save_map_plot(lfmc, f"Smoke-test LFMC {SMOKE_DATE}", plot_paths["lfmc"], cmap="viridis")
        save_map_plot(clim, f"Smoke-test climatology for {SMOKE_DATE}", plot_paths["climatology"], cmap="viridis")
        save_map_plot(
            anomaly,
            f"Smoke-test LFMC anomaly {SMOKE_DATE}",
            plot_paths["anomaly"],
            cmap="RdBu_r",
            norm=norm,
        )
        save_histogram(anomaly, plot_paths["histogram"])
        tile_path = write_smoke_tile(anomaly, max_abs=max_abs)
        point_report = verify_point(ds, y_slice, x_slice, climatology, smoke_date_idx)

        report = {
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_zarr": str(SOURCE_ZARR),
            "smoke_zarr": str(smoke_zarr),
            "subset": subset_metadata,
            "climatology": {
                "day_count": 365,
                "window_offsets": list(WINDOW_OFFSETS),
                "window_size": len(WINDOW_OFFSETS),
                "feb29_policy": "map_to_feb28",
                "smoke_date": SMOKE_DATE,
                "smoke_day_label": labels[smoke_day_idx],
                "smoke_day_index_zero_based": int(smoke_day_idx),
                "rolling_count_min": int(np.nanmin(rolling_count)),
                "rolling_count_max": int(np.nanmax(rolling_count)),
            },
            "anomaly_distribution": {
                "percentile_labels": [1, 2, 5, 25, 50, 75, 95, 98, 99],
                "percentiles": anomaly_percentiles,
                "temporary_tile_color_max_abs": float(max_abs),
            },
            "plots": {key: str(value) for key, value in plot_paths.items()},
            "tile": str(tile_path),
            "point_verification": point_report,
        }
        report_path = SMOKE_ROOT / "smoke_test_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"Wrote smoke report {report_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
