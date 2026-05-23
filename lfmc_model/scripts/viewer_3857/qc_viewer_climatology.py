#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "viewer_pipeline_config.yaml"
DEFAULT_DATE = "2024-12-31"


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def load_config(config_path: Path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def datestr(value) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


def calendar_index_365(date_value: dt.date) -> int:
    if date_value.month == 2 and date_value.day == 29:
        date_value = dt.date(date_value.year, 2, 28)
    return int(dt.date(2001, date_value.month, date_value.day).timetuple().tm_yday - 1)


def finite_percentiles(values: np.ndarray, percentiles):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [None for _ in percentiles]
    return [float(np.nanpercentile(finite, pct)) for pct in percentiles]


def save_map(data: np.ndarray, title: str, path: Path, cmap: str, norm=None) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    image = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log(f"Wrote {path}")


def qc(args) -> None:
    cfg = load_config(args.config)
    clim_cfg = cfg["climatology"]
    viewer_path = Path(str(cfg["output"]["viewer_dataset_path"]))
    plot_date = str(args.date)
    plot_dir = Path(os.environ.get("SCRATCH", "/scratch/users/trobinet")) / (
        "long_lfmc/final_lfmc/lfmc_model/plots/viewer_3857_climatology_qc"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    log(f"Opening viewer zarr {viewer_path}")
    ds = xr.open_zarr(viewer_path, consolidated=False)
    try:
        dates = [datestr(value) for value in ds["time"].values]
        time_idx = dates.index(plot_date)
        day_idx = calendar_index_365(dt.date.fromisoformat(plot_date))
        display_variable = str(cfg["dataset"]["display_variable"])
        tile_variable = str(clim_cfg["viewer_tile_variable"])
        point_variable = str(clim_cfg["viewer_point_variable"])

        log(f"Loading viewer maps for {plot_date} time_idx={time_idx} climatology_day_idx={day_idx}")
        lfmc = np.asarray(ds[display_variable].isel(time=time_idx).values, dtype=np.float32)
        tile_clim = np.asarray(ds[tile_variable].isel(climatology_day=day_idx).values, dtype=np.float32)
        anomaly = lfmc - tile_clim

        y_idx = int(np.asarray(np.where(np.isfinite(anomaly))[0])[0])
        x_idx = int(np.asarray(np.where(np.isfinite(anomaly))[1])[0])
        point_value = float(ds[point_variable].isel(climatology_day=day_idx, y=y_idx, x=x_idx).values)
        tile_value = float(tile_clim[y_idx, x_idx])
        point_tile_abs_diff = abs(point_value - tile_value)

        percentile_labels = [1, 2, 5, 25, 50, 75, 95, 98, 99]
        percentiles = finite_percentiles(anomaly, percentile_labels)
        finite_percentiles_only = [value for value in percentiles if value is not None]
        max_abs = max(abs(percentiles[1]), abs(percentiles[-2]), 1.0) if finite_percentiles_only else 1.0

        lfmc_path = plot_dir / f"lfmc_{plot_date}_viewer_grid.png"
        clim_path = plot_dir / f"climatology_{plot_date}_viewer_grid.png"
        anom_path = plot_dir / f"anomaly_{plot_date}_viewer_grid.png"
        hist_path = plot_dir / f"anomaly_histogram_{plot_date}_viewer_grid.png"
        panel_path = plot_dir / f"lfmc_climatology_anomaly_{plot_date}_viewer_grid.png"
        report_path = plot_dir / f"qc_report_{plot_date}.json"

        save_map(lfmc, f"Viewer LFMC {plot_date}", lfmc_path, "viridis")
        save_map(tile_clim, f"Viewer LFMC climatology {plot_date}", clim_path, "viridis")
        save_map(
            anomaly,
            f"Viewer LFMC anomaly {plot_date}",
            anom_path,
            "RdBu_r",
            TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        )

        finite = anomaly[np.isfinite(anomaly)]
        fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
        ax.hist(finite, bins=120, color="#536b7d")
        ax.axvline(0.0, color="#202020", linewidth=1.2)
        ax.set_title(f"Viewer LFMC anomaly distribution {plot_date}")
        ax.set_xlabel("LFMC anomaly (%)")
        ax.set_ylabel("Pixel count")
        fig.savefig(hist_path, dpi=180)
        plt.close(fig)
        log(f"Wrote {hist_path}")

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), constrained_layout=True)
        panels = [
            (lfmc, f"LFMC {plot_date}", "viridis", None),
            (tile_clim, "Climatology", "viridis", None),
            (anomaly, "Anomaly", "RdBu_r", TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)),
        ]
        for ax, (data, title, cmap, norm) in zip(axes, panels):
            image = ax.imshow(data, cmap=cmap, norm=norm)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, shrink=0.72)
        fig.savefig(panel_path, dpi=180)
        plt.close(fig)
        log(f"Wrote {panel_path}")

        report = {
            "status": "completed",
            "plot_date": plot_date,
            "viewer_dataset_path": str(viewer_path),
            "time_idx": int(time_idx),
            "climatology_day_idx_zero_based": int(day_idx),
            "tile_variable": tile_variable,
            "point_variable": point_variable,
            "point_tile_check": {
                "y": y_idx,
                "x": x_idx,
                "tile_value": tile_value,
                "point_value": point_value,
                "absolute_difference": point_tile_abs_diff,
            },
            "anomaly_percentile_labels": percentile_labels,
            "anomaly_percentiles": percentiles,
            "max_abs_for_plot": float(max_abs),
            "finite_anomaly_fraction": float(np.isfinite(anomaly).mean()),
            "plots": {
                "lfmc": str(lfmc_path),
                "climatology": str(clim_path),
                "anomaly": str(anom_path),
                "histogram": str(hist_path),
                "panel": str(panel_path),
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"Wrote {report_path}")
    finally:
        ds.close()


def parse_args():
    parser = argparse.ArgumentParser(description="QC plot viewer-grid LFMC climatology and anomaly.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--date", default=DEFAULT_DATE)
    return parser.parse_args()


def main() -> None:
    qc(parse_args())


if __name__ == "__main__":
    main()
