#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import yaml
import zarr


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "rao_s1_source_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def date_strings(root):
    values = np.asarray(root["time"][:]).astype("datetime64[D]")
    return [np.datetime_as_string(value, unit="D") for value in values]


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value))


def calendar_index_365(date_value: dt.date) -> int:
    if date_value.month == 2 and date_value.day == 29:
        date_value = dt.date(date_value.year, 2, 28)
    return int(dt.date(2001, date_value.month, date_value.day).timetuple().tm_yday - 1)


def window_indices(center_idx: int, start_offset: int, end_offset: int) -> np.ndarray:
    return np.asarray([(center_idx + offset) % 365 for offset in range(start_offset, end_offset + 1)], dtype=np.int16)


def finite_percentiles(values: np.ndarray, percentiles: Tuple[float, ...]):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [None for _ in percentiles]
    return [float(value) for value in np.percentile(finite, percentiles)]


def save_map(data: np.ndarray, title: str, path: Path, cmap: str, norm=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, shrink=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log(f"Saved {path}")


def save_histogram(data: np.ndarray, path: Path) -> None:
    finite = data[np.isfinite(data)]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(finite, bins=80, color="#6d8e60", edgecolor="none")
    ax.axvline(0.0, color="#202020", linewidth=1.2)
    ax.set_title("Sentinel-1 LFMC anomaly distribution")
    ax.set_xlabel("LFMC anomaly (%)")
    ax.set_ylabel("Pixel count")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log(f"Saved {path}")


def selected_subset(root, size: int) -> Tuple[slice, slice]:
    height = int(root["y"].shape[0])
    width = int(root["x"].shape[0])
    y_start = max(0, height // 2 - size // 2)
    x_start = max(0, width // 2 - size // 2)
    return slice(y_start, min(y_start + size, height)), slice(x_start, min(x_start + size, width))


def run(args) -> None:
    cfg = load_config(args.config)
    clim_cfg = cfg["climatology"]
    viewer_path = Path(str(cfg["paths"]["viewer_zarr_path"]))
    root = zarr.open_group(str(viewer_path), mode="r")
    dates = date_strings(root)
    target_date = args.date or dates[-1]
    if target_date not in dates:
        raise ValueError(f"Date {target_date} not found in {viewer_path}")
    target_idx = dates.index(target_date)
    baseline_start = parse_date(clim_cfg["baseline_start_date"])
    baseline_end = parse_date(clim_cfg["baseline_end_date"])
    baseline_indices = [idx for idx, value in enumerate(dates) if baseline_start <= parse_date(value) <= baseline_end]
    baseline_days = np.asarray([calendar_index_365(parse_date(dates[idx])) for idx in baseline_indices], dtype=np.int16)
    target_day_idx = calendar_index_365(parse_date(target_date))
    window_start, window_end = [int(value) for value in clim_cfg["window_offsets"]]
    target_window = set(window_indices(target_day_idx, window_start, window_end).tolist())
    selected_indices = [idx for idx, day_idx in zip(baseline_indices, baseline_days) if int(day_idx) in target_window]
    if not selected_indices:
        raise ValueError(f"No baseline dates found for {target_date} anomaly window")

    y_slice, x_slice = selected_subset(root, int(args.size))
    log(f"Loading subset y={y_slice.start}:{y_slice.stop} x={x_slice.start}:{x_slice.stop} for {target_date}")
    lfmc = np.asarray(root["lfmc"][target_idx, y_slice, x_slice], dtype=np.float32)
    stack = np.asarray(
        root["lfmc"].get_orthogonal_selection((np.asarray(selected_indices, dtype=np.int64), y_slice, x_slice)),
        dtype=np.float32,
    )
    climatology = np.nanmean(stack, axis=0).astype(np.float32)
    anomaly = lfmc - climatology

    scratch = Path(os.environ.get("SCRATCH", "/scratch/users/trobinet"))
    plot_dir = scratch / "long_lfmc/final_lfmc/rao_s1_lfmc/plots/anomaly_smoke"
    plot_dir.mkdir(parents=True, exist_ok=True)
    max_abs = max(abs(value) for value in finite_percentiles(anomaly, (2, 98)) if value is not None)
    max_abs = max(float(max_abs), 1.0)
    paths = {
        "lfmc": plot_dir / f"lfmc_{target_date}.png",
        "climatology": plot_dir / f"climatology_{target_date}.png",
        "anomaly": plot_dir / f"anomaly_{target_date}.png",
        "histogram": plot_dir / f"anomaly_histogram_{target_date}.png",
        "report": plot_dir / f"anomaly_smoke_{target_date}.json",
    }
    save_map(lfmc, f"Sentinel-1 LFMC {target_date}", paths["lfmc"], "viridis")
    save_map(climatology, f"Sentinel-1 LFMC climatology {target_date}", paths["climatology"], "viridis")
    save_map(
        anomaly,
        f"Sentinel-1 LFMC anomaly {target_date}",
        paths["anomaly"],
        "RdBu",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    save_histogram(anomaly, paths["histogram"])
    report = {
        "date": target_date,
        "viewer_zarr_path": str(viewer_path),
        "subset": {"y_start": y_slice.start, "y_stop": y_slice.stop, "x_start": x_slice.start, "x_stop": x_slice.stop},
        "baseline_start_date": baseline_start.isoformat(),
        "baseline_end_date": baseline_end.isoformat(),
        "baseline_dates_in_window": len(selected_indices),
        "window_offsets": [window_start, window_end],
        "anomaly_percentile_labels": [2, 50, 98],
        "anomaly_percentiles": finite_percentiles(anomaly, (2, 50, 98)),
        "finite_anomaly_fraction": float(np.isfinite(anomaly).mean()),
        "plots": {key: str(value) for key, value in paths.items() if key != "report"},
    }
    paths["report"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log(f"Saved {paths['report']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test Rao S1 anomaly climatology on a small viewer-grid subset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--date", default=None)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
