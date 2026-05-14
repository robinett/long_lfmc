#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr

from viewer_pipeline_common import (
    asset_root,
    date_status_path,
    datestr_values,
    load_config,
    log,
    tile_status_dir,
    viewer_dataset_path,
)


def grid_extent(x_values, y_values):
    dx = float(np.median(np.diff(x_values)))
    dy = float(np.median(np.diff(y_values)))
    pixel_width = abs(dx)
    pixel_height = abs(dy)
    return {
        "west": float(x_values[0] - pixel_width / 2.0),
        "east": float(x_values[-1] + pixel_width / 2.0),
        "north": float(y_values[0] + pixel_height / 2.0),
        "south": float(y_values[-1] - pixel_height / 2.0),
    }, pixel_width, pixel_height


def geo_bounds(ds) -> Dict[str, float]:
    lat_values = np.asarray(ds["lat"].values, dtype=np.float64)
    lon_values = np.asarray(ds["lon"].values, dtype=np.float64)
    valid = np.isfinite(lat_values) & np.isfinite(lon_values)
    return {
        "west": float(np.nanmin(lon_values[valid])),
        "east": float(np.nanmax(lon_values[valid])),
        "south": float(np.nanmin(lat_values[valid])),
        "north": float(np.nanmax(lat_values[valid])),
    }


def tile_resolutions(pixel_width: float, min_zoom: int, max_zoom: int):
    return [pixel_width * (2 ** (max_zoom - zoom)) for zoom in range(min_zoom, max_zoom + 1)]


def view_resolutions(pixel_width: float, min_zoom: int, max_zoom: int, extras):
    base = tile_resolutions(pixel_width, min_zoom, max_zoom)
    return base + [float(value) for value in extras if float(value) < pixel_width]


def completed_tile_dates(cfg, available_dates):
    out = []
    for date_str in available_dates:
        status_path = date_status_path(tile_status_dir(cfg), date_str)
        if status_path.exists():
            out.append(date_str)
    return out


def finalize(args):
    cfg = load_config(args.config)
    ds = xr.open_zarr(viewer_dataset_path(cfg), consolidated=False)
    try:
        dates = datestr_values(ds["time"].values)
        publish_dates = completed_tile_dates(cfg, dates)
        if not publish_dates:
            raise RuntimeError("No completed tile dates found; refusing to publish empty manifest")
        if args.require_all_dates and len(publish_dates) != len(dates):
            missing = sorted(set(dates) - set(publish_dates))
            raise RuntimeError(f"Missing tile completion markers for {len(missing)} dates; first missing={missing[:5]}")

        x_values = np.asarray(ds["x"].values, dtype=np.float64)
        y_values = np.asarray(ds["y"].values, dtype=np.float64)
        extent, pixel_width, pixel_height = grid_extent(x_values, y_values)
        tiles_cfg = cfg["tiles"]
        min_zoom = int(tiles_cfg["min_zoom"])
        max_zoom = int(tiles_cfg["max_zoom"])
        tile_size = int(tiles_cfg["tile_size"])
        initial_date_cfg = str(cfg["dataset"].get("initial_date", "latest"))
        initial_date = publish_dates[-1] if initial_date_cfg == "latest" else initial_date_cfg

        manifest = {
            "dataset_label": str(cfg["dataset"]["dataset_label"]),
            "initial_date": initial_date,
            "dates": publish_dates,
            "grid_crs": str(cfg["dataset"]["viewer_grid_crs"]),
            "grid_shape": {"height": int(y_values.size), "width": int(x_values.size)},
            "grid_resolution": {"dx": pixel_width, "dy": pixel_height},
            "grid_extent": extent,
            "geo_bounds": geo_bounds(ds),
            "tiles": {
                "tile_size": tile_size,
                "min_zoom": min_zoom,
                "max_zoom": max_zoom,
                "origin": [extent["west"], extent["north"]],
                "resolutions": tile_resolutions(pixel_width, min_zoom, max_zoom),
                "view_resolutions": view_resolutions(pixel_width, min_zoom, max_zoom, tiles_cfg.get("extra_view_resolutions", [])),
            },
            "layers": {},
            "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        for layer_key, layer_cfg in cfg["layers"].items():
            status = json.loads(date_status_path(tile_status_dir(cfg), publish_dates[-1]).read_text(encoding="utf-8"))
            manifest["layers"][layer_key] = {
                "label": str(layer_cfg["label"]),
                "unit": str(layer_cfg["unit"]),
                "min": float(layer_cfg["min"]),
                "max": float(layer_cfg["max"]),
                "palette": layer_cfg["palette"],
                "stops": layer_cfg["stops"],
                "tile_root_template": f"tiles/{layer_key}" + "/{date}/{z}/{x}/{y}.png",
                "tile_counts": status["layers"][layer_key],
            }

        root = asset_root(cfg)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / str(cfg["output"]["manifest_filename"])
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_path, manifest_path)
        log(f"Published manifest {manifest_path} with {len(publish_dates)} dates")
    finally:
        ds.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Publish viewer tile manifest from completed tile markers.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "viewer_pipeline_config.yaml")
    parser.add_argument("--require-all-dates", action="store_true")
    return parser.parse_args()


def main():
    finalize(parse_args())


if __name__ == "__main__":
    main()
