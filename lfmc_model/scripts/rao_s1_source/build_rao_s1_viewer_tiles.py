#!/usr/bin/env python3

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from PIL import Image
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


def date_strings(root) -> Sequence[str]:
    values = np.asarray(root["time"][:]).astype("datetime64[D]")
    return [np.datetime_as_string(value, unit="D") for value in values]


def build_color_ramp(normalized_values: np.ndarray, stops: np.ndarray, colors: np.ndarray) -> np.ndarray:
    out = np.empty(normalized_values.shape + (3,), dtype=np.uint8)
    flat = normalized_values.ravel()
    flat_rgb = out.reshape(-1, 3)
    for channel_idx in range(3):
        flat_rgb[:, channel_idx] = np.interp(flat, stops, colors[:, channel_idx]).astype(np.uint8)
    return out


def write_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG", optimize=True)


def tile_counts(height: int, width: int, tile_size: int):
    return int(math.ceil(width / tile_size)), int(math.ceil(height / tile_size))


def downsample_rgba(rgba: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return rgba
    return np.ascontiguousarray(rgba[::factor, ::factor, :])


def rgba_for_data(data: np.ndarray, cfg: Dict[str, object]) -> np.ndarray:
    rendering = cfg["rendering"]
    valid_mask = np.isfinite(data)
    value_min = float(rendering["min"])
    value_max = float(rendering["max"])
    normalized = (data - value_min) / (value_max - value_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid_mask] = 0.0
    stops = np.asarray(rendering["stops"], dtype=np.float32)
    colors = np.asarray(rendering["palette"], dtype=np.float32)
    rgb = build_color_ramp(normalized, stops, colors)
    alpha = np.where(valid_mask, int(rendering.get("default_valid_alpha", 191)), 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def tile_relpath(date_str: str, zoom: int, tile_x: int, tile_y: int) -> Path:
    return Path("tiles") / "lfmc" / date_str / str(zoom) / str(tile_x) / f"{tile_y}.png"


def write_tiles(asset_root: Path, rgba: np.ndarray, cfg: Dict[str, object], date_str: str):
    viewer = cfg["viewer"]
    tile_size = int(viewer["tile_size"])
    min_zoom = int(viewer["min_zoom"])
    max_zoom = int(viewer["max_zoom"])
    layer_counts = {}
    for zoom in range(min_zoom, max_zoom + 1):
        factor = 2 ** (max_zoom - zoom)
        zoom_rgba = downsample_rgba(rgba, factor)
        height, width, _ = zoom_rgba.shape
        tiles_x, tiles_y = tile_counts(height, width, tile_size)
        layer_counts[str(zoom)] = {"x": tiles_x, "y": tiles_y}
        log(f"Tiling lfmc {date_str} zoom {zoom}: {tiles_x}x{tiles_y} tiles")
        for tile_y in range(tiles_y):
            row_start = tile_y * tile_size
            row_end = min(row_start + tile_size, height)
            for tile_x in range(tiles_x):
                col_start = tile_x * tile_size
                col_end = min(col_start + tile_size, width)
                tile_rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
                tile_rgba[: row_end - row_start, : col_end - col_start, :] = zoom_rgba[row_start:row_end, col_start:col_end, :]
                write_png(asset_root / tile_relpath(date_str, zoom, tile_x, tile_y), tile_rgba)
    return layer_counts


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def build_manifest(root, cfg: Dict[str, object], dates: Sequence[str], layer_counts: Dict[str, object]) -> Dict[str, object]:
    x_values = np.asarray(root["x"][:], dtype=np.float64)
    y_values = np.asarray(root["y"][:], dtype=np.float64)
    dx = abs(float(np.median(np.diff(x_values))))
    dy = abs(float(np.median(np.diff(y_values))))
    bounds = root.attrs.get("geospatial_bounds", {})
    tile_resolutions = [dx * (2 ** (int(cfg["viewer"]["max_zoom"]) - zoom)) for zoom in range(int(cfg["viewer"]["min_zoom"]), int(cfg["viewer"]["max_zoom"]) + 1)]
    view_resolutions = tile_resolutions + [float(value) for value in cfg["viewer"].get("extra_view_resolutions", []) if float(value) < dx]
    return {
        "dataset_label": str(cfg["dataset"]["dataset_label"]),
        "initial_date": str(cfg["dataset"]["initial_date"]),
        "dates": list(dates),
        "grid_crs": str(cfg["dataset"]["viewer_crs"]),
        "grid_shape": {"height": int(y_values.size), "width": int(x_values.size)},
        "grid_resolution": {"dx": dx, "dy": dy},
        "grid_extent": {
            "west": float(x_values[0] - dx / 2.0),
            "east": float(x_values[-1] + dx / 2.0),
            "north": float(y_values[0] + dy / 2.0),
            "south": float(y_values[-1] - dy / 2.0),
        },
        "geo_bounds": bounds,
        "tiles": {
            "tile_size": int(cfg["viewer"]["tile_size"]),
            "min_zoom": int(cfg["viewer"]["min_zoom"]),
            "max_zoom": int(cfg["viewer"]["max_zoom"]),
            "origin": [float(x_values[0] - dx / 2.0), float(y_values[0] + dy / 2.0)],
            "resolutions": tile_resolutions,
            "view_resolutions": view_resolutions,
        },
        "layers": {
            "lfmc": {
                "label": str(cfg["dataset"]["variable_label"]),
                "unit": str(cfg["dataset"]["units"]),
                "min": float(cfg["rendering"]["min"]),
                "max": float(cfg["rendering"]["max"]),
                "palette": cfg["rendering"]["palette"],
                "stops": cfg["rendering"]["stops"],
                "tile_root_template": "tiles/lfmc/{date}/{z}/{x}/{y}.png",
                "tile_counts": layer_counts,
            }
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def build(args) -> None:
    cfg = load_config(args.config)
    viewer_path = Path(str(cfg["paths"]["viewer_zarr_path"]))
    asset_root = Path(str(cfg["paths"]["asset_root"]))
    variable_name = str(cfg["dataset"]["variable_name"])

    if args.dry_run:
        log(f"Would read viewer zarr: {viewer_path}")
        log(f"Would write absolute LFMC tiles under: {asset_root}")
        return

    root = zarr.open_group(str(viewer_path), mode="r")
    all_dates = date_strings(root)
    dates = args.dates or all_dates
    missing = [date for date in dates if date not in all_dates]
    if missing:
        raise ValueError(f"Dates missing from viewer zarr: {', '.join(missing)}")
    date_index = {date: idx for idx, date in enumerate(all_dates)}
    if args.clear and asset_root.exists():
        log(f"Removing existing asset root {asset_root}")
        shutil.rmtree(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)

    layer_counts = {}
    for date_str in dates:
        date_dir = asset_root / "tiles" / "lfmc" / date_str
        if date_dir.exists():
            shutil.rmtree(date_dir)
        log(f"Rendering absolute LFMC tiles for {date_str}")
        data = np.asarray(root[variable_name][date_index[date_str], :, :], dtype=np.float32)
        rgba = rgba_for_data(data, cfg)
        layer_counts = write_tiles(asset_root, rgba, cfg, date_str)
    manifest = build_manifest(root, cfg, all_dates, layer_counts)
    write_json(asset_root / "manifest.json", manifest)
    log(f"Viewer assets ready: {asset_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Rao S1-informed absolute LFMC viewer tiles.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
