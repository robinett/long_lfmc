#!/usr/bin/env python3

import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr
import yaml
from PIL import Image


here = Path(__file__).resolve().parent
config_path = here / "viewer_build_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config() -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def datetime64_to_datestr(value) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


def build_color_ramp(normalized_values: np.ndarray, stops: np.ndarray, colors: np.ndarray) -> np.ndarray:
    out = np.empty(normalized_values.shape + (3,), dtype=np.uint8)
    flat = normalized_values.ravel()
    flat_rgb = out.reshape(-1, 3)
    for channel_idx in range(3):
        flat_rgb[:, channel_idx] = np.interp(flat, stops, colors[:, channel_idx]).astype(np.uint8)
    return out


class ViewerAssetBuilder:
    def __init__(self, cfg: Dict[str, object]):
        dataset_cfg = cfg["dataset"]
        output_cfg = cfg["output"]
        tiles_cfg = cfg["tiles"]

        self.dataset_label = str(dataset_cfg["dataset_label"])
        self.dataset_path = str(dataset_cfg["local_dataset_path"])
        self.grid_crs = str(dataset_cfg["grid_crs"])
        self.initial_date = str(dataset_cfg["initial_date"])
        self.selected_dates = [str(value) for value in dataset_cfg["dates"]]

        self.asset_root = Path(str(output_cfg["asset_root"]))
        self.manifest_filename = str(output_cfg["manifest_filename"])
        self.clear_output_root = bool(output_cfg.get("clear_output_root", True))

        self.tile_size = int(tiles_cfg["tile_size"])
        self.min_zoom = int(tiles_cfg["min_zoom"])
        self.max_zoom = int(tiles_cfg["max_zoom"])
        self.extra_view_resolutions = [float(value) for value in tiles_cfg.get("extra_view_resolutions", [250.0, 125.0])]

        self.layers = {str(name): layer_cfg for name, layer_cfg in cfg["layers"].items()}

        log(f"Opening local dataset {self.dataset_path}")
        self.ds = xr.open_zarr(self.dataset_path, consolidated=False)
        self.available_dates = [datetime64_to_datestr(value) for value in self.ds["time"].values]
        self.selected_indices = [self.available_dates.index(date_str) for date_str in self.selected_dates]

        self.x_values = np.asarray(self.ds["x"].values, dtype=np.float64)
        self.y_values = np.asarray(self.ds["y"].values, dtype=np.float64)
        self.lat_array = self.ds["lat"]
        self.lon_array = self.ds["lon"]

        self.dx = float(np.median(np.diff(self.x_values)))
        self.dy = float(np.median(np.diff(self.y_values)))
        self.pixel_width = abs(self.dx)
        self.pixel_height = abs(self.dy)
        self.grid_extent = self._build_grid_extent()
        self.geo_bounds = self._build_geo_bounds()
        self.origin = [self.grid_extent["west"], self.grid_extent["north"]]
        self.tile_resolutions = self._build_tile_resolutions()
        self.view_resolutions = self._build_view_resolutions()

    def _build_grid_extent(self) -> Dict[str, float]:
        return {
            "west": float(self.x_values[0] - self.pixel_width / 2.0),
            "east": float(self.x_values[-1] + self.pixel_width / 2.0),
            "north": float(self.y_values[0] + self.pixel_height / 2.0),
            "south": float(self.y_values[-1] - self.pixel_height / 2.0),
        }

    def _build_geo_bounds(self) -> Dict[str, float]:
        lat_values = np.asarray(self.lat_array.values, dtype=np.float64)
        lon_values = np.asarray(self.lon_array.values, dtype=np.float64)
        valid_mask = np.isfinite(lat_values) & np.isfinite(lon_values)
        return {
            "west": float(np.nanmin(lon_values[valid_mask])),
            "east": float(np.nanmax(lon_values[valid_mask])),
            "south": float(np.nanmin(lat_values[valid_mask])),
            "north": float(np.nanmax(lat_values[valid_mask])),
        }

    def _build_tile_resolutions(self):
        return [self.pixel_width * (2 ** (self.max_zoom - zoom)) for zoom in range(self.min_zoom, self.max_zoom + 1)]

    def _build_view_resolutions(self):
        base = [self.pixel_width * (2 ** (self.max_zoom - zoom)) for zoom in range(self.min_zoom, self.max_zoom + 1)]
        extras = [value for value in self.extra_view_resolutions if value < self.pixel_width]
        return base + extras

    def _rgba_for_layer(self, layer_key: str, time_idx: int) -> np.ndarray:
        layer_cfg = self.layers[layer_key]
        variable_name = str(layer_cfg["variable"])
        data = np.asarray(self.ds[variable_name].isel(time=time_idx).values, dtype=np.float32)
        valid_mask = np.isfinite(data)

        value_min = float(layer_cfg["min"])
        value_max = float(layer_cfg["max"])
        normalized = (data - value_min) / (value_max - value_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized[~valid_mask] = 0.0

        stops = np.asarray(layer_cfg["stops"], dtype=np.float32)
        colors = np.asarray(layer_cfg["palette"], dtype=np.float32)
        rgb = build_color_ramp(normalized, stops, colors)
        alpha = np.where(valid_mask, 235, 0).astype(np.uint8)
        return np.dstack([rgb, alpha])

    def _downsample_rgba(self, rgba: np.ndarray, factor: int) -> np.ndarray:
        if factor == 1:
            return rgba
        return np.ascontiguousarray(rgba[::factor, ::factor, :])

    def _tile_counts(self, array_height: int, array_width: int) -> Tuple[int, int]:
        tiles_x = int(math.ceil(array_width / self.tile_size))
        tiles_y = int(math.ceil(array_height / self.tile_size))
        return tiles_x, tiles_y

    def _write_tile(self, output_path: Path, tile_rgba: np.ndarray) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(tile_rgba, mode="RGBA")
        image.save(output_path, format="PNG", optimize=True)

    def _tile_relpath(self, layer_key: str, date_str: str, zoom: int, tile_x: int, tile_y: int) -> str:
        return str(Path("tiles") / layer_key / date_str / str(zoom) / str(tile_x) / f"{tile_y}.png")

    def _build_tiles_for_image(self, rgba: np.ndarray, layer_key: str, date_str: str, manifest_layer: Dict[str, object]) -> None:
        for zoom in range(self.min_zoom, self.max_zoom + 1):
            factor = 2 ** (self.max_zoom - zoom)
            zoom_rgba = self._downsample_rgba(rgba, factor=factor)
            array_height, array_width, _ = zoom_rgba.shape
            tiles_x, tiles_y = self._tile_counts(array_height=array_height, array_width=array_width)
            manifest_layer["tile_counts"][str(zoom)] = {"x": tiles_x, "y": tiles_y}

            log(
                f"Tiling {layer_key} {date_str} zoom {zoom} "
                f"({array_width}x{array_height} pixels -> {tiles_x}x{tiles_y} tiles)"
            )
            for tile_y in range(tiles_y):
                row_start = tile_y * self.tile_size
                row_end = min(row_start + self.tile_size, array_height)
                for tile_x in range(tiles_x):
                    col_start = tile_x * self.tile_size
                    col_end = min(col_start + self.tile_size, array_width)
                    tile_rgba = np.zeros((self.tile_size, self.tile_size, 4), dtype=np.uint8)
                    tile_rgba[: row_end - row_start, : col_end - col_start, :] = zoom_rgba[row_start:row_end, col_start:col_end, :]
                    relpath = self._tile_relpath(
                        layer_key=layer_key,
                        date_str=date_str,
                        zoom=zoom,
                        tile_x=tile_x,
                        tile_y=tile_y,
                    )
                    self._write_tile(self.asset_root / relpath, tile_rgba)

    def build(self) -> None:
        if self.clear_output_root and self.asset_root.exists():
            log(f"Clearing output root {self.asset_root}")
            shutil.rmtree(self.asset_root)
        self.asset_root.mkdir(parents=True, exist_ok=True)

        manifest = {
            "dataset_label": self.dataset_label,
            "initial_date": self.initial_date,
            "dates": self.selected_dates,
            "grid_crs": self.grid_crs,
            "grid_shape": {
                "height": int(self.y_values.size),
                "width": int(self.x_values.size),
            },
            "grid_resolution": {
                "dx": self.pixel_width,
                "dy": self.pixel_height,
            },
            "grid_extent": self.grid_extent,
            "geo_bounds": self.geo_bounds,
            "tiles": {
                "tile_size": self.tile_size,
                "min_zoom": self.min_zoom,
                "max_zoom": self.max_zoom,
                "origin": self.origin,
                "resolutions": self.tile_resolutions,
                "view_resolutions": self.view_resolutions,
            },
            "layers": {},
        }

        for layer_key, layer_cfg in self.layers.items():
            manifest["layers"][layer_key] = {
                "label": str(layer_cfg["label"]),
                "unit": str(layer_cfg["unit"]),
                "min": float(layer_cfg["min"]),
                "max": float(layer_cfg["max"]),
                "palette": layer_cfg["palette"],
                "stops": layer_cfg["stops"],
                "tile_root_template": f"tiles/{layer_key}" + "/{date}/{z}/{x}/{y}.png",
                "tile_counts": {},
            }
            manifest_layer = manifest["layers"][layer_key]

            for time_idx, date_str in zip(self.selected_indices, self.selected_dates):
                log(f"Rendering {layer_key} for {date_str}")
                rgba = self._rgba_for_layer(layer_key=layer_key, time_idx=time_idx)
                self._build_tiles_for_image(
                    rgba=rgba,
                    layer_key=layer_key,
                    date_str=date_str,
                    manifest_layer=manifest_layer,
                )

        manifest_path = self.asset_root / self.manifest_filename
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log(f"Wrote manifest to {manifest_path}")


def main() -> None:
    cfg = load_config()
    builder = ViewerAssetBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()
