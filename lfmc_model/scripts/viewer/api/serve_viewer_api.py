#!/usr/bin/env python3

import json
import mimetypes
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple

import dask
import numpy as np
import xarray as xr
import yaml
from pyproj import Transformer


here = Path(__file__).resolve().parent
viewer_root = here.parent
config_path = viewer_root / "viewer_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def load_config() -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_float(value):
    return float(value) if np.isfinite(value) else None


def datetime64_to_datestr(value) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


class ViewerDataset:
    def __init__(self, cfg: Dict[str, object]):
        data_cfg = cfg["data"]
        assets_cfg = cfg["assets"]

        self.dataset_label = str(data_cfg["dataset_label"])
        self.data_source = str(data_cfg.get("data_source", "local")).strip().lower()
        self.local_dataset_path = str(data_cfg.get("local_dataset_path", "")).strip()
        self.source_store = str(data_cfg.get("source_store", "")).strip()
        self.source_endpoint_url = str(data_cfg.get("source_endpoint_url", "")).strip()
        self.grid_crs = str(data_cfg["grid_crs"])
        self.display_variable = str(data_cfg["display_variable"])
        self.spread_variable = str(data_cfg["spread_variable"])
        self.quality_variable = str(data_cfg["quality_variable"])
        self.landcover_variable = str(data_cfg["landcover_variable"])
        self.initial_date = str(data_cfg["initial_date"])

        self.asset_mode = str(assets_cfg.get("asset_mode", "local")).strip().lower()
        self.asset_root = Path(str(assets_cfg["local_asset_root"])).resolve()
        self.manifest_filename = str(assets_cfg["manifest_filename"])

        self.dataset_path = self._dataset_path_label()
        self.ds = self._open_dataset()
        self.grid_to_wgs84 = Transformer.from_crs(self.grid_crs, "EPSG:4326", always_xy=True)
        self.wgs84_to_grid = Transformer.from_crs("EPSG:4326", self.grid_crs, always_xy=True)
        self.dates = [datetime64_to_datestr(value) for value in self.ds["time"].values]
        self.x_values = np.asarray(self.ds["x"].values, dtype=np.float64)
        self.y_values = np.asarray(self.ds["y"].values, dtype=np.float64)
        self.dx = float(np.median(np.diff(self.x_values)))
        self.dy = float(np.median(np.diff(self.y_values)))
        self.pixel_width = abs(self.dx)
        self.pixel_height = abs(self.dy)
        self.grid_extent = self._grid_extent()

        self.lat_array = self.ds["lat"]
        self.lon_array = self.ds["lon"]
        self.landcover_da = self.ds[self.landcover_variable]
        self.landcover_years = np.asarray(self.landcover_da["landcover_year"].values)
        self.landcover_labels = self._landcover_mapping()
        self.quality_labels = self._quality_mapping()

    def _dataset_path_label(self) -> str:
        if self.data_source == "source":
            if not self.source_store:
                raise ValueError("viewer_config data.source_store is required when data_source=source")
            return self.source_store
        if self.data_source == "local":
            if not self.local_dataset_path:
                raise ValueError("viewer_config data.local_dataset_path is required when data_source=local")
            return self.local_dataset_path
        raise ValueError(f"Unsupported data_source {self.data_source!r}; expected 'local' or 'source'")

    def _open_dataset(self) -> xr.Dataset:
        if self.data_source == "source":
            storage_options = {"anon": True}
            if self.source_endpoint_url:
                storage_options["client_kwargs"] = {"endpoint_url": self.source_endpoint_url}
            return xr.open_zarr(
                self.source_store,
                consolidated=False,
                storage_options=storage_options,
            )
        return xr.open_zarr(self.local_dataset_path, consolidated=False)

    def _grid_extent(self) -> Dict[str, float]:
        return {
            "west": float(self.x_values[0] - self.pixel_width / 2.0),
            "east": float(self.x_values[-1] + self.pixel_width / 2.0),
            "north": float(self.y_values[0] + self.pixel_height / 2.0),
            "south": float(self.y_values[-1] - self.pixel_height / 2.0),
        }

    def _landcover_mapping(self) -> Dict[int, str]:
        code_to_name = self.landcover_da.attrs.get("code_to_name", {})
        nodata_code = self.landcover_da.attrs.get("nodata_code")
        if isinstance(code_to_name, dict):
            mapping = {int(key): str(value) for key, value in code_to_name.items()}
            if nodata_code is not None:
                mapping[int(nodata_code)] = "nodata"
            return mapping
        dataset_key = self.ds.attrs.get("dominant_landcover_code_key")
        if isinstance(dataset_key, str):
            parsed = json.loads(dataset_key)
            mapping = {int(key): str(value) for key, value in parsed.items()}
            if nodata_code is not None:
                mapping[int(nodata_code)] = "nodata"
            return mapping
        return {}

    def _quality_mapping(self) -> Dict[int, str]:
        values = self.ds.attrs.get("quality_flag_values")
        if isinstance(values, dict):
            return {int(value): str(key) for key, value in values.items()}
        flag_values = self.ds[self.quality_variable].attrs.get("flag_values", [])
        flag_meanings = str(self.ds[self.quality_variable].attrs.get("flag_meanings", "")).split()
        return {int(value): meaning for value, meaning in zip(flag_values, flag_meanings)}

    def manifest_path(self) -> Path:
        return self.asset_root / self.manifest_filename

    def metadata(self) -> Dict[str, object]:
        return {
            "dataset_label": self.dataset_label,
            "data_source": self.data_source,
            "dataset_path": self.dataset_path,
            "initial_date": self.initial_date,
            "asset_mode": self.asset_mode,
            "asset_root": str(self.asset_root),
            "manifest_path": str(self.manifest_path()),
            "dates": self.dates,
            "grid_crs": self.grid_crs,
            "grid_extent": self.grid_extent,
            "grid_resolution": {
                "dx": self.pixel_width,
                "dy": self.pixel_height,
            },
        }

    def _date_index(self, date_str: str) -> int:
        try:
            return self.dates.index(date_str)
        except ValueError as exc:
            raise ValueError(f"Date not available: {date_str}") from exc

    def _lat_value(self, y_idx: int, x_idx: int):
        return safe_float(np.asarray(self.lat_array.isel(y=y_idx, x=x_idx).values).item())

    def _lon_value(self, y_idx: int, x_idx: int):
        return safe_float(np.asarray(self.lon_array.isel(y=y_idx, x=x_idx).values).item())

    def _landcover_year_index(self, date_str: str) -> int:
        year = int(date_str[:4])
        if self.landcover_years.size == 1:
            return 0
        diffs = np.abs(self.landcover_years.astype(np.int64) - year)
        return int(np.argmin(diffs))

    def _cell_index_for_grid_xy(self, grid_x: float, grid_y: float) -> Tuple[int, int]:
        west = self.grid_extent["west"]
        east = self.grid_extent["east"]
        north = self.grid_extent["north"]
        south = self.grid_extent["south"]
        if grid_x < west or grid_x > east or grid_y < south or grid_y > north:
            raise ValueError("Requested point is outside the LFMC grid extent")

        x_idx = int(np.floor((grid_x - west) / self.pixel_width))
        y_idx = int(np.floor((north - grid_y) / self.pixel_height))
        x_idx = min(max(x_idx, 0), self.x_values.size - 1)
        y_idx = min(max(y_idx, 0), self.y_values.size - 1)
        return x_idx, y_idx

    def _cell_bounds(self, x_idx: int, y_idx: int) -> Dict[str, float]:
        center_x = float(self.x_values[x_idx])
        center_y = float(self.y_values[y_idx])
        return {
            "west": center_x - self.pixel_width / 2.0,
            "east": center_x + self.pixel_width / 2.0,
            "south": center_y - self.pixel_height / 2.0,
            "north": center_y + self.pixel_height / 2.0,
        }

    def point_payload(self, date_str: str, grid_x: float = None, grid_y: float = None, lat: float = None, lon: float = None) -> Dict[str, object]:
        time_idx = self._date_index(date_str)
        if grid_x is None or grid_y is None:
            if lat is None or lon is None:
                raise ValueError("Provide either grid x/y or lat/lon")
            grid_x, grid_y = self.wgs84_to_grid.transform(lon, lat)
        requested_lon, requested_lat = self.grid_to_wgs84.transform(grid_x, grid_y)
        x_idx, y_idx = self._cell_index_for_grid_xy(grid_x=float(grid_x), grid_y=float(grid_y))
        cell_bounds = self._cell_bounds(x_idx=x_idx, y_idx=y_idx)

        mean_series = np.asarray(
            self.ds[self.display_variable].isel(y=y_idx, x=x_idx).values,
            dtype=np.float32,
        )
        std_series = np.asarray(
            self.ds[self.spread_variable].isel(y=y_idx, x=x_idx).values,
            dtype=np.float32,
        )
        quality_series = np.asarray(self.ds[self.quality_variable].values, dtype=np.uint8)
        landcover_year_idx = self._landcover_year_index(date_str)
        landcover_code = int(
            np.asarray(
                self.landcover_da.isel(landcover_year=landcover_year_idx, y=y_idx, x=x_idx).values
            ).item()
        )

        center_x = float(self.x_values[x_idx])
        center_y = float(self.y_values[y_idx])
        center_lon, center_lat = self.grid_to_wgs84.transform(center_x, center_y)

        return {
            "date": date_str,
            "requested_grid_x": float(grid_x),
            "requested_grid_y": float(grid_y),
            "requested_lat": safe_float(requested_lat),
            "requested_lon": safe_float(requested_lon),
            "cell_center_x": center_x,
            "cell_center_y": center_y,
            "nearest_lat": self._lat_value(y_idx, x_idx),
            "nearest_lon": self._lon_value(y_idx, x_idx),
            "cell_center_lat": safe_float(center_lat),
            "cell_center_lon": safe_float(center_lon),
            "cell_bounds": cell_bounds,
            "cell_index": {
                "x": int(x_idx),
                "y": int(y_idx),
            },
            "lfmc_ens_mean": safe_float(mean_series[time_idx]),
            "lfmc_ens_std": safe_float(std_series[time_idx]),
            "quality_flag": int(quality_series[time_idx]),
            "data_product_level": self.quality_labels.get(int(quality_series[time_idx]), "unknown"),
            "landcover_code": landcover_code,
            "landcover_name": self.landcover_labels.get(landcover_code, "unknown"),
            "timeseries": {
                "dates": self.dates,
                "lfmc_ens_mean": [safe_float(value) for value in mean_series],
                "lfmc_ens_std": [safe_float(value) for value in std_series],
                "quality_flag": [int(value) for value in quality_series],
            },
        }

    def resolve_asset_path(self, request_path: str) -> Path:
        relpath = request_path.removeprefix("/viewer-assets/").strip("/")
        candidate = (self.asset_root / relpath).resolve()
        if not str(candidate).startswith(str(self.asset_root)):
            raise ValueError("Attempted path traversal outside asset root")
        return candidate


cfg = load_config()
dask.config.set(scheduler="synchronous")
viewer_dataset = ViewerDataset(cfg)


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server_version = "LongLFMCViewer/0.3"

    def log_message(self, format_string: str, *args) -> None:
        print(timestamped_message(format_string % args), flush=True)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path.startswith("/viewer-assets/"):
                self._static_response(viewer_dataset.resolve_asset_path(parsed.path), send_body=False)
                return
            if parsed.path == "/api/health":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._json_response({"status": "ok"})
                return
            if parsed.path == "/api/metadata":
                self._json_response(viewer_dataset.metadata())
                return
            if parsed.path == "/api/point":
                date_str = self._require_param(query, "date")
                grid_x = self._optional_float(query, "x")
                grid_y = self._optional_float(query, "y")
                lat = self._optional_float(query, "lat")
                lon = self._optional_float(query, "lon")
                self._json_response(
                    viewer_dataset.point_payload(
                        date_str=date_str,
                        grid_x=grid_x,
                        grid_y=grid_y,
                        lat=lat,
                        lon=lon,
                    )
                )
                return
            if parsed.path.startswith("/viewer-assets/"):
                self._static_response(viewer_dataset.resolve_asset_path(parsed.path))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _require_param(self, query: Dict[str, List[str]], name: str) -> str:
        values = query.get(name)
        if not values or not values[0]:
            raise ValueError(f"Missing required query parameter: {name}")
        return values[0]

    def _optional_float(self, query: Dict[str, List[str]], name: str):
        values = query.get(name)
        if not values or values[0] == "":
            return None
        return float(values[0])

    def _json_response(self, payload: Dict[str, object], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static_response(self, asset_path: Path, send_body: bool = True) -> None:
        if not asset_path.exists() or not asset_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
            return

        mime_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        body = asset_path.read_bytes() if send_body else b""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(asset_path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(body)


def main() -> None:
    server_cfg = cfg["server"]
    host = str(server_cfg["host"])
    port = int(server_cfg["port"])
    server = ThreadingHTTPServer((host, port), ViewerRequestHandler)
    print(timestamped_message(f"Serving viewer API at http://{host}:{port}"), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
