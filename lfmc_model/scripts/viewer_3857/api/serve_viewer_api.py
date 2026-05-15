#!/usr/bin/env python3

import csv
import hmac
import io
import json
import mimetypes
import os
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Dict, List, Optional, Tuple

try:
    from http.server import ThreadingHTTPServer
except ImportError:
    from http.server import HTTPServer

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True


class DatasetLoadingError(RuntimeError):
    pass


here = Path(__file__).resolve().parent
viewer_root = here.parent
config_path = viewer_root / "viewer_config.yaml"
DATASET_LOAD_WAIT_SECONDS = 45.0
DEFAULT_POINT_TIMESERIES_DAYS = 365
MAX_POINT_TIMESERIES_DAYS = 366

dask = None
np = None
xr = None
yaml = None
Transformer = None
runtime_dependencies_loaded = False
runtime_dependencies_lock = threading.Lock()


def load_runtime_dependencies() -> None:
    global dask, np, xr, yaml, Transformer, runtime_dependencies_loaded

    if runtime_dependencies_loaded:
        return

    with runtime_dependencies_lock:
        if runtime_dependencies_loaded:
            return

        import dask as dask_module
        import numpy as np_module
        import xarray as xr_module
        from pyproj import Transformer as transformer_class

        dask = dask_module
        np = np_module
        xr = xr_module
        Transformer = transformer_class
        dask.config.set(scheduler="synchronous")
        runtime_dependencies_loaded = True


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def load_config() -> Dict[str, object]:
    global yaml

    if yaml is None:
        import yaml as yaml_module

        yaml = yaml_module
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def safe_float(value):
    return float(value) if np.isfinite(value) else None


def datetime64_to_datestr(value) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


def join_url_parts(base_url: str, relpath: str) -> str:
    return f"{base_url.rstrip('/')}/{relpath.lstrip('/')}"


def nearest_index(sorted_values, target_value: float) -> int:
    values = np.asarray(sorted_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("nearest_index requires a non-empty 1D coordinate array")

    ascending = values[0] <= values[-1]
    work_values = values if ascending else values[::-1]
    insert_idx = int(np.searchsorted(work_values, target_value, side="left"))

    if insert_idx <= 0:
        nearest_work_idx = 0
    elif insert_idx >= work_values.size:
        nearest_work_idx = work_values.size - 1
    else:
        left_idx = insert_idx - 1
        right_idx = insert_idx
        left_diff = abs(target_value - work_values[left_idx])
        right_diff = abs(work_values[right_idx] - target_value)
        nearest_work_idx = left_idx if left_diff <= right_diff else right_idx

    if ascending:
        return nearest_work_idx
    return values.size - 1 - nearest_work_idx


def open_dataset_for_config(data_cfg: Dict[str, object]):
    data_source = str(data_cfg.get("data_source", "local")).strip().lower()
    local_dataset_path = str(data_cfg.get("local_dataset_path", "")).strip()
    source_store = str(data_cfg.get("source_store", "")).strip()
    source_endpoint_url = str(data_cfg.get("source_endpoint_url", "")).strip()

    if data_source == "source":
        if not source_store:
            raise ValueError("source_store is required when data_source=source")
        storage_options = {"anon": True}
        if source_endpoint_url:
            storage_options["client_kwargs"] = {"endpoint_url": source_endpoint_url}
        consolidated = bool(data_cfg.get("consolidated", False))
        return xr.open_zarr(
            source_store,
            consolidated=consolidated,
            storage_options=storage_options,
        )

    if data_source == "local":
        if not local_dataset_path:
            raise ValueError("local_dataset_path is required when data_source=local")
        consolidated = bool(data_cfg.get("consolidated", False))
        return xr.open_zarr(local_dataset_path, consolidated=consolidated)

    raise ValueError(f"Unsupported data_source {data_source!r}; expected 'local' or 'source'")


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
        self.uncertainty_variable = str(data_cfg["uncertainty_variable"])
        self.quality_variable = str(data_cfg["quality_variable"])
        self.landcover_variable = str(data_cfg["landcover_variable"])
        self.initial_date = str(data_cfg["initial_date"])

        self.asset_mode = str(assets_cfg.get("asset_mode", "local")).strip().lower()
        self.asset_root = Path(str(assets_cfg["local_asset_root"])).resolve()
        self.source_asset_base_url = str(assets_cfg.get("source_asset_base_url", "")).strip()
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

    def _open_dataset(self):
        return open_dataset_for_config(
            {
                "data_source": self.data_source,
                "local_dataset_path": self.local_dataset_path,
                "source_store": self.source_store,
                "source_endpoint_url": self.source_endpoint_url,
                "consolidated": True if self.data_source == "source" else False,
            }
        )

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
        if self.asset_mode == "source":
            asset_base_url = self.source_asset_base_url
            manifest_url = join_url_parts(asset_base_url, self.manifest_filename)
        else:
            asset_base_url = "/viewer-assets"
            manifest_url = f"{asset_base_url}/{self.manifest_filename}"
        return {
            "dataset_label": self.dataset_label,
            "data_source": self.data_source,
            "dataset_path": self.dataset_path,
            "initial_date": self.initial_date,
            "asset_mode": self.asset_mode,
            "asset_root": str(self.asset_root),
            "manifest_path": str(self.manifest_path()),
            "asset_base_url": asset_base_url,
            "asset_manifest_url": manifest_url,
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

    def _value_for_cell(self, variable_name: str, time_idx: int, y_idx: int, x_idx: int):
        return np.asarray(
            self.ds[variable_name].isel(time=time_idx, y=y_idx, x=x_idx).values
        ).item()

    def _series_for_cell(
        self,
        variable_name: str,
        start_idx: int,
        end_idx: int,
        y_idx: int,
        x_idx: int,
    ) -> Tuple[float, ...]:
        values = np.asarray(
            self.ds[variable_name].isel(time=slice(start_idx, end_idx), y=y_idx, x=x_idx).values,
            dtype=np.float32,
        )
        return tuple(float(value) for value in values)

    def _quality_value(self, time_idx: int) -> int:
        return int(np.asarray(self.ds[self.quality_variable].isel(time=time_idx).values).item())

    def _quality_series_for_window(self, start_idx: int, end_idx: int) -> Tuple[int, ...]:
        values = np.asarray(self.ds[self.quality_variable].isel(time=slice(start_idx, end_idx)).values)
        if values.ndim != 1:
            raise ValueError(
                f"Expected viewer quality variable {self.quality_variable!r} to be 1D over time; "
                f"found shape {values.shape}"
            )
        return tuple(int(value) for value in values)

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

    def point_payload(
        self,
        date_str: str,
        grid_x: float = None,
        grid_y: float = None,
        lat: float = None,
        lon: float = None,
        include_timeseries: bool = False,
        timeseries_days: int = DEFAULT_POINT_TIMESERIES_DAYS,
    ) -> Dict[str, object]:
        time_idx = self._date_index(date_str)
        if grid_x is None or grid_y is None:
            if lat is None or lon is None:
                raise ValueError("Provide either grid x/y or lat/lon")
            grid_x, grid_y = self.wgs84_to_grid.transform(lon, lat)
        requested_lon, requested_lat = self.grid_to_wgs84.transform(grid_x, grid_y)
        x_idx, y_idx = self._cell_index_for_grid_xy(grid_x=float(grid_x), grid_y=float(grid_y))
        cell_bounds = self._cell_bounds(x_idx=x_idx, y_idx=y_idx)
        mean_value = safe_float(self._value_for_cell(self.display_variable, time_idx, y_idx, x_idx))
        uncertainty_value = safe_float(self._value_for_cell(self.uncertainty_variable, time_idx, y_idx, x_idx))
        landcover_year_idx = self._landcover_year_index(date_str)
        raw_landcover_value = np.asarray(
            self.landcover_da.isel(landcover_year=landcover_year_idx, y=y_idx, x=x_idx).values
        ).item()
        landcover_code = safe_float(raw_landcover_value)
        if landcover_code is None:
            log(
                "Warning: missing landcover for point query "
                f"date={date_str} x_idx={x_idx} y_idx={y_idx} "
                f"grid_x={float(grid_x):.2f} grid_y={float(grid_y):.2f} "
                f"lfmc={mean_value}"
            )
        else:
            landcover_code = int(landcover_code)

        center_x = float(self.x_values[x_idx])
        center_y = float(self.y_values[y_idx])
        center_lon, center_lat = self.grid_to_wgs84.transform(center_x, center_y)
        quality_value = self._quality_value(time_idx)

        payload = {
            "date": date_str,
            "requested_grid_x": float(grid_x),
            "requested_grid_y": float(grid_y),
            "requested_lat": safe_float(requested_lat),
            "requested_lon": safe_float(requested_lon),
            "cell_center_x": center_x,
            "cell_center_y": center_y,
            "nearest_lat": safe_float(center_lat),
            "nearest_lon": safe_float(center_lon),
            "cell_center_lat": safe_float(center_lat),
            "cell_center_lon": safe_float(center_lon),
            "cell_bounds": cell_bounds,
            "cell_index": {
                "x": int(x_idx),
                "y": int(y_idx),
            },
            "lfmc_ens_mean": mean_value,
            "lfmc_ens_std": uncertainty_value,
            "quality_flag": quality_value,
            "data_product_level": self.quality_labels.get(quality_value, "unknown"),
            "landcover_code": landcover_code,
            "landcover_name": self.landcover_labels.get(landcover_code, "unknown") if landcover_code is not None else "unknown",
            "timeseries": None,
        }

        if include_timeseries:
            bounded_days = min(max(int(timeseries_days), 1), MAX_POINT_TIMESERIES_DAYS)
            start_idx = max(0, time_idx - bounded_days + 1)
            end_idx = time_idx + 1
            mean_series = self._series_for_cell(
                self.display_variable,
                start_idx=start_idx,
                end_idx=end_idx,
                y_idx=y_idx,
                x_idx=x_idx,
            )
            uncertainty_series = self._series_for_cell(
                self.uncertainty_variable,
                start_idx=start_idx,
                end_idx=end_idx,
                y_idx=y_idx,
                x_idx=x_idx,
            )
            quality_series = self._quality_series_for_window(start_idx=start_idx, end_idx=end_idx)
            payload["timeseries"] = {
                "dates": self.dates[start_idx:end_idx],
                "lfmc_ens_mean": [safe_float(value) for value in mean_series],
                "lfmc_ens_std": [safe_float(value) for value in uncertainty_series],
                "quality_flag": [int(value) for value in quality_series],
                "window_days": bounded_days,
            }

        return payload

    def resolve_asset_path(self, request_path: str) -> Path:
        relpath = request_path.removeprefix("/viewer-assets/").strip("/")
        candidate = (self.asset_root / relpath).resolve()
        if not str(candidate).startswith(str(self.asset_root)):
            raise ValueError("Attempted path traversal outside asset root")
        return candidate

    def close(self) -> None:
        self.ds.close()


class ScientificDataset:
    def __init__(self, cfg: Dict[str, object]):
        data_cfg = cfg["scientific_data"]

        self.data_source = str(data_cfg.get("data_source", "local")).strip().lower()
        self.local_dataset_path = str(data_cfg.get("local_dataset_path", "")).strip()
        self.source_store = str(data_cfg.get("source_store", "")).strip()
        self.source_endpoint_url = str(data_cfg.get("source_endpoint_url", "")).strip()
        self.grid_crs = str(data_cfg["grid_crs"])
        self.display_variable = str(data_cfg["display_variable"])
        self.uncertainty_variable = str(data_cfg["uncertainty_variable"])
        self.quality_variable = str(data_cfg["quality_variable"])

        self.ds = open_dataset_for_config(
            {
                "data_source": self.data_source,
                "local_dataset_path": self.local_dataset_path,
                "source_store": self.source_store,
                "source_endpoint_url": self.source_endpoint_url,
                "consolidated": bool(data_cfg.get("consolidated", False)),
            }
        )
        self.wgs84_to_grid = Transformer.from_crs("EPSG:4326", self.grid_crs, always_xy=True)
        self.grid_to_wgs84 = Transformer.from_crs(self.grid_crs, "EPSG:4326", always_xy=True)
        self.dates = [datetime64_to_datestr(value) for value in self.ds["time"].values]
        self.date_to_index = {date_str: idx for idx, date_str in enumerate(self.dates)}
        self.x_values = np.asarray(self.ds["x"].values, dtype=np.float64)
        self.y_values = np.asarray(self.ds["y"].values, dtype=np.float64)

    def _date_index(self, date_str: str) -> int:
        try:
            return self.dates.index(date_str)
        except ValueError as exc:
            raise ValueError(f"Date not available in scientific dataset: {date_str}") from exc

    def _cell_for_latlon(self, lat: float, lon: float) -> Tuple[int, int, float, float]:
        grid_x, grid_y = self.wgs84_to_grid.transform(lon, lat)
        x_idx = nearest_index(self.x_values, float(grid_x))
        y_idx = nearest_index(self.y_values, float(grid_y))
        center_x = float(self.x_values[x_idx])
        center_y = float(self.y_values[y_idx])
        return x_idx, y_idx, center_x, center_y

    def download_csv_bytes_for_sites(
        self,
        sites: List[Tuple[float, float]],
        start_date: str,
        end_date: str,
    ) -> Tuple[bytes, str]:
        start_idx = self._date_index(start_date)
        end_idx = self._date_index(end_date)
        if end_idx < start_idx:
            raise ValueError("end_date must be on or after start_date")
        if not sites:
            raise ValueError("At least one site is required for CSV download")
        if len(sites) > 10:
            raise ValueError("CSV download supports at most 10 sites")

        time_slice = slice(start_idx, end_idx + 1)
        date_values = self.dates[start_idx : end_idx + 1]
        quality_values = np.asarray(self.ds[self.quality_variable].isel(time=time_slice).values)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "site_index",
                "site_lat",
                "site_lon",
                "date",
                "lfmc_ens_mean",
                "lfmc_ens_std",
                "quality_flag",
            ]
        )

        for site_idx, (lat, lon) in enumerate(sites, start=1):
            x_idx, y_idx, center_x, center_y = self._cell_for_latlon(lat=lat, lon=lon)
            center_lon, center_lat = self.grid_to_wgs84.transform(center_x, center_y)
            mean_values = np.asarray(
                self.ds[self.display_variable].isel(time=time_slice, y=y_idx, x=x_idx).values,
                dtype=np.float32,
            )
            uncertainty_values = np.asarray(
                self.ds[self.uncertainty_variable].isel(time=time_slice, y=y_idx, x=x_idx).values,
                dtype=np.float32,
            )

            for time_idx, date_str in enumerate(date_values):
                mean_value = safe_float(mean_values[time_idx])
                uncertainty_value = safe_float(uncertainty_values[time_idx])
                quality_raw = quality_values[time_idx]
                quality_value = int(quality_raw) if np.isfinite(quality_raw) else ""
                writer.writerow(
                    [
                        site_idx,
                        f"{float(center_lat):.6f}",
                        f"{float(center_lon):.6f}",
                        date_str,
                        "" if mean_value is None else f"{mean_value:.6f}",
                        "" if uncertainty_value is None else f"{uncertainty_value:.6f}",
                        quality_value,
                    ]
                )

        filename = f"lfmc_sites_{start_date}_to_{end_date}.csv"
        return buffer.getvalue().encode("utf-8"), filename

    def close(self) -> None:
        self.ds.close()


def build_viewer_dataset(fresh_cfg: Dict[str, object]) -> ViewerDataset:
    load_runtime_dependencies()
    return ViewerDataset(fresh_cfg)


def build_scientific_dataset(fresh_cfg: Dict[str, object]) -> ScientificDataset:
    load_runtime_dependencies()
    return ScientificDataset(fresh_cfg)


def build_datasets() -> Tuple[Dict[str, object], ViewerDataset, ScientificDataset]:
    fresh_cfg = load_config()
    return fresh_cfg, build_viewer_dataset(fresh_cfg), build_scientific_dataset(fresh_cfg)


cfg: Dict[str, object] = {}
viewer_dataset: Optional[ViewerDataset] = None
scientific_dataset: Optional[ScientificDataset] = None
dataset_lock = threading.RLock()
dataset_condition = threading.Condition(dataset_lock)
viewer_dataset_loading = False
scientific_dataset_loading = False
last_refresh_time = 0.0
last_refresh_error: Optional[str] = None


def refresh_datasets(reason: str = "manual") -> Dict[str, object]:
    global cfg, viewer_dataset, scientific_dataset
    global viewer_dataset_loading, scientific_dataset_loading
    global last_refresh_time, last_refresh_error

    log(f"Unloading datasets reason={reason}")
    with dataset_condition:
        old_viewer_dataset = viewer_dataset
        old_scientific_dataset = scientific_dataset
        cfg = load_config()
        viewer_dataset = None
        scientific_dataset = None
        viewer_dataset_loading = False
        scientific_dataset_loading = False
        last_refresh_time = time.time()
        last_refresh_error = None
        dataset_condition.notify_all()

    if old_viewer_dataset is not None:
        old_viewer_dataset.close()
    if old_scientific_dataset is not None:
        old_scientific_dataset.close()

    payload = {
        "status": "refreshed",
        "reason": reason,
        "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_refresh_time)),
        "viewer_dataset_loaded": False,
        "scientific_dataset_loaded": False,
    }
    log("Dataset handles unloaded; zarrs will reopen on demand")
    return payload


def load_viewer_dataset(reason: str) -> None:
    global cfg, viewer_dataset, viewer_dataset_loading, last_refresh_error

    try:
        log(f"Opening viewer dataset reason={reason}")
        fresh_cfg = load_config()
        fresh_viewer_dataset = build_viewer_dataset(fresh_cfg)
        with dataset_condition:
            old_viewer_dataset = viewer_dataset
            cfg = fresh_cfg
            viewer_dataset = fresh_viewer_dataset
            viewer_dataset_loading = False
            last_refresh_error = None
            dataset_condition.notify_all()
        if old_viewer_dataset is not None:
            old_viewer_dataset.close()
        log(f"Viewer dataset open complete viewer_dates={len(fresh_viewer_dataset.dates)}")
    except Exception as exc:
        with dataset_condition:
            viewer_dataset_loading = False
            last_refresh_error = str(exc)
            dataset_condition.notify_all()
        log(f"Viewer dataset open failed reason={reason}: {exc}")


def load_scientific_dataset(reason: str) -> None:
    global cfg, scientific_dataset, scientific_dataset_loading, last_refresh_error

    try:
        log(f"Opening scientific dataset reason={reason}")
        fresh_cfg = load_config()
        fresh_scientific_dataset = build_scientific_dataset(fresh_cfg)
        with dataset_condition:
            old_scientific_dataset = scientific_dataset
            cfg = fresh_cfg
            scientific_dataset = fresh_scientific_dataset
            scientific_dataset_loading = False
            last_refresh_error = None
            dataset_condition.notify_all()
        if old_scientific_dataset is not None:
            old_scientific_dataset.close()
        log(f"Scientific dataset open complete scientific_dates={len(fresh_scientific_dataset.dates)}")
    except Exception as exc:
        with dataset_condition:
            scientific_dataset_loading = False
            last_refresh_error = str(exc)
            dataset_condition.notify_all()
        log(f"Scientific dataset open failed reason={reason}: {exc}")


def start_viewer_dataset_load(reason: str) -> None:
    global viewer_dataset_loading, last_refresh_error

    with dataset_condition:
        if viewer_dataset is not None or viewer_dataset_loading:
            return
        viewer_dataset_loading = True
        last_refresh_error = None
        dataset_condition.notify_all()

    thread = threading.Thread(target=load_viewer_dataset, args=(reason,), daemon=True)
    thread.start()


def start_scientific_dataset_load(reason: str) -> None:
    global scientific_dataset_loading, last_refresh_error

    with dataset_condition:
        if scientific_dataset is not None or scientific_dataset_loading:
            return
        scientific_dataset_loading = True
        last_refresh_error = None
        dataset_condition.notify_all()

    thread = threading.Thread(target=load_scientific_dataset, args=(reason,), daemon=True)
    thread.start()


def require_loaded_viewer_dataset() -> ViewerDataset:
    with dataset_lock:
        if viewer_dataset is None:
            if last_refresh_error:
                raise RuntimeError(f"Viewer dataset is not loaded; last refresh error: {last_refresh_error}")
            raise RuntimeError("Viewer dataset is still loading")
        return viewer_dataset


def require_loaded_scientific_dataset() -> ScientificDataset:
    with dataset_lock:
        if scientific_dataset is None:
            if last_refresh_error:
                raise RuntimeError(f"Scientific dataset is not loaded; last refresh error: {last_refresh_error}")
            raise RuntimeError("Scientific dataset is still loading")
        return scientific_dataset


def wait_for_viewer_dataset(timeout_seconds: float = DATASET_LOAD_WAIT_SECONDS) -> ViewerDataset:
    if viewer_dataset is None and not viewer_dataset_loading:
        start_viewer_dataset_load(reason="on_demand_viewer")

    deadline = time.time() + timeout_seconds
    with dataset_condition:
        while viewer_dataset is None:
            if last_refresh_error and not viewer_dataset_loading:
                raise RuntimeError(f"Viewer dataset is not loaded; last refresh error: {last_refresh_error}")
            remaining = deadline - time.time()
            if remaining <= 0:
                raise DatasetLoadingError("Viewer dataset is still loading")
            dataset_condition.wait(timeout=remaining)
        return viewer_dataset


def wait_for_scientific_dataset(timeout_seconds: float = DATASET_LOAD_WAIT_SECONDS) -> ScientificDataset:
    if scientific_dataset is None and not scientific_dataset_loading:
        start_scientific_dataset_load(reason="on_demand_scientific")

    deadline = time.time() + timeout_seconds
    with dataset_condition:
        while scientific_dataset is None:
            if last_refresh_error and not scientific_dataset_loading:
                raise RuntimeError(f"Scientific dataset is not loaded; last refresh error: {last_refresh_error}")
            remaining = deadline - time.time()
            if remaining <= 0:
                raise DatasetLoadingError("Scientific dataset is still loading")
            dataset_condition.wait(timeout=remaining)
        return scientific_dataset


def should_refresh_for_date_error(exc: Exception) -> bool:
    return "Date not available" in str(exc)


def config_metadata_payload() -> Dict[str, object]:
    fresh_cfg = load_config()
    data_cfg = fresh_cfg["data"]
    assets_cfg = fresh_cfg["assets"]
    asset_mode = str(assets_cfg.get("asset_mode", "local")).strip().lower()
    if asset_mode == "source":
        asset_base_url = str(assets_cfg.get("source_asset_base_url", "")).strip()
        manifest_url = join_url_parts(asset_base_url, str(assets_cfg["manifest_filename"]))
    else:
        asset_base_url = "/viewer-assets"
        manifest_url = f"{asset_base_url}/{assets_cfg['manifest_filename']}"

    return {
        "dataset_label": str(data_cfg["dataset_label"]),
        "data_source": str(data_cfg.get("data_source", "local")),
        "dataset_path": str(data_cfg.get("source_store") or data_cfg.get("local_dataset_path", "")),
        "initial_date": str(data_cfg["initial_date"]),
        "asset_mode": asset_mode,
        "asset_root": str(assets_cfg["local_asset_root"]),
        "manifest_path": str(Path(str(assets_cfg["local_asset_root"])) / str(assets_cfg["manifest_filename"])),
        "asset_base_url": asset_base_url,
        "asset_manifest_url": manifest_url,
        "dates": [],
        "grid_crs": str(data_cfg["grid_crs"]),
        "grid_extent": None,
        "grid_resolution": None,
        "dataset_loaded": False,
        "last_refresh_epoch": last_refresh_time,
    }


def viewer_metadata_payload() -> Dict[str, object]:
    with dataset_lock:
        loaded_viewer_dataset = viewer_dataset
        if loaded_viewer_dataset is not None:
            payload = loaded_viewer_dataset.metadata()
            payload["dataset_loaded"] = True
            payload["last_refresh_epoch"] = last_refresh_time
            return payload

    payload = config_metadata_payload()
    with dataset_lock:
        payload["last_refresh_error"] = last_refresh_error
        return payload


def viewer_point_payload_with_refresh(
    date_str: str,
    grid_x: float = None,
    grid_y: float = None,
    lat: float = None,
    lon: float = None,
    include_timeseries: bool = False,
    timeseries_days: int = DEFAULT_POINT_TIMESERIES_DAYS,
) -> Dict[str, object]:
    try:
        loaded_viewer_dataset = wait_for_viewer_dataset()
        with dataset_lock:
            return loaded_viewer_dataset.point_payload(
                date_str=date_str,
                grid_x=grid_x,
                grid_y=grid_y,
                lat=lat,
                lon=lon,
                include_timeseries=include_timeseries,
                timeseries_days=timeseries_days,
            )
    except ValueError as exc:
        if not should_refresh_for_date_error(exc):
            raise

    refresh_datasets(reason=f"point_date_miss:{date_str}")
    with dataset_lock:
        loaded_viewer_dataset = wait_for_viewer_dataset()
        return loaded_viewer_dataset.point_payload(
            date_str=date_str,
            grid_x=grid_x,
            grid_y=grid_y,
            lat=lat,
            lon=lon,
            include_timeseries=include_timeseries,
            timeseries_days=timeseries_days,
        )


def scientific_csv_with_refresh(
    sites: List[Tuple[float, float]],
    start_date: str,
    end_date: str,
) -> Tuple[bytes, str]:
    try:
        loaded_scientific_dataset = wait_for_scientific_dataset()
        with dataset_lock:
            return loaded_scientific_dataset.download_csv_bytes_for_sites(
                sites=sites,
                start_date=start_date,
                end_date=end_date,
            )
    except ValueError as exc:
        if not should_refresh_for_date_error(exc):
            raise

    refresh_datasets(reason=f"csv_date_miss:{start_date}:{end_date}")
    with dataset_lock:
        loaded_scientific_dataset = wait_for_scientific_dataset()
        return loaded_scientific_dataset.download_csv_bytes_for_sites(
            sites=sites,
            start_date=start_date,
            end_date=end_date,
        )


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server_version = "LongLFMCViewer/0.3"

    def log_message(self, format_string: str, *args) -> None:
        print(timestamped_message(format_string % args), flush=True)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path.startswith("/viewer-assets/"):
                loaded_viewer_dataset = require_loaded_viewer_dataset()
                self._static_response(loaded_viewer_dataset.resolve_asset_path(parsed.path), send_body=False)
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
                with dataset_condition:
                    datasets_loaded = viewer_dataset is not None and scientific_dataset is not None
                    payload = {
                        "status": "ok",
                        "datasets_loaded": datasets_loaded,
                        "viewer_dataset_loaded": viewer_dataset is not None,
                        "scientific_dataset_loaded": scientific_dataset is not None,
                        "viewer_dataset_loading": viewer_dataset_loading,
                        "scientific_dataset_loading": scientific_dataset_loading,
                        "last_refresh_epoch": last_refresh_time,
                        "last_refresh_error": last_refresh_error,
                    }
                self._json_response(payload)
                return
            if parsed.path == "/api/metadata":
                self._json_response(viewer_metadata_payload())
                return
            if parsed.path == "/api/point":
                date_str = self._require_param(query, "date")
                grid_x = self._optional_float(query, "x")
                grid_y = self._optional_float(query, "y")
                lat = self._optional_float(query, "lat")
                lon = self._optional_float(query, "lon")
                include_timeseries = self._optional_bool(query, "include_timeseries", default=False)
                timeseries_days = self._optional_int(
                    query,
                    "timeseries_days",
                    default=DEFAULT_POINT_TIMESERIES_DAYS,
                    minimum=1,
                    maximum=MAX_POINT_TIMESERIES_DAYS,
                )
                payload = viewer_point_payload_with_refresh(
                    date_str=date_str,
                    grid_x=grid_x,
                    grid_y=grid_y,
                    lat=lat,
                    lon=lon,
                    include_timeseries=include_timeseries,
                    timeseries_days=timeseries_days,
                )
                self._json_response(payload)
                return
            if parsed.path == "/api/download_csv":
                start_date = self._require_param(query, "start_date")
                end_date = self._require_param(query, "end_date")
                sites = []
                for raw_site in query.get("site", []):
                    parts = [value.strip() for value in raw_site.split(",", maxsplit=1)]
                    if len(parts) != 2:
                        raise ValueError(f"Invalid site parameter {raw_site!r}; expected 'lat,lon'")
                    sites.append((float(parts[0]), float(parts[1])))
                if not sites:
                    lat = float(self._require_param(query, "lat"))
                    lon = float(self._require_param(query, "lon"))
                    sites = [(lat, lon)]
                csv_bytes, filename = scientific_csv_with_refresh(
                    sites=sites,
                    start_date=start_date,
                    end_date=end_date,
                )
                self._csv_response(csv_bytes, filename=filename)
                return
            if parsed.path.startswith("/viewer-assets/"):
                loaded_viewer_dataset = require_loaded_viewer_dataset()
                self._static_response(loaded_viewer_dataset.resolve_asset_path(parsed.path))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except DatasetLoadingError as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/refresh":
                self._require_refresh_auth()
                payload = refresh_datasets(reason="refresh_endpoint")
                self._json_response(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except PermissionError as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.UNAUTHORIZED)
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _require_refresh_auth(self) -> None:
        expected_token = str(os.environ.get("LONG_LFMC_API_REFRESH_TOKEN", "")).strip()
        if not expected_token:
            raise PermissionError("Refresh endpoint is disabled; LONG_LFMC_API_REFRESH_TOKEN is not set")
        header = str(self.headers.get("Authorization", "")).strip()
        expected_header = f"Bearer {expected_token}"
        if not hmac.compare_digest(header, expected_header):
            raise PermissionError("Invalid refresh token")

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

    def _optional_bool(self, query: Dict[str, List[str]], name: str, default: bool = False) -> bool:
        values = query.get(name)
        if not values or values[0] == "":
            return default
        value = values[0].strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"Invalid boolean query parameter {name}: {values[0]!r}")

    def _optional_int(
        self,
        query: Dict[str, List[str]],
        name: str,
        default: int,
        minimum: int = None,
        maximum: int = None,
    ) -> int:
        values = query.get(name)
        if not values or values[0] == "":
            return default
        value = int(values[0])
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be at most {maximum}")
        return value

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

    def _csv_response(self, body: bytes, filename: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    host = str(os.environ.get("HOST", "0.0.0.0")).strip() or "0.0.0.0"
    port = int(str(os.environ.get("PORT", "8001")).strip())
    server = ThreadingHTTPServer((host, port), ViewerRequestHandler)
    print(timestamped_message(f"Serving viewer API at http://{host}:{port}"), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
