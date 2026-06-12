#!/usr/bin/env python3

import csv
import datetime as dt
import hmac
import io
import json
import mimetypes
import os
import threading
import time
import urllib.parse
import urllib.request
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
DEFAULT_DATASET_KEY = "modis"
DEFAULT_POINT_TIMESERIES_DAYS = 90
MAX_POINT_TIMESERIES_DAYS = 90
MAX_CSV_DOWNLOAD_YEARS = 3
DEFAULT_POINT_CLIMATOLOGY_VARIABLE = "lfmc_climatology_mean_point"

fsspec = None
np = None
yaml = None
zarr = None
Transformer = None
runtime_dependencies_loaded = False
runtime_dependencies_lock = threading.Lock()


def load_runtime_dependencies() -> None:
    global fsspec, np, yaml, zarr, Transformer, runtime_dependencies_loaded

    if runtime_dependencies_loaded:
        return

    with runtime_dependencies_lock:
        if runtime_dependencies_loaded:
            return

        import fsspec as fsspec_module
        import numpy as np_module
        import yaml as yaml_module
        import zarr as zarr_module
        from pyproj import Transformer as transformer_class

        fsspec = fsspec_module
        np = np_module
        yaml = yaml_module
        zarr = zarr_module
        Transformer = transformer_class
        runtime_dependencies_loaded = True


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config() -> Dict[str, object]:
    load_runtime_dependencies()
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value) if np.isfinite(value) else None
    except TypeError:
        return None


def safe_difference(left, right):
    if left is None or right is None:
        return None
    return safe_float(float(left) - float(right))


def datetime64_to_datestr(value) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


def calendar_day_index_365(date_str: str) -> int:
    date_value = dt.date.fromisoformat(date_str)
    if date_value.month == 2 and date_value.day == 29:
        date_value = dt.date(date_value.year, 2, 28)
    return int(dt.date(2001, date_value.month, date_value.day).timetuple().tm_yday - 1)


def max_csv_end_date(start_date: str) -> dt.date:
    parsed_start = dt.date.fromisoformat(start_date)
    try:
        return parsed_start.replace(year=parsed_start.year + MAX_CSV_DOWNLOAD_YEARS)
    except ValueError:
        return parsed_start.replace(year=parsed_start.year + MAX_CSV_DOWNLOAD_YEARS, day=28)


def shift_year(date_value: dt.date, year: int) -> dt.date:
    try:
        return date_value.replace(year=year)
    except ValueError:
        return date_value.replace(year=year, day=28)


def join_url_parts(base_url: str, relpath: str) -> str:
    return f"{base_url.rstrip('/')}/{relpath.lstrip('/')}"


def zarr_attrs(array_or_group) -> Dict[str, object]:
    attrs = getattr(array_or_group, "attrs", {})
    if hasattr(attrs, "asdict"):
        return attrs.asdict()
    return dict(attrs)


def dataset_entries(cfg: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    if "datasets" in cfg:
        return dict(cfg["datasets"])
    return {
        DEFAULT_DATASET_KEY: {
            "label": str(cfg["data"]["dataset_label"]),
            "data": cfg["data"],
            "assets": cfg["assets"],
        }
    }


def default_dataset_key(cfg: Dict[str, object]) -> str:
    return str(cfg.get("default_dataset", DEFAULT_DATASET_KEY))


def normalize_dataset_key(cfg: Dict[str, object], dataset_key: Optional[str]) -> str:
    entries = dataset_entries(cfg)
    key = str(dataset_key or default_dataset_key(cfg)).strip() or default_dataset_key(cfg)
    if key not in entries:
        raise ValueError(f"Unknown dataset {key!r}; expected one of {sorted(entries)}")
    return key


class ViewerDataset:
    def __init__(self, dataset_key: str, cfg: Dict[str, object]):
        load_runtime_dependencies()

        self.dataset_key = dataset_key
        self.entry_cfg = cfg
        data_cfg = cfg["data"]
        assets_cfg = cfg["assets"]

        self.dataset_label = str(data_cfg.get("dataset_label") or cfg.get("label") or dataset_key)
        self.data_source = str(data_cfg.get("data_source", "local")).strip().lower()
        self.local_dataset_path = str(data_cfg.get("local_dataset_path", "")).strip()
        self.source_store = str(data_cfg.get("source_store", "")).strip()
        self.source_endpoint_url = str(data_cfg.get("source_endpoint_url", "")).strip()
        self.grid_crs = str(data_cfg["grid_crs"])
        self.display_variable = str(data_cfg["display_variable"])
        self.uncertainty_variable = str(data_cfg.get("uncertainty_variable", "")).strip()
        self.quality_variable = str(data_cfg.get("quality_variable", "")).strip()
        self.landcover_variable = str(data_cfg.get("landcover_variable", "")).strip()
        self.climatology_variable = str(data_cfg.get("climatology_variable", DEFAULT_POINT_CLIMATOLOGY_VARIABLE)).strip()
        self.landcover_source_dataset = str(data_cfg.get("landcover_source_dataset", "")).strip()
        self.initial_date = str(data_cfg["initial_date"])
        self.layer_keys = list(data_cfg.get("layer_keys", []))

        self.asset_mode = str(assets_cfg.get("asset_mode", "local")).strip().lower()
        self.asset_root = Path(str(assets_cfg["local_asset_root"])).resolve()
        self.source_asset_base_url = str(assets_cfg.get("source_asset_base_url", "")).strip()
        self.manifest_filename = str(assets_cfg["manifest_filename"])

        self.dataset_path = self._dataset_path_label()
        self.manifest = self._load_manifest()
        self.root = self._open_zarr_root()
        self.display_array = self.root[self.display_variable]
        self.uncertainty_array = self._optional_array(self.uncertainty_variable)
        self.quality_array = self._optional_array(self.quality_variable)
        self.landcover_array = self._optional_array(self.landcover_variable)
        self.landcover_year_array = self._optional_array("landcover_year") if self.landcover_array is not None else None
        self.climatology_array = self._optional_array(self.climatology_variable)
        self.lat_array = self._optional_array("lat")
        self.lon_array = self._optional_array("lon")

        self.dates = [str(value) for value in self.manifest.get("dates", [])]
        if not self.dates:
            self.dates = [datetime64_to_datestr(value) for value in self.root["time"][:]]
        self.date_values = [dt.date.fromisoformat(value) for value in self.dates]
        self.date_to_index = {date_str: idx for idx, date_str in enumerate(self.dates)}

        self.grid_extent = dict(self.manifest["grid_extent"])
        grid_resolution = self.manifest["grid_resolution"]
        self.pixel_width = abs(float(grid_resolution["dx"]))
        self.pixel_height = abs(float(grid_resolution["dy"]))
        self.x_size = int(self.display_array.shape[2])
        self.y_size = int(self.display_array.shape[1])
        self.grid_to_wgs84 = Transformer.from_crs(self.grid_crs, "EPSG:4326", always_xy=True)
        self.wgs84_to_grid = Transformer.from_crs("EPSG:4326", self.grid_crs, always_xy=True)

        if self.landcover_year_array is not None:
            self.landcover_years = np.asarray(self.landcover_year_array[:], dtype=np.int64)
        else:
            self.landcover_years = np.asarray([], dtype=np.int64)
        self.landcover_labels = self._landcover_mapping()
        self.quality_labels = self._quality_mapping()

    def _optional_array(self, variable_name: str):
        if not variable_name:
            return None
        try:
            return self.root[variable_name]
        except KeyError:
            return None

    def _dataset_path_label(self) -> str:
        if self.data_source == "source":
            if not self.source_store:
                raise ValueError("source_store is required when data_source=source")
            return self.source_store
        if self.data_source == "local":
            if not self.local_dataset_path:
                raise ValueError("local_dataset_path is required when data_source=local")
            return self.local_dataset_path
        raise ValueError(f"Unsupported data_source {self.data_source!r}; expected 'local' or 'source'")

    def _manifest_url(self) -> str:
        return join_url_parts(self.source_asset_base_url, self.manifest_filename)

    def _load_manifest(self) -> Dict[str, object]:
        if self.asset_mode == "source":
            request = urllib.request.Request(
                self._manifest_url(),
                headers={"User-Agent": "LongLFMCViewer/0.3"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        with self.manifest_path().open("r", encoding="utf-8") as f:
            return json.load(f)

    def _open_zarr_root(self):
        if self.data_source == "source":
            if not self.source_store.startswith("s3://"):
                raise ValueError(f"Expected s3:// Source store path, got {self.source_store!r}")
            storage_options = {"anon": True}
            if self.source_endpoint_url:
                storage_options["client_kwargs"] = {"endpoint_url": self.source_endpoint_url}
            fs = fsspec.filesystem("s3", **storage_options)
            return zarr.open_consolidated(fs.get_mapper(self.source_store.removeprefix("s3://")), mode="r")
        return zarr.open_consolidated(self.local_dataset_path, mode="r")

    def _landcover_mapping(self) -> Dict[int, str]:
        if self.landcover_array is None:
            return {}
        attrs = zarr_attrs(self.landcover_array)
        code_to_name = attrs.get("code_to_name", {})
        nodata_code = attrs.get("nodata_code")
        if isinstance(code_to_name, dict):
            mapping = {int(key): str(value) for key, value in code_to_name.items()}
            if nodata_code is not None:
                mapping[int(nodata_code)] = "nodata"
            return mapping
        dataset_key = zarr_attrs(self.root).get("dominant_landcover_code_key")
        if isinstance(dataset_key, str):
            parsed = json.loads(dataset_key)
            mapping = {int(key): str(value) for key, value in parsed.items()}
            if nodata_code is not None:
                mapping[int(nodata_code)] = "nodata"
            return mapping
        return {}

    def _quality_mapping(self) -> Dict[int, str]:
        values = zarr_attrs(self.root).get("quality_flag_values")
        if isinstance(values, dict):
            return {int(value): str(key) for key, value in values.items()}
        if self.quality_array is None:
            return {}
        attrs = zarr_attrs(self.quality_array)
        flag_values = attrs.get("flag_values", [])
        flag_meanings = str(attrs.get("flag_meanings", "")).split()
        return {int(value): meaning for value, meaning in zip(flag_values, flag_meanings)}

    def manifest_path(self) -> Path:
        return self.asset_root / self.manifest_filename

    def metadata(self) -> Dict[str, object]:
        if self.asset_mode == "source":
            asset_base_url = self.source_asset_base_url
            manifest_url = self._manifest_url()
        else:
            asset_base_url = f"/viewer-assets/{self.dataset_key}"
            manifest_url = f"{asset_base_url}/{self.manifest_filename}"
        payload = config_dataset_metadata(self.dataset_key, self.entry_cfg)
        payload.update(
            {
                "dataset_loaded": True,
                "dates": self.dates,
                "grid_extent": self.grid_extent,
                "grid_resolution": {
                    "dx": self.pixel_width,
                    "dy": self.pixel_height,
                },
                "asset_base_url": asset_base_url,
                "asset_manifest_url": manifest_url,
                "manifest_path": str(self.manifest_path()),
            }
        )
        return payload

    def _date_index(self, date_str: str) -> int:
        try:
            return self.date_to_index[date_str]
        except KeyError as exc:
            raise ValueError(f"Date not available: {date_str}") from exc

    def _date_range_indices(self, start_date: dt.date, end_date: dt.date) -> Tuple[int, int]:
        start_idx = 0
        while start_idx < len(self.date_values) and self.date_values[start_idx] < start_date:
            start_idx += 1
        end_idx = start_idx
        while end_idx < len(self.date_values) and self.date_values[end_idx] <= end_date:
            end_idx += 1
        return start_idx, end_idx

    def _cell_index_for_grid_xy(self, grid_x: float, grid_y: float) -> Tuple[int, int]:
        west = float(self.grid_extent["west"])
        east = float(self.grid_extent["east"])
        north = float(self.grid_extent["north"])
        south = float(self.grid_extent["south"])
        if grid_x < west or grid_x > east or grid_y < south or grid_y > north:
            raise ValueError("Requested point is outside the LFMC grid extent")

        x_idx = int(np.floor((grid_x - west) / self.pixel_width))
        y_idx = int(np.floor((north - grid_y) / self.pixel_height))
        x_idx = min(max(x_idx, 0), self.x_size - 1)
        y_idx = min(max(y_idx, 0), self.y_size - 1)
        return x_idx, y_idx

    def _cell_center(self, x_idx: int, y_idx: int) -> Tuple[float, float]:
        center_x = float(self.grid_extent["west"]) + self.pixel_width / 2.0 + x_idx * self.pixel_width
        center_y = float(self.grid_extent["north"]) - self.pixel_height / 2.0 - y_idx * self.pixel_height
        return center_x, center_y

    def _cell_bounds(self, x_idx: int, y_idx: int) -> Dict[str, float]:
        center_x, center_y = self._cell_center(x_idx=x_idx, y_idx=y_idx)
        return {
            "west": center_x - self.pixel_width / 2.0,
            "east": center_x + self.pixel_width / 2.0,
            "south": center_y - self.pixel_height / 2.0,
            "north": center_y + self.pixel_height / 2.0,
        }

    def _landcover_year_index(self, date_str: str) -> int:
        if self.landcover_years.size == 0:
            return 0
        year = int(date_str[:4])
        if self.landcover_years.size == 1:
            return 0
        diffs = np.abs(self.landcover_years - year)
        return int(np.argmin(diffs))

    def _array_value(self, array, time_idx: int, y_idx: int, x_idx: int):
        if array is None:
            return None
        return safe_float(np.asarray(array[time_idx, y_idx, x_idx]).item())

    def _series_for_cell(self, array, start_idx: int, end_idx: int, y_idx: int, x_idx: int) -> List[Optional[float]]:
        if array is None:
            return [None] * max(end_idx - start_idx, 0)
        values = np.asarray(array[start_idx:end_idx, y_idx, x_idx], dtype=np.float32)
        return [safe_float(value) for value in values]

    def _values_for_indices(self, array, indices: List[int], y_idx: int, x_idx: int) -> List[Optional[float]]:
        if array is None:
            return [None] * len(indices)
        if not indices:
            return []
        values = np.asarray(array.oindex[indices, y_idx, x_idx], dtype=np.float32)
        return [safe_float(value) for value in values]

    def _quality_value(self, time_idx: int):
        if self.quality_array is None:
            return None
        value = np.asarray(self.quality_array[time_idx]).item()
        return int(value) if np.isfinite(value) else None

    def _quality_series_for_window(self, start_idx: int, end_idx: int) -> List[Optional[int]]:
        if self.quality_array is None:
            return [None] * max(end_idx - start_idx, 0)
        values = np.asarray(self.quality_array[start_idx:end_idx])
        return [int(value) if np.isfinite(value) else None for value in values]

    def _quality_for_indices(self, indices: List[int]) -> List[Optional[int]]:
        if self.quality_array is None:
            return [None] * len(indices)
        if not indices:
            return []
        values = np.asarray(self.quality_array.oindex[indices])
        return [int(value) if np.isfinite(value) else None for value in values]

    def _climatology_value_for_cell(self, date_str: str, y_idx: int, x_idx: int):
        if self.climatology_array is None:
            return None
        day_idx = calendar_day_index_365(date_str)
        return safe_float(np.asarray(self.climatology_array[day_idx, y_idx, x_idx]).item())

    def _climatology_series_for_cell(self, dates: List[str], y_idx: int, x_idx: int) -> List[Optional[float]]:
        if self.climatology_array is None:
            return [None] * len(dates)
        day_indices = np.asarray([calendar_day_index_365(date_str) for date_str in dates], dtype=np.int64)
        point_climatology = np.asarray(self.climatology_array[:, y_idx, x_idx], dtype=np.float32)
        return [safe_float(value) for value in point_climatology[day_indices]]

    def _day_offsets(self, dates: List[str], window_start: dt.date) -> List[int]:
        return [(dt.date.fromisoformat(date_str) - window_start).days for date_str in dates]

    def _window_series_from_lookups(
        self,
        indices: List[int],
        start_date: dt.date,
        mean_lookup: Dict[int, Optional[float]],
        uncertainty_lookup: Dict[int, Optional[float]],
        quality_lookup: Dict[int, Optional[int]],
        y_idx: int,
        x_idx: int,
    ) -> Dict[str, object]:
        dates = [self.dates[idx] for idx in indices]
        mean_series = [mean_lookup.get(idx) for idx in indices]
        uncertainty_series = [uncertainty_lookup.get(idx) for idx in indices]
        return {
            "dates": dates,
            "day_offsets": self._day_offsets(dates, start_date),
            "lfmc_ens_mean": mean_series,
            "lfmc_ens_std": uncertainty_series,
            "lfmc_climatology_mean": self._climatology_series_for_cell(dates, y_idx, x_idx),
            "lfmc_anomaly": [
                safe_difference(mean, climatology)
                for mean, climatology in zip(mean_series, self._climatology_series_for_cell(dates, y_idx, x_idx))
            ],
            "quality_flag": [quality_lookup.get(idx) for idx in indices],
            "window_days": DEFAULT_POINT_TIMESERIES_DAYS,
        }

    def _series_windows_for_cell(self, selected_date: dt.date, y_idx: int, x_idx: int) -> Dict[str, object]:
        current_start = selected_date - dt.timedelta(days=DEFAULT_POINT_TIMESERIES_DAYS - 1)
        current_start_idx, current_end_idx = self._date_range_indices(current_start, selected_date)
        current_indices = list(range(current_start_idx, current_end_idx))

        selected_year = selected_date.year
        first_year = self.date_values[0].year
        historical_specs = []
        all_indices = set(current_indices)
        for year in range(first_year, selected_year):
            hist_end = shift_year(selected_date, year)
            hist_start = hist_end - dt.timedelta(days=DEFAULT_POINT_TIMESERIES_DAYS - 1)
            start_idx, end_idx = self._date_range_indices(hist_start, hist_end)
            indices = list(range(start_idx, end_idx))
            if len(indices) < 2:
                continue
            historical_specs.append((year, hist_start, indices))
            all_indices.update(indices)

        sorted_indices = sorted(all_indices)
        mean_lookup = dict(zip(sorted_indices, self._values_for_indices(self.display_array, sorted_indices, y_idx, x_idx)))
        uncertainty_lookup = dict(zip(sorted_indices, self._values_for_indices(self.uncertainty_array, sorted_indices, y_idx, x_idx)))
        quality_lookup = dict(zip(sorted_indices, self._quality_for_indices(sorted_indices)))

        current = self._window_series_from_lookups(
            current_indices,
            current_start,
            mean_lookup,
            uncertainty_lookup,
            quality_lookup,
            y_idx,
            x_idx,
        )
        windows = []
        for year, hist_start, indices in historical_specs:
            series = self._window_series_from_lookups(
                indices,
                hist_start,
                mean_lookup,
                uncertainty_lookup,
                quality_lookup,
                y_idx,
                x_idx,
            )
            series["year"] = year
            windows.append(series)
        current["historical_windows"] = windows
        return current

    def landcover_payload_for_latlon(self, date_str: str, lat: float, lon: float) -> Dict[str, object]:
        if self.landcover_array is None:
            return {"landcover_code": None, "landcover_name": "unavailable"}
        grid_x, grid_y = self.wgs84_to_grid.transform(lon, lat)
        x_idx, y_idx = self._cell_index_for_grid_xy(grid_x=float(grid_x), grid_y=float(grid_y))
        landcover_year_idx = self._landcover_year_index(date_str)
        raw_landcover_value = np.asarray(self.landcover_array[landcover_year_idx, y_idx, x_idx]).item()
        landcover_code = safe_float(raw_landcover_value)
        if landcover_code is None:
            return {"landcover_code": None, "landcover_name": "unknown"}
        landcover_code = int(landcover_code)
        return {
            "landcover_code": landcover_code,
            "landcover_name": self.landcover_labels.get(landcover_code, "unknown"),
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
        _ = timeseries_days
        time_idx = self._date_index(date_str)
        if grid_x is None or grid_y is None:
            if lat is None or lon is None:
                raise ValueError("Provide either grid x/y or lat/lon")
            grid_x, grid_y = self.wgs84_to_grid.transform(lon, lat)

        requested_lon, requested_lat = self.grid_to_wgs84.transform(grid_x, grid_y)
        x_idx, y_idx = self._cell_index_for_grid_xy(grid_x=float(grid_x), grid_y=float(grid_y))
        cell_bounds = self._cell_bounds(x_idx=x_idx, y_idx=y_idx)
        mean_value = self._array_value(self.display_array, time_idx, y_idx, x_idx)
        uncertainty_value = self._array_value(self.uncertainty_array, time_idx, y_idx, x_idx)
        climatology_value = self._climatology_value_for_cell(date_str, y_idx, x_idx)
        anomaly_value = safe_difference(mean_value, climatology_value)
        center_x, center_y = self._cell_center(x_idx=x_idx, y_idx=y_idx)
        if self.lon_array is not None and self.lat_array is not None:
            center_lon = safe_float(np.asarray(self.lon_array[y_idx, x_idx]).item())
            center_lat = safe_float(np.asarray(self.lat_array[y_idx, x_idx]).item())
        else:
            center_lon, center_lat = self.grid_to_wgs84.transform(center_x, center_y)
            center_lon = safe_float(center_lon)
            center_lat = safe_float(center_lat)

        landcover = self.landcover_payload_for_latlon(date_str, center_lat, center_lon)
        quality_value = self._quality_value(time_idx)
        payload = {
            "dataset_key": self.dataset_key,
            "date": date_str,
            "requested_grid_x": float(grid_x),
            "requested_grid_y": float(grid_y),
            "requested_lat": safe_float(requested_lat),
            "requested_lon": safe_float(requested_lon),
            "cell_center_x": center_x,
            "cell_center_y": center_y,
            "nearest_lat": center_lat,
            "nearest_lon": center_lon,
            "cell_center_lat": center_lat,
            "cell_center_lon": center_lon,
            "cell_bounds": cell_bounds,
            "cell_index": {
                "x": int(x_idx),
                "y": int(y_idx),
            },
            "lfmc_ens_mean": mean_value,
            "lfmc_ens_std": uncertainty_value,
            "lfmc_climatology_mean": climatology_value,
            "lfmc_anomaly": anomaly_value,
            "quality_flag": quality_value,
            "data_product_level": self.quality_labels.get(quality_value, "unknown") if quality_value is not None else "unknown",
            "landcover_code": landcover["landcover_code"],
            "landcover_name": landcover["landcover_name"],
            "timeseries": None,
        }

        if include_timeseries:
            selected_date = dt.date.fromisoformat(date_str)
            payload["timeseries"] = self._series_windows_for_cell(selected_date, y_idx, x_idx)

        return payload

    def download_csv_bytes_for_sites(
        self,
        sites: List[Tuple[float, float, str, str]],
        start_date: str,
        end_date: str,
        landcover_dataset=None,
    ) -> Tuple[bytes, str]:
        if not sites:
            raise ValueError("At least one site is required for CSV download")
        if len(sites) > 10:
            raise ValueError("CSV download supports at most 10 sites")

        normalized_sites = []
        for site_idx, site in enumerate(sites, start=1):
            lat, lon, site_start_date, site_end_date = site
            if site_end_date < site_start_date:
                raise ValueError(f"Site {site_idx} end_date must be on or after start_date")
            if dt.date.fromisoformat(site_end_date) > max_csv_end_date(site_start_date):
                raise ValueError(
                    f"Site {site_idx} CSV download range must be {MAX_CSV_DOWNLOAD_YEARS} years or less"
                )
            range_start = dt.date.fromisoformat(site_start_date)
            range_end = dt.date.fromisoformat(site_end_date)
            start_idx, end_idx = self._date_range_indices(range_start, range_end)
            if end_idx <= start_idx:
                raise ValueError(f"Site {site_idx} has no {self.dataset_label} dates in requested range")
            normalized_sites.append((lat, lon, site_start_date, site_end_date, start_idx, end_idx))

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "dataset",
                "site_index",
                "site_lat",
                "site_lon",
                "site_start_date",
                "site_end_date",
                "date",
                "lfmc",
                "lfmc_uncertainty",
                "lfmc_anomaly",
                "lfmc_climatology",
                "quality_flag",
                "landcover_code",
                "landcover_name",
            ]
        )

        for site_idx, site in enumerate(normalized_sites, start=1):
            lat, lon, site_start_date, site_end_date, start_idx, end_idx = site
            grid_x, grid_y = self.wgs84_to_grid.transform(lon, lat)
            x_idx, y_idx = self._cell_index_for_grid_xy(grid_x=float(grid_x), grid_y=float(grid_y))
            center_x, center_y = self._cell_center(x_idx, y_idx)
            center_lon, center_lat = self.grid_to_wgs84.transform(center_x, center_y)
            landcover_source = landcover_dataset or self
            landcover = landcover_source.landcover_payload_for_latlon(site_end_date, float(center_lat), float(center_lon))
            dates = self.dates[start_idx:end_idx]
            mean_values = self._series_for_cell(self.display_array, start_idx, end_idx, y_idx, x_idx)
            uncertainty_values = self._series_for_cell(self.uncertainty_array, start_idx, end_idx, y_idx, x_idx)
            climatology_values = self._climatology_series_for_cell(dates, y_idx, x_idx)
            anomaly_values = [
                safe_difference(mean, climatology)
                for mean, climatology in zip(mean_values, climatology_values)
            ]
            quality_values = self._quality_series_for_window(start_idx, end_idx)

            for row_idx, date_str in enumerate(dates):
                writer.writerow(
                    [
                        self.dataset_key,
                        site_idx,
                        f"{float(center_lat):.6f}",
                        f"{float(center_lon):.6f}",
                        site_start_date,
                        site_end_date,
                        date_str,
                        "" if mean_values[row_idx] is None else f"{mean_values[row_idx]:.6f}",
                        "" if uncertainty_values[row_idx] is None else f"{uncertainty_values[row_idx]:.6f}",
                        "" if anomaly_values[row_idx] is None else f"{anomaly_values[row_idx]:.6f}",
                        "" if climatology_values[row_idx] is None else f"{climatology_values[row_idx]:.6f}",
                        "" if quality_values[row_idx] is None else quality_values[row_idx],
                        "" if landcover["landcover_code"] is None else landcover["landcover_code"],
                        landcover["landcover_name"],
                    ]
                )

        ranges = {(site[2], site[3]) for site in normalized_sites}
        if len(ranges) == 1:
            filename = f"{self.dataset_key}_lfmc_sites_{start_date}_to_{end_date}.csv"
        else:
            filename = f"{self.dataset_key}_lfmc_sites_{start_date}_to_{end_date}_site_ranges.csv"
        return buffer.getvalue().encode("utf-8"), filename

    def resolve_asset_path(self, request_path: str) -> Path:
        relpath = request_path.removeprefix(f"/viewer-assets/{self.dataset_key}/").strip("/")
        candidate = (self.asset_root / relpath).resolve()
        if not str(candidate).startswith(str(self.asset_root)):
            raise ValueError("Attempted path traversal outside asset root")
        return candidate

    def close(self) -> None:
        self.root = None


def config_dataset_metadata(dataset_key: str, entry_cfg: Dict[str, object]) -> Dict[str, object]:
    data_cfg = entry_cfg["data"]
    assets_cfg = entry_cfg["assets"]
    asset_mode = str(assets_cfg.get("asset_mode", "local")).strip().lower()
    if asset_mode == "source":
        asset_base_url = str(assets_cfg.get("source_asset_base_url", "")).strip()
        manifest_url = join_url_parts(asset_base_url, str(assets_cfg["manifest_filename"]))
    else:
        asset_base_url = f"/viewer-assets/{dataset_key}"
        manifest_url = f"{asset_base_url}/{assets_cfg['manifest_filename']}"

    layer_keys = data_cfg.get("layer_keys", [])
    return {
        "dataset_key": dataset_key,
        "dataset_label": str(data_cfg.get("dataset_label") or entry_cfg.get("label") or dataset_key),
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
        "supports_anomaly": "anomaly" in layer_keys,
        "supports_uncertainty": bool(str(data_cfg.get("uncertainty_variable", "")).strip()),
        "supports_climatology": bool(str(data_cfg.get("climatology_variable", "")).strip()),
        "layer_keys": layer_keys,
    }


cfg: Dict[str, object] = {}
viewer_datasets: Dict[str, ViewerDataset] = {}
dataset_loading: Dict[str, bool] = {}
dataset_lock = threading.RLock()
dataset_condition = threading.Condition(dataset_lock)
last_refresh_time = 0.0
last_refresh_error: Optional[str] = None


def load_dataset(dataset_key: str, reason: str) -> None:
    global cfg, last_refresh_error

    try:
        log(f"Opening viewer dataset={dataset_key} reason={reason}")
        fresh_cfg = load_config()
        entry = dataset_entries(fresh_cfg)[dataset_key]
        fresh_dataset = ViewerDataset(dataset_key, entry)
        with dataset_condition:
            old_dataset = viewer_datasets.get(dataset_key)
            cfg = fresh_cfg
            viewer_datasets[dataset_key] = fresh_dataset
            dataset_loading[dataset_key] = False
            last_refresh_error = None
            dataset_condition.notify_all()
        if old_dataset is not None:
            old_dataset.close()
        log(f"Viewer dataset open complete dataset={dataset_key} dates={len(fresh_dataset.dates)}")
    except Exception as exc:
        with dataset_condition:
            dataset_loading[dataset_key] = False
            last_refresh_error = str(exc)
            dataset_condition.notify_all()
        log(f"Viewer dataset open failed dataset={dataset_key} reason={reason}: {exc}")


def start_dataset_load(dataset_key: str, reason: str) -> None:
    with dataset_condition:
        if dataset_key in viewer_datasets or dataset_loading.get(dataset_key, False):
            return
        dataset_loading[dataset_key] = True
        dataset_condition.notify_all()

    thread = threading.Thread(target=load_dataset, args=(dataset_key, reason), daemon=True)
    thread.start()


def wait_for_dataset(dataset_key: str, timeout_seconds: float = DATASET_LOAD_WAIT_SECONDS) -> ViewerDataset:
    if dataset_key not in viewer_datasets and not dataset_loading.get(dataset_key, False):
        start_dataset_load(dataset_key, reason="on_demand")

    deadline = time.time() + timeout_seconds
    with dataset_condition:
        while dataset_key not in viewer_datasets:
            if last_refresh_error and not dataset_loading.get(dataset_key, False):
                raise RuntimeError(f"Dataset {dataset_key} is not loaded; last refresh error: {last_refresh_error}")
            remaining = deadline - time.time()
            if remaining <= 0:
                raise DatasetLoadingError(f"Dataset {dataset_key} is still loading")
            dataset_condition.wait(timeout=remaining)
        return viewer_datasets[dataset_key]


def require_loaded_dataset(dataset_key: str) -> ViewerDataset:
    with dataset_lock:
        if dataset_key not in viewer_datasets:
            if last_refresh_error:
                raise RuntimeError(f"Dataset {dataset_key} is not loaded; last refresh error: {last_refresh_error}")
            raise RuntimeError(f"Dataset {dataset_key} is still loading")
        return viewer_datasets[dataset_key]


def refresh_datasets(reason: str = "manual") -> Dict[str, object]:
    global cfg, viewer_datasets, dataset_loading, last_refresh_time, last_refresh_error

    log(f"Unloading datasets reason={reason}")
    with dataset_condition:
        old_datasets = viewer_datasets
        cfg = load_config()
        viewer_datasets = {}
        dataset_loading = {}
        last_refresh_time = time.time()
        last_refresh_error = None
        dataset_condition.notify_all()

    for dataset in old_datasets.values():
        dataset.close()

    payload = {
        "status": "refreshed",
        "reason": reason,
        "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_refresh_time)),
        "datasets_loaded": [],
    }
    log("Dataset handles unloaded; zarrs will reopen on demand")
    return payload


def should_refresh_for_date_error(exc: Exception) -> bool:
    return "Date not available" in str(exc)


def metadata_payload(dataset_key: Optional[str] = None) -> Dict[str, object]:
    fresh_cfg = load_config()
    entries = dataset_entries(fresh_cfg)
    requested_key = normalize_dataset_key(fresh_cfg, dataset_key)
    with dataset_lock:
        loaded_dataset = viewer_datasets.get(requested_key)
        loaded_keys = sorted(viewer_datasets)
        loading_keys = sorted(key for key, value in dataset_loading.items() if value)

    datasets = {}
    for key, entry in entries.items():
        with dataset_lock:
            loaded = viewer_datasets.get(key)
        datasets[key] = loaded.metadata() if loaded is not None else config_dataset_metadata(key, entry)

    payload = datasets[requested_key].copy()
    payload.update(
        {
            "default_dataset": default_dataset_key(fresh_cfg),
            "active_dataset": requested_key,
            "datasets": datasets,
            "loaded_datasets": loaded_keys,
            "loading_datasets": loading_keys,
            "last_refresh_epoch": last_refresh_time,
            "last_refresh_error": last_refresh_error,
        }
    )
    if loaded_dataset is not None:
        payload["dataset_loaded"] = True
    return payload


def point_payload_with_refresh(
    dataset_key: str,
    date_str: str,
    grid_x: float = None,
    grid_y: float = None,
    lat: float = None,
    lon: float = None,
    include_timeseries: bool = False,
    timeseries_days: int = DEFAULT_POINT_TIMESERIES_DAYS,
) -> Dict[str, object]:
    try:
        loaded_dataset = wait_for_dataset(dataset_key)
        with dataset_lock:
            payload = loaded_dataset.point_payload(
                date_str=date_str,
                grid_x=grid_x,
                grid_y=grid_y,
                lat=lat,
                lon=lon,
                include_timeseries=include_timeseries,
                timeseries_days=timeseries_days,
            )
            if loaded_dataset.landcover_source_dataset and payload.get("landcover_code") is None:
                source_dataset = wait_for_dataset(loaded_dataset.landcover_source_dataset)
                landcover = source_dataset.landcover_payload_for_latlon(
                    date_str=date_str,
                    lat=float(payload["cell_center_lat"]),
                    lon=float(payload["cell_center_lon"]),
                )
                payload.update(landcover)
            return payload
    except ValueError as exc:
        if not should_refresh_for_date_error(exc):
            raise

    refresh_datasets(reason=f"point_date_miss:{dataset_key}:{date_str}")
    loaded_dataset = wait_for_dataset(dataset_key)
    with dataset_lock:
        payload = loaded_dataset.point_payload(
            date_str=date_str,
            grid_x=grid_x,
            grid_y=grid_y,
            lat=lat,
            lon=lon,
            include_timeseries=include_timeseries,
            timeseries_days=timeseries_days,
        )
        if loaded_dataset.landcover_source_dataset and payload.get("landcover_code") is None:
            source_dataset = wait_for_dataset(loaded_dataset.landcover_source_dataset)
            payload.update(
                source_dataset.landcover_payload_for_latlon(
                    date_str=date_str,
                    lat=float(payload["cell_center_lat"]),
                    lon=float(payload["cell_center_lon"]),
                )
            )
        return payload


def csv_with_refresh(
    dataset_key: str,
    sites: List[Tuple[float, float, str, str]],
    start_date: str,
    end_date: str,
) -> Tuple[bytes, str]:
    try:
        loaded_dataset = wait_for_dataset(dataset_key)
        landcover_dataset = None
        if loaded_dataset.landcover_source_dataset:
            landcover_dataset = wait_for_dataset(loaded_dataset.landcover_source_dataset)
        with dataset_lock:
            return loaded_dataset.download_csv_bytes_for_sites(
                sites=sites,
                start_date=start_date,
                end_date=end_date,
                landcover_dataset=landcover_dataset,
            )
    except ValueError as exc:
        if not should_refresh_for_date_error(exc):
            raise

    refresh_datasets(reason=f"csv_date_miss:{dataset_key}:{start_date}:{end_date}")
    loaded_dataset = wait_for_dataset(dataset_key)
    landcover_dataset = None
    if loaded_dataset.landcover_source_dataset:
        landcover_dataset = wait_for_dataset(loaded_dataset.landcover_source_dataset)
    with dataset_lock:
        return loaded_dataset.download_csv_bytes_for_sites(
            sites=sites,
            start_date=start_date,
            end_date=end_date,
            landcover_dataset=landcover_dataset,
        )


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server_version = "LongLFMCViewer/0.4"

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
                dataset_key = parsed.path.strip("/").split("/")[1]
                loaded_dataset = require_loaded_dataset(dataset_key)
                self._static_response(loaded_dataset.resolve_asset_path(parsed.path), send_body=False)
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
            fresh_cfg = load_config()
            dataset_key = normalize_dataset_key(fresh_cfg, self._optional_param(query, "dataset"))
            if parsed.path == "/api/health":
                entries = dataset_entries(fresh_cfg)
                with dataset_condition:
                    payload = {
                        "status": "ok",
                        "datasets": sorted(entries),
                        "loaded_datasets": sorted(viewer_datasets),
                        "loading_datasets": sorted(key for key, value in dataset_loading.items() if value),
                        "last_refresh_epoch": last_refresh_time,
                        "last_refresh_error": last_refresh_error,
                    }
                self._json_response(payload)
                return
            if parsed.path == "/api/metadata":
                self._json_response(metadata_payload(dataset_key))
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
                payload = point_payload_with_refresh(
                    dataset_key=dataset_key,
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
                    parts = [value.strip() for value in raw_site.split(",")]
                    if len(parts) == 2:
                        sites.append((float(parts[0]), float(parts[1]), start_date, end_date))
                    elif len(parts) == 4:
                        sites.append((float(parts[0]), float(parts[1]), parts[2], parts[3]))
                    else:
                        raise ValueError(
                            f"Invalid site parameter {raw_site!r}; expected 'lat,lon' or 'lat,lon,start_date,end_date'"
                        )
                if not sites:
                    lat = float(self._require_param(query, "lat"))
                    lon = float(self._require_param(query, "lon"))
                    sites = [(lat, lon, start_date, end_date)]
                csv_bytes, filename = csv_with_refresh(
                    dataset_key=dataset_key,
                    sites=sites,
                    start_date=start_date,
                    end_date=end_date,
                )
                self._csv_response(csv_bytes, filename=filename)
                return
            if parsed.path.startswith("/viewer-assets/"):
                dataset_key = parsed.path.strip("/").split("/")[1]
                loaded_dataset = require_loaded_dataset(dataset_key)
                self._static_response(loaded_dataset.resolve_asset_path(parsed.path))
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

    def _optional_param(self, query: Dict[str, List[str]], name: str):
        values = query.get(name)
        if not values or values[0] == "":
            return None
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
