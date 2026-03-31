#!/usr/bin/env python3

import json
import shutil
import time
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr
import yaml
import zarr
from pyproj import Transformer
from rasterio.transform import Affine, from_origin
from rasterio.warp import transform_bounds


here = Path(__file__).resolve().parent
config_path = here / "viewer_dataset_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_config() -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def center_coordinates_from_transform(transform: Affine, width: int, height: int):
    x_values = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y_values = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x_values, y_values


class ViewerDataset3857Builder:
    def __init__(self, cfg: Dict[str, object]):
        dataset_cfg = cfg["dataset"]
        output_cfg = cfg["output"]

        self.dataset_label = str(dataset_cfg["dataset_label"])
        self.source_dataset_path = Path(str(dataset_cfg["scientific_dataset_path"]))
        self.source_crs = str(dataset_cfg["scientific_grid_crs"])
        self.target_crs = str(dataset_cfg["viewer_grid_crs"])
        self.target_grid_anchor_mode = str(dataset_cfg.get("viewer_grid_anchor_mode", "source_extent_center"))
        self.viewer_sampling_method = str(dataset_cfg.get("viewer_sampling_method", "nearest_source_cell"))
        self.display_variable = str(dataset_cfg["display_variable"])
        self.quality_variable = str(dataset_cfg["quality_variable"])
        self.landcover_variable = str(dataset_cfg["landcover_variable"])

        self.viewer_dataset_path = Path(str(output_cfg["viewer_dataset_path"]))
        self.clear_existing = bool(output_cfg.get("clear_existing", True))
        self.viewer_resolution_m = float(output_cfg["viewer_resolution_m"])
        self.time_chunk_size = int(output_cfg.get("time_chunk_size", 128))
        self.spatial_chunk_size = int(output_cfg.get("spatial_chunk_size", 256))
        self.latlon_row_chunk_size = int(output_cfg.get("latlon_row_chunk_size", 256))

        log(f"Opening scientific dataset {self.source_dataset_path}")
        self.ds = xr.open_zarr(self.source_dataset_path, consolidated=False)

        self.x_values = np.asarray(self.ds["x"].values, dtype=np.float64)
        self.y_values = np.asarray(self.ds["y"].values, dtype=np.float64)
        self.time_values = np.asarray(self.ds["time"].values)
        self.landcover_year_values = np.asarray(self.ds["landcover_year"].values)
        self.dx = abs(float(np.median(np.diff(self.x_values))))
        self.dy = abs(float(np.median(np.diff(self.y_values))))

        self.source_extent = {
            "west": float(self.x_values[0] - self.dx / 2.0),
            "east": float(self.x_values[-1] + self.dx / 2.0),
            "north": float(self.y_values[0] + self.dy / 2.0),
            "south": float(self.y_values[-1] - self.dy / 2.0),
        }
        self.source_transform = from_origin(
            self.source_extent["west"],
            self.source_extent["north"],
            self.dx,
            self.dy,
        )

        self.source_to_target = Transformer.from_crs(self.source_crs, self.target_crs, always_xy=True)
        self.target_bounds = transform_bounds(
            self.source_crs,
            self.target_crs,
            self.source_extent["west"],
            self.source_extent["south"],
            self.source_extent["east"],
            self.source_extent["north"],
            densify_pts=21,
        )
        self.target_transform, self.target_width, self.target_height = self._build_explicit_target_grid()
        self.target_x_values, self.target_y_values = center_coordinates_from_transform(
            self.target_transform,
            self.target_width,
            self.target_height,
        )
        self.grid_to_wgs84 = Transformer.from_crs(self.target_crs, "EPSG:4326", always_xy=True)
        self.target_to_source = Transformer.from_crs(self.target_crs, self.source_crs, always_xy=True)
        self.source_row_lookup = None
        self.source_col_lookup = None

    def _anchor_target_center(self):
        if self.target_grid_anchor_mode != "source_extent_center":
            raise ValueError(f"Unsupported viewer_grid_anchor_mode {self.target_grid_anchor_mode!r}")
        source_center_x = 0.5 * (self.source_extent["west"] + self.source_extent["east"])
        source_center_y = 0.5 * (self.source_extent["south"] + self.source_extent["north"])
        return self.source_to_target.transform(source_center_x, source_center_y)

    def _build_explicit_target_grid(self):
        anchor_x, anchor_y = self._anchor_target_center()
        target_west, target_south, target_east, target_north = self.target_bounds
        resolution = self.viewer_resolution_m

        col_min = int(np.floor(((target_west - anchor_x) / resolution) + 0.5))
        col_max = int(np.ceil(((target_east - anchor_x) / resolution) - 0.5))
        row_min = int(np.floor(((anchor_y - target_north) / resolution) + 0.5))
        row_max = int(np.ceil(((anchor_y - target_south) / resolution) - 0.5))

        target_width = col_max - col_min + 1
        target_height = row_max - row_min + 1
        target_west_edge = anchor_x + col_min * resolution - (0.5 * resolution)
        target_north_edge = anchor_y - row_min * resolution + (0.5 * resolution)
        target_transform = from_origin(
            target_west_edge,
            target_north_edge,
            resolution,
            resolution,
        )
        return target_transform, target_width, target_height

    def _copy_dataset_attrs(self) -> Dict[str, object]:
        attrs = dict(self.ds.attrs)
        attrs["viewer_dataset_label"] = self.dataset_label
        attrs["viewer_dataset_source_path"] = str(self.source_dataset_path)
        attrs["viewer_dataset_source_crs"] = self.source_crs
        attrs["viewer_dataset_target_crs"] = self.target_crs
        attrs["viewer_dataset_target_grid_anchor_mode"] = self.target_grid_anchor_mode
        attrs["viewer_dataset_sampling_method"] = self.viewer_sampling_method
        attrs["viewer_dataset_target_bounds"] = [float(value) for value in self.target_bounds]
        attrs["viewer_dataset_nominal_resolution_m"] = self.viewer_resolution_m
        attrs["viewer_dataset_created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return attrs

    def _open_output_root(self):
        if self.clear_existing and self.viewer_dataset_path.exists():
            log(f"Clearing existing viewer dataset {self.viewer_dataset_path}")
            shutil.rmtree(self.viewer_dataset_path)
        self.viewer_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        return zarr.open_group(
            str(self.viewer_dataset_path),
            mode="w",
            zarr_format=2,
            attributes=self._copy_dataset_attrs(),
        )

    def _create_array(self, root, name: str, *, shape=None, dtype=None, chunks=None, data=None, dims=None, attrs=None, fill_value=None):
        array_attrs = dict(attrs or {})
        array_attrs["_ARRAY_DIMENSIONS"] = list(dims or [])
        create_kwargs = {
            "name": name,
            "chunks": chunks,
            "attributes": array_attrs,
            "overwrite": True,
        }
        if data is not None:
            create_kwargs["data"] = data
        else:
            create_kwargs["shape"] = shape
            create_kwargs["dtype"] = dtype
            create_kwargs["fill_value"] = fill_value
        return root.create_array(**create_kwargs)

    def _create_output_arrays(self, root):
        coord_chunk = min(self.spatial_chunk_size * 4, max(self.target_width, self.target_height))
        self._create_array(root, "x", data=self.target_x_values, chunks=(min(coord_chunk, self.target_width),), dims=["x"])
        self._create_array(root, "y", data=self.target_y_values, chunks=(min(coord_chunk, self.target_height),), dims=["y"])
        self._create_array(root, "time", data=self.time_values, chunks=(min(len(self.time_values), self.time_chunk_size),), dims=["time"])
        self._create_array(
            root,
            "landcover_year",
            data=self.landcover_year_values,
            chunks=(max(1, len(self.landcover_year_values)),),
            dims=["landcover_year"],
        )

        lat_attrs = dict(self.ds["lat"].attrs)
        lon_attrs = dict(self.ds["lon"].attrs)
        landcover_attrs = dict(self.ds[self.landcover_variable].attrs)
        mean_attrs = dict(self.ds[self.display_variable].attrs)
        quality_attrs = dict(self.ds[self.quality_variable].attrs)

        self._create_array(
            root,
            "lat",
            shape=(self.target_height, self.target_width),
            dtype=np.float64,
            chunks=(self.latlon_row_chunk_size, self.spatial_chunk_size),
            dims=["y", "x"],
            attrs=lat_attrs,
            fill_value=np.nan,
        )
        self._create_array(
            root,
            "lon",
            shape=(self.target_height, self.target_width),
            dtype=np.float64,
            chunks=(self.latlon_row_chunk_size, self.spatial_chunk_size),
            dims=["y", "x"],
            attrs=lon_attrs,
            fill_value=np.nan,
        )
        self._create_array(
            root,
            self.display_variable,
            shape=(len(self.time_values), self.target_height, self.target_width),
            dtype=np.float32,
            chunks=(min(len(self.time_values), self.time_chunk_size), self.spatial_chunk_size, self.spatial_chunk_size),
            dims=["time", "y", "x"],
            attrs=mean_attrs,
            fill_value=np.nan,
        )
        self._create_array(
            root,
            self.landcover_variable,
            shape=(len(self.landcover_year_values), self.target_height, self.target_width),
            dtype=np.uint8,
            chunks=(1, self.spatial_chunk_size, self.spatial_chunk_size),
            dims=["landcover_year", "y", "x"],
            attrs=landcover_attrs,
            fill_value=int(landcover_attrs.get("nodata_code", 255)),
        )
        self._create_array(
            root,
            self.quality_variable,
            shape=(len(self.time_values),),
            dtype=np.uint8,
            chunks=(min(len(self.time_values), self.time_chunk_size),),
            dims=["time"],
            attrs=quality_attrs,
            fill_value=255,
        )

    def _write_lat_lon(self, root) -> None:
        lat_arr = root["lat"]
        lon_arr = root["lon"]
        for row_start in range(0, self.target_height, self.latlon_row_chunk_size):
            row_end = min(row_start + self.latlon_row_chunk_size, self.target_height)
            y_block = self.target_y_values[row_start:row_end]
            x_grid, y_grid = np.meshgrid(self.target_x_values, y_block)
            lon_block, lat_block = self.grid_to_wgs84.transform(x_grid, y_grid)
            lat_arr[row_start:row_end, :] = lat_block.astype(np.float64)
            lon_arr[row_start:row_end, :] = lon_block.astype(np.float64)
            log(
                "Computed lat/lon rows "
                f"{row_start}-{row_end - 1} of {self.target_height - 1}"
            )

    @staticmethod
    def _nearest_index_for_sorted_axis(axis_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
        insert_idx = np.searchsorted(axis_values, query_values, side="left")
        insert_idx = np.clip(insert_idx, 0, axis_values.size - 1)
        left_idx = np.clip(insert_idx - 1, 0, axis_values.size - 1)
        right_idx = insert_idx

        left_dist = np.abs(query_values - axis_values[left_idx])
        right_dist = np.abs(axis_values[right_idx] - query_values)
        use_left = left_dist <= right_dist
        return np.where(use_left, left_idx, right_idx).astype(np.int32)

    def _build_source_lookup(self) -> None:
        if self.viewer_sampling_method != "nearest_source_cell":
            raise ValueError(f"Unsupported viewer_sampling_method {self.viewer_sampling_method!r}")

        row_lookup = np.empty((self.target_height, self.target_width), dtype=np.int32)
        col_lookup = np.empty((self.target_height, self.target_width), dtype=np.int32)
        source_x_sorted = self.x_values
        source_y_sorted_asc = self.y_values[::-1].copy()

        for row_start in range(0, self.target_height, self.latlon_row_chunk_size):
            row_end = min(row_start + self.latlon_row_chunk_size, self.target_height)
            y_block = self.target_y_values[row_start:row_end]
            x_grid, y_grid = np.meshgrid(self.target_x_values, y_block)
            source_x_grid, source_y_grid = self.target_to_source.transform(x_grid, y_grid)
            col_block = self._nearest_index_for_sorted_axis(source_x_sorted, source_x_grid)
            row_block_asc = self._nearest_index_for_sorted_axis(source_y_sorted_asc, source_y_grid)
            row_block = (self.y_values.size - 1 - row_block_asc).astype(np.int32)
            row_lookup[row_start:row_end, :] = row_block
            col_lookup[row_start:row_end, :] = col_block
            log(
                "Built source lookup rows "
                f"{row_start}-{row_end - 1} of {self.target_height - 1}"
            )

        self.source_row_lookup = row_lookup
        self.source_col_lookup = col_lookup

    def _sample_nearest_source_2d(self, source_2d: np.ndarray):
        if self.source_row_lookup is None or self.source_col_lookup is None:
            raise RuntimeError("Source lookup arrays have not been built")
        return source_2d[self.source_row_lookup, self.source_col_lookup]

    def _write_time_stack(self, root, variable_name: str) -> None:
        output_arr = root[variable_name]
        total_steps = len(self.time_values)
        for time_idx, time_value in enumerate(self.time_values, start=1):
            source_2d = np.asarray(self.ds[variable_name].isel(time=time_idx - 1).values, dtype=np.float32)
            destination = self._sample_nearest_source_2d(source_2d)
            output_arr[time_idx - 1, :, :] = destination.astype(np.float32)
            log(
                f"Sampled {variable_name} time step {time_idx}/{total_steps} "
                f"({np.datetime_as_string(np.datetime64(time_value), unit='D')})"
            )

    def _write_landcover(self, root) -> None:
        output_arr = root[self.landcover_variable]
        total_years = len(self.landcover_year_values)
        for year_idx, year_value in enumerate(self.landcover_year_values, start=1):
            source_2d = np.asarray(
                self.ds[self.landcover_variable].isel(landcover_year=year_idx - 1).values,
                dtype=np.uint8,
            )
            destination = self._sample_nearest_source_2d(source_2d)
            output_arr[year_idx - 1, :, :] = destination.astype(np.uint8)
            log(f"Sampled {self.landcover_variable} year {year_idx}/{total_years} ({year_value})")

    def _write_quality(self, root) -> None:
        output_arr = root[self.quality_variable]
        source_values = np.asarray(self.ds[self.quality_variable].values, dtype=np.uint8)
        output_arr[:] = source_values
        log(
            f"Wrote {self.quality_variable} with {source_values.size} values "
            f"and time chunk size {min(len(self.time_values), self.time_chunk_size)}"
        )

    def build(self) -> None:
        log(
            f"Target viewer grid: {self.target_width} columns x {self.target_height} rows "
            f"at nominal {self.viewer_resolution_m:.1f} m in {self.target_crs}"
        )
        root = self._open_output_root()
        self._create_output_arrays(root)
        self._write_lat_lon(root)
        self._build_source_lookup()
        self._write_quality(root)
        self._write_landcover(root)
        self._write_time_stack(root, self.display_variable)
        log(f"Wrote derived viewer dataset to {self.viewer_dataset_path}")


def main() -> None:
    cfg = load_config()
    builder = ViewerDataset3857Builder(cfg)
    builder.build()


if __name__ == "__main__":
    main()
