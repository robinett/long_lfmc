#!/usr/bin/env python3

"""
Minimal example for opening a long_lfmc Zarr store from Source Cooperative.

Install dependencies first if needed:
    pip install "xarray[complete]" s3fs zarr pyproj
"""

import xarray as xr
from pyproj import Transformer


SOURCE_ENDPOINT_URL = "https://data.source.coop"
SOURCE_PRODUCT_PREFIX = "rseg/long-lfmc-test/"
REMOTE_ZARR_RELPATH = "lfmc_maps.zarr"

TIME_START = "2023-07-10"
TIME_END = "2023-07-15"
POINT_LAT = 37.5
POINT_LON = -122.2

WGS84_TO_LFMC_GRID = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)


def source_store_uri(product_prefix: str, remote_zarr_relpath: str) -> str:
    product_prefix = product_prefix.strip().strip("/")
    remote_zarr_relpath = remote_zarr_relpath.strip().strip("/")
    return f"s3://{product_prefix}/{remote_zarr_relpath}"


def open_remote_lfmc_dataset(product_prefix: str, remote_zarr_relpath: str) -> xr.Dataset:
    store_uri = source_store_uri(product_prefix, remote_zarr_relpath)
    return xr.open_zarr(
        store_uri,
        chunks={},
        consolidated=False,
        storage_options={
            "anon": True,
            "client_kwargs": {"endpoint_url": SOURCE_ENDPOINT_URL},
        },
    )


def coordinate_slice(values, lower_value: float, upper_value: float) -> slice:
    if values[0] <= values[-1]:
        return slice(lower_value, upper_value)
    return slice(upper_value, lower_value)


def example_time_subset(ds: xr.Dataset) -> xr.Dataset:
    subset = ds.sel(time=slice(TIME_START, TIME_END))
    print("Time subset")
    print(subset[["lfmc_ens_mean", "lfmc_ens_std"]])
    return subset


def example_point_timeseries(ds: xr.Dataset) -> xr.DataArray:
    grid_x, grid_y = WGS84_TO_LFMC_GRID.transform(POINT_LON, POINT_LAT)
    point_series = ds["lfmc_ens_mean"].sel(x=grid_x, y=grid_y, method="nearest")
    nearest_x = float(point_series["x"].item())
    nearest_y = float(point_series["y"].item())

    print("")
    print(f"Nearest LFMC grid point to lon={POINT_LON}, lat={POINT_LAT}")
    print(f"Grid coordinates: x={nearest_x:.1f}, y={nearest_y:.1f}")
    print(point_series.to_series().head())
    return point_series


def example_bbox_subset(ds: xr.Dataset) -> xr.Dataset:
    west_lon = -123.3
    east_lon = -121.8
    south_lat = 36.7
    north_lat = 38.2

    west_x, south_y = WGS84_TO_LFMC_GRID.transform(west_lon, south_lat)
    east_x, north_y = WGS84_TO_LFMC_GRID.transform(east_lon, north_lat)

    subset = ds.sel(
        x=coordinate_slice(ds["x"].values, min(west_x, east_x), max(west_x, east_x)),
        y=coordinate_slice(ds["y"].values, min(south_y, north_y), max(south_y, north_y)),
        time=slice(TIME_START, TIME_END),
    )

    print("")
    print("Bounding-box subset")
    print(subset[["lfmc_ens_mean", "lfmc_ens_std"]])
    return subset


def main():
    ds = open_remote_lfmc_dataset(
        product_prefix=SOURCE_PRODUCT_PREFIX,
        remote_zarr_relpath=REMOTE_ZARR_RELPATH,
    )

    print("Opened dataset")
    print(ds)
    print("")
    print(f"Public Source endpoint: {SOURCE_ENDPOINT_URL}")
    print(
        "Zarr path:",
        source_store_uri(SOURCE_PRODUCT_PREFIX, REMOTE_ZARR_RELPATH),
    )

    example_time_subset(ds)
    example_point_timeseries(ds)
    example_bbox_subset(ds)


if __name__ == "__main__":
    main()
