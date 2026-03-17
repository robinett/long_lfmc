import os
import shutil
import sys

import pandas as pd
import xarray as xr

sys.path.append('/scratch/users/trobinet/long_lfmc/trent_datasets/shared/')
from pandas_to_xarray import pandas_to_xarray
import regridder
import plotting as plot


SCRATCH_ROOT = "/scratch/users/trobinet/long_lfmc"
FINAL_ROOT = f"{SCRATCH_ROOT}/final_lfmc"
OAK_TRENT_ROOT = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets"
SCRATCH_STATIC_DIR = f"{FINAL_ROOT}/static"
SCRATCH_STATIC_RAW_DIR = f"{SCRATCH_STATIC_DIR}/raw"
SCRATCH_PLOTS_DIR = f"{SCRATCH_STATIC_DIR}/plots"
TARGET_GRID_FNAME = f"{FINAL_ROOT}/grid/epsg5070_500m_westUS_grid.nc4"


def _stage_to_scratch(scratch_path, oak_path):
    os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
    if os.path.exists(scratch_path):
        print(f"Using scratch copy: {scratch_path}")
        return
    if not os.path.exists(oak_path):
        raise FileNotFoundError(
            f"Scratch copy missing at {scratch_path} and source missing at {oak_path}"
        )
    print(f"Copying raw static file to scratch: {oak_path} -> {scratch_path}")
    shutil.copy2(oak_path, scratch_path)


def main():
    pd_to_xr = False
    regrid_xr = True
    raw_static_fname = f"{SCRATCH_STATIC_RAW_DIR}/static_features_p36_250m_latlon_float32.pkl"
    ds_fname = f"{SCRATCH_STATIC_RAW_DIR}/static_features_250m_latlon_float32.nc"
    target_grid_fname = TARGET_GRID_FNAME
    regrid_ds_fname = f"{SCRATCH_STATIC_DIR}/static_features_500m_epsg5070_float32.nc"
    src_crs = 'EPSG:4326'
    target_crs = 'EPSG:5070'
    _stage_to_scratch(
        raw_static_fname,
        f"{OAK_TRENT_ROOT}/static/static_features_p36_250m_latlon_float32.pkl",
    )
    _stage_to_scratch(
        ds_fname,
        f"{OAK_TRENT_ROOT}/static/static_features_250m_latlon_float32.nc",
    )
    if pd_to_xr:
        raw_df = pd.read_pickle(raw_static_fname)
        print('raw_df:')
        print(raw_df)
        print('raw_df columns:')
        print(raw_df.columns)
        columns_to_keep = {
            'latitude': 'lat',
            'longitude': 'lon',
            'slope(t)': 'slope',
            'elevation(t)': 'elevation',
            'canopy_height(t)': 'canopy_height',
            'forest_cover(t)': 'forest_cover',
            'silt(t)': 'silt',
            'clay(t)': 'clay',
            'sand(t)': 'sand'
        }
        ex_var_to_plot = 'elevation'
        ds = pandas_to_xarray(
            raw_df,
            columns_to_keep,
            ex_var_to_plot
        )
        ds.to_netcdf(
            ds_fname,
            format='NETCDF4',
            engine='netcdf4',
            encoding={
                'lat': {'dtype': 'float32'},
                'lon': {'dtype': 'float32'},
                'slope': {'dtype': 'float32'},
                'elevation': {'dtype': 'float32'},
                'canopy_height': {'dtype': 'float32'},
                'forest_cover': {'dtype': 'float32'},
                'silt': {'dtype': 'float32'},
                'clay': {'dtype': 'float32'},
                'sand': {'dtype': 'float32'}
            }
        )
    if  regrid_xr:
        if not pd_to_xr:
            ds = xr.open_dataset(
                ds_fname,
                engine='netcdf4'
            )
        ds = ds.rename({'lat':'y', 'lon':'x'})
        ds.rio.write_crs(src_crs, inplace=True)
        target_grid = xr.open_dataset(
            target_grid_fname,
            engine='netcdf4'
        )
        target_chunks,_,_ = regridder.chunk_xr_dataset(
            target_grid,
            chunk_size=1000
        )
        regridded_ds = regridder.reproject_and_regrid_single_file(
            target_grid,
            ds,
            target_crs,
            src_crs,
            target_chunks,
            plot_tests=False,
            target_dir_last_ext='static',
            chunk_buffer=1000
        )
        plot.plot_from_xarray(
            'ds',
            regridded_ds,
            'elevation',
            target_crs,
            target_crs,
            f'{SCRATCH_PLOTS_DIR}/elev_regrid.png'
        )
        regridder.save_xarray_w_encoding(
            regridded_ds,
            regrid_ds_fname
        )


if __name__ == '__main__':
    main()
