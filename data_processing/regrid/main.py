import os
import sys
from datetime import datetime
import argparse

import regridder

def main():
    # inputs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_grid",
        type=str,
        help="Path to the target grid .nc file",
        required=True
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        help="Path to the source directory with .nc files to regrid",
        required=True
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="Path to the output directory for regridded .nc files",
        required=True
    )
    parser.add_argument(
        "--src_crs",
        type=str,
        help="Coordinate reference system of the source data",
        required=True
    )
    parser.add_argument(
        "--target_crs",
        type=str,
        help="Coordinate reference system of the target data",
        required=True
    )
    parser.add_argument(
        "--fill_value",
        type=str,
        help="Fill used in src data",
        default='none'
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size for regridding",
        default=2000
    )
    parser.add_argument(
        "--chunk_buffer",
        type=int,
        help="Buffer size for chunking",
        default=200
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Optional first YYYY-MM-DD date to regrid when source filenames contain daily dates",
        default=None
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="Optional last YYYY-MM-DD date to regrid when source filenames contain daily dates",
        default=None
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip source files whose expected target file already exists",
    )
    target_grid_dir = parser.parse_args().target_grid
    src_dir = parser.parse_args().src_dir
    target_dir = parser.parse_args().target_dir
    src_crs = parser.parse_args().src_crs
    target_crs = parser.parse_args().target_crs
    chunk_size = parser.parse_args().chunk_size
    chunk_buffer = parser.parse_args().chunk_buffer
    start_date = parser.parse_args().start_date
    end_date = parser.parse_args().end_date
    skip_existing = parser.parse_args().skip_existing
    if parser.parse_args().fill_value != 'none':
        fill_value = float(parser.parse_args().fill_value)
    else:
        fill_value = parser.parse_args().fill_value
    # general directories that we shoudl have
    home_dir = '/home/users/trobinet/long_lfmc/data_processing/regrid'
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets'
    # we are going to pass in a lot of our arguments. do that here
    # give the directory for an .nc that is the target dataset. This should the
    # target in terms of crs, resolution, and bounds.
    #target_grid_dir = os.path.join(
    #    scratch_dir,
    #    'grid',
    #    'epsg5070_500m_westUS_grid.nc4'
    #)
    # pass directory to regrid. all files in this directory and all
    # sub-directories will be regird and same directory structure will be
    # maintained for outputs. all files need to be the same projection,
    # resolution, and spatial extent.
    #src_dir = os.path.join(
    #    scratch_dir,
    #    'modis',
    #    'modis_processed_daily'
    #)
    # where should we put the modis outputs of the regridding?
    #target_dir = os.path.join(
    #    scratch_dir,
    #    'modis',
    #    'modis_regridded'
    #)
    # what is the crs of the src data?
    #src_crs = '+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs'
    # what is the crs of the target data?
    #target_crs = 'EPSG:5070'
    # what is the chunk size for the regridding?
    # alright, let's do this!
    print('calling regridder')
    regridder.reproject_and_regrid_whole_directory(
        src_dir,
        target_dir,
        target_grid_dir,
        src_crs,
        target_crs,
        chunk_size=chunk_size,
        fill_value=fill_value,
        chunk_buffer=chunk_buffer,
        start_date=start_date,
        end_date=end_date,
        skip_existing=skip_existing,
    )


if __name__ == '__main__':
    main()
