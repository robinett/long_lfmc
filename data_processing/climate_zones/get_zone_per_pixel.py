import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import sys
import os
import numpy as np

def get_climate_zone_per_pixel(
    target_grid_path,
    climate_zone_path,
    output_path
):
    # load the target grid
    target_grid = xr.open_dataset(target_grid_path)
    # load the climate zone files
    climate_zones = rxr.open_rasterio(climate_zone_path)
    # get the climate zone per pixel
    climate_zones_resampled = climate_zones.rio.reproject_match(
        target_grid, resampling=Resampling.nearest
    )
    # convert to xarray dataset with var name 'climate_zone'
    climate_zones_resampled = climate_zones_resampled.to_dataset(name='climate_zone')
    # assign the correct coordinates
    climate_zones_resampled = climate_zones_resampled.assign_coords(
        {
            'x':target_grid['x'],
            'y':target_grid['y']
        }
    )
    print(climate_zones_resampled)
    # save the output
    climate_zones_resampled.to_netcdf(output_path)

def main():
    target_grid_path = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/grid/'
        'epsg5070_500m_westUS_grid.nc4'
    )
    climate_zone_path = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/climate_zones/1991_2020/'
        'koppen_geiger_0p1.tif'
    )
    output_path = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/climate_zones/'
        'climate_zone_per_pixel_westUS.nc4'
    )
    get_climate_zone_per_pixel(
        target_grid_path,
        climate_zone_path,
        output_path
    )


if __name__ == "__main__":
    main()