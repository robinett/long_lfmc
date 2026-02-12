import sys
import os

sys.path.append('/home/users/trobinet/long_lfmc/data_processing/shared')
import create_grid
import plotting

def main():
    base_dir = '/home/users/trobinet/long_lfmc/data_processing/grid'
    scratch_dir = '/scratch/users/trobinet/long_lfmc/final_lfmc/grid'
    # Define the bounding box via points for the extent in each direction
    x_west = [-124.826015, 40.430961]   # westernmost point of CONUS
    x_east = [-93.508292,  29.764377]   # easternmost point of TX (near Sabine Pass)
    y_south = [-97.396381, 25.837163]   # southernmost point of TX (near Brownsville)
    y_north = [-123.369077, 49.199870]  # northern border across WA/MT
    my_proj = "EPSG:5070"
    bounding_points = [x_west, y_south, x_east, y_north]
    res = 500
    # where is the conus shapefile?
    conus_shp_fname = os.path.join(
        scratch_dir,
        'conus_shapefile/cb_2024_us_state_5m.shp'
    )
    #base_dir = '/home/users/trobinet/long_lfmc/data_processing/grid'
    #scratch_dir = '/scratch/users/trobinet/long_lfmc/final_lfmc/grid'
    ## Define the bounding box. we will do this via the points for the extent in
    ## each direction
    #x_west = [-124.826015,40.430961] # wetsernmost point of CONUS in EPSG:5070
    #x_east = [-101.933353,41.004590] # eastern border of colorado
    #y_south = [-103.147714,28.909267] # southernmost point of NM
    #y_north = [-123.369077,49.199870] # northern border across WA/MT
    ## define the projection we want to use
    #my_proj = "EPSG:5070" # Albers Equal Area Conic projection
    ## Create the bounding box
    #bounding_points = [x_west, y_south, x_east, y_north]
    ## define the resolution, in meters
    #res = 500
    ## where is the conus shapefile?
    #conus_shp_fname = os.path.join(
    #    scratch_dir,
    #    'conus_shapefile/cb_2024_us_nation_5m.shp'
    #)
    # where do we want to save our grid?
    grid_fname = os.path.join(
        scratch_dir,
        'epsg5070_500m_westUS_grid.nc4'
    )
    # Create the grid
    print('creating grid')
    grid = create_grid.create_grid(
        bounding_points, res, my_proj, conus_shp_fname
    )
    print('plotting grid')
    plotting.plot_from_xarray(
        'ds',grid,'random_vals',
        'EPSG:5070','EPSG:5070',
        os.path.join(
            scratch_dir,
            'plots',
            'grid_w_random_values.png'
        )
    )
    print('saving grid')
    grid.to_netcdf(
        grid_fname,
        format='NETCDF4',
        engine='netcdf4',
        encoding={
            var: {"zlib": True, "complevel": 4}
            for var in grid.data_vars
        }
    )

if __name__ == "__main__":
    main()
