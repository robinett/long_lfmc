import numpy as np
import xarray as xr
import rioxarray
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box
import sys
import geopandas as gpd
from shapely import points

def create_grid(
    bounding_points,
    res,
    final_proj,
    conus_shp_fname
):
    '''
    Create a grid with the specified resolution and projection. Takes as an
    anchor the northwest point and uses the specified resolution to get as
    close to south and east as possible.
    Inputs:
        bounding_box (list): List of bounding box coordinates
        res (float): Resolution of the grid (in coordinates of your crs)
        final_proj (str): Projection string for the final grid
    '''
    # get your min and max values
    x_min, y_min, x_max, y_max = bounding_points
    # define the transformer that we will use throughout here
    transformer = Transformer.from_crs(
        "EPSG:4326", final_proj, always_xy=True
    )
    back_transformer = Transformer.from_crs(
        final_proj, "EPSG:4326", always_xy=True
    )
    # get our anchor piont
    x_west_limit, _ = transformer.transform(x_min[0],x_min[1])
    _, y_north_limit = transformer.transform(y_max[0],y_max[1])
    _, y_south_limit = transformer.transform(y_min[0],y_min[1])
    x_east_limit, _ = transformer.transform(x_max[0],x_max[1])
    # number of steps. we are going never going to go short
    nx = int(np.ceil((x_east_limit - x_west_limit) / res))
    ny = int(np.ceil((y_north_limit - y_south_limit) / res))
    # get our pixel centers
    x_centers = x_west_limit + (np.arange(nx) + 0.5) * res
    y_centers = y_north_limit - (np.arange(ny) + 0.5) * res
    xv,yv = np.meshgrid(x_centers, y_centers)
    # we want to fill with random numbers
    random_vals = np.random.random((ny, nx))
    # create the xarray
    grid = xr.Dataset(
        {
            "random_vals": (["y", "x"], random_vals)
        },
        coords={
            "x": ("x", x_centers),
            "y": ("y", y_centers)
        }
    )
    grid.rio.write_crs("EPSG:5070", inplace=True)
    # let's add back in lat and lon in wgs84 in case this is useful in
    # the future
    lon,lat = back_transformer.transform(
        xv,yv
    )
    grid = grid.assign_coords(
        lon=(("y", "x"), lon),
        lat=(("y", "x"), lat)
    )
    # load the shapefile that we will use to block out non-conus pixels
    print('masking non-conus pixels')
    conus_boundary = gpd.read_file(conus_shp_fname)
    conus_boundary = conus_boundary.to_crs("EPSG:5070")
    # Create a 2D grid of centers
    flat_coords = points(xv.ravel(), yv.ravel())
    # Mask grid based on U.S. boundary (convert grid to GeoDataFrame)
    grid_points = gpd.GeoDataFrame(
        geometry=flat_coords,
        crs="EPSG:5070"
    )
    joined = gpd.sjoin(
        grid_points,
        conus_boundary,
        how='left',
        predicate='within'
    )
    mask = ~joined.index_right.isna().to_numpy()
    mask_2d = mask.reshape(xv.shape)
    masked_grid = grid.where(mask_2d)
    # finally, let's get rid of any columns or rows that are all NaN
    print('removing empty rows and columns')
    masked_vals = masked_grid["random_vals"]
    valid_x = ~masked_vals.isnull().all(dim='y')
    valid_y = ~masked_vals.isnull().all(dim='x')
    final_grid = masked_grid.sel(
        x=masked_grid["x"][valid_x],
        y=masked_grid["y"][valid_y]
    )
    return final_grid




