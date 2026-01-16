import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
import xarray as xr
from pyproj import CRS
import rioxarray

def get_proj(proj):
    """
    Get the projection object from a string.
    Parameters
    ----------
    proj : str
        Projection to get.
    Returns
    -------
    proj : cartopy.crs.Projection
        Projection object.
    """
    # we need to define the daymet lambers projection with a custom proj string
    lambers_proj_str = (
        "+proj=lcc +lat_1=25 + lat_2=60 +lat_0=42.5 +lon_0=-100 "
        "+x_0=0 + y_0=0 +datum=WGS84 +units=km +no_defs"
    )
    lambers_proj_crs = CRS.from_proj4(lambers_proj_str)
    larbers_proj_cartopy = ccrs.Projection(lambers_proj_crs)
    proj_dict = {
        'EPSG:5070':ccrs.AlbersEqualArea(
            central_longitude=-96,
            central_latitude=23,
            false_easting=0,
            false_northing=0,
            standard_parallels=(29.5, 45.5),
            globe=ccrs.Globe(datum='NAD83')
        ),
        'EPSG:4326':ccrs.PlateCarree(),
        '+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs':ccrs.Sinusoidal.MODIS,
        '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=km +no_defs':larbers_proj_cartopy,
    }
    try:
        coded_proj = proj_dict[proj]
    except KeyError:
        raise KeyError(
            f"Projection {proj} not found in proj_dict. "
            "Available projections are {proj_dict.keys()}"
        )
    return coded_proj

def plot_from_xarray(
    load_type, type_obj, var,
    proj_in, proj_out,
    fname, cmap='rainbow',
    extent=None
):
    # --- load ---
    if load_type == 'fname':
        if str(type_obj).endswith('.tif'):
            da = rioxarray.open_rasterio(
                type_obj,
                engine='rasterio',
                mask_and_scale=True
            ).squeeze(drop=True)
            ds = da.to_dataset(name=var)
        else:
            ds = xr.open_dataset(type_obj, engine='netcdf4')
    elif load_type == 'ds':
        ds = type_obj
    elif load_type == 'da':
        ds = type_obj.to_dataset(name=var)
    else:
        raise ValueError("load_type must be 'fname' or 'ds'.")

    if var not in ds:
        raise KeyError(f"'{var}' not in dataset.")

    da = ds[var]

    # --- reduce to 2D (auto-slice) ---
    # common spatial names
    y_names = ['y', 'latitude', 'lat']
    x_names = ['x', 'longitude', 'lon']
    dims = list(da.dims)

    # squeeze length-1 dims first
    da = da.squeeze(drop=True)

    def pick_first(name_list, dims):
        for n in name_list:
            if n in dims:
                return n
        return None

    ydim = pick_first(y_names, da.dims)
    xdim = pick_first(x_names, da.dims)

    # if still 3+ dims, drop non-spatial by .isel(index=0)
    # prefer common non-spatial dims in this order
    drop_order = ['band', 'time', 'variable', 'layer']
    for d in drop_order:
        if d in da.dims and da.ndim > 2:
            da = da.isel({d: 0}).squeeze(drop=True)

    # if unknown extra dims remain, drop first indices
    while da.ndim > 2:
        for d in da.dims:
            if d not in (ydim, xdim):
                da = da.isel({d: 0}).squeeze(drop=True)
                break

    # final guard: ensure 2D
    if da.ndim != 2:
        raise RuntimeError(
            f"Could not reduce '{var}' to 2D; "
            f"got dims={da.dims}."
        )

    # --- projections & axes ---
    coded_in = get_proj(proj_in)
    coded_out = get_proj(proj_out)

    fig, ax = plt.subplots(
        subplot_kw={'projection': coded_out},
        figsize=(7, 6)
    )

    # if not extent given, use western US
    if extent is None:
        west_us = [-126, -99, 20, 55]
        ax.set_extent(west_us, crs=get_proj('EPSG:4326'))
    else:
        ax.set_extent(extent, crs=coded_in)



    # --- plot as image (2D -> cmap ok) ---
    im = da.plot.imshow(
        ax=ax,
        transform=coded_in,
        cmap=cmap,
        robust=True,
        add_colorbar=True,
        rasterized=True
    )

    # --- decorations ---
    ax.add_feature(cfeature.COASTLINE, linewidth=0.15)
    ax.add_feature(cfeature.STATES, linewidth=0.10)

    # --- save ---
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


#def plot_from_xarray(
#    load_type,type_obj,var,
#    proj_in,proj_out,
#    fname,cmap='rainbow'
#):
#    # if load type is 'fname', load the file
#    if load_type == 'fname':
#        #ds = xr.open_dataset(type_obj,engine='netcdf4')
#        # check if file extension is tif
#        if type_obj.endswith('.tif'):
#            da = rioxarray.open_rasterio(
#                type_obj,engine='rasterio',mask_and_scale=True
#            ).squeeze()
#            ds = da.to_dataset(name=var)
#        else:
#            ds = xr.open_dataset(type_obj,engine='netcdf4')
#    if load_type == 'ds':
#        ds = type_obj
#    # get the in projection
#    coded_proj_in = get_proj(proj_in)
#    # get the out projection
#    coded_proj_out = get_proj(proj_out)
#    fig,ax = plt.subplots(subplot_kw={'projection':coded_proj_out})
#    # set extent to western US
#    west_us_extent = [-126,-99,20,55]
#    ax.set_extent(west_us_extent,crs=get_proj('EPSG:4326'))
#    ds[var].plot(
#        ax=ax,transform=coded_proj_in,
#        cmap=cmap
#    )
#    ax.add_feature(cfeature.COASTLINE,linewidth=0.15)
#    ax.add_feature(cfeature.STATES,linewidth=0.1)
#    # save the figure
#    savename = '{fname}'.format(fname=fname)
#    plt.savefig(savename,dpi=300,bbox_inches='tight')
#    plt.close()

def plot_multiple_xarray_datasets(
    load_types,type_objs,vars_to_plot,
    projs_in,proj_out,
    fname,cmaps,alphas
):
    """
    Plot multiple xarray datasets on the same figure.
    Parameters
    ----------
    load_types : list
        List of load types. 'fname' or 'ds'.
    type_objs : list
        List of xarray datasets or file names.
    vars_to_plot : list
        List of variables to plot. Each variable should correspond to one
        type_obj that was passed.
    projs_in : str
        List of projection of the input data.
    proj_out : str
        Projection to plot at.
    fname : str
        File name to save the figure as.
    cmaps : str
        List of colormaps to use for each variable. Each colormap should
        correspond to one variable that was passed.
    alphas : float
        List of alpha values to use for each variable. Each alpha value
        should correspond to one variable that was passed.
    """
    # get the out projection
    coded_proj_out = get_proj(proj_out)
    #set up the plot
    fig,ax = plt.subplots(subplot_kw={'projection':coded_proj_out})
    # set extent to western US
    west_us_extent = [-126,-99,20,55]
    ax.set_extent(west_us_extent,crs=get_proj('EPSG:4326'))
    for o,obj in enumerate(type_objs):
        # if load type is 'fname', load the file
        if load_types[o] == 'fname':
            ds = xr.open_dataset(type_objs[o],engine='h5netcdf')
        if load_types[o] == 'ds':
            ds = type_objs[o]
        coded_proj_in = get_proj(projs_in[o])
        ds[vars_to_plot[o]].plot(
            ax=ax,transform=coded_proj_in,
            cmap=cmaps[o],alpha=alphas[o]
        )
    ax.add_feature(cfeature.COASTLINE,linewidth=0.15)
    ax.add_feature(cfeature.STATES,linewidth=0.1)
    # save the figure
    savename = '{fname}'.format(fname=fname)
    plt.savefig(savename,dpi=300,bbox_inches='tight')
    plt.close()

def plot_timeseries(times,vals,xlabel,ylabel,save_name,title=None,time_bound=None):
    # get rid of NaNs
    times = times[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]

    plt.figure(figsize=(12, 6))
    plt.plot(times, vals, marker='o', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    if time_bound:
        plt.xlim(time_bound)
    if title:
        plt.title(title)
    plt.savefig(save_name)
    plt.close()