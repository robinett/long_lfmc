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
        "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    # Use a concrete cartopy projection to ensure x_limits/y_limits exist
    larbers_proj_cartopy = ccrs.LambertConformal(
        central_longitude=-100,
        central_latitude=42.5,
        standard_parallels=(25, 60),
    )
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
        '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs':larbers_proj_cartopy,
    }
    try:
        coded_proj = proj_dict[proj]
    except KeyError:
        raise KeyError(
            f"Projection {proj} not found in proj_dict. "
            f"Available projections are {proj_dict.keys()}"
        )
    return coded_proj

def plot_from_xarray(
    load_type, type_obj, var,
    proj_in, proj_out,
    fname, cmap='rainbow',
    extent=None,
    extent_crs=None,
    title=None,
    cbar_label=None,
    vmin=None,
    vmax=None,
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
        west_us = [-120, -93, 23, 52]
        ax.set_extent(west_us, crs=get_proj('EPSG:4326'))
    else:
        extent_proj = coded_in if extent_crs is None else get_proj(extent_crs)
        ax.set_extent(extent, crs=extent_proj)

    # --- plot as image (2D -> cmap ok) ---
    im = da.plot.imshow(
        ax=ax,
        transform=coded_in,
        cmap=cmap,
        robust=True,
        add_colorbar=True,
        cbar_kwargs=None if cbar_label is None else {"label": cbar_label},
        vmin=vmin,
        vmax=vmax,
        rasterized=True
    )

    # --- decorations ---
    ax.add_feature(cfeature.COASTLINE, linewidth=0.15)
    ax.add_feature(cfeature.STATES, linewidth=0.10)
    if title is not None:
        ax.set_title(title)

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


def choose_random_valid_time_index_stacked(da_time_yx, rng, sample_stride=256):
    """
    Choose a random time index with at least one finite value using a sparse
    spatial sample for speed.
    """
    sample = da_time_yx
    if "y" in sample.dims and "x" in sample.dims:
        sample = sample.isel(y=slice(None, None, sample_stride),
                             x=slice(None, None, sample_stride))
        valid_any = sample.notnull().any(dim=("y", "x")).compute().values
    else:
        valid_any = sample.notnull().any().compute().values
    idx = np.flatnonzero(valid_any)
    if len(idx) == 0:
        return None
    return int(rng.choice(idx))


def save_qc_map_from_dataarray(
    da2d,
    out_path,
    title,
    colorbar_label=None,
    cmap='viridis',
    downsample_stride=8,
    dpi=150,
):
    """
    Save a quick 2D QC map from an xarray DataArray. If lon/lat coords exist and
    are 2D, use pcolormesh; otherwise fall back to imshow.
    """
    plot_da = da2d
    if "y" in da2d.dims and "x" in da2d.dims and downsample_stride and downsample_stride > 1:
        plot_da = da2d.isel(y=slice(None, None, downsample_stride),
                            x=slice(None, None, downsample_stride))
    plot_da = plot_da.load()

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if ("lon" in plot_da.coords and "lat" in plot_da.coords and
            getattr(plot_da["lon"], "ndim", 0) == 2 and getattr(plot_da["lat"], "ndim", 0) == 2):
        mesh = ax.pcolormesh(
            plot_da["lon"].values,
            plot_da["lat"].values,
            plot_da.values,
            shading="auto",
            cmap=cmap,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        mesh = ax.imshow(plot_da.values, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(mesh, ax=ax, shrink=0.85, label=colorbar_label)
    ax.set_title(title)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_interpolation_diagnostic_timeseries(
    times,
    original_series_list,
    interpolated_series_list,
    status_series_list,
    point_labels,
    save_name,
    title=None,
):
    """
    Plot original vs interpolated time series for several sampled pixels.

    Parameters
    ----------
    times : array-like
        Time axis shared by all series.
    original_series_list : list[np.ndarray]
        Original values (with NaNs) for each sampled point.
    interpolated_series_list : list[np.ndarray]
        Interpolated values for each sampled point.
    status_series_list : list[np.ndarray]
        Fill status arrays (0 original-valid, 1 interpolated, 2 still missing).
    point_labels : list[str]
        Labels for each sampled point.
    save_name : str
        Output figure path.
    title : str, optional
        Figure title.
    """
    n_panels = len(point_labels)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 3.5 * n_panels),
        sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        orig = np.asarray(original_series_list[i], dtype=float)
        interp = np.asarray(interpolated_series_list[i], dtype=float)
        status = np.asarray(status_series_list[i])

        ax.plot(times, interp, color='tab:blue', linewidth=1.8, label='interpolated')
        ax.plot(
            times, orig,
            color='black',
            linewidth=1.0,
            linestyle='--',
            marker='o',
            markersize=2.5,
            alpha=0.8,
            label='original'
        )

        filled_mask = status == 1
        if np.any(filled_mask):
            ax.scatter(
                np.asarray(times)[filled_mask],
                interp[filled_mask],
                s=22,
                color='tab:red',
                label='filled days',
                zorder=3
            )

        still_missing_mask = status == 2
        if np.any(still_missing_mask):
            y_min, y_max = ax.get_ylim()
            y_mark = y_min + 0.05 * (y_max - y_min if y_max > y_min else 1.0)
            ax.scatter(
                np.asarray(times)[still_missing_mask],
                np.full(still_missing_mask.sum(), y_mark),
                s=12,
                color='gray',
                alpha=0.7,
                label='still missing',
                zorder=2
            )

        ax.set_ylabel("Value")
        ax.set_title(point_labels[i])
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc='best')

    axes[-1].set_xlabel("Time")
    plt.xticks(rotation=45)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    save_dir = os.path.dirname(save_name)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_name, dpi=200)
    plt.close(fig)
