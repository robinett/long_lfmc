import xarray as xr
from rasterio.enums import Resampling
import os
import sys
import datetime
import re
import numpy as np
from pyproj import Transformer

sys.path.append('/home/users/trobinet/long_lfmc/data_processing/shared')
import plotting as plot


DAILY_DATE_PATTERN = re.compile(r'_(\d{8})(?:_regridded)?\.[^.]+$')


def format_time_value_as_yyyymmdd(time_value):
    """
    Convert an xarray time value (numpy datetime64 or cftime-like object) to
    YYYYMMDD for daily output file naming.
    """
    if isinstance(time_value, np.ndarray):
        time_value = time_value.item()
    if isinstance(time_value, np.datetime64):
        if np.isnat(time_value):
            raise ValueError('Encountered NaT time value in source dataset')
        time_str = np.datetime_as_string(time_value, unit='D')
        return time_str.replace('-', '')
    if (
        hasattr(time_value, 'year') and
        hasattr(time_value, 'month') and
        hasattr(time_value, 'day')
    ):
        return (
            f"{int(time_value.year):04d}"
            f"{int(time_value.month):02d}"
            f"{int(time_value.day):02d}"
        )
    raise ValueError(
        'Unable to parse time value for output naming: {}'.format(
            repr(time_value)
        )
    )


def get_regrid_output_path(
    src_file_path,
    src_dir,
    target_dir,
    date_str=None
):
    """
    Build the output path for a regridded file. If date_str is provided, insert
    a YYYYMMDD token into the basename so multi-time inputs can emit one file
    per day using the same naming convention as single-day inputs.
    """
    relative_subpath = os.path.dirname(
        os.path.relpath(
            src_file_path,
            src_dir
        )
    )
    target_save_full_dir = os.path.join(
        target_dir,
        relative_subpath
    )
    this_base, this_ext = os.path.splitext(os.path.basename(src_file_path))
    if this_ext in ['.tif', '.tiff', '.img', '.hdf', '.hdf4', '.hdf5', '.h5']:
        this_ext = '.nc4'
    elif this_ext in ['.nc', '.nc4', '.ncdf']:
        pass
    else:
        print('unrecognized file extension: {}'.format(this_ext))
        print('exiting')
        sys.exit()
    if date_str is not None:
        if re.search(r'_\d{8}$', this_base):
            # Already looks like a single-day file basename.
            pass
        elif re.search(r'_\d{4}$', this_base):
            # Replace trailing year token (e.g., Daymet yearly file) with day.
            this_base = re.sub(r'_\d{4}$', '_{}'.format(date_str), this_base)
        else:
            # Generic fallback for other multi-time files.
            this_base = '{}_{}'.format(this_base, date_str)
    this_fname = f"{this_base}_regridded{this_ext}"
    return os.path.join(
        target_save_full_dir,
        this_fname
    )

def reproject_and_regrid_whole_directory(
    src_dir,
    target_dir,
    target_grid_fname,
    src_crs,
    target_crs,
    chunk_size=500,
    fill_value='none',
    chunk_buffer=200,
    single_file=False,
    resampling=Resampling.nearest,
    start_date=None,
    end_date=None,
    skip_existing=False,
):
    """
    Parameters
    ----------
    src_dir : str
        Directory containing the source dataset files to be regridded.
    target_dir : str
        Directory where the regridded dataset files will be saved.
    target_grid_fname : str
        Filename of the target grid dataset.
    src_crs : str
        Coordinate reference system of the source dataset.
    target_crs : str
        Coordinate reference system of the target dataset.
    chunk_size : int, optional
        Size of the chunks to use for regridding. Default is 500.
    """
    # just setup; throughout all the check plots in this class, we want to
    # select the first variable that is not x/y/lon/lat/time
    print('opening the target grid')
    target_grid = xr.open_dataset(
        target_grid_fname,
        engine='h5netcdf'
    )
    # add the target crs here for safety
    target_grid.rio.write_crs(target_crs, inplace=True)
    # set up the directory structure for the output
    print('setting up output directory structure')
    mirror_directory_structure(src_dir, target_dir)
    # call the appropriate regridding function
    # let's get the last extension of the target dir. this is just what we
    # are going to name our various plots so that we can differentiate them
    # later
    target_dir_last_ext = os.path.basename(
        os.path.normpath(target_dir)
    )
    print('chunking the target grid')
    # chunk the target grid in to reasonable sizes
    target_chunks,target_chunk_x_idxs,target_chunk_y_idxs = chunk_xr_dataset(
        target_grid,
        chunk_size
    )
    num_target_chunks = len(target_chunks)
    target_chunk_mem = target_chunks[0].nbytes / 1024**2
    print('target grid chunked into {} chunks with size of {} MB'.format(
        num_target_chunks,
        target_chunk_mem
    ))
    # get all of the files that have been passed to us to regrid
    src_file_paths = get_all_file_paths(src_dir)
    if start_date is not None or end_date is not None:
        start_dt = (
            datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            if start_date is not None else None
        )
        end_dt = (
            datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            if end_date is not None else None
        )
        src_file_paths = filter_daily_files_by_date(src_file_paths, start_dt, end_dt)
        print(
            'filtered source files to {} daily files between {} and {}'.format(
                len(src_file_paths),
                start_dt,
                end_dt,
            )
        )
    plotted_first_output = False
    # loop over each file
    for sfp,this_src_file_path in enumerate(src_file_paths):
        expected_target_path = get_regrid_output_path(
            this_src_file_path,
            src_dir,
            target_dir,
        )
        if skip_existing and os.path.exists(expected_target_path):
            print('skipping existing regrid output {}'.format(expected_target_path))
            continue
        print('working on file {}'.format(
            this_src_file_path
        ))
        print('this is file file {}/{}'.format(
            sfp+1,
            len(src_file_paths)
        ))
        # load the file that we are regridding.
        this_src_ds = xr.open_dataset(
            this_src_file_path
        )
        drop_if_present = ["time_bnds", "yearday"]
        this_src_ds = this_src_ds.drop_vars(
            [v for v in drop_if_present if v in this_src_ds.data_vars]
        )
        #print(np.unique(this_src_ds['band_data'].values))
        #print(np.where(np.isnan(this_src_ds['band_data'].values)))
        #continue
        if fill_value != 'none':
            # fill the fill value with nans
            this_src_ds = this_src_ds.where(
                this_src_ds != fill_value
            )
        # add the source crs here for safety
        this_src_ds.rio.write_crs(src_crs, inplace=True)
        num_time_steps = this_src_ds.sizes.get('time', 0)
        is_multi_time_file = ('time' in this_src_ds.dims and num_time_steps > 1)
        if is_multi_time_file:
            print(
                'detected multi-time source file with {} time steps; '
                'regridding one day at a time'.format(num_time_steps)
            )
            time_values = this_src_ds['time'].values
            src_ds_iter = []
            for ti in range(num_time_steps):
                date_str = format_time_value_as_yyyymmdd(time_values[ti])
                src_ds_iter.append((
                    this_src_ds.isel(time=ti, drop=True),
                    date_str,
                    ti + 1,
                    num_time_steps
                ))
        else:
            print('detected single-time/daily source file')
            src_ds_iter = [(this_src_ds, None, 1, 1)]

        for this_src_ds_to_regrid, date_str, ti, nt in src_ds_iter:
            if date_str is not None:
                print('working on day {}/{} ({})'.format(ti, nt, date_str))
            this_regridded_ds = reproject_and_regrid_single_file(
                target_grid,
                this_src_ds_to_regrid,
                target_crs,
                src_crs,
                target_chunks,
                plot_tests=False,
                target_dir_last_ext=target_dir_last_ext,
                chunk_buffer=chunk_buffer,
                resampling=resampling
            )
            # make sure that we have a crs written to the output
            if 'crs' not in this_regridded_ds.coords:
                this_regridded_ds.rio.write_crs(target_crs, inplace=True)
            if not plotted_first_output:
                print('regridded ds from first output:')
                print(this_regridded_ds)
                print('plotting the final regridded ds')
                # get the first var that isn't excluded
                exclude_when_plotting = [
                    'x',
                    'y',
                    'lon',
                    'lat',
                    'time'
                ]
                for var in this_regridded_ds.data_vars:
                    if var.lower() not in exclude_when_plotting:
                        this_var = var
                        break
                plot.plot_from_xarray(
                    'ds',
                    this_regridded_ds,
                    this_var,
                    target_crs,
                    target_crs,
                    (
                        '/scratch/users/trobinet/long_lfmc/final_lfmc/' +
                        'regridding/plots/final_regridded_ds_{}.png'.format(
                            target_dir_last_ext
                        )
                    )
                )
                plotted_first_output = True
            target_save_fname = get_regrid_output_path(
                this_src_file_path,
                src_dir,
                target_dir,
                date_str=date_str
            )
            if os.path.exists(target_save_fname):
                print(
                    'WARNING: overwriting existing regridded file: {}'.format(
                        target_save_fname
                    )
                )
            save_xarray_w_encoding(
                this_regridded_ds,
                target_save_fname
            )
            this_regridded_ds.close()
            if this_src_ds_to_regrid is not this_src_ds:
                this_src_ds_to_regrid.close()
        this_src_ds.close()
    target_grid.close()
    
def reproject_and_regrid_single_file(
    target_grid,
    this_src_ds,
    target_crs,
    src_crs,
    target_chunks,
    plot_tests=False,
    target_dir_last_ext='',
    chunk_buffer=200,
    resampling=Resampling.nearest
):
    # create what will be the final dataset. this is just the target
    # grid without the data that was included in this.
    this_regridded_ds = target_grid.copy()
    this_regridded_ds = this_regridded_ds.drop_vars(
        list(this_regridded_ds.data_vars)
    )
    # let's drop all dimensions of size 1 here
    this_src_ds = this_src_ds.squeeze()
    # add all variables that we will include. Initialize as nan
    vars_to_add = list(this_src_ds.data_vars)
    #dims_to_add = this_src_ds[vars_to_add[0]].dims
    dims_to_add = ['y','x']
    shape_to_add = tuple(
        this_regridded_ds.sizes[dim] for dim in dims_to_add
    )
    for var in vars_to_add:
        this_regridded_ds[var] = xr.DataArray(
            data=np.full(shape_to_add,np.nan),
            dims=dims_to_add
        )
    # loop over each chunk of the target grid
    for tc,this_target_chunk in enumerate(target_chunks):
        print('working on chunk {}/{}'.format(
            tc+1,
            len(target_chunks)
        ))
        # get the corresponding, padded chunk of the source grid
        # we want this chunk to extend at least 10 pixels beyond the
        # boundary of the target chunk in each direction.
        #print('getting chunk')
        this_padded_src_chunk = get_padded_chunk(
            this_target_chunk,
            this_src_ds,
            num_padding_pixels=chunk_buffer
        )
        if plot_tests:
            # fill all nans for this plot to show the full extent
            print('plotting datasets on top of each other')
            this_padded_src_chunk_plot = this_padded_src_chunk.fillna(0.0)
            if 'time' in this_padded_src_chunk_plot.dims:
                this_padded_src_chunk_plot = this_padded_src_chunk_plot.isel(
                    time=0,
                    drop=True
                )
            # get the first var that isn't excluded
            exclude_when_plotting = [
                'x',
                'y',
                'lon',
                'lat',
                'time'
            ]
            for sv in this_padded_src_chunk_plot.data_vars:
                if sv.lower() not in exclude_when_plotting:
                    src_var = sv
                    break
            for tv in this_target_chunk.data_vars:
                if tv.lower() not in exclude_when_plotting:
                    tgt_var = tv
                    break
            plot.plot_multiple_xarray_datasets(
                ['ds','ds'],
                [this_target_chunk,this_padded_src_chunk_plot],
                [tgt_var,src_var],
                [
                    target_crs,
                    src_crs
                ],
                target_crs,
                (
                    '/scratch/users/trobinet/long_lfmc/final_lfmc/' +
                    'regridding/plots/target_chunk_and_src_chunk_w_buffer_{}.png'.format(
                        target_dir_last_ext
                    )
                ),
                ['copper','winter'],
                [0.5,0.5]
            )
        # match the target grid
        #print('reprojecting')
        this_padded_src_chunk_reproj = this_padded_src_chunk.rio.reproject_match(
            this_target_chunk,
            resampling=resampling
        )
        if plot_tests:
            print('plotting the regridded chunk')
            plot.plot_from_xarray(
                'ds',
                this_target_chunk,
                tgt_var,
                target_crs,
                target_crs,
                (
                    '/scratch/users/trobinet/long_lfmc/final_lfmc/' +
                    'regridding/plots/target_chunk_{}.png'.format(
                        target_dir_last_ext
                    )
                )
            )
            plot.plot_from_xarray(
                'ds',
                this_padded_src_chunk,
                src_var,
                src_crs,
                target_crs,
                (
                    '/scratch/users/trobinet/long_lfmc/final_lfmc/' +
                    'regridding/plots/this_padded_src_chunk_{}.png'.format(
                        target_dir_last_ext
                    )
                )
            )
            plot.plot_from_xarray(
                'ds',
                this_padded_src_chunk_reproj,
                src_var,
                target_crs,
                target_crs,
                (
                    '/scratch/users/trobinet/long_lfmc/final_lfmc/' +
                    'regridding/plots/src_chunk_regridded_{}.png'.format(
                        target_dir_last_ext
                    )
                )
            )
        # add to this_regridded_ds
        #print('combining')
        #print(this_regridded_ds)
        #print(this_padded_src_chunk_reproj)
        this_regridded_ds = this_regridded_ds.combine_first(
            this_padded_src_chunk_reproj
        )
    # first, nan out all the values that were nan in our target grid,
    # since presumably we don't want these
    target_grid_mask = target_grid[list(target_grid.data_vars)[0]].isnull()
    this_regridded_ds = this_regridded_ds.where(
        ~target_grid_mask
    )
    return this_regridded_ds

def save_xarray_w_encoding(
    this_regridded_ds,
    target_save_fname
):
    encoding = {}
    # Encode all data variables
    for var in this_regridded_ds.data_vars:
        dims = this_regridded_ds[var].dims
        shape = this_regridded_ds[var].shape
        chunks = []
        for dim, size in zip(dims, shape):
            if 'time' in dim:
                chunks.append(min(size, 1))
            elif 'x' in dim or 'lon' in dim:
                chunks.append(min(size, 1000))
            elif 'y' in dim or 'lat' in dim:
                chunks.append(min(size, 1000))
            else:
                chunks.append(size)
        encoding[var] = {
            'zlib': True,
            'complevel': 5,
            'chunksizes': tuple(chunks),
            'dtype': 'float32'
        }
    # Optionally compress 2D coordinate variables (e.g., lat/lon)
    for coord in ['lat', 'lon']:
        if coord in this_regridded_ds.coords:
            dims = this_regridded_ds[coord].dims
            shape = this_regridded_ds[coord].shape
            if len(dims) == 2:
                chunks = []
                for dim, size in zip(dims, shape):
                    if 'x' in dim or 'lon' in dim:
                        chunks.append(min(size, 1000))
                    elif 'y' in dim or 'lat' in dim:
                        chunks.append(min(size, 1000))
                    else:
                        chunks.append(size)
                encoding[coord] = {
                    'zlib': True,
                    'complevel': 5,
                    'chunksizes': tuple(chunks),
                    'dtype': 'float32'
                }
    # Save the dataset
    this_regridded_ds.to_netcdf(
        target_save_fname,
        format='NETCDF4',
        encoding=encoding
    )
def get_all_file_paths(src_dir):
    """
    Get all file paths in the source directory and its subdirectories.

    Parameters
    ----------
    src_dir : str
        Source directory to search for files.

    Returns
    -------
    list
        List of file paths.
    """
    file_paths = []
    for dirpath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    file_paths_sorted = sorted(file_paths)
    return file_paths_sorted


def extract_daily_date_from_path(path):
    match = DAILY_DATE_PATTERN.search(os.path.basename(path))
    if match is None:
        return None
    return datetime.datetime.strptime(match.group(1), '%Y%m%d').date()


def filter_daily_files_by_date(file_paths, start_date, end_date):
    filtered = []
    for path in file_paths:
        file_date = extract_daily_date_from_path(path)
        if file_date is None:
            continue
        if start_date is not None and file_date < start_date:
            continue
        if end_date is not None and file_date > end_date:
            continue
        filtered.append(path)
    return filtered
def mirror_directory_structure(src_root,dst_root):
    for dir_path,dir_names,_ in os.walk(src_root):
        # get the relative path to the source root
        rel_path = os.path.relpath(dir_path, src_root)
        # create the corresponding directory in the destination root
        dst_dir = os.path.join(dst_root, rel_path)
        # make the directory if it doesn't exist
        os.makedirs(dst_dir,exist_ok=True)
def chunk_xr_dataset(
    ds,
    chunk_size,
    x_dim_name='x',
    y_dim_name='y'
):
    """
    Chunk an xarray dataset into smaller chunks for processing.
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be chunked.
    chunk_size : int
        The size of the chunks.
    x_dim_name : str, optional
        The name of the x dimension in the dataset. Default is 'x'.
    y_dim_name : str, optional
        The name of the y dimension in the dataset. Default is 'y'.
    Returns
    -------
    chunks: list
        A list containing the chunked datasets.
    """
    x_chunks = np.arange(
        0,
        ds.sizes[x_dim_name],
        chunk_size
    )
    y_chunks = np.arange(
        0,
        ds.sizes[y_dim_name],
        chunk_size
    )
    chunks = []
    for i in x_chunks:
        for j in y_chunks:
            this_chunk = ds.isel({
                x_dim_name: slice(i, i + chunk_size),
                y_dim_name: slice(j, j + chunk_size)
            })
            chunks.append(this_chunk.copy())
    return chunks,x_chunks,y_chunks
def get_padded_chunk(
    target_chunk,
    src_ds,
    num_padding_pixels=10,
    target_x_dim_name='x',
    target_y_dim_name='y',
    src_x_dim_name='x',
    src_y_dim_name='y'
):
    """
    Get a padded chunk of the source dataset based on the target chunk.

    Parameters
    ----------
    target_chunk : xarray.Dataset
        The target chunk to be padded.
    src_ds : xarray.Dataset
        The source dataset.
    num_padding_pixels : int, optional
        Number of pixels to pad around the target chunk. Default is 10.
    target_x_dim_name : str, optional
        The name of the x dimension in the target dataset. Default is 'x'.
    target_y_dim_name : str, optional
        The name of the y dimension in the target dataset. Default is 'y'.
    src_x_dim_name : str, optional
        The name of the x dimension in the source dataset. Default is 'x'.
    src_y_dim_name : str, optional
        The name of the y dimension in the source dataset. Default is 'y'.

    Returns
    -------
    xarray.Dataset
        Padded chunk of the source dataset.
    """
    # get the bounds of the target chunk in its crs
    target_x_min = target_chunk[target_x_dim_name].min().item()
    target_x_max = target_chunk[target_x_dim_name].max().item()
    target_y_min = target_chunk[target_y_dim_name].min().item()
    target_y_max = target_chunk[target_y_dim_name].max().item()
    # get the resolution of the target
    target_x_res = abs(
        target_chunk[target_x_dim_name][1] -
        target_chunk[target_x_dim_name][0]
    ).item()
    target_y_res = abs(
        target_chunk[target_y_dim_name][1] -
        target_chunk[target_y_dim_name][0]
    ).item()
    # add a 10-pixel buffer
    buffer_x = target_x_res * num_padding_pixels
    buffer_y = target_y_res * num_padding_pixels
    target_x_min_buf = target_x_min - buffer_x
    target_x_max_buf = target_x_max + buffer_x
    target_y_min_buf = target_y_min - buffer_y
    target_y_max_buf = target_y_max + buffer_y
    # Transform all four buffered target corners to source CRS so rotated or
    # non-linear projections keep the full source footprint for this chunk.
    transformer = Transformer.from_crs(
        target_chunk.rio.crs,
        src_ds.rio.crs,
        always_xy=True
    )
    src_corners = [
        transformer.transform(target_x_min_buf, target_y_min_buf),
        transformer.transform(target_x_min_buf, target_y_max_buf),
        transformer.transform(target_x_max_buf, target_y_min_buf),
        transformer.transform(target_x_max_buf, target_y_max_buf),
    ]
    src_x_vals = [corner[0] for corner in src_corners]
    src_y_vals = [corner[1] for corner in src_corners]
    src_x_min = min(src_x_vals)
    src_x_max = max(src_x_vals)
    src_y_min = min(src_y_vals)
    src_y_max = max(src_y_vals)
    # get the subset
    # make this robust to coordinates possibly descending by default in
    # src_ds as opposed to asscending
    x_ascending = src_ds.x.values[0] < src_ds.x.values[-1]
    y_ascending = src_ds.y.values[0] < src_ds.y.values[-1]
    if x_ascending:
        x_slice = slice(src_x_min,src_x_max)
    else:
        x_slice = slice(src_x_max,src_x_min)
    if y_ascending:
        y_slice = slice(src_y_min,src_y_max)
    else:
        y_slice = slice(src_y_max,src_y_min)
    src_subset = src_ds.sel({
        src_x_dim_name: x_slice,
        src_y_dim_name: y_slice
    })
    return src_subset
