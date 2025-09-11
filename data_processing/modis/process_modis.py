from pyhdf.SD import SD, SDC
import re
import sys
import numpy as np
import glob
import xarray as xr
import datetime
import os

sys.path.append('/home/users/trobinet/long_lfmc/data_processing/shared')
import plotting as plot

def explore_single_file(fname):
    '''
    Function for exploring a single modis file
    '''
    # open the file
    file = SD(fname, SDC.READ)
    print(file.info())
    datasets = file.datasets()
    for idx,sds in enumerate(datasets.keys()):
        print(f'{idx}: {sds}')
    reflectance_1 = file.select('Nadir_Reflectance_Band1')
    r_1_data = reflectance_1.get()
    print('r_1_data:')
    print(r_1_data)
    # let's see if we can get metadata
    attrs = file.attributes()
    for idx,attr in enumerate(attrs.keys()):
        print(f'{idx}: {attr}')
    metadata = attrs.get('StructMetadata.0', '')
    print(metadata)

def get_metadata(
    modis_raw_dir,
    start_date
):
    '''
    Function for creating daily netcdf files from modis hdf files
    '''
    print('getting metadata for all tiles')
    # get the tiles that we are dealing with
    first_day_year = start_date.strftime('%Y')
    first_day_dir = os.path.join(
        modis_raw_dir,
        first_day_year
    )
    modis_files = sorted(
        glob.glob(
            os.path.join(
                first_day_dir,
                'MCD43A4*.hdf'
            )
        )
    )
    first_day_files = [
        f for f in modis_files
        if f.split('.')[1] == 'A' + start_date.strftime('%Y%j')
    ]
    # get the iv and ih tiles from these files
    ivs = np.zeros(0)
    ihs = np.zeros(0)
    for f in first_day_files:
        # get the tile number
        tile_numbers = f.split('/')[-1].split('.')[2]
        #print(tile_numbers)
        # get the iv and ih from the tile number
        ih = int(tile_numbers[1:3])
        iv = int(tile_numbers[5:])
        # append to the arrays
        ihs = np.append(ihs, ih)
        ivs = np.append(ivs, iv)
    # get the metadata for each tile so that we don't have to do this every day
    # later
    extracted_metadata = {}
    for h,htile in enumerate(ihs):
        vtile = ivs[h]
        # get the file for the first day corresponding to this tile
        this_date_section = 'A' + start_date.strftime('%Y%j')
        this_tile_section = 'h' + f"{int(htile):02}" + 'v' + f"{int(vtile):02}"
        tile_fname = [
            f for f in first_day_files
            if (
                f.split('.')[1] == this_date_section
                and f.split('.')[2] == this_tile_section
            )
        ][0]
        # open the file and get the metadata
        sd = SD(tile_fname, SDC.READ)
        # Read the global attribute
        struct_meta_str = sd.attributes()["StructMetadata.0"]
        xdim = int(re.search(r"XDim=(\d+)", struct_meta_str).group(1))
        ydim = int(re.search(r"YDim=(\d+)", struct_meta_str).group(1))
        ul_match = re.search(
            r"UpperLeftPointMtrs=\((-?\d+\.\d+),(-?\d+\.\d+)\)",
            struct_meta_str
        )
        lr_match = re.search(
            r"LowerRightMtrs=\((-?\d+\.\d+),(-?\d+\.\d+)\)",
            struct_meta_str
        )
        ul = (float(ul_match.group(1)), float(ul_match.group(2)))
        lr = (float(lr_match.group(1)), float(lr_match.group(2)))
        # create the grid for this tile
        ulx = ul[0]
        uly = ul[1]
        lrx = lr[0]
        lry = lr[1]
        # compute pixel size
        dx = np.abs((lrx - ulx) / xdim)
        dy = np.abs((uly - lry) / ydim)
        # 1d array of pixel centers
        x = np.linspace(ulx + dx/2, lrx - dx/2, xdim)
        y = np.linspace(uly + dy/2, lry - dy/2, ydim)
        # 2d meshgrid of pixel centers
        # xx, yy = np.meshgrid(x, y)
        extracted_metadata[this_tile_section] = {
            'xdim': xdim,
            'ydim': ydim,
            'ul': ul,
            'lr': lr,
            'x': x,
            'y': y,
        }
        # close the file
        sd.end()
    return extracted_metadata

def regrid_to_daily_ncs(
    modis_raw_dir,
    metadata,
    start_date,
    end_date,
    layer_names,
    tiles_per_day,
    out_dir,
    quality_flag=0,
    precision=-9999
):
    '''
    Function for regridding modis files to daily netcdf files.
    Parameters
    ----------
    modis_raw_dir : str
        Directory where the modis raw files are located.
    metadata : dict
        Metadata for each tile. Comes from get_metadata function.
    start_date : datetime
        Start date for the data.
    end_date : datetime
        End date for the data.
    layer_names : dict
        Dictionary of layer names. Keys are 'data' and 'quality'.
        Values are lists of layer names.
    out_dir : str
        Directory where the output files will be saved.
    quality_flag : int
        Quality flags to use. Default is 0. Includes all quality flags <=
        passed quality_flag. For example, if quality_flags = 2, then data with
        quality flags 0, 1, and 2 will be included. Quality flag definitions
        are:
            0 = best quality, full inversion (WoDs, RMSE majority good)
            1 = good quality, full inversion (also including the cases that
                no clear sky observations over the day of interest or the
                Solar Zenith Angle is too large even WoDs, RMSE majority good)
            2 = Magnitude inversion (numobs >=7)
            3 = Magnitude inversion (numobs >=2&<7)
            4 = Fill value
    '''
    current_date = start_date
    data_layers = layer_names['data']
    quality_layers = layer_names['quality']
    num_layers = len(data_layers)
    # our data has a scale factor. Make sure to include that here
    modis_scale_factor = 0.0001
    while current_date <= end_date:
        # files only for today
        today_files = sorted(
            glob.glob(
                os.path.join(
                    modis_raw_dir,
                    current_date.strftime('%Y'),
                    'MCD43A4.A{}.*.hdf'.format(
                        current_date.strftime('%Y%j')
                    )
                )
            )
        )
        # check if we have the right number of files
        if len(today_files) != tiles_per_day:
            raise ValueError(
                'Number of files for {} is {}. We expect {}.'.format(
                    current_date.strftime('%Y-%m-%d'),
                    len(today_files),
                    tiles_per_day
                )
            )
        today_quality_files = sorted(
            glob.glob(
                os.path.join(
                    modis_raw_dir,
                    current_date.strftime('%Y'),
                    'MCD43A2.A{}.*.hdf'.format(
                        current_date.strftime('%Y%j')
                    )
                )
            )
        )
        all_datasets = []
        print('working on {}'.format(
            current_date.strftime('%Y-%m-%d')
        ))
        print('extracting data')
        for f,file in enumerate(today_files):
            #print('processing file {}'.format(file.split('/')[-1]))
            this_sd = SD(file, SDC.READ)
            # get the modis-formatted date and tile number from this file
            modis_date = file.split('/')[-1].split('.')[1]
            tile_number = file.split('/')[-1].split('.')[2]
            #print('extracting data on {} for tile {}'.format(
            #    current_date.strftime('%Y-%m-%d'),
            #    tile_number
            #))
            quality_file = []
            # there is at least one quality date/tile combination that is
            # not avialable on the LP DAAC. Super weird. But in this case
            # we just assume that all data is bad out of safety.
            for q_file in today_quality_files:
                if (
                    q_file.split('/')[-1].split('.')[1] == modis_date
                    and q_file.split('/')[-1].split('.')[2] == tile_number
                ):
                    quality_file.append(q_file)
                    break
            if len(quality_file) > 0:
                this_quality_file = quality_file[0]
                this_quality_sd = SD(this_quality_file, SDC.READ)
            else:
                print('no quality file for {}'.format(file))
            for l,layer in enumerate(data_layers):
                # get the layer  
                this_band = this_sd.select(layer)
                this_band_data = np.array(this_band.get()).astype(np.float32)
                # get rid of all fill values
                this_band_data[
                    this_band_data == 32767
                ] = np.nan
                # apply the scale factor
                this_band_data = this_band_data * modis_scale_factor
                ### WE USED TO ONLY CHECK QUALITY FILE IN SOME CIRCUMSTANCES.
                ### DOING THIS FOR ALL FILES NOW
                ## get the quality layer
                #this_quality = this_sd.select(quality_layers[l])
                #this_quality_data = this_quality.get()
                ## check for need for further investiation
                #low_quality_idx = np.where(
                #    (this_quality_data > 0) & (this_quality_data < 255)
                #)
                #has_low_quality = low_quality_idx[0].size > 0
                ## we need to do some further checks if quality is questionable.
                ## we will only not use if it is snowy or it is water
                # get the corresponding quality file
                if len(quality_file) == 0:
                    # for the strange case where we don't have any modis
                    # quality information
                    # set all data to nan
                    this_band_data[
                        this_band_data != np.nan
                    ] = np.nan
                else:
                    # else perform a normal quality check for inversion
                    # quality, snow, and water
                    quality_info = this_quality_sd.select(
                        quality_layers[l]
                    ).get()
                    snow_info = this_quality_sd.select(
                        'Snow_BRDF_Albedo'
                    ).get()
                    water_info = this_quality_sd.select(
                        'BRDF_Albedo_LandWaterType'
                    ).get()
                    water_vals = np.unique(water_info)
                    # find places with acceptable quality,
                    # no snow, and containing land (for now we will roll with
                    # coastlines as well)
                    # condition 1: acceptable quality
                    condition_1 = (quality_info <= quality_flag)
                    # condition 2: no snow
                    condition_2 = (snow_info == 0)
                    # condition 3: contains land
                    condition_3 = (
                        (water_info == 1) | (water_info == 2)
                    )
                    # set the idx of the locations where each of these three
                    # contitions are not met and this_band_data is not already
                    # nan
                    bad_quality_idx = np.where(
                        (
                            (condition_1 == False) |
                            (condition_2 == False) |
                            (condition_3 == False)
                        ) & (
                            this_band_data != np.nan
                        )
                    )
                    # set these to nan
                    this_band_data[
                        bad_quality_idx
                    ] = np.nan
                # if this is the first layer, we need to create teh dataset
                if l == 0:
                    # we are going to round coords to the nearest tenth of a
                    # meter to facilitate the datasets being concatenated.
                    if precision != -9999:
                        x_rounded = np.round(
                            metadata[tile_number]['x']/ precision
                        ) * precision
                        y_rounded = np.round(
                            metadata[tile_number]['y']/ precision
                        ) * precision
                    else:
                        x_rounded = metadata[tile_number]['x']
                        y_rounded = metadata[tile_number]['y']
                    this_ds = xr.Dataset(
                        {
                            layer: (["y", "x"], this_band_data)
                        },
                        coords={
                            "x":("x", x_rounded),
                            "y":("y", y_rounded)
                        }
                    )
                    this_ds.attrs["crs"] = "EPSG:SR-ORG:6974"
                    # testing our plotting function
                    #plot.plot_from_xarray(
                    #    'ds',
                    #    this_ds,
                    #    layer,
                    #    'modis_sinusoidal',
                    #    'modis_sinusoidal',
                    #    os.path.join(
                    #        '/scratch/users/trobinet/long_lfmc/trent_datasets/modis_plots',
                    #        'modis_raw_{date}_{tile}_{layer}.png'.format(
                    #            date=current_date.strftime('%Y-%m-%d'),
                    #            tile=tile_number,
                    #            layer=layer
                    #        )
                    #    )
                    #)
                else:
                    this_ds[layer] = (["y","x"], this_band_data)
            # add the attributes to the dataset
            copied_ds = this_ds.copy(deep=True)
            all_datasets.append(copied_ds)
        # combine all these into a single dataset
        # get the x and y coords
        x_coords = np.zeros(0)
        y_coords = np.zeros(0)
        for ds in all_datasets:
            this_x = ds.x.values
            this_y = ds.y.values
            if this_x[0] not in x_coords:
                x_coords = np.append(x_coords, this_x)
            if this_y[0] not in y_coords:
                y_coords = np.append(y_coords, this_y)
        # sort the x and y coords
        full_x = np.sort(x_coords)
        full_y = np.sort(y_coords)
        # create the dataset that we will place everything into
        nan_data = np.full(
            (len(full_y), len(full_x)),
            np.nan
        )
        combined_ds = xr.Dataset(
            coords={
                "x": ("x", full_x),
                "y": ("y", full_y)
            }
        )
        combined_ds.attrs["crs"] = "EPSG:SR-ORG:6974"
        # add nan data to the dataset
        print('creating combined dataset')
        for l,layer in enumerate(data_layers):
            #print('{}: adding layer {} to combined dataset'.format(
            #    current_date.strftime('%Y-%m-%d'),
            #    layer
            #))
            # add the data to the dataset
            nan_data = np.full(
                (len(full_y), len(full_x)),
                np.nan
            )
            combined_ds[layer] = (["y", "x"], nan_data)
            for d,ds in enumerate(all_datasets):
                # get the x and y coords
                x_coords = ds.x.values
                y_coords = ds.y.values
                # get the data
                data = ds[layer].values
                # add the data to the dataset
                combined_ds[layer].loc[
                    {
                        "x": x_coords,
                        "y": y_coords
                    }
                ] = data.copy()
        if current_date == start_date:
            print('example of combined dataset:')
            print(combined_ds)
            plot.plot_from_xarray(
                'ds',
                combined_ds,
                data_layers[0],
                '+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs',
                '+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +units=m +no_defs',
                os.path.join(
                    '/scratch/users/trobinet/long_lfmc/trent_datasets/modis',
                    'modis_plots',
                    'combined_dataset_{}_{}.png'.format(
                        data_layers[0],
                        current_date.strftime('%Y%m%d')
                    )
                )
            )
        # save the dataset to a netcdf file
        this_fname = 'modis_reflectance_{}.nc4'.format(
            current_date.strftime('%Y%m%d')
        )
        this_out_dir = os.path.join(
            out_dir,
            current_date.strftime('%Y'),
            current_date.strftime('%m'),
            this_fname
        )
        # ensure this directory already exists
        os.makedirs(
            os.path.dirname(this_out_dir),
            exist_ok=True
        )
        # save to this path
        combined_ds.to_netcdf(
            this_out_dir,
            format='NETCDF4',
            encoding={
                var: {
                    'zlib': True,
                    'complevel': 5,
                    'chunksizes': (1000, 1000),
                    'dtype': 'float32'
                } for var in combined_ds.data_vars
            }
        )
        # increment the date
        current_date += datetime.timedelta(days=1)

