import matplotlib
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import datetime
import copy
import os
import sys
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

sys.path.append('/home/users/trobinet/long_lfmc/data_processing/shared')
import plotting as plot

np.random.seed(42)

def main():
    """
    Main function to check MODIS data availability when different quality
    levels are used during the regridding.
    We randomly select 1000 pixels within western US and do a daily data check
    of data availability from 2003-2023.
    Then we plot the selected pixels and bar plot of the selected data
    available across all pixels.
    """
    # get the target grid so that we can selected 1000 relevant pixels
    target_grid_fname = (
        "/scratch/users/trobinet/long_lfmc/trent_datasets/grid/"
        "epsg5070_500m_westUS_grid.nc4"
    )
    target_grid = xr.open_dataset(
        target_grid_fname,
        engine="netcdf4"
    )
    data = target_grid["random_vals"].values
    valid_mask = ~np.isnan(data)
    valid_indices = np.argwhere(valid_mask)
    # select 1000 of these indices
    n_select = 10000
    selected_indices = valid_indices[
        np.random.choice(len(valid_indices), n_select, replace=False)
    ]
    selected_mask = np.full(data.shape, False)
    selected_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
    data_for_ex = np.where(selected_mask, data, np.nan)
    target_grid['for_ex'] = (('y', 'x'), data_for_ex)
    #plot.plot_from_xarray(
    #    'ds', target_grid, 'for_ex',
    #    proj_in='EPSG:5070', proj_out='EPSG:5070',
    #    fname='/scratch/users/trobinet/long_lfmc/trent_datasets/grid/'
    #          'example_modis_avail.png'
    #)
    #sys.exit()
    #x_coords = target_grid['x'].values[selected_indices[:, 1]]
    #y_coords = target_grid['y'].values[selected_indices[:, 0]]
    # loop over each day for each modis quality product and simply record the
    # number of pixel days that are not nan
    products = ['quality_0', 'quality_1', 'quality_2', 'quality_3']
    start_date = datetime.datetime(2010, 1, 1)
    current_date = copy.deepcopy(start_date)
    end_date = datetime.datetime(2019, 12, 31) #inclusive
    all_data_avail = {}
    for p,prod in enumerate(products):
        print(prod)
        this_prod_avail = 0
        this_prod_total = 0
        base_path = (
            "/scratch/users/trobinet/long_lfmc/trent_datasets/modis/"
            "modis_regridded/{}".format(prod)
        )
        # get an example file to extract variable names just once
        example_fname = os.path.join(
            base_path,
            current_date.strftime("%Y"),
            current_date.strftime("%m"),
            'modis_reflectance_{}{}{}_regridded.nc4'.format(
                current_date.strftime("%Y"),
                current_date.strftime("%m"),
                current_date.strftime("%d")
            )
        )
        example_ds = xr.open_dataset(
            example_fname,
            engine="netcdf4"
        )
        # get the variable names
        var_names = list(example_ds.data_vars)
        example_ds.close()
        while current_date <= end_date:
            print('prod: {} date: {}'.format(prod, current_date))
            this_date_fname = os.path.join(
                base_path,
                current_date.strftime("%Y"),
                current_date.strftime("%m"),
                'modis_reflectance_{}{}{}_regridded.nc4'.format(
                    current_date.strftime("%Y"),
                    current_date.strftime("%m"),
                    current_date.strftime("%d")
                )
            )
            this_ds = xr.open_dataset(
                this_date_fname,
                engine="netcdf4"
            )
            # loop over the variable names to check the data availability
            for var in var_names:
                this_data = this_ds[var].values
                #this_data = this_ds[var].sel(
                #    x=xr.DataArray(x_coords, dims='points'),
                #    y=xr.DataArray(y_coords, dims='points')
                #)
                chosen_data = this_data[selected_indices[:, 0], selected_indices[:, 1]]
                num_not_nan = np.sum(~np.isnan(chosen_data))
                this_prod_avail += num_not_nan
                this_prod_total += n_select
            this_ds.close()
            current_date += relativedelta(months=1)
        if p == 0:
            all_data_avail['total_obs'] = this_prod_total
        all_data_avail[prod] = this_prod_avail
        all_data_avail['{}_percent'.format(prod)] = (
            this_prod_avail / this_prod_total
        ) * 100
        print(all_data_avail)
        current_date = copy.deepcopy(start_date)
    # make a bar plot of data availability by product
    names = [
        'quality_0_percent', 'quality_1_percent',
        'quality_2_percent', 'quality_3_percent'
    ]
    vals = [
        all_data_avail['quality_0_percent'],
        all_data_avail['quality_1_percent'],
        all_data_avail['quality_2_percent'],
        all_data_avail['quality_3_percent']
    ]
    plt.figure()
    plt.bar(names, vals)
    plt.ylabel('Data availability (%)')
    plt.title('MODIS data availability by product')
    plt.savefig(
        '/scratch/users/trobinet/long_lfmc/trent_datasets/modis/'
        'modis_plots/availability_by_product.png',
        dpi=300,
        bbox_inches='tight'
    )

if __name__ == "__main__":
    main()

