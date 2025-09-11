import pickle
import pandas as pd
import xarray as xr
import sys

sys.path.append('/home/users/trobinet/long_lfmc/data_processing/shared')
import plotting as plot

def pandas_to_xarray(
    df,
    columns_to_keep_mapping,
    ex_var_to_plot
):
    df = df[
        list(
            columns_to_keep_mapping.keys()
        )
    ].rename(
        columns=columns_to_keep_mapping
    )
    print('sorting')
    df = df.sort_values(['lat', 'lon'])
    num_dups = df.duplicated(subset=['lat', 'lon']).sum()
    if num_dups > 0:
        raise ValueError(
            'There are duplicate lat/lon pairs in the dataframe.'
            'Please remove'
        )
    # make sure that we don't have any duplicate lat/lon pairs
    df_indexed = df.set_index(['lat', 'lon'])
    print('df indexed by lat and lon:')
    print(df_indexed)
    # convert to dataset
    ds = df_indexed.to_xarray()
    print('ds')
    print(ds)
    print('plotting example variable')
    example_fname = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/regridding/'
        'plots/static_features_{}.png'.format(ex_var_to_plot)
    )
    plot.plot_from_xarray(
        load_type='ds',
        type_obj=ds,
        var=ex_var_to_plot,
        proj_in='EPSG:4326',
        proj_out='EPSG:4326',
        fname=example_fname
    )
    return ds


