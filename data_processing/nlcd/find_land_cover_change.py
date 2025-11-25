import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
import os
import numpy as np
import sys

sys.path.append(os.path.join(
    os.path.dirname(__file__),
    '..',
    'shared'
))

import plotting


def main():
    # when are we interested if there was land cover change?
    start_date = pd.Timestamp("2016-01-01")
    end_date = pd.Timestamp("2021-12-31")
    # load the nlcd @ 500m
    nlcd_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/nlcd/'
    nlcd_ds = xr.open_zarr(
        os.path.join(nlcd_dir, 'nlcd_target_grid_2003_2023.zarr'),
        chunks="auto"
    )
    nlcd_ds['nan'] = xr.DataArray(
        np.zeros_like(nlcd_ds['barren']),
        coords=nlcd_ds['barren'].coords,
        dims=nlcd_ds['barren'].dims
    )
    print('Loaded NLCD dataset:')
    print(nlcd_ds)
    # get the dominant land cover and % coverage at the end of the period
    print('Calculating dominant land cover at end of period...')
    ds_end = nlcd_ds.sel(year=pd.Timestamp(end_date.year, 1, 1))
    vars_lc = [
        "barren", "crops", "deciduous_forest", "developed",
        "evergreen_forest", "grass", "mixed_forest", "other",
        "shrub", "water", "wetlands", "nan"
    ]
    stacked = xr.concat(
        [ds_end[v] for v in vars_lc],
        dim=xr.DataArray(vars_lc, dims="lc", name="lc"),
    )
    dominant_name = stacked.idxmax("lc")
    dominant_percent = stacked.max("lc")
    out = xr.Dataset(
        {
            'dominant_land_cover': dominant_name,
            'dominant_percentage': dominant_percent
        },
        coords={
            'y': ds_end['y'],
            'x': ds_end['x'],
        }
    )
    out = out.chunk({'y':500, 'x':500})
    # great now check if the dominant land cover type changed by >20% for
    # any year in the period
    ds_sel = nlcd_ds.sel(
        year=slice(pd.Timestamp(start_date.year, 1, 1), pd.Timestamp(end_date.year - 1, 1, 1))
    )
    da_hist = ds_sel[vars_lc].to_array(dim="lc")
    with ProgressBar():
        da_hist = da_hist.compute()
    dom_name_end = out['dominant_land_cover']
    # replace float(nan) with 'nan' string to match da_hist lc names
    print('Replacing nan values...')
    dom_name_end = dom_name_end.fillna('nan')
    dom_perc_end = out['dominant_percentage']
    print('Selecting historical data for dominant land cover type...')
    dom_labels = dom_name_end.data  # np.ndarray of shape (y, x)
    # Map to integer positions along the `lc` dim
    # any missing label will get -1
    lc_index = da_hist['lc'].to_index()
    codes = lc_index.get_indexer(dom_labels.ravel())
    codes = codes.reshape(dom_labels.shape)
    dom_codes = xr.DataArray(
        codes,
        dims=dom_name_end.dims,
        coords={k: v for k, v in dom_name_end.coords.items()
                if k in dom_name_end.dims},
        name='lc_index'
    )
    dom_history = da_hist.isel(lc=dom_codes)
    print('Calculating delta in dominant land cover percentage over time...')
    delta = abs(dom_history - dom_perc_end)
    thresh = 0.2
    print(f'Applying threshold of {thresh}% to determine land cover change...')
    change_flag = (delta > thresh).any(dim="year")
    change_flag = change_flag.astype("int8")
    # apply nans according to original data
    print('Applying original data mask to change flag...')
    original_mask = ds_end['barren'].isnull()
    change_flag = change_flag.where(~original_mask)
    print('Adding land cover change flag to output dataset...')
    out['land_cover_change_flag'] = change_flag
    # plot the land cover change flag for sanity
    print('Plotting land cover change flag...')
    plotting.plot_from_xarray(
        load_type='ds',
        type_obj=out,
        var='land_cover_change_flag',
        proj_in='EPSG:5070',
        proj_out='EPSG:5070',
        fname=os.path.join(
            nlcd_dir,
            'plots',
            f'nlcd_land_cover_change_flag_{start_date.year}_{end_date.year}.png'
        ),
    )
    print(out)
    print('Saving output dataset...')
    save_name = os.path.join(
        nlcd_dir,
        f'nlcd_land_cover_change_{start_date.year}_{end_date.year}.zarr'
    )
    with ProgressBar():
        out.to_zarr(save_name, mode='w')

if __name__ == "__main__":
    main()