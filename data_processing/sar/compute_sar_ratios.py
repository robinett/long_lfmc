import os
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import shutil
import sys

def calc_ratio(sar_da, modis_ds, modis_name):
    #sar_da = sar_da.sel(time=slice(start_date, end_date))
    modis_da = (
        modis_ds["data"]
        .sel(variable=modis_name)
    )
    r = (sar_da / modis_da).squeeze(drop=True)
    # drop junk coords if present
    to_drop = [c for c in ["spatial_ref", "variable"] if c in r.coords]
    if to_drop:
        r = r.drop_vars(to_drop)
    return r

def write_var(name, da, out_store):
    ds = da.to_dataset(name=name)
    # chunk for safety
    ds = ds.chunk({"time": 1, "y": 512, "x": 512})
    with ProgressBar():
        ds.to_zarr(out_store, mode="a")

def main():
    oak_dir = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets"
    scratch_dir = "/scratch/users/trobinet/long_lfmc/trent_datasets"
    start = pd.Timestamp(2016, 4, 24)
    end   = pd.Timestamp(2023, 10, 3)
    daily_index = pd.date_range(start=start, end=end, freq='D')
    out_store = os.path.join(
        scratch_dir, "sar", "sar_500m_filled_ratios.zarr"
    )
    ## we need to delete the zarr store if it exists to avoid contamination
    #if os.path.exists(out_store):
    #    resp = input(f"{out_store} exists. Delete? [y/n] ")
    #    if resp.lower().startswith("y"):
    #        if out_store.endswith(".zarr"):
    #            shutil.rmtree(out_store)
    #        else:
    #            raise RuntimeError("Refusing to delete non-.zarr path")
    chunks = {"time": 1, "y": 512, "x": 512}
    vv_ds = xr.open_zarr(
        os.path.join(scratch_dir, "sar", "sar_500m_filled_vv.zarr"),
        chunks=chunks,
    )
    vv_ds = vv_ds.sel(time=slice(start, end))
    vv_ds = vv_ds.reindex(time=daily_index, method="ffill")
    #vv_ds = vv_da.to_dataset(name="vv_backscatter")
    vh_ds = xr.open_zarr(
        os.path.join(scratch_dir, "sar", "sar_500m_filled.zarr"),
        chunks=chunks,
    )
    vh_ds = vh_ds.sel(time=slice(start, end))
    vh_ds = vh_ds.reindex(time=daily_index, method="ffill")
    #vh_ds = vh_da.to_dataset(name="vh_backscatter")
    modis_ds = xr.open_zarr(
        os.path.join(
            oak_dir, "modis", "modis_regridded_gapfilled",
            "quality_1", "interpolated", "modis_all_vars.zarr"
        ),
        chunks=chunks,
    )
    modis_ds = modis_ds.sel(time=slice(start, end))
    modis_ds = modis_ds.reindex(time=daily_index, method="ffill")
    #master_time = modis_ds.time.sel(time=slice(start, end))
    #vv_da = vv_ds["vv_backscatter"].reindex(time=master_time)
    #vh_da = vh_ds["vh_backscatter"].reindex(time=master_time)
    ## forward fill SAR onto MODIS daily grid
    #tchunk = 64
    #vv_da = vv_da.chunk({"time": tchunk, 'y': 512, 'x': 512})
    #vh_da = vh_da.chunk({"time": tchunk, 'y': 512, 'x': 512})
    #vv_da = vv_da.ffill("time", limit=10)
    #vh_da = vh_da.ffill("time", limit=10)
    #vv_ds = vv_da.to_dataset(name="vv_backscatter")
    #vh_ds = vh_da.to_dataset(name="vh_backscatter")
    #vv_ds = vv_ds.chunk({"time": 1, "y": 512, "x": 512})
    #vh_ds = vh_ds.chunk({"time": 1, "y": 512, "x": 512})
    # Initialize output with coords/time slice only (cheap)
    base = vv_ds.drop_vars(
        [v for v in vv_ds.data_vars]  # keep coords only
    )
    
    print(base)
    print(vv_ds)
    print(vh_ds)
    print(modis_ds)
    with ProgressBar():
        # chunk for safety
        base = base.chunk({"time": 1, "y": 512, "x": 512})
        base.to_zarr(out_store, mode="w")
    # vv - vh
    print('Writing vv_minus_vh')
    now = pd.Timestamp.now()
    print(f'now: {now}')
    write_var(
        "vv_minus_vh",
        (vv_ds["vv_backscatter"] - vh_ds["vh_backscatter"]),
        out_store
    )
    bands = [
        "Nadir_Reflectance_Band1_filled",
        "Nadir_Reflectance_Band2_filled",
        "Nadir_Reflectance_Band3_filled",
        "Nadir_Reflectance_Band4_filled",
        "Nadir_Reflectance_Band5_filled",
        "Nadir_Reflectance_Band6_filled",
        "Nadir_Reflectance_Band7_filled",
    ]
    for i, b in enumerate(bands, start=1):
        print(f"Writing vv ratio for {b}")
        now = pd.Timestamp.now()
        print(f'now: {now}')
        write_var(
            f"vv_over_{i}",
            calc_ratio(vv_ds["vv_backscatter"], modis_ds, b),
            out_store
        )
        print(f"Writing vh ratio for {b}")
        write_var(
            f"vh_over_{i}",
            calc_ratio(vh_ds["vh_backscatter"], modis_ds, b),
            out_store
        )

if __name__ == "__main__":
    main()



#import os
#import xarray as xr
#import sys
#from dask.diagnostics import ProgressBar
#import pandas as pd
#
#def calc_ratio(sar_ds,sar_name,modis_ds,modis_name,start_date,end_date):
#    # trim modis data by dates when we have the specific sar data
#    sar_ds = sar_ds.sel(time=slice(start_date, end_date))
#    modis_ds = modis_ds.sel(time=slice(start_date, end_date))
#    this_sar = sar_ds[sar_name]
#    this_modis = modis_ds['data'].sel(variable=modis_name)
#    # ensure consitent chunking
#    #this_sar = this_sar.chunk({'time':1,'x':512,'y':512})
#    #this_modis = this_modis.chunk({'time':1,'x':512,'y':512})
#    ratio = (
#        this_sar /
#        this_modis
#    )
#    ratio = ratio.squeeze()
#    to_drop = [v for v in ["spatial_ref", "variable"] if v in ratio.coords]
#    ratio = ratio.drop_vars(to_drop)
#    # ensure chunked correctly
#    #ratio = ratio.chunk({'time':1,'x':512,'y':512})
#    return ratio
#
#def main():
#    oak_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets'
#    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets'
#    sar_start = pd.Timestamp(2016,4,24)
#    sar_end = pd.Timestamp(2023,10,3)
#    chunks = {'time': 1, 'x': 512, 'y': 512}
#    vv_ds = xr.open_zarr(
#        os.path.join(
#            scratch_dir,
#            'sar',
#            'sar_500m_filled_vv.zarr'
#        ),
#        chunks=chunks
#    )
#    vh_ds = xr.open_zarr(
#        os.path.join(
#            scratch_dir,
#            'sar',
#            'sar_500m_filled.zarr'
#        ),
#        chunks=chunks
#    )
#    modis_ds = xr.open_zarr(
#        os.path.join(
#            oak_dir,
#            'modis',
#            'modis_regridded_gapfilled',
#            'quality_1',
#            'interpolated',
#            'modis_all_vars.zarr'
#        ),
#        chunks=chunks
#    )
#    print('datasets:')
#    print(vv_ds)
#    print(vh_ds)
#    print(modis_ds)
#    vv_minus_vh = vv_ds['vv_backscatter'] - vh_ds['vh_backscatter']
#    #print('vv_minus_vh')
#    #print(vv_minus_vh)
#    print('calc 1 ratios')
#    vv_over_1 = calc_ratio(vv_ds, 'vv_backscatter', modis_ds, 'Nadir_Reflectance_Band1_filled', sar_start, sar_end)
#    vh_over_1 = calc_ratio(vh_ds, 'vh_backscatter', modis_ds, 'Nadir_Reflectance_Band1_filled', sar_start, sar_end)
#    print('vv_over_1')
#    print(vv_over_1)
#    print('calc 2 ratios')
#    vv_over_2 = calc_ratio(vv_ds, 'vv_backscatter', modis_ds, 'Nadir_Reflectance_Band2_filled', sar_start, sar_end)
#    vh_over_2 = calc_ratio(vh_ds, 'vh_backscatter', modis_ds, 'Nadir_Reflectance_Band2_filled', sar_start, sar_end)
#    print('calc 3 ratios')
#    vv_over_3 = calc_ratio(vv_ds, 'vv_backscatter', modis_ds, 'Nadir_Reflectance_Band3_filled', sar_start, sar_end)
#    vh_over_3 = calc_ratio(vh_ds, 'vh_backscatter', modis_ds, 'Nadir_Reflectance_Band3_filled', sar_start, sar_end)
#    print('calc 4 ratios')
#    vv_over_4 = calc_ratio(vv_ds, 'vv_backscatter', modis_ds, 'Nadir_Reflectance_Band4_filled', sar_start, sar_end)
#    vh_over_4 = calc_ratio(vh_ds, 'vh_backscatter', modis_ds, 'Nadir_Reflectance_Band4_filled', sar_start, sar_end)
#    print('calc 5 ratios')
#    vv_over_5 = calc_ratio(vv_ds, 'vv_backscatter', modis_ds, 'Nadir_Reflectance_Band5_filled', sar_start, sar_end)
#    vh_over_5 = calc_ratio(vh_ds, 'vh_backscatter', modis_ds, 'Nadir_Reflectance_Band5_filled', sar_start, sar_end)
#    print('calc 6 ratios')
#    vv_over_6 = calc_ratio(vv_ds, 'vv_backscatter', modis_ds, 'Nadir_Reflectance_Band6_filled', sar_start, sar_end)
#    vh_over_6 = calc_ratio(vh_ds, 'vh_backscatter', modis_ds, 'Nadir_Reflectance_Band6_filled', sar_start, sar_end)
#    print('calc 7 ratios')
#    vv_over_7 = calc_ratio(vv_ds, 'vv_backscatter', modis_ds, 'Nadir_Reflectance_Band7_filled', sar_start, sar_end)
#    vh_over_7 = calc_ratio(vh_ds, 'vh_backscatter', modis_ds, 'Nadir_Reflectance_Band7_filled', sar_start, sar_end)
#    ratio_ds = vv_ds.copy()
#    ratio_ds = ratio_ds.sel(time=slice(sar_start, sar_end))
#    print('adding minus')
#    ratio_ds['vv_minus_vh'] = vv_minus_vh.sel(time=slice(sar_start, sar_end))
#    print('adding 1')
#    ratio_ds['vv_over_1'] = vv_over_1
#    ratio_ds['vh_over_1'] = vh_over_1
#    print('adding 2')
#    ratio_ds['vv_over_2'] = vv_over_2
#    ratio_ds['vh_over_2'] = vh_over_2
#    print('adding 3')
#    ratio_ds['vv_over_3'] = vv_over_3
#    ratio_ds['vh_over_3'] = vh_over_3
#    print('adding 4')
#    ratio_ds['vv_over_4'] = vv_over_4
#    ratio_ds['vh_over_4'] = vh_over_4
#    print('adding 5')
#    ratio_ds['vv_over_5'] = vv_over_5
#    ratio_ds['vh_over_5'] = vh_over_5
#    print('adding 6')
#    ratio_ds['vv_over_6'] = vv_over_6
#    ratio_ds['vh_over_6'] = vh_over_6
#    print('adding 7')
#    ratio_ds['vv_over_7'] = vv_over_7
#    ratio_ds['vh_over_7'] = vh_over_7
#    # get rid of vv_backscatter
#    print('dropping')
#    ratio_ds = ratio_ds.drop_vars(['vv_backscatter'])
#    print(ratio_ds)
#    with ProgressBar():
#        ratio_ds.to_zarr(
#            os.path.join(
#                scratch_dir,
#                'sar',
#                'sar_500m_filled_ratios.zarr'
#            )
#        )
#
#
#if __name__ == '__main__':
#    main()