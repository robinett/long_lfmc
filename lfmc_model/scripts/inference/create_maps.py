import os
import sys
import json
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from dask.cache import Cache
import dask

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, '..', '..','..')
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

from point_tool_new import build_tensors, run_model_forward
from plotting import plot_timeseries_by_site

cache = Cache(64e9)
cache.register()

def main():
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/'
    oak_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/'
    plots_dir = os.path.join(scratch_dir,'lfmc_model','data','inference','plots')
    # where to save the outputs of the model
    # information about the sites to run
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-12-31')
    model_grid = xr.open_dataset(
        os.path.join(oak_dir, 'grid','epsg5070_500m_westUS_grid.nc4')
    )
    print(model_grid)
    vals = model_grid['random_vals']
    lats = model_grid['lat'].values
    lons = model_grid['lon'].values
    mask = vals.notnull()
    iy,ix = np.where(mask.data)
    tile_iy = iy // 10
    tile_ix = ix // 10
    idx_df = pd.DataFrame({
        'ix': ix,
        'iy': iy,
        'tile_ix': tile_ix,
        'tile_iy': tile_iy
    })
    unique_tiles = idx_df[['tile_ix', 'tile_iy']].drop_duplicates()
    tile_lons_lats = {}
    grouped_idx_df = idx_df.groupby(['tile_ix', 'tile_iy'],sort=False)
    #for (tx,ty), group in tqdm(grouped_idx_df, total=grouped_idx_df.ngroups, desc='Creating tile chunks'):
    #    iy = group['iy'].to_numpy()
    #    ix = group['ix'].to_numpy()
    #    tile_lons_lats[f"{tx}_{ty}"] = (lons[iy, ix], lats[iy, ix])
    for (tx, ty), group in tqdm(
        grouped_idx_df,
        total=grouped_idx_df.ngroups,
        desc="Creating tile chunks",
):
        iy = group["iy"].to_numpy()
        ix = group["ix"].to_numpy()
        this_lons = lons[iy, ix]
        this_lats = lats[iy, ix]
        # shape: (N, 2) → [[lon, lat], [lon, lat], ...]
        tile_lons_lats[f"{tx}_{ty}"] = np.column_stack(
            (this_lons, this_lats)
        )
    # information that we load from the model
    model_name = 'news1_multitask_5_1'
    var_names_path = os.path.join(
        scratch_dir,'lfmc_model','data','inputs','news1_multitask','var_names.json'
    )
    with open(var_names_path) as f:
        var_names = json.load(f)
    model_dir = os.path.join(
        scratch_dir,
        'lfmc_model','data',
        'outputs', model_name,
        'transformer_dm32_nh1_nl2_df64_do0.15_bs128_lr0.0005_warmup2458_wd0.0001_iobs30638_vvobs0_vhobs119237_dmlong64_nhlong2_nllong3_dflong128_outlong32_basic',
        'fold_9998'
    )
    norm_params = os.path.join(
        model_dir,'norm_params.json'
    )
    model_path = os.path.join(
        model_dir,
        'model_epoch4.pt'
    )
    with open(norm_params) as f:
        norm_params = json.load(f)
    # lets lay out where the varaibles are that we are going to need to find
    # location of possible long input variables
    var_locs = {
        'daymet':[
            'prcp','srad','swe','tmax','vp'
        ],
        'modis':[
            'Nadir_Reflectance_Band1_filled',
            'Nadir_Reflectance_Band2_filled',
            'Nadir_Reflectance_Band3_filled',
            'Nadir_Reflectance_Band4_filled',
            'Nadir_Reflectance_Band5_filled',
            'Nadir_Reflectance_Band6_filled',
            'Nadir_Reflectance_Band7_filled'
        ],
        'static':[
            'slope',
            'elevation',
            'canopy_height',
            'clay',
            'sand'
        ],
        'climate_zone':[
            'climate_zone_1','climate_zone_2','climate_zone_3',
            'climate_zone_4','climate_zone_5','climate_zone_6',
            'climate_zone_7','climate_zone_8','climate_zone_9',
            'climate_zone_10','climate_zone_11','climate_zone_12',
            'climate_zone_13','climate_zone_14','climate_zone_15',
            'climate_zone_16','climate_zone_17','climate_zone_18',
            'climate_zone_19','climate_zone_20','climate_zone_21',
            'climate_zone_22','climate_zone_23','climate_zone_24',
            'climate_zone_25','climate_zone_26','climate_zone_27',
            'climate_zone_28','climate_zone_29',
        ],
        'landcover_frac':[
            'barren',
            'crops',
            'deciduous_forest',
            'developed',
            'evergreen_forest',
            'grass',
            'mixed_forest',
            'other',
            'shrub',
            'water',
            'wetlands'
        ]
    }
    print('opening datasets...')
    dss = {
        'daymet': xr.open_zarr(
            os.path.join(oak_dir, 'daymet/daymet_all_vars.zarr'),
            consolidated=False
        ),
        'modis': xr.open_zarr(
            os.path.join(
                oak_dir,
                'modis/modis_regridded_gapfilled/quality_1/interpolated/modis_all_vars.zarr'
            )
        ),
        'static': xr.open_dataset(
            os.path.join(oak_dir, 'static', 'static_features_500m_epsg5070_float32.nc')
        ),
        'climate_zone': xr.open_dataset(
            os.path.join(oak_dir, 'climate_zones', 'climate_zone_per_pixel_westUS.nc4')
        ),
        'landcover_frac': xr.open_zarr(
            os.path.join(oak_dir, 'nlcd', 'nlcd_target_grid_2003_2023.zarr')
        ),
    }
    #daymet = dss['daymet']
    #blk_1 = daymet.isel(x=0, y=0, time=0)
    #blk_2 = daymet.isel(x=1, y=1, time=0)
    #print(pd.Timestamp.now())
    #blk_1.load()
    #print(pd.Timestamp.now())
    #blk_1.load()
    #print(pd.Timestamp.now())
    #blk_2.load()
    #print(pd.Timestamp.now())
    #blk_2.load()
    #print(pd.Timestamp.now())
    #sys.exit()
    short_lag_days = [
        0,1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30
    ]
    long_lag_days = [
        0,1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,
        51,52,53,54,55,56,57,58,59,60,
        61,62,63,64,65,66,67,68,69,70,
        71,72,73,74,75,76,77,78,79,80,
        81,82,83,84,85,86,87,88,89,90,
        91,92,93,94,95,96,97,98,99,100,
        101,102,103,104,105,106,107,108,109,110,
        111,112,113,114,115,116,117,118,119,120,
        121,122,123,124,125,126,127,128,129,130,
        131,132,133,134,135,136,137,138,139,140,
        141,142,143,144,145,146,147,148,149,150,
        151,152,153,154,155,156,157,158,159,160,
        161,162,163,164,165,166,167,168,169,170,
        171,172,173,174,175,176,177,178,179,180,
    ]
    # locations of possible static input variables
    for t,(tile_name,tile_locs) in tqdm(enumerate(tile_lons_lats.items()), total=len(tile_lons_lats), desc="Running model at tiles"):
        start_dates = [start_date for _ in range(tile_locs.shape[0])]
        end_dates = [end_date for _ in range(tile_locs.shape[0])]
        short_tensor, long_tensor, static_tensor, info_df = build_tensors(
            tile_locs,
            start_dates,
            end_dates,
            var_names,
            var_locs,
            dss,
            short_lag_days,
            long_lag_days,
            norm_params,
            all_nearby=True,
        )
        preds_df = run_model_forward(
            short_tensor,
            long_tensor,
            static_tensor,
            info_df,
            model_path,
            norm_params
        )
        plot_timeseries_by_site(preds_df, plots_dir, 'lfmc_pred', "LFMC Prediction (%)")
        print(preds_df)
        sys.exit()

if __name__ == "__main__":
    main()
