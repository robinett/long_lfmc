import os
import xarray as xr
import sys
import pandas as pd
import json
from tqdm import tqdm
from pyproj import Transformer
import numpy as np
from dask.diagnostics import ProgressBar
import torch
import re
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, '..', '..','..')
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

from transformer_multitask_longclimate import LFMCTransformer
from transformer_multitask_longclimate_uncertainty import LFMCTransformer as LFMCTransformerUncertainty
from plotting import plot_timeseries_by_site

def parse_model_path(model_path: str) -> dict:
    name = Path(model_path).parts[-3]  # transformer_..._basic
    patterns = {
        "d_model": r"dm(\d+)",
        "nhead": r"nh(\d+)",
        "num_layers": r"nl(\d+)",
        "dim_feedforward": r"df(\d+)",
        "dropout": r"do([\d.]+)",
        "long_d_model": r"dmlong(\d+)",
        "long_nhead": r"nhlong(\d+)",
        "long_num_layers": r"nllong(\d+)",
        "long_dim_feedforward": r"dflong(\d+)",
        "long_out_dim": r"outlong(\d+)",
    }
    params = {}
    for key, pat in patterns.items():
        m = re.search(pat, name)
        if m:
            val = m.group(1)
            params[key] = float(val) if "." in val else int(val)
    return params

def build_tensors(
    locs,
    start_dates,
    end_dates,
    var_names,
    var_locs,
    dss,
    short_lag_days,
    long_lag_days,
    norm_params,
):
    # long input array: shape [B, Ts, Din_long]
    # short input array: shape [B, Ts, Din_short]
    # static features array: shape [B, 1, Din_static]
    # we're going to need to invert to get the correct dataset depending on our var
    var_to_ds = {
        var: ds_key
        for ds_key, vars_ in var_locs.items()
        for var in vars_
    }
    short_vars_needed = var_names['short_vars']
    long_vars_needed = var_names['long_vars']
    static_vars_needed = var_names['static_vars']
    total_preds = 0
    for l,loc in enumerate(locs):
        this_start = start_dates[l]
        this_end = end_dates[l]
        num_days = (this_end - this_start).days + 1
        total_preds += num_days
    # initialize our tensors
    short_input = np.zeros((total_preds, len(short_lag_days), len(short_vars_needed)))
    long_input = np.zeros((total_preds, len(long_lag_days), len(long_vars_needed)))
    static_input = np.zeros((total_preds, 1, len(static_vars_needed)))
    short_input[:, :, :] = np.nan
    long_input[:, :, :] = np.nan
    static_input[:, :, :] = np.nan
    short_lag_needed = short_lag_days[-1]
    long_lag_needed = long_lag_days[-1]
    max_lag_needed = max(short_lag_needed, long_lag_needed)
    trns = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    starting_B = 0
    #for l,loc in enumerate(locs):
    for l, loc in tqdm(
        enumerate(locs),
        total=len(locs),
        desc="Creating tensor for each location"
    ):
        this_x, this_y = trns.transform(loc[0],loc[1])
        this_start = start_dates[l]
        this_end = end_dates[l]
        date_range = pd.date_range(this_start, this_end)
        # create the lists that we will turn into our info df
        lats_to_add = np.repeat(loc[1], len(date_range))
        lons_to_add = np.repeat(loc[0], len(date_range))
        times_to_add = date_range
        if l == 0:
            info_df = pd.DataFrame({
                'lat': lats_to_add,
                'lon': lons_to_add,
                'date': times_to_add
            })
        else:
            info_df = pd.concat([info_df, pd.DataFrame({
                'lat': lats_to_add,
                'lon': lons_to_add,
                'date': times_to_add
            })])
        if l > 0:
            starting_B += (end_dates[l-1] - start_dates[l-1]).days + 1
        modis_data = dss['modis'].sel(
            x=this_x,
            y=this_y,
            method='nearest'
        )
        modis_data = modis_data.sel(
            time=slice(this_start - pd.Timedelta(days=short_lag_needed), this_end),
        ).compute()
        daymet_data = dss['daymet'].sel(
            x=this_x,
            y=this_y,
            method='nearest'
        )
        daymet_data = (
            daymet_data
            .sel(time=slice(this_start - pd.Timedelta(days=long_lag_needed), this_end + pd.Timedelta(days=1)))
            .compute()
        )
        # set daymet to midnight
        #daymet_data = daymet_data.resample(time='1D').mean()
        daymet_data['time'] = daymet_data['time'].dt.floor('D')
        # for daymet we need to copy Dec 30 to be Dec 31 on leap years because they
        # refuse to give us leap years 
        ds = daymet_data
        years = np.unique(ds.time.dt.year.values)
        new_slices = []
        for y in years:
            if (
                (pd.Timestamp(f"{y}-01-01").is_leap_year) and
                this_end >= pd.Timestamp(f"{y}-12-31")
            ):
                dec30 = pd.Timestamp(f"{y}-12-30")
                dec31 = pd.Timestamp(f"{y}-12-31")
                if dec31 not in ds.time.values:
                    s31 = ds.sel(time=dec30).copy(deep=True)
                    s31 = s31.assign_coords(time=dec31)
                    new_slices.append(s31)
        if new_slices:
            ds = xr.concat([ds] + new_slices, dim="time").sortby("time")
        daymet_data = ds
        for s,s_var in enumerate(short_vars_needed):
            this_norm_mean = norm_params['train_short_mean'][s]
            this_norm_std = norm_params['train_short_std'][s]
            if s_var == 'lfrac':
                this_vals = np.arange(short_lag_needed+1)
                this_vals = this_vals / short_lag_needed
                #this_vals = this_vals[::-1]
                this_vals = (this_vals - this_norm_mean) / this_norm_std
                for d,date in enumerate(date_range):
                    short_input[starting_B + d, :, s] = this_vals
                continue
            this_ds_name = var_to_ds[s_var]
            if this_ds_name == 'modis':
                this_vals = modis_data['data'].sel(variable=s_var)
                # we now need to fill in for all days and lags at this site
                for d,date in enumerate(date_range):
                    vals_to_add = this_vals.sel(
                        time=slice(date - pd.Timedelta(days=len(short_lag_days)-1), date),
                    )
                    # reverse vals to add
                    vals_to_add = vals_to_add[::-1]
                    vals_to_add_norm = (vals_to_add - this_norm_mean) / this_norm_std
                    #print(np.where(vals_to_add_norm.isnull()))
                    #sys.exit()
                    # normalize
                    short_input[starting_B + d, :, s] = vals_to_add_norm
            else:
                raise NotImplementedError(
                    f'Dataset {this_ds_name} not implemented'
                )
        # add the long vars
        for l,l_var in enumerate(long_vars_needed):
            this_norm_mean = norm_params['train_long_mean'][l]
            this_norm_std = norm_params['train_long_std'][l]
            if l_var == 'lfrac':
                this_vals = np.arange(long_lag_needed+1)
                this_vals = this_vals / long_lag_needed
                #this_vals = this_vals[::-1]
                this_vals = (this_vals - this_norm_mean) / this_norm_std
                for d,date in enumerate(date_range):
                    long_input[starting_B + d, :, l] = this_vals
                continue
            this_ds_name = var_to_ds[l_var]
            if this_ds_name == 'daymet':
                this_vals = daymet_data['data'].sel(variable=l_var)
                # we now need to fill in for all days and lags at this site
                for d,date in enumerate(date_range):
                    # because daymet is super annoying, if it is the 31st of a leap year,
                    # we need to pretend that the day is yesterday
                    this_date_start = date - pd.Timedelta(days=(len(long_lag_days)-1))
                    this_date_end = date
                    vals_to_add = this_vals.sel(
                        time=slice(this_date_start, this_date_end)
                    )
                    # reverse vals to add
                    vals_to_add = vals_to_add[::-1]
                    # normalize
                    vals_to_add_norm = (vals_to_add - this_norm_mean) / this_norm_std
                    long_input[starting_B + d, :, l] = vals_to_add_norm
            else:
                raise NotImplementedError(
                    f'Dataset {this_ds_name} not implemented'
                )
        for st,st_var in enumerate(static_vars_needed):
            #print(st_var)
            this_norm_mean = norm_params['train_static_mean'][st]
            this_norm_std = norm_params['train_static_std'][st]
            if st_var == 'latitude':
                this_vals = loc[1]  # latitude
                vals_to_add= (this_vals - this_norm_mean) / this_norm_std
            elif st_var == 'longitude':
                this_vals = loc[0] # longitude
                vals_to_add = (this_vals - this_norm_mean) / this_norm_std
            elif 'climate_zone' in st_var:
                this_ds = dss['climate_zone']
                this_vals = this_ds['climate_zone'].sel(
                    x=this_x,
                    y=this_y,
                    method='nearest'
                )
                climate_zone_here = int(this_vals[0].values)
                climate_zone_checking = int(
                    st_var.split('_')[-1]
                )
                if climate_zone_here == climate_zone_checking:
                    vals_to_add = 1
                else:
                    vals_to_add = 0
            elif (
                'barren' in st_var or
                'crops' in st_var or
                'forest' in st_var or
                'developed' in st_var or
                'grass' in st_var or
                'other' in st_var or
                'shrub' in st_var or
                'water' in st_var or
                'wetlands' in st_var
            ):
                this_ds = dss['landcover_frac']
                this_vals = this_ds[st_var].sel(
                    x=this_x,
                    y=this_y,
                    method='nearest'
                ).compute()
            else:
                this_ds_name = var_to_ds[st_var]
                this_ds = dss[this_ds_name]
                this_vals = this_ds[st_var].sel(
                    x=this_x,
                    y=this_y,
                    method='nearest'
                )
                vals_to_add = (this_vals - this_norm_mean) / this_norm_std
            for d,date in enumerate(date_range):
                if (
                    'barren' in st_var or
                    'crops' in st_var or
                    'forest' in st_var or
                    'developed' in st_var or
                    'grass' in st_var or
                    'other' in st_var or
                    'shrub' in st_var or
                    'water' in st_var or
                    'wetlands' in st_var
                ):
                    vals_to_add = this_vals.sel(
                        year=pd.Timestamp(date.year, 1, 1),
                        #method='nearest'
                    ).values
                static_input[starting_B + d, :, st] = vals_to_add
    # convert to tensors and make dataloaders
    long_tensor = torch.tensor(long_input)
    short_tensor = torch.tensor(short_input)
    static_tensor = torch.tensor(static_input)
    return [short_tensor, long_tensor, static_tensor, info_df]

def run_model_forward(
    short_tensor,
    long_tensor,
    static_tensor,
    info_df,
    model_path,
    norm_params,
    batch_size=512,
    model_num_queries=2,
    model_task_weights=2,
    model_type = 'standard'
):
    dataset = TensorDataset(
        short_tensor,
        long_tensor,
        static_tensor
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    # get the model and restore the saved weights
    short_input_dim = short_tensor.shape[-1]
    long_input_dim = long_tensor.shape[-1]
    static_input_dim = static_tensor.shape[-1]
    model_params = parse_model_path(model_path)
    if model_type == 'standard':
        model = LFMCTransformer(
            short_input_dim = short_input_dim,
            long_input_dim = long_input_dim,
            static_input_dim = static_input_dim,
            d_model = model_params['d_model'],
            nhead = model_params['nhead'],
            num_layers = model_params['num_layers'],
            dim_feedforward = model_params['dim_feedforward'],
            dropout = model_params['dropout'],
            num_queries = model_num_queries,
            long_d_model = model_params['long_d_model'],
            long_nhead = model_params['long_nhead'],
            long_num_layers = model_params['long_num_layers'],
            long_dim_feedforward = model_params['long_dim_feedforward'],
            long_out_dim = model_params['long_out_dim'],
            num_task_weights = model_task_weights
        )
    elif model_type == 'uncertainty':
        model = LFMCTransformerUncertainty(
            short_input_dim = short_input_dim,
            long_input_dim = long_input_dim,
            static_input_dim = static_input_dim,
            d_model = model_params['d_model'],
            nhead = model_params['nhead'],
            num_layers = model_params['num_layers'],
            dim_feedforward = model_params['dim_feedforward'],
            dropout = model_params['dropout'],
            num_queries = model_num_queries,
            long_d_model = model_params['long_d_model'],
            long_nhead = model_params['long_nhead'],
            long_num_layers = model_params['long_num_layers'],
            long_dim_feedforward = model_params['long_dim_feedforward'],
            long_out_dim = model_params['long_out_dim'],
            num_task_weights = model_task_weights
        )
    else:
        raise notImplementedError(
            f"Model choice '{model_choice}' is not implemented"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    ckpt = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt,
        strict=False
    )
    #sys.exit()
    model.eval()
    # run the model
    pbar = tqdm(
        dataloader,
        desc='Running model batches'
    )
    preds_i = np.zeros(short_tensor.shape[0])
    batch_counter = 0
    with torch.no_grad():
        for Xsh_b, Xl_b, Xst_b in pbar:
            Xsh_b = Xsh_b.to(device=device, dtype=torch.float32)
            Xl_b = Xl_b.to(device=device, dtype=torch.float32)
            Xst_b = Xst_b.to(device=device, dtype=torch.float32)
            output = model(Xsh_b, Xl_b, Xst_b)
            preds_i_b = output['mu_insitu']
            start_idx = batch_counter * batch_size
            end_idx = start_idx + Xsh_b.shape[0]
            preds_i[start_idx:end_idx] = preds_i_b.cpu().numpy()
            batch_counter += 1
    # renormalize
    preds_i = preds_i * norm_params['lfmc_std'] + norm_params['lfmc_mean']
    info_df['lfmc_pred'] = preds_i
    #print(info_df)
    return info_df



def main():
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/'
    oak_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/'
    # where to save the outputs of the model
    out_dir = os.path.join(oak_dir,'lfmc_model','data','infer','for_mitch_20260124')
    plots_dir = os.path.join(scratch_dir,'inference','plots')
    os.makedirs(out_dir, exist_ok=True)
    # information about the sites to run
    site_info = pd.read_csv(
        os.path.join(
            scratch_dir,
            'inference/site_info',
            'valid_locations_wus.csv'
        )
    )
    # just take the first location for now for testing
    #site_info = site_info.iloc[:1]
    print('info about where we will be running the model:')
    print(site_info)
    locs = site_info[['lon', 'lat']].values.tolist()
    start_dates = pd.to_datetime(site_info['start_date'])
    end_dates = pd.to_datetime(site_info['end_date'])
    # information that we load from the model
    model_name = 'news1_multitask_5_1'
    var_names_path = os.path.join(
        scratch_dir,'inputs','news1_multitask','var_names.json'
    )
    with open(var_names_path) as f:
        var_names = json.load(f)
    model_dir = os.path.join(
        scratch_dir,
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
    short_tensor, long_tensor, static_tensor, info_df = build_tensors(
        locs,
        start_dates,
        end_dates,
        var_names,
        var_locs,
        dss,
        short_lag_days,
        long_lag_days,
        norm_params,
    )
    preds_df = run_model_forward(
        short_tensor,
        long_tensor,
        static_tensor,
        info_df,
        model_path,
        norm_params
    )
    # save the output
    print('Saving output...')
    preds_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    # plot each site
    #preds_df = pd.read_csv(
    #    os.path.join(out_dir, 'predictions.csv')
    #)
    print('Plotting by site...')
    plot_timeseries_by_site(preds_df, plots_dir, 'lfmc_pred', "LFMC Prediction (%)")


if __name__ == "__main__":
    main()