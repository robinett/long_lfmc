import os
import xarray as xr
import sys
import pandas as pd
import json
import warnings
from tqdm import tqdm
from pyproj import Transformer
import numpy as np
from dask.diagnostics import ProgressBar
import torch
import re
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from dask.cache import Cache

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, '..', '..','..')
sys.path.append(os.path.join(project_root,'lfmc_model','models','transformer'))
sys.path.append(os.path.join(project_root,'lfmc_model','models','multisource_fusion'))
sys.path.append(os.path.join(project_root,'lfmc_model','utils'))

from transformer_multitask_longclimate import LFMCTransformer
from transformer_multitask_longclimate_uncertainty import LFMCTransformer as LFMCTransformerUncertainty
from multisource_fusion_model import LFMCMultiSourceFusion
from plotting import plot_timeseries_by_site

cache = Cache(64e9)
cache.register()

def parse_model_path(model_path: str) -> dict:
    name = Path(model_path).parts[-3]  # transformer_..._basic
    model_family = 'multisource_fusion' if name.startswith('multisource_fusion_') else 'transformer'
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
        "weather_kernel_size": r"kw(\d+)",
        "weather_max_dilation": r"wdil(\d+)",
        "weather_d_model": r"_dw(\d+)",
        "modis_d_model": r"_dm(\d+)",
        "static_d_model": r"_ds(\d+)",
        "common_d_model": r"_dc(\d+)",
        "shared_latent_dim": r"_sh(\d+)",
        "lfmc_private_dim": r"_lfp(\d+)",
        "sar_private_dim": r"_sarp(\d+)",
    }
    params = {"model_family": model_family}
    for key, pat in patterns.items():
        m = re.search(pat, name)
        if m:
            val = m.group(1)
            params[key] = float(val) if "." in val else int(val)
    return params

def _nearest_index(coords, val):
    coords = np.asarray(coords)
    if coords.ndim != 1:
        raise ValueError("Expected 1D coordinate array")
    if coords.size == 0:
        raise ValueError("Empty coordinate array")
    if coords.size == 1:
        return 0
    if coords[0] <= coords[-1]:
        idx = int(np.searchsorted(coords, val))
        if idx <= 0:
            return 0
        if idx >= coords.size:
            return coords.size - 1
        left = coords[idx - 1]
        right = coords[idx]
        return idx - 1 if abs(val - left) <= abs(right - val) else idx
    coords_rev = coords[::-1]
    idx_rev = int(np.searchsorted(coords_rev, val))
    if idx_rev <= 0:
        return coords.size - 1
    if idx_rev >= coords_rev.size:
        return 0
    left = coords_rev[idx_rev - 1]
    right = coords_rev[idx_rev]
    nearest_rev = idx_rev - 1 if abs(val - left) <= abs(right - val) else idx_rev
    return coords.size - 1 - nearest_rev

def _get_chunk_size(ds, dim, fallback=64):
    if hasattr(ds, "chunksizes") and ds.chunksizes and dim in ds.chunksizes:
        return int(ds.chunksizes[dim][0])
    if hasattr(ds, "chunks") and ds.chunks:
        dim_to_axis = {d: i for i, d in enumerate(ds.dims)}
        if dim in dim_to_axis:
            axis = dim_to_axis[dim]
            return int(ds.chunks[axis][0])
    return int(fallback)


def _merge_daymet_datasets(daymet_ds, anomaly_ds):
    if anomaly_ds is None:
        return daymet_ds
    anomaly_ds = anomaly_ds.reindex(time=daymet_ds["time"])
    return xr.concat(
        [daymet_ds, anomaly_ds],
        dim="variable",
        compat="override",
        coords="minimal",
        join="exact",
    )

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
    all_nearby=False
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
    loc_xys = [trns.transform(loc[0], loc[1]) for loc in locs]
    preloaded_chunks = {}
    loc_chunk_keys = None
    preloaded_static = None
    preloaded_soils = None
    preloaded_canopy_height = None
    preloaded_climate_zone = None
    if all_nearby:
        preloaded_static = dss['static'].load()
        if 'soils' in dss:
            preloaded_soils = dss['soils'].load()
        if 'canopy_height' in dss:
            preloaded_canopy_height = dss['canopy_height'].load()
        if 'climate_zone' in dss:
            preloaded_climate_zone = dss['climate_zone'].load()
        modis_ds = dss['modis']
        modis_x = modis_ds['x'].values
        modis_y = modis_ds['y'].values
        if 'data' in modis_ds:
            modis_chunk_ref = modis_ds['data']
        else:
            modis_data_vars = list(modis_ds.data_vars)
            if len(modis_data_vars) == 0:
                raise KeyError("MODIS dataset has no data variables available for chunking")
            modis_chunk_ref = modis_ds[modis_data_vars[0]]
        x_chunk = _get_chunk_size(modis_chunk_ref, 'x', fallback=64)
        y_chunk = _get_chunk_size(modis_chunk_ref, 'y', fallback=64)
        chunk_groups = {}
        loc_chunk_keys = []
        for i, (this_x, this_y) in enumerate(loc_xys):
            x_idx = _nearest_index(modis_x, this_x)
            y_idx = _nearest_index(modis_y, this_y)
            chunk_key = (x_idx // x_chunk, y_idx // y_chunk)
            loc_chunk_keys.append(chunk_key)
            if chunk_key not in chunk_groups:
                chunk_groups[chunk_key] = []
            chunk_groups[chunk_key].append(i)
        for chunk_key, loc_idxs in tqdm(
            chunk_groups.items(),
            total=len(chunk_groups),
            desc="Preloading nearby chunks",
        ):
            x0 = chunk_key[0] * x_chunk
            x1 = min((chunk_key[0] + 1) * x_chunk, modis_x.size)
            y0 = chunk_key[1] * y_chunk
            y1 = min((chunk_key[1] + 1) * y_chunk, modis_y.size)
            chunk_start = min(start_dates[i] for i in loc_idxs)
            chunk_end = max(end_dates[i] for i in loc_idxs)
            preloaded_chunks[chunk_key] = {
                'modis': (
                    dss['modis']
                    .isel(x=slice(x0, x1), y=slice(y0, y1))
                    .sel(
                        time=slice(
                            chunk_start - pd.Timedelta(days=short_lag_needed),
                            chunk_end
                        )
                    )
                    .compute()
                ),
                'daymet': (
                    dss['daymet']
                    .isel(x=slice(x0, x1), y=slice(y0, y1))
                    .sel(
                        time=slice(
                            chunk_start - pd.Timedelta(days=long_lag_needed),
                            chunk_end + pd.Timedelta(days=1)
                        )
                    )
                    .compute()
                ),
                'landcover_frac': (
                    dss['landcover_frac']
                    .isel(x=slice(x0, x1), y=slice(y0, y1))
                    .load()
                ),
            }
    starting_B = 0
    #for l,loc in enumerate(locs):
    for l, loc in tqdm(
        enumerate(locs),
        total=len(locs),
        desc="Creating tensor for each location"
    ):
        this_x, this_y = loc_xys[l]
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
        if all_nearby and loc_chunk_keys is not None:
            chunk_ds = preloaded_chunks[loc_chunk_keys[l]]
            modis_data = chunk_ds['modis'].sel(
                x=this_x,
                y=this_y,
                method='nearest'
            )
            daymet_data = chunk_ds['daymet'].sel(
                x=this_x,
                y=this_y,
                method='nearest'
            )
        else:
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
                if 'data' in modis_data:
                    this_vals = modis_data['data'].sel(variable=s_var)
                else:
                    if s_var not in modis_data.data_vars:
                        raise KeyError(
                            f"MODIS variable '{s_var}' not found. "
                            f"Available variables: {list(modis_data.data_vars)}"
                        )
                    this_vals = modis_data[s_var]
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
                if all_nearby and loc_chunk_keys is not None:
                    this_ds = preloaded_climate_zone
                else:
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
                if all_nearby and loc_chunk_keys is not None:
                    this_ds = preloaded_chunks[loc_chunk_keys[l]]['landcover_frac']
                else:
                    this_ds = dss['landcover_frac']
                this_vals = this_ds[st_var].sel(
                    x=this_x,
                    y=this_y,
                    method='nearest'
                )
                if not (all_nearby and loc_chunk_keys is not None):
                    this_vals = this_vals.compute()
            else:
                this_ds_name = var_to_ds[st_var]
                if all_nearby and loc_chunk_keys is not None and this_ds_name == 'static':
                    this_ds = preloaded_static
                elif all_nearby and loc_chunk_keys is not None and this_ds_name == 'soils':
                    this_ds = preloaded_soils
                elif all_nearby and loc_chunk_keys is not None and this_ds_name == 'canopy_height':
                    this_ds = preloaded_canopy_height
                else:
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

def _build_inference_model(
    short_input_dim,
    long_input_dim,
    static_input_dim,
    model_path,
    model_num_queries=2,
    model_task_weights=2,
    model_type='standard',
):
    model_params = parse_model_path(model_path)
    inferred_family = model_params.get('model_family', 'transformer')
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "enable_nested_tensor is True, but self.use_nested_tensor is False "
                "because encoder_layer.self_attn.num_heads is odd"
            ),
            category=UserWarning,
        )
        if inferred_family == 'multisource_fusion':
            model = LFMCMultiSourceFusion(
                short_input_dim=short_input_dim,
                long_input_dim=long_input_dim,
                static_input_dim=static_input_dim,
                weather_d_model=model_params['weather_d_model'],
                modis_d_model=model_params['modis_d_model'],
                static_d_model=model_params['static_d_model'],
                common_d_model=model_params['common_d_model'],
                weather_kernel_size=model_params['weather_kernel_size'],
                weather_max_dilation=model_params['weather_max_dilation'],
                shared_latent_dim=model_params.get('shared_latent_dim', 0),
                lfmc_private_dim=model_params.get('lfmc_private_dim', 0),
                sar_private_dim=model_params.get('sar_private_dim', 0),
                dropout=model_params['dropout'],
                num_task_weights=model_task_weights,
            )
        elif model_type == 'standard':
            model = LFMCTransformer(
                short_input_dim=short_input_dim,
                long_input_dim=long_input_dim,
                static_input_dim=static_input_dim,
                d_model=model_params['d_model'],
                nhead=model_params['nhead'],
                num_layers=model_params['num_layers'],
                dim_feedforward=model_params['dim_feedforward'],
                dropout=model_params['dropout'],
                num_queries=model_num_queries,
                long_d_model=model_params['long_d_model'],
                long_nhead=model_params['long_nhead'],
                long_num_layers=model_params['long_num_layers'],
                long_dim_feedforward=model_params['long_dim_feedforward'],
                long_out_dim=model_params['long_out_dim'],
                num_task_weights=model_task_weights,
            )
        elif model_type == 'uncertainty':
            model = LFMCTransformerUncertainty(
                short_input_dim=short_input_dim,
                long_input_dim=long_input_dim,
                static_input_dim=static_input_dim,
                d_model=model_params['d_model'],
                nhead=model_params['nhead'],
                num_layers=model_params['num_layers'],
                dim_feedforward=model_params['dim_feedforward'],
                dropout=model_params['dropout'],
                num_queries=model_num_queries,
                long_d_model=model_params['long_d_model'],
                long_nhead=model_params['long_nhead'],
                long_num_layers=model_params['long_num_layers'],
                long_dim_feedforward=model_params['long_dim_feedforward'],
                long_out_dim=model_params['long_out_dim'],
                num_task_weights=model_task_weights,
            )
        else:
            raise NotImplementedError(
                f"Model type '{model_type}' is not implemented"
            )
    return model


def load_model_for_inference(
    short_input_dim,
    long_input_dim,
    static_input_dim,
    model_path,
    model_num_queries=2,
    model_task_weights=2,
    model_type='standard',
    device=None,
):
    model = _build_inference_model(
        short_input_dim=short_input_dim,
        long_input_dim=long_input_dim,
        static_input_dim=static_input_dim,
        model_path=model_path,
        model_num_queries=model_num_queries,
        model_task_weights=model_task_weights,
        model_type=model_type,
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(
        ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt,
        strict=False,
    )
    model.eval()
    return model, device


def predict_with_loaded_model(
    short_tensor,
    long_tensor,
    static_tensor,
    model,
    device,
    norm_params,
    batch_size=512,
    use_cuda_autocast=True,
):
    dataset = TensorDataset(short_tensor, long_tensor, static_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_obs = short_tensor.shape[0]
    preds_i = np.zeros(n_obs, dtype=np.float64)
    preds_vv = np.full(n_obs, np.nan, dtype=np.float64)
    preds_vh = np.full(n_obs, np.nan, dtype=np.float64)
    preds_i_std = np.full(n_obs, np.nan, dtype=np.float64)
    preds_vv_std = np.full(n_obs, np.nan, dtype=np.float64)
    preds_vh_std = np.full(n_obs, np.nan, dtype=np.float64)
    batch_counter = 0
    use_cuda_autocast = bool(use_cuda_autocast) and device.type == "cuda"
    autocast_dtype = torch.float16
    with torch.inference_mode():
        for Xsh_b, Xl_b, Xst_b in dataloader:
            Xsh_b = Xsh_b.to(device=device, dtype=torch.float32)
            Xl_b = Xl_b.to(device=device, dtype=torch.float32)
            Xst_b = Xst_b.to(device=device, dtype=torch.float32)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_cuda_autocast,
            ):
                output = model(Xsh_b, Xl_b, Xst_b)
            preds_i_b = np.asarray(output['mu_insitu'].detach().cpu().numpy()).reshape(-1)
            preds_vv_b = np.asarray(output['mu_vv'].detach().cpu().numpy()).reshape(-1)
            preds_vh_b = np.asarray(output['mu_vh'].detach().cpu().numpy()).reshape(-1)
            if 'log_var_insitu' in output:
                preds_i_std_b = np.sqrt(
                    np.exp(np.asarray(output['log_var_insitu'].detach().cpu().numpy()).reshape(-1))
                )
            else:
                preds_i_std_b = np.full_like(preds_i_b, np.nan)
            if 'log_var_vv' in output:
                preds_vv_std_b = np.sqrt(
                    np.exp(np.asarray(output['log_var_vv'].detach().cpu().numpy()).reshape(-1))
                )
            else:
                preds_vv_std_b = np.full_like(preds_vv_b, np.nan)
            if 'log_var_vh' in output:
                preds_vh_std_b = np.sqrt(
                    np.exp(np.asarray(output['log_var_vh'].detach().cpu().numpy()).reshape(-1))
                )
            else:
                preds_vh_std_b = np.full_like(preds_vh_b, np.nan)
            start_idx = batch_counter * batch_size
            end_idx = start_idx + Xsh_b.shape[0]
            preds_i[start_idx:end_idx] = preds_i_b
            preds_vv[start_idx:end_idx] = preds_vv_b
            preds_vh[start_idx:end_idx] = preds_vh_b
            preds_i_std[start_idx:end_idx] = preds_i_std_b
            preds_vv_std[start_idx:end_idx] = preds_vv_std_b
            preds_vh_std[start_idx:end_idx] = preds_vh_std_b
            batch_counter += 1
    lfmc_mean = norm_params.get('lfmc_mean', np.nan)
    lfmc_std = norm_params.get('lfmc_std', np.nan)
    vv_mean = norm_params.get('vv_mean', np.nan)
    vv_std = norm_params.get('vv_std', np.nan)
    vh_mean = norm_params.get('vh_mean', np.nan)
    vh_std = norm_params.get('vh_std', np.nan)
    if np.isfinite(lfmc_mean) and np.isfinite(lfmc_std) and lfmc_std != 0:
        preds_i = preds_i * lfmc_std + lfmc_mean
        preds_i_std = preds_i_std * lfmc_std
    else:
        preds_i[:] = np.nan
        preds_i_std[:] = np.nan
    if np.isfinite(vv_mean) and np.isfinite(vv_std) and vv_std != 0:
        preds_vv = preds_vv * vv_std + vv_mean
        preds_vv_std = preds_vv_std * vv_std
    else:
        preds_vv[:] = np.nan
        preds_vv_std[:] = np.nan
    if np.isfinite(vh_mean) and np.isfinite(vh_std) and vh_std != 0:
        preds_vh = preds_vh * vh_std + vh_mean
        preds_vh_std = preds_vh_std * vh_std
    else:
        preds_vh[:] = np.nan
        preds_vh_std[:] = np.nan
    return {
        'lfmc_pred': preds_i,
        'lfmc_pred_std': preds_i_std,
        'vv_pred': preds_vv,
        'vv_pred_std': preds_vv_std,
        'vh_pred': preds_vh,
        'vh_pred_std': preds_vh_std,
    }


def _attach_predictions_to_info_df(info_df, preds):
    info_df = info_df.copy()
    info_df['lfmc_pred'] = preds['lfmc_pred']
    info_df['lfmc_pred_std'] = preds['lfmc_pred_std']
    info_df['vv_pred'] = preds['vv_pred']
    info_df['vv_pred_std'] = preds['vv_pred_std']
    info_df['vh_pred'] = preds['vh_pred']
    info_df['vh_pred_std'] = preds['vh_pred_std']
    return info_df


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
    model_type='standard',
    use_cuda_autocast=True,
):
    model, device = load_model_for_inference(
        short_input_dim=short_tensor.shape[-1],
        long_input_dim=long_tensor.shape[-1],
        static_input_dim=static_tensor.shape[-1],
        model_path=model_path,
        model_num_queries=model_num_queries,
        model_task_weights=model_task_weights,
        model_type=model_type,
    )
    preds = predict_with_loaded_model(
        short_tensor=short_tensor,
        long_tensor=long_tensor,
        static_tensor=static_tensor,
        model=model,
        device=device,
        norm_params=norm_params,
        batch_size=batch_size,
        use_cuda_autocast=use_cuda_autocast,
    )
    return _attach_predictions_to_info_df(info_df, preds)



def main():
    scratch_root = '/scratch/users/trobinet/long_lfmc/final_lfmc'
    scratch_model_dir = os.path.join(scratch_root, 'lfmc_model')
    # where to save the outputs of the model
    out_dir = os.path.join(scratch_model_dir, 'inference', 'for_mitch_20260124')
    plots_dir = os.path.join(scratch_model_dir, 'inference', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    # information about the sites to run
    site_info = pd.read_csv(
        os.path.join(
            scratch_model_dir,
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
        scratch_model_dir, 'inputs', 'news1_multitask', 'var_names.json'
    )
    with open(var_names_path) as f:
        var_names = json.load(f)
    model_dir = os.path.join(
        scratch_model_dir,
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
            'prcp','srad','swe','tmax','vpd',
            'srad_daily_anom','prcp_rolling30_anom','swe_daily_anom','tmax_daily_anom','vpd_daily_anom'
        ],
        'modis':[
            'Nadir_Reflectance_Band1_interp',
            'Nadir_Reflectance_Band2_interp',
            'Nadir_Reflectance_Band3_interp',
            'Nadir_Reflectance_Band4_interp',
            'Nadir_Reflectance_Band5_interp',
            'Nadir_Reflectance_Band6_interp',
            'Nadir_Reflectance_Band7_interp'
        ],
        'static':[
            'slope',
            'elevation',
        ],
        'soils':[
            'clay',
            'sand'
        ],
        'canopy_height':[
            'canopy_height',
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
    daymet_ds = xr.open_zarr(
        os.path.join(scratch_root, 'daymet', 'daymet_vars_and_anoms.zarr'),
        consolidated=False
    )
    dss = {
        'daymet': _merge_daymet_datasets(daymet_ds, None),
        'modis': xr.open_zarr(
            os.path.join(
                scratch_root,
                'modis',
                'modis_regrid_interpolated',
                'modis_interp_5d.zarr'
            )
        ),
        'static': xr.open_dataset(
            os.path.join(scratch_root, 'static', 'static_features_500m_epsg5070_float32.nc')
        ),
        'soils': xr.open_dataset(
            os.path.join(scratch_root, 'soils', 'soilgrids_top_500m_epsg5070.nc')
        ),
        'canopy_height': xr.open_dataset(
            os.path.join(scratch_root, 'canopy_height', 'gedi_canopy_height_2019_500m_epsg5070.nc')
        ),
        'landcover_frac': xr.open_zarr(
            os.path.join(scratch_root, 'nlcd', 'nlcd_target_grid_2000_2024.zarr')
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
