#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401
import xarray as xr
import yaml
from rasterio.enums import Resampling


REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
DEFAULT_REGISTRY_PATH = REPO_ROOT / 'lfmc_model/scripts/inference/source_registry.yaml'
DEFAULT_GRID_PATH = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4')
TARGET_CRS = 'EPSG:5070'
PRISM_CRS = 'EPSG:4269'
PRISM_RELEASE_URL = 'https://services.nacse.org/prism/data/get/releaseDate/us/800m/{var_name}/{date_str}?json=true'
PRISM_DOWNLOAD_VARS = ['ppt', 'tmax', 'tdmean', 'soltotal']

PRISM_VAR_ATTRS = {
    'prcp': {
        'long_name': 'daily total precipitation',
        'units': 'mm/day',
        'cell_methods': 'area: mean time: sum',
    },
    'tmax': {
        'long_name': 'daily maximum temperature',
        'units': 'degrees C',
        'cell_methods': 'area: mean time: maximum',
    },
    'vp': {
        'long_name': 'daily average vapor pressure',
        'units': 'Pa',
        'cell_methods': 'area: mean time: mean',
    },
    'srad': {
        'long_name': 'daylight average incident shortwave radiation',
        'units': 'W/m2',
        'cell_methods': 'area: mean time: mean',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download PRISM low-latency climate inputs and write target-grid daily files.'
    )
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--registry_path', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--grid_path', type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument('--raw_root', type=Path, default=None)
    parser.add_argument('--extracted_root', type=Path, default=None)
    parser.add_argument('--output_root', type=Path, default=None)
    parser.add_argument('--plots_dir', type=Path, default=None)
    parser.add_argument('--release_latency_days', type=int, default=None)
    parser.add_argument('--overwrite_existing', action='store_true')
    parser.add_argument('--check_only', action='store_true')
    return parser.parse_args()


def load_registry(registry_path: Path) -> dict:
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


def apply_registry_defaults(args):
    registry = load_registry(args.registry_path)
    prism_cfg = registry.get('processing', {}).get('prism', {})
    climate_cfg = registry.get('processing', {}).get('climate_low_latency', {})
    args.raw_root = args.raw_root or Path(prism_cfg['raw_root'])
    args.extracted_root = args.extracted_root or Path(prism_cfg['extracted_root'])
    args.output_root = args.output_root or Path(climate_cfg['regrid_root'])
    args.plots_dir = args.plots_dir or Path(prism_cfg['plots_dir'])
    args.release_latency_days = args.release_latency_days or int(prism_cfg['release_latency_days'])
    return args


def parse_date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f'end_date {end_date} is before start_date {start_date}')
    return pd.date_range(start, end, freq='D')


def release_info(var_name: str, date_value: pd.Timestamp):
    url = PRISM_RELEASE_URL.format(
        var_name=var_name,
        date_str=date_value.strftime('%Y%m%d'),
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = json.loads(response.text)
    if not isinstance(payload, list) or len(payload) < 5:
        raise ValueError(f'Unexpected PRISM release payload for {var_name} {date_value.date()}: {payload!r}')
    return {
        'requested_date': payload[0],
        'release_date': payload[1],
        'variable': payload[2],
        'status_code': payload[3],
        'download_url': payload[4],
    }


def target_file_path(output_root: Path, var_name: str, date_value: pd.Timestamp) -> Path:
    return (
        output_root
        / date_value.strftime('%Y')
        / f'daymet_v4_daily_na_{var_name}_{date_value.strftime("%Y%m%d")}_regridded.nc'
    )


def range_complete(output_root: Path, dates: pd.DatetimeIndex) -> bool:
    expected = [
        target_file_path(output_root, var_name, ts)
        for ts in dates
        for var_name in ['prcp', 'tmax', 'vp', 'srad']
    ]
    return all(path.exists() for path in expected)


def range_available(dates: pd.DatetimeIndex):
    results = []
    for date_value in dates:
        date_status = {'date': date_value.strftime('%Y-%m-%d'), 'vars': {}}
        for var_name in PRISM_DOWNLOAD_VARS:
            try:
                info = release_info(var_name, date_value)
                date_status['vars'][var_name] = {'available': True, 'release_date': info['release_date']}
            except Exception as exc:
                date_status['vars'][var_name] = {'available': False, 'error': str(exc)}
        results.append(date_status)
    available = all(
        var_info.get('available', False)
        for date_status in results
        for var_info in date_status['vars'].values()
    )
    return available, results


def maybe_rate_limit_error(content_disposition: str, payload_path: Path) -> str | None:
    if content_disposition and 'error_msg' in content_disposition:
        with open(payload_path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    return None


def download_zip_if_needed(var_name: str, date_value: pd.Timestamp, raw_root: Path) -> Path:
    raw_dir = raw_root / date_value.strftime('%Y') / date_value.strftime('%m') / var_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f'prism_{var_name}_{date_value.strftime("%Y%m%d")}.zip'
    if zip_path.exists():
        if not zipfile.is_zipfile(zip_path):
            print(f'[WARN] Existing PRISM payload is not a valid zip; removing {zip_path}')
            zip_path.unlink()
        else:
            print(f'[SKIP] PRISM zip exists: {zip_path}')
            return zip_path

    info = release_info(var_name, date_value)
    print(f'Downloading PRISM {var_name} for {date_value.date()} from {info["download_url"]}')
    with requests.get(info['download_url'], timeout=300, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        content_disposition = response.headers.get('content-disposition', '')

    rate_limit_msg = maybe_rate_limit_error(content_disposition, zip_path)
    if rate_limit_msg is not None:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f'PRISM rate-limit response for {var_name} {date_value.date()}: {rate_limit_msg.strip()}')
    return zip_path


def extract_tif_if_needed(var_name: str, date_value: pd.Timestamp, zip_path: Path, extracted_root: Path) -> Path:
    extract_dir = extracted_root / date_value.strftime('%Y') / date_value.strftime('%m') / var_name
    extract_dir.mkdir(parents=True, exist_ok=True)
    tif_path = extract_dir / f'prism_{var_name}_us_30s_{date_value.strftime("%Y%m%d")}.tif'
    if tif_path.exists():
        print(f'[SKIP] PRISM tif exists: {tif_path}')
        return tif_path

    with zipfile.ZipFile(zip_path) as zf:
        tif_names = [name for name in zf.namelist() if name.endswith('.tif')]
        if len(tif_names) != 1:
            raise RuntimeError(f'Expected exactly one tif in {zip_path}; found {tif_names}')
        with zf.open(tif_names[0]) as src, open(tif_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    return tif_path


def target_template(grid_path: Path):
    grid = xr.open_dataset(grid_path)
    template = grid['lat'].copy(deep=True)
    template.name = 'template'
    template = template.rio.write_crs(TARGET_CRS)
    template = template.rio.set_spatial_dims(x_dim='x', y_dim='y')
    return grid, template


def reproject_prism_tif(tif_path: Path, template: xr.DataArray) -> xr.DataArray:
    da = rioxarray.open_rasterio(tif_path, masked=True).squeeze(drop=True)
    if da.rio.crs is None:
        da = da.rio.write_crs(PRISM_CRS)
    da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    projected = da.rio.reproject_match(template, resampling=Resampling.bilinear)
    return projected.astype(np.float32)


def daylight_seconds(lat_values: np.ndarray, date_value: pd.Timestamp) -> np.ndarray:
    doy = int(date_value.dayofyear)
    lat_rad = np.deg2rad(lat_values.astype(np.float64))
    decl = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0))
    day_hours = 24.0 * ws / np.pi
    return np.maximum(day_hours * 3600.0, 1.0)


def derive_vp_from_tdmean(tdmean_c: np.ndarray) -> np.ndarray:
    td = tdmean_c.astype(np.float64)
    vp_pa = 611.2 * np.exp((17.67 * td) / (td + 243.5))
    return vp_pa.astype(np.float32)


def derive_srad_from_soltotal(soltotal: np.ndarray, lat_values: np.ndarray, date_value: pd.Timestamp) -> np.ndarray:
    sol = soltotal.astype(np.float64)
    daylight = daylight_seconds(lat_values, date_value)
    srad = (sol * 1_000_000.0) / daylight
    return srad.astype(np.float32)


def write_target_daily_file(
    target_grid: xr.Dataset,
    values: np.ndarray,
    var_name: str,
    date_value: pd.Timestamp,
    output_root: Path,
    overwrite_existing: bool = False,
):
    out_path = target_file_path(output_root, var_name, date_value)
    if out_path.exists() and not overwrite_existing:
        print(f'[SKIP] Target daily file exists: {out_path}')
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = {
        'y': target_grid['y'],
        'x': target_grid['x'],
    }
    ds_out = xr.Dataset(
        {
            var_name: xr.DataArray(
                values.astype(np.float32),
                dims=('y', 'x'),
                coords={'y': target_grid['y'], 'x': target_grid['x']},
                attrs=PRISM_VAR_ATTRS[var_name],
            )
        },
        coords=coords,
    )
    ds_out = ds_out.rio.write_crs(TARGET_CRS)
    ds_out.to_netcdf(out_path)
    print(f'Wrote {out_path}')
    return out_path


def process_day(date_value: pd.Timestamp, args, target_grid: xr.Dataset, template: xr.DataArray):
    daily_arrays = {}
    for prism_var in PRISM_DOWNLOAD_VARS:
        zip_path = download_zip_if_needed(prism_var, date_value, args.raw_root)
        tif_path = extract_tif_if_needed(prism_var, date_value, zip_path, args.extracted_root)
        daily_arrays[prism_var] = reproject_prism_tif(tif_path, template).values

    prcp = daily_arrays['ppt']
    tmax = daily_arrays['tmax']
    vp = derive_vp_from_tdmean(daily_arrays['tdmean'])
    srad = derive_srad_from_soltotal(daily_arrays['soltotal'], np.asarray(target_grid['lat'].values), date_value)

    write_target_daily_file(target_grid, prcp, 'prcp', date_value, args.output_root, overwrite_existing=args.overwrite_existing)
    write_target_daily_file(target_grid, tmax, 'tmax', date_value, args.output_root, overwrite_existing=args.overwrite_existing)
    write_target_daily_file(target_grid, vp, 'vp', date_value, args.output_root, overwrite_existing=args.overwrite_existing)
    write_target_daily_file(target_grid, srad, 'srad', date_value, args.output_root, overwrite_existing=args.overwrite_existing)


def main():
    args = apply_registry_defaults(parse_args())
    dates = parse_date_range(args.start_date, args.end_date)
    print(f'Updating PRISM for {dates[0].date()} -> {dates[-1].date()}')
    if range_complete(args.output_root, dates) and not args.overwrite_existing:
        print('PRISM target daily files already cover requested range; nothing to do')
        raise SystemExit(0)

    available, details = range_available(dates)
    if args.check_only:
        for date_status in details:
            print(date_status)
        raise SystemExit(0 if available else 1)
    if not available:
        missing = [
            date_status['date']
            for date_status in details
            if not all(var_info.get('available', False) for var_info in date_status['vars'].values())
        ]
        raise SystemExit(f'PRISM is not fully available for requested range; missing dates: {missing}')

    target_grid, template = target_template(args.grid_path)
    for date_value in dates:
        missing_vars = [
            var_name for var_name in ['prcp', 'tmax', 'vp', 'srad']
            if not target_file_path(args.output_root, var_name, date_value).exists()
        ]
        if len(missing_vars) == 0 and not args.overwrite_existing:
            print(f'[SKIP] PRISM outputs already exist for {date_value.date()}')
            continue
        print(f'Processing PRISM date {date_value.date()}')
        process_day(date_value, args, target_grid, template)


if __name__ == '__main__':
    main()
