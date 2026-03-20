#!/usr/bin/env python3

import argparse
import gzip
import os
import re
import tarfile
import tempfile
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
SNODAS_CRS = 'EPSG:4326'
SNODAS_BASE_URL = 'https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{year}/{month_num:02d}_{month_name}/SNODAS_{date_str}.tar'
TXT_FIELD_RE = re.compile(r'^(.*?):\s*(.*?)\s*$')

SWE_ATTRS = {
    'long_name': 'snow water equivalent',
    'units': 'kg/m2',
    'cell_methods': 'area: mean time: mean',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download SNODAS SWE and write target-grid daily files.'
    )
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--registry_path', type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument('--grid_path', type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument('--raw_root', type=Path, default=None)
    parser.add_argument('--output_root', type=Path, default=None)
    parser.add_argument('--plots_dir', type=Path, default=None)
    parser.add_argument('--swe_product_token', type=str, default=None)
    parser.add_argument('--overwrite_existing', action='store_true')
    parser.add_argument('--check_only', action='store_true')
    return parser.parse_args()


def load_registry(registry_path: Path) -> dict:
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


def apply_registry_defaults(args):
    registry = load_registry(args.registry_path)
    snodas_cfg = registry.get('processing', {}).get('snodas', {})
    climate_cfg = registry.get('processing', {}).get('climate_low_latency', {})
    args.raw_root = args.raw_root or Path(snodas_cfg['raw_root'])
    args.output_root = args.output_root or Path(climate_cfg['regrid_root'])
    args.plots_dir = args.plots_dir or Path(snodas_cfg['plots_dir'])
    args.swe_product_token = args.swe_product_token or str(snodas_cfg['swe_product_token'])
    return args


def parse_date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f'end_date {end_date} is before start_date {start_date}')
    return pd.date_range(start, end, freq='D')


def snodas_tar_url(date_value: pd.Timestamp) -> str:
    return SNODAS_BASE_URL.format(
        year=date_value.year,
        month_num=date_value.month,
        month_name=date_value.strftime('%b'),
        date_str=date_value.strftime('%Y%m%d'),
    )


def target_file_path(output_root: Path, date_value: pd.Timestamp) -> Path:
    return (
        output_root
        / date_value.strftime('%Y')
        / f'daymet_v4_daily_na_swe_{date_value.strftime("%Y%m%d")}_regridded.nc'
    )


def range_complete(output_root: Path, dates: pd.DatetimeIndex) -> bool:
    return all(target_file_path(output_root, ts).exists() for ts in dates)


def range_available(dates: pd.DatetimeIndex):
    details = []
    for date_value in dates:
        url = snodas_tar_url(date_value)
        try:
            response = requests.head(url, allow_redirects=True, timeout=60)
            available = bool(response.ok)
            details.append({'date': date_value.strftime('%Y-%m-%d'), 'available': available, 'url': url})
        except Exception as exc:
            details.append({'date': date_value.strftime('%Y-%m-%d'), 'available': False, 'error': str(exc), 'url': url})
    available = all(item['available'] for item in details)
    return available, details


def download_tar_if_needed(date_value: pd.Timestamp, raw_root: Path) -> Path:
    tar_dir = raw_root / date_value.strftime('%Y') / date_value.strftime('%m')
    tar_dir.mkdir(parents=True, exist_ok=True)
    tar_path = tar_dir / f'SNODAS_{date_value.strftime("%Y%m%d")}.tar'
    if tar_path.exists():
        print(f'[SKIP] SNODAS tar exists: {tar_path}')
        return tar_path
    url = snodas_tar_url(date_value)
    print(f'Downloading SNODAS SWE for {date_value.date()} from {url}')
    with requests.get(url, timeout=300, stream=True) as response:
        response.raise_for_status()
        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return tar_path


def parse_metadata_text(text: str) -> dict:
    metadata = {}
    for line in text.splitlines():
        match = TXT_FIELD_RE.match(line)
        if match:
            metadata[match.group(1).strip()] = match.group(2).strip()
    return metadata


def extract_swe_members(tar_path: Path, product_token: str):
    with tarfile.open(tar_path) as tar:
        txt_members = [m for m in tar.getmembers() if m.name.endswith('.txt.gz') and product_token in m.name]
        dat_members = [m for m in tar.getmembers() if m.name.endswith('.dat.gz') and product_token in m.name]
        if len(txt_members) != 1 or len(dat_members) != 1:
            raise RuntimeError(
                f'Expected one SWE txt/dat member for token {product_token} in {tar_path}; '
                f'found txt={len(txt_members)} dat={len(dat_members)}'
            )
        metadata_text = gzip.decompress(tar.extractfile(txt_members[0]).read()).decode('utf-8', errors='replace')
        raw_bytes = gzip.decompress(tar.extractfile(dat_members[0]).read())
    return parse_metadata_text(metadata_text), raw_bytes


def snodas_array_from_raw(metadata: dict, raw_bytes: bytes) -> xr.DataArray:
    rows = int(metadata['Number of rows'])
    cols = int(metadata['Number of columns'])
    nodata = float(metadata['No data value'])
    x0 = float(metadata['Benchmark x-axis coordinate'])
    y0 = float(metadata['Benchmark y-axis coordinate'])
    xres = float(metadata['X-axis resolution'])
    yres = float(metadata['Y-axis resolution'])

    arr = np.frombuffer(raw_bytes, dtype='>i2').reshape(rows, cols).astype(np.float32)
    arr[arr == nodata] = np.nan
    # SNODAS SWE metadata reports "Meters / 1000"; converting to kg/m2 yields the raw integer values.
    values = arr

    x = x0 + (np.arange(cols, dtype=np.float64) * xres)
    y = y0 - (np.arange(rows, dtype=np.float64) * yres)
    da = xr.DataArray(values, dims=('y', 'x'), coords={'y': y, 'x': x}, name='swe')
    da = da.rio.write_crs(SNODAS_CRS)
    da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    return da


def target_template(grid_path: Path):
    grid = xr.open_dataset(grid_path)
    template = grid['lat'].copy(deep=True)
    template.name = 'template'
    template = template.rio.write_crs(TARGET_CRS)
    template = template.rio.set_spatial_dims(x_dim='x', y_dim='y')
    return grid, template


def write_target_daily_file(target_grid: xr.Dataset, values: np.ndarray, date_value: pd.Timestamp, output_root: Path, overwrite_existing: bool = False):
    out_path = target_file_path(output_root, date_value)
    if out_path.exists() and not overwrite_existing:
        print(f'[SKIP] Target daily file exists: {out_path}')
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        {
            'swe': xr.DataArray(
                values.astype(np.float32),
                dims=('y', 'x'),
                coords={'y': target_grid['y'], 'x': target_grid['x']},
                attrs=SWE_ATTRS,
            )
        },
        coords={
            'y': target_grid['y'],
            'x': target_grid['x'],
        },
    )
    ds_out = ds_out.rio.write_crs(TARGET_CRS)
    ds_out.to_netcdf(out_path)
    print(f'Wrote {out_path}')
    return out_path


def process_day(date_value: pd.Timestamp, args, target_grid: xr.Dataset, template: xr.DataArray):
    tar_path = download_tar_if_needed(date_value, args.raw_root)
    metadata, raw_bytes = extract_swe_members(tar_path, args.swe_product_token)
    src_da = snodas_array_from_raw(metadata, raw_bytes)
    projected = src_da.rio.reproject_match(template, resampling=Resampling.bilinear)
    write_target_daily_file(target_grid, projected.values, date_value, args.output_root, overwrite_existing=args.overwrite_existing)


def main():
    args = apply_registry_defaults(parse_args())
    dates = parse_date_range(args.start_date, args.end_date)
    print(f'Updating SNODAS SWE for {dates[0].date()} -> {dates[-1].date()}')
    if range_complete(args.output_root, dates) and not args.overwrite_existing:
        print('SNODAS target daily files already cover requested range; nothing to do')
        raise SystemExit(0)

    available, details = range_available(dates)
    if args.check_only:
        for item in details:
            print(item)
        raise SystemExit(0 if available else 1)
    if not available:
        missing = [item['date'] for item in details if not item['available']]
        raise SystemExit(f'SNODAS is not fully available for requested range; missing dates: {missing}')

    target_grid, template = target_template(args.grid_path)
    for date_value in dates:
        out_path = target_file_path(args.output_root, date_value)
        if out_path.exists() and not args.overwrite_existing:
            print(f'[SKIP] SNODAS output exists for {date_value.date()}')
            continue
        print(f'Processing SNODAS date {date_value.date()}')
        process_day(date_value, args, target_grid, template)


if __name__ == '__main__':
    main()
