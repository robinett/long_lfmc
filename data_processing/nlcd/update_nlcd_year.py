#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import xarray as xr

REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
RAW_URL_FILE = REPO_ROOT / 'data_processing/nlcd/raw_download_urls.txt'
RAW_DIR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_raw')
RAW_ZARR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_2000_2024.zarr')
TARGET_ZARR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_target_grid_2000_2024.zarr')


def parse_args():
    parser = argparse.ArgumentParser(description='Download and append one annual NLCD year into the raw and target zarr stores.')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--url_file', type=Path, default=RAW_URL_FILE)
    parser.add_argument('--raw_dir', type=Path, default=RAW_DIR)
    parser.add_argument('--raw_zarr', type=Path, default=RAW_ZARR)
    parser.add_argument('--target_zarr', type=Path, default=TARGET_ZARR)
    parser.add_argument('--check_only', action='store_true')
    return parser.parse_args()


def run_cmd(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def zarr_has_year(zarr_path: Path, coord_name: str, year: int) -> bool:
    if not zarr_path.exists():
        return False
    ds = xr.open_zarr(zarr_path)
    years = pd.DatetimeIndex(pd.to_datetime(ds[coord_name].values)).year
    return int(year) in set(years.tolist())


def year_url(url_file: Path, year: int) -> str:
    target = f'_{year}_'
    for raw_line in url_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if target in line:
            return line
    raise FileNotFoundError(f'No NLCD download URL found for year {year} in {url_file}')


def remote_url_available(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            return 200 <= getattr(response, 'status', 200) < 400
    except Exception:
        try:
            req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                return 200 <= getattr(response, 'status', 200) < 400
        except Exception:
            return False


def check_year_available(year: int, url_file: Path, raw_dir: Path, raw_zarr: Path, target_zarr: Path):
    del raw_dir
    raw_done = zarr_has_year(raw_zarr, 'time', year)
    target_done = zarr_has_year(target_zarr, 'year', year)
    if raw_done and target_done:
        return True, 'raw_and_target_zarr_already_contain_year'
    try:
        url = year_url(url_file, year)
    except FileNotFoundError:
        return False, 'year_missing_from_nlcd_url_manifest'
    if remote_url_available(url):
        return True, 'remote_probe_ok'
    return False, 'remote_probe_failed'


def local_raw_tif(year: int, raw_dir: Path) -> Path | None:
    existing = sorted(raw_dir.glob(f'*{year}*.tif'))
    if len(existing) == 0:
        return None
    if len(existing) > 1:
        raise ValueError(f'Multiple raw NLCD tifs found for year {year}: {existing}')
    return existing[0]


def ensure_raw_tif(year: int, url_file: Path, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = local_raw_tif(year, raw_dir)
    if existing is not None:
        print(f'NLCD raw tif already present for {year}: {existing.name}')
        return existing

    url = year_url(url_file, year)
    zip_name = Path(urlparse(url).path).name
    zip_path = raw_dir / zip_name
    print(f'Downloading NLCD year {year} from {url}')
    run_cmd(['wget', '-c', '-O', zip_path, url])

    with tempfile.TemporaryDirectory(dir=str(raw_dir)) as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f'Extracting {zip_path.name} into {tmp_path}')
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        tifs = sorted(tmp_path.rglob('*.tif'))
        if len(tifs) == 0:
            raise FileNotFoundError(f'No tif found inside {zip_path}')
        target_path = raw_dir / f'{Path(zip_name).stem}.tif'
        shutil.move(str(tifs[0]), str(target_path))
        print(f'Extracted {target_path}')
    if zip_path.exists():
        zip_path.unlink()
    return target_path


def main():
    args = parse_args()
    year = int(args.year)
    print(f'Updating NLCD for year {year}')

    available, reason = check_year_available(year, args.url_file, args.raw_dir, args.raw_zarr, args.target_zarr)
    if args.check_only:
        print(f'NLCD availability check for {year}: available={available} reason={reason}')
        raise SystemExit(0 if available else 1)

    raw_done = zarr_has_year(args.raw_zarr, 'time', year)
    target_done = zarr_has_year(args.target_zarr, 'year', year)
    if raw_done and target_done:
        print(f'Both NLCD zarr stores already contain year {year}; nothing to do')
        return

    ensure_raw_tif(year, args.url_file, args.raw_dir)

    if not raw_done:
        run_cmd([
            sys.executable,
            REPO_ROOT / 'data_processing/nlcd/append_nlcd_year.py',
            '--raw_dir', args.raw_dir,
            '--out_zarr', args.raw_zarr,
            '--year', str(year),
        ])
    else:
        print(f'Raw NLCD zarr already contains year {year}; skipping raw append')

    if not target_done:
        run_cmd([
            sys.executable,
            REPO_ROOT / 'data_processing/nlcd/append_nlcd_target_year.py',
            '--raw_zarr', args.raw_zarr,
            '--out_zarr', args.target_zarr,
            '--year', str(year),
        ])
    else:
        print(f'Target-grid NLCD zarr already contains year {year}; skipping target append')

    print(f'Finished NLCD update for year {year}')


if __name__ == '__main__':
    main()
