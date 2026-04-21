#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import earthaccess
import pandas as pd
import xarray as xr

REPO_ROOT = Path('/home/users/trobinet/long_lfmc')
GRID_PATH = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4')
DAYMET_EARTHACCESS_ROOT = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_earthaccess')
DAYMET_DAILY_ROOT = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_earthaccess_daily')
DAYMET_REGRID_ROOT = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_regrid')
DAYMET_ARCHIVE_ZARR = Path('/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_all_vars.zarr')
DAYMET_VARS = ['tmax', 'prcp', 'vp', 'swe', 'srad']
DAYMET_PROBE_VARS = ['tmax', 'vp']
DAYMET_URL_TEMPLATE = (
    'https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/'
    'data/daymet_v4_daily_na_{var}_{year}.nc'
)
DAYMET_CRS = '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
TARGET_CRS = 'EPSG:5070'
DAYMET_TO_DAILY_WORKER = REPO_ROOT / 'data_processing/daymet/run_daymet_to_daily_var.sbatch'
DAYMET_REGRID_WORKER = REPO_ROOT / 'data_processing/daymet/run_daymet_regrid_var.sbatch'
DAYMET_PARALLEL_LOG_ROOT = REPO_ROOT / 'logs/daymet_parallel'


def parse_args():
    parser = argparse.ArgumentParser(description='Download, split, regrid, and append one archive-quality Daymet year.')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--grid_path', type=Path, default=GRID_PATH)
    parser.add_argument('--earthaccess_root', type=Path, default=DAYMET_EARTHACCESS_ROOT)
    parser.add_argument('--daily_root', type=Path, default=DAYMET_DAILY_ROOT)
    parser.add_argument('--regrid_root', type=Path, default=DAYMET_REGRID_ROOT)
    parser.add_argument('--archive_zarr', type=Path, default=DAYMET_ARCHIVE_ZARR)
    parser.add_argument('--chunk_buffer', type=int, default=200)
    parser.add_argument('--chunk_size', type=int, default=2000)
    parser.add_argument('--check_only', action='store_true')
    return parser.parse_args()


def _fallback_expected_days(year: int) -> int:
    return 366 if pd.Timestamp(f'{year}-01-01').is_leap_year else 365


def _first_available_annual_file(year: int, earthaccess_root: Path) -> Path | None:
    year_dir = earthaccess_root / str(year)
    for var_name in DAYMET_VARS:
        candidate = year_dir / f'daymet_v4_daily_na_{var_name}_{year}.nc'
        if candidate.exists():
            return candidate
    return None


def source_daymet_dates(year: int, earthaccess_root: Path) -> pd.DatetimeIndex:
    annual_path = _first_available_annual_file(year, earthaccess_root)
    if annual_path is None:
        return pd.DatetimeIndex([])
    with xr.open_dataset(annual_path) as ds:
        return pd.DatetimeIndex(pd.to_datetime(ds['time'].values)).normalize().unique()


def expected_output_dates(year: int, earthaccess_root: Path) -> pd.DatetimeIndex:
    source_dates = source_daymet_dates(year, earthaccess_root)
    if len(source_dates) == 0:
        return pd.date_range(f'{year}-01-01', periods=_fallback_expected_days(year), freq='D')
    expected_dates = list(source_dates)
    needs_synthetic_dec31 = (
        pd.Timestamp(f'{year}-01-01').is_leap_year
        and len(source_dates) == 365
        and pd.Timestamp(f'{year}-02-29') in source_dates
        and source_dates[-1] == pd.Timestamp(f'{year}-12-30')
        and pd.Timestamp(f'{year}-12-31') not in source_dates
    )
    if needs_synthetic_dec31:
        expected_dates.append(pd.Timestamp(f'{year}-12-31'))
    return pd.DatetimeIndex(expected_dates).unique().sort_values()


def _maybe_copy_dec30_to_dec31(var_name: str, src_path: Path, dst_path: Path, label: str) -> None:
    if dst_path.exists():
        return
    if not src_path.exists():
        print(f'Cannot synthesize {label} Dec 31 for var={var_name}; missing Dec 30 source {src_path}')
        return
    shutil.copy2(src_path, dst_path)
    print(f'Synthesized {label} Dec 31 for var={var_name}: {dst_path}')


def ensure_synthetic_dec31_daily(year: int, earthaccess_root: Path, daily_root: Path) -> None:
    expected_dates = expected_output_dates(year, earthaccess_root)
    if pd.Timestamp(f'{year}-12-31') not in expected_dates:
        return
    daily_year_dir = daily_root / str(year)
    for var_name in DAYMET_VARS:
        _maybe_copy_dec30_to_dec31(
            var_name,
            daily_year_dir / f'daymet_v4_daily_na_{var_name}_{year}1230.nc',
            daily_year_dir / f'daymet_v4_daily_na_{var_name}_{year}1231.nc',
            'daily',
        )


def ensure_synthetic_dec31_regrid(year: int, earthaccess_root: Path, regrid_root: Path) -> None:
    expected_dates = expected_output_dates(year, earthaccess_root)
    if pd.Timestamp(f'{year}-12-31') not in expected_dates:
        return
    regrid_year_dir = regrid_root / str(year)
    for var_name in DAYMET_VARS:
        _maybe_copy_dec30_to_dec31(
            var_name,
            regrid_year_dir / f'daymet_v4_daily_na_{var_name}_{year}1230_regridded.nc',
            regrid_year_dir / f'daymet_v4_daily_na_{var_name}_{year}1231_regridded.nc',
            'regridded',
        )


def archive_has_year(archive_zarr: Path, year: int, earthaccess_root: Path) -> bool:
    if not archive_zarr.exists():
        return False
    ds = xr.open_zarr(archive_zarr)
    times = pd.DatetimeIndex(pd.to_datetime(ds['time'].values)).normalize()
    year_times = times[times.year == int(year)]
    if len(year_times) == 0:
        return False
    expected_dates = expected_output_dates(year, earthaccess_root)
    if len(expected_dates) == 0:
        expected_dates = pd.date_range(f'{year}-01-01', periods=_fallback_expected_days(year), freq='D')
    actual_dates = pd.DatetimeIndex(year_times.unique()).sort_values()
    return len(actual_dates) >= len(expected_dates) and actual_dates.max() >= expected_dates.max()


def _probe_remote_daymet_var(session, year: int, var_name: str) -> bool:
    url = DAYMET_URL_TEMPLATE.format(var=var_name, year=year)
    try:
        response = session.head(url, allow_redirects=True, timeout=30)
        if response.ok:
            return True
        response = session.get(url, stream=True, timeout=30)
        try:
            return response.ok
        finally:
            response.close()
    except Exception:
        return False


def remote_daymet_year_available(year: int) -> bool:
    earthaccess.login()
    session = earthaccess.get_requests_https_session()
    return all(_probe_remote_daymet_var(session, year, var_name) for var_name in DAYMET_PROBE_VARS)


def check_year_available(year: int, archive_zarr: Path, earthaccess_root: Path):
    if archive_has_year(archive_zarr, year, earthaccess_root):
        return True, 'archive_zarr_already_contains_year'
    if remote_daymet_year_available(year):
        return True, f'remote_probe_ok:{"-".join(DAYMET_PROBE_VARS)}'
    return False, f'remote_probe_failed:{"-".join(DAYMET_PROBE_VARS)}'


def run_cmd(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def run_capture(cmd):
    print('Running command:')
    print('  ' + ' '.join(str(part) for part in cmd))
    return subprocess.run([str(part) for part in cmd], check=True, capture_output=True, text=True)


def ensure_annual_downloads(year: int, earthaccess_root: Path):
    year_dir = earthaccess_root / str(year)
    missing = [
        var for var in DAYMET_VARS
        if not (year_dir / f'daymet_v4_daily_na_{var}_{year}.nc').exists()
    ]
    if len(missing) == 0:
        print(f'All Daymet annual files already present for {year} in {year_dir}')
        return
    print(f'Missing annual Daymet files for {year}: {missing}')
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/daymet/get_daymet.py',
        '--start_date', f'{year}-01-01',
        '--end_date', f'{year}-12-31',
        '--output_root', earthaccess_root,
        '--skip_existing',
    ])


def job_is_active(job_id: str) -> bool:
    result = subprocess.run(
        ['squeue', '-h', '-j', str(job_id), '-o', '%T'],
        check=True,
        capture_output=True,
        text=True,
    )
    states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return len(states) > 0


def final_job_state(job_id: str) -> str:
    result = subprocess.run(
        ['sacct', '-j', str(job_id), '--format=State', '-n', '-P'],
        check=True,
        capture_output=True,
        text=True,
    )
    states = [line.strip().split('|')[0] for line in result.stdout.splitlines() if line.strip()]
    if not states:
        return 'UNKNOWN'
    for state in states:
        if state not in {'COMPLETED'}:
            return state
    return 'COMPLETED'


def submit_sbatch_job(script_path: Path, env_vars: dict[str, str], label: str) -> str:
    env_arg = ','.join(['ALL'] + [f'{key}={value}' for key, value in env_vars.items()])
    result = run_capture(['sbatch', '--parsable', '--export=' + env_arg, script_path])
    job_id = result.stdout.strip()
    print(f'Submitted {label} job {job_id}')
    return job_id


def wait_for_jobs(job_ids: dict[str, str], label: str, poll_seconds: int = 30) -> None:
    pending = dict(job_ids)
    print(f'Waiting for {label} jobs to complete: {pending}')
    while pending:
        completed_vars = []
        for var_name, job_id in pending.items():
            if job_is_active(job_id):
                print(f'  {label} var={var_name} job={job_id} state=ACTIVE; sleeping')
                continue
            state = final_job_state(job_id)
            print(f'  {label} var={var_name} job={job_id} final_state={state}')
            if state != 'COMPLETED':
                raise RuntimeError(f'{label} job failed for var={var_name} job={job_id} state={state}')
            completed_vars.append(var_name)
        for var_name in completed_vars:
            pending.pop(var_name, None)
        if pending:
            time.sleep(poll_seconds)


def daily_counts_by_var(daily_year_dir: Path) -> dict[str, int]:
    counts = {var_name: 0 for var_name in DAYMET_VARS}
    if not daily_year_dir.exists():
        return counts
    for var_name in DAYMET_VARS:
        counts[var_name] = len(list(daily_year_dir.glob(f'daymet_v4_daily_na_{var_name}_*.nc')))
    return counts


def regrid_counts_by_var(regrid_year_dir: Path) -> dict[str, int]:
    counts = {var_name: 0 for var_name in DAYMET_VARS}
    if not regrid_year_dir.exists():
        return counts
    for var_name in DAYMET_VARS:
        counts[var_name] = len(list(regrid_year_dir.glob(f'daymet_v4_daily_na_{var_name}_*_regridded.nc*')))
    return counts


def ensure_daily_split(year: int, earthaccess_root: Path, daily_root: Path):
    year_dir = earthaccess_root / str(year)
    annual_files = sorted(year_dir.glob('daymet_v4_daily_na_*_*.nc'))
    if len(annual_files) == 0:
        raise FileNotFoundError(f'No annual Daymet files found for {year} in {year_dir}')

    daily_year_dir = daily_root / str(year)
    expected_days = len(expected_output_dates(year, earthaccess_root))
    counts = daily_counts_by_var(daily_year_dir)
    if all(counts[var_name] >= expected_days for var_name in DAYMET_VARS):
        print(f'Daymet daily split already looks populated for {year}: {counts}')
        return
    DAYMET_PARALLEL_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    job_ids = {}
    for var_name in DAYMET_VARS:
        if counts[var_name] >= expected_days:
            print(f'Daymet daily split already complete for var={var_name}: {counts[var_name]} files')
            continue
        job_ids[var_name] = submit_sbatch_job(
            DAYMET_TO_DAILY_WORKER,
            {
                'YEAR': str(year),
                'VAR': var_name,
                'INPUT_ROOT': str(earthaccess_root),
                'OUTPUT_ROOT': str(daily_root),
                'SKIP_EXISTING': '1',
                'PROGRESS_EVERY': '10',
            },
            f'daymet_to_daily[{var_name}]',
        )
    if job_ids:
        wait_for_jobs(job_ids, 'daymet_to_daily')
    ensure_synthetic_dec31_daily(year, earthaccess_root, daily_root)
    counts = daily_counts_by_var(daily_year_dir)
    if not all(counts[var_name] >= expected_days for var_name in DAYMET_VARS):
        raise RuntimeError(f'Daymet daily split did not complete for year {year}: {counts}')
    print(f'Daymet daily split complete for {year}: {counts}')


def ensure_regridded_year(year: int, grid_path: Path, earthaccess_root: Path, daily_root: Path, regrid_root: Path, chunk_size: int, chunk_buffer: int):
    src_dir = daily_root / str(year)
    target_dir = regrid_root / str(year)
    if not src_dir.exists():
        raise FileNotFoundError(f'Daymet daily directory missing for {year}: {src_dir}')
    ensure_synthetic_dec31_daily(year, earthaccess_root, daily_root)
    expected_days = len(expected_output_dates(year, earthaccess_root))
    counts = regrid_counts_by_var(target_dir)
    ensure_synthetic_dec31_regrid(year, earthaccess_root, regrid_root)
    counts = regrid_counts_by_var(target_dir)
    if all(counts[var_name] >= expected_days for var_name in DAYMET_VARS):
        print(f'Daymet regridded output already looks populated for {year}: {counts}')
        return
    DAYMET_PARALLEL_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    job_ids = {}
    for var_name in DAYMET_VARS:
        if counts[var_name] >= expected_days:
            print(f'Daymet regrid already complete for var={var_name}: {counts[var_name]} files')
            continue
        job_ids[var_name] = submit_sbatch_job(
            DAYMET_REGRID_WORKER,
            {
                'YEAR': str(year),
                'VAR': var_name,
                'GRID_PATH': str(grid_path),
                'SRC_DIR': str(src_dir),
                'TARGET_DIR': str(target_dir),
                'CHUNK_SIZE': str(chunk_size),
                'CHUNK_BUFFER': str(chunk_buffer),
                'SRC_CRS': DAYMET_CRS,
                'TARGET_CRS': TARGET_CRS,
            },
            f'daymet_regrid[{var_name}]',
        )
    if job_ids:
        wait_for_jobs(job_ids, 'daymet_regrid')
    ensure_synthetic_dec31_regrid(year, earthaccess_root, regrid_root)
    counts = regrid_counts_by_var(target_dir)
    if not all(counts[var_name] >= expected_days for var_name in DAYMET_VARS):
        raise RuntimeError(f'Daymet regrid did not complete for year {year}: {counts}')
    print(f'Daymet regridded output complete for {year}: {counts}')


def append_archive_year(year: int, archive_zarr: Path, regrid_root: Path):
    run_cmd([
        sys.executable,
        REPO_ROOT / 'data_processing/convert_to_zarr/append_daymet_archive_year.py',
        '--root', regrid_root,
        '--out_zarr', archive_zarr,
        '--year', str(year),
    ])


def main():
    args = parse_args()
    year = int(args.year)
    print(f'Updating archive Daymet for year {year}')
    available, reason = check_year_available(year, args.archive_zarr, args.earthaccess_root)
    if args.check_only:
        print(f'Daymet availability check for {year}: available={available} reason={reason}')
        raise SystemExit(0 if available else 1)
    if archive_has_year(args.archive_zarr, year, args.earthaccess_root):
        print(f'Archive Daymet zarr already contains year {year}; nothing to do')
        return
    ensure_annual_downloads(year, args.earthaccess_root)
    ensure_daily_split(year, args.earthaccess_root, args.daily_root)
    ensure_regridded_year(year, args.grid_path, args.earthaccess_root, args.daily_root, args.regrid_root, args.chunk_size, args.chunk_buffer)
    append_archive_year(year, args.archive_zarr, args.regrid_root)
    print(f'Finished archive Daymet update for year {year}')


if __name__ == '__main__':
    main()
