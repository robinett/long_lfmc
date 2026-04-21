import argparse
import os
import re
import sys

import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import plotting as plot

FILE_RE = re.compile(
    r'daymet_v4(?P<latency>ll)?_daily_na_(?P<var>[a-z0-9]+)_(?P<stamp>\d{4}(?:\d{2})?)\.nc$'
)
DAYMET_LCC_PROJ = (
    '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 '
    '+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
)
DEFAULT_INPUT_ROOT = '/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_earthaccess'
DEFAULT_OUTPUT_ROOT = '/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/daymet_earthaccess_daily'
DEFAULT_PLOTS_DIR = '/scratch/users/trobinet/long_lfmc/final_lfmc/daymet/plots'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--plots_dir', type=str, default=DEFAULT_PLOTS_DIR)
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--var', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--run_smoke_test', action='store_true')
    parser.add_argument('--progress_every', type=int, default=10)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--plot_first_day', dest='plot_first_day', action='store_true')
    parser.add_argument('--no-plot_first_day', dest='plot_first_day', action='store_false')
    parser.set_defaults(plot_first_day=True)
    return parser.parse_args()


def _safe_strftime(dt, fmt):
    try:
        return dt.strftime(fmt)
    except Exception:
        try:
            import numpy as np
            s = np.datetime_as_string(dt, unit='D')
            return s.replace('-', '')
        except Exception:
            return f'{dt.year:04d}{dt.month:02d}{dt.day:02d}'


def process_file(
    in_path,
    out_root,
    year_override=None,
    dry_run=False,
    progress_every=10,
    plot_first_day=False,
    plot_dir=None,
    skip_existing=False,
):
    m = FILE_RE.search(os.path.basename(in_path))
    if not m:
        raise ValueError(f'Unexpected filename format: {in_path}')

    var_name = m.group('var')
    stamp = m.group('stamp')
    year = int(year_override or stamp[:4])
    out_year_dir = os.path.join(out_root, f'{year:04d}')
    os.makedirs(out_year_dir, exist_ok=True)

    with xr.open_dataset(in_path, chunks={'time': 1}) as ds:
        time_vals = ds['time'].values
        total = len(time_vals)

        for i, t in enumerate(time_vals):
            date_str = _safe_strftime(t, '%Y%m%d')
            out_name = f'daymet_v4_daily_na_{var_name}_{date_str}.nc'
            out_path = os.path.join(out_year_dir, out_name)

            if progress_every and (i == 0 or (i + 1) % progress_every == 0 or (i + 1) == total):
                print(f'  {os.path.basename(in_path)}: day {i + 1}/{total} -> {out_name}')

            if skip_existing and os.path.exists(out_path):
                continue
            if dry_run:
                print(out_path)
                continue

            day = ds.isel(time=i)
            day = day.expand_dims(time=[t])
            if plot_first_day and i == 0:
                if plot_dir is None:
                    raise ValueError('plot_dir must be set when plot_first_day is True')
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f'daymet_{var_name}_{date_str}.png')
                print(f'  plotting first day to {plot_path}')
                plot.plot_from_xarray(
                    load_type='ds',
                    type_obj=day,
                    var=var_name,
                    proj_in=DAYMET_LCC_PROJ,
                    proj_out='EPSG:4326',
                    fname=plot_path,
                )
            with ProgressBar():
                day.to_netcdf(out_path)


def discover_files(input_root, years=None, months=None, vars_filter=None):
    files = []
    if years:
        year_dirs = [os.path.join(input_root, str(y)) for y in years]
    else:
        year_dirs = [os.path.join(input_root, d) for d in os.listdir(input_root)]

    allowed_months = None if not months else {int(month) for month in months}
    allowed_vars = None if not vars_filter else {str(var_name) for var_name in vars_filter}

    for ydir in year_dirs:
        if not os.path.isdir(ydir):
            continue
        for name in os.listdir(ydir):
            if not name.endswith('.nc'):
                continue
            match = FILE_RE.search(name)
            if not match:
                continue
            if allowed_vars is not None and match.group('var') not in allowed_vars:
                continue
            stamp = match.group('stamp')
            if allowed_months is not None and len(stamp) == 6 and int(stamp[4:6]) not in allowed_months:
                continue
            files.append(os.path.join(ydir, name))
    return sorted(files)


def main():
    args = parse_args()
    years = [int(args.year)] if args.year is not None else None
    if years is None:
        year_env = os.environ.get('YEAR')
        if year_env:
            years = [int(year_env)]
    months = [int(args.month)] if args.month is not None else None
    vars_filter = [args.var] if args.var is not None else None

    files = discover_files(args.input_root, years, months, vars_filter)
    if not files:
        raise SystemExit('No matching NetCDF files found.')

    if args.run_smoke_test:
        _smoke_test(files[0], args.output_root)
        return

    total_files = len(files)
    for idx, path in enumerate(files, start=1):
        print(f'Processing file {idx}/{total_files}: {os.path.basename(path)}')
        process_file(
            path,
            args.output_root,
            dry_run=args.dry_run,
            progress_every=args.progress_every,
            plot_first_day=args.plot_first_day,
            plot_dir=args.plots_dir,
            skip_existing=args.skip_existing,
        )


def _smoke_test(sample_path, output_root):
    print('Running smoke test on:', sample_path)
    process_file(sample_path, output_root, dry_run=False)
    m = FILE_RE.search(os.path.basename(sample_path))
    var_name = m.group('var')
    stamp = m.group('stamp')
    year = stamp[:4]
    with xr.open_dataset(sample_path) as ds:
        t0 = ds['time'].values[0]
        date_str = _safe_strftime(t0, '%Y%m%d')
    out_path = os.path.join(output_root, year, f'daymet_v4_daily_na_{var_name}_{date_str}.nc')
    if not os.path.exists(out_path):
        raise SystemExit(f'Smoke test failed: missing output {out_path}')
    with xr.open_dataset(out_path) as out_ds:
        if out_ds.dims['time'] != 1:
            raise SystemExit('Smoke test failed: time dimension not 1')
    print('Smoke test OK:', out_path)


if __name__ == '__main__':
    main()
