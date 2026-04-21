import argparse
import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm
from dask.diagnostics import ProgressBar

DATE_PATTERN = re.compile(r"(\d{8})")
ZARR_VERSION = 3

warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
)


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


def extract_date_from_filename(path):
    match = DATE_PATTERN.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Date not found in filename: {path}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def candidate_modis_paths(base_path, dt):
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    base_dir = Path(base_path) / year / month
    date_tag = f"{year}{month}{day}"
    return [
        base_dir / f"modis_reflectance_{date_tag}.nc4",
        base_dir / f"modis_reflectance_{date_tag}_regridded.nc4",
    ]


def build_daily_file_map(base_path, start_dt, end_dt):
    file_map = {}
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq="D")
    for dt in all_dates:
        for path in candidate_modis_paths(base_path, dt.to_pydatetime()):
            if path.exists():
                file_map[pd.Timestamp(dt)] = str(path)
                break
    return all_dates, file_map


def find_reference_file(file_map):
    for path in file_map.values():
        return path
    raise FileNotFoundError("No input files found in the requested range.")


def infer_dims_and_vars(ds):
    if "y" in ds.dims and "x" in ds.dims:
        y_dim, x_dim = "y", "x"
    else:
        spatial_pairs = []
        for y_name in ["y", "lat", "latitude"]:
            for x_name in ["x", "lon", "longitude"]:
                if y_name in ds.dims and x_name in ds.dims:
                    spatial_pairs.append((y_name, x_name))
        if not spatial_pairs:
            raise ValueError(f"Could not infer spatial dims from dims={dict(ds.dims)}")
        y_dim, x_dim = spatial_pairs[0]

    interp_vars = []
    for var_name, da in ds.data_vars.items():
        if da.dims == (y_dim, x_dim) and np.issubdtype(da.dtype, np.number):
            if var_name == "spatial_ref":
                continue
            interp_vars.append(var_name)
    if not interp_vars:
        raise ValueError("No 2D numeric interpolation variables found.")
    return y_dim, x_dim, interp_vars


def choose_check_band(ds, check_band):
    if check_band and check_band in ds.data_vars:
        return check_band
    for candidate in ds.data_vars:
        if "Reflectance" in candidate:
            return candidate
    for candidate, da in ds.data_vars.items():
        if len(da.dims) == 2 and np.issubdtype(da.dtype, np.number) and candidate != "spatial_ref":
            return candidate
    raise ValueError("Could not select a band for interpolation checking.")


def _interpolate_2d_time_gap(arr_2d, max_gap_days):
    """Fill internal NaN gaps up to max_gap_days along axis 0 (time)."""
    out = np.array(arr_2d, dtype=np.float32, copy=True)
    is_valid = np.isfinite(out)
    is_nan = ~is_valid
    t_len, n_pix = out.shape

    if t_len == 0 or n_pix == 0:
        return out, np.zeros_like(out, dtype=np.uint8)

    time_idx = np.arange(t_len, dtype=np.int32)[:, None]

    prev_idx = np.where(is_valid, time_idx, -1)
    np.maximum.accumulate(prev_idx, axis=0, out=prev_idx)

    next_idx = np.where(is_valid, time_idx, t_len)
    next_idx = np.minimum.accumulate(next_idx[::-1], axis=0)[::-1]

    has_prev = prev_idx >= 0
    has_next = next_idx < t_len
    interior_missing = is_nan & has_prev & has_next

    gap_len = next_idx - prev_idx - 1
    fillable = interior_missing & (gap_len <= max_gap_days)

    fill_r, fill_c = np.where(fillable)
    if fill_r.size > 0:
        p_idx = prev_idx[fill_r, fill_c]
        n_idx = next_idx[fill_r, fill_c]
        p_val = out[p_idx, fill_c]
        n_val = out[n_idx, fill_c]
        weight = (fill_r - p_idx).astype(np.float32) / (n_idx - p_idx).astype(np.float32)
        out[fill_r, fill_c] = p_val + (n_val - p_val) * weight

    status = np.zeros(out.shape, dtype=np.uint8)
    status[is_nan & np.isfinite(out)] = 1
    status[is_nan & ~np.isfinite(out)] = 2
    return out, status


def month_labels():
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def format_elapsed(seconds):
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def maybe_tqdm(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def run_check_interpolation(
    base_path,
    sample_size,
    thresholds,
    plot_path,
    map_plot_path,
    check_band=None,
    check_year=2023,
    seed=0,
):
    year_start = datetime(check_year, 1, 1)
    year_end = datetime(check_year, 12, 31)
    max_thresh = max(thresholds)
    proc_start = year_start - timedelta(days=max_thresh)
    proc_end = year_end + timedelta(days=max_thresh)

    proc_dates, proc_file_map = build_daily_file_map(base_path, proc_start, proc_end)
    if len(proc_file_map) == 0:
        raise FileNotFoundError(f"No files found for interpolation check around {check_year}.")

    ref_file = find_reference_file(proc_file_map)
    with xr.open_dataset(ref_file, engine="netcdf4") as ref_ds:
        y_dim, x_dim, _ = infer_dims_and_vars(ref_ds)
        band_name = choose_check_band(ref_ds, check_band)
        y_size = ref_ds.sizes[y_dim]
        x_size = ref_ds.sizes[x_dim]
        x_coord_vals = ref_ds[x_dim].values if x_dim in ref_ds.coords else None
        y_coord_vals = ref_ds[y_dim].values if y_dim in ref_ds.coords else None
        lon_grid = ref_ds["lon"].values if "lon" in ref_ds.coords else None
        lat_grid = ref_ds["lat"].values if "lat" in ref_ds.coords else None

    print(f"check_interpolation band: {band_name}")
    print(f"building valid-pixel mask for {check_year}")

    valid_any = np.zeros((y_size, x_size), dtype=bool)
    year_dates = pd.date_range(year_start, year_end, freq="D")
    for i, dt in enumerate(year_dates, start=1):
        path = proc_file_map.get(pd.Timestamp(dt))
        if path is None:
            continue
        if i % 30 == 0 or i == 1 or i == len(year_dates):
            print(f"  scanning day {i}/{len(year_dates)}")
        with xr.open_dataset(path, engine="netcdf4") as ds_day:
            arr = ds_day[band_name].values
            valid_any |= np.isfinite(arr)

    valid_flat = np.flatnonzero(valid_any.ravel())
    if valid_flat.size == 0:
        raise ValueError(f"No valid {band_name} pixels found in {check_year}.")

    n_sample = min(sample_size, valid_flat.size)
    rng = np.random.default_rng(seed)
    sample_flat = rng.choice(valid_flat, size=n_sample, replace=False)
    sample_y, sample_x = np.unravel_index(sample_flat, (y_size, x_size))
    print(f"sampled {n_sample} pixels for check_interpolation")

    if lon_grid is not None and lat_grid is not None:
        map_x = lon_grid[sample_y, sample_x]
        map_y = lat_grid[sample_y, sample_x]
        map_xlabel = "Longitude"
        map_ylabel = "Latitude"
        map_title_suffix = "lon/lat"
    else:
        map_x = x_coord_vals[sample_x] if x_coord_vals is not None else sample_x
        map_y = y_coord_vals[sample_y] if y_coord_vals is not None else sample_y
        map_xlabel = x_dim
        map_ylabel = y_dim
        map_title_suffix = f"{x_dim}/{y_dim}"

    series = np.full((len(proc_dates), n_sample), np.nan, dtype=np.float32)
    for i, dt in enumerate(proc_dates, start=1):
        path = proc_file_map.get(pd.Timestamp(dt))
        if path is None:
            continue
        if i % 60 == 0 or i == 1 or i == len(proc_dates):
            print(f"  loading sampled values day {i}/{len(proc_dates)}")
        with xr.open_dataset(path, engine="netcdf4") as ds_day:
            arr = ds_day[band_name].values
            series[i - 1, :] = arr[sample_y, sample_x]

    proc_months = pd.DatetimeIndex(proc_dates).month.values
    target_mask = (proc_dates >= pd.Timestamp(year_start)) & (proc_dates <= pd.Timestamp(year_end))
    raw_target = series[target_mask, :]
    target_months = proc_months[target_mask]

    may_mask = target_months == 5
    if np.any(may_mask):
        may_missing_counts = np.isnan(raw_target[may_mask, :]).sum(axis=0)
    else:
        may_missing_counts = np.zeros(n_sample, dtype=int)

    monthly_pct = {}
    for threshold in thresholds:
        print(f"  evaluating threshold={threshold} days")
        filled, _ = _interpolate_2d_time_gap(series, threshold)
        filled_target = filled[target_mask, :]
        monthly_vals = []
        for month in range(1, 13):
            m_mask = target_months == month
            if not np.any(m_mask):
                monthly_vals.append(np.nan)
                continue
            pct = 100.0 * np.isfinite(filled_target[m_mask, :]).mean()
            monthly_vals.append(pct)
        monthly_pct[threshold] = monthly_vals

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(12)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(thresholds))
    for offset, threshold in zip(offsets, thresholds):
        ax.bar(x + offset, monthly_pct[threshold], width=width, label=f"{threshold} days")
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels())
    ax.set_ylabel("Available sampled pixel-days (%)")
    ax.set_title(f"{check_year} MODIS availability after interpolation ({band_name})")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Max gap")
    fig.tight_layout()
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"saved interpolation check plot: {plot_path}")

    map_plot_path = Path(map_plot_path)
    map_plot_path.parent.mkdir(parents=True, exist_ok=True)
    if lon_grid is not None and lat_grid is not None:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from lfmc_model.utils.plotting import map_points

        map_points(
            lons=map_x,
            lats=map_y,
            counts_per_point=may_missing_counts,
            colors=may_missing_counts,
            cmap="viridis",
            colorbar_label="Missing days in May (raw data)",
            cbar_lim=(0, 31),
            save_path=str(map_plot_path),
            s_min=20,
            s_max=140,
            clip_quantiles=(0.0, 0.98),
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            map_x,
            map_y,
            c=may_missing_counts,
            s=18,
            cmap="viridis",
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Missing days in May (raw data)")
        ax.set_xlabel(map_xlabel)
        ax.set_ylabel(map_ylabel)
        ax.set_title(f"{check_year} May raw missingness for sampled pixels ({band_name}, {map_title_suffix})")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(map_plot_path, dpi=200)
        plt.close(fig)
    print(f"saved May raw missingness map: {map_plot_path}")


def plot_interpolation_diagnostics(
    start_date,
    end_date,
    base_path,
    output_zarr,
    plot_path,
    map_plot_path=None,
    band_name=None,
    n_points=5,
    seed=0,
):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from data_processing.shared.plotting import (
        plot_from_xarray,
        plot_interpolation_diagnostic_timeseries,
    )

    target_dates = pd.date_range(parse_date(start_date), parse_date(end_date), freq="D")
    if len(target_dates) == 0:
        raise ValueError("No target dates for diagnostic plotting.")

    print(f"opening zarr for diagnostics: {output_zarr}")
    ds_zarr = xr.open_zarr(output_zarr, consolidated=False)
    try:
        y_dim, x_dim = None, None
        for d in ["y", "lat", "latitude"]:
            if d in ds_zarr.dims:
                y_dim = d
                break
        for d in ["x", "lon", "longitude"]:
            if d in ds_zarr.dims:
                x_dim = d
                break
        if y_dim is None or x_dim is None:
            raise ValueError(f"Could not infer spatial dims from zarr dims={dict(ds_zarr.dims)}")

        if band_name is None:
            interp_candidates = [v for v in ds_zarr.data_vars if v.endswith("_interp")]
            if not interp_candidates:
                raise ValueError("No *_interp variables found in zarr.")
            band_interp_name = interp_candidates[0]
        else:
            band_interp_name = band_name if band_name.endswith("_interp") else f"{band_name}_interp"
            if band_interp_name not in ds_zarr.data_vars:
                raise KeyError(f"{band_interp_name} not found in zarr.")

        base_band = band_interp_name[:-7] if band_interp_name.endswith("_interp") else band_interp_name
        status_name = f"fill_status_{base_band}"
        if status_name not in ds_zarr.data_vars:
            raise KeyError(f"{status_name} not found in zarr.")

        interp_da = ds_zarr[band_interp_name]
        status_da = ds_zarr[status_name]
        if "time" in interp_da.dims:
            interp_da = interp_da.sel(time=target_dates)
            status_da = status_da.sel(time=target_dates)

        print(f"diagnostic band: {base_band}")
        rng = np.random.default_rng(seed)
        y_size = int(ds_zarr.sizes[y_dim])
        x_size = int(ds_zarr.sizes[x_dim])
        time_values = pd.DatetimeIndex(interp_da["time"].values)
        n_time = len(time_values)
        if n_time == 0:
            raise ValueError("No time values available for diagnostics.")

        # Fast diagnostics: find a few points that actually have interpolation events,
        # then plot only short windows around one event per point.
        max_events = max(1, int(n_points))
        window_radius = 30
        offsets = np.arange(-window_radius, window_radius + 1, dtype=int)
        window_len = len(offsets)
        max_attempts = max(200, max_events * 250)

        all_dates, file_map = build_daily_file_map(
            base_path,
            target_dates[0].to_pydatetime(),
            target_dates[-1].to_pydatetime(),
        )
        _ = all_dates  # retained for interface parity

        def _read_raw_point(ts, yi, xi):
            path = file_map.get(pd.Timestamp(ts))
            if path is None:
                return np.nan
            with xr.open_dataset(path, engine="netcdf4") as ds_day:
                day_da = ds_day[base_band]
                day_y_dim = next((d for d in ["y", "lat", "latitude"] if d in day_da.dims), None)
                day_x_dim = next((d for d in ["x", "lon", "longitude"] if d in day_da.dims), None)
                if day_y_dim is None or day_x_dim is None:
                    raise ValueError(
                        f"Could not infer spatial dims for {base_band} in {path}; dims={day_da.dims}"
                    )
                point = day_da.isel({day_y_dim: int(yi), day_x_dim: int(xi)}).values
                return float(np.asarray(point).squeeze())

        point_labels = []
        original_list = []
        interp_list = []
        status_list = []
        chosen_points = set()
        attempts = 0
        while len(point_labels) < max_events and attempts < max_attempts:
            attempts += 1
            yi = int(rng.integers(0, y_size))
            xi = int(rng.integers(0, x_size))
            if (yi, xi) in chosen_points:
                continue

            point_status = status_da.isel({y_dim: yi, x_dim: xi}).values
            event_indices = np.flatnonzero(point_status == 1)
            if event_indices.size == 0:
                continue

            chosen_points.add((yi, xi))
            event_idx = int(rng.choice(event_indices))
            event_time = pd.Timestamp(time_values[event_idx])
            point_interp = interp_da.isel({y_dim: yi, x_dim: xi}).values

            point_original_window = np.full(window_len, np.nan, dtype=np.float32)
            point_interp_window = np.full(window_len, np.nan, dtype=np.float32)
            point_status_window = np.full(window_len, 2, dtype=np.uint8)

            for j, off in enumerate(offsets):
                t_idx = event_idx + int(off)
                if t_idx < 0 or t_idx >= n_time:
                    continue
                ts = pd.Timestamp(time_values[t_idx])
                point_interp_window[j] = float(point_interp[t_idx])
                point_status_window[j] = int(point_status[t_idx])
                point_original_window[j] = _read_raw_point(ts, yi, xi)

            n_filled = int(event_indices.size)
            point_labels.append(
                f"Point {len(point_labels)+1}: y={yi}, x={xi}, event={event_time.date()} ({n_filled} filled days)"
            )
            original_list.append(point_original_window)
            interp_list.append(point_interp_window)
            status_list.append(point_status_window)

        if not point_labels:
            raise ValueError(f"No pixels with interpolated values found for {base_band} in requested dates.")

        plot_interpolation_diagnostic_timeseries(
            times=offsets,
            original_series_list=original_list,
            interpolated_series_list=interp_list,
            status_series_list=status_list,
            point_labels=point_labels,
            save_name=str(plot_path),
            title=f"Interpolation diagnostic: {base_band} (event-centered ±{window_radius} days)",
        )
        print(f"saved diagnostic plot: {plot_path}")

        if map_plot_path is not None:
            summer_mask = pd.DatetimeIndex(target_dates).month.isin([6, 7, 8])
            if np.any(summer_mask):
                summer_dates = target_dates[summer_mask]
                summer_date = summer_dates[len(summer_dates) // 2]
            else:
                summer_date = target_dates[len(target_dates) // 2]

            print(f"plotting diagnostic map for {summer_date.date()} ({base_band})")
            map_ds = ds_zarr[[band_interp_name]].sel(time=summer_date)
            plot_from_xarray(
                load_type="ds",
                type_obj=map_ds,
                var=band_interp_name,
                proj_in="EPSG:5070",
                proj_out="EPSG:5070",
                fname=str(map_plot_path),
                cmap="viridis",
            )
            print(f"saved diagnostic map plot: {map_plot_path}")
    finally:
        ds_zarr.close()


def build_output_template(
    output_zarr,
    ref_file,
    interp_vars,
    y_dim,
    x_dim,
    target_dates,
    xy_chunk_size,
    time_chunk_size,
    overwrite=False,
):
    output_zarr = Path(output_zarr)
    if output_zarr.exists() and overwrite:
        if output_zarr.is_dir():
            import shutil

            shutil.rmtree(output_zarr)
        else:
            output_zarr.unlink()
    elif output_zarr.exists() and not overwrite:
        raise FileExistsError(
            f"Output zarr store already exists: {output_zarr}. Use --overwrite_zarr to replace it."
        )

    print(f'Creating zarr store based on {ref_file}')
    with xr.open_dataset(ref_file, engine="netcdf4") as ref_ds:
        y_size = ref_ds.sizes[y_dim]
        x_size = ref_ds.sizes[x_dim]
        chunks = (
            min(time_chunk_size, len(target_dates)),
            min(xy_chunk_size, y_size),
            min(xy_chunk_size, x_size),
        )
        coords = {
            "time": ("time", pd.DatetimeIndex(target_dates)),
            y_dim: ref_ds[y_dim],
            x_dim: ref_ds[x_dim],
        }
        for coord_name in ["lat", "lon"]:
            if coord_name in ref_ds.coords:
                coords[coord_name] = ref_ds.coords[coord_name]

        # Write only coords/static metadata with xarray (small graph), then create
        # large interpolation arrays directly with the Zarr API so init does not
        # build a massive dask graph for every output chunk.
        coord_data_vars = {}
        if "spatial_ref" in ref_ds.data_vars:
            coord_data_vars["spatial_ref"] = ref_ds["spatial_ref"]

        coord_ds = xr.Dataset(coords=coords, data_vars=coord_data_vars, attrs=dict(ref_ds.attrs))
        coord_ds.attrs["interpolation_output"] = "time-gap-limited linear interpolation"
        coord_ds.to_zarr(output_zarr, mode="w", zarr_format=ZARR_VERSION)

        root = zarr.open_group(str(output_zarr), mode="a", zarr_format=ZARR_VERSION)
        shape = (len(target_dates), y_size, x_size)

        for var in interp_vars:
            interp_name = f"{var}_interp"
            status_name = f"fill_status_{var}"

            interp_arr = root.create_array(
                interp_name,
                shape=shape,
                chunks=chunks,
                dtype=np.float32,
                fill_value=np.nan,
                overwrite=False,
                dimension_names=("time", y_dim, x_dim),
            )
            interp_arr.attrs.update(dict(ref_ds[var].attrs))
            interp_arr.attrs["_ARRAY_DIMENSIONS"] = ["time", y_dim, x_dim]

            status_arr = root.create_array(
                status_name,
                shape=shape,
                chunks=chunks,
                dtype=np.uint8,
                fill_value=0,
                overwrite=False,
                dimension_names=("time", y_dim, x_dim),
            )
            status_arr.attrs.update(
                {
                    "description": "0=original or originally valid, 1=interpolated, 2=missing after interpolation",
                    "_ARRAY_DIMENSIONS": ["time", y_dim, x_dim],
                }
            )

    return output_zarr


def iter_spatial_slices(y_size, x_size, xy_chunk_size):
    for y0 in range(0, y_size, xy_chunk_size):
        y1 = min(y0 + xy_chunk_size, y_size)
        for x0 in range(0, x_size, xy_chunk_size):
            x1 = min(x0 + xy_chunk_size, x_size)
            yield slice(y0, y1), slice(x0, x1)


def iter_spatial_slices_with_index(y_size, x_size, xy_chunk_size):
    chunk_index = 0
    for ys, xs in iter_spatial_slices(y_size, x_size, xy_chunk_size):
        yield chunk_index, ys, xs
        chunk_index += 1


def get_processing_context(
    start_date,
    end_date,
    base_path,
    max_interpolation_days,
    buffer_days=None,
):
    dt_start = parse_date(start_date)
    dt_end = parse_date(end_date)
    if dt_end < dt_start:
        raise ValueError("end_date must be on or after start_date")

    if buffer_days is None:
        buffer_days = max_interpolation_days

    proc_start = dt_start - timedelta(days=buffer_days)
    proc_end = dt_end + timedelta(days=buffer_days)
    proc_dates, proc_file_map = build_daily_file_map(base_path, proc_start, proc_end)
    if len(proc_file_map) == 0:
        raise FileNotFoundError("No input files found for the processing range.")

    ref_file = find_reference_file(proc_file_map)
    with xr.open_dataset(ref_file, engine="netcdf4") as ref_ds:
        y_dim, x_dim, interp_vars = infer_dims_and_vars(ref_ds)
        y_size = ref_ds.sizes[y_dim]
        x_size = ref_ds.sizes[x_dim]

    target_dates = pd.date_range(dt_start, dt_end, freq="D")
    target_mask = (proc_dates >= pd.Timestamp(dt_start)) & (proc_dates <= pd.Timestamp(dt_end))
    target_indices = np.flatnonzero(target_mask)
    if target_indices.size == 0:
        raise ValueError("No target dates in requested range.")

    return {
        "dt_start": dt_start,
        "dt_end": dt_end,
        "buffer_days": buffer_days,
        "proc_dates": proc_dates,
        "proc_file_map": proc_file_map,
        "ref_file": ref_file,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "interp_vars": interp_vars,
        "y_size": y_size,
        "x_size": x_size,
        "target_dates": target_dates,
        "target_indices": target_indices,
    }


def print_chunk_plan(y_size, x_size, xy_chunk_size, num_workers):
    y_chunks = (y_size + xy_chunk_size - 1) // xy_chunk_size
    x_chunks = (x_size + xy_chunk_size - 1) // xy_chunk_size
    total_chunks = y_chunks * x_chunks
    print(f"chunk grid: {y_chunks} y-chunks x {x_chunks} x-chunks = {total_chunks} total chunks")
    if num_workers > 0:
        counts = [0] * num_workers
        for chunk_index in range(total_chunks):
            counts[chunk_index % num_workers] += 1
        print(f"num_workers={num_workers}, chunks per worker min/max={min(counts)}/{max(counts)}")
    return total_chunks


def process_interpolation_to_zarr(
    start_date,
    end_date,
    base_path,
    output_zarr,
    max_interpolation_days,
    buffer_days=None,
    xy_chunk_size=128,
    time_chunk_size=1,
    overwrite=False,
    mode="all",
    worker_id=0,
    num_workers=1,
    finalize_metadata=True,
    dry_run_chunk_plan=False,
    max_chunks_per_worker=None,
    zero_fill_skipped_chunks=False,
):
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")

    ctx = get_processing_context(
        start_date=start_date,
        end_date=end_date,
        base_path=base_path,
        max_interpolation_days=max_interpolation_days,
        buffer_days=buffer_days,
    )
    dt_start = ctx["dt_start"]
    dt_end = ctx["dt_end"]
    buffer_days = ctx["buffer_days"]
    proc_dates = ctx["proc_dates"]
    proc_file_map = ctx["proc_file_map"]
    ref_file = ctx["ref_file"]
    y_dim = ctx["y_dim"]
    x_dim = ctx["x_dim"]
    interp_vars = ctx["interp_vars"]
    y_size = ctx["y_size"]
    x_size = ctx["x_size"]
    target_dates = ctx["target_dates"]
    target_indices = ctx["target_indices"]

    print(f"processing range: {dt_start.date()} to {dt_end.date()}")
    print(f"buffer_days={buffer_days}, max_interpolation_days={max_interpolation_days}")
    print(f"proc dates={len(proc_dates)} (files present={len(proc_file_map)})")
    print(f"spatial dims: {y_dim}={y_size}, {x_dim}={x_size}")
    print(f"output chunking: time={time_chunk_size}, xy={xy_chunk_size}")
    print(f"interp vars ({len(interp_vars)}): {interp_vars}")
    total_chunks = print_chunk_plan(y_size, x_size, xy_chunk_size, num_workers)
    print(f"mode={mode}")
    if mode == "worker":
        print(f"worker_id={worker_id}, num_workers={num_workers}")
        if max_chunks_per_worker is not None:
            print(f"max_chunks_per_worker={max_chunks_per_worker}")
            print(f"zero_fill_skipped_chunks={zero_fill_skipped_chunks}")

    if dry_run_chunk_plan:
        return

    if mode in {"all", "init"}:
        output_zarr = build_output_template(
            output_zarr=output_zarr,
            ref_file=ref_file,
            interp_vars=interp_vars,
            y_dim=y_dim,
            x_dim=x_dim,
            target_dates=target_dates,
            xy_chunk_size=xy_chunk_size,
            time_chunk_size=time_chunk_size,
            overwrite=overwrite,
        )
        print(f"initialized zarr store: {output_zarr}")
        if mode == "init":
            return

    if mode == "finalize":
        zarr.consolidate_metadata(str(output_zarr))
        print(f"consolidated metadata: {output_zarr}")
        return

    if mode not in {"all", "worker"}:
        raise ValueError("mode must be one of: all, init, worker, finalize")

    if not Path(output_zarr).exists():
        raise FileNotFoundError(
            f"Output zarr store not found for worker mode: {output_zarr}. Run --mode init first."
        )

    if mode == "worker":
        remainder = total_chunks % num_workers
        assigned_total = (total_chunks // num_workers) + (1 if worker_id < remainder else 0)
    else:
        assigned_total = total_chunks

    assigned_chunks = 0
    processed_chunks = 0
    worker_start_time = time.monotonic()
    for chunk_index, ys, xs in iter_spatial_slices_with_index(y_size, x_size, xy_chunk_size):
        if mode == "worker" and (chunk_index % num_workers) != worker_id:
            continue
        assigned_chunks += 1
        y_len = ys.stop - ys.start
        x_len = xs.stop - xs.start
        should_process_chunk = True
        if (
            mode == "worker"
            and max_chunks_per_worker is not None
            and processed_chunks >= max_chunks_per_worker
        ):
            should_process_chunk = False

        elapsed_before = time.monotonic() - worker_start_time
        print(
            f"chunk {chunk_index + 1}/{total_chunks} "
            f"(worker chunk {assigned_chunks}/{assigned_total}): "
            f"{y_dim}[{ys.start}:{ys.stop}] {x_dim}[{xs.start}:{xs.stop}] "
            f"| elapsed={format_elapsed(elapsed_before)}"
        )

        region = {"time": slice(0, len(target_dates)), y_dim: ys, x_dim: xs}
        chunk_write_bytes = 0
        chunk_write_elapsed = 0.0
        if should_process_chunk:
            processed_chunks += 1
            chunk_process_start = time.monotonic()
            cubes = {
                var: np.full((len(proc_dates), y_len, x_len), np.nan, dtype=np.float32)
                for var in interp_vars
            }

            day_iter = maybe_tqdm(
                enumerate(proc_dates),
                total=len(proc_dates),
                desc="  read days",
                unit="day",
                leave=False,
            )
            for t_i, dt in day_iter:
                path = proc_file_map.get(pd.Timestamp(dt))
                if path is None:
                    continue
                if (t_i + 1) % 60 == 0 or t_i == 0 or t_i == len(proc_dates) - 1:
                    print(f"  reading day {t_i + 1}/{len(proc_dates)}")
                with xr.open_dataset(path, engine="netcdf4") as ds_day:
                    for var in interp_vars:
                        cubes[var][t_i, :, :] = ds_day[var].isel({y_dim: ys, x_dim: xs}).values.astype(
                            np.float32
                        )

            interp_start = time.monotonic()
            var_iter = maybe_tqdm(
                enumerate(interp_vars, start=1),
                total=len(interp_vars),
                desc="  interpolate vars",
                unit="var",
                leave=False,
            )
            for v, var in var_iter:
                print(f"  Processing variable {v}/{len(interp_vars)}: {var}")
                var_start = time.monotonic()
                flat = cubes[var].reshape(len(proc_dates), -1)
                filled_flat, status_flat = _interpolate_2d_time_gap(flat, max_interpolation_days)
                filled_target = filled_flat[target_indices, :].reshape(len(target_dates), y_len, x_len)
                status_target = status_flat[target_indices, :].reshape(len(target_dates), y_len, x_len)
                write_ds_var = xr.Dataset(
                    {
                        f"{var}_interp": (
                            ("time", y_dim, x_dim),
                            filled_target.astype(np.float32),
                        ),
                        f"fill_status_{var}": (
                            ("time", y_dim, x_dim),
                            status_target.astype(np.uint8),
                        ),
                    }
                )
                n_filled = int(np.sum(status_target == 1))
                interp_elapsed = time.monotonic() - var_start
                print(
                    f"    done {var} | filled_pixels_days={n_filled} "
                    f"| interp_elapsed={format_elapsed(interp_elapsed)}"
                )
                print(f"    writing {var} + fill_status_{var} to zarr")
                write_start = time.monotonic()
                with ProgressBar():
                    write_ds_var.to_zarr(
                        output_zarr,
                        mode="a",
                        region=region,
                        zarr_format=ZARR_VERSION,
                    )
                write_elapsed = time.monotonic() - write_start
                pair_bytes = 0
                for da in write_ds_var.data_vars.values():
                    pair_bytes += da.size * da.dtype.itemsize
                chunk_write_bytes += pair_bytes
                chunk_write_elapsed += write_elapsed
                pair_gib = pair_bytes / (1024 ** 3)
                pair_rate = (pair_gib / write_elapsed) if write_elapsed > 0 else float("inf")
                print(
                    f"    zarr write complete ({var}) | approx_data={pair_gib:.2f} GiB "
                    f"| write_elapsed={format_elapsed(write_elapsed)} "
                    f"| approx_rate={pair_rate:.2f} GiB/s"
                )
            print(
                f"  interpolation stage complete | elapsed={format_elapsed(time.monotonic() - interp_start)}"
            )
        elif zero_fill_skipped_chunks:
            print("  skipping interpolation for this chunk and writing zeros (debug mode)")
            zero_interp = np.zeros((len(target_dates), y_len, x_len), dtype=np.float32)
            zero_status = np.zeros((len(target_dates), y_len, x_len), dtype=np.uint8)
            zero_iter = maybe_tqdm(
                enumerate(interp_vars, start=1),
                total=len(interp_vars),
                desc="  write zero vars",
                unit="var",
                leave=False,
            )
            for v, var in zero_iter:
                print(f"  zero-write variable {v}/{len(interp_vars)}: {var}")
                write_ds_var = xr.Dataset(
                    {
                        f"{var}_interp": (("time", y_dim, x_dim), zero_interp),
                        f"fill_status_{var}": (("time", y_dim, x_dim), zero_status),
                    }
                )
                write_start = time.monotonic()
                with ProgressBar():
                    write_ds_var.to_zarr(
                        output_zarr,
                        mode="a",
                        region=region,
                        zarr_format=ZARR_VERSION,
                    )
                write_elapsed = time.monotonic() - write_start
                pair_bytes = 0
                for da in write_ds_var.data_vars.values():
                    pair_bytes += da.size * da.dtype.itemsize
                chunk_write_bytes += pair_bytes
                chunk_write_elapsed += write_elapsed
                pair_gib = pair_bytes / (1024 ** 3)
                pair_rate = (pair_gib / write_elapsed) if write_elapsed > 0 else float("inf")
                print(
                    f"    zarr zero-write complete ({var}) | approx_data={pair_gib:.2f} GiB "
                    f"| write_elapsed={format_elapsed(write_elapsed)} "
                    f"| approx_rate={pair_rate:.2f} GiB/s"
                )
        else:
            print("  skipping interpolation for this chunk (debug mode, no write)")
            elapsed_after = time.monotonic() - worker_start_time
            avg_time = elapsed_after / assigned_chunks
            eta = avg_time * max(assigned_total - assigned_chunks, 0)
            print(
                f"  progress: {assigned_chunks}/{assigned_total} chunks "
                f"| elapsed={format_elapsed(elapsed_after)} "
                f"| time_remaining={format_elapsed(eta)}"
            )
            continue

        if chunk_write_bytes > 0:
            write_gib = chunk_write_bytes / (1024 ** 3)
            rate = (write_gib / chunk_write_elapsed) if chunk_write_elapsed > 0 else float("inf")
            print(
                f"  chunk zarr writes complete | approx_data={write_gib:.2f} GiB "
                f"| write_elapsed={format_elapsed(chunk_write_elapsed)} | approx_rate={rate:.2f} GiB/s"
            )
        if should_process_chunk:
            chunk_total_elapsed = time.monotonic() - chunk_process_start
            print(f"  chunk processing total (interp+write)={format_elapsed(chunk_total_elapsed)}")
        elapsed_after = time.monotonic() - worker_start_time
        avg_time = elapsed_after / assigned_chunks
        eta = avg_time * max(assigned_total - assigned_chunks, 0)
        print(
            f"  progress: {assigned_chunks}/{assigned_total} chunks "
            f"| elapsed={format_elapsed(elapsed_after)} "
            f"| time_remaining={format_elapsed(eta)}"
        )

    print(f"worker assigned {assigned_chunks} chunks, processed {processed_chunks} chunks")

    if finalize_metadata:
        zarr.consolidate_metadata(str(output_zarr))
        print(f"finished writing zarr store: {output_zarr}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast gap-limited temporal interpolation for daily MODIS regridded files."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid",
        help="Base directory containing daily MODIS regridded files in YYYY/MM subfolders.",
    )
    parser.add_argument("--start_date", type=str, help="Target output start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, help="Target output end date (YYYY-MM-DD).")
    parser.add_argument(
        "--output_zarr",
        type=str,
        default="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_regrid_interpolated/modis_interpolated.zarr",
        help="Output zarr store path.",
    )
    parser.add_argument(
        "--max_interpolation_days",
        type=int,
        default=15,
        help="Fill only internal gaps with length <= this number of days.",
    )
    parser.add_argument(
        "--buffer_days",
        type=int,
        default=None,
        help="Buffer days around target range. Defaults to max_interpolation_days.",
    )
    parser.add_argument(
        "--xy_chunk_size",
        type=int,
        default=128,
        help="Spatial chunk size for processing and zarr writing.",
    )
    parser.add_argument(
        "--time_chunk_size",
        type=int,
        default=1,
        help="Time chunk size for output zarr arrays (reduces file count when >1).",
    )
    parser.add_argument(
        "--overwrite_zarr",
        action="store_true",
        help="Overwrite output zarr store if it exists.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "init", "worker", "finalize"],
        help="Execution mode for parallel-safe zarr writing.",
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="Worker index for --mode worker (0-based).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Total number of workers for --mode worker.",
    )
    parser.add_argument(
        "--dry_run_chunk_plan",
        action="store_true",
        help="Print chunk grid / worker assignment info and exit before writing.",
    )
    parser.add_argument(
        "--max_chunks_per_worker",
        type=int,
        default=None,
        help="Debug option: process only the first N assigned chunks for each worker.",
    )
    parser.add_argument(
        "--zero_fill_skipped_chunks",
        action="store_true",
        help="When using --max_chunks_per_worker, write zeros for remaining assigned chunks.",
    )
    parser.add_argument(
        "--check_interpolation",
        action="store_true",
        help="Run 2023 interpolation availability check plot (single-band sample) and exit unless dates are also provided.",
    )
    parser.add_argument(
        "--check_year",
        type=int,
        default=2023,
        help="Year for interpolation availability check plot.",
    )
    parser.add_argument(
        "--check_band",
        type=str,
        default=None,
        help="Band variable name to use for interpolation availability check.",
    )
    parser.add_argument(
        "--check_sample_size",
        type=int,
        default=1000,
        help="Number of pixels to sample for interpolation availability check.",
    )
    parser.add_argument(
        "--check_thresholds",
        type=int,
        nargs="+",
        default=[7, 15, 30, 60],
        help="Interpolation max-gap thresholds to compare in check mode.",
    )
    parser.add_argument(
        "--check_seed",
        type=int,
        default=0,
        help="Random seed for pixel sampling in check mode.",
    )
    parser.add_argument(
        "--check_plot_path",
        type=str,
        default="./data_processing/interpolate/interpolation_check_2023.png",
        help="Path for the check_interpolation monthly bar plot.",
    )
    parser.add_argument(
        "--check_map_plot_path",
        type=str,
        default="./data_processing/interpolate/interpolation_check_2023_may_missing_map.png",
        help="Path for the check_interpolation May raw missingness scatter map.",
    )
    parser.add_argument(
        "--plot_interpolation_diagnostics",
        action="store_true",
        help="Plot original vs interpolated time series for random pixels with interpolation.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Generate diagnostic plots from an existing output zarr and exit (no interpolation writes).",
    )
    parser.add_argument(
        "--diagnostic_plot_path",
        type=str,
        default="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/plots/interpolation_diagnostic.png",
        help="Output path for diagnostic interpolation plot.",
    )
    parser.add_argument(
        "--diagnostic_map_plot_path",
        type=str,
        default="/scratch/users/trobinet/long_lfmc/final_lfmc/modis/plots/interpolation_diagnostic_map.png",
        help="Output path for diagnostic spatial map plot (summer day if available).",
    )
    parser.add_argument(
        "--diagnostic_band",
        type=str,
        default=None,
        help="Band to diagnose (base band name or *_interp name). Default: first *_interp variable.",
    )
    parser.add_argument(
        "--diagnostic_n_points",
        type=int,
        default=5,
        help="Number of random interpolated pixels to plot in diagnostic output.",
    )
    parser.add_argument(
        "--diagnostic_seed",
        type=int,
        default=0,
        help="Random seed for diagnostic point sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.check_interpolation:
        run_check_interpolation(
            base_path=args.base_path,
            sample_size=args.check_sample_size,
            thresholds=sorted(set(args.check_thresholds)),
            plot_path=args.check_plot_path,
            map_plot_path=args.check_map_plot_path,
            check_band=args.check_band,
            check_year=args.check_year,
            seed=args.check_seed,
        )
        if not (args.start_date and args.end_date):
            return

    if not args.start_date or not args.end_date:
        raise ValueError("start_date and end_date are required for interpolation output.")

    if args.plot_interpolation_diagnostics or args.plot_only:
        plot_interpolation_diagnostics(
            start_date=args.start_date,
            end_date=args.end_date,
            base_path=args.base_path,
            output_zarr=args.output_zarr,
            plot_path=args.diagnostic_plot_path,
            map_plot_path=args.diagnostic_map_plot_path,
            band_name=args.diagnostic_band,
            n_points=args.diagnostic_n_points,
            seed=args.diagnostic_seed,
        )
        return

    process_interpolation_to_zarr(
        start_date=args.start_date,
        end_date=args.end_date,
        base_path=args.base_path,
        output_zarr=args.output_zarr,
        max_interpolation_days=args.max_interpolation_days,
        buffer_days=args.buffer_days,
        xy_chunk_size=args.xy_chunk_size,
        time_chunk_size=args.time_chunk_size,
        overwrite=args.overwrite_zarr,
        mode=args.mode,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        finalize_metadata=(args.mode == "all"),
        dry_run_chunk_plan=args.dry_run_chunk_plan,
        max_chunks_per_worker=args.max_chunks_per_worker,
        zero_fill_skipped_chunks=args.zero_fill_skipped_chunks,
    )

    # Always generate final diagnostic plots after a completed full/finalize run.
    # Do not run in init/worker/dry-run modes because the store is incomplete.
    if args.mode in {"all", "finalize"} and not args.dry_run_chunk_plan:
        try:
            plot_interpolation_diagnostics(
                start_date=args.start_date,
                end_date=args.end_date,
                base_path=args.base_path,
                output_zarr=args.output_zarr,
                plot_path=args.diagnostic_plot_path,
                map_plot_path=args.diagnostic_map_plot_path,
                band_name=args.diagnostic_band,
                n_points=args.diagnostic_n_points,
                seed=args.diagnostic_seed,
            )
        except Exception as exc:
            print(f"Warning: final diagnostic plotting failed: {exc}")


if __name__ == "__main__":
    main()
