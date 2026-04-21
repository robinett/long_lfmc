#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr


def saturation_vapor_pressure_pa(temp_c: xr.DataArray) -> xr.DataArray:
    temp_c = temp_c.astype(np.float32)
    return 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one Daymet year into a clim20 store using a saved climatology store."
    )
    parser.add_argument("--raw_archive_zarr", type=Path, required=True)
    parser.add_argument("--combined_zarr", type=Path, required=True)
    parser.add_argument("--climatology_zarr", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    return parser.parse_args()


def select_raw_var(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    out = ds["data"].sel(variable=var_name).drop_vars("variable", errors="ignore")
    out.name = var_name
    return out.astype(np.float32)


def select_clim_var(ds: xr.Dataset, var_name: str, month_day_indexer: xr.DataArray) -> xr.DataArray:
    out = ds["data"].sel(variable=var_name).sel(month_day=month_day_indexer)
    out = out.drop_vars("variable", errors="ignore").drop_vars("month_day", errors="ignore")
    out.name = var_name
    return out.astype(np.float32)


def build_temp_var_path(base_dir: Path, month_tag: str, var_name: str) -> Path:
    return base_dir / month_tag / f"{var_name}.zarr"


def month_windows(start_date: pd.Timestamp, end_date: pd.Timestamp):
    current = start_date.normalize()
    while current <= end_date:
        month_end = min(current + pd.offsets.MonthEnd(0), end_date)
        yield current, month_end
        current = (month_end + pd.Timedelta(days=1)).normalize()


def require_raw_year_time(raw_ds: xr.Dataset, year: int) -> xr.Dataset:
    expected_times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    raw_len = int(raw_ds.sizes.get("time", 0))
    if raw_len != len(expected_times):
        raise ValueError(
            f"Raw archive time length does not match expected year length: "
            f"year={year} raw_len={raw_len} expected_len={len(expected_times)}"
        )
    actual_times = pd.to_datetime(raw_ds["time"].values)
    is_expected = len(actual_times) == len(expected_times) and np.array_equal(
        actual_times.values,
        expected_times.values,
    )
    if not is_expected:
        diffs = np.diff(actual_times.values).astype("timedelta64[D]")
        raise ValueError(
            f"Raw archive time coordinate is invalid for year={year}: "
            f"first_actual={actual_times[0]} last_actual={actual_times[-1]} "
            f"expected_first={expected_times[0]} expected_last={expected_times[-1]} "
            f"first_diffs={diffs[:5]}"
        )
    return raw_ds


def get_existing_combined_last_time(combined_zarr: Path) -> pd.Timestamp:
    combined_ds = xr.open_zarr(combined_zarr, consolidated=False)
    last_time = pd.Timestamp(combined_ds["time"].values[-1])
    combined_ds.close()
    return last_time


def write_temp_var(
    temp_dir: Path,
    month_tag: str,
    var_name: str,
    data: xr.DataArray,
) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = build_temp_var_path(temp_dir, month_tag, var_name)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_path.exists():
        shutil.rmtree(temp_path)
    out = xr.Dataset(
        {var_name: data.chunk({"time": 32, "y": 512, "x": 512}).astype(np.float32)},
        coords={"time": data["time"], "y": data["y"], "x": data["x"]},
    )
    out[var_name].encoding = {}
    for coord_name in out.coords:
        out[coord_name].encoding = {}
    out.to_zarr(temp_path, mode="w", consolidated=False, safe_chunks=False)
    print(
        f"  wrote temp var={var_name} days={int(data.sizes['time'])} "
        f"chunks={out[var_name].chunksizes} path={temp_path}"
    )


def assemble_month_output(
    month_dir: Path,
    variable_order: list[str],
    coords_source: xr.Dataset,
) -> xr.Dataset:
    arrays = []
    for var_name in variable_order:
        temp_path = month_dir / f"{var_name}.zarr"
        ds = xr.open_zarr(temp_path, consolidated=False)
        arr = ds[var_name].expand_dims(variable=[var_name]).astype(np.float32)
        arrays.append(arr)
    out = xr.concat(
        arrays,
        dim="variable",
        coords="minimal",
        compat="override",
        join="exact",
    ).transpose("time", "variable", "y", "x")
    out = out.chunk({"time": 32, "variable": 1, "y": 512, "x": 512})
    return xr.Dataset(
        {"data": out},
        coords={
            "time": out["time"],
            "variable": out["variable"],
            "y": coords_source["y"],
            "x": coords_source["x"],
            "lat": coords_source["lat"],
            "lon": coords_source["lon"],
        },
    )


def main() -> None:
    args = parse_args()
    year = int(args.year)
    year_str = str(year)
    year_start = pd.Timestamp(f"{year_str}-01-01")
    year_end = pd.Timestamp(f"{year_str}-12-31")
    requested_start = pd.Timestamp(args.start_date) if args.start_date else year_start
    requested_end = pd.Timestamp(args.end_date) if args.end_date else year_end
    if requested_start < year_start or requested_end > year_end or requested_start > requested_end:
        raise ValueError(
            f"Requested range must fall within {year_str}: "
            f"start={requested_start.date()} end={requested_end.date()}"
        )

    raw_ds = xr.open_zarr(args.raw_archive_zarr, consolidated=False)
    clim_ds = xr.open_zarr(args.climatology_zarr, consolidated=False)
    combined_meta_ds = xr.open_zarr(args.combined_zarr, consolidated=False)
    variable_order = [str(val) for val in combined_meta_ds["variable"].values]
    temp_root = args.combined_zarr.parent / "append_daymet_monthly_tmp" / year_str

    existing_last_time = get_existing_combined_last_time(args.combined_zarr)
    if existing_last_time >= requested_end:
        print(
            f"{args.combined_zarr} already contains requested range through "
            f"{existing_last_time.date()}; nothing to append"
        )
        combined_meta_ds.close()
        raw_ds.close()
        clim_ds.close()
        return

    raw_ds = require_raw_year_time(raw_ds, year)
    year_raw = raw_ds.sel(time=slice(f"{year_str}-01-01", f"{year_str}-12-31"))
    if int(year_raw.sizes.get("time", 0)) == 0:
        raise ValueError(f"{args.raw_archive_zarr} does not contain year {year}")

    next_start = max(existing_last_time + pd.Timedelta(days=1), requested_start)
    if next_start > requested_end:
        print(
            f"{args.combined_zarr} already contains requested range through "
            f"{existing_last_time.date()}; nothing to append"
        )
        combined_meta_ds.close()
        raw_ds.close()
        clim_ds.close()
        return

    print(
        f"Appending Daymet clim20 year={year} range={next_start.date()} -> {requested_end.date()} "
        f"from raw={args.raw_archive_zarr} into combined={args.combined_zarr}"
    )
    print(f"Variable order: {variable_order}")

    months_appended = 0
    for chunk_start, chunk_end in month_windows(next_start, requested_end):
        month_tag = chunk_start.strftime("%Y-%m")
        print(
            f"Processing month={month_tag} start={chunk_start.date()} end={chunk_end.date()} "
            f"days={(chunk_end - chunk_start).days + 1}"
        )
        month_dir = temp_root / month_tag
        if month_dir.exists():
            shutil.rmtree(month_dir)
        month_raw = year_raw.sel(time=slice(chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        if int(month_raw.sizes.get("time", 0)) == 0:
            print(f"  skipping month={month_tag}; no raw days in requested slice")
            continue

        month_times = pd.to_datetime(month_raw["time"].values)
        month_day = xr.DataArray(
            month_times.month * 100 + month_times.day,
            dims=("time",),
            coords={"time": month_raw["time"]},
            name="month_day",
        )

        print("  computing tmax and tmax_daily_anom")
        tmax = select_raw_var(month_raw, "tmax")
        clim_tmax = select_clim_var(clim_ds, "tmax_daily_clim", month_day)
        write_temp_var(temp_root, month_tag, "tmax", tmax)
        write_temp_var(temp_root, month_tag, "tmax_daily_anom", (tmax - clim_tmax).astype(np.float32))

        print("  computing vpd and vpd_daily_anom")
        vp = select_raw_var(month_raw, "vp")
        vpd = (saturation_vapor_pressure_pa(tmax) - vp).clip(min=0.0).astype(np.float32)
        vpd.name = "vpd"
        clim_vpd = select_clim_var(clim_ds, "vpd_daily_clim", month_day)
        write_temp_var(temp_root, month_tag, "vpd", vpd)
        write_temp_var(temp_root, month_tag, "vpd_daily_anom", (vpd - clim_vpd).astype(np.float32))

        print("  computing prcp and prcp_rolling30_anom")
        combined_tail_ds = xr.open_zarr(args.combined_zarr, consolidated=False)
        prior_prcp = (
            combined_tail_ds["data"]
            .sel(variable="prcp")
            .drop_vars("variable", errors="ignore")
            .isel(time=slice(-29, None))
            .astype(np.float32)
        )
        prcp = select_raw_var(month_raw, "prcp")
        prcp_hist = xr.concat([prior_prcp, prcp], dim="time")
        prcp_rolling30 = prcp_hist.rolling(time=30, min_periods=30).sum().isel(
            time=slice(-int(prcp.sizes["time"]), None)
        ).astype(np.float32)
        prcp_rolling30 = prcp_rolling30.assign_coords(time=prcp["time"])
        clim_prcp_roll30 = select_clim_var(clim_ds, "prcp_rolling30_daily_clim", month_day)
        write_temp_var(temp_root, month_tag, "prcp", prcp)
        write_temp_var(
            temp_root,
            month_tag,
            "prcp_rolling30_anom",
            (prcp_rolling30 - clim_prcp_roll30).astype(np.float32),
        )
        combined_tail_ds.close()

        print("  computing srad and srad_daily_anom")
        srad = select_raw_var(month_raw, "srad")
        clim_srad = select_clim_var(clim_ds, "srad_daily_clim", month_day)
        write_temp_var(temp_root, month_tag, "srad", srad)
        write_temp_var(temp_root, month_tag, "srad_daily_anom", (srad - clim_srad).astype(np.float32))

        print("  computing swe and swe_daily_anom")
        swe = select_raw_var(month_raw, "swe")
        clim_swe = select_clim_var(clim_ds, "swe_daily_clim", month_day)
        write_temp_var(temp_root, month_tag, "swe", swe)
        write_temp_var(temp_root, month_tag, "swe_daily_anom", (swe - clim_swe).astype(np.float32))

        print(f"  assembling monthly output for month={month_tag}")
        month_out_ds = assemble_month_output(month_dir, variable_order, combined_meta_ds)
        month_out_ds.to_zarr(
            args.combined_zarr,
            mode="a",
            append_dim="time",
            consolidated=False,
            safe_chunks=False,
        )
        zarr.consolidate_metadata(str(args.combined_zarr))
        months_appended += 1
        new_last = get_existing_combined_last_time(args.combined_zarr)
        print(
            f"Completed month={month_tag}; appended_days={int(month_raw.sizes['time'])} "
            f"new_combined_max_date={new_last.date()}"
        )
        shutil.rmtree(month_dir)

    if temp_root.exists() and not any(temp_root.iterdir()):
        temp_root.rmdir()

    combined_meta_ds.close()
    raw_ds.close()
    clim_ds.close()
    print(
        f"Finished Daymet clim20 append for year={year}; months_appended={months_appended} "
        f"combined_zarr={args.combined_zarr}"
    )


if __name__ == "__main__":
    main()
