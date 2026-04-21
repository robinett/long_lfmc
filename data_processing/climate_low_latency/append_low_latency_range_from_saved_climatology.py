#!/usr/bin/env python3

import argparse
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
        description=(
            "Append a low-latency climate date range into a combined clim20-style store "
            "using the saved Daymet climatology."
        )
    )
    parser.add_argument("--standard_zarr", type=Path, required=True)
    parser.add_argument("--combined_zarr", type=Path, required=True)
    parser.add_argument("--climatology_zarr", type=Path, required=True)
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    return parser.parse_args()


def select_standard_var(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    out = ds["data"].sel(variable=var_name).drop_vars("variable", errors="ignore")
    out.name = var_name
    return out.astype(np.float32)


def select_clim_var(ds: xr.Dataset, var_name: str, month_day_indexer: xr.DataArray) -> xr.DataArray:
    out = ds["data"].sel(variable=var_name).sel(month_day=month_day_indexer)
    out = out.drop_vars("variable", errors="ignore").drop_vars("month_day", errors="ignore")
    out.name = var_name
    return out.astype(np.float32)


def truncate_store_before_date(combined_zarr: Path, combined_ds: xr.Dataset, start_date: pd.Timestamp) -> None:
    times = pd.to_datetime(combined_ds["time"].values).normalize()
    prefix_count = int(np.searchsorted(times.values, start_date.to_datetime64(), side="left"))
    print(
        f"Truncating combined low-latency/weather store before {start_date.date()}; "
        f"keeping {prefix_count} time steps"
    )
    root = zarr.open_group(str(combined_zarr), mode="a")
    root["time"].resize(prefix_count)
    root["data"].resize(
        prefix_count,
        root["data"].shape[1],
        root["data"].shape[2],
        root["data"].shape[3],
    )


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    if end_date < start_date:
        raise ValueError(f"end_date {end_date.date()} is before start_date {start_date.date()}")

    combined_ds = xr.open_zarr(args.combined_zarr, consolidated=False)
    standard_ds = xr.open_zarr(args.standard_zarr, consolidated=False)
    clim_ds = xr.open_zarr(args.climatology_zarr, consolidated=False)

    try:
        standard_range = standard_ds.sel(time=slice(str(start_date.date()), str(end_date.date())))
        if int(standard_range.sizes.get("time", 0)) == 0:
            raise ValueError(
                f"{args.standard_zarr} does not contain requested range "
                f"{start_date.date()} -> {end_date.date()}"
            )

        combined_times = pd.to_datetime(combined_ds["time"].values).normalize()
        if len(combined_times) > 0 and combined_times.max() >= start_date:
            truncate_store_before_date(args.combined_zarr, combined_ds, start_date)
            combined_ds.close()
            combined_ds = xr.open_zarr(args.combined_zarr, consolidated=False)
            combined_times = pd.to_datetime(combined_ds["time"].values).normalize()

        prcp = select_standard_var(standard_range, "prcp")
        srad = select_standard_var(standard_range, "srad")
        swe = select_standard_var(standard_range, "swe")
        tmax = select_standard_var(standard_range, "tmax")
        vp = select_standard_var(standard_range, "vp")
        vpd = (saturation_vapor_pressure_pa(tmax) - vp).clip(min=0.0).astype(np.float32)
        vpd.name = "vpd"

        month_day = xr.DataArray(
            pd.to_datetime(standard_range["time"].values).month * 100
            + pd.to_datetime(standard_range["time"].values).day,
            dims=("time",),
            coords={"time": standard_range["time"]},
            name="month_day",
        )

        clim_tmax = select_clim_var(clim_ds, "tmax_daily_clim", month_day)
        clim_vpd = select_clim_var(clim_ds, "vpd_daily_clim", month_day)
        clim_prcp_roll30 = select_clim_var(clim_ds, "prcp_rolling30_daily_clim", month_day)
        clim_srad = select_clim_var(clim_ds, "srad_daily_clim", month_day)
        clim_swe = select_clim_var(clim_ds, "swe_daily_clim", month_day)

        prior_prcp = (
            combined_ds["data"]
            .sel(variable="prcp")
            .drop_vars("variable", errors="ignore")
            .sel(time=slice(None, str((start_date - pd.Timedelta(days=1)).date())))
            .isel(time=slice(-29, None))
            .astype(np.float32)
        )
        if int(prior_prcp.sizes.get("time", 0)) < 29:
            raise ValueError(
                "Need at least 29 prior precipitation days in combined store to compute "
                f"rolling anomalies for {start_date.date()}"
            )

        prcp_hist = xr.concat([prior_prcp, prcp], dim="time")
        prcp_rolling30 = (
            prcp_hist.rolling(time=30, min_periods=30)
            .sum()
            .isel(time=slice(-int(prcp.sizes["time"]), None))
            .astype(np.float32)
        )
        prcp_rolling30 = prcp_rolling30.assign_coords(time=prcp["time"])

        derived = {
            "tmax": tmax,
            "vpd": vpd,
            "prcp": prcp,
            "srad": srad,
            "swe": swe,
            "tmax_daily_anom": (tmax - clim_tmax).astype(np.float32),
            "vpd_daily_anom": (vpd - clim_vpd).astype(np.float32),
            "prcp_rolling30_anom": (prcp_rolling30 - clim_prcp_roll30).astype(np.float32),
            "srad_daily_anom": (srad - clim_srad).astype(np.float32),
            "swe_daily_anom": (swe - clim_swe).astype(np.float32),
        }

        variable_order = [str(val) for val in combined_ds["variable"].values]
        arrays = [derived[var_name].expand_dims(variable=[var_name]) for var_name in variable_order]
        out = xr.concat(arrays, dim="variable").transpose("time", "variable", "y", "x")
        out = out.chunk({"time": 32, "variable": 1, "y": 512, "x": 512})
        out_ds = xr.Dataset(
            {"data": out},
            coords={
                "time": out["time"],
                "variable": out["variable"],
                "y": combined_ds["y"],
                "x": combined_ds["x"],
                "lat": combined_ds["lat"],
                "lon": combined_ds["lon"],
            },
        )

        out_ds.to_zarr(
            args.combined_zarr,
            mode="a",
            append_dim="time",
            consolidated=False,
            safe_chunks=False,
        )
        zarr.consolidate_metadata(str(args.combined_zarr))
        print(
            f"Appended low-latency combined climate/weather range "
            f"{start_date.date()} -> {end_date.date()} into {args.combined_zarr}"
        )
    finally:
        combined_ds.close()
        standard_ds.close()
        clim_ds.close()


if __name__ == "__main__":
    main()
