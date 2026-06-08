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


def ensure_xy_match(left: xr.DataArray, right: xr.DataArray, label: str) -> None:
    for coord_name in ["x", "y"]:
        if coord_name not in left.coords or coord_name not in right.coords:
            raise ValueError(f"Cannot verify {label}: missing {coord_name} coordinate")
        if not np.array_equal(left[coord_name].values, right[coord_name].values):
            raise ValueError(f"{label} has incompatible {coord_name} coordinates")


def align_to_reference_grid(da: xr.DataArray, reference: xr.Dataset, label: str) -> xr.DataArray:
    out = da
    for coord_name in ["x", "y"]:
        if coord_name not in out.coords or coord_name not in reference.coords:
            raise ValueError(f"Cannot align {label}: missing {coord_name} coordinate")
        source_values = out[coord_name].values
        target_values = reference[coord_name].values
        if np.array_equal(source_values, target_values):
            continue
        if source_values.shape != target_values.shape or set(source_values.tolist()) != set(target_values.tolist()):
            raise ValueError(f"Cannot align {label}: incompatible {coord_name} coordinates")
        out = out.reindex({coord_name: target_values})
    for coord_name in ["x", "y"]:
        if not np.array_equal(out[coord_name].values, reference[coord_name].values):
            raise ValueError(f"Failed to align {label}: incompatible {coord_name} coordinates")
    return out


def drop_aux_spatial_coords(da: xr.DataArray) -> xr.DataArray:
    return da.drop_vars(["lat", "lon", "spatial_ref"], errors="ignore")


def truncate_store_before_date(combined_zarr: Path, combined_ds: xr.Dataset, start_date: pd.Timestamp) -> None:
    times = pd.to_datetime(combined_ds["time"].values).normalize()
    prefix_count = int(np.searchsorted(times.values, start_date.to_datetime64(), side="left"))
    print(
        f"Truncating combined low-latency/weather store before {start_date.date()}; "
        f"keeping {prefix_count} time steps"
    )
    root = zarr.open_group(str(combined_zarr), mode="a")
    root["time"].resize((prefix_count,))
    root["data"].resize(
        (
            prefix_count,
            root["data"].shape[1],
            root["data"].shape[2],
            root["data"].shape[3],
        )
    )


def monthly_spans(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans = []
    current = pd.Timestamp(start_date).normalize()
    final = pd.Timestamp(end_date).normalize()
    while current <= final:
        month_end = current + pd.offsets.MonthEnd(0)
        chunk_end = min(pd.Timestamp(month_end).normalize(), final)
        spans.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)
    return spans


def append_chunk(args: argparse.Namespace, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    combined_ds = xr.open_zarr(args.combined_zarr, consolidated=False)
    standard_ds = xr.open_zarr(args.standard_zarr, consolidated=False)
    clim_ds = xr.open_zarr(args.climatology_zarr, consolidated=False)

    try:
        print(f"Appending combined weather chunk {start_date.date()} -> {end_date.date()}")
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

        prcp = align_to_reference_grid(select_standard_var(standard_range, "prcp"), combined_ds, "prcp")
        srad = align_to_reference_grid(select_standard_var(standard_range, "srad"), combined_ds, "srad")
        swe = align_to_reference_grid(select_standard_var(standard_range, "swe"), combined_ds, "swe")
        tmax = align_to_reference_grid(select_standard_var(standard_range, "tmax"), combined_ds, "tmax")
        vp = align_to_reference_grid(select_standard_var(standard_range, "vp"), combined_ds, "vp")
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

        ensure_xy_match(prior_prcp, prcp, "Precipitation rolling-history concat")
        prcp_hist = xr.concat(
            [drop_aux_spatial_coords(prior_prcp), drop_aux_spatial_coords(prcp)],
            dim="time",
            compat="override",
            coords="minimal",
            join="exact",
        )
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
        arrays = [
            drop_aux_spatial_coords(derived[var_name]).expand_dims(variable=[var_name])
            for var_name in variable_order
        ]
        out = xr.concat(arrays, dim="variable", coords="minimal", compat="override").transpose(
            "time", "variable", "y", "x"
        )
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
        print(
            f"Appended low-latency combined climate/weather chunk "
            f"{start_date.date()} -> {end_date.date()} into {args.combined_zarr}"
        )
    finally:
        combined_ds.close()
        standard_ds.close()
        clim_ds.close()


def verify_requested_range(combined_zarr: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    ds = xr.open_zarr(combined_zarr, consolidated=False)
    try:
        times = pd.to_datetime(ds["time"].values).normalize()
        requested = pd.date_range(start_date, end_date, freq="D")
        missing = requested.difference(pd.DatetimeIndex(times))
        if len(missing) > 0:
            raise RuntimeError(
                f"Combined weather store is missing {len(missing)} requested dates; "
                f"first missing={missing[0].date()}"
            )
        print(
            f"Verified combined weather store covers "
            f"{start_date.date()} -> {end_date.date()}"
        )
    finally:
        ds.close()


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    if end_date < start_date:
        raise ValueError(f"end_date {end_date.date()} is before start_date {start_date.date()}")

    spans = monthly_spans(start_date, end_date)
    print(
        f"Appending low-latency combined climate/weather range "
        f"{start_date.date()} -> {end_date.date()} in {len(spans)} monthly chunk(s)"
    )
    for index, (chunk_start, chunk_end) in enumerate(spans, start=1):
        print(f"Monthly chunk {index}/{len(spans)}")
        append_chunk(args, chunk_start, chunk_end)

    verify_requested_range(args.combined_zarr, start_date, end_date)
    zarr.consolidate_metadata(str(args.combined_zarr))
    print(
        f"Appended low-latency combined climate/weather range "
        f"{start_date.date()} -> {end_date.date()} into {args.combined_zarr}"
    )


if __name__ == "__main__":
    main()
