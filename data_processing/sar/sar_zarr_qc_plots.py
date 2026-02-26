#!/usr/bin/env python3
"""Make quick SAR zarr QC maps: one random valid date per variable."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import xarray as xr
sys.path.append(str(Path(__file__).resolve().parents[2]))
from data_processing.shared.plotting import (
    choose_random_valid_time_index_stacked,
    plot_from_xarray,
)


def plot_var_date(ds: xr.Dataset, var_name: str, out_dir: Path, rng: np.random.Generator):
    da = ds["data"].sel(variable=var_name)
    tidx = choose_random_valid_time_index_stacked(da, rng, sample_stride=256)
    if tidx is None:
        print(f"Skipping {var_name}: no valid data found")
        return

    tval = np.datetime_as_string(ds["time"].values[tidx], unit="D")
    slab = da.isel(time=tidx).rename(var_name)

    out_path = out_dir / f"{var_name}_{tval}.png"
    plot_from_xarray(
        load_type="da",
        type_obj=slab,
        var=var_name,
        proj_in="EPSG:5070",
        proj_out="EPSG:5070",
        fname=str(out_path),
        cmap="viridis",
        title=f"{var_name} on {tval}",
    )
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(args.zarr_path, consolidated=False)
    try:
        rng = np.random.default_rng(args.seed)
        if "data" not in ds or "variable" not in ds.coords:
            raise ValueError("Expected stacked zarr with data variable and variable coord")
        var_names = [str(v) for v in ds["variable"].values.tolist()]
        print("QC plotting variables:", var_names)
        for var_name in var_names:
            plot_var_date(ds, var_name, args.out_dir, rng)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
