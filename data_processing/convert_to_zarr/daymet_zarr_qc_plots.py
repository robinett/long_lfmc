#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr
sys.path.append(str(Path(__file__).resolve().parents[2]))
from data_processing.shared.plotting import plot_from_xarray


def parse_args():
    ap = argparse.ArgumentParser("Create random-date QC maps from Daymet zarr")
    ap.add_argument("--zarr", type=str, required=True, help="Final Daymet zarr store path")
    ap.add_argument("--out-dir", type=str, required=True, help="Directory to save QC plots")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print(f"Opening zarr: {args.zarr}")
    ds = xr.open_zarr(args.zarr, consolidated=False)
    try:
        if "data" not in ds.data_vars:
            raise KeyError("Expected 'data' variable in zarr store")
        if "time" not in ds.dims or "variable" not in ds.dims:
            raise KeyError("Expected time and variable dimensions in zarr store")

        times = pd.to_datetime(ds["time"].values)
        var_names = [str(v) for v in ds["variable"].values]
        if len(times) == 0:
            raise ValueError("No time steps found in zarr store")

        for var_idx, var_name in enumerate(var_names):
            t_idx = int(rng.integers(0, len(times)))
            date = pd.Timestamp(times[t_idx]).strftime("%Y-%m-%d")
            da = ds["data"].isel(time=t_idx, variable=var_idx).rename(var_name)

            out_path = out_dir / f"daymet_qc_{var_name}_{date}.png"
            plot_from_xarray(
                load_type="da",
                type_obj=da,
                var=var_name,
                proj_in="EPSG:5070",
                proj_out="EPSG:5070",
                fname=str(out_path),
                cmap="viridis",
                title=f"Daymet QC: {var_name} on {date}",
            )
            print(f"Saved QC plot: {out_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
