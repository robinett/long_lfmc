import re
import numpy as np
import xarray as xr
import glob
import sys
import pandas as pd

# extract time from filename
def date_from_path(p): 
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
    return np.datetime64(date_re.search(p).group(1))

def preprocess(ds, path=None):
    # rename 'band_data' -> 'lfmc'
    ds = ds.rename({"band_data": "lfmc"})
    # attach time from filename
    if path is not None:
        t = date_from_path(path)
        ds = ds.expand_dims({"time": [t]})
    return ds

def sample_lfmc_points_non_nan(
    ds,
    num_to_sample=1_000_000,
    seed=42,
    batch_size=200_000,
    max_draws_multiplier=100, # e.g. if 100, function exist if it finds less than 1% valid points
):
    """
    Sample exactly `num_to_sample` valid (non-NaN) points across (time,y,x),
    without replacement among valid points. Works without .vindex.
    """
    # ensure consistent dim order
    da = ds["lfmc"].transpose("time", "y", "x")
    T, Y, X = (int(da.sizes[d]) for d in ("time", "y", "x"))
    N = T * Y * X
    if num_to_sample > N:
        raise ValueError(f"num_to_sample={num_to_sample} exceeds total cells={N}")

    rng = np.random.default_rng(seed)
    sel_flat = set()
    times_list, ys_list, xs_list = [], [], []
    lfmc_list, lat_list, lon_list = [], [], []

    tyx_per_t = Y * X
    def flat_to_tyx(flat_idx):
        t_idx = flat_idx // tyx_per_t
        rem   = flat_idx %  tyx_per_t
        y_idx = rem // X
        x_idx = rem %  X
        return t_idx, y_idx, x_idx

    total_draws_cap = max_draws_multiplier * num_to_sample
    total_draws = 0

    # helper: wrap numpy arrays so xarray does vectorized point indexing
    def idx_da(a): return xr.DataArray(a, dims="points")

    while len(sel_flat) < num_to_sample:
        remaining = num_to_sample - len(sel_flat)
        draw_n = min(batch_size, remaining * 4)  # oversample
        flat = rng.integers(0, N, size=draw_n)   # with replacement
        total_draws += draw_n
        if total_draws > total_draws_cap:
            raise RuntimeError(
                f"Too many draws ({total_draws}) for {len(sel_flat)} valid samples. "
                "NaN fraction may be very high; raise max_draws_multiplier or lower target."
            )

        # keep only new unique candidates
        unique_flat = np.unique(flat)
        mask_new = ~np.isin(unique_flat, list(sel_flat))
        new_flat = unique_flat[mask_new]
        if new_flat.size == 0:
            continue

        # map to (t,y,x)
        t_idx, y_idx, x_idx = flat_to_tyx(new_flat)

        # vectorized point selection WITHOUT .vindex
        lfmc_vals = da.isel(time=idx_da(t_idx), y=idx_da(y_idx), x=idx_da(x_idx)).values
        valid_mask = np.isfinite(lfmc_vals)
        if not valid_mask.any():
            continue

        # keep only valid subset (and cap to what's still needed)
        take_flat = new_flat[valid_mask]
        t_take    = t_idx[valid_mask]
        y_take    = y_idx[valid_mask]
        x_take    = x_idx[valid_mask]
        lfmc_take = lfmc_vals[valid_mask]

        k = min(remaining, take_flat.size)
        take_flat = take_flat[:k]
        t_take    = t_take[:k]
        y_take    = y_take[:k]
        x_take    = x_take[:k]
        lfmc_take = lfmc_take[:k]

        # lat/lon are (y,x)-only
        lat_take = ds["lat"].isel(y=idx_da(y_take), x=idx_da(x_take)).values
        lon_take = ds["lon"].isel(y=idx_da(y_take), x=idx_da(x_take)).values
        time_vals = ds["time"].values[t_take]

        sel_flat.update(take_flat.tolist())
        times_list.append(time_vals)
        ys_list.append(y_take)
        xs_list.append(x_take)
        lfmc_list.append(lfmc_take)
        lat_list.append(lat_take)
        lon_list.append(lon_take)
    out = pd.DataFrame({
        "date": np.concatenate(times_list),
        "latitude": np.concatenate(lat_list),
        "longitude": np.concatenate(lon_list),
        "lfmc": np.concatenate(lfmc_list),
    })
    # randomize the order of the df
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    assert len(out) == num_to_sample
    return out

def main():
    # glob pattern to pick up everything
    paths = sorted(
        glob.glob("/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/krishna_regrid/*/*/*.nc4")
    )
    # build dataset lazily with Dask
    print("Opening datasets...")
    for p,pth in enumerate(paths):
        print(f'Opening dataset {p+1} of {len(paths)}: {pth}')
        this_ds = xr.open_dataset(
            pth,
            engine='netcdf4',
        )
        this_ds = preprocess(this_ds, path=pth)
        if p == 0:
            ds = this_ds
        else:
            ds = xr.concat([ds, this_ds], dim="time")
    sampled_lfmc = sample_lfmc_points_non_nan(ds, num_to_sample=500_000, seed=42)
    sampled_lfmc.to_csv(
        "/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/krishna_lfmc_samples.csv",
        index=False
    )

if __name__ == "__main__":
    main()