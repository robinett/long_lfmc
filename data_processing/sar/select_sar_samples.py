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

def sample_by_pixels_non_nan(
    ds: xr.Dataset,
    var_to_sample,
    num_to_sample: int = 1_000_000,
    seed: int = 42,
    pixel_batch_size: int = 10_000,   # number of pixels to evaluate per batch
    max_pixel_draws_multiplier: int = 50,  # safety cap on draws vs requested samples
):
    """
    Randomly sample unique pixels (y,x). For each selected pixel, take *all* non-NaN
    LFMC time steps at that pixel, then move to the next pixel. Continue until we've
    collected exactly `num_to_sample` non-NaN observations (truncating the last pixel
    if needed). Counts only non-NaN entries toward the total.

    Returns a pandas.DataFrame with columns:
        ['date', 'latitude', 'longitude', 'lfmc']
    """
    # Ensure consistent dimension order
    da = ds[var_to_sample].transpose("time", "y", "x")
    T, Y, X = (int(da.sizes[d]) for d in ("time", "y", "x"))
    total_pixels = Y * X

    if num_to_sample <= 0:
        return pd.DataFrame(columns=["date", "latitude", "longitude", var_to_sample])

    rng = np.random.default_rng(seed)
    time_vals_full = ds["time"].values

    # Helper to make xarray do vectorized point indexing
    def idx_da(a): 
        return xr.DataArray(a, dims="points")

    # Track which pixels we've already tried (flatten yx -> y*X + x)
    selected_pixels = set()

    # Output buffers
    dates_buf = []
    lats_buf  = []
    lons_buf  = []
    var_buf  = []

    # Draw safety
    # This caps how many pixel *draws* we're willing to try
    # before concluding that valid coverage is too sparse.
    max_pixel_draws = max_pixel_draws_multiplier * int(np.ceil(num_to_sample / max(1, T)))
    pixel_draws = 0

    # Helper: map flat yx -> (y, x)
    def flat_to_yx(flat_idx):
        y_idx = flat_idx // X
        x_idx = flat_idx %  X
        return y_idx, x_idx

    # Main loop: keep drawing pixels until we hit target or exhaust options
    total_kept = 0
    while total_kept < num_to_sample:
        print(f"Collected {total_kept}/{num_to_sample} samples so far...")
        # If we've considered all pixels, bail out
        if len(selected_pixels) >= total_pixels:
            raise RuntimeError(
                f"Ran out of pixels ({len(selected_pixels)}/{total_pixels}) "
                f"with total_kept={total_kept} < num_to_sample={num_to_sample}. "
                "Coverage may be too sparse."
            )

        # Draw a batch of candidate pixels (with replacement), then uniquify
        # and remove those we've already attempted.
        draw_n = min(
            pixel_batch_size,
            total_pixels - len(selected_pixels)
        )
        # Overdraw a bit to counter all-NaN pixels
        draw_n = max(draw_n, 1)
        flat = rng.integers(0, total_pixels, size=draw_n * 2)
        pixel_draws += flat.size
        if pixel_draws > max_pixel_draws * pixel_batch_size:
            raise RuntimeError(
                f"Too many pixel draws ({pixel_draws}) for {total_kept} collected samples. "
                "Data may be very sparse; increase max_pixel_draws_multiplier or lower target."
            )

        # Keep only new pixels
        unique_flat = np.unique(flat)
        mask_new = ~np.isin(unique_flat, list(selected_pixels))
        new_pix_flat = unique_flat[mask_new]
        if new_pix_flat.size == 0:
            continue

        # Convert to y/x arrays
        y_idx, x_idx = flat_to_yx(new_pix_flat)

        # Grab LFMC for all time at these pixels -> shape (T, P)
        vals = da.isel(y=idx_da(y_idx), x=idx_da(x_idx)).values  # (T, P)
        valid_mask = np.isfinite(vals)                            # (T, P)

        # For each pixel (column), collect all non-NaN times
        # Stop early if we reach num_to_sample.
        P = new_pix_flat.size
        for p in range(P):
            pix_flat = int(new_pix_flat[p])
            if pix_flat in selected_pixels:
                continue  # shouldn't happen, but cheap guard

            t_valid = np.nonzero(valid_mask[:, p])[0]
            if t_valid.size == 0:
                # Mark pixel as checked (all-NaN) and move on
                selected_pixels.add(pix_flat)
                continue

            # Values and times for this pixel
            var_p = vals[t_valid, p]
            dates_p = time_vals_full[t_valid]

            # Lat/Lon for this pixel (scalar each, repeat to match t_valid)
            y_p, x_p = flat_to_yx(pix_flat)
            lat_p = ds["lat"].values[y_p, x_p]
            lon_p = ds["lon"].values[y_p, x_p]

            remain = num_to_sample - total_kept
            if t_valid.size > remain:
                # Truncate this last pixel to hit target exactly
                t_valid = t_valid[:remain]
                var_p = var_p[:remain]
                dates_p = dates_p[:remain]

            dates_buf.append(dates_p)
            var_buf.append(var_p)
            # Repeat scalars per valid time
            lats_buf.append(np.full_like(var_p, fill_value=lat_p, dtype=float))
            lons_buf.append(np.full_like(var_p, fill_value=lon_p, dtype=float))

            total_kept += var_p.size
            selected_pixels.add(pix_flat)

            if total_kept >= num_to_sample:
                break  # done for this outer batch
        # continue outer while until we hit target
    # Build DataFrame
    out = pd.DataFrame({
        "date":      np.concatenate(dates_buf),
        "latitude":  np.concatenate(lats_buf),
        "longitude": np.concatenate(lons_buf),
        var_to_sample:  np.concatenate(var_buf),
    })
    assert len(out) == num_to_sample, (len(out), num_to_sample)
    return out


def main():
    # glob pattern to pick up everything
    sar_ds = xr.open_zarr(
        "/scratch/users/trobinet/long_lfmc/trent_datasets/sar/sar_formatted.zarr",
        chunks="auto",
    )
    vars_to_sample = ['VV', 'VH', 'vv_minus_vh']
    for var in vars_to_sample:
        print(f"Sampling {var}...")
        sampled_sar = sample_by_pixels_non_nan(sar_ds, var, num_to_sample=100_000, seed=42)
        var_fmt = var.lower()
        sampled_sar.to_csv(
            f"/scratch/users/trobinet/long_lfmc/trent_datasets/sar/sampled/{var_fmt}_samples.csv",
            index=False
        )

if __name__ == "__main__":
    main()