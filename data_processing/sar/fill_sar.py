import xarray as xr
import os
import numpy as np
from dask.diagnostics import ProgressBar

def main():
    oak_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets'
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets'
    print('opening daily SAR dataset')
    daily_sar = xr.open_zarr(
        os.path.join(
            scratch_dir,
            'sar',
            'sar_500m_full_vv.zarr'
        )
    )
    print(daily_sar)
    da = daily_sar["vv_backscatter"]
    # IMPORTANT: ffill is a cumulative op along time.
    # Your current chunks are (time=1, y=512, x=512) which will be *slow*.
    # Rechunk time to something reasonable (e.g., 128–512).
    print('re-chunking')
    da = da.chunk({"time": 256, "y": 512, "x": 512})
    t = daily_sar["time"]  # (time,)
    # 1) Forward-fill values
    print('forward filling')
    filled = da.ffill("time")
    # 2) Track "time of last valid observation" per pixel
    #    Put timestamp where valid, NaT where NaN
    print('tracking last valid observation times')
    valid_time = xr.where(da.notnull(), t, np.datetime64("NaT"))
    # Forward-fill that timestamp too
    print('forward filling last valid observation times')
    last_valid_time = valid_time.ffill("time")
    # Age since last valid obs (NaT propagates -> becomes NaT/NaN-ish mask behavior)
    age = t - last_valid_time
    # 3) Keep filled values only if last valid is within 90 days
    print('filtering for valid based on time')
    filled_90d = filled.where(age <= np.timedelta64(90, "D"))
    # Put back into a dataset (optional)
    out = daily_sar.copy()
    out["vv_backscatter"] = filled_90d
    out = out.chunk(
        {'time':1, 'y':512, 'x':512}
    )
    print(out)
    print('saving filled SAR dataset')
    with ProgressBar():
        out.to_zarr(
            os.path.join(
                scratch_dir,
                'sar',
                'sar_500m_filled_vv.zarr'
            ),
            mode='w',
            consolidated=True
        )

if __name__ == "__main__":
    main()