import re
import numpy as np
import xarray as xr
import glob
import sys
import pandas as pd
from pyproj import Transformer

np.random.seed(42)

# extract time from filename
def date_from_path(p): 
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
    return np.datetime64(date_re.search(p).group(1))

def main():
    # which sampling should we do
    sample_at_sites = True
    sample_at_random = True
    # glob pattern to pick up everything
    sar_ds = xr.open_zarr(
        "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_500m_full.zarr",
        chunks="auto",
    )
    lfmc_df = pd.read_csv(
        "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/nfmd/nfmd_processed_landcovermatches.csv"
    )
    #vars_to_sample = ['VV', 'VH', 'vv_minus_vh']
    vars_to_sample = ['vh_backscatter']
    trns = Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)
    trns_back = Transformer.from_crs('EPSG:5070','EPSG:4326',always_xy=True)
    # load up the land cover data that we are going to use to make sure that we are only drawing
    # meaningful land cover types
    land_cover_ds = xr.open_zarr(
        '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/nlcd/nlcd_target_grid_2003_2023.zarr'
    )
    all_land_covers = list(land_cover_ds.data_vars)
    allowed_landcovers = [
        'deciduous_forest',
        'evergreen_forest',
        'grass',
        'mixed_forest',
        'shrub'
    ]
    for var in vars_to_sample:
        if not sample_at_sites:
            continue
        print(f"Sampling {var} at sites.")
        # generate a random integer between 1 and 100 to be our seed
        this_seed = np.random.randint(1, 100)
        # sample all timeseries for locations where we have LFMC samples
        # get all unique lat/lon combinations
        all_lats = lfmc_df["latitude"]
        all_lons = lfmc_df["longitude"]
        all_lat_lon = pd.DataFrame({"latitude": all_lats, "longitude": all_lons})
        all_lat_lon = all_lat_lon.drop_duplicates().reset_index(drop=True)
        for r,row in all_lat_lon.iterrows():
            if r % 10 == 0:
                print(f"Processing site {r}/{len(all_lat_lon)}")
            this_lat = row["latitude"]
            this_lon = row["longitude"]
            this_x, this_y = trns.transform(this_lon, this_lat)
            sampled_ds = sar_ds[var].sel(x=this_x, y=this_y, method="nearest")
            this_vals = sampled_ds.values
            this_dates = sampled_ds.coords["time"].values
            lats_rep = np.full_like(this_vals, fill_value=this_lat, dtype=float)
            lons_rep = np.full_like(this_vals, fill_value=this_lon, dtype=float)
            if r == 0:
                sampled_sar_at_sites = pd.DataFrame({
                    "date": this_dates,
                    "latitude": lats_rep,
                    "longitude": lons_rep,
                    var: this_vals
                })
            else:
                df_to_append = pd.DataFrame({
                    "date": this_dates,
                    "latitude": lats_rep,
                    "longitude": lons_rep,
                    var: this_vals
                })
                sampled_sar_at_sites = pd.concat([sampled_sar_at_sites, df_to_append], ignore_index=True)
        # drop nan pixels
        sampled_sar_at_sites = sampled_sar_at_sites.dropna()
        print(sampled_sar_at_sites)
        var_fmt = var.lower()
        sampled_sar_at_sites.to_csv(
            f"/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sampled/{var_fmt}_samples_at_sites_matching.csv"
        )
    for var in vars_to_sample:
        created_df = False
        if not sample_at_random:
            continue
        var_fmt = var.lower()
        print(f"Sampling {var} at random.")
        # get all the possible points
        xs = sar_ds[var].coords["x"].values
        ys = sar_ds[var].coords["y"].values
        all_xs, all_ys = np.meshgrid(xs, ys)
        all_points = pd.DataFrame({"x": all_xs.ravel(), "y": all_ys.ravel()})
        all_points = all_points.drop_duplicates().reset_index(drop=True)
        # repeadly sample from this dataframe until we have achieved our goal number of points
        num_points = len(all_points)
        perm = np.random.permutation(num_points)
        # for reference, batch size of 1000 gives about 1800 points per month
        batch_size = 500
        # stops at whichever comes first: desired points or max batches
        desired_points = 100_000
        max_batches = 1
        # our counter
        batch_num = 1
        for i in range(0, num_points, batch_size):
            if created_df and out_df.shape[0] >= desired_points:
                break
            if batch_num > max_batches:
                break
            # Get the current batch of points
            idx = perm[i:i+batch_size]
            batch = all_points.iloc[idx]
            x_arr = batch["x"].to_numpy()
            y_arr = batch["y"].to_numpy()
            # we need to run two checks on these points
            # first, is the sar data there nan?
            # second, is the land cover there something that we find acceptable?
            # start by getting the sar data for each of these points
            sar_data = sar_ds[var].sel(
                x=xr.DataArray(x_arr,dims='points'),
                y=xr.DataArray(y_arr,dims='points'),
                method="nearest"
            )
            # get the dates and check the land cover for each
            dates = sar_data.coords["time"].values
            last_month = 0
            for d,date in enumerate(dates):
                this_month = pd.to_datetime(date).month
                if this_month != last_month:
                    print(f'Processing {date} for batch {batch_num}')
                    print(f'Current number of samples: {out_df.shape[0]}' if created_df else 'No samples yet')
                    last_month = this_month
                this_sar = sar_data.sel(time=date).values
                this_lc = land_cover_ds.sel(
                    x=xr.DataArray(x_arr, dims="points"),
                    y=xr.DataArray(y_arr, dims="points"),
                    method="nearest",
                )
                this_lc = this_lc.sel(year=date, method="nearest")
                lc_arr = this_lc.to_array(dim='landcover').values
                valid = np.where(~np.isnan(lc_arr))
                okay_cols = np.unique(valid[1])
                lc_valid = lc_arr[:,okay_cols]
                sar_valid = this_sar[okay_cols]
                x_arr_valid = x_arr[okay_cols]
                y_arr_valid = y_arr[okay_cols]
                # get rid of anywhere that doesn't have a sar value
                valid_locs = np.where(~np.isnan(sar_valid))
                if valid_locs[0].size == 0:
                    continue
                lc_valid = lc_valid[:,valid_locs].squeeze()
                sar_valid = sar_valid[valid_locs]
                x_arr_valid = x_arr_valid[valid_locs]
                y_arr_valid = y_arr_valid[valid_locs]
                dom_lc_idx = np.argmax(lc_valid, axis=0)
                dom_lc_perc = np.max(lc_valid, axis=0)
                if type(dom_lc_idx) != np.ndarray:
                    dom_lc_idx = [dom_lc_idx]
                    dom_lc_perc = [dom_lc_perc]
                
                #    dom_lc = [0]
                #    idx_okay = [False]
                #else:
                dom_lc = [0 for n in range(len(dom_lc_idx))]
                idx_okay = [False for n in range(len(dom_lc_idx))]
                for n in range(len(dom_lc_idx)):
                    dom_lc[n] = all_land_covers[dom_lc_idx[n]]
                    if dom_lc[n] in allowed_landcovers and dom_lc_perc[n] >= 0.5:
                        idx_okay[n] = True
                final_xs = x_arr_valid[idx_okay]
                final_ys = y_arr_valid[idx_okay]
                final_sar = sar_valid[idx_okay]
                final_date = [date for _ in range(len(final_xs))]
                lons,lats = trns_back.transform(final_xs, final_ys)
                if not created_df:
                    created_df = True
                    out_df = pd.DataFrame({
                        "longitude": lons,
                        "latitude": lats,
                        "date": final_date,
                        var_fmt: final_sar
                    })
                else:
                    out_df = pd.concat([out_df, pd.DataFrame({
                        "longitude": lons,
                        "latitude": lats,
                        "date": final_date,
                        var_fmt: final_sar
                    })], ignore_index=True)
            print(out_df)
            batch_num += 1
        lat_lon_combos = list(zip(out_df["longitude"], out_df["latitude"]))
        unique_lat_lon_combos = list(set(lat_lon_combos))
        print(f"Sampled {len(out_df)} points from {len(unique_lat_lon_combos)} unique locations")
        out_df.to_csv(
            f"/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sampled/{var_fmt}_samples_random_matching.csv",
            index=False
        )

if __name__ == "__main__":
    main()