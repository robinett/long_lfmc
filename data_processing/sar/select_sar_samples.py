import numpy as np
import xarray as xr
import pandas as pd
from pyproj import Transformer
from datetime import datetime
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from dask.diagnostics import ProgressBar
except ImportError:
    ProgressBar = None

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    print(f"[{_ts()}] {msg}")

def _get_sar_var_da(sar_ds, var_name):
    if var_name in sar_ds.data_vars:
        return sar_ds[var_name]
    if "data" in sar_ds.data_vars and "variable" in sar_ds.coords:
        available = set(str(v) for v in sar_ds["variable"].values)
        if var_name not in available:
            raise KeyError(
                f"SAR variable '{var_name}' not found in stacked 'data' variable. "
                f"Available variable labels: {sorted(available)}"
            )
        return sar_ds["data"].sel(variable=var_name)
    raise KeyError(
        f"SAR variable '{var_name}' not found. Data variables: {list(sar_ds.data_vars)}"
    )

def _get_sar_vars_stack_da(sar_ds, var_names):
    if "data" in sar_ds.data_vars and "variable" in sar_ds.coords:
        available = set(str(v) for v in sar_ds["variable"].values)
        missing = [v for v in var_names if v not in available]
        if missing:
            raise KeyError(
                f"SAR variables missing from stacked 'data' variable: {missing}. "
                f"Available variable labels: {sorted(available)}"
            )
        return sar_ds["data"].sel(variable=var_names)
    arrs = []
    for v in var_names:
        arrs.append(_get_sar_var_da(sar_ds, v))
    return xr.concat(arrs, dim=pd.Index(var_names, name="variable"))

def _select_prefix_whole_sites(
    ordered_flat_idxs,
    ordered_counts,
    target_num_observations,
    strict_target=False,
):
    if ordered_flat_idxs.size == 0:
        return np.array([], dtype=np.int64), 0

    cumulative_obs = np.cumsum(ordered_counts, dtype=np.int64)
    stop_idx = np.searchsorted(cumulative_obs, target_num_observations, side="left")

    if stop_idx >= ordered_flat_idxs.size:
        selected_flat_idxs = ordered_flat_idxs
        selected_obs_total = int(cumulative_obs[-1])
        if strict_target and selected_obs_total < target_num_observations:
            raise RuntimeError(
                f"Could only collect {selected_obs_total} observations, below target "
                f"{target_num_observations}."
            )
        return selected_flat_idxs, selected_obs_total

    # Keep whole sites and pick the closer side around target.
    obs_over = int(cumulative_obs[stop_idx])
    if stop_idx == 0:
        return ordered_flat_idxs[:1], obs_over
    obs_under = int(cumulative_obs[stop_idx - 1])
    if abs(obs_under - target_num_observations) <= abs(obs_over - target_num_observations):
        return ordered_flat_idxs[:stop_idx], obs_under
    return ordered_flat_idxs[: stop_idx + 1], obs_over

def main():
    # which sampling should we do
    sample_at_sites = False
    sample_at_random = True
    random_seed = 42
    target_num_observations = 55_000
    random_count_batch_points = 2_000
    show_within_batch_progress = True
    min_dominant_landcover_fraction = 0.5
    strict_target = False
    _log("Starting SAR sample selection script.")
    _log(
        f"Settings: sample_at_sites={sample_at_sites}, sample_at_random={sample_at_random}, "
        f"random_seed={random_seed}, target_num_observations={target_num_observations}, "
        f"random_count_batch_points={random_count_batch_points}, "
        f"show_within_batch_progress={show_within_batch_progress}, "
        f"min_dominant_landcover_fraction={min_dominant_landcover_fraction}, "
        f"strict_target={strict_target}"
    )
    # glob pattern to pick up everything
    _log("Opening SAR zarr dataset...")
    sar_ds = xr.open_zarr(
        "/scratch/users/trobinet/long_lfmc/final_lfmc/sar/sar_all_vars.zarr",
        chunks="auto",
    )
    _log(
        f"SAR schema: data_vars={list(sar_ds.data_vars)}, "
        f"has_variable_coord={'variable' in sar_ds.coords}"
    )
    _log("Reading LFMC CSV...")
    lfmc_df = pd.read_csv(
        "/scratch/users/trobinet/long_lfmc/final_lfmc/nfmd/nfmd_processed.csv"
    )
    vars_to_sample = ['vv', 'vh', 'vv_minus_vh', 'vv_over_vh']
    _log(f"Variables to sample: {vars_to_sample}")
    trns = Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)
    trns_back = Transformer.from_crs('EPSG:5070','EPSG:4326',always_xy=True)
    # load up the land cover data that we are going to use to make sure that we are only drawing
    # meaningful land cover types
    _log("Opening NLCD zarr dataset...")
    land_cover_ds = xr.open_zarr(
        '/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_target_grid_2000_2024.zarr'
    )
    all_land_covers = list(land_cover_ds.data_vars)
    allowed_landcovers = [
        'deciduous_forest',
        'evergreen_forest',
        'grass',
        'mixed_forest',
        'shrub'
    ]
    allowed_landcover_idxs = [
        i for i, lc_name in enumerate(all_land_covers) if lc_name in allowed_landcovers
    ]
    if len(allowed_landcover_idxs) == 0:
        raise ValueError("No allowed land cover classes were found in the land cover dataset.")

    _log("Precomputing dominant land cover masks...")
    lc_array = land_cover_ds.to_array(dim="landcover")
    lc_any_valid = lc_array.notnull().any(dim="landcover")
    lc_array_filled = lc_array.fillna(-1.0)
    dominant_lc_idx = lc_array_filled.argmax(dim="landcover").astype(np.int16)
    dominant_lc_frac = lc_array_filled.max(dim="landcover")
    lc_allowed = xr.apply_ufunc(
        np.isin,
        dominant_lc_idx,
        kwargs={"test_elements": np.asarray(allowed_landcover_idxs, dtype=np.int16)},
        dask="parallelized",
        output_dtypes=[bool],
    )
    lc_ok_by_year = lc_any_valid & lc_allowed & (dominant_lc_frac >= min_dominant_landcover_fraction)
    sar_time = sar_ds.coords["time"]
    sar_time_dt = pd.to_datetime(sar_time.values)
    sar_year_dt = pd.to_datetime(
        {"year": sar_time_dt.year, "month": np.ones(sar_time_dt.size, dtype=int), "day": np.ones(sar_time_dt.size, dtype=int)}
    )
    sar_year = xr.DataArray(
        sar_year_dt.values,
        dims=["time"],
        coords={"time": sar_time.values},
    )
    lc_ok_by_time = lc_ok_by_year.sel(year=sar_year, method="nearest")
    _log("Land cover masks prepared.")
    if sample_at_sites:
        _log("Sampling all variables at sites with shared site selection.")
        all_lats = lfmc_df["latitude"]
        all_lons = lfmc_df["longitude"]
        all_lat_lon = pd.DataFrame({"latitude": all_lats, "longitude": all_lons})
        all_lat_lon = all_lat_lon.drop_duplicates().reset_index(drop=True)
        _log(f"Sites: {len(all_lat_lon)} unique LFMC locations.")
        site_lats = all_lat_lon["latitude"].to_numpy()
        site_lons = all_lat_lon["longitude"].to_numpy()
        _log("Sites: transforming coordinates to EPSG:5070...")
        site_xs, site_ys = trns.transform(site_lons, site_lats)
        _log("Sites: extracting all variables in one vectorized pull...")
        sites_stack_da = _get_sar_vars_stack_da(sar_ds, vars_to_sample).sel(
            x=xr.DataArray(site_xs, dims="points"),
            y=xr.DataArray(site_ys, dims="points"),
            method="nearest",
        ).transpose("time", "variable", "points")
        if ProgressBar:
            with ProgressBar():
                sites_stack_da = sites_stack_da.compute()
        else:
            _log("Dask ProgressBar unavailable; computing without progress bar.")
            sites_stack_da = sites_stack_da.compute()
        site_values = sites_stack_da.values
        site_dates = sites_stack_da.coords["time"].values
        # shape: (variable, points)
        site_obs_counts = np.sum(~np.isnan(site_values), axis=0).astype(np.int64)
        _log(f"Sites: obs-count matrix shape={site_obs_counts.shape} (variable, points).")
        site_rng = np.random.default_rng(random_seed)
        shuffled_site_idxs = site_rng.permutation(site_lats.size)

        site_vars_iter = tqdm(vars_to_sample, desc="Site vars", unit="var") if tqdm else vars_to_sample
        for var_idx, var in enumerate(site_vars_iter):
            _log(f"{var}: selecting random LFMC sites to hit target observations...")
            ordered_counts = site_obs_counts[var_idx, shuffled_site_idxs]
            eligible_mask = ordered_counts > 0
            eligible_site_idxs = shuffled_site_idxs[eligible_mask]
            eligible_counts = ordered_counts[eligible_mask]
            selected_site_idxs, selected_obs_total = _select_prefix_whole_sites(
                ordered_flat_idxs=eligible_site_idxs,
                ordered_counts=eligible_counts,
                target_num_observations=target_num_observations,
                strict_target=strict_target,
            )
            if selected_site_idxs.size == 0:
                raise RuntimeError(f"No eligible LFMC sites were found for variable '{var}'.")

            var_values = site_values[:, var_idx, selected_site_idxs]
            valid_mask = ~np.isnan(var_values)
            time_idx, point_idx = np.where(valid_mask)
            sampled_sar_at_sites = pd.DataFrame(
                {
                    "date": site_dates[time_idx],
                    "latitude": site_lats[selected_site_idxs][point_idx],
                    "longitude": site_lons[selected_site_idxs][point_idx],
                    var: var_values[time_idx, point_idx],
                }
            )
            _log(
                f"{var}: target={target_num_observations}, selected_sites={selected_site_idxs.size}, "
                f"estimated_obs={selected_obs_total}, written_obs={len(sampled_sar_at_sites)}."
            )
            var_fmt = var.lower()
            out_path = f"/scratch/users/trobinet/long_lfmc/final_lfmc/sar/{var_fmt}_samples_at_sites_matching.csv"
            sampled_sar_at_sites.to_csv(out_path)
            _log(f"{var}: wrote site samples to {out_path}")

    if sample_at_random:
        _log("Sampling random locations with shared all-variable extraction.")
        random_stack_da = _get_sar_vars_stack_da(sar_ds, vars_to_sample)
        xs = random_stack_da.coords["x"].values
        ys = random_stack_da.coords["y"].values
        n_x = len(xs)
        total_flat_points = len(xs) * len(ys)
        _log(f"Random grid dimensions x={len(xs)}, y={len(ys)}.")
        _log("Random: progressive batched location counting started.")
        rng = np.random.default_rng(random_seed)
        shuffled_flat_idxs_all = rng.permutation(total_flat_points)

        selected_flat_by_var = {var: [] for var in vars_to_sample}
        selected_obs_by_var = {var: 0 for var in vars_to_sample}
        done_by_var = {var: False for var in vars_to_sample}

        n_batches = int(np.ceil(total_flat_points / random_count_batch_points))
        batch_iter = range(n_batches)
        if tqdm:
            batch_iter = tqdm(batch_iter, total=n_batches, desc="Random count batches", unit="batch")
        random_batch_start_time = time.time()

        for batch_idx in batch_iter:
            if all(done_by_var.values()):
                _log(f"Random: all variables reached target by batch {batch_idx}.")
                break
            batch_wall_start = time.time()
            start = batch_idx * random_count_batch_points
            end = min(total_flat_points, start + random_count_batch_points)
            batch_flat_idxs = shuffled_flat_idxs_all[start:end]
            batch_y_idx = batch_flat_idxs // n_x
            batch_x_idx = batch_flat_idxs % n_x
            batch_xs = xs[batch_x_idx]
            batch_ys = ys[batch_y_idx]
            elapsed_total = batch_wall_start - random_batch_start_time
            completed_batches = batch_idx
            avg_batch_sec = elapsed_total / completed_batches if completed_batches > 0 else 0.0
            remaining_batches = n_batches - batch_idx
            eta_sec = avg_batch_sec * remaining_batches
            pct_done = (100.0 * start / total_flat_points) if total_flat_points > 0 else 0.0
            _log(
                f"Random batch {batch_idx + 1}/{n_batches} start: points={start}:{end} "
                f"({pct_done:.2f}%); elapsed={elapsed_total/60.0:.1f}m; eta={eta_sec/60.0:.1f}m."
            )

            sampled_batch_stack = random_stack_da.sel(
                x=xr.DataArray(batch_xs, dims="points"),
                y=xr.DataArray(batch_ys, dims="points"),
                method="nearest",
            ).transpose("time", "variable", "points")
            sampled_batch_lc_ok = lc_ok_by_time.sel(
                x=xr.DataArray(batch_xs, dims="points"),
                y=xr.DataArray(batch_ys, dims="points"),
                method="nearest",
            ).transpose("time", "points")
            if ProgressBar:
                with ProgressBar():
                    sampled_batch_stack = sampled_batch_stack.compute()
                    sampled_batch_lc_ok = sampled_batch_lc_ok.compute()
            else:
                sampled_batch_stack = sampled_batch_stack.compute()
                sampled_batch_lc_ok = sampled_batch_lc_ok.compute()
            batch_values = sampled_batch_stack.values
            batch_lc_mask = sampled_batch_lc_ok.values
            batch_counts = np.sum(
                (~np.isnan(batch_values)) & batch_lc_mask[:, None, :],
                axis=0,
                dtype=np.int64,
            )

            point_iter = range(batch_flat_idxs.size)
            if tqdm and show_within_batch_progress:
                point_iter = tqdm(
                    point_iter,
                    total=batch_flat_idxs.size,
                    desc=f"Batch {batch_idx + 1}/{n_batches} points",
                    unit="pt",
                    leave=False,
                )
            for point_i in point_iter:
                flat_idx = int(batch_flat_idxs[point_i])
                for var_idx, var in enumerate(vars_to_sample):
                    if done_by_var[var]:
                        continue
                    point_count = int(batch_counts[var_idx, point_i])
                    if point_count <= 0:
                        continue

                    current_obs = selected_obs_by_var[var]
                    obs_if_added = current_obs + point_count

                    if obs_if_added < target_num_observations:
                        selected_flat_by_var[var].append(flat_idx)
                        selected_obs_by_var[var] = obs_if_added
                        continue

                    if obs_if_added == target_num_observations:
                        selected_flat_by_var[var].append(flat_idx)
                        selected_obs_by_var[var] = obs_if_added
                        done_by_var[var] = True
                        continue

                    diff_under = target_num_observations - current_obs
                    diff_over = obs_if_added - target_num_observations
                    if diff_over < diff_under:
                        selected_flat_by_var[var].append(flat_idx)
                        selected_obs_by_var[var] = obs_if_added
                    done_by_var[var] = True

            batch_wall_end = time.time()
            batch_duration = batch_wall_end - batch_wall_start
            should_log_batch = (
                batch_idx == 0
                or (batch_idx + 1) % 25 == 0
                or end == total_flat_points
                or all(done_by_var.values())
            )
            if should_log_batch:
                status = ", ".join(
                    [
                        f"{var}:{selected_obs_by_var[var]}{'*' if done_by_var[var] else ''}"
                        for var in vars_to_sample
                    ]
                )
                _log(
                    f"Random batch {batch_idx + 1}/{n_batches}: processed_points={end}/{total_flat_points}; "
                    f"batch_duration={batch_duration:.1f}s; obs_status={status}"
                )

        for var in vars_to_sample:
            selected_flat_by_var[var] = np.asarray(selected_flat_by_var[var], dtype=np.int64)
            selected_obs_total = int(selected_obs_by_var[var])
            if selected_flat_by_var[var].size == 0:
                raise RuntimeError(f"No eligible random locations were found for variable '{var}'.")
            if strict_target and selected_obs_total < target_num_observations:
                raise RuntimeError(
                    f"Could only collect {selected_obs_total} observations for '{var}', "
                    f"below target {target_num_observations}."
                )
            _log(
                f"{var}: target={target_num_observations}, selected_locations={selected_flat_by_var[var].size}, "
                f"estimated_obs={selected_obs_total}."
            )

        union_selected_flat = np.unique(
            np.concatenate([selected_flat_by_var[var] for var in vars_to_sample])
        )
        union_y_idx = union_selected_flat // n_x
        union_x_idx = union_selected_flat % n_x
        union_xs = xs[union_x_idx]
        union_ys = ys[union_y_idx]
        _log(
            f"Random: extracting all vars once for union of selected locations "
            f"(n={union_selected_flat.size})."
        )
        sampled_random_stack = random_stack_da.sel(
            x=xr.DataArray(union_xs, dims="points"),
            y=xr.DataArray(union_ys, dims="points"),
            method="nearest",
        ).transpose("time", "variable", "points")
        sampled_random_lc_ok = lc_ok_by_time.sel(
            x=xr.DataArray(union_xs, dims="points"),
            y=xr.DataArray(union_ys, dims="points"),
            method="nearest",
        ).transpose("time", "points")
        if ProgressBar:
            with ProgressBar():
                sampled_random_stack = sampled_random_stack.compute()
                sampled_random_lc_ok = sampled_random_lc_ok.compute()
        else:
            sampled_random_stack = sampled_random_stack.compute()
            sampled_random_lc_ok = sampled_random_lc_ok.compute()
        random_values = sampled_random_stack.values
        random_dates = sampled_random_stack.coords["time"].values
        random_lc_mask = sampled_random_lc_ok.values

        rand_vars_iter = tqdm(list(enumerate(vars_to_sample)), desc="Random vars", unit="var") if tqdm else enumerate(vars_to_sample)
        for var_idx, var in rand_vars_iter:
            selected_flat_idxs = np.sort(selected_flat_by_var[var])
            union_positions = np.searchsorted(union_selected_flat, selected_flat_idxs)
            var_values = random_values[:, var_idx, union_positions]
            var_valid = (~np.isnan(var_values)) & random_lc_mask[:, union_positions]
            time_idx, point_idx = np.where(var_valid)
            sampled_vals = var_values[time_idx, point_idx]
            sampled_dates = random_dates[time_idx]

            selected_y_idx = selected_flat_idxs // n_x
            selected_x_idx = selected_flat_idxs % n_x
            selected_xs = xs[selected_x_idx]
            selected_ys = ys[selected_y_idx]
            sampled_points_x = selected_xs[point_idx]
            sampled_points_y = selected_ys[point_idx]
            sampled_lons, sampled_lats = trns_back.transform(sampled_points_x, sampled_points_y)

            var_fmt = var.lower()
            out_df = pd.DataFrame(
                {
                    "longitude": sampled_lons,
                    "latitude": sampled_lats,
                    "date": sampled_dates,
                    var_fmt: sampled_vals,
                }
            ).drop_duplicates(subset=["longitude", "latitude", "date"])
            unique_locations = out_df[["longitude", "latitude"]].drop_duplicates().shape[0]
            _log(
                f"{var}: target={target_num_observations}, selected_locations={selected_flat_idxs.size}, "
                f"estimated_obs={selected_obs_by_var[var]}, written_obs={len(out_df)}, "
                f"unique_locations={unique_locations}."
            )
            out_path = f"/scratch/users/trobinet/long_lfmc/final_lfmc/sar/{var_fmt}_samples_random_matching.csv"
            out_df.to_csv(out_path, index=False)
            _log(f"{var}: wrote random samples to {out_path}")

    _log("Script complete.")

if __name__ == "__main__":
    main()
