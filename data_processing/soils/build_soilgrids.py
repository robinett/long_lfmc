import argparse
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
import rioxarray
import xarray as xr
import yaml
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REGRID_DIR = os.path.join(REPO_ROOT, "data_processing", "regrid")
SHARED_DIR = os.path.join(REPO_ROOT, "data_processing", "shared")

for extra_path in [REGRID_DIR, SHARED_DIR]:
    if extra_path not in sys.path:
        sys.path.append(extra_path)

import plotting as plot
import regridder


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build SoilGrids-based clay and sand layers on the LFMC target grid."
        )
    )
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "configs.yaml"),
        help="Path to SoilGrids config YAML.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_resampling(method_name: str) -> Resampling:
    method_lookup = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }
    try:
        return method_lookup[method_name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported resampling method '{method_name}'. "
            f"Expected one of {sorted(method_lookup)}."
        ) from exc


def get_target_bounds(target_grid: xr.Dataset) -> Tuple[float, float, float, float]:
    x_vals = target_grid.coords["x"].values
    y_vals = target_grid.coords["y"].values
    x_res = float(np.abs(x_vals[1] - x_vals[0]))
    y_res = float(np.abs(y_vals[1] - y_vals[0]))
    x_min = float(np.min(x_vals) - x_res / 2.0)
    x_max = float(np.max(x_vals) + x_res / 2.0)
    y_min = float(np.min(y_vals) - y_res / 2.0)
    y_max = float(np.max(y_vals) + y_res / 2.0)
    return x_min, y_min, x_max, y_max


def get_plot_extent_from_target_grid(target_grid: xr.Dataset) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = get_target_bounds(target_grid)
    return x_min, x_max, y_min, y_max


def transform_bounds(
    bounds: Tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bounds
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = []
    ys = []
    for x_val, y_val in [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    ]:
        tx, ty = transformer.transform(x_val, y_val)
        xs.append(tx)
        ys.append(ty)
    return min(xs), min(ys), max(xs), max(ys)


def open_target_grid(target_grid_path: str, target_crs: str) -> xr.Dataset:
    print(f"Opening target grid: {target_grid_path}")
    target_grid = xr.open_dataset(target_grid_path)
    target_grid.rio.write_crs(target_crs, inplace=True)
    return target_grid


def get_clip_bounds_in_source_crs(
    target_grid: xr.Dataset,
    source_crs,
    target_crs: str,
    clip_buffer_m: float,
) -> Tuple[float, float, float, float]:
    target_bounds = get_target_bounds(target_grid)
    buffered_target_bounds = (
        target_bounds[0] - clip_buffer_m,
        target_bounds[1] - clip_buffer_m,
        target_bounds[2] + clip_buffer_m,
        target_bounds[3] + clip_buffer_m,
    )
    clip_bounds = transform_bounds(
        buffered_target_bounds,
        target_crs,
        source_crs,
    )
    print(
        "Target-grid clip bounds transformed to source CRS: "
        f"{clip_bounds}"
    )
    return clip_bounds


def get_soilgrids_vrt_path(
    base_url: str,
    property_name: str,
    depth_interval: str,
    prediction_statistic: str,
) -> str:
    return (
        f"/vsicurl/{base_url}/{property_name}/"
        f"{property_name}_{depth_interval}_{prediction_statistic}.vrt"
    )


def format_elapsed(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_partial_stage_path(raw_output_path: str) -> str:
    return f"{raw_output_path}.partial"


def resolve_soilgrids_source_path(
    property_config: Dict,
    soilgrids_config: Dict,
) -> Tuple[str, str]:
    full_local_path = property_config.get("full_local_path", "")
    if full_local_path:
        if os.path.exists(full_local_path):
            print(
                f"Using existing full local SoilGrids source for "
                f"{property_config['name']}: {full_local_path}"
            )
            return full_local_path, "local_full"
        print(
            f"Configured full local SoilGrids source not found for "
            f"{property_config['name']}: {full_local_path}"
        )
    vrt_path = get_soilgrids_vrt_path(
        soilgrids_config["base_url"],
        property_config["name"],
        soilgrids_config["depth_interval"],
        soilgrids_config["prediction_statistic"],
    )
    return vrt_path, "remote_vrt"


def stage_window_to_local_tiff(
    src,
    clip_window,
    raw_output_path: str,
    conversion_factor: float,
    stripe_height: int,
    progress_interval_pct: float,
    label: str,
) -> None:
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=int(clip_window.height),
        width=int(clip_window.width),
        transform=src.window_transform(clip_window),
        count=1,
        dtype="float32",
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=np.nan,
    )
    total_rows = int(clip_window.height)
    next_progress_pct = 0.0
    start_time = time.time()
    print(f"Writing staged local raster: {raw_output_path}")
    with rasterio.open(raw_output_path, "w", **profile) as dst:
        for row_start in range(0, total_rows, stripe_height):
            stripe_rows = min(stripe_height, total_rows - row_start)
            src_window = Window(
                col_off=int(clip_window.col_off),
                row_off=int(clip_window.row_off) + row_start,
                width=int(clip_window.width),
                height=int(stripe_rows),
            )
            dst_window = Window(
                col_off=0,
                row_off=row_start,
                width=int(clip_window.width),
                height=int(stripe_rows),
            )
            stripe = src.read(1, window=src_window, masked=True).astype(np.float32)
            stripe = stripe / conversion_factor
            dst.write(stripe.filled(np.nan), 1, window=dst_window)

            rows_done = row_start + stripe_rows
            pct_done = 100.0 * rows_done / total_rows
            should_log = (
                row_start == 0 or
                rows_done == total_rows or
                pct_done >= next_progress_pct
            )
            if should_log:
                elapsed = time.time() - start_time
                rate = rows_done / elapsed if elapsed > 0 else 0.0
                eta = (total_rows - rows_done) / rate if rate > 0 else float("nan")
                print(
                    f"{label}: staged rows {rows_done}/{total_rows} "
                    f"({pct_done:.1f}%), elapsed={format_elapsed(elapsed)}, "
                    f"eta={format_elapsed(eta) if np.isfinite(eta) else 'n/a'}"
                )
                while next_progress_pct <= pct_done:
                    next_progress_pct += progress_interval_pct


def download_and_clip_property(
    property_config: Dict,
    soilgrids_config: Dict,
    processing_config: Dict,
    clip_bounds: Tuple[float, float, float, float],
    raw_dir: str,
) -> xr.DataArray:
    property_name = property_config["name"]
    raw_output_path = os.path.join(
        raw_dir,
        (
            f"{property_name}_{soilgrids_config['depth_interval']}_"
            f"{soilgrids_config['prediction_statistic']}_westus_pct.tif"
        ),
    )
    partial_output_path = get_partial_stage_path(raw_output_path)
    if os.path.exists(partial_output_path):
        print(
            f"Removing incomplete staged raster for {property_name}: "
            f"{partial_output_path}"
        )
        os.remove(partial_output_path)
    if os.path.exists(raw_output_path):
        print(f"Using staged local raster for {property_name}: {raw_output_path}")
        staged = rioxarray.open_rasterio(raw_output_path, masked=True).squeeze(drop=True)
        staged.name = property_name
        return staged
    source_path, source_kind = resolve_soilgrids_source_path(
        property_config,
        soilgrids_config,
    )
    print(
        f"Staging SoilGrids AOI locally for {property_name} "
        f"from {source_kind}: {source_path}"
    )
    with rasterio.open(source_path) as src:
        print(
            f"{property_name}: source shape=({src.height}, {src.width}), "
            f"crs={src.crs}, bounds={src.bounds}"
        )
        window = src.window(*clip_bounds)
        window = window.round_offsets().round_lengths()
        stage_window_to_local_tiff(
            src,
            window,
            partial_output_path,
            float(property_config["conversion_factor"]),
            int(processing_config["stage_stripe_height"]),
            float(processing_config["stage_progress_interval_pct"]),
            property_name,
        )
    os.replace(partial_output_path, raw_output_path)
    clipped = rioxarray.open_rasterio(raw_output_path, masked=True).squeeze(drop=True)
    clipped.name = property_name
    clipped.attrs["units"] = property_config["output_units"]
    clipped.attrs["soilgrids_depth_interval"] = soilgrids_config["depth_interval"]
    clipped.attrs["soilgrids_prediction_statistic"] = (
        soilgrids_config["prediction_statistic"]
    )
    clipped.attrs["soilgrids_source_units"] = property_config["source_units"]
    print(
        f"{property_name}: clipped shape={clipped.shape}, "
        f"min={float(clipped.min(skipna=True).values):.3f}, "
        f"max={float(clipped.max(skipna=True).values):.3f}"
    )
    return clipped


def build_source_dataset(
    config: Dict,
    clip_bounds: Tuple[float, float, float, float],
) -> xr.Dataset:
    raw_dir = config["paths"]["raw_dir"]
    data_arrays = []
    source_crs = None
    for property_config in config["soilgrids"]["properties"]:
        clipped = download_and_clip_property(
            property_config,
            config["soilgrids"],
            config["processing"],
            clip_bounds,
            raw_dir,
        )
        if source_crs is None:
            source_crs = clipped.rio.crs
        data_arrays.append(clipped.to_dataset(name=property_config["name"]))
    source_ds = xr.merge(data_arrays, compat="override")
    source_ds.rio.write_crs(source_crs, inplace=True)
    print("Merged source dataset:")
    print(source_ds)
    return source_ds


def regrid_soils(
    source_ds: xr.Dataset,
    target_grid: xr.Dataset,
    config: Dict,
) -> xr.Dataset:
    processing = config["processing"]
    target_chunks, _, _ = regridder.chunk_xr_dataset(
        target_grid,
        chunk_size=processing["chunk_size"],
    )
    resampling = get_resampling(processing["resampling"])
    print(
        "Regridding source dataset onto target grid with "
        f"chunk_size={processing['chunk_size']}, "
        f"chunk_buffer_pixels={processing['chunk_buffer_pixels']}, "
        f"resampling={resampling.name}"
    )
    regridded = regridder.reproject_and_regrid_single_file(
        target_grid,
        source_ds,
        processing["target_crs"],
        str(source_ds.rio.crs),
        target_chunks,
        plot_tests=False,
        target_dir_last_ext="soils",
        chunk_buffer=processing["chunk_buffer_pixels"],
        resampling=resampling,
    )
    for property_config in config["soilgrids"]["properties"]:
        var_name = property_config["name"]
        regridded[var_name].attrs["units"] = property_config["output_units"]
    print("Finished regridding.")
    return regridded


def load_primary_water_wetlands_mask(
    target_grid: xr.Dataset,
    config: Dict,
) -> xr.DataArray:
    nlcd_path = config["paths"]["nlcd_target_path"]
    reference_year = np.datetime64(config["fill"]["landcover_reference_year"])
    print(
        "Loading NLCD fractions for shoreline fill mask: "
        f"{nlcd_path} ({reference_year})"
    )
    with xr.open_zarr(nlcd_path) as nlcd_ds:
        aligned = nlcd_ds.sel(
            year=reference_year,
            x=target_grid["x"],
            y=target_grid["y"],
        )
        landcover_vars = [
            "barren",
            "crops",
            "deciduous_forest",
            "developed",
            "evergreen_forest",
            "grass",
            "mixed_forest",
            "other",
            "shrub",
            "water",
            "wetlands",
        ]
        class_stack = xr.concat(
            [aligned[var_name] for var_name in landcover_vars],
            dim=xr.IndexVariable("landcover_class", landcover_vars),
        )
        primary_class = class_stack.idxmax("landcover_class")
        primary_water_wetlands = (
            (primary_class == "water") | (primary_class == "wetlands")
        )
    primary_water_wetlands.name = "primary_water_wetlands"
    return primary_water_wetlands


def get_neighbor_offsets() -> List[Tuple[int, int]]:
    offsets = []
    for y_offset in [-1, 0, 1]:
        for x_offset in [-1, 0, 1]:
            if x_offset == 0 and y_offset == 0:
                continue
            offsets.append((y_offset, x_offset))
    return offsets


def fill_missing_from_immediate_neighbors(
    data_array: xr.DataArray,
    primary_water_wetlands_mask: xr.DataArray,
) -> Tuple[xr.DataArray, Dict[str, int]]:
    values = data_array.values.copy()
    missing_mask = ~np.isfinite(values)
    eligible_mask = missing_mask & ~primary_water_wetlands_mask.values
    filled_values = values.copy()
    fill_count = 0
    retained_missing_count = 0
    neighbor_offsets = get_neighbor_offsets()

    eligible_rows, eligible_cols = np.where(eligible_mask)
    for row_idx, col_idx in zip(eligible_rows, eligible_cols):
        neighbor_values = []
        for row_offset, col_offset in neighbor_offsets:
            neighbor_row = row_idx + row_offset
            neighbor_col = col_idx + col_offset
            if (
                neighbor_row < 0 or
                neighbor_row >= values.shape[0] or
                neighbor_col < 0 or
                neighbor_col >= values.shape[1]
            ):
                continue
            neighbor_value = values[neighbor_row, neighbor_col]
            if np.isfinite(neighbor_value):
                neighbor_values.append(neighbor_value)
        if neighbor_values:
            filled_values[row_idx, col_idx] = float(np.mean(neighbor_values))
            fill_count += 1
        else:
            retained_missing_count += 1

    filled = xr.DataArray(
        filled_values,
        coords=data_array.coords,
        dims=data_array.dims,
        attrs=data_array.attrs,
        name=data_array.name,
    )
    fill_stats = {
        "initial_missing": int(missing_mask.sum()),
        "eligible_missing": int(eligible_mask.sum()),
        "filled": fill_count,
        "retained_missing": int((~np.isfinite(filled_values)).sum()),
        "retained_eligible_missing": retained_missing_count,
        "protected_primary_water_wetlands": int(
            (missing_mask & primary_water_wetlands_mask.values).sum()
        ),
    }
    return filled, fill_stats


def apply_shoreline_gap_fill(
    regridded_ds: xr.Dataset,
    target_grid: xr.Dataset,
    config: Dict,
) -> xr.Dataset:
    if not config["fill"]["enabled"]:
        return regridded_ds
    primary_water_wetlands_mask = load_primary_water_wetlands_mask(
        target_grid,
        config,
    )
    filled_vars = {}
    for property_config in config["soilgrids"]["properties"]:
        var_name = property_config["name"]
        print(f"Applying immediate-neighbor fill for {var_name}")
        filled_da, fill_stats = fill_missing_from_immediate_neighbors(
            regridded_ds[var_name],
            primary_water_wetlands_mask,
        )
        filled_da.attrs["fill_method"] = "8_neighbor_non_primary_water_wetlands"
        for stat_name, stat_value in fill_stats.items():
            print(f"  {var_name} {stat_name}={stat_value}")
        filled_vars[var_name] = filled_da
    filled_ds = xr.Dataset(filled_vars, coords=regridded_ds.coords, attrs=regridded_ds.attrs)
    filled_ds.rio.write_crs(config["processing"]["target_crs"], inplace=True)
    return filled_ds


def compute_difference_limit(diff_da: xr.DataArray, quantile: float) -> float:
    abs_values = np.abs(diff_da.values)
    finite_values = abs_values[np.isfinite(abs_values)]
    if finite_values.size == 0:
        return 1.0
    limit = float(np.quantile(finite_values, quantile))
    return max(limit, 1.0e-6)


def write_summary(
    regridded_ds: xr.Dataset,
    legacy_ds: xr.Dataset,
    config: Dict,
) -> None:
    summary_lines = []
    summary_lines.append("soilgrids_vs_legacy_summary")
    summary_lines.append("")
    for property_config in config["soilgrids"]["properties"]:
        var_name = property_config["name"]
        new_vals = regridded_ds[var_name].values
        old_vals = legacy_ds[var_name].values
        valid_mask = np.isfinite(new_vals) & np.isfinite(old_vals)
        diff_vals = new_vals[valid_mask] - old_vals[valid_mask]
        if diff_vals.size == 0:
            summary_lines.append(f"{var_name}: no overlapping valid pixels")
            summary_lines.append("")
            continue
        corr = float(np.corrcoef(new_vals[valid_mask], old_vals[valid_mask])[0, 1])
        summary_lines.extend(
            [
                f"{var_name}:",
                f"  overlap_valid_pixels={int(valid_mask.sum())}",
                f"  new_min={float(np.nanmin(new_vals)):.4f}",
                f"  new_max={float(np.nanmax(new_vals)):.4f}",
                f"  new_mean={float(np.nanmean(new_vals)):.4f}",
                f"  old_min={float(np.nanmin(old_vals)):.4f}",
                f"  old_max={float(np.nanmax(old_vals)):.4f}",
                f"  old_mean={float(np.nanmean(old_vals)):.4f}",
                f"  diff_mean={float(np.mean(diff_vals)):.4f}",
                f"  diff_median={float(np.median(diff_vals)):.4f}",
                f"  diff_p05={float(np.quantile(diff_vals, 0.05)):.4f}",
                f"  diff_p95={float(np.quantile(diff_vals, 0.95)):.4f}",
                f"  corr={corr:.6f}",
                "",
            ]
        )
    summary_path = config["paths"]["summary_path"]
    print(f"Writing summary report: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


def make_plots(
    regridded_ds: xr.Dataset,
    legacy_ds: xr.Dataset,
    target_grid: xr.Dataset,
    config: Dict,
) -> None:
    plots_dir = config["paths"]["plots_dir"]
    plotting_config = config["plotting"]
    target_crs = config["processing"]["target_crs"]
    plot_extent = get_plot_extent_from_target_grid(target_grid)
    for property_config in config["soilgrids"]["properties"]:
        var_name = property_config["name"]
        abs_plot_path = os.path.join(plots_dir, f"soilgrids_{var_name}_abs.png")
        diff_plot_path = os.path.join(
            plots_dir,
            f"soilgrids_{var_name}_minus_legacy.png",
        )
        print(f"Plotting absolute map for {var_name}: {abs_plot_path}")
        plot.plot_from_xarray(
            "da",
            regridded_ds[var_name],
            var_name,
            target_crs,
            target_crs,
            abs_plot_path,
            cmap=plotting_config["absolute_cmap"],
            extent=plot_extent,
            extent_crs=target_crs,
            title=f"SoilGrids {var_name} ({property_config['output_units']})",
            cbar_label=f"{var_name} ({property_config['output_units']})",
        )
        diff_da = regridded_ds[var_name] - legacy_ds[var_name]
        diff_da.name = var_name
        diff_limit = compute_difference_limit(
            diff_da,
            plotting_config["difference_quantile"],
        )
        print(
            f"Plotting difference map for {var_name}: {diff_plot_path} "
            f"(symmetric limit={diff_limit:.3f})"
        )
        plot.plot_from_xarray(
            "da",
            diff_da,
            var_name,
            target_crs,
            target_crs,
            diff_plot_path,
            cmap=plotting_config["difference_cmap"],
            extent=plot_extent,
            extent_crs=target_crs,
            title=f"SoilGrids minus legacy {var_name} ({property_config['output_units']})",
            cbar_label=f"{var_name} difference ({property_config['output_units']})",
            vmin=-diff_limit,
            vmax=diff_limit,
        )


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    ensure_dirs([paths["output_dir"], paths["raw_dir"], paths["plots_dir"]])

    target_grid = open_target_grid(
        paths["target_grid_path"],
        config["processing"]["target_crs"],
    )
    first_property = config["soilgrids"]["properties"][0]["name"]
    sample_vrt_path = get_soilgrids_vrt_path(
        config["soilgrids"]["base_url"],
        first_property,
        config["soilgrids"]["depth_interval"],
        config["soilgrids"]["prediction_statistic"],
    )
    print(f"Opening sample SoilGrids raster to resolve source CRS: {sample_vrt_path}")
    sample_da = rioxarray.open_rasterio(sample_vrt_path, masked=True).squeeze(drop=True)
    clip_bounds = get_clip_bounds_in_source_crs(
        target_grid,
        sample_da.rio.crs,
        config["processing"]["target_crs"],
        config["processing"]["clip_buffer_m"],
    )
    sample_da.close()

    source_ds = build_source_dataset(config, clip_bounds)
    regridded_ds = regrid_soils(source_ds, target_grid, config)
    regridded_ds = apply_shoreline_gap_fill(regridded_ds, target_grid, config)
    output_dataset_path = paths["output_dataset_path"]
    print(f"Saving regridded SoilGrids dataset: {output_dataset_path}")
    regridder.save_xarray_w_encoding(regridded_ds, output_dataset_path)

    print(f"Opening legacy static dataset for comparison: {paths['legacy_static_path']}")
    legacy_ds = xr.open_dataset(paths["legacy_static_path"])[["clay", "sand"]]
    make_plots(regridded_ds, legacy_ds, target_grid, config)
    write_summary(regridded_ds, legacy_ds, config)

    legacy_ds.close()
    regridded_ds.close()
    source_ds.close()
    target_grid.close()
    print("SoilGrids build completed successfully.")


if __name__ == "__main__":
    main()
