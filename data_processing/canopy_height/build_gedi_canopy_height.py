import argparse
import gc
import os
import resource
import shutil
import subprocess
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import rasterio
import rioxarray
import xarray as xr
import yaml
from pyproj import Transformer
from rasterio.enums import Resampling

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
            "Build GEDI-based canopy height on the LFMC target grid."
        )
    )
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "configs.yaml"),
        help="Path to canopy-height config YAML.",
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


def get_plot_extent_from_target_grid(
    target_grid: xr.Dataset,
) -> Tuple[float, float, float, float]:
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


def ensure_full_local_gedi_source(config: Dict) -> str:
    full_download_path = config["paths"]["full_download_path"]
    if os.path.exists(full_download_path):
        print(f"Using existing full local GEDI source: {full_download_path}")
        return full_download_path

    source_url = config["gedi"]["source_url"]
    download_dir = os.path.dirname(full_download_path)
    os.makedirs(download_dir, exist_ok=True)
    wget_path = shutil.which("wget")
    curl_path = shutil.which("curl")
    attempted_cmds = []
    if wget_path is not None:
        attempted_cmds.append([
            wget_path,
            "--continue",
            "--progress=bar:force",
            "-O",
            full_download_path,
            source_url,
        ])
    if curl_path is not None:
        attempted_cmds.append([
            curl_path,
            "-L",
            "-C",
            "-",
            "-#",
            "-o",
            full_download_path,
            source_url,
        ])
    if not attempted_cmds:
        raise RuntimeError("Neither wget nor curl is available for GEDI download.")

    last_error = None
    for cmd in attempted_cmds:
        downloader_name = os.path.basename(cmd[0])
        print(
            f"Downloading full GEDI source to scratch with shell progress "
            f"using {downloader_name}:"
        )
        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return full_download_path
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(
                f"{downloader_name} download attempt failed with exit code "
                f"{exc.returncode}; trying next downloader if available."
            )
    raise RuntimeError(
        "All GEDI download attempts failed."
    ) from last_error
    return full_download_path


def decode_gedi_chunk(source_da: xr.DataArray, config: Dict) -> xr.DataArray:
    valid_mask = (
        (source_da >= float(config["gedi"]["valid_min"])) &
        (source_da <= float(config["gedi"]["valid_max"]))
    )
    for nodata_code in config["gedi"]["nodata_codes"]:
        valid_mask = valid_mask & (source_da != nodata_code)
    decoded = source_da.where(valid_mask).astype(np.float32)
    decoded.name = "canopy_height"
    decoded.attrs["units"] = config["gedi"]["output_units"]
    decoded.attrs["source_url"] = config["gedi"]["source_url"]
    decoded.attrs["valid_range"] = (
        f"{config['gedi']['valid_min']}-{config['gedi']['valid_max']}"
    )
    return decoded


def build_source_dataset(
    config: Dict,
) -> xr.Dataset:
    source_path = ensure_full_local_gedi_source(config)
    print(f"Opening GEDI source directly from full local file: {source_path}")
    source_da = rioxarray.open_rasterio(
        source_path,
        masked=False,
    ).squeeze(drop=True)
    print(
        f"GEDI source shape={source_da.shape}, "
        f"crs={source_da.rio.crs}, "
        f"x_range=({float(source_da.x.min()):.4f}, {float(source_da.x.max()):.4f}), "
        f"y_range=({float(source_da.y.min()):.4f}, {float(source_da.y.max()):.4f})"
    )
    source_da.name = "canopy_height"
    source_da.attrs["units"] = config["gedi"]["output_units"]
    source_da.attrs["source_url"] = config["gedi"]["source_url"]
    source_da.attrs["valid_range"] = (
        f"{config['gedi']['valid_min']}-{config['gedi']['valid_max']}"
    )
    source_ds = source_da.to_dataset(name="canopy_height")
    source_ds.rio.write_crs(str(source_da.rio.crs), inplace=True)
    print("Merged source dataset:")
    print(source_ds)
    return source_ds


def maxrss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 / 1024.0


def regrid_canopy_height(
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
    regridded = target_grid.copy().drop_vars(list(target_grid.data_vars))
    regridded["canopy_height"] = xr.DataArray(
        data=np.full(
            (
                regridded.sizes["y"],
                regridded.sizes["x"],
            ),
            np.nan,
            dtype=np.float32,
        ),
        dims=["y", "x"],
    )
    x_vals = regridded["x"].values
    y_vals = regridded["y"].values
    for chunk_index, target_chunk in enumerate(target_chunks, start=1):
        print(
            f"working on chunk {chunk_index}/{len(target_chunks)} "
            f"(maxrss_gib={maxrss_gib():.2f})"
        )
        padded_src_chunk = regridder.get_padded_chunk(
            target_chunk,
            source_ds,
            num_padding_pixels=processing["chunk_buffer_pixels"],
        )
        decoded_chunk = decode_gedi_chunk(
            padded_src_chunk["canopy_height"],
            config,
        ).to_dataset(name="canopy_height")
        decoded_chunk.rio.write_crs(str(source_ds.rio.crs), inplace=True)
        decoded_reproj = decoded_chunk.rio.reproject_match(
            target_chunk,
            resampling=resampling,
        )
        x_start = int(np.where(x_vals == target_chunk["x"].values[0])[0][0])
        y_start = int(np.where(y_vals == target_chunk["y"].values[0])[0][0])
        x_stop = x_start + int(target_chunk.sizes["x"])
        y_stop = y_start + int(target_chunk.sizes["y"])
        regridded["canopy_height"].values[y_start:y_stop, x_start:x_stop] = (
            decoded_reproj["canopy_height"].values
        )
        try:
            padded_src_chunk.close()
        except Exception:
            pass
        try:
            decoded_chunk.close()
        except Exception:
            pass
        try:
            decoded_reproj.close()
        except Exception:
            pass
        del padded_src_chunk, decoded_chunk, decoded_reproj
        gc.collect()
        print(
            f"finished chunk {chunk_index}/{len(target_chunks)} "
            f"(maxrss_gib={maxrss_gib():.2f})"
        )
    target_grid_mask = target_grid[list(target_grid.data_vars)[0]].isnull()
    regridded = regridded.where(~target_grid_mask)
    regridded["canopy_height"].attrs["units"] = config["gedi"]["output_units"]
    print("Finished regridding.")
    return regridded


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
    new_vals = regridded_ds["canopy_height"].values
    old_vals = legacy_ds["canopy_height"].values
    valid_mask = np.isfinite(new_vals) & np.isfinite(old_vals)
    summary_lines = ["gedi_vs_legacy_summary", ""]
    if valid_mask.sum() == 0:
        summary_lines.append("canopy_height: no overlapping valid pixels")
    else:
        diff_vals = new_vals[valid_mask] - old_vals[valid_mask]
        corr = float(np.corrcoef(new_vals[valid_mask], old_vals[valid_mask])[0, 1])
        summary_lines.extend(
            [
                "canopy_height:",
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
    abs_plot_path = os.path.join(plots_dir, "gedi_canopy_height_abs.png")
    diff_plot_path = os.path.join(plots_dir, "gedi_canopy_height_minus_legacy.png")
    print(f"Plotting absolute map: {abs_plot_path}")
    plot.plot_from_xarray(
        "da",
        regridded_ds["canopy_height"],
        "canopy_height",
        target_crs,
        target_crs,
        abs_plot_path,
        cmap=plotting_config["absolute_cmap"],
        extent=plot_extent,
        extent_crs=target_crs,
        title="GEDI canopy height (meters)",
        cbar_label="canopy_height (meters)",
    )
    diff_da = regridded_ds["canopy_height"] - legacy_ds["canopy_height"]
    diff_da.name = "canopy_height"
    diff_limit = compute_difference_limit(
        diff_da,
        plotting_config["difference_quantile"],
    )
    print(
        f"Plotting difference map: {diff_plot_path} "
        f"(symmetric limit={diff_limit:.3f})"
    )
    plot.plot_from_xarray(
        "da",
        diff_da,
        "canopy_height",
        target_crs,
        target_crs,
        diff_plot_path,
        cmap=plotting_config["difference_cmap"],
        extent=plot_extent,
        extent_crs=target_crs,
        title="GEDI minus legacy canopy height (meters)",
        cbar_label="canopy_height difference (meters)",
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
    source_ds = build_source_dataset(config)
    regridded_ds = regrid_canopy_height(source_ds, target_grid, config)

    output_dataset_path = paths["output_dataset_path"]
    print(f"Saving regridded GEDI dataset: {output_dataset_path}")
    regridder.save_xarray_w_encoding(regridded_ds, output_dataset_path)

    print(f"Opening legacy static dataset for comparison: {paths['legacy_static_path']}")
    legacy_ds = xr.open_dataset(paths["legacy_static_path"])[["canopy_height"]]
    make_plots(regridded_ds, legacy_ds, target_grid, config)
    write_summary(regridded_ds, legacy_ds, config)

    legacy_ds.close()
    regridded_ds.close()
    source_ds.close()
    target_grid.close()
    print("GEDI canopy-height build completed successfully.")


if __name__ == "__main__":
    main()
