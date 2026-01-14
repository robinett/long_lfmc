import warnings
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
    module=r"urllib3(\..*)?",
)

import xarray as xr
import pandas as pd
import os
import numpy as np
import sys
import rioxarray as rxr
import asf_search as asf
from shapely.geometry import box
from pyproj import Transformer
from rioxarray.merge import merge_arrays
import glob
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from tqdm import tqdm
import shutil
from dask.diagnostics import ProgressBar
import copy
import argparse
from collections import defaultdict
import re

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, '..', '..')
sys.path.append(os.path.join(project_root, 'data_processing','shared'))

import plotting


track_re = re.compile(r"_T(?P<track>\d+)-")

def get_track(path: str) -> int:
    m = track_re.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse track: {path}")
    return int(m.group("track"))

def chunkify(seq, cap):
    for i in range(0, len(seq), cap):
        yield seq[i:i+cap]

def process_range(
    start_date,
    end_date,
    bounding_box,
    target_grid_path,
    out_dir,
    scratch_dir
):
    # localize start and end to pacific time; then get us utc version
    start_date_pac = start_date.tz_localize('America/Los_Angeles')
    end_date_pac = end_date.tz_localize('America/Los_Angeles')
    start_date_utc = start_date_pac.tz_convert('UTC')
    end_date_utc = end_date_pac.tz_convert('UTC') + pd.Timedelta(days=1)
    days_utc = pd.date_range(start=start_date_utc, end=end_date_utc, freq='D')
    tgt_grid = xr.open_dataset(target_grid_path)
    for d,date_utc in enumerate(days_utc):
        print(f'Processing {date_utc.strftime("%Y-%m-%d")}...')
        out_ds = tgt_grid.rename({'random_vals':'vh_backscatter'})
        #out_ds = out_ds.expand_dims(time=days_utc)
        out_ds['vh_backscatter'] = xr.full_like(
            out_ds['vh_backscatter'],
            fill_value=np.nan
        )
        # Implement the processing logic for each date here
        out_path = os.path.join(
            out_dir,
            f's1_{date_utc.strftime("%Y%m%d")}.nc'
        )
        date_utc_iso = date_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        date_utc_iso_end = (date_utc + pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        lon_min, lat_min, lon_max, lat_max = bounding_box
        geom = box(lon_min, lat_min, lon_max, lat_max)
        max_tries = 10
        sleep_seconds = 60
        for attempt in range(1,max_tries+1):
            try:
                results = asf.geo_search(
                    dataset="OPERA-S1",
                    processingLevel="RTC",
                    #beamMode="IW",
                    polarization="VH",
                    flightDirection="DESCENDING",
                    start=date_utc_iso,
                    end=date_utc_iso_end,
                    intersectsWith=geom.wkt,
                )
                break  # If successful, exit the loop
            except Exception as e:
                print(f"Attempt {attempt} failed for search: {e}")
                if attempt == max_tries:
                    raise
                print(f"Retrying search in {sleep_seconds} seconds...")
        #results = results[:10]
        vh_urls = []
        mask_urls = []
        skipped = []
        for r, res in enumerate(results):
            this_urls = res.properties.get("additionalUrls", []) or []
            this_vh_tif = next(
                (u for u in this_urls if u.endswith("_VH.tif")),
                None,
            )
            this_mask_tif = next(
                (u for u in this_urls if u.endswith("mask.tif")),
                None,
            )
            if this_vh_tif is None or this_mask_tif is None:
                skipped.append({
                    "idx": r,
                    "has_vh": this_vh_tif is not None,
                    "has_mask": this_mask_tif is not None,
                    "n_urls": len(this_urls),
                })
                continue
            vh_urls.append(this_vh_tif)
            mask_urls.append(this_mask_tif)
        all_urls = vh_urls + mask_urls
        print(f"downloading files: {len(vh_urls)} VH + {len(mask_urls)} masks")
        if skipped:
            print(f"skipped {len(skipped)} results missing VH/mask")
        #vh_urls = []
        #mask_urls = []
        #for r,res in enumerate(results):
        #    this_urls = res.properties['additionalUrls']
        #    this_vh_tif = next(
        #        u for u in this_urls
        #        if u.endswith('_VH.tif')
        #    )
        #    this_mask_tif = next(
        #        u for u in this_urls
        #        if u.endswith('mask.tif')
        #    )
        #    vh_urls.append(this_vh_tif)
        #    mask_urls.append(this_mask_tif)
        #    #print(this_url)
        #all_urls = vh_urls + mask_urls
        #print('downloading files')
        #for url in tqdm(all_urls, desc="Downloading SAR"):
        #    asf.download_url(url, path=scratch_dir)
        max_tries = 20
        sleep_seconds = 100
        for attempt in range(1,max_tries+1):
            try:
                asf.download_urls(urls=all_urls,path=scratch_dir,processes=1)
                break  # If successful, exit the loop
            except Exception as e:
                print(f"Attempt {attempt} failed for download: {e}")
                if attempt == max_tries:
                    raise
                # remove everything for the temp file so we don't get confused
                for name in os.listdir(scratch_dir):
                    os.remove(os.path.join(scratch_dir, name))
                print(f"Retrying download in {sleep_seconds} seconds...")
        # open these tifs as a single dataset
        # we just do chunk_size files at a time to keep things reasonable
        chunk_cap = 15
        vh_files = sorted(
            glob.glob(os.path.join(scratch_dir, "*_VH.tif"))
        )
        # group by track
        by_track = defaultdict(list)
        for f in vh_files:
            this_track = get_track(f)
            by_track[this_track].append(f)
        # build chunks: each chunk is only one track, max 25
        vh_chunks = []
        for track in sorted(by_track.keys()):
            files = sorted(by_track[track])
            for ch in chunkify(files, chunk_cap):
                vh_chunks.append((track, ch))
        #chunk_size = 10
        #vh_files = glob.glob(
        #    os.path.join(scratch_dir, '*_VH.tif')
        #)
        #vh_files = sorted(vh_files)
        #num_files = len(vh_files)
        #num_chunks = (num_files // chunk_size) + 1
        for c, (track, this_vh_files) in enumerate(
            tqdm(vh_chunks, desc="Processing Chunks")
        ):
            # skip if we are out of files
            if not this_vh_files:
                continue
            this_mask_files = []
            for f in this_vh_files:
                mask_file = f.replace('_VH.tif', '_mask.tif')
                if not os.path.exists(mask_file):
                    raise FileNotFoundError(f'Mask file not found: {mask_file}')
                this_mask_files.append(mask_file)
            vh_das = []
            #for f in tqdm(this_vh_files, desc="Opening VH TIFs"):
            for f in this_vh_files:
                vh_das.append(
                    rxr.open_rasterio(f, masked=True, chunks={'x': 2048, 'y': 2048})
                        .squeeze(drop=True)
                )
            #print('merging vh arrays')
            vh_mosaic = merge_arrays(vh_das)
            mask_das = []
            #for f in tqdm(this_mask_files, desc="Opening Mask TIFs"):
            for f in this_mask_files:
                mask_das.append(
                    rxr.open_rasterio(f, masked=True, chunks={'x': 2048, 'y': 2048})
                        .squeeze(drop=True)
                )
            #print('merging mask arrays')
            #with ProgressBar(): 
            mask_mosaic = merge_arrays(mask_das)
            #print('applying mask to vh mosaic')
            valid = (
                (mask_mosaic == 0.) |
                (mask_mosaic == 1.) |
                (mask_mosaic == 2.) |
                (mask_mosaic == 3.)
            )
            vh_mosaic = vh_mosaic.where(valid)
            ## plot this mosaic
            ##print('plotting vh mosaic')
            #vh_mosaic_converted = vh_mosaic.rio.reproject('EPSG:5070')
            #vh_mosaic_converted_db = 10 * np.log10(vh_mosaic_converted + 1e-10)
            ## Boolean mask of valid data
            #valid = np.isfinite(vh_mosaic_converted_db.values)
            ## Rows / columns that contain at least one valid value
            #rows = valid.any(axis=1)
            #cols = valid.any(axis=0)
            ## Bounds in coordinate space
            #xmin = float(vh_mosaic_converted_db.x.values[cols].min())
            #xmax = float(vh_mosaic_converted_db.x.values[cols].max())
            #ymin = float(vh_mosaic_converted_db.y.values[rows].min())
            #ymax = float(vh_mosaic_converted_db.y.values[rows].max())
            #bounds = [xmin, xmax, ymin, ymax]
            #plotting.plot_from_xarray(
            #    'da',
            #    vh_mosaic_converted_db,
            #    'vh_backscatter',
            #    'EPSG:5070',
            #    'EPSG:5070',
            #    f'/scratch/users/trobinet/long_lfmc/trent_datasets/sar/plots/vh_mosaic_c{c}.png',
            #    extent=bounds
            #)
            ## plot a hist if we want to see how things are looking
            #print('plotting vh histogram')
            #vh_vals = vh_mosaic.values.flatten()
            #vh_vals = vh_vals[~np.isnan(vh_vals)]
            #vh_vals_db = 10 * np.log10(vh_vals + 1e-10)
            #print(np.unique(vh_vals_db))
            ## round to the nearest 0.1
            #vh_vals_db = np.round(vh_vals_db * 10) / 10
            #idxs = np.arange(len(vh_vals_db))
            #rand_sel = np.random.choice(idxs, size=10_000, replace=False)
            #vh_vals_sel = vh_vals_db[rand_sel]
            #vh_u, vh_counts = np.unique(vh_vals_sel, return_counts=True)
            #plt.figure(figsize=(6, 4))
            #plt.bar(vh_u, vh_counts, width=1/10)
            #plt.xlabel("VH value (db)")
            #plt.ylabel("Count")
            #plt.savefig(os.path.join(
            #    '/scratch/users/trobinet/long_lfmc/trent_datasets/sar/',
            #    'plots',
            #    f"vh_histogram_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            #))
            # convert to our target grid for this day.
            #print('averaging up to target grid')
            vh_mask = xr.where(
                np.isfinite(vh_mosaic),1.0,0.0
            ).astype('float32')
            num = (vh_mosaic.fillna(0) * vh_mask).rio.reproject_match(
                out_ds,resampling = Resampling.average
            )
            num = num.where(num != np.finfo(np.float32).max)
            coverage = vh_mask.rio.reproject_match(
                out_ds,resampling=Resampling.average
            )
            coverage = coverage.where(coverage != np.finfo(np.float32).max)
            vh_mean = (num / coverage)
            vh_mean = vh_mean.where(coverage >= 0.75)
            vh_mean_db = 10 * np.log10(vh_mean + 1e-10)
            #plotting.plot_from_xarray(
            #    'da',
            #    vh_mean_db,
            #    'vh_backscatter',
            #    'EPSG:5070',
            #    'EPSG:5070',
            #    f'/scratch/users/trobinet/long_lfmc/trent_datasets/sar/plots/vh_mean_c{c}.png',
            #    extent=bounds
            #)
            # trim vh_mean_db to same shape as out_ds
            # add to out_ds only where valid
            # Boolean mask of valid data
            valid = (np.isfinite(vh_mean_db))
            out_ds['vh_backscatter'] = xr.where(
                valid,
                vh_mean_db,
                out_ds['vh_backscatter']
            )
            #plotting.plot_from_xarray(
            #    'ds',
            #    out_ds,
            #    'vh_backscatter',
            #    'EPSG:5070',
            #    'EPSG:5070',
            #    f'/scratch/users/trobinet/long_lfmc/trent_datasets/sar/plots/out_ds_c{c}.png',
            #    extent=None
            #)
        # save the daily .nc
        # but only save if not all empty
        valid = out_ds['vh_backscatter'].notnull().any()
        if valid:
            comp = dict(zlib=True, complevel=4)
            encoding = {
                var: comp
                for var in out_ds.data_vars
            }
            out_ds.to_netcdf(out_path, encoding=encoding)
        # clear the scratch directory
        for name in os.listdir(scratch_dir):
            path = os.path.join(scratch_dir, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        # clear everything in memory here
        out_ds.close()


def main():
    p = argparse.ArgumentParser(
        description="Process S1 SAR mosaics to daily 500m .nc4s"
    )
    p.add_argument(
        '--start_date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    p.add_argument(
        '--end_date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    p.add_argument(
        '--job_num',
        type=int,
        required=True,
        help='Job number for processing'
    )
    args = p.parse_args()
    bounding_box = [
        -130.0,23,-96.5,52.0
    ]
    target_grid_path = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/grid/epsg5070_500m_westUS_grid.nc4'
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    out_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_raw_daily/'
    scratch_dir = f'/scratch/users/trobinet/long_lfmc/trent_datasets/sar/temp/{args.job_num}/'
    # remove the temp dir if it exists
    shutil.rmtree(scratch_dir, ignore_errors=True)
    # re-create scratch dir
    os.makedirs(scratch_dir)
    process_range(
        start_date,
        end_date,
        bounding_box,
        target_grid_path,
        out_dir,
        scratch_dir
    )

if __name__ == "__main__":
    main()
