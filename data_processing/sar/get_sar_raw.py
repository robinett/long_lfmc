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
from pystac_client import Client
import odc.stac
import planetary_computer as pc
import sys
import rioxarray as rxr
import asf_search as asf
from shapely.geometry import box
from pyproj import Transformer
from rioxarray.merge import merge_arrays
import glob
import matplotlib.pyplot as plt

def has_vh_asset(it):
    keys = {k.lower() for k in it.assets.keys()}
    return "vh" in keys or any("vh" in k for k in keys)

def get_vh_asset(item):
    # exact 'vh' first
    for k, a in item.assets.items():
        if k.lower() == "vh":
            return a
    # fallback: anything with 'vh' in the name
    for k, a in item.assets.items():
        if "vh" in k.lower():
            return a
    raise KeyError(f"No VH asset in {item.id}. keys={list(item.assets.keys())}")

def process_range(
    start_date,
    end_date,
    bounding_box,
    stac_url,
    collection,
    target_grid_path,
    out_dir,
    scratch_dir
):
    # localize start and end to pacific time; then get us utc version
    start_date_pac = start_date.tz_localize('America/Los_Angeles')
    end_date_pac = end_date.tz_localize('America/Los_Angeles')
    start_date_utc = start_date_pac.tz_convert('UTC')
    end_date_utc = end_date_pac.tz_convert('UTC')
    days_utc = pd.date_range(start=start_date_utc, end=end_date_utc, freq='D')
    tgt_grid = xr.open_dataset(target_grid_path)
    out_ds = tgt_grid.rename({'random_vals':'vh_backscatter'})
    out_ds = out_ds.expand_dims(time=days_utc)
    out_ds['vh_backscatter'] = xr.full_like(
        out_ds['vh_backscatter'],
        fill_value=np.nan
    )
    for d,date_utc in enumerate(days_utc):
        # Implement the processing logic for each date here
        out_path = os.path.join(
            out_dir,
            f's1_{date_utc.strftime("%Y%m%d")}.nc'
        )
        date_utc_iso = date_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        date_utc_iso_end = (date_utc + pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        lon_min, lat_min, lon_max, lat_max = bounding_box
        geom = box(lon_min, lat_min, lon_max, lat_max)
        print(date_utc_iso)
        print(date_utc_iso_end)
        print(geom.wkt)
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
        results = results[:10]
        vh_urls = []
        mask_urls = []
        for r,res in enumerate(results):
            this_urls = res.properties['additionalUrls']
            this_vh_tif = next(
                u for u in this_urls
                if u.endswith('_VH.tif')
            )
            this_mask_tif = next(
                u for u in this_urls
                if u.endswith('mask.tif')
            )
            vh_urls.append(this_vh_tif)
            mask_urls.append(this_mask_tif)
            #print(this_url)
        all_urls = vh_urls + mask_urls
        print('downloading files')
        asf.download_urls(urls=all_urls,path=scratch_dir)
        # open these tifs as a single dataset
        vh_files = glob.glob(
            os.path.join(scratch_dir, '*_VH.tif')
        )
        vh_das = [
            rxr.open_rasterio(f,masked=True,chunks={'x':2048,'y':2048})
                .squeeze(drop=True)
            for f in vh_files
        ]
        vh_mosaic = merge_arrays(vh_das)
        mask_files = glob.glob(
            os.path.join(scratch_dir, '*_mask.tif')
        )
        mask_das = [
            rxr.open_rasterio(f,masked=True,chunks={'x':2048,'y':2048})
                .squeeze(drop=True)
            for f in mask_files
        ]
        mask_mosaic = merge_arrays(mask_das)
        valid = (
            (mask_mosaic == 0.) |
            (mask_mosaic == 1.) |
            (mask_mosaic == 2.) |
            (mask_mosaic == 3.)
        )
        vh_mosaic = vh_mosaic.where(valid)
        # plot a hist if we want to see how things are looking
        vh_vals = vh_mosaic.values.flatten()
        vh_vals = vh_vals[~np.isnan(vh_vals)]
        vh_vals_db = 10 * np.log10(vh_vals + 1e-10)
        print(np.unique(vh_vals_db))
        # round to the nearest 0.1
        vh_vals_db = np.round(vh_vals_db * 10) / 10
        idxs = np.arange(len(vh_vals_db))
        rand_sel = np.random.choice(idxs, size=10_000, replace=False)
        vh_vals_sel = vh_vals_db[rand_sel]
        vh_u, vh_counts = np.unique(vh_vals_sel, return_counts=True)
        plt.figure(figsize=(6, 4))
        plt.bar(vh_u, vh_counts, width=1/10)
        plt.xlabel("VH value (db)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(
            '/scratch/users/trobinet/long_lfmc/trent_datasets/sar/',
            'plots',
            f"vh_histogram_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
        ))
        
        
        
        
        
        
        #cat = Client.open(
        #    stac_url,
        #    modifier=pc.sign_inplace,
        #)
        #s1 = cat.get_collection(collection)
        #search_kwargs = dict(
        #    collections=[s1.id],
        #    datetime=f"{date_utc_iso}/{date_utc_iso_end}",
        #    query={'sat:orbit_state':{'eq':'descending'}},
        #    bbox=bounding_box,
        #    max_items=5000
        #)
        #search = cat.search(**search_kwargs)
        #items = list(search.items())
        #items_vh = [it for it in items if has_vh_asset(it)]
        #print(len(items_vh))
        #print('descending + VH:', len(items_vh))
        ## extract the data from the items
        #trns = Transformer.from_crs("EPSG:32611", "EPSG:5070", always_xy=True)
        #for i,item in enumerate(items_vh):
        #    asset = get_vh_asset(item)
        #    this_s1_da = rxr.open_rasterio(asset.href, masked=True)
        #    print('this_s1_da')
        #    print(this_s1_da)
        #    print('this_s1_da crs:')
        #    print(this_s1_da.rio.crs)
        #    print('out_ds')
        #    print(out_ds)
        #    print('out_ds crs:')
        #    print(out_ds.rio.crs)
        #    # get the piece of our target grid that overlaps with the S1 data
        #    sys.exit()
        #    sys.exit()


        


def main():
    bounding_box = [
        -130.0,23,-96.5,52.0
    ]
    stac_url = (
        "https://planetarycomputer"
        ".microsoft.com/api/stac/v1"
    )
    collection = "sentinel-1-rtc"
    target_grid_path = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/grid/epsg5070_500m_westUS_grid.nc4'
    start_date = pd.Timestamp('2018-07-01')
    end_date = pd.Timestamp('2018-07-31')
    out_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_raw_daily/'
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/sar/temp/'
    process_range(
        start_date,
        end_date,
        bounding_box,
        stac_url,
        collection,
        target_grid_path,
        out_dir,
        scratch_dir
    )

if __name__ == "__main__":
    main()
