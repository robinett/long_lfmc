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

def process_range(
    start_date,
    end_date,
    bounding_box,
    stac_url,
    collection,
    target_grid_path,
    out_dir
):
    days = pd.date_range(start=start_date, end=end_date, freq='D')
    tgt_grid = xr.open_dataset(target_grid_path)
    for d,date in enumerate(days):
        # Implement the processing logic for each date here
        out_path = os.path.join(
            out_dir,
            f's1_{date.strftime("%Y%m%d")}.nc'
        )
        cat = Client.open(stac_url)
        s1 = cat.get_collection(collection)
        print(s1)

        print("ID:", s1.id)
        print("Title:", s1.title)
        print("Description:", s1.description)

        print("\nExtent:")
        print("Spatial:", s1.extent.spatial)
        print("Temporal:", s1.extent.temporal)

        print("\nSummaries:")
        for k, v in s1.summaries.lists.items():
            print(k, v[:5] if isinstance(v, list) else v)
        sys.exit()
        


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
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    out_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_raw_daily/'
    process_range(
        start_date,
        end_date,
        bounding_box,
        stac_url,
        collection,
        target_grid_path,
        out_dir
    )

if __name__ == "__main__":
    main()
