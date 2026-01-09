import xarray as xr
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, '..', '..')
sys.path.append(os.path.join(project_root, 'data_processing','shared'))

import plotting

def main():
    dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets/sar/sar_raw_daily'
    fnames = [
        's1_20190101.nc',
        's1_20160424.nc',
    ]
    for fname in fnames:
        ds = xr.open_dataset(os.path.join(dir, fname))
        plotting.plot_from_xarray(
            'ds',
            ds,
            'vh_backscatter',
            'EPSG:5070',
            'EPSG:5070',
            f'/scratch/users/trobinet/long_lfmc/trent_datasets/sar/plots/{fname.split(".")[0]}.png',
        )

if __name__ == '__main__':
    main()