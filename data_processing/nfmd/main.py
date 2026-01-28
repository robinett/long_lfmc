import os
import sys
import datetime

import process_nfmd

def main():
    orig_csv_fname = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/nfmd/'
        'fieldSample.csv'
    )
    nfmd_loc_fname = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/nfmd/'
        'Site_Metadata.csv'
    )
    #nlcd_raw_fname = (
    #    '/scratch/users/trobinet/long_lfmc/trent_datasets/nlcd/'
    #    'nlcd_2003_2023.zarr'
    #)
    nlcd_fname = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/nlcd/'
        'nlcd_target_grid_2003_2023.zarr'
    )
    species_to_landcover_name = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/nfmd/'
        'species_to_landcover_mapping.csv'
    )
    start = datetime.datetime(2003, 1, 1)
    end = datetime.datetime(2023, 12, 31)
    nfmd_process_fname = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/nfmd/'
        'nfmd_processed_landcovermatches.csv'
    )
    # in the future we need to update this but quick and dirty for now.
    bound_box = [
        -125.0,
        29.0,
        -102.0,
        49.0
    ]
    process_nfmd.process(
        orig_csv_fname,
        nfmd_loc_fname,
        nlcd_fname,
        species_to_landcover_name,
        start,
        end,
        bound_box,
        nfmd_process_fname
    )


if __name__ == '__main__':
    main()
