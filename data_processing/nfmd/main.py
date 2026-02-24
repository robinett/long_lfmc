import os
import sys
import datetime

import process_nfmd

def main():
    scratch_dir = '/scratch/users/trobinet/long_lfmc/final_lfmc'
    orig_csv_fname = os.path.join(
        scratch_dir,
        'nfmd',
        'lfmc_samples_20000101_20260224.csv'
    )
    nfmd_loc_fname = os.path.join(
        scratch_dir,
        'nfmd',
        'site_info.csv'
    )
    nlcd_fname = os.path.join(
        scratch_dir,
        'nlcd',
        'nlcd_target_grid_2000_2023.zarr'
    )
    species_to_landcover_name = os.path.join(
        scratch_dir,
        'nfmd',
        'species_to_landcover_mapping.csv'
    )
    start = datetime.datetime(2003, 1, 1)
    end = datetime.datetime(2023, 12, 31)
    nfmd_process_fname = os.path.join(
        scratch_dir,
        'nfmd',
        'nfmd_processed.csv'
    )
    # in the future we need to update this but quick and dirty for now.
    bound_box = [
        -125.0,
        29.0,
        -102.0,
        49.0
    ]
    filter_mismatch_landcover = True
    process_nfmd.process(
        orig_csv_fname,
        nfmd_loc_fname,
        nlcd_fname,
        species_to_landcover_name,
        start,
        end,
        bound_box,
        nfmd_process_fname,
        filter_mismatch_landcover=filter_mismatch_landcover
    )


if __name__ == '__main__':
    main()
