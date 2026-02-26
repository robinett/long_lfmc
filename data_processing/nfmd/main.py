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
        'nlcd_target_grid_2000_2024.zarr'
    )
    species_to_landcover_name = os.path.join(
        scratch_dir,
        'nfmd',
        'species_to_landcover_mapping.csv'
    )
    valid_grid_fname = os.path.join(
        scratch_dir,
        'grid',
        'epsg5070_500m_westUS_grid.nc4'
    )
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2024, 12, 31)
    nfmd_process_fname = os.path.join(
        scratch_dir,
        'nfmd',
        'nfmd_processed.csv'
    )
    # legacy arg retained for process() signature; filtering now uses the
    # valid area of the target grid inside process_nfmd.py.
    bound_box = None
    filter_mismatch_landcover = True
    process_nfmd.process(
        orig_csv_fname,
        nfmd_loc_fname,
        nlcd_fname,
        species_to_landcover_name,
        valid_grid_fname,
        start,
        end,
        bound_box,
        nfmd_process_fname,
        filter_mismatch_landcover=filter_mismatch_landcover
    )


if __name__ == '__main__':
    main()
