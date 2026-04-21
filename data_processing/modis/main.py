import glob
import os
import datetime
import sys
import argparse

import process_modis as p_modis
from get_modis import dict_to_key, load_missing_granules_manifest

def main():
    # pass in the start and end dates from the submitting script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date in YYYY-MM-DD format",
        required=True
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date in YYYY-MM-DD format",
        required=True
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory",
        required=True
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        help="Root directory containing raw MODIS HDF files",
        required=False,
        default=None
    )
    parser.add_argument(
        "--quality_flag",
        type=str,
        help="Quality flag to use. All flags <= quality_flag are used",
        required=True
    )
    args = parser.parse_args()
    start_date_str = args.start_date
    end_date_str = args.end_date
    modis_processed_dir = args.out_dir
    quality_flag = int(args.quality_flag)
    process_start_date = datetime.datetime.strptime(
        start_date_str,
        '%Y-%m-%d'
    )
    process_end_date = datetime.datetime.strptime(
        end_date_str,
        '%Y-%m-%d'
    )
    print('start date:', process_start_date)
    print('end date:', process_end_date)
    # general scratch directory for this project
    scratch_dir = '/scratch/users/trobinet/long_lfmc'
    # scratch directory for raw modis
    modis_raw_dir = os.path.join(
        scratch_dir,
        'final_lfmc',
        'modis',
        'modis_earthaccess'
    )
    if args.raw_root is not None:
        modis_raw_dir = args.raw_root
    # scratch directory for processed modis
    #modis_processed_dir = os.path.join(
    #    scratch_dir,
    #    'trent_datasets',
    #    'modis',
    #    'modis_processed_daily'
    #)
    # what layers do we want to extract?
    layer_names = {
        'data':[
            'Nadir_Reflectance_Band1',
            'Nadir_Reflectance_Band2',
            'Nadir_Reflectance_Band3',
            'Nadir_Reflectance_Band4',
            'Nadir_Reflectance_Band5',
            'Nadir_Reflectance_Band6',
            'Nadir_Reflectance_Band7'
        ],
        'quality':[
            'BRDF_Albedo_Band_Quality_Band1',
            'BRDF_Albedo_Band_Quality_Band2',
            'BRDF_Albedo_Band_Quality_Band3',
            'BRDF_Albedo_Band_Quality_Band4',
            'BRDF_Albedo_Band_Quality_Band5',
            'BRDF_Albedo_Band_Quality_Band6',
            'BRDF_Albedo_Band_Quality_Band7'
        ]
    }
    # finally, how many raw files should there be per day? If this many files
    # do not exist for that day, we will throw an error because this means that
    # we are missing data that would be needed to create the relevant daily netcdf
    tiles_per_day = 10
    # let's get all the raw modis files and sort them by date
    #modis_files_2003 = sorted(
    #    glob.glob(
    #        os.path.join(
    #            modis_raw_dir,
    #            '2003',
    #            'MCD43A4*.hdf'
    #        )
    #    )
    #)
    #modis_quality_files_2003 = sorted(
    #    glob.glob(
    #        os.path.join(
    #            modis_raw_dir,
    #            '2003',
    #            'MCD43A2*.hdf'
    #        )
    #    )
    #)
    ## let's explore this first file
    #p_modis.explore_single_file(
    #    modis_files_2003[0]
    #)
    #p_modis.explore_single_file(
    #    modis_files_2003[1]
    #)
    #p_modis.explore_single_file(
    #    modis_quality_files_2003[0]
    #)
    # create the modis grid
    # let's process these hdf files into daily netcdf files to actually work
    # with them
    #process_start_date = datetime.datetime(2005, 1, 1)
    #process_end_date = datetime.datetime(2005, 1, 31)
    metadata = p_modis.get_metadata(
        modis_raw_dir,
        process_start_date,
        process_end_date
    )
    missing_manifest = load_missing_granules_manifest(modis_raw_dir)
    allowed_missing_keys = {
        dict_to_key(record)
        for records in missing_manifest.values()
        for record in records
    }
    if allowed_missing_keys:
        print(
            'allowing {} MODIS logical gaps to be written as NaN tiles'.format(
                len(allowed_missing_keys)
            )
        )
    p_modis.regrid_to_daily_ncs(
        modis_raw_dir,
        metadata,
        process_start_date,
        process_end_date,
        layer_names,
        tiles_per_day,
        modis_processed_dir,
        quality_flag=quality_flag,
        allowed_missing_keys=allowed_missing_keys,
    )

if __name__ == '__main__':
    main()
