import argparse
import os
import datetime
from compile_training_data import compile_data

def main():
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
    start_date = datetime.datetime.strptime(
        parser.parse_args().start_date,
        '%Y-%m-%d'
    )
    end_date = datetime.datetime.strptime(
        parser.parse_args().end_date,
        '%Y-%m-%d'
    )
    # general scratch directory for this project
    scratch_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets'
    # get all the information that we need for the features and labels
    features = {
        'dirs':{
            #'daymet_prcp':os.path.join(
            #    scratch_dir,
            #    'daymet/daymet_regrid/prcp/{year:04d}/{month:02d}',
            #    'prcp_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            #),
            #'daymet_srad':os.path.join(
            #    scratch_dir,
            #    'daymet/daymet_regrid/srad/{year:04d}/{month:02d}',
            #    'srad_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            #),
            #'daymet_swe':os.path.join(
            #    scratch_dir,
            #    'daymet/daymet_regrid/swe/{year:04d}/{month:02d}',
            #    'swe_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            #),
            #'daymet_tmax':os.path.join(
            #    scratch_dir,
            #    'daymet/daymet_regrid/tmax/{year:04d}/{month:02d}',
            #    'tmax_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            #),
            #'daymet_tmin':os.path.join(
            #    scratch_dir,
            #    'daymet/daymet_regrid/tmin/{year:04d}/{month:02d}',
            #    'tmin_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            #),
            #'daymet_vpd':os.path.join(
            #    scratch_dir,
            #    'daymet/daymet_regrid/vp/{year:04d}/{month:02d}',
            #    'vp_{year:04d}_{month:02d}_{day:02d}_regridded.nc'
            #),
            'daymet':os.path.join(
                scratch_dir,
                'daymet/daymet_all_vars.zarr'
            ),
            #'modis':os.path.join(
            #    scratch_dir,
            #    'modis/modis_regridded_gapfilled/quality_1/interpolated',
            #    '{year:04d}/{month:02d}',
            #    'modis_filled_{year:04d}{month:02d}{day:02d}.nc4'
            #),
            'modis':os.path.join(
                scratch_dir,
                'modis/modis_regridded_gapfilled/quality_1/interpolated',
                'modis_all_vars.zarr'
            ),
            'static':os.path.join(
                scratch_dir,
                'static/static_features_500m_epsg5070_float32.nc'
            ),
            'krishna_stats':os.path.join(
                scratch_dir,
                'krishna/stats/krishna_lfmc_statistics.nc4'
            ),
        },
        'vars':{
            #'daymet_prcp': ['prcp'],
            #'daymet_srad': ['srad'],
            #'daymet_swe': ['swe'],
            #'daymet_tmax': ['tmax'],
            #'daymet_tmin': ['tmin'],
            #'daymet_vpd': ['vp'],
            'daymet':[
                'prcp',
                'srad',
                'swe',
                'tmax',
                'tmin',
                'vp'
            ],
            'modis': [
                'Nadir_Reflectance_Band1_filled',
                'Nadir_Reflectance_Band2_filled',
                'Nadir_Reflectance_Band3_filled',
                'Nadir_Reflectance_Band4_filled',
                'Nadir_Reflectance_Band5_filled',
                'Nadir_Reflectance_Band6_filled',
                'Nadir_Reflectance_Band7_filled',
                'filled_1',
                'filled_2',
                'filled_3',
                'filled_4',
                'filled_5',
                'filled_6',
                'filled_7'
            ],
            'static':[
                'slope',
                'elevation',
                'canopy_height',
                'forest_cover',
                'clay',
                'sand'
            ],
            'krishna_stats':[
                'retrieved_lfmc_mean',
                'retrieved_lfmc_std',
                'retrieved_lfmc_min',
                'retrieved_lfmc_max',
                'retrieved_lfmc_djf_mean',
                'retrieved_lfmc_mam_mean',
                'retrieved_lfmc_jja_mean',
                'retrieved_lfmc_son_mean'
            ],
        },
        'type':{
            #'daymet_dayl':'spatial_temporal',
            #'daymet_prcp':'spatial_temporal',
            #'daymet_srad':'spatial_temporal',
            #'daymet_swe':'spatial_temporal',
            #'daymet_tmax':'spatial_temporal',
            #'daymet_tmin':'spatial_temporal',
            #'daymet_vpd':'spatial_temporal',
            'daymet':'spatial_temporal',
            'modis':'spatial_temporal',
            'static':'spatial_static',
            'krishna_stats':'spatial_static',
        }
    }
    # labels must be a csv with a "date"/"lat"/"lon" column and then a columns
    # corresponding to "var," which we take to be the truth value
    labels = {
        'dirs':{
            'nfmd':os.path.join(
                scratch_dir,
                'nfmd/nfmd_processed.csv'
            ),
            'rs':os.path.join(
                scratch_dir,
                'krishna/krishna_lfmc_samples.csv'
            )
        },
        'vars':{
            'nfmd':['lfmc'],
            'rs':['lfmc']
        }
    }
    # number of random samples to include from RS data
    num_samples_if_available = 15000
    # what does should be included? 0 is current day, 1 is previous day, 5 is 5
    # days before, etc.
    # this is only relevant for transformer
    days_to_include = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    ]
    inputs_outputs = (
        'y_Insitu_X_ModisfilledDaymetStaticKrishnastatsWeatherstats_30days_testing'
    )
    out_dir = os.path.join(
        scratch_dir,
        'compiled',
        inputs_outputs
    )
    os.makedirs(out_dir, exist_ok=True)
    out_fname = f'compiled_data_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv'
    out_fname = os.path.join(out_dir, out_fname)
    # compile the data
    compile_data(
        start_date,
        end_date,
        features,
        labels,
        days_to_include,
        out_fname,
        num_rs_samples = num_samples_if_available
    )


if __name__ == "__main__":
    main()
