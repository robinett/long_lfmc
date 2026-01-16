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
    oak_dir = '/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets'
    # get all the information that we need for the features and labels
    features = {
        'dirs':{
            'daymet':os.path.join(
                oak_dir,
                'daymet/daymet_all_vars.zarr'
            ),
            'modis':os.path.join(
                oak_dir,
                'modis/modis_regridded_gapfilled/quality_1/interpolated',
                'modis_all_vars.zarr'
            ),
            'static':os.path.join(
                oak_dir,
                'static/static_features_500m_epsg5070_float32.nc'
            ),
            'climate_zone':os.path.join(
                oak_dir,
                'climate_zones/climate_zone_per_pixel_westUS.nc4'
            ),
            'sar_stats':os.path.join(
                oak_dir,
                'sar/sar_stats.zarr'
            ),
            'landcover_frac':os.path.join(
                oak_dir,
                'nlcd/nlcd_target_grid_2003_2023.zarr'
            ),
            'nlcd_class':os.path.join(
                oak_dir,
                'nlcd/nlcd_2003_2023.zarr'
            ),
            'landcover_change':os.path.join(
                oak_dir,
                'nlcd/nlcd_land_cover_change_2016_2021.zarr'
            ),
            #'krishna_stats':os.path.join(
            #    scratch_dir,
            #    'krishna/stats/krishna_lfmc_statistics.nc4'
            #),
            #'weather_stats':os.path.join(
            #    scratch_dir,
            #    'daymet/stats/stats_{year}.zarr'
            #)
        },
        'vars':{
            'daymet':[
                'prcp',
                'srad',
                'swe',
                'tmax',
                #'tmin',
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
            ],
            'static':[
                'slope',
                'elevation',
                'canopy_height',
                #'forest_cover',
                'clay',
                'sand'
            ],
            'climate_zone':['climate_zone'],
            'sar_stats':[
                #'sar_vv_mean',
                'sar_vh_mean',
                #'sar_vv_minus_vh_mean',
                #'sar_vv_std',
                'sar_vh_std',
                #'sar_vv_minus_vh_std',
                #'sar_vv_min',
                'sar_vh_min',
                #'sar_vv_minus_vh_min',
                #'sar_vv_max',
                'sar_vh_max',
                #'sar_vv_minus_vh_max',
                #'sar_vv_jan_mean',
                #'sar_vv_feb_mean',
                #'sar_vv_mar_mean',
                #'sar_vv_apr_mean',
                #'sar_vv_may_mean',
                #'sar_vv_jun_mean',
                #'sar_vv_jul_mean',
                #'sar_vv_aug_mean',
                #'sar_vv_sep_mean',
                #'sar_vv_oct_mean',
                #'sar_vv_nov_mean',
                #'sar_vv_dec_mean',
                'sar_vh_jan_mean',
                'sar_vh_feb_mean',
                'sar_vh_mar_mean',
                'sar_vh_apr_mean',
                'sar_vh_may_mean',
                'sar_vh_jun_mean',
                'sar_vh_jul_mean',
                'sar_vh_aug_mean',
                'sar_vh_sep_mean',
                'sar_vh_oct_mean',
                'sar_vh_nov_mean',
                'sar_vh_dec_mean',
                #'sar_vv_minus_vh_jan_mean',
                #'sar_vv_minus_vh_feb_mean',
                #'sar_vv_minus_vh_mar_mean',
                #'sar_vv_minus_vh_apr_mean',
                #'sar_vv_minus_vh_may_mean',
                #'sar_vv_minus_vh_jun_mean',
                #'sar_vv_minus_vh_jul_mean',
                #'sar_vv_minus_vh_aug_mean',
                #'sar_vv_minus_vh_sep_mean',
                #'sar_vv_minus_vh_oct_mean',
                #'sar_vv_minus_vh_nov_mean',
                #'sar_vv_minus_vh_dec_mean',
                #'vv_skewness',
                'vh_skewness',
                #'vv_minus_vh_skewness',
                #'vv_kurtosis',
                'vh_kurtosis',
                #'vv_minus_vh_kurtosis',
                #'vv_autocorr1',
                'vh_autocorr1',
                #'vv_minus_vh_autocorr1',
                #'vv_autocorr2',
                'vh_autocorr2',
                #'vv_minus_vh_autocorr2'
            ],
            'landcover_frac':[
                'barren',
                'crops',
                'deciduous_forest',
                'developed',
                'evergreen_forest',
                'grass',
                'mixed_forest',
                'other',
                'shrub',
                'water',
                'wetlands'
            ],
            'nlcd_class':[
                'nlcd'
            ],
            'landcover_change':[
                'land_cover_change_flag'
            ],
            #'krishna_stats':[
            #    'retrieved_lfmc_mean',
            #    'retrieved_lfmc_std',
            #    'retrieved_lfmc_min',
            #    'retrieved_lfmc_max',
            #    'retrieved_Jan_mean',
            #    'retrieved_Feb_mean',
            #    'retrieved_Mar_mean',
            #    'retrieved_Apr_mean',
            #    'retrieved_May_mean',
            #    'retrieved_Jun_mean',
            #    'retrieved_Jul_mean',
            #    'retrieved_Aug_mean',
            #    'retrieved_Sep_mean',
            #    'retrieved_Oct_mean',
            #    'retrieved_Nov_mean',
            #    'retrieved_Dec_mean',
            #    'retrieved_lfmc_skewness',
            #    'retrieved_lfmc_kurtosis',
            #    'retrieved_lfmc_autocorr1',
            #    'retrieved_lfmc_autocorr2'
            #],
            #'weather_stats':[
            #    'prcp_cum_0d_4d','prcp_cum_5d_9d','prcp_cum_10d_14d',
            #    'prcp_cum_15d_19d','prcp_cum_20d_24d','prcp_cum_25d_29d',
            #    'prcp_cum_30d_34d','prcp_cum_35d_39d','prcp_cum_40d_44d',
            #    'prcp_cum_45d_49d','prcp_cum_50d_54d','prcp_cum_55d_59d',
            #    'prcp_cum_60d_64d','prcp_cum_65d_69d','prcp_cum_70d_74d',
            #    'prcp_cum_75d_79d','prcp_cum_80d_84d','prcp_cum_85d_89d',
            #    'prcp_cum_90d_94d','prcp_cum_95d_99d','prcp_cum_100d_104d',
            #    'prcp_cum_105d_109d','prcp_cum_110d_114d','prcp_cum_115d_119d',
            #    'prcp_cum_120d_124d','prcp_cum_125d_129d','prcp_cum_130d_134d',
            #    'prcp_cum_135d_139d','prcp_cum_140d_144d','prcp_cum_145d_149d',
            #    'prcp_cum_150d_154d','prcp_cum_155d_159d','prcp_cum_160d_164d',
            #    'prcp_cum_165d_169d','prcp_cum_170d_174d','prcp_cum_175d_179d',
            #    'tmax_max_0d_4d','tmax_max_5d_9d','tmax_max_10d_14d',
            #    'tmax_max_15d_19d','tmax_max_20d_24d','tmax_max_25d_29d',
            #    'tmax_max_30d_34d','tmax_max_35d_39d','tmax_max_40d_44d',
            #    'tmax_max_45d_49d','tmax_max_50d_54d','tmax_max_55d_59d',
            #    'tmax_max_60d_64d','tmax_max_65d_69d','tmax_max_70d_74d',
            #    'tmax_max_75d_79d','tmax_max_80d_84d','tmax_max_85d_89d',
            #    'tmax_max_90d_94d','tmax_max_95d_99d','tmax_max_100d_104d',
            #    'tmax_max_105d_109d','tmax_max_110d_114d','tmax_max_115d_119d',
            #    'tmax_max_120d_124d','tmax_max_125d_129d','tmax_max_130d_134d',
            #    'tmax_max_135d_139d','tmax_max_140d_144d','tmax_max_145d_149d',
            #    'tmax_max_150d_154d','tmax_max_155d_159d','tmax_max_160d_164d',
            #    'tmax_max_165d_169d','tmax_max_170d_174d','tmax_max_175d_179d',
            #    'vp_min_0d_4d','vp_min_5d_9d','vp_min_10d_14d',
            #    'vp_min_15d_19d','vp_min_20d_24d','vp_min_25d_29d',
            #    'vp_min_30d_34d','vp_min_35d_39d','vp_min_40d_44d',
            #    'vp_min_45d_49d','vp_min_50d_54d','vp_min_55d_59d',
            #    'vp_min_60d_64d','vp_min_65d_69d','vp_min_70d_74d',
            #    'vp_min_75d_79d','vp_min_80d_84d','vp_min_85d_89d',
            #    'vp_min_90d_94d','vp_min_95d_99d','vp_min_100d_104d',
            #    'vp_min_105d_109d','vp_min_110d_114d','vp_min_115d_119d',
            #    'vp_min_120d_124d','vp_min_125d_129d','vp_min_130d_134d',
            #    'vp_min_135d_139d','vp_min_140d_144d','vp_min_145d_149d',
            #    'vp_min_150d_154d','vp_min_155d_159d','vp_min_160d_164d',
            #    'vp_min_165d_169d','vp_min_170d_174d','vp_min_175d_179d',
            #    'swe_max_0d_4d','swe_max_5d_9d','swe_max_10d_14d',
            #    'swe_max_15d_19d','swe_max_20d_24d','swe_max_25d_29d',
            #    'swe_max_30d_34d','swe_max_35d_39d','swe_max_40d_44d',
            #    'swe_max_45d_49d','swe_max_50d_54d','swe_max_55d_59d',
            #    'swe_max_60d_64d','swe_max_65d_69d','swe_max_70d_74d',
            #    'swe_max_75d_79d','swe_max_80d_84d','swe_max_85d_89d',
            #    'swe_max_90d_94d','swe_max_95d_99d','swe_max_100d_104d',
            #    'swe_max_105d_109d','swe_max_110d_114d','swe_max_115d_119d',
            #    'swe_max_120d_124d','swe_max_125d_129d','swe_max_130d_134d',
            #    'swe_max_135d_139d','swe_max_140d_144d','swe_max_145d_149d',
            #    'swe_max_150d_154d','swe_max_155d_159d','swe_max_160d_164d',
            #    'swe_max_165d_169d','swe_max_170d_174d','swe_max_175d_179d'
            #]
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
            'climate_zone':'spatial_static_onehot',
            'sar_stats':'spatial_static',
            'landcover_frac':'spatial_yearly',
            'nlcd_class':'nlcd',
            'landcover_change':'spatial_static',
            #'krishna_stats':'spatial_static',
            #'weather_stats':'spatial_nolag'
        }
    }
    # labels must be a csv with a "date"/"lat"/"lon" column and then a columns
    # corresponding to "var," which we take to be the truth value
    labels = {
        'dirs':{
            'nfmd':os.path.join(
                oak_dir,
                'nfmd/nfmd_processed.csv'
            ),
            #'vv':os.path.join(
            #    scratch_dir,
            #    'sar/sampled/vv_samples.csv'
            #),
            'vh_at_sites':os.path.join(
                oak_dir,
                'sar/sampled/vh_backscatter_samples_at_sites.csv'
            ),
            'vh_at_random':os.path.join(
                oak_dir,
                'sar/sampled/vh_backscatter_samples_random.csv'
            )
            #'vv_minus_vh':os.path.join(
            #    scratch_dir,
            #    'sar/sampled/vv_minus_vh_samples.csv'
            #),
        },
        'vars':{
            'nfmd':['lfmc'],
            #'vv':['VV'],
            'vh_at_sites':['vh_backscatter'],
            'vh_at_random':['vh_backscatter'],
            #'vv_minus_vh':['vv_minus_vh']
        }
    }
    # number of random samples to include from RS data
    # WE CONTROL THIS NOW IN THE SAMPLING; JUST TAKE ALL
    num_samples_if_available = 100000000.0
    # what does should be included? 0 is current day, 1 is previous day, 5 is 5
    # days before, etc.
    # this is only relevant for transformer
    days_to_include = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
        141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
        151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
        161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
        171, 172, 173, 174, 175, 176, 177, 178, 179, 180
    ]
    inputs_outputs = (
        'y_InsituVh_X_ModisfilledDaymetStaticClimatezoneSarstatsLandcoverfracLandcoverchange_Z_Nlcdclass_180d'
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
