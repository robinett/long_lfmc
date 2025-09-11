import torch
import pandas as pd
import pickle
import re
import numpy as np
from pathlib import Path
import argparse
import sys
from collections import defaultdict, Counter
import glob
import datetime
import copy
from datetime import timedelta
import os
import ast

# append up one dir to path
sys.path.append(sys.path[0] + '/../..')

from utils import plotting  # Assuming utils is a module with plotting function

def build_training_dataset(
    csv_names,
    first_label_date,
    last_label_date,
    static_features,
    dynamic_features,
    target_col,
    lag_days_keep,
    val_minor_mid_major_split=None, # Example split percentages: (0.05, 0.05, 0.05)
    train_split=0.8,
    split_type='random',
    plot_distributions=None,
    acceptable_target_range=(30,500), # Example range for LFMC, adjust
    model_name='transformer',
    train_on_filled=False,
    filled_cols = [],
    filled_cols_labs = []
):
    '''
    Function that takes a directory of csv files (with same features as columns,
    each row as a label) and builds a train/test dataset for a temporal
    transformer.
    '''
    # combine all of the csvs into one dataframe
    print("Combining CSV files...")
    df = combine_csvs(
        csv_names,
        first_label_date,
        last_label_date
    )
    # eliminate unreasonably high and low values
    # remove rows that have target values outside of the acceptable range
    num_rows_before = len(df.index)
    print("Removing rows with target values outside of acceptable range...")
    df = df[
        (df[target_col] >= acceptable_target_range[0]) &
        (df[target_col] <= acceptable_target_range[1])
    ].reset_index(drop=True)
    num_rows_after = len(df.index)
    print(
        'Number of rows removed due to target value range: '
        '{}'.format(num_rows_before - num_rows_after)
    )
    # first things first: remove the validation set. This ensure that  it will
    # be the same across all of our different model setups
    if val_minor_mid_major_split is not None:
        print("Isolating validation set...")
        (
            val_df,
            remaining_df,
            low_fill_vals,
            med_fill_vals,
            high_fill_vals
        ) = val_spatial_split(
            df,
            minor_mid_major_split,
            filled_cols,
            filled_cols_labs,
            plot_distributions=plot_distributions
        )
    else:
        remaining_df = df.copy()
    # remove non-specified lag days
    print("Removing non-specified lag days...")
    allowed_suffixes = {f"_day_minus_{lag}" for lag in lag_days_keep}
    cols_to_keep = [
        col for col in remaining_df.columns
        if (
            (any(col.endswith(suffix) for suffix in allowed_suffixes)) or
            (col in static_features) or
            (col == target_col) or
            (col in ['latitude', 'longitude', 'date'])
        )
    ]
    remaining_df = remaining_df[cols_to_keep]
    val_df = val_df[cols_to_keep] if val_df is not None else None
    # remove rows that have nan values
    if val_df is not None:
        print(f"Total number of val labels: {len(val_df.index)}")
    print(f"Total number of non-val labels: {len(remaining_df.index)}")
    #print(f"Total number of features: {len(remaining_df.columns)}")
    # hard coding in here that statistics on Krishna's data are not allowed to
    # have nans
    num_val_before_drop = len(val_df.index) if val_df is not None else 0
    num_non_val_before_drop = len(remaining_df.index)
    if 'retrieved_lfmc_mean' in remaining_df.columns:
        retrieved_cols = val_df.columns[
            val_df.columns.str.contains(
                'retrieved', case=False, regex=False
            )
        ]
        val_df = val_df.dropna(subset=retrieved_cols) if val_df is not None else None
        remaining_df = remaining_df.dropna(subset=retrieved_cols)
        num_val_after_drop = len(val_df.index) if val_df is not None else 0
        num_non_val_after_drop = len(remaining_df.index)
        print(
            'Number of val labels with no Krishna stats: '
            '{}'.format(num_val_before_drop - num_val_after_drop)
        )
        print(
            'Number of non-val labels with no Krishna stats: '
            '{}'.format(num_non_val_before_drop - num_non_val_after_drop)
        )
    num_obs_before_drop = len(remaining_df.index)
    remaining_df = remaining_df.dropna()
    num_obs_after_drop = len(remaining_df.index)
    print(
        'Number of non-val labels that couldnt be gap filled:'
        ' {}'.format(num_obs_before_drop - num_obs_after_drop)
    )
    val_df = val_df.dropna() if val_df is not None else None
    # isolate our validation set. this will tell us how good we are at
    # extrapolating to new locations that are a) heavily gap-filled, b) semi
    # gap-filled, or c) only minorly gap filled.
    # each of these should be about 5% of the data
    actual_dynamic_features = []
    for feature in dynamic_features:
        this_feature_give = feature + '_day_minus_'
        this_col = remaining_df.columns[
                remaining_df.columns.str.contains(
                    this_feature_give, case=False, regex=False
                )
            ]
        if len(this_col) > 0:
            for adf in this_col:
                actual_dynamic_features.append(adf)
    cols_to_use = set(static_features + actual_dynamic_features + [target_col])
    cols_dont_use = set(remaining_df.columns) - cols_to_use
    # now that these dfs have been processed, only keep the columns that we are
    # actually going to use as features/labels
    # if we don't want to train with filled data, get rid of that now
    num_remain_obs_with_filled = len(remaining_df.index)
    if not train_on_filled:
        print("Removing filled data from remaining dataset...")
        # get the rows that have 1 or 2 in any of filled col labs
        true_filled_cols_labs = remaining_df.columns[
            remaining_df.columns.str.contains(
                'filled', case=False, regex=False
            )
        ]
        filled_mask = remaining_df[true_filled_cols_labs].isin([1, 2]).any(axis=1)
        # remove these rows from the remaining_df
        remaining_df = remaining_df[~filled_mask].reset_index(drop=True)
    num_remain_obs_with_filled_after = len(remaining_df.index)
    print(
        'Number of non-val labels removed becuase of fill data: '
        '{}'.format(num_remain_obs_with_filled - num_remain_obs_with_filled_after)
    )
    # perform the train/test split on the remaining data
    # this can be:
        # 1. Random split. Lables are split completely randomly
        # 2. Temporal split. Labels are split based on time, with training data
        #    occuring before the test data.
        # 3. Spatial split. Labels are split based on spatial location, with
        #    all locations in the test set not appearing in the training set.
    print("Performing train/test split...")
    split_dict = {
        'random': random_split(remaining_df, train_split),
        'temporal': temporal_split(remaining_df, train_split),
        'spatial': train_test_spatial_split(remaining_df, train_split)
    }
    if split_type not in split_dict:
        raise ValueError(
            f"Split type {split_type} not recognized. "
            "Must be one of: " + ", ".join(split_dict.keys())
        )
    train_df, test_df, crit_1, crit_2 = split_dict[split_type]
    site_assignments = {
        'train':(
            set(zip(train_df['latitude'], train_df['longitude']))
        ),
        'test': (
            set(zip(test_df['latitude'], test_df['longitude']))
        ),
        'val_low': set(low_fill_vals) if low_fill_vals else None,
        'val_mid': set(med_fill_vals) if med_fill_vals else None,
        'val_high': set(high_fill_vals) if high_fill_vals else None
    }
    # if random, crit_1 and crit_2 are not used
    # if temporal, crit_1 is the date of the split and crit_2 not used
    # if spatial, crit_1 is the training locations and crit_2 is the test locations
    # remove the columns that we don't want to use
    print("Removing unused columns...")
    train_df = train_df.drop(columns=cols_dont_use)
    test_df = test_df.drop(columns=cols_dont_use)
    if val_df is not None:
        val_df = val_df.drop(columns=cols_dont_use)
    # normalize the train set across all features + label
    # one mean and std for each feature counting all lag days
    print("Normalizing train set...")
    train_df,feature_norm_df = normalize_dataset_across_lag_days(
        train_df,
        static_features,
        target_col
    )
    # apply the same normalization to the test set
    print("Normalizing test set...")
    for col in test_df.columns:
        mean = feature_norm_df.loc[
            feature_norm_df['feature'] == col, 'mean'
        ].values[0]
        std = feature_norm_df.loc[
            feature_norm_df['feature'] == col, 'std'
        ].values[0]
        test_df[col] = (test_df[col] - mean) / std
    if val_df is not None:
        print("Normalizing Validation set...")
        for col in val_df.columns:
            mean = feature_norm_df.loc[
                feature_norm_df['feature'] == col, 'mean'
            ].values[0]
            std = feature_norm_df.loc[
                feature_norm_df['feature'] == col, 'std'
            ].values[0]
            val_df[col] = (val_df[col] - mean) / std
    # plot the distributions of the features and labels across train and test sets
    if plot_distributions not in (None, False):
        # make sure that our plotting location exists
        os.makedirs(plot_distributions, exist_ok=True)
        # get column names independent of lag
        base_to_cols = get_lag_base_to_cols(train_df.columns)
        # loop over lag-independent column names
        for base_name, cols in base_to_cols.items():
            print('plotting train/test distribution for: {}'.format(base_name))
            # get the train values for this column across all lag days
            all_train_vals = np.array(pd.concat(
                [train_df[col] for col in cols], axis=0
            ))
            # get the train values for this column across all lag days
            all_test_vals = np.array(pd.concat(
                [test_df[col] for col in cols], axis=0
            ))
            # if we are plotting the validation set, get those values too
            if val_df is not None:
                all_val_vals = np.array(pd.concat(
                    [val_df[col] for col in cols], axis=0
                ))
            # call kde plot for these features across lag days
            this_fname = os.path.join(
                plot_distributions,
                f"{base_name}_distribution.png"
            )
            plotting.kde_plot(
                (
                    [all_train_vals, all_test_vals] +
                    ([all_val_vals] if val_df is not None else [])
                ),
                ['train', 'test'] + (['val'] if val_df is not None else []),
                this_fname,
                xlabel=base_name,
                ylabel='Density'
            )
    # get the X and y tensors for the transformer
    print("Creating train tensors...")
    train_X, train_y = get_X_y_from_df(
        train_df,
        static_features,
        target_col,
        np.array(lag_days_keep, dtype=np.float32),
        model_name
    )
    print("Creating test tensors...")
    test_X, test_y = get_X_y_from_df(
        test_df,
        static_features,
        target_col,
        np.array(lag_days_keep, dtype=np.float32),
        model_name
    )
    if val_df is not None:
        print("Creating validation tensors...")
        val_X, val_y = get_X_y_from_df(
            val_df,
            static_features,
            target_col,
            np.array(lag_days_keep, dtype=np.float32),
            model_name
        )
    print('Train features shape: {}'.format(train_X.shape))
    print('Train labels shape: {}'.format(train_y.shape))
    print('Test features shape: {}'.format(test_X.shape))
    print('Test labels shape: {}'.format(test_y.shape))
    # if we have a validation set, get the X and y tensors for it
    if val_df is not None:
        print('Validation features shape: {}'.format(val_X.shape))
        print('Validation labels shape: {}'.format(val_y.shape))
    # return the tensors
    to_return = (
        train_X,train_y,
        test_X,test_y,
        val_X,val_y,
        feature_norm_df,
        site_assignments
    )
    return to_return

def combine_csvs(csvs, start_date, end_date):
    # create one df from all of the csv paths
    created_final = False
    # get all the csv paths
    csv_paths = glob.glob(csvs)
    csv_paths.sort()
    for c,csv in enumerate(csv_paths):
        print(f"Processing CSV {c+1}/{len(csv_paths)}: {csv}")
        this_csv_start_str = csv.split('_')[-2]
        this_csv_end_str = csv.split('_')[-1].split('.')[0]
        this_csv_start = datetime.date(
            int(this_csv_start_str[:4]),
            int(this_csv_start_str[4:6]),
            int(this_csv_start_str[6:])
        )
        this_csv_end = datetime.date(
            int(this_csv_end_str[:4]),
            int(this_csv_end_str[4:6]),
            int(this_csv_end_str[6:])
        )
        # check if the csv is within the date range
        # we want to use if at all overlaps with my desired date range
        use_csv = (
            ((this_csv_start >= start_date) and (this_csv_start <= end_date)) or
            ((this_csv_end >= start_date) and (this_csv_end <= end_date)) or
            ((this_csv_start <= start_date) and (this_csv_end >= end_date)) or
            ((this_csv_start <= start_date) and (this_csv_end >= start_date)) or
            ((this_csv_start <= end_date) and (this_csv_end >= end_date))
        )
        if use_csv:
            this_df = pd.read_csv(csv)
            suspicious_cols = [
                col for col in this_df.columns
                if this_df[col].dropna().astype(str).head(50).map(is_list_like_string).any()
            ]
            for col in suspicious_cols:
                this_df[col] = this_df[col].astype(str).map(parse_list_first)
            # eliminate any rows that are outside of the date range
            dates_str = this_df['date']
            dates = pd.to_datetime(dates_str, format='%Y-%m-%d')
            conforming_dates = (
                (dates >= pd.Timestamp(start_date)) &
                (dates <= pd.Timestamp(end_date))
            )
            # get the index of the conforming dates
            conforming_dates_idx = conforming_dates[conforming_dates].index
            # filter the dataframe to only include conforming dates
            this_df_use = this_df.iloc[conforming_dates_idx]
            # check if we have created the final dataframe yet
            if not created_final:
                all_df = copy.deepcopy(this_df_use)
                created_final = True
            else:
                # append to the final dataframe
                all_df = pd.concat([all_df, this_df_use], ignore_index=True)
    return all_df

def parse_list_first(val):
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            # Handle known edge case where val is "[nan]"
            if val.lower() == "[nan]":
                return np.nan
            parsed = ast.literal_eval(val)
            return parsed[0] if parsed else np.nan
        except:
            return np.nan
    return val

def is_list_like_string(val):
    if not isinstance(val, str):
        return False
    val = val.strip()
    return val.startswith("[") and val.endswith("]")

def random_split(df, train_split):
    # randomly split the dataframe into train and test sets
    train_size = int(len(df) * train_split)
    train_df = df.sample(n=train_size, random_state=42)
    test_df = df.drop(train_df.index)
    crit_1 = None  # not used in random split
    crit_2 = None  # not used in random split
    return train_df, test_df, crit_1, crit_2

def temporal_split(df, train_split, split_type='percent'):
    df = df.sort_values(by='date')
    # split the dataframe into train and test sets based on date
    if split_type == 'percent':
        date_counts = df['date'].value_counts().sort_index()
        cum_counts = date_counts.cumsum()
        total_rows = len(df)
        split_row_target = int(total_rows * train_split)
        split_date = cum_counts[cum_counts >= split_row_target].index[0]
    elif split_type == 'date':
        # split based on a specific date
        split_date = pd.Timestamp(train_split)
    else:
        raise ValueError("split_type must be 'percent' or 'date'")
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]
    crit_1 = train_df['date'].max()  # last date in training set
    crit_2 = None  # not used in temporal split
    return train_df, test_df, crit_1, crit_2

def train_test_spatial_split(df, train_split):
    latlon_series = list(zip(df['latitude'], df['longitude']))
    latlon_counts = Counter(latlon_series)
    sorted_locations = list(latlon_counts.keys())
    # randomize the locations to ensure randomness in selection
    np.random.shuffle(sorted_locations)
    cumulative = 0
    train_locs = []
    total_rows = len(df)
    target_rows = int(total_rows * train_split)
    for loc in sorted_locations:
        if cumulative >= target_rows:
            break
        train_locs.append(loc)
        cumulative += latlon_counts[loc]
    # covert to set for efficient lookup
    train_locs_set = set(train_locs)
    # get our train/test
    mask = [loc in train_locs_set for loc in latlon_series]
    train_df = df[mask]
    test_df = df[[not m for m in mask]]
    test_locs = list(set(latlon_series) - train_locs_set)
    crit_1 = train_locs_set  # training locations
    crit_2 = set(test_locs)  # test locations
    return train_df, test_df, crit_1, crit_2

def val_spatial_split(
    df, minor_mid_major_split,filled_cols,filled_cols_labs,
    plot_distributions=None
):
    minor = minor_mid_major_split[0]
    mid = minor_mid_major_split[1]
    major = minor_mid_major_split[2]
    # get the locations
    latlon_series = list(zip(df['latitude'], df['longitude']))
    latlon_counts = Counter(latlon_series)
    sorted_locations = list(latlon_counts.keys())
    # randomize the locations to ensure randomness in selection
    np.random.shuffle(sorted_locations)
    cumulative = np.zeros(3, dtype=int)
    val_locs = [[], [], []]  # minor, mid, major
    total_rows = len(df)
    target_rows = np.array([
        int(total_rows * minor),
        int(total_rows * mid),
        int(total_rows * major)
    ])
    # how many modis observations are at each of our sites?
    cols = df.columns
    num_fill_inps_per_obs = 0
    for c in cols:
        for f in filled_cols:
            if f in c:
                num_fill_inps_per_obs += 1
    all_filled_perc = np.zeros(len(sorted_locations), dtype=float)
    for l,loc in enumerate(sorted_locations):
        # the true number of possible filled obs will be the number of filled cols
        # times the number of observations at this location
        loc_count = latlon_counts[loc]
        num_possible_filled_obs = loc_count * num_fill_inps_per_obs
        # get all of the columns for this location
        loc_mask = [l == loc for l in latlon_series]
        loc_df = df[loc_mask]
        # check how many modis observations are filled
        filled_count = 0
        for filled_col in filled_cols_labs:
            true_filled_col_labs = loc_df.columns[
                loc_df.columns.str.contains(
                    filled_col, case=False, regex=False
                )
            ]
            this_site_filled_col = loc_df[true_filled_col_labs].values
            # count how many are equal to 1
            filled_count += np.sum(this_site_filled_col == 1)
        filled_percent = filled_count / num_possible_filled_obs
        all_filled_perc[l] = filled_percent
        if filled_percent < 0.02 and cumulative[0] < target_rows[0]:
            # minor gap filled
            val_locs[0].append(loc)
            cumulative[0] += loc_count
        elif filled_percent >= 0.02 and filled_percent < 0.05 and cumulative[1] < target_rows[1]:
            # mid gap filled
            val_locs[1].append(loc)
            cumulative[1] += loc_count
        elif filled_percent >= 0.05 and cumulative[2] < target_rows[2]:
            # major gap filled
            val_locs[2].append(loc)
            cumulative[2] += loc_count
    # put val locs into a dataframe and remove from the original df
    val_locs_set = set(val_locs[0] + val_locs[1] + val_locs[2])
    val_mask = np.array(
        [loc in val_locs_set for loc in latlon_series]
    )
    val_df = df[val_mask].reset_index(drop=True)
    # remove the validation locations from the original dataframe
    df = df[~val_mask].reset_index(drop=True)
    # plot hte distribution of filled perc
    if plot_distributions not in (None, False):
        # plot the filled perc distribution
        plotting.kde_plot(
            [all_filled_perc],
            ['filled_percent'],
            os.path.join(plot_distributions, 'filled_percent_distribution.png'),
            xlabel='Filled Percent',
            ylabel='Density'
        )
    # return the validation dataframe and the original dataframe
    return val_df, df, val_locs[0], val_locs[1], val_locs[2]

def normalize_dataset_across_lag_days(df, static_features, target_col):
    df_norm = df.copy()
    stats = []
    base_to_cols = get_lag_base_to_cols(df.columns)
    # normalize each group
    for base, cols in base_to_cols.items():
        # combine values across lagged cols (if any) to compute mean and std
        all_vals = None
        for c,col in enumerate(cols):
            if c == 0:
                all_vals = df[col].values
            else:
                all_vals = np.concatenate((all_vals, df[col].values))
        mean = all_vals.mean()
        std = all_vals.std(ddof=0)
        std = std if std != 0 else 1 # avoid division by zero
        # normalize each column in the group
        for col in cols:
            df_norm[col] = (df[col] - mean) / std
            stats.append({
                'feature': col,
                'mean': mean,
                'std': std
            })
    # convert stats to a DataFrame
    stats_df = pd.DataFrame(stats)
    return df_norm, stats_df

def get_lag_base_to_cols(feature_cols):
    # group based on variable
    base_to_cols = {}
    for col in feature_cols:
        match = re.match(r'(.+)_day_minus_\d+$', col)
        if match:
            base = match.group(1)
            base_to_cols.setdefault(base, []).append(col)
        else:
            base_to_cols[col] = [col]  # static features or non-lagged features
    return base_to_cols

def extract_day_include(columns, static_features):
    day_include = set()
    for col in columns:
        if any(col.startswith(p) for p in static_features):
            continue
        match = re.search(r"_(\d+)$", col)
        day = int(match.group(1)) if match else 0
        day_include.add(day)
    return sorted(day_include)


def get_lag_day(col):
    match = re.search(r'day_minus_(\d+)', col)
    return int(match.group(1)) if match else -1  # static or non-lagged

def get_X_y_from_df(
    df,
    static_features,
    target_col,
    time_lag_vector,
    model_name
):
    # extract directly from the df the lagged days that we will be including
    all_cols = df.columns
    if target_col is not None:
        feature_names = all_cols.drop(target_col)
    else:
        feature_names = all_cols
    day_include = extract_day_include(feature_names, static_features)
    # now get the list of our dynamic features
    exclude_cols = set(static_features + [target_col])
    dynamic_cols = [
        col for col in feature_names if col not in exclude_cols
    ]
    grouped = defaultdict(list)
    for col in dynamic_cols:
        prefix = col.split('_day_minus_')[0] if '_day_minus_' in col else col
        grouped[prefix].append(col)
    lagged_features = []
    for var,col in grouped.items():
        sorted_cols = sorted(col, key=get_lag_day)
        lagged_features.extend(sorted_cols)
    seq_len = len(day_include)
    # create the tensors
    samples = []
    targets = []
    # get the base features
    base_lagged_features = set()
    for feat in lagged_features:
        if "_day_minus_" in feat:
            base = feat.split("_day_minus_")[0]
            base_lagged_features.add(base)
    base_lagged_features = sorted(base_lagged_features)
    for i,(idx,row) in enumerate(df.iterrows()):
        if i%1000 == 0:
            print(f"Creating tensor for df row {i} of {len(df)}")
        # create the input tensor
        lagged_vals = []
        for lag_day in day_include:
            step_features = []
            for base_feat in base_lagged_features:
                feat = f"{base_feat}_day_minus_{lag_day}"
                val = row[feat]
                step_features.append(val)
            lagged_vals.append(step_features)
        lagged_vals = np.array(lagged_vals, dtype=np.float32)
        # add static features
        static_vals = row[static_features].values.astype(np.float32)
        static_vals_rep = np.repeat(
            static_vals[np.newaxis, :], seq_len, axis=0
        )
        if model_name == 'transformer':
            time_lag_vector_norm = time_lag_vector / np.max(time_lag_vector)
            time_lag_feature = time_lag_vector_norm[:, np.newaxis]
            sample_features = np.concatenate(
                [lagged_vals, static_vals_rep, time_lag_feature], axis=1
            )
        elif model_name == 'temporal_cnn':
            sample_features = np.concatenate(
                [lagged_vals, static_vals_rep], axis=1
            )
        else:
            raise ValueError(f"Model name {model_name} not recognized.")
        samples.append(sample_features)
        if target_col is not None:
            targets.append(row[target_col])
    # convert to tensors
    X = torch.tensor(np.array(samples), dtype=torch.float32)
    if target_col is None:
        y = None
    else:
        y = torch.tensor(np.array(targets), dtype=torch.float32)
    return X, y


if __name__ == "__main__":
    # set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # fill in follwing necessary information for producing the correct dataset
    csv_names = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/compiled/'
        'y_Insitu_X_ModisfilledDaymetStaticKrishnastatsWeatherstats_30days/'
        'compiled_data_*.csv'
    )
    first_label_date = datetime.date(2003, 1, 1)
    last_label_date = datetime.date(2023, 12, 31)
    static_features = [
        'slope','elevation','canopy_height','forest_cover',
        'clay','sand','latitude','longitude'
        #'retrieved_lfmc_mean',
        #'retrieved_lfmc_std','retrieved_lfmc_min','retrieved_lfmc_max',
        #'retrieved_lfmc_djf_mean','retrieved_lfmc_mam_mean',
        #'retrieved_lfmc_jja_mean','retrieved_lfmc_son_mean'
    ]
    lagged_features = [
        'srad','prcp','swe','tmax','tmin','vp',
        'Nadir_Reflectance_Band1_filled',
        'Nadir_Reflectance_Band2_filled',
        'Nadir_Reflectance_Band3_filled',
        'Nadir_Reflectance_Band4_filled',
        'Nadir_Reflectance_Band5_filled',
        'Nadir_Reflectance_Band6_filled',
        'Nadir_Reflectance_Band7_filled'
        'days_since_rain','max_precip_14_days',
        'rolling_precip_14_days','max_temp_14_days',
        'rolling_temp_14_days','max_vp_14_days',
        'rolling_vp_14_days'
    ]
    target_col = 'lfmc'
    filled_cols = [
        'Nadir_Reflectance_Band1_filled',
        'Nadir_Reflectance_Band2_filled',
        'Nadir_Reflectance_Band3_filled',
        'Nadir_Reflectance_Band4_filled',
        'Nadir_Reflectance_Band5_filled',
        'Nadir_Reflectance_Band6_filled',
        'Nadir_Reflectance_Band7_filled'
    ]
    filled_cols_labs = [
        'filled_1',
        'filled_2',
        'filled_3',
        'filled_4',
        'filled_5',
        'filled_6',
        'filled_7'
    ]
    #lag_days_keep = [
    #    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #    21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    #]
    lag_days_keep = [
        0,1,2,3,4,7,10,15,20,25,30
    ]
    # percent of data that goes to validation set
    # minor, mid, major split
    # minor is sites with < 1% gap filled, mid is sites with 1-5% gap filled,
    # major is sites with > 5% gap filled
    # generally set to mirror distribution in gap filling across all data
    minor_mid_major_split = (0.05, 0.04, 0.01) # ex: (0.07, 0.02, 0.01) or None
    train_split = 0.85
    split_type = 'spatial'
    model_name = 'transformer'
    features_name = 'ModisDaymetStaticLatlonWeatherstats'
    labels_name = 'Insitu'
    lag_days_name = 'intermittent' # full or intermittent
    start_date_name = first_label_date.strftime('%Y%m%d')
    end_date_name = last_label_date.strftime('%Y%m%d')
    train_on_filled = False  # whether to train on filled data only
    gen_file_out = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/'
        'splits/{model_name}_{split_type}_{dataset_type}_{start_date_name}_{end_date_name}_'
        'y_{labels_name}_x_{features_name}.npy'
    )
    gen_file_out_filled = gen_file_out.format(
        model_name=model_name,
        split_type=split_type,
        dataset_type='{dataset_type}',
        start_date_name=start_date_name,
        end_date_name=end_date_name,
        labels_name=labels_name,
        features_name=features_name
    )
    train_file_out = gen_file_out_filled.format(dataset_type='train')
    test_file_out = gen_file_out_filled.format(dataset_type='test')
    val_file_out = gen_file_out_filled.format(dataset_type='val')
    plot_distributions = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/outputs/viz/'
        'distributions/{model_name}_{split_type}_'
        '{start_date_name}_{end_date_name}_y_{labels_name}_x_{features_name}'.format(
            model_name=model_name,
            split_type=split_type,
            start_date_name=start_date_name,
            end_date_name=end_date_name,
            labels_name=labels_name,
            features_name=features_name
        )
    )
    #plot_distributions = False
    norm_df_out = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/'
        'norm_df/{model_name}_{split_type}_'
        '{start_date_name}_{end_date_name}_y_{labels_name}_x_{features_name}.csv'
    )
    norm_df_out = norm_df_out.format(
        model_name=model_name,
        split_type=split_type,
        start_date_name=start_date_name,
        end_date_name=end_date_name,
        labels_name=labels_name,
        features_name=features_name
    )
    site_locs_out = (
        '/scratch/users/trobinet/long_lfmc/trent_datasets/lfmc_model/data/'
        'site_assignments/{model_name}_{split_type}_'
        '{start_date_name}_{end_date_name}_y_{labels_name}_x_{features_name}.pickle'
    )
    site_locs_out = site_locs_out.format(
        model_name=model_name,
        split_type=split_type,
        start_date_name=start_date_name,
        end_date_name=end_date_name,
        labels_name=labels_name,
        features_name=features_name
    )
    # make sure the output directories exist
    os.makedirs(os.path.dirname(train_file_out), exist_ok=True)
    os.makedirs(os.path.dirname(test_file_out), exist_ok=True)
    os.makedirs(os.path.dirname(val_file_out), exist_ok=True)
    os.makedirs(os.path.dirname(norm_df_out), exist_ok=True)
    os.makedirs(os.path.dirname(site_locs_out), exist_ok=True)
    if plot_distributions is not False:
        os.makedirs(plot_distributions, exist_ok=True)
    (
        train_X,train_y,
        test_X,test_Y,
        val_X,val_y,
        norm_df,
        site_assignments
    ) = build_training_dataset(
        csv_names,
        first_label_date,
        last_label_date,
        static_features,
        lagged_features,
        target_col,
        lag_days_keep,
        val_minor_mid_major_split=minor_mid_major_split,
        train_split=train_split,
        split_type=split_type,
        plot_distributions=plot_distributions,
        model_name=model_name,
        train_on_filled=train_on_filled,
        filled_cols=filled_cols,
        filled_cols_labs=filled_cols_labs
    )
    torch.save({"X": train_X, "y": train_y}, train_file_out)
    torch.save({"X": test_X, "y": test_Y}, test_file_out)
    if val_X is not None and val_y is not None:
        torch.save({"X": val_X, "y": val_y}, val_file_out)
    norm_df.to_csv(norm_df_out, index=False)
    with open(site_locs_out, 'wb') as f:
        pickle.dump(site_assignments, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Train dataset saved to {train_file_out}")
    print(f"Test dataset saved to {test_file_out}")
    if val_X is not None and val_y is not None:
        print(f"Validation dataset saved to {val_file_out}")
    print(f"Normalization statistics saved to {norm_df_out}")
    print(f"Site assignments saved to {site_locs_out}")
