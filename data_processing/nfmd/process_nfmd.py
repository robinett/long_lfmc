import pandas as pd
import sys
import numpy as np
import copy

def process(
    orig_fname,
    nfmd_loc_fname,
    start,
    end,
    bound_box,
    out_fname
):
    orig = pd.read_csv(orig_fname,dtype={6:str})
    # let's rename some columns here to make life easier
    orig = orig.rename(columns={
        'Sample Id':'sample_id',
        'Date-Time (ex. 2024-01-26T00:00:00+00:00)':'date',
        'Site Name':'site_name',
        'SiteId':'site_id',
        'Fuel Type':'fuel_type',
        'Category':'category',
        'Sub-Category':'sub_category',
        'Method':'method',
        'Sample Avg Value':'lfmc',
        'Sample Status':'sample_status'
    })
    orig['date'] = pd.to_datetime(orig['date'])
    # we don't want dead fuel moisture. Remove this.
    orig = orig[orig['category'] != 'Dead']
    loc_data = pd.read_csv(nfmd_loc_fname)
    loc_data = loc_data.set_index('Site Name')
    loc_data_idx = sorted(np.array(loc_data.index))
    #print([s for s in loc_data_idx if 'Rid' in s])
    #print([s for s in loc_data_idx if 'rid' in s])
    #print([s for s in loc_data_idx if 'RID' in s])
    #print([s for s in loc_data_idx if 'Cor' in s])
    cols = list(orig.columns)
    cols.append('latitude')
    cols.append('longitude')
    # for each site, go through and eliminate all times outside of relevant
    # time period
    # then check if there are multiple species. if multiple species, check if
    # the species have pearson's r > 0.5. If so, average them at eahc time
    # point. otherwise rid.
    # get the unique site names
    site_names = orig['site_name'].unique()
    final_created = False
    for s,site in enumerate(site_names):
        print(f'Processing site {s+1}/{len(site_names)}: {site}')
        # get the data for this site
        site_data = orig[orig['site_name'] == site]
        # get the coordinates for this site
        this_lat = loc_data.loc[site]['Latitude']
        this_lon = loc_data.loc[site]['Longitude']
        # make sure that we only got one lat and lon
        if isinstance(this_lat, pd.Series):
            this_lat = this_lat.iloc[0]
            this_lon = this_lon.iloc[0]
        # check if the coordinates are in the bounding box
        if (
            this_lat < bound_box[1] or
            this_lat > bound_box[3] or
            this_lon < bound_box[0] or
            this_lon > bound_box[2]
        ):
            continue
        # eliminate data outside of relevant time period
        site_data = site_data[
            (
                site_data['date'] >= pd.Timestamp(start).tz_localize('UTC')
            ) & (
                site_data['date'] <= pd.Timestamp(end).tz_localize('UTC')
            )
        ]
        # if there is no data, skip this site
        if site_data.shape[0] == 0:
            continue
        # get the unique species names
        species_names = site_data['fuel_type'].unique()
        # if there is only one species, just keep it
        if len(species_names) == 1:
            #print('one species')
            num_rows = site_data.shape[0]
            lats = np.full(num_rows, this_lat)
            lons = np.full(num_rows, this_lon)
            site_data['latitude'] = lats
            site_data['longitude'] = lons
            if not final_created:
                final = copy.deepcopy(site_data)
                final_created = True
            else:
                final = pd.concat([final, site_data])
        else:
            #print('multiple species')
            site_data = site_data.sort_values(by='date')
            pivot_df = site_data.pivot_table(
                index='date',columns='fuel_type',values='lfmc'
            )
            corr_matrix = pivot_df.corr()
            corr_matrix_np = np.array(corr_matrix)
            non_diag = corr_matrix_np[~np.eye(corr_matrix_np.shape[0],dtype=bool)]
            #print(corr_matrix)
            all_above_threshold = (non_diag > 0.5).all()
            # if all above threshold, average lfmc at each timepoint
            #all_above_threshold = True
            if all_above_threshold:
                # average the lfmc at each timepoint
                avg_lfmc = pivot_df.mean(axis=1)
                # get a single string that is all the species we averaged
                # together
                species_str = '; '.join(species_names)
                num_rows = avg_lfmc.shape[0]
                lats = np.full(num_rows, this_lat)
                lons = np.full(num_rows, this_lon)
                # create a new dataframe with the averaged lfmc
                avg_df = pd.DataFrame({
                    'sample_id':site_data['sample_id'].iloc[0],
                    'site_id':site_data['site_id'].iloc[0],
                    'category':site_data['category'].iloc[0],
                    'sub_category':site_data['sub_category'].iloc[0],
                    'method':site_data['method'].iloc[0],
                    'sample_status':site_data['sample_status'].iloc[0],
                    'site_name':site,
                    'date':avg_lfmc.index,
                    'fuel_type':species_str,
                    'lfmc':avg_lfmc.values,
                    'latitude':lats,
                    'longitude':lons
                })
                if not final_created:
                    final = copy.deepcopy(avg_df)
                    final_created = True
                else:
                    final = pd.concat([final, avg_df])
    print(final)
    num_timepoints = final.shape[0]
    num_unique_sites = final['site_name'].nunique()
    print('num measurements:', num_timepoints)
    print('num unique sites:', num_unique_sites)
    # write the final dataframe to a csv
    final.to_csv(out_fname, index=False)
