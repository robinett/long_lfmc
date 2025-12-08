import pandas as pd
import sys
import numpy as np
import copy
import xarray as xr
from pyproj import Transformer
import re
import os

def classify_fast(name,rules):
    name_l = name.lower()
    for lc, pats in rules.items():
        for p in pats:
            if re.search(p, name_l):
                return lc
    return 'no_match'

def get_circle_mask(ds, lat, lon, radius_m=400, transformer=None):
    """
    Return (circle_mask, subset) where circle_mask is
    an xarray.DataArray (boolean) aligned with subset['nlcd'].
    """
    if transformer is None:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x0, y0 = transformer.transform(lon, lat)
    # bounding box slice
    x_slice = slice(x0 - radius_m, x0 + radius_m)
    y_slice = slice(y0 + radius_m, y0 - radius_m)  # y usually decreasing
    sub = ds.sel(x=x_slice, y=y_slice)
    # xarray will broadcast coords for us
    X = xr.DataArray(sub["x"]).broadcast_like(sub["nlcd"])
    Y = xr.DataArray(sub["y"]).broadcast_like(sub["nlcd"])
    dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    circle = dist <= radius_m
    return circle, sub


def process(
    orig_fname,
    nfmd_loc_fname,
    nlcd_fname,
    species_to_landcover_name,
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
    #print('fuel type')
    #print(len(orig['fuel_type'].unique()))
    #print('category')
    #print(len(orig['category'].unique()))
    #print('sub_category')
    #print(orig['sub_category'].unique())
    #sys.exit()
    # if species to landcover doensn't exist, create it
    if not os.path.exists(species_to_landcover_name):
        rules = {
            "evergreen_forest": [
                r"pine", r"fir", r"spruce", r"hemlock", r"juniper",
                r"cedar", r"lodgepole", r"pinyon"
            ],
            "deciduous_forest": [
                r"oak", r"maple", r"aspen", r"birch", r"cottonwood",
                r"willow", r"ash", r"cherry"
            ],
            "shrub": [
                r"sage", r"manzanita", r"rabbitbrush",
                r"mesquite", r"sumac", r"ceanothus", r"brittle", r"horsebrush"
            ],
            "grass": [
                r"grass", r"grama", r"brome", r"sedge",
                r"reed", r"pinegrass", r"wildrye", r"squirreltail",
            ],
        }
        species_list = orig['fuel_type'].unique()
        # standarize
        draft_map = {
            species: classify_fast(species, rules)
            for species in species_list
        }
        df_map = pd.DataFrame({
            'species': list(draft_map.keys()),
            'landcover': list(draft_map.values())
        })
        df_map.to_csv(species_to_landcover_name, index=False)
        print('Created species to landcover mapping at', species_to_landcover_name)
        print('Go check it! Exiting.')
        sys.exit()
    species_to_landcover = pd.read_csv(species_to_landcover_name)
    # check if there is still any no_match
    no_match_species = species_to_landcover[
        species_to_landcover['landcover'] == 'no_match'
    ]['species'].values
    if len(no_match_species) > 0:
        print('The following species have no landcover match:')
        for s in no_match_species:
            print('-', s)
        print('Please update the species to landcover mapping file at:')
        print(species_to_landcover_name)
        print('Exiting.')
        sys.exit()
    # add this mapping to orig
    species_to_landcover_dict = {
        row['species']: row['landcover']
        for _, row in species_to_landcover.iterrows()
    }
    orig['landcover'] = orig['fuel_type'].map(species_to_landcover_dict)
    loc_data = pd.read_csv(nfmd_loc_fname)
    loc_data = loc_data.set_index('Site ID')
    loc_data_idx = sorted(np.array(loc_data.index))
    cols = list(orig.columns)
    cols.append('latitude')
    cols.append('longitude')
    # load up the land cover database so that we can check the surrounding land cover
    nlcd = xr.open_zarr(nlcd_fname)
    tfm = Transformer.from_crs("EPSG:4326",nlcd.rio.crs,always_xy=True)
    tfm_back = Transformer.from_crs(nlcd.rio.crs,"EPSG:4326",always_xy=True)
    # get rid of any measurements outside the bounds of our target grid
    for i,row in loc_data.iterrows():
        this_lat = row['Latitude']
        this_lon = row['Longitude']
        this_x,this_y = tfm.transform(this_lon,this_lat)
        if (
            this_x < nlcd['x'].min().values or
            this_x > nlcd['x'].max().values or
            this_y < nlcd['y'].min().values or
            this_y > nlcd['y'].max().values
        ):
            # remove all entries from orig with this site id
            orig = orig[orig['site_id'] != i]
    # for each site, go through and eliminate all times outside of relevant
    # time period
    # then check if there are multiple species. if multiple species, check if
    # the species have pearson's r > 0.8. If so, average them at eahc time
    # point. otherwise rid.
    # get the unique site names
    # instead of using site names, we need to get all lat/lon pairs that are in
    # common pixels, using the grid from the nlcd data
    site_ids = orig['site_id'].unique()
    site_coord_indices = []
    grid_xs = nlcd['x'].values
    grid_ys = nlcd['y'].values
    for s,site in enumerate(site_ids):
        site_data = orig[orig['site_id'] == site]
        this_lat = np.array(np.unique(loc_data.loc[site]['Latitude']))
        this_lon = np.array(np.unique(loc_data.loc[site]['Longitude']))
        # make sure that the site has only one lat/lon
        if type(this_lat) == np.ndarray and len(this_lat) != 1:
            print('Site has multiple latitudes:', site)
            sys.exit()
        if type(this_lon) == np.ndarray and len(this_lon) != 1:
            print('Site has multiple longitudes:', site)
            sys.exit()
        x,y = tfm.transform(this_lon,this_lat)
        # get the corresponding indexes in the nlcd data
        this_x = nlcd['x'].sel(x=x, method='nearest')
        this_y = nlcd['y'].sel(y=y, method='nearest')
        this_x_idx = int(np.where(grid_xs == this_x.values)[0][0])
        this_y_idx = int(np.where(grid_ys == this_y.values)[0][0])
        site_coord_indices.append((this_x_idx, this_y_idx))
    # group sites by their coordinate indices
    latlon_to_sites = {}
    for i,idxs in enumerate(site_coord_indices):
        this_x = grid_xs[idxs[0]]
        this_y = grid_ys[idxs[1]]
        this_lat, this_lon = tfm_back.transform(this_x, this_y)
        if f'{this_lat}_{this_lon}' not in latlon_to_sites:
            latlon_to_sites[f'{this_lat}_{this_lon}'] = []
        latlon_to_sites[f'{this_lat}_{this_lon}'].append(site_ids[i])
    for latlon, sites in latlon_to_sites.items():
        if len(sites) > 1:
            print(f'Lat/Lon {latlon} has multiple sites:')
            for s in sites:
                print('-', s)
    #site_names = orig['site_name'].unique()
    final_created = False
    total_samples_removed = 0
    for l,latlon in enumerate(latlon_to_sites.keys()):
        this_lat = float(latlon.split('_')[0])
        this_lon = float(latlon.split('_')[1])
        sites = latlon_to_sites[latlon]
        #if orig[orig['site_id'] == site]['site_name'].values[0] != 'Red Canyon':
        #    continue
        # get the data for this site
        site_data = orig[orig['site_id'].isin(sites)]
        #site_name = site_data['site_name'].values[0]
        print(f'Processing pixel {l+1}/{len(latlon_to_sites)}: {latlon}')
        # get the coordinates for this site
        x,y = tfm.transform(this_lat,this_lon)
        ## check if the coordinates are in the bounding box
        #if (
        #    this_lat < bound_box[1] or
        #    this_lat > bound_box[3] or
        #    this_lon < bound_box[0] or
        #    this_lon > bound_box[2]
        #):
        #    continue
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
        # get the unique list of years
        years = site_data['date'].dt.year.unique()
        years = np.sort(years)
        # check the land cover for each year. if doesn't match, skip
        keep_years = []
        classes = [
            'barren',
            'crops',
            'deciduous_forest',
            'developed',
            'evergreen_forest',
            'grass',
            'mixed_forest',
            'shrub',
            'water',
            'wetlands'
        ]
        classes_dict = {
            11:'water',
            12:'water',
            21:'developed',
            22:'developed',
            23:'developed',
            24:'developed',
            31:'barren',
            41:'deciduous_forest',
            42:'evergreen_forest',
            43:'mixed_forest',
            52:'shrub',
            71:'grass',
            81:'crops',
            82:'crops',
            90:'wetlands',
            95:'wetlands'
        }
        unexpected_classes = [
            'barren',
            'crops',
            'developed',

        ]
        for year in years:
            # get the land cover for this year
            nlcd_year = nlcd.sel(
                x=x, y=y, method='nearest'
            ).sel(year=str(year))
            # get the prevalence of unexpected classes
            total_unexpected_perc = 0.0
            for c,cla in enumerate(classes):
                if cla not in unexpected_classes:
                    continue
                this_unexpected_perc = nlcd_year[cla].values[0]
                total_unexpected_perc += this_unexpected_perc
            if total_unexpected_perc >= 0.25:
                print(f'Removing year {year} due to unexpected land cover presence: {total_unexpected_perc:.2f}')
                samples_before = site_data.shape[0]
                site_data = site_data[site_data['date'].dt.year != year]
                samples_after = site_data.shape[0]
                total_samples_removed += (samples_before - samples_after)
        # make sure that we haven't gotten rid of all the data
        if site_data.shape[0] == 0:
            print('No data left after land cover check')
            continue
        # get the unique species names
        species_names = site_data['fuel_type'].unique()
        # if there is only one species, just keep it
        if len(species_names) == 1:
            # if there are dates with multiple entries, take the average for lfmc
            site_data = site_data.groupby(['date'],as_index=False).agg({
                'sample_id':'first',
                'site_id':'first',
                'category':'first',
                'sub_category':'first',
                'method':'first',
                'sample_status':'first',
                'site_name':'first',
                'fuel_type':'first',
                'lfmc':'mean',
                'landcover':'first'
            })
            num_rows = site_data.shape[0]
            lats = np.full(num_rows, this_lat)
            lons = np.full(num_rows, this_lon)
            site_data['latitude'] = lats
            site_data['longitude'] = lons
            # confirm there are no duplicate date entries
            date_counts = site_data['date'].value_counts()
            duplicate_dates = date_counts[date_counts > 1]
            if duplicate_dates.shape[0] > 0:
                print(site_data)
                raise ValueError('Still found duplicate date entries after consolidating species')
            if not final_created:
                final = copy.deepcopy(site_data)
                final_created = True
            else:
                final = pd.concat([final, site_data])
        else:
            site_data = site_data.sort_values(by='date')
            pivot_df = site_data.pivot_table(
                index='date',columns='fuel_type',values='lfmc'
            )
            corr_matrix = pivot_df.corr()
            corr_matrix_np = np.array(corr_matrix)
            non_diag = corr_matrix_np[~np.eye(corr_matrix_np.shape[0],dtype=bool)]
            all_above_threshold = (non_diag > 0.8).all()
            # if all above threshold, average lfmc at each timepoint
            #all_above_threshold = True
            if all_above_threshold:
                # average the lfmc at each timepoint, only if we have measurements from all species for that day
                pivot_df_clean = pivot_df.dropna()
                avg_lfmc = pivot_df_clean.mean(axis=1)
                # get a single string that is all the species we averaged
                # together
                species_str = '; '.join(species_names)
                num_rows = avg_lfmc.shape[0]
                lats = np.full(num_rows, this_lat)
                lons = np.full(num_rows, this_lon)
                # check if the landcover si the same; if not, add 'mixed_sample'
                landcover_list = [
                    species_to_landcover_dict[sp]
                    for sp in species_names
                ]
                landcover_set = set(landcover_list)
                if len(landcover_set) == 1:
                    landcover = landcover_list[0]
                else:
                    landcover = 'mixed_sample'
                # make sure there are no duplicate date entries
                date_counts = avg_lfmc.index.value_counts()
                duplicate_dates = date_counts[date_counts > 1]
                if duplicate_dates.shape[0] > 0:
                    print(avg_lfmc)
                    raise ValueError('Still found duplicate date entries after consolidating species')
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
                    'longitude':lons,
                    'landcover':landcover
                })
                if not final_created:
                    final = copy.deepcopy(avg_df)
                    final_created = True
                else:
                    final = pd.concat([final, avg_df])
    num_timepoints = final.shape[0]
    num_unique_sites = final['site_name'].nunique()
    print('total samples removed due to land cover check:', total_samples_removed)
    print('num measurements:', num_timepoints)
    print('num unique sites:', num_unique_sites)
    # write the final dataframe to a csv
    final.to_csv(out_fname, index=False)
