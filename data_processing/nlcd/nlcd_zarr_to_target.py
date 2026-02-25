import os
import xarray as xr
import numpy as np
import sys
from pyproj import Transformer
from dask.diagnostics import ProgressBar
from tqdm import tqdm

sys.path.append(os.path.join(
    os.path.dirname(__file__),
    '..',
    'shared'
))

import plotting

def main():
    data_dir = '/scratch/users/trobinet/long_lfmc/final_lfmc'
    # get the nlcd data
    print('getting original nlcd data...')
    nlcd_zarr_path = os.path.join(
        data_dir,
        'nlcd',#'nlcd_full',
        #'nlcd.zarr'
        'nlcd_2000_2024.zarr'
    )
    nlcd_orig = xr.open_zarr(nlcd_zarr_path)
    nlcd_orig = nlcd_orig.sortby('x')
    nlcd_orig = nlcd_orig.sortby('y')
    print('original nlcd:')
    print(nlcd_orig)
    nlcd_dict = {
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
    # get the unique land covers that we are looking for
    unique_lc = list(set(nlcd_dict.values()))
    classes = unique_lc + ["other"]
    n_classes = len(classes)
    # map from NLCD code -> class index
    max_code = max(nlcd_dict.keys())  # e.g. 95
    code_to_class_idx = np.full(max_code + 1, n_classes - 1, dtype=np.int16)
    # default is "other" at index n_classes-1
    for lc_idx, lc_name in enumerate(unique_lc):
        # add all codes mapping to this lc_name
        for code, name in nlcd_dict.items():
            if name == lc_name:
                code_to_class_idx[code] = lc_idx
    #nlcd_orig = nlcd_orig.assign(
    #    water=lambda ds: ds.landcover.isin([11, 12]).astype("uint8"),
    #    developed=lambda ds: ds.landcover.isin([21, 22, 23, 24]).astype("uint8"),
    #    barren=lambda ds: (ds.landcover == 31).astype("uint8"),
    #    deciduous_forest=lambda ds: (ds.landcover == 41).astype("uint8"),
    #    evergreen_forest=lambda ds: (ds.landcover == 42).astype("uint8"),
    #    mixed_forest=lambda ds: (ds.landcover == 43).astype("uint8"),
    #    shrub=lambda ds: (ds.landcover == 52).astype("uint8"),
    #    grass=lambda ds: (ds.landcover == 71).astype("uint8"),
    #    crops=lambda ds: ds.landcover.isin([81, 82]).astype("uint8"),
    #    wetlands=lambda ds: ds.landcover.isin([90, 95]).astype("uint8"),
    #).rio.write_crs("EPSG:4326").rename({"lat": "y", "lon": "x"})
    #nlcd_orig = nlcd_orig.rio.write_crs("EPSG:4326").rename({"lat": "y", "lon": "x"})
    nlcd_years = nlcd_orig['time'].values
    # open up our target grid
    print('creating our empty target dataset...')
    target_grid_path = os.path.join(
        data_dir,
        'grid',
        'epsg5070_500m_westUS_grid.nc4'
    )
    target_grid = xr.open_dataset(target_grid_path)
    target_grid = target_grid.sortby('x')
    target_grid = target_grid.sortby('y')
    target_nlcd = target_grid.copy().drop_vars('random_vals')
    # add 'year' dimension to target_nlcd
    target_nlcd = target_nlcd.expand_dims({'year': nlcd_years})
    target_nlcd = target_nlcd.assign_coords({'year': nlcd_years})
    ex_vals = target_grid['random_vals'].values.copy()
    ex_vals[:] = np.nan
    # add year dimension of len (nlcd_years) to ex_vals
    ex_vals = np.repeat(
        ex_vals[:, :, np.newaxis],
        len(nlcd_years),
        axis=2
    )
    for lc in unique_lc:
        target_nlcd[lc] = (('y', 'x','year'), ex_vals.copy())
        target_grid[lc] = (('y', 'x','year'), ex_vals.copy())
    target_nlcd['other'] = (('y', 'x','year'), ex_vals.copy())
    target_grid['other'] = (('y', 'x','year'), ex_vals.copy())
    x_res = np.abs(
        target_grid['x'].values[1] - target_grid['x'].values[0]
    )
    y_res = np.abs(
        target_grid['y'].values[1] - target_grid['y'].values[0]
    )
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    # break our target dataset into 100x100 boxes so we can load all nlcd for those boxes
    # at once, then move onto the next
    x_len = target_grid.sizes['x']
    y_len = target_grid.sizes['y']
    box_size = 500
    x_box_dim = np.ceil(x_len / box_size).astype(int)
    y_box_dim = np.ceil(y_len / box_size).astype(int)
    count = 0
    for y in range(y_box_dim):
        for x in range(x_box_dim):
            box_xr = target_grid.isel(
                x=slice(x*box_size, min((x+1)*box_size, x_len)),
                y=slice(y*box_size, min((y+1)*box_size, y_len))
            )
            box = np.array([
                np.min(box_xr['x'].values),
                np.max(box_xr['x'].values),
                np.min(box_xr['y'].values),
                np.max(box_xr['y'].values)
            ])
            if count == 0:
                all_boxes = box[np.newaxis, :]
            else:
                all_boxes = np.vstack((all_boxes, box[np.newaxis, :]))
            count += 1
    boxes_used = 0
    #for b,box in enumerate(all_boxes):
    for b,box in tqdm(enumerate(all_boxes), total=len(all_boxes), desc='Processing boxes'):
        print(f"Processing box {b+1} / {len(all_boxes)}")
        # load nlcd for this
        min_x_box, max_x_box, min_y_box, max_y_box = box
        min_x_box_res = min_x_box - x_res
        max_x_box_res = max_x_box + x_res
        min_y_box_res = min_y_box - y_res
        max_y_box_res = max_y_box + y_res
        #min_x_box_4326, min_y_box_4326 = transformer.transform(min_x_box_res, min_y_box_res)
        #max_x_box_4326, max_y_box_4326 = transformer.transform(max_x_box_res, max_y_box_res)
        #print(min_x_box_4326, max_x_box_4326, min_y_box_4326, max_y_box_4326)
        print("   loading target subset...")
        target_subset = target_grid.sel(
            x=slice(min_x_box, max_x_box),
            y=slice(min_y_box, max_y_box)
        ).copy()
        valid_mask = ~np.isnan(
            target_subset["random_vals"].values
        )  # shape (ny_c, nx_c), True where pixel is inside domain
        # if there are no valid pixels in this box, skip it
        if np.all(~valid_mask):
            print("  no valid pixels in this box, skipping...")
            continue
        
        print("   loading nlcd subset...")
        nlcd_subset = nlcd_orig.sel(
            x=slice(min_x_box_res, max_x_box_res),
            y=slice(min_y_box_res, max_y_box_res)
        ).compute()
        # build mappings so we can use numpy
        print("   building mappings...")
        nlcd_x = nlcd_subset["x"].values      # (nx_f,)
        nlcd_y = nlcd_subset["y"].values      # (ny_f,)
        nlcd_data = nlcd_subset["nlcd"].values   # shape (time, ny_f, nx_f)
        # get rid of nan and make int
        nlcd_data_mask = np.isnan(nlcd_data)
        #nlcd_data = nlcd_data[~nlcd_data_mask]
        nlcd_data = nlcd_data.astype(np.int32)
        #print(nlcd_data)
        x_t = target_subset["x"].values       # (nx_c,)
        y_t = target_subset["y"].values       # (ny_c,)
        dx = np.abs(x_t[1] - x_t[0])
        dy = np.abs(y_t[1] - y_t[0])
        # build edges for coarse cells in x/y
        x_edges = np.empty(x_t.size + 1)
        x_edges[1:-1] = (x_t[:-1] + x_t[1:]) / 2.0
        x_edges[0] = x_t[0] - dx / 2.0
        x_edges[-1] = x_t[-1] + dx / 2.0
        y_edges = np.empty(y_t.size + 1)
        y_edges[1:-1] = (y_t[:-1] + y_t[1:]) / 2.0
        y_edges[0] = y_t[0] - dy / 2.0
        y_edges[-1] = y_t[-1] + dy / 2.0
        # map fine coords -> coarse indices
        fine_x_idx = np.searchsorted(x_edges, nlcd_x) - 1  # (nx_f,)
        fine_y_idx = np.searchsorted(y_edges, nlcd_y) - 1  # (ny_f,)
        # 2D arrays of coarse indices for every fine pixel
        fine_x_idx_2d, fine_y_idx_2d = np.meshgrid(
            fine_x_idx, fine_y_idx
        )  # shape (ny_f, nx_f) each
        # mask out fine pixels that map outside this coarse box
        inside = (
            (fine_x_idx_2d >= 0) & (fine_x_idx_2d < x_t.size) &
            (fine_y_idx_2d >= 0) & (fine_y_idx_2d < y_t.size)
        )
        fine_x_idx_2d = fine_x_idx_2d[inside]
        fine_y_idx_2d = fine_y_idx_2d[inside]
        # flatten mapping: which coarse pixel does each fine pixel hit?
        flat_coarse_idx = (
            fine_y_idx_2d * x_t.size + fine_x_idx_2d
        )  # (n_valid_fine_pixels,)
        n_coarse = x_t.size * y_t.size
        n_years = nlcd_data.shape[0]
        # prepare output for this box:
        # (n_classes, ny_c, nx_c, n_years)
        box_counts = np.zeros(
            (n_classes, y_t.size, x_t.size, n_years),
            dtype=np.int32,
        )
        # loop over years ONLY (no x/y loops)
        for t in range(n_years):
            print(f"     processing year {nlcd_years[t]} ({t+1} / {n_years})")
            # fine NLCD for this year
            lc = nlcd_data[t, :, :]      # (ny_f, nx_f)
            lc = lc[inside]              # (n_valid_fine_pixels,)
            # mask out nodata / "nan" code
            valid_lc = lc != -2147483648
            if not np.any(valid_lc):
                # no valid fine pixels for this year in this box
                # leave box_counts[:, :, :, t] as zeros and continue
                continue
            # keep only valid landcover codes and matching coarse indices
            lc_valid = lc[valid_lc]
            flat_coarse_idx_valid = flat_coarse_idx[valid_lc]
            # map codes -> class index, safely
            # (if lc_valid is int32 this is fine; if not, cast)
            #lc_valid = lc_valid.astype(np.int16)
            #lc_valid = np.clip(lc_valid, 0, 255)
            lc_idx = code_to_class_idx[lc_valid]   # (n_valid_lc,)
            # flatten [coarse_pixel, class] and accumulate
            year_counts = np.zeros(n_coarse * n_classes, dtype=np.int32)
            flat_idx = flat_coarse_idx_valid * n_classes + lc_idx
            np.add.at(year_counts, flat_idx, 1)
            # reshape to (ny, nx, n_classes)
            year_counts = year_counts.reshape(y_t.size, x_t.size, n_classes)
            # move class axis to front: (n_classes, ny, nx)
            box_counts[:, :, :, t] = np.moveaxis(year_counts, -1, 0)
        print("    adding to target_nlcd...")
        # convert counts to fractions
        # total pixels per coarse pixel per year:
        box_totals = box_counts.sum(axis=0, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            box_fracs = box_counts / box_totals
        # apply valid mask: set to nan where target pixel is invalid
        for k, lc_name in enumerate(classes):
            arr = box_fracs[k]
            arr[~valid_mask, :] = np.nan
            target_nlcd[lc_name].loc[
                dict(x=x_t, y=y_t)
            ] = arr
        # plot deciduous forest every 5 boxes for checking
        if (boxes_used % 5) == 0:
            plotting.plot_from_xarray(
                load_type='ds',
                type_obj=target_nlcd,
                var='evergreen_forest',
                proj_in='EPSG:5070',
                proj_out='EPSG:5070',
                fname=(
                    '/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/plots/deciduous_forest_box_{}.png'.format(b)
                )
            )
        boxes_used += 1
        #print(target_subset)
        #print(nlcd_subset)
        ## check if there are any valid pixels in this target subset
        #if np.all(np.isnan(target_subset['random_vals'].values)):
        #    print("  no valid pixels in this box, skipping...")
        #    continue
        ## go through each pixel of our target grid and get the overlapping nlcd pixels
        #for i in tqdm(range(target_subset.sizes['y']), desc=f'box {b+1} / {len(all_boxes)}'):
        #    #print(f"Processing row {i} / len {target_subset.sizes['y']} for box {b+1} / {len(all_boxes)}")
        #    for j in range(target_subset.sizes['x']):
        #        #target_grid_subset = target_grid.sel(
        #        #    x=slice(min_x_box, max_x_box),
        #        #    y=slice(min_y_box, max_y_box)
        #        #).copy()
        #        #for lc in unique_lc:
        #        #    target_grid_subset[lc] = (
        #        #        ('y', 'x','year'),
        #        #        np.full(
        #        #            (target_subset.sizes['y'],
        #        #             target_subset.sizes['x'],
        #        #             len(nlcd_years)),
        #        #            np.nan
        #        #        )
        #        #    )
        #        target_pixel = target_subset.isel(y=i, x=j)
        #        if np.isnan(target_pixel['random_vals'].values.item()):
        #            continue
        #        # get the bounds of the target pixel
        #        min_x = (
        #            target_pixel['x'].values.item() - (x_res / 2)
        #        )
        #        max_x = (
        #            target_pixel['x'].values.item() + (x_res / 2)
        #        )
        #        min_y = (
        #            target_pixel['y'].values.item() - (y_res / 2)
        #        )
        #        max_y = (
        #            target_pixel['y'].values.item() + (y_res / 2)
        #        )
        #        # transform to nlcd crs
        #        #min_x_4326, min_y_4326 = transformer.transform(min_x, min_y)
        #        #max_x_4326, max_y_4326 = transformer.transform(max_x, max_y)
        #        # select the overlapping nlcd pixels
        #        nlcd_pix_sub = nlcd_subset.sel(
        #            x=slice(min_x, max_x),
        #            y=slice(min_y, max_y)
        #        )
        #        for y, year in enumerate(nlcd_years):
        #            nlcd_year = nlcd_pix_sub.sel(time=year)
        #            #total_pixels = nlcd_year['nlcd'].size
        #            lc_counts = {lc:0 for lc in unique_lc}
        #            lc_counts['other'] = 0
        #            landcover_data = nlcd_year['nlcd'].values.flatten()
        #            vals, counts = np.unique(landcover_data, return_counts=True)
        #            total_pixels = counts.sum()
        #            for val, count in zip(vals, counts):
        #                lc_name = nlcd_dict.get(val, 'other')
        #                lc_counts[lc_name] += count
        #            #    lc_name = nlcd_dict.get(val, 'other')
        #            #    lc_counts[lc_name] += 1
        #            # now calculate fractions and assign to target_nlcd
        #            for lc in unique_lc:
        #                if total_pixels == 0:
        #                    fraction = np.nan
        #                else:
        #                    fraction = lc_counts[lc] / total_pixels
        #                target_subset[lc][i,j,y] = fraction
        #            if total_pixels == 0:
        #                other_fraction = np.nan
        #            else:
        #                other_fraction = lc_counts['other'] / total_pixels
        #            target_subset['other'][i,j,y] = other_fraction
        #    #if i == 5:
        #    #    break
        ## replace the values in target_nlcd with those from target_grid_subset
        #print(target_subset)
        #print(target_nlcd)
        #for lc in unique_lc + ['other']:
        #    target_nlcd[lc].loc[
        #        dict(
        #            x=target_subset['x'].values,
        #            y=target_subset['y'].values
        #        )
        #    ] = target_subset[lc].values
        ##break  # temporary break for testing
    # save this dataset to zarr
    output_path = os.path.join(
        data_dir,
        'nlcd',
        'nlcd_target_grid_2000_2024.zarr'
    )
    print(f'saving to {output_path}...')
    # chunk the dataset before saving
    target_nlcd = target_nlcd.chunk({'y':500, 'x':500, 'year':1})
    with ProgressBar():
        target_nlcd.to_zarr(output_path, mode='w')








if __name__ == "__main__":
    main()