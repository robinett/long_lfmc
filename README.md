# Long LFMC Dataset

This repository contains code and data workflows for the long LFMC project.

The long LFMC dataset is a gridded map product for live fuel moisture content across the western United States. The public release is distributed as a scientific `.zarr` store on Source Cooperative in its native `EPSG:5070` grid.

The model estimates LFMC from a multisource set of remote-sensing and environmental inputs. In the current pipeline, those inputs include MODIS surface reflectance history, Daymet weather and anomaly history, and static land-surface features such as topography, soils, canopy height, and land-cover fractions. The main product is a time-resolved gridded LFMC prediction field, with ensemble mean and uncertainty variables included in the published map store.

The model family in this repository is a deep learning multisource temporal model. The training stack includes transformer-based and fusion-model variants that combine short-term satellite history, longer-term climate history, and static predictors. Training labels come from in situ LFMC observations from NFMD-based processed site records, and the multitask training code also uses SAR VV/VH supervision as auxiliary targets in some model variants.

You can explore the public dataset remotely without downloading the full store, or download the full `.zarr` store to use locally.

## Links

- Viewer: `https://example.com/long-lfmc-viewer`

  The viewer is the easiest way to explore the dataset if you want to look at patterns quickly before writing any code. It is an interactive web map for browsing LFMC across space and time, and it is designed to make a very large spatiotemporal product feel immediately usable. It is also the best option if you want to download data for only a small number of points rather than working with the full map dataset.

  In the viewer, you will be able to:

  - pan and zoom across the western United States
  - move through available dates to see how modeled LFMC changes over time
  - click on a location to inspect the local LFMC value and related map information
  - inspect time series at individual pixels
  - point and click up to 10 sites and download a `.csv` of daily LFMC values for those locations
  - view uncertainty and quality-related layers alongside the main LFMC field
  - use it as a fast visual screening tool before downloading data or writing analysis code

- Example notebook: [example_use_long_lfmc_remote.ipynb](/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/example_use_long_lfmc_remote.ipynb)

  The notebook is a hands-on example of using the public `.zarr` store directly from Python. It walks through opening the remote dataset, inspecting its dimensions and variables, pulling a small time slice, extracting a point time series, selecting an area with a polygon, and making simple plots.

  The main reason this is useful is that you can work with the public dataset without downloading the full store to your machine. `xarray` opens the remote `.zarr` lazily, so you usually only fetch metadata and the specific chunks needed for the subset or plot you request. In practice, that means you can do real analysis with very little local storage and very little up-front transfer cost.

## Download The Full Dataset

If you want the entire public long LFMC dataset on your local machine, the recommended method is to download the full `.zarr` store with `aws s3 sync`.

This is better than writing a custom Python downloader because:

- `.zarr` stores are directories made up of many files
- `aws s3 sync` handles recursive download cleanly
- interrupted downloads can be resumed by running the same command again
- the downloaded directory structure stays in the format that `xarray.open_zarr(...)` expects

### Public Dataset Location

- Source endpoint: `https://data.source.coop`
- Public S3 bucket: `us-west-2.opendata.source.coop`
- Dataset prefix: `rseg/long-lfmc-test/lfmc_maps.zarr`

### Recommended Command

Choose the output directory you want, then run:

```bash
aws s3 sync \
  s3://us-west-2.opendata.source.coop/rseg/long-lfmc-test/lfmc_maps.zarr \
  /desired/local/path/lfmc_maps.zarr \
  --no-sign-request \
  --region us-west-2
```

Example:

```bash
aws s3 sync \
  s3://us-west-2.opendata.source.coop/rseg/long-lfmc-test/lfmc_maps.zarr \
  ./lfmc_maps.zarr \
  --no-sign-request \
  --region us-west-2
```

### After Download

You can open the downloaded dataset locally with:

```python
import xarray as xr

ds = xr.open_zarr("./lfmc_maps.zarr")
ds
```

### Notes

- The full dataset can be large, so make sure you have enough disk space before downloading.
- If the transfer stops partway through, run the same `aws s3 sync` command again to continue.
- The notebook is a better option if you only need to inspect or subset the data.
