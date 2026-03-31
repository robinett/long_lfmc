# Long LFMC Viewer 3857

This directory contains a separate Web Mercator viewer fork for the long LFMC map product. It is intentionally isolated from `lfmc_model/scripts/viewer` so the native-grid viewer can remain untouched.

The prototype has three pieces:

- `build_viewer_dataset_3857.py`: derives a viewer-only `EPSG:3857` Zarr from the scientific `EPSG:5070` LFMC Zarr
- `build_viewer_assets.py`: builds local tile pyramids and a manifest from that derived viewer Zarr
- `api/serve_viewer_api.py`: serves point queries plus the local viewer assets
- `frontend/`: a Vite + React app using OpenLayers in `EPSG:3857` with a standard basemap underlay

## What It Does

- Opens the local scientific LFMC Zarr from `viewer_dataset_config.yaml`
- Builds a separate viewer-only Web Mercator Zarr
- Builds a two-date local tiled artifact set for:
  - LFMC mean
- Serves those local `EPSG:3857` tiles locally for the frontend
- Lets you change dates, toggle map layers, and click for:
  - LFMC
  - viewer-grid cell location
  - land cover
  - product level
  - full time series

## Build First

Build the derived 3857 viewer Zarr first:

```bash
cd /home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857
bash run_viewer_build_dataset.sh
```

Then build the local viewer assets:

```bash
cd /home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857
bash run_viewer_build.sh
```

The dataset build writes the derived viewer Zarr to:

```text
/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/viewer_3857/lfmc_maps_3857.zarr
```

The asset build writes tiles plus `manifest.json` to:

```text
/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/viewer_assets/web_mercator_3857
```

## Stage 1 Source Publish

Stage 1 publishes only the derived `EPSG:3857` viewer Zarr to Source. Tiles remain local in this stage.

Run:

```bash
cd /home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out
bash run_upload_viewer_3857_dataset.sh
```

That transfer-side helper does three things:

- consolidates metadata on the local viewer Zarr
- uploads the viewer Zarr to the existing Source product prefix using the shared transfer config
- opens the remote store back from Source and performs a small sample read

The remote store path is configured in [viewer_config.yaml](viewer_config.yaml) as:

```text
s3://rseg/long-lfmc-test/viewer_3857/lfmc_maps_3857.zarr
```

The live `viewer_3857` runtime still stays on `data_source: local` until the remote store is verified and latency is acceptable.

All upload and Source-transfer logic now lives in:

```text
/home/users/trobinet/long_lfmc/lfmc_model/scripts/transfer_out
```

## Startup

Run the API directly:

```bash
cd /home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857
bash run_viewer_api.sh
```

Run the frontend directly:

```bash
cd /home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857
bash run_viewer_frontend.sh
```

Or start both together in a tmux session:

```bash
cd /home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857
bash start_viewer_tmux.sh
```

## SSH Tunnel

If the viewer is running on a Sherlock login node, tunnel the frontend port from your laptop:

```bash
ssh -J your_sunetid@sherlock.stanford.edu -L 4174:127.0.0.1:4174 your_sunetid@your_login_node
```

Then open:

```text
http://127.0.0.1:4174
```

The Vite config proxies both `/api/*` and `/viewer-assets/*` to the Python API on port `8001`, so one tunnel is enough.

## Configuration

Edit `viewer_dataset_config.yaml` to change:

- scientific LFMC Zarr path
- derived 3857 viewer Zarr path
- target nominal resolution
- viewer sampling method

Edit `viewer_build_config.yaml` to change:

- viewer Zarr path
- selected dates
- output root
- layer ranges and palettes

Edit `viewer_config.yaml` to change:

- runtime viewer dataset source
- local asset root
- API host and port

## Notes

- This viewer is intentionally a derived display product, not the scientific source of truth.
- The scientific LFMC dataset remains in native `EPSG:5070`; this fork builds a separate `EPSG:3857` viewer dataset for display and click consistency with standard basemaps.
- Because the viewer dataset is in Web Mercator, its nominal `500 m` pixel size is not an exact on-the-ground 500 m everywhere.
