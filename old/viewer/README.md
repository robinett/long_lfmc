# Long LFMC Viewer Prototype

This directory now contains a local-first exact-grid viewer prototype for the long LFMC map product.

The prototype has three pieces:

- `build_viewer_assets.py`: builds native-grid tile pyramids and a manifest from the local LFMC Zarr
- `api/serve_viewer_api.py`: serves exact point queries plus the local viewer assets
- `frontend/`: a Vite + React app using OpenLayers in native `EPSG:5070`

## What It Does

- Opens the local Sherlock LFMC Zarr from `viewer_build_config.yaml`
- Builds a two-date local tiled artifact set for:
  - LFMC mean
  - uncertainty
- Serves those native `EPSG:5070` tiles locally for the frontend
- Lets you change dates, toggle map layers, and click for:
  - LFMC
  - uncertainty
  - exact click location
  - selected 500 m cell
  - land cover
  - product level
  - full time series with uncertainty band

## Build First

Build the local viewer assets before launching the app:

```bash
cd /home/users/trobinet/long_lfmc/old/viewer
bash run_viewer_build.sh
```

The build reads the local Zarr and writes native-grid tiles plus `manifest.json` to:

```text
/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/viewer_assets/local_two_date_native_grid
```

## Startup

Run the API directly:

```bash
cd /home/users/trobinet/long_lfmc/old/viewer
bash run_viewer_api.sh
```

Run the frontend directly:

```bash
cd /home/users/trobinet/long_lfmc/old/viewer
bash run_viewer_frontend.sh
```

Or start both together in a tmux session:

```bash
cd /home/users/trobinet/long_lfmc/old/viewer
bash start_viewer_tmux.sh
```

## SSH Tunnel

If the viewer is running on a Sherlock login node, tunnel the frontend port from your laptop:

```bash
ssh -J your_sunetid@sherlock.stanford.edu -L 4173:127.0.0.1:4173 your_sunetid@your_login_node
```

Then open:

```text
http://127.0.0.1:4173
```

The Vite config proxies both `/api/*` and `/viewer-assets/*` to the Python API on port `8000`, so one tunnel is enough.

## Configuration

Edit `viewer_build_config.yaml` to change:

- local Zarr path
- selected dates
- output root
- layer ranges and palettes

Edit `viewer_config.yaml` to change:

- runtime dataset source
- local asset root
- API host and port

## Notes

- This viewer is intentionally local-first so we can validate exact 500m cell behavior before moving anything to Source.
- The frontend now renders native `EPSG:5070` static tiles rather than reprojected Web Mercator tiles or full-scene images.
- Once the local exact-grid workflow is solid, the static tile assets can be moved to Source while the click API remains separate.
