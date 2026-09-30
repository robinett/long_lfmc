# Long LFMC Viewer 3857

Viewer-specific code for the long LFMC web map lives in this directory.

For the public dataset description, download instructions, and example notebook link, see the repository [README](/home/users/trobinet/long_lfmc/README.md).

For remote API deployment (for example Render), install the Python runtime dependencies from
[requirements.txt](/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/requirements.txt).
This file is intended for the Source-backed viewer API/runtime path, not the local tile/dataset build workflow.

The production frontend at `https://lfmc.stanford.edu` is built by
[deploy-viewer.yml](/home/users/trobinet/long_lfmc/.github/workflows/deploy-viewer.yml)
on pushes to `main` that change this viewer directory, then uploaded to GCS.
[frontend/.env.production](/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/frontend/.env.production)
points the app at the deployed API at `https://long-lfmc.onrender.com`.

Create the repository GitHub Actions secret `VITE_CARTO_BASEMAP_KEY` with the
CARTO Basemaps API key before deploying. Production builds require it. The key
is included in browser requests, so restrict it to `lfmc.stanford.edu` in the
CARTO dashboard and keep the existing OpenStreetMap and CARTO attribution visible.

Recommended frontend deployment settings:
- root directory: `lfmc_model/scripts/viewer_3857/frontend`
- build command: `npm install && npm run build`
- output directory: `dist`

The included
[frontend/vercel.json](/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/frontend/vercel.json)
provides a simple SPA rewrite for static hosting.
