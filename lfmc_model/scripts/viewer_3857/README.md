# Long LFMC Viewer 3857

Viewer-specific code for the long LFMC web map lives in this directory.

For the public dataset description, download instructions, and example notebook link, see the repository [README](/home/users/trobinet/long_lfmc/README.md).

For remote API deployment (for example Render), install the Python runtime dependencies from
[requirements.txt](/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/requirements.txt).
This file is intended for the Source-backed viewer API/runtime path, not the local tile/dataset build workflow.

For static frontend deployment (for example Vercel), use
[frontend/.env.production](/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/frontend/.env.production)
to point the app at the deployed API. The current production target is
`https://long-lfmc.onrender.com`.

Recommended frontend deployment settings:
- root directory: `lfmc_model/scripts/viewer_3857/frontend`
- build command: `npm install && npm run build`
- output directory: `dist`

The included
[frontend/vercel.json](/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857/frontend/vercel.json)
provides a simple SPA rewrite for static hosting.
