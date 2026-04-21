Archive-forward scratch test setup for `2024`.

Files here:
- `sync_current_scientific_and_climatology_to_oak.sbatch`: one-off manual sync job for the current scientific zarr and Daymet climatology if OAK is missing those baselines.
- `prepare_archive_forward_setup.py`: stage job payload; requires the needed OAK baselines to already exist and then stages `OAK -> scratch` only.
  It stages a full scratch scientific zarr through `2024`, then intentionally corrupts only `2024` and sets `quality_flag=1` for that year so the yearly coordinator sees it as low-latency output that needs replacement.
  It stages the annual/model-resolution NLCD baseline through `2023`, the Daymet clim20 baseline through `2023`, and the Daymet climatology store.
- `stage_archive_forward_setup.sbatch`: stages baseline assets from OAK to scratch and writes test configs.
- `run_archive_forward_inference_test.sbatch`: runs the real yearly final coordinator against the scratch registry/config.
  The coordinator detects that `2024` is fully present but low-latency quality, confirms Daymet/NLCD availability, updates the scratch Daymet/NLCD stores for `2024`, runs final inference on `owners` GPUs only, and overwrites `2024` in the scratch scientific zarr.
  Default owners cap is `100`, the owners GPU memory constraint is cleared, and OAK sync-back is disabled for this test run.
- `run_archive_forward_test.sh`: submit-only wrapper that chains the stage job and the yearly coordinator job with dependencies.

Expected flow:
1. If a required baseline is missing on OAK, run `sync_current_scientific_and_climatology_to_oak.sbatch` manually first.
2. Run `bash /home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/tests/archive_forward/run_archive_forward_test.sh`.

Important:
- The test startup does not move anything from `scratch -> OAK`.
- The stage job is strict: it fails if the required OAK baselines are missing.
