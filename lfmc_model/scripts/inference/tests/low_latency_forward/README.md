# Low-Latency Forward Test

This harness stages a scratch-only pre-2024 environment and then runs the real low-latency coordinator against it with a simulated `today` of `2025-01-07`.

Files:
- `prepare_low_latency_forward_setup.py`: stage job payload. It restores a clean scratch baseline through `2023`, stages a single scratch LFMC target zarr, and writes the scratch registry/config used by the test.
- `stage_low_latency_forward_setup.sbatch`: stages the scratch baseline and writes test configs.
- `run_low_latency_forward_inference_test.sbatch`: runs the real low-latency coordinator against the scratch registry/config, with `TODAY_OVERRIDE=2025-01-07`, `SKIP_OAK_SYNC=1`, and owners-only GPU submission.
- `run_low_latency_forward_test.sh`: submit-only wrapper that chains the stage job and the low-latency coordinator with dependencies.

Expected behavior:
- The scratch LFMC target starts with data only through `2023-12-31`.
- The coordinator resolves `safe_end = 2024-12-31`.
- It detects that `2024-01-01 -> 2024-12-31` is missing.
- It updates MODIS with 5-day tail context, downloads/regrids PRISM and SNODAS for `2024`, appends low-latency anomalies into the scratch combined weather store, runs inference, and appends `2024` into the scratch LFMC zarr.

Submit:

```bash
bash /home/users/trobinet/long_lfmc/lfmc_model/scripts/inference/tests/low_latency_forward/run_low_latency_forward_test.sh
```
