#!/usr/bin/env python3

import time
from pathlib import Path

import xarray as xr
import yaml
import zarr


here = Path(__file__).resolve().parent
transfer_config_path = here / "source_coop_transfer_configs.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_transfer_config():
    with transfer_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_transfer_config()
    dataset_cfg = cfg["datasets"]["scientific_lfmc_maps"]
    scientific_dataset_path = Path(str(dataset_cfg["source_path"])).expanduser().resolve()
    if not scientific_dataset_path.exists():
        raise FileNotFoundError(
            f"Scientific dataset does not exist yet: {scientific_dataset_path}."
        )

    has_zarr_v3 = (scientific_dataset_path / "zarr.json").exists()
    has_zarr_v2 = (scientific_dataset_path / ".zgroup").exists()
    if has_zarr_v3:
        log(f"Detected Zarr v3 store at {scientific_dataset_path}")
    elif has_zarr_v2:
        log(f"Detected Zarr v2 store at {scientific_dataset_path}")
    else:
        raise ValueError(
            "Expected a Zarr store with either zarr.json (Zarr v3) "
            f"or .zgroup (Zarr v2) at the top level, but found neither in {scientific_dataset_path}"
        )

    log(f"Consolidating metadata for {scientific_dataset_path}")
    zarr.consolidate_metadata(str(scientific_dataset_path))

    log(f"Verifying consolidated open for {scientific_dataset_path}")
    ds = xr.open_zarr(str(scientific_dataset_path), consolidated=True)
    try:
        log(
            "Verified consolidated scientific dataset "
            f"with dims {dict(ds.sizes)} and variables {list(ds.data_vars)}"
        )
    finally:
        ds.close()

    log("Scientific dataset is ready for Source upload")


if __name__ == "__main__":
    main()
