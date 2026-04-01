#!/usr/bin/env python3

import time
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


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
    source_cfg = cfg["source_coop"]
    dataset_cfg = cfg["datasets"]["scientific_lfmc_maps"]
    bucket = str(source_cfg["bucket"]).strip().strip("/")
    product_prefix = str(source_cfg["product_prefix"]).strip().strip("/")
    destination_relpath = str(dataset_cfg["destination_relpath"]).strip().strip("/")
    source_endpoint_url = "https://data.source.coop"
    source_store = f"s3://{product_prefix}/{destination_relpath}"

    log(f"Opening remote scientific dataset {source_store}")
    ds = xr.open_zarr(
        source_store,
        consolidated=True,
        storage_options={
            "anon": True,
            "client_kwargs": {"endpoint_url": source_endpoint_url},
        },
    )
    try:
        sample_value = np.asarray(ds["lfmc_ens_mean"].isel(time=0, y=0, x=0).values).item()
        log(
            "Verified remote scientific dataset "
            f"from bucket={bucket} dims={dict(ds.sizes)} sample_lfmc_ens_mean={sample_value}"
        )
    finally:
        ds.close()


if __name__ == "__main__":
    main()
