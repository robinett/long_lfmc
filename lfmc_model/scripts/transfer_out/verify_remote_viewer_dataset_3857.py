#!/usr/bin/env python3

import time
from pathlib import Path

import xarray as xr
import yaml


here = Path(__file__).resolve().parent
viewer_config_path = here.parent / "viewer_3857" / "viewer_config.yaml"
transfer_config_path = here / "source_coop_transfer_configs.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_viewer_config():
    with viewer_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_transfer_config():
    with transfer_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_viewer_config()
    transfer_cfg = load_transfer_config()
    viewer_dataset_cfg = transfer_cfg["datasets"]["viewer_3857_lfmc_maps"]
    data_cfg = cfg["data"]
    source_store = str(data_cfg["source_store"]).strip()
    source_endpoint_url = str(data_cfg["source_endpoint_url"]).strip()
    display_variable = str(data_cfg["display_variable"])
    uncertainty_variable = str(data_cfg["uncertainty_variable"])
    required_variables = sorted(
        set(
            [display_variable, uncertainty_variable]
            + [str(name) for name in viewer_dataset_cfg.get("required_variables", [])]
        )
    )
    if not source_store:
        raise ValueError("viewer_config data.source_store is empty")

    storage_options = {"anon": True}
    if source_endpoint_url:
        storage_options["client_kwargs"] = {"endpoint_url": source_endpoint_url}

    log(f"Opening remote viewer dataset {source_store}")
    started = time.time()
    ds = xr.open_zarr(
        source_store,
        consolidated=True,
        storage_options=storage_options,
        chunks={},
    )
    try:
        elapsed = time.time() - started
        log(
            "Opened remote viewer dataset in "
            f"{elapsed:.2f}s with dims {dict(ds.sizes)}"
        )
        for variable_name in required_variables:
            if variable_name not in ds.data_vars:
                raise ValueError(f"Remote viewer dataset is missing expected variable {variable_name!r}")
        x_size = int(ds.sizes["x"])
        y_size = int(ds.sizes["y"])
        center_x = x_size // 2
        center_y = y_size // 2
        mean_value = float(ds[display_variable].isel(time=0, y=center_y, x=center_x).values)
        uncertainty_value = float(ds[uncertainty_variable].isel(time=0, y=center_y, x=center_x).values)
        climatology_dim = next(
            dim_name
            for dim_name in ds["lfmc_climatology_mean_point"].dims
            if dim_name in {"climatology_day", "dayofyear"}
        )
        climatology_value = float(
            ds["lfmc_climatology_mean_point"]
            .isel({climatology_dim: 0, "y": center_y, "x": center_x})
            .values
        )
        anomaly_value = mean_value - climatology_value
        log(
            "Sample read succeeded at "
            f"time=0, y={center_y}, x={center_x}: "
            f"{display_variable}={mean_value:.3f}, "
            f"{uncertainty_variable}={uncertainty_value:.3f}, "
            f"lfmc_climatology_mean_point={climatology_value:.3f}, "
            f"lfmc_anomaly={anomaly_value:.3f}"
        )
    finally:
        ds.close()


if __name__ == "__main__":
    main()
