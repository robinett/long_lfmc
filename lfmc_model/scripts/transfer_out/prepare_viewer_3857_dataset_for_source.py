#!/usr/bin/env python3

import time
from pathlib import Path

import xarray as xr
import yaml
import zarr


here = Path(__file__).resolve().parent
viewer_dataset_config_path = here.parent / "viewer_3857" / "viewer_dataset_config.yaml"


def timestamped_message(message: str) -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ") + message


def log(message: str) -> None:
    print(timestamped_message(message), flush=True)


def load_dataset_config():
    with viewer_dataset_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_dataset_config()
    viewer_dataset_path = Path(str(cfg["output"]["viewer_dataset_path"])).expanduser().resolve()
    if not viewer_dataset_path.exists():
        raise FileNotFoundError(
            f"Viewer dataset does not exist yet: {viewer_dataset_path}. "
            "Run viewer_3857/run_viewer_build_dataset.sh first."
        )

    has_zarr_v3 = (viewer_dataset_path / "zarr.json").exists()
    has_zarr_v2 = (viewer_dataset_path / ".zgroup").exists()
    if has_zarr_v3:
        log(f"Detected Zarr v3 store at {viewer_dataset_path}")
    elif has_zarr_v2:
        log(f"Detected Zarr v2 store at {viewer_dataset_path}")
    else:
        raise ValueError(
            "Expected a Zarr store with either zarr.json (Zarr v3) "
            f"or .zgroup (Zarr v2) at the top level, but found neither in {viewer_dataset_path}"
        )

    log(f"Consolidating metadata for {viewer_dataset_path}")
    zarr.consolidate_metadata(str(viewer_dataset_path))

    log(f"Verifying consolidated open for {viewer_dataset_path}")
    ds = xr.open_zarr(str(viewer_dataset_path), consolidated=True)
    try:
        expected_variables = [
            str(cfg["dataset"]["display_variable"]),
            str(cfg["dataset"]["uncertainty_variable"]),
            str(cfg["dataset"]["quality_variable"]),
            str(cfg["dataset"]["landcover_variable"]),
        ]
        missing_variables = [name for name in expected_variables if name not in ds.data_vars]
        if missing_variables:
            raise ValueError(
                "Viewer dataset is missing expected variables after rebuild: "
                f"{missing_variables}"
            )
        log(
            "Verified consolidated viewer dataset "
            f"with dims {dict(ds.sizes)} and variables {list(ds.data_vars)}"
        )
    finally:
        ds.close()

    log("Viewer dataset is ready for Source upload")


if __name__ == "__main__":
    main()
