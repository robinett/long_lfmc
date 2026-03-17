from pathlib import Path

import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
import os


SCRATCH_ROOT = Path("/scratch/users/trobinet/long_lfmc")
FINAL_ROOT = SCRATCH_ROOT / "final_lfmc"
CLIMATE_ROOT = FINAL_ROOT / "climate_zones"
TARGET_GRID_PATH = FINAL_ROOT / "grid" / "epsg5070_500m_westUS_grid.nc4"
DEFAULT_OUTPUT_PATH = CLIMATE_ROOT / "climate_zone_per_pixel_fullgrid.nc4"


def resolve_climate_zone_raster():
    candidates = [
        CLIMATE_ROOT / "raw" / "koppen_geiger_0p1.tif",
        CLIMATE_ROOT / "raw" / "1991_2020" / "koppen_geiger_0p1.tif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    tif_candidates = sorted(CLIMATE_ROOT.rglob("*.tif"))
    if len(tif_candidates) == 1:
        return tif_candidates[0]

    preferred = [
        path for path in tif_candidates
        if "koppen" in path.name.lower() or "geiger" in path.name.lower()
    ]
    if len(preferred) == 1:
        return preferred[0]

    raise FileNotFoundError(
        f"Could not find a unique climate raster under {CLIMATE_ROOT}. "
        "Run get_climate_zones.sh first and confirm the extracted TIFF path."
    )

def get_climate_zone_per_pixel(
    target_grid_path,
    climate_zone_path,
    output_path
):
    print(f"Loading target grid from {target_grid_path}")
    # load the target grid
    target_grid = xr.open_dataset(target_grid_path)
    print(f"Loading raw climate zones from {climate_zone_path}")
    # load the climate zone files
    climate_zones = rxr.open_rasterio(climate_zone_path)
    # get the climate zone per pixel
    climate_zones_resampled = climate_zones.rio.reproject_match(
        target_grid, resampling=Resampling.nearest
    )
    # convert to xarray dataset with var name 'climate_zone'
    climate_zones_resampled = climate_zones_resampled.to_dataset(name='climate_zone')
    # assign the correct coordinates
    climate_zones_resampled = climate_zones_resampled.assign_coords(
        {
            'x':target_grid['x'],
            'y':target_grid['y']
        }
    )
    print(climate_zones_resampled)
    # save the output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    climate_zones_resampled.to_netcdf(output_path)
    print(f"Saved climate zones per pixel to {output_path}")

def main():
    target_grid_path = str(TARGET_GRID_PATH)
    climate_zone_path = str(resolve_climate_zone_raster())
    output_path = str(DEFAULT_OUTPUT_PATH)
    get_climate_zone_per_pixel(
        target_grid_path,
        climate_zone_path,
        output_path
    )


if __name__ == "__main__":
    main()
