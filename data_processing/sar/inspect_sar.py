import rioxarray as rxr
import rasterio
import numpy as np

def main():
    sar_file = (
        '/oak/stanford/groups/konings/datasets/Sentinel1_15days_250m/2016-04-15_sar.tif'
    )
    this_file = rxr.open_rasterio(sar_file)
    print(this_file)
    vals = this_file.sel(
        band=1
    ).values
    print(np.unique(vals))

if __name__ == "__main__":
    main()