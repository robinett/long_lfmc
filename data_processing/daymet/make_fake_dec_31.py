import os
import shutil
from datetime import datetime

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def fill_missing_1231(base_dir, var_list, start_year=1980, end_year=2025):
    for var in var_list:
        print(f"\nProcessing variable: {var}")
        for year in range(start_year, end_year + 1):
            if not is_leap_year(year):
                continue

            if var == 'none':
                dec_dir = os.path.join(base_dir, str(year), "12")
                f_1230 = os.path.join(dec_dir, f"{year}_12_30_regridded.nc")
                f_1231 = os.path.join(dec_dir, f"{year}_12_31_regridded.nc")
            else:
                dec_dir = os.path.join(base_dir, var, str(year), "12")
                f_1230 = os.path.join(dec_dir, f"{var}_{year}_12_30_regridded.nc")
                f_1231 = os.path.join(dec_dir, f"{var}_{year}_12_31_regridded.nc")
            if not os.path.isdir(dec_dir):
                print(f"  Directory not found: {dec_dir}")
                continue


            if not os.path.exists(f_1230):
                print(f"  Missing 12/30 file: {f_1230}")
                continue

            if os.path.exists(f_1231):
                print(f"  12/31 already exists for {year}")
                continue

            shutil.copyfile(f_1230, f_1231)
            print(f"  Copied 12/30 to 12/31 for {year}")

# Example usage:
fill_missing_1231(
    base_dir="/scratch/users/trobinet/long_lfmc/trent_datasets/daymet/",
    var_list=['stats'],
    start_year=2003,
    end_year=2023
)

