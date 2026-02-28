import json
import os

from longweather_direct_pipeline import build_training_sample_index_from_label_sources


def main():
    # All configuration is explicitly set here.
    scratch_dir = "/scratch/users/trobinet/long_lfmc/final_lfmc"
    oak_dir = "/oak/stanford/groups/konings/trobinet/long_lfmc/trent_datasets"

    start_date = "2000-01-01"
    end_date = "2024-12-31"

    out_path = os.path.join(
        scratch_dir,
        "lfmc_model",
        "indexes",
        "sample_index_longweather_2000_2024_lfmc_vv_minus_vh.parquet",
    )

    label_sources = {
        "nfmd": os.path.join(
            scratch_dir,
            "nfmd",
            "nfmd_processed.csv",
        ),
        #"vh_at_sites": os.path.join(
        #   scratch_dir,
        #   'sar',
        #   'vh_samples_at_sites_matching.csv'
        #),
        #"vh_at_random": os.path.join(
        #   scratch_dir,
        #   'sar',
        #   'vh_samples_random_matching.csv'
        #),
        #"vv_at_sites": os.path.join(
        #   scratch_dir,
        #   'sar',
        #   'vv_samples_at_sites_matching.csv'
        #),
        #"vv_at_random": os.path.join(
        #   scratch_dir,
        #   'sar',
        #   'vv_samples_random_matching.csv'
        #),
        "vv_minus_vh_at_sites": os.path.join(
           scratch_dir,
           'sar',
           'vv_minus_vh_samples_at_sites_matching.csv'
        ),
        "vv_minus_vh_at_random": os.path.join(
           scratch_dir,
           'sar',
           'vv_minus_vh_samples_random_matching.csv'
        ),
        #"vv_over_vh_at_sites": os.path.join(
        #   scratch_dir,
        #   'sar',
        #   'vv_over_vh_samples_at_sites_matching.csv'
        #),
        #"vv_over_vh_at_random": os.path.join(
        #   scratch_dir,
        #   'sar',
        #   'vv_over_vh_samples_random_matching.csv'
        #)
    }
    target_cols = ["lfmc","vv_minus_vh"]
    acceptable_lfmc_range = (30.0, 500.0)
    vh_locations = "all"  # all | at_sites | at_random
    random_seed = 42

    target_sample_n = {
        "lfmc": -1,  # -1 means keep all
        "vv_minus_vh": -1,
    }

    target_sample_fraction = {
        # "vh_backscatter": 0.25,
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = build_training_sample_index_from_label_sources(
        label_sources=label_sources,
        start_date=start_date,
        end_date=end_date,
        out_path=out_path,
        target_cols=target_cols,
        acceptable_lfmc_range=acceptable_lfmc_range,
        target_sample_n=target_sample_n,
        target_sample_fraction=target_sample_fraction,
        vh_locations=vh_locations,
        random_seed=random_seed,
        sort_by=["date", "latitude", "longitude"],
    )

    print(f"Wrote {len(df):,} rows to {out_path}")
    print('df:')
    print(df)
    if "target_name" in df.columns:
        print(f"Target counts: {df['target_name'].value_counts(dropna=False).to_dict()}")
    if "source_legible" in df.columns:
        print(f"Source counts: {df['source_legible'].value_counts(dropna=False).to_dict()}")
    print(
        json.dumps(
            {
                "target_cols": target_cols,
                "acceptable_lfmc_range": list(acceptable_lfmc_range),
                "vh_locations": vh_locations,
                "random_seed": random_seed,
                "target_sample_n": target_sample_n,
                "target_sample_fraction": target_sample_fraction,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

