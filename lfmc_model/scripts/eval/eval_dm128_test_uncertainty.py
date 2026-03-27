import argparse
import json
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

here = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(here, "../../.."))
sys.path.append(os.path.join(project_root, "lfmc_model", "utils"))
sys.path.append(os.path.join(project_root, "lfmc_model", "scripts", "eval"))

from plotting import generic_hexbin
from compare_timeseries import (
    _to_naive_datetime,
    aggregate_site_errors,
    get_model_inference_series,
    get_site_error,
    get_site_state_annotation,
    model_color,
)
from eval_deep import (
    build_lfmc_y2y_df,
    build_site_landcover_lookup,
    load_fold_predictions,
    select_ensemble_member_dirs,
)


DEFAULT_ENSEMBLE_ROOT = (
    "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/"
    "lfmc_vh_vv_365_shared_ensemble"
)
DEFAULT_MEMBER_PREFIX = "transformer_dm128_"
DEFAULT_INPUTS_ROOT = "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/inputs"
DEFAULT_INPUT_DATA_NAME = "ensemble/lfmc_vh_vv_365_shared_ensemble"
DEFAULT_ENSEMBLE_SUBSET_SIZE = 16
DEFAULT_ENSEMBLE_SUBSET_SEED = 0
LANDCOVER_CLASS_ORDER = [
    "deciduous_forest",
    "evergreen_forest",
    "shrub",
    "grass",
    "mixed_forest",
]
LANDCOVER_DISPLAY = {
    "overall": "Overall",
    "shrub": "Shrub",
    "evergreen_forest": "Evergreen Forest",
    "deciduous_forest": "Deciduous Forest",
    "grass": "Grass",
    "mixed_forest": "Mixed Forest",
    "unknown": "Unknown",
}
SITE_R2_KDE_MIN = -3.0
SITE_R2_KDE_MAX = 1.0
MIN_SITE_OBS = 10
NUM_EVERGREEN_TIMESERIES = 10
TIMESERIES_YEARS = 3
TIMESERIES_FORWARD_BATCH_SIZE = 4096
TIMESERIES_SELECTION_MODE = "random_sample"
TIMESERIES_SELECTION_SEED = 0


def paper_rc_params(fontsize):
    return {
        "font.family": "DejaVu Serif",
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "axes.linewidth": 1.0,
        "xtick.labelsize": max(fontsize - 1, 8),
        "ytick.labelsize": max(fontsize - 1, 8),
        "legend.fontsize": max(fontsize - 2, 8),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def format_landcover_label(label):
    return LANDCOVER_DISPLAY.get(str(label), str(label).replace("_", " ").title())


def coord_site_key_from_lat_lon(latitude, longitude):
    return f"{float(latitude):.5f}_{float(longitude):.5f}"


def build_timeseries_site_lookup(site_keys, round_decimals=6):
    records = []
    for site_key in sorted(site_keys):
        try:
            lat_str, lon_str = str(site_key).split("_", 1)
            lat = float(lat_str)
            lon = float(lon_str)
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "timeseries_site_key": str(site_key),
                "latitude_round": round(lat, round_decimals),
                "longitude_round": round(lon, round_decimals),
            }
        )
    if len(records) == 0:
        return pd.DataFrame(
            columns=["timeseries_site_key", "latitude_round", "longitude_round"]
        )
    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates(subset=["timeseries_site_key"], keep="first")
        .reset_index(drop=True)
    )


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the specific dm128 multitask test-fold evaluation requested for "
            "land-cover R2 distributions and uncertainty-error hexbins."
        )
    )
    parser.add_argument(
        "--ensemble_outputs_root",
        type=str,
        default=DEFAULT_ENSEMBLE_ROOT,
        help="Shared ensemble root containing transformer_dm128_* member dirs.",
    )
    parser.add_argument(
        "--ensemble_member_name_prefix",
        type=str,
        default=DEFAULT_MEMBER_PREFIX,
        help="Prefix used to select dm128 ensemble members.",
    )
    parser.add_argument(
        "--ensemble_subset_size",
        type=int,
        default=DEFAULT_ENSEMBLE_SUBSET_SIZE,
        help="Number of ensemble members to keep via deterministic random subset.",
    )
    parser.add_argument(
        "--ensemble_subset_seed",
        type=int,
        default=DEFAULT_ENSEMBLE_SUBSET_SEED,
        help="Seed for deterministic ensemble subsetting.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to write figures and tables.",
    )
    parser.add_argument(
        "--hexbin_gridsize",
        type=int,
        default=70,
        help="Hexbin grid size for uncertainty plots.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=16,
        help="Base font size for all plots.",
    )
    parser.add_argument(
        "--inputs_root",
        type=str,
        default=DEFAULT_INPUTS_ROOT,
        help="Root directory containing LFMC model inference inputs.",
    )
    parser.add_argument(
        "--input_data_name",
        type=str,
        default=DEFAULT_INPUT_DATA_NAME,
        help="Input data name used by compare_timeseries inference utilities.",
    )
    parser.add_argument(
        "--num_evergreen_sites",
        type=int,
        default=NUM_EVERGREEN_TIMESERIES,
        help="Number of evergreen sites to include in the example timeseries figure.",
    )
    parser.add_argument(
        "--timeseries_years",
        type=int,
        default=TIMESERIES_YEARS,
        help="Number of consecutive years to show in each evergreen timeseries panel.",
    )
    parser.add_argument(
        "--timeseries_forward_batch_size",
        type=int,
        default=TIMESERIES_FORWARD_BATCH_SIZE,
        help="Batch size used by point-tool inference for evergreen timeseries panels.",
    )
    return parser.parse_args()


def resolve_plot_dir(plot_dir):
    if plot_dir is not None:
        return plot_dir
    scratch_root = os.environ.get("SCRATCH", "/scratch/users/trobinet")
    return os.path.join(
        scratch_root,
        "long_lfmc",
        "final_lfmc",
        "lfmc_model",
        "plots",
        "eval_dm128_test_uncertainty",
    )


def ordered_landcovers(categories):
    categories = list(categories)
    present_classes = [cls for cls in LANDCOVER_CLASS_ORDER if cls in categories]
    remaining_classes = sorted([cls for cls in categories if cls not in present_classes])
    return present_classes + remaining_classes


def selected_member_dirs(ensemble_outputs_root, member_name_prefix, subset_size, subset_seed):
    member_dirs = select_ensemble_member_dirs(
        ensemble_outputs_root,
        member_name_prefix=member_name_prefix,
    )
    if subset_size in {None, "", "None"}:
        return member_dirs
    subset_size = int(subset_size)
    if subset_size <= 0:
        raise ValueError("ensemble_subset_size must be >= 1")
    if subset_size > len(member_dirs):
        raise ValueError(
            f"ensemble_subset_size={subset_size} exceeds available member count "
            f"({len(member_dirs)}) under {ensemble_outputs_root}"
        )
    rng = np.random.default_rng(int(subset_seed))
    selected_idx = np.sort(rng.choice(len(member_dirs), size=subset_size, replace=False))
    return [member_dirs[int(idx)] for idx in selected_idx]


def ensemble_selection_note(ensemble_outputs_root, member_name_prefix, subset_size, subset_seed):
    return (
        f"random subset of {int(subset_size)} ensemble members under {ensemble_outputs_root} "
        f"with prefix {member_name_prefix}, seed={int(subset_seed)}"
    )


def ensemble_member_names(member_dirs):
    return "|".join(os.path.basename(str(path)) for path in member_dirs)


def build_eval_row_keys(frame):
    key_cols = [
        col for col in [
            "target", "fold", "date", "latitude", "longitude",
            "source", "source_legible", "site_name", "fuel_type", "fuel",
        ] if col in frame.columns
    ]
    if len(key_cols) == 0:
        return pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index)
    work = frame[key_cols].copy()
    for col in work.columns:
        if np.issubdtype(np.asarray(work[col]).dtype, np.datetime64):
            work[col] = pd.to_datetime(work[col], errors="coerce").astype(str)
        else:
            work[col] = work[col].fillna("__nan__").astype(str)
    return work.agg("|".join, axis=1)


def aggregate_lfmc_member_eval_frames(member_eval_dfs):
    lfmc_frames = []
    for frame_idx, frame in enumerate(member_eval_dfs):
        lfmc_frame = frame[frame["target"] == "lfmc"].copy().reset_index(drop=True)
        if len(lfmc_frame) == 0:
            raise ValueError(f"Member {frame_idx + 1} has no LFMC evaluation rows")
        lfmc_frame["_row_key"] = build_eval_row_keys(lfmc_frame)
        lfmc_frames.append(lfmc_frame)
    common_keys = set(lfmc_frames[0]["_row_key"].tolist())
    for frame_idx, frame in enumerate(lfmc_frames[1:], start=2):
        common_keys &= set(frame["_row_key"].tolist())
        if len(common_keys) == 0:
            raise ValueError(
                f"No common LFMC evaluation rows remain after intersecting member {frame_idx}"
            )
    common_keys = sorted(common_keys)
    template = (
        lfmc_frames[0][lfmc_frames[0]["_row_key"].isin(common_keys)]
        .copy()
        .sort_values("_row_key", kind="mergesort")
        .reset_index(drop=True)
    )
    pred_stack = []
    obs_template = template["obs"].to_numpy(dtype=float)
    for frame_idx, frame in enumerate(lfmc_frames, start=1):
        work = (
            frame[frame["_row_key"].isin(common_keys)]
            .copy()
            .sort_values("_row_key", kind="mergesort")
            .reset_index(drop=True)
        )
        if work["_row_key"].tolist() != template["_row_key"].tolist():
            raise ValueError(f"LFMC row-key alignment mismatch in member {frame_idx}")
        work_obs = work["obs"].to_numpy(dtype=float)
        if not np.allclose(work_obs, obs_template, rtol=0.0, atol=1e-4, equal_nan=True):
            raise ValueError(f"LFMC truth mismatch in member {frame_idx} after row-key alignment")
        pred_stack.append(work["pred"].to_numpy(dtype=float))
    pred_stack = np.stack(pred_stack, axis=1)
    out = template.drop(columns=["_row_key"]).copy()
    out["pred"] = pred_stack.mean(axis=1)
    out["pred_std_ensemble"] = pred_stack.std(axis=1, ddof=0)
    return out


def load_subset_eval_context(args):
    member_dirs = selected_member_dirs(
        ensemble_outputs_root=args.ensemble_outputs_root,
        member_name_prefix=args.ensemble_member_name_prefix,
        subset_size=args.ensemble_subset_size,
        subset_seed=args.ensemble_subset_seed,
    )
    print(
        f"Using 16-member subset from {args.ensemble_outputs_root}: "
        f"{ensemble_member_names(member_dirs)}"
    )
    member_eval_dfs = [load_fold_predictions(member_dir) for member_dir in member_dirs]
    return {
        "mode": "ensemble",
        "model_dir": args.ensemble_outputs_root,
        "member_dirs": member_dirs,
        "eval_df": aggregate_lfmc_member_eval_frames(member_eval_dfs),
        "member_eval_dfs": member_eval_dfs,
        "ensemble_member_name_prefix": args.ensemble_member_name_prefix,
        "ensemble_selection_note": ensemble_selection_note(
            args.ensemble_outputs_root,
            args.ensemble_member_name_prefix,
            args.ensemble_subset_size,
            args.ensemble_subset_seed,
        ),
        "ensemble_member_names": ensemble_member_names(member_dirs),
    }


def top_consecutive_observation_years(dates, n_years):
    dt = pd.to_datetime(dates, errors="coerce")
    dt = dt[dt.notna()]
    if len(dt) == 0:
        return []
    n_years = int(max(n_years, 1))
    year_counts = pd.Series(dt.year).value_counts().to_dict()
    min_year = int(dt.year.min())
    max_year = int(dt.year.max())
    span_years = max_year - min_year + 1
    if span_years < n_years:
        return list(range(min_year, min_year + n_years))
    best_start = min_year
    best_score = -1
    for start_year in range(min_year, max_year - n_years + 2):
        years_here = list(range(start_year, start_year + n_years))
        score = int(sum(year_counts.get(year, 0) for year in years_here))
        if score > best_score:
            best_score = score
            best_start = start_year
    return list(range(best_start, best_start + n_years))


def panel_limits_from_series(series_list, pad_fraction=0.08):
    vals = []
    for series in series_list:
        for key in ["values", "lower", "upper"]:
            arr = np.asarray(series.get(key, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                vals.append(arr)
    if len(vals) == 0:
        return None
    data = np.concatenate(vals)
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    span = data_max - data_min
    pad = max(span * pad_fraction, 2.0 if span < 20 else 0.0)
    if span <= 0:
        pad = max(abs(data_min) * 0.1, 1.0)
    return data_min - pad, data_max + pad


def apply_panel_series(ax, series):
    dates = pd.to_datetime(series["dates"], errors="coerce")
    values = np.asarray(series["values"], dtype=float)
    lower = series.get("lower")
    upper = series.get("upper")
    color = series.get("color")
    if lower is not None and upper is not None:
        ax.fill_between(
            dates,
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            color=color,
            alpha=series.get("fill_alpha", 0.14),
            linewidth=0,
            zorder=series.get("zorder", 2) - 1,
        )
    ax.plot(
        dates,
        values,
        color=color,
        linestyle=series.get("linestyle", "-"),
        linewidth=series.get("linewidth", 2.0),
        marker=series.get("marker", None),
        markersize=series.get("markersize", 5),
        alpha=series.get("alpha", 1.0),
        zorder=series.get("zorder", 3),
        markerfacecolor=series.get("markerfacecolor", color),
        markeredgecolor=series.get("markeredgecolor", color),
        markeredgewidth=series.get("markeredgewidth", 0.8 if series.get("marker") else 0.0),
    )


def prepare_lfmc_test_df(eval_df):
    print("Preparing LFMC-only test-fold evaluation table")
    lfmc_df = build_lfmc_y2y_df(eval_df)
    if len(lfmc_df) == 0:
        raise ValueError("No LFMC test rows found in evaluation context")
    if "pred_std_ensemble" not in lfmc_df.columns:
        raise KeyError(
            "Expected ensemble uncertainty column pred_std_ensemble in LFMC evaluation table"
        )
    lfmc_df = lfmc_df.copy()
    lfmc_df["pred_std_ensemble"] = pd.to_numeric(
        lfmc_df["pred_std_ensemble"],
        errors="coerce",
    )
    lfmc_df["error"] = lfmc_df["pred"] - lfmc_df["obs"]
    lfmc_df["abs_error"] = np.abs(lfmc_df["error"])
    pred_abs = np.abs(pd.to_numeric(lfmc_df["pred"], errors="coerce"))
    valid_pred_mask = np.isfinite(pred_abs) & (pred_abs > 0)
    lfmc_df["uncertainty_norm_by_pred"] = np.nan
    lfmc_df.loc[valid_pred_mask, "uncertainty_norm_by_pred"] = (
        lfmc_df.loc[valid_pred_mask, "pred_std_ensemble"] / pred_abs.loc[valid_pred_mask]
    )
    lfmc_df["abs_error_norm_by_pred"] = np.nan
    lfmc_df.loc[valid_pred_mask, "abs_error_norm_by_pred"] = (
        lfmc_df.loc[valid_pred_mask, "abs_error"] / pred_abs.loc[valid_pred_mask]
    )
    finite_norm = lfmc_df["uncertainty_norm_by_pred"].to_numpy(dtype=float)
    finite_norm = finite_norm[np.isfinite(finite_norm)]
    lfmc_df["uncertainty_norm_by_pred_scaled"] = np.nan
    if finite_norm.size > 0:
        min_norm = float(np.min(finite_norm))
        max_norm = float(np.max(finite_norm))
        if max_norm > min_norm:
            lfmc_df["uncertainty_norm_by_pred_scaled"] = (
                (lfmc_df["uncertainty_norm_by_pred"] - min_norm) / (max_norm - min_norm)
            )
        else:
            lfmc_df["uncertainty_norm_by_pred_scaled"] = 0.0
    return lfmc_df


def build_site_r2_df(lfmc_df, plot_dir):
    print("Computing site-level R2 values and attaching dominant land cover")
    site_df = (
        lfmc_df.groupby("site_key", dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "latitude": float(pd.to_numeric(group["latitude"], errors="coerce").iloc[0]),
                    "longitude": float(pd.to_numeric(group["longitude"], errors="coerce").iloc[0]),
                    "year": int(pd.to_numeric(group["year"], errors="coerce").dropna().min()),
                    "n_points": int(len(group)),
                    "obs": np.asarray(group["obs"], dtype=float),
                    "pred": np.asarray(group["pred"], dtype=float),
                }
            )
        )
        .reset_index()
    )
    site_df = site_df[site_df["n_points"] >= MIN_SITE_OBS].copy()
    if len(site_df) == 0:
        raise ValueError(
            f"No LFMC test sites have at least {MIN_SITE_OBS} observations for site-level R2"
        )
    site_df["site_r2"] = site_df.apply(
        lambda row: float(r2_score(row["obs"], row["pred"])),
        axis=1,
    )
    lookup_path = os.path.join(plot_dir, "lfmc_test_site_landcover_lookup.csv")
    site_lookup_df = build_site_landcover_lookup(
        site_df[["site_key", "latitude", "longitude", "year"]].copy(),
        lookup_path,
    )
    site_lookup_df["site_key"] = site_lookup_df["site_key"].astype(str)
    site_df["site_key"] = site_df["site_key"].astype(str)
    site_df = site_df.merge(
        site_lookup_df[["site_key", "dominant_landcover", "dominant_landcover_frac"]],
        on="site_key",
        how="left",
    )
    site_df = site_df[site_df["dominant_landcover"].notna()].copy()
    site_df = site_df.drop(columns=["obs", "pred"])
    site_df["coord_site_key"] = site_df.apply(
        lambda row: coord_site_key_from_lat_lon(row["latitude"], row["longitude"]),
        axis=1,
    )
    site_df = site_df.sort_values(
        ["dominant_landcover", "site_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    return site_df


def plot_site_r2_landcover_pdf(site_r2_df, save_base, fontsize):
    print("Plotting site-level R2 density curves by land cover")
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("colorblind", n_colors=max(site_r2_df["dominant_landcover"].nunique(), 1))
    color_lookup = {
        landcover: palette[idx]
        for idx, landcover in enumerate(
            ordered_landcovers(site_r2_df["dominant_landcover"].dropna().unique().tolist())
        )
    }
    legend_labels = []
    for landcover in ordered_landcovers(site_r2_df["dominant_landcover"].dropna().unique().tolist()):
        class_df = site_r2_df[site_r2_df["dominant_landcover"] == landcover].copy()
        if len(class_df) < 2:
            print(
                f"Skipping KDE for land cover {landcover}: only {len(class_df)} site(s) with valid R2"
            )
            continue
        class_df["site_r2_clipped"] = class_df["site_r2"].clip(SITE_R2_KDE_MIN, SITE_R2_KDE_MAX)
        sns.kdeplot(
            data=class_df,
            x="site_r2_clipped",
            ax=ax,
            linewidth=2.2,
            label=landcover,
            color=color_lookup[landcover],
            fill=False,
            common_norm=False,
            bw_adjust=0.3,
            cut=0,
            clip=(SITE_R2_KDE_MIN, SITE_R2_KDE_MAX),
            gridsize=512,
        )
        legend_labels.append(f"{format_landcover_label(landcover)} (n={len(class_df)})")
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(
            handles,
            legend_labels,
            frameon=False,
            fontsize=max(fontsize - 2, 8),
            title=None,
        )
    ax.set_xlim(SITE_R2_KDE_MIN, SITE_R2_KDE_MAX)
    ax.set_xlabel("Site-level LFMC R2", fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=max(fontsize - 2, 8))
    ax.text(
        0.97,
        0.97,
        (
            f"Sites = {len(site_r2_df)}\n"
            f"Land covers = {site_r2_df['dominant_landcover'].nunique()}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "0.4",
        },
        fontsize=max(fontsize - 3, 8),
    )
    fig.tight_layout()
    png_path = f"{save_base}.png"
    pdf_path = f"{save_base}.pdf"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png_path": png_path, "pdf_path": pdf_path}


def build_lfmc_transition_df(lfmc_df, site_r2_df):
    print("Building adjacent-observation LFMC transition table")
    site_meta = site_r2_df[
        ["site_key", "dominant_landcover", "n_points", "coord_site_key"]
    ].drop_duplicates().copy()
    site_meta["site_key"] = site_meta["site_key"].astype(str)
    work_df = lfmc_df.copy()
    work_df["site_key"] = work_df["site_key"].astype(str)
    work_df = work_df.merge(site_meta, on="site_key", how="inner")
    work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
    work_df["obs"] = pd.to_numeric(work_df["obs"], errors="coerce")
    work_df = work_df.dropna(subset=["date", "obs"]).copy()
    work_df = work_df.sort_values(["site_key", "date"], kind="mergesort").reset_index(drop=True)
    work_df["next_date"] = work_df.groupby("site_key")["date"].shift(-1)
    work_df["next_obs"] = work_df.groupby("site_key")["obs"].shift(-1)
    work_df = work_df[work_df["next_date"].notna()].copy()
    work_df["delta_days"] = (
        (pd.to_datetime(work_df["next_date"]) - pd.to_datetime(work_df["date"]))
        .dt.total_seconds() / 86400.0
    )
    work_df["delta_lfmc"] = work_df["next_obs"] - work_df["obs"]
    work_df["slope"] = np.nan
    slope_mask = np.isfinite(work_df["delta_days"]) & (work_df["delta_days"] > 0)
    work_df.loc[slope_mask, "slope"] = (
        work_df.loc[slope_mask, "delta_lfmc"] / work_df.loc[slope_mask, "delta_days"]
    )
    work_df["next_slope"] = work_df.groupby("site_key")["slope"].shift(-1)
    work_df["turning_angle_deg"] = np.nan
    turning_mask = np.isfinite(work_df["slope"]) & np.isfinite(work_df["next_slope"])
    work_df.loc[turning_mask, "turning_angle_deg"] = np.degrees(
        np.abs(
            np.arctan(work_df.loc[turning_mask, "next_slope"])
            - np.arctan(work_df.loc[turning_mask, "slope"])
        )
    )
    work_df["relative_abs_change"] = np.nan
    rel_mask = np.isfinite(work_df["obs"]) & (work_df["obs"] > 0)
    work_df.loc[rel_mask, "relative_abs_change"] = (
        np.abs(work_df.loc[rel_mask, "delta_lfmc"]) / work_df.loc[rel_mask, "obs"]
    )
    keep_cols = [
        "site_key",
        "coord_site_key",
        "dominant_landcover",
        "n_points",
        "date",
        "next_date",
        "obs",
        "next_obs",
        "delta_days",
        "delta_lfmc",
        "slope",
        "next_slope",
        "turning_angle_deg",
        "relative_abs_change",
    ]
    return work_df[keep_cols].copy()


def summarize_turning_angle_by_landcover(transition_df):
    print("Summarizing turning angles by land cover")
    records = []
    for landcover in ordered_landcovers(transition_df["dominant_landcover"].dropna().unique().tolist()):
        class_df = transition_df[transition_df["dominant_landcover"] == landcover].copy()
        turning_vals = pd.to_numeric(
            class_df["turning_angle_deg"],
            errors="coerce",
        ).to_numpy(dtype=float)
        turning_vals = turning_vals[np.isfinite(turning_vals)]
        records.append(
            {
                "dominant_landcover": landcover,
                "n_sites": int(class_df["site_key"].astype(str).nunique()),
                "n_transitions_total": int(len(class_df)),
                "n_turning_angles": int(turning_vals.size),
                "turning_angle_mean_deg": (
                    float(np.mean(turning_vals)) if turning_vals.size > 0 else np.nan
                ),
                "turning_angle_median_deg": (
                    float(np.median(turning_vals)) if turning_vals.size > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_turning_angle_landcover_pdf(transition_df, save_base, fontsize):
    print("Plotting turning-angle density curves by land cover")
    plot_df = transition_df.copy()
    plot_df["turning_angle_deg"] = pd.to_numeric(plot_df["turning_angle_deg"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["turning_angle_deg"])].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("colorblind", n_colors=max(plot_df["dominant_landcover"].nunique(), 1))
    color_lookup = {
        landcover: palette[idx]
        for idx, landcover in enumerate(
            ordered_landcovers(plot_df["dominant_landcover"].dropna().unique().tolist())
        )
    }
    legend_labels = []
    for landcover in ordered_landcovers(plot_df["dominant_landcover"].dropna().unique().tolist()):
        class_df = plot_df[plot_df["dominant_landcover"] == landcover].copy()
        if len(class_df) < 2:
            print(
                f"Skipping turning-angle KDE for land cover {landcover}: "
                f"only {len(class_df)} valid value(s)"
            )
            continue
        sns.kdeplot(
            data=class_df,
            x="turning_angle_deg",
            ax=ax,
            linewidth=2.2,
            label=landcover,
            color=color_lookup[landcover],
            fill=False,
            common_norm=False,
            bw_adjust=0.35,
            cut=0,
            clip=(0.0, 180.0),
            gridsize=512,
        )
        legend_labels.append(f"{format_landcover_label(landcover)} (n={len(class_df)})")
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(
            handles,
            legend_labels,
            frameon=False,
            fontsize=max(fontsize - 2, 8),
            title=None,
        )
    ax.set_xlim(0.0, 180.0)
    ax.set_xlabel("Turning angle between consecutive LFMC segments (degrees)", fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=max(fontsize - 2, 8))
    ax.text(
        0.97,
        0.97,
        (
            f"Turning angles = {len(plot_df)}\n"
            f"Land covers = {plot_df['dominant_landcover'].nunique()}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "0.4",
        },
        fontsize=max(fontsize - 3, 8),
    )
    fig.tight_layout()
    png_path = f"{save_base}.png"
    pdf_path = f"{save_base}.pdf"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png_path": png_path, "pdf_path": pdf_path}


def load_multitask_timeseries_entry(args):
    print("Loading dm128 multitask ensemble entry for evergreen timeseries inference")
    member_dirs = selected_member_dirs(
        ensemble_outputs_root=args.ensemble_outputs_root,
        member_name_prefix=args.ensemble_member_name_prefix,
        subset_size=args.ensemble_subset_size,
        subset_seed=args.ensemble_subset_seed,
    )
    member_site_errors = {}
    member_site_error_list = []
    for member_idx, member_dir in enumerate(member_dirs, start=1):
        print(
            f"  timeseries member {member_idx}/{len(member_dirs)}: "
            f"{os.path.basename(member_dir)}"
        )
        this_site_error = get_site_error(
            member_dir,
            progress_label=(
                f"multitask member {member_idx}/{len(member_dirs)} "
                f"({os.path.basename(member_dir)})"
            ),
        )
        member_site_errors[member_dir] = this_site_error
        member_site_error_list.append(this_site_error)
    return {
        "name": "multitask",
        "outputs_root": args.ensemble_outputs_root,
        "input_data_name": args.input_data_name,
        "model_type": "standard",
        "model_num_tasks": 3,
        "model_dir": args.ensemble_outputs_root,
        "is_ensemble": True,
        "ensemble_member_name_prefix": args.ensemble_member_name_prefix,
        "member_dirs": member_dirs,
        "member_site_errors": member_site_errors,
        "site_error": aggregate_site_errors(member_site_error_list),
    }


def select_landcover_timeseries_sites(site_r2_df, model_entry, n_sites, landcover_name):
    landcover_label = format_landcover_label(landcover_name)
    print(f"Selecting {landcover_label.lower()} sites for example timeseries figure")
    available_sites = set(model_entry["site_error"].keys())
    site_lookup_df = build_timeseries_site_lookup(available_sites)
    candidates = site_r2_df[site_r2_df["dominant_landcover"] == landcover_name].copy()
    candidates = candidates[np.isfinite(pd.to_numeric(candidates["site_r2"], errors="coerce"))]
    print(f"  {landcover_label} sites with finite site R2: {len(candidates)}")
    candidates["latitude_round"] = pd.to_numeric(candidates["latitude"], errors="coerce").round(6)
    candidates["longitude_round"] = pd.to_numeric(candidates["longitude"], errors="coerce").round(6)
    candidates = candidates.merge(
        site_lookup_df,
        on=["latitude_round", "longitude_round"],
        how="left",
    )
    candidates = candidates[candidates["timeseries_site_key"].notna()].copy()
    print(f"  {landcover_label} sites with compatible timeseries keys: {len(candidates)}")
    candidates["coord_site_key"] = candidates["timeseries_site_key"].astype(str)
    candidates = candidates.sort_values(
        ["site_r2", "n_points", "coord_site_key"],
        ascending=[False, False, True],
        kind="mergesort",
    ).drop_duplicates(subset=["coord_site_key"], keep="first")
    if len(candidates) < n_sites:
        raise ValueError(
            f"Only found {len(candidates)} {landcover_label.lower()} sites with {MIN_SITE_OBS}+ observations "
            f"and compatible timeseries inference, need {n_sites}"
        )
    candidates = candidates.sort_values(["coord_site_key"], kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(TIMESERIES_SELECTION_SEED)
    selected_idx = np.sort(rng.choice(len(candidates), size=int(n_sites), replace=False))
    selected = candidates.iloc[selected_idx].reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1, dtype=int)
    selected["selection_mode"] = TIMESERIES_SELECTION_MODE
    selected["selection_seed"] = TIMESERIES_SELECTION_SEED
    return selected


def build_timeseries_panel_title(site_row, selected_years):
    state_text = get_site_state_annotation(site_row["coord_site_key"])
    location_text = str(state_text) if state_text else str(site_row["coord_site_key"])
    years_text = (
        f"{selected_years[0]}-{selected_years[-1]}"
        if len(selected_years) > 0 else "Unknown years"
    )
    return (
        f"Site {site_row['site_key']} | {location_text}\n"
        f"R2 = {site_row['site_r2']:.2f} | N = {int(site_row['n_points'])} | Years = {years_text}"
    )


def build_evergreen_timeseries_panels(selected_sites_df, model_entry, args):
    print("Building evergreen timeseries panels")
    prediction_color = model_color(model_entry["name"]) or "#e75480"
    inference_cache = {}
    tensor_cache = {}
    runtime_cache = {}
    panels = []
    metadata_rows = []
    for _, site_row in selected_sites_df.iterrows():
        site_key = str(site_row["coord_site_key"])
        site_entry = model_entry["site_error"][site_key]
        obs_dates = _to_naive_datetime(site_entry["dates"])
        obs_vals = np.asarray(site_entry["true_values"], dtype=float)
        selected_years = top_consecutive_observation_years(obs_dates, args.timeseries_years)
        if len(selected_years) == 0:
            continue
        obs_dt = pd.to_datetime(obs_dates, errors="coerce")
        obs_mask = obs_dt.year.isin(selected_years)
        obs_dates = obs_dt[obs_mask].to_numpy(dtype="datetime64[ns]")
        obs_vals = obs_vals[obs_mask]
        start_date = pd.Timestamp(year=int(selected_years[0]), month=1, day=1)
        end_date = pd.Timestamp(year=int(selected_years[-1]), month=12, day=31)
        infer_out = get_model_inference_series(
            model_entry,
            site_key,
            start_date,
            end_date,
            inference_cache,
            tensor_cache,
            runtime_cache,
            args.inputs_root,
            args.timeseries_forward_batch_size,
        )
        pred_dates = pd.to_datetime(infer_out["dates"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        pred_vals = np.asarray(infer_out["lfmc_pred"], dtype=float)
        pred_std = np.asarray(infer_out.get("lfmc_pred_std", []), dtype=float)
        lower = None
        upper = None
        if len(pred_std) == len(pred_vals):
            lower = pred_vals - pred_std
            upper = pred_vals + pred_std
        panels.append(
            {
                "title": build_timeseries_panel_title(site_row, selected_years),
                "series": [
                    {
                        "label": "Prediction",
                        "dates": pred_dates,
                        "values": pred_vals,
                        "lower": lower,
                        "upper": upper,
                        "color": prediction_color,
                        "linewidth": 2.2,
                        "linestyle": "-",
                        "alpha": 0.95,
                        "fill_alpha": 0.14,
                        "zorder": 3,
                    },
                    {
                        "label": "Observed LFMC",
                        "dates": obs_dates,
                        "values": obs_vals,
                        "color": "#111111",
                        "linestyle": "",
                        "marker": "o",
                        "markersize": 4.8,
                        "alpha": 0.92,
                        "linewidth": 0.0,
                        "zorder": 4,
                    },
                ],
            }
        )
        metadata_rows.append(
            {
                "selection_rank": int(site_row["selection_rank"]),
                "site_key": site_row["site_key"],
                "coord_site_key": site_key,
                "dominant_landcover": site_row["dominant_landcover"],
                "site_r2": float(site_row["site_r2"]),
                "n_points": int(site_row["n_points"]),
                "latitude": float(site_row["latitude"]),
                "longitude": float(site_row["longitude"]),
                "selected_year_start": int(selected_years[0]),
                "selected_year_end": int(selected_years[-1]),
                "n_prediction_days": int(len(pred_vals)),
                "n_observation_points": int(len(obs_vals)),
            }
        )
    if len(panels) == 0:
        raise ValueError("No evergreen timeseries panels could be built")
    return panels, pd.DataFrame.from_records(metadata_rows)


def plot_evergreen_timeseries_grid(panels, save_base, fontsize, dpi):
    print("Plotting evergreen timeseries grid")
    n_panels = len(panels)
    ncols = 2
    nrows = int(np.ceil(n_panels / float(ncols)))
    prediction_color = panels[0]["series"][0]["color"]
    with plt.rc_context(paper_rc_params(fontsize)):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(15.5, max(12.0, 2.9 * nrows + 1.2)),
            constrained_layout=False,
        )
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        for panel_idx, panel in enumerate(panels):
            row_idx = panel_idx // ncols
            col_idx = panel_idx % ncols
            ax = axes[row_idx, col_idx]
            for series in panel["series"]:
                apply_panel_series(ax, series)
            ax.set_title(panel["title"], loc="left", pad=4)
            ax.set_ylabel("LFMC (%)")
            y_limits = panel_limits_from_series(panel["series"])
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            ax.grid(False)
            locator = mdates.MonthLocator(interval=6)
            formatter = mdates.DateFormatter("%Y-%m")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis="x", rotation=30)
            if row_idx < (nrows - 1):
                ax.tick_params(axis="x", labelbottom=False)
        for empty_idx in range(n_panels, nrows * ncols):
            row_idx = empty_idx // ncols
            col_idx = empty_idx % ncols
            axes[row_idx, col_idx].set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Date")
        legend_handles = [
            Line2D([0], [0], color=prediction_color, linewidth=2.2, label="Prediction"),
            Line2D(
                [0],
                [0],
                color="#111111",
                marker="o",
                linestyle="",
                markersize=5,
                label="Observed LFMC",
            ),
            Patch(
                facecolor=prediction_color,
                alpha=0.14,
                edgecolor="none",
                label="Ensemble-based uncertainty",
            ),
        ]
        fig.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.subplots_adjust(
            left=0.07,
            right=0.99,
            top=0.97,
            bottom=0.08,
            hspace=0.36,
            wspace=0.18,
        )
        outputs = {}
        for ext in ["png", "pdf"]:
            save_path = f"{save_base}.{ext}"
            fig.savefig(save_path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
            outputs[f"{ext}_path"] = save_path
        plt.close(fig)
    return outputs


def plot_uncertainty_hexbin(
    df,
    x_col,
    y_col,
    save_path,
    xlabel,
    ylabel,
    fontsize,
    gridsize,
):
    work_df = df[[x_col, y_col]].copy()
    work_df[x_col] = pd.to_numeric(work_df[x_col], errors="coerce")
    work_df[y_col] = pd.to_numeric(work_df[y_col], errors="coerce")
    work_df = work_df.dropna(subset=[x_col, y_col]).copy()
    if len(work_df) == 0:
        raise ValueError(f"No finite rows available for plot {os.path.basename(save_path)}")
    x_vals = work_df[x_col].to_numpy(dtype=float)
    y_vals = work_df[y_col].to_numpy(dtype=float)
    if len(work_df) < 2:
        raise ValueError(
            f"Need at least two finite rows for best-fit line in plot {os.path.basename(save_path)}"
        )
    corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    fit_vals = slope * x_vals + intercept
    fit_r2 = float(r2_score(y_vals, fit_vals))
    stats_text = (
        f"N = {len(work_df)}\n"
        f"mean x = {np.mean(x_vals):.3f}\n"
        f"mean y = {np.mean(y_vals):.3f}\n"
        f"slope = {slope:.3f}\n"
        f"intercept = {intercept:.3f}\n"
        f"r = {corr:.3f}\n"
        f"fit R2 = {fit_r2:.3f}"
    )
    generic_hexbin(
        x_vals,
        y_vals,
        save_path,
        gridsize=gridsize,
        xlabel=xlabel,
        ylabel=ylabel,
        cbar_label="Count",
        fontsize=fontsize,
        line_to_plot="correlation",
        stats_text=stats_text,
    )
    return {
        "plot_path": save_path,
        "n": int(len(work_df)),
        "x_mean": float(np.mean(x_vals)),
        "y_mean": float(np.mean(y_vals)),
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "fit_r": float(corr),
        "fit_r2": float(fit_r2),
    }


def write_summary(summary_dict, plot_dir):
    summary_path = os.path.join(plot_dir, "summary.json")
    with open(summary_path, "w") as file_obj:
        json.dump(summary_dict, file_obj, indent=2)
    print(f"Wrote summary JSON: {summary_path}")


def main():
    args = get_args()
    plot_dir = resolve_plot_dir(args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Writing outputs to: {plot_dir}")

    context = load_subset_eval_context(args)
    print(
        f"Loaded ensemble context with {len(context['member_dirs'])} members from "
        f"{args.ensemble_outputs_root}"
    )

    lfmc_df = prepare_lfmc_test_df(context["eval_df"])
    lfmc_csv_path = os.path.join(plot_dir, "lfmc_test_points_with_uncertainty.csv")
    lfmc_df.to_csv(lfmc_csv_path, index=False)
    print(f"Wrote LFMC test-point table: {lfmc_csv_path}")

    site_r2_df = build_site_r2_df(lfmc_df, plot_dir)
    site_r2_csv_path = os.path.join(plot_dir, "lfmc_test_site_r2_by_landcover.csv")
    site_r2_df.to_csv(site_r2_csv_path, index=False)
    print(f"Wrote site-level R2 table: {site_r2_csv_path}")

    summary = {
        "lfmc_test_points_csv": lfmc_csv_path,
        "site_r2_csv": site_r2_csv_path,
        "n_lfmc_test_points": int(len(lfmc_df)),
        "n_sites_with_r2": int(len(site_r2_df)),
        "n_ensemble_members": int(len(context["member_dirs"])),
        "ensemble_selection_note": context["ensemble_selection_note"],
        "ensemble_member_names": context["ensemble_member_names"],
    }

    site_r2_plot_paths = plot_site_r2_landcover_pdf(
        site_r2_df=site_r2_df,
        save_base=os.path.join(plot_dir, "lfmc_site_r2_landcover_pdf"),
        fontsize=args.fontsize,
    )
    summary["site_r2_landcover_pdf"] = site_r2_plot_paths

    transition_df = build_lfmc_transition_df(lfmc_df, site_r2_df)
    transition_csv_path = os.path.join(plot_dir, "lfmc_site_transitions_by_landcover.csv")
    transition_df.to_csv(transition_csv_path, index=False)
    print(f"Wrote LFMC transition table: {transition_csv_path}")
    summary["lfmc_transition_csv"] = transition_csv_path

    turning_angle_df = summarize_turning_angle_by_landcover(transition_df)
    turning_angle_csv_path = os.path.join(plot_dir, "lfmc_turning_angle_by_landcover.csv")
    turning_angle_df.to_csv(turning_angle_csv_path, index=False)
    print(f"Wrote turning-angle summary: {turning_angle_csv_path}")
    summary["lfmc_turning_angle_csv"] = turning_angle_csv_path

    turning_angle_plot_paths = plot_turning_angle_landcover_pdf(
        transition_df,
        save_base=os.path.join(plot_dir, "lfmc_turning_angle_landcover_pdf"),
        fontsize=args.fontsize,
    )
    summary["lfmc_turning_angle_landcover_pdf"] = {
        **turning_angle_plot_paths,
        "n_landcovers": int(len(turning_angle_df)),
        "min_site_observations": int(MIN_SITE_OBS),
        "metric_basis": "consecutive_adjacent_segment_turning_angles_pooled_by_landcover",
    }

    abs_error_plot = plot_uncertainty_hexbin(
        df=lfmc_df,
        x_col="pred_std_ensemble",
        y_col="abs_error",
        save_path=os.path.join(plot_dir, "lfmc_abs_error_vs_uncertainty_hexbin.png"),
        xlabel="Ensemble uncertainty (LFMC std across dm128 members)",
        ylabel="Absolute LFMC error (%)",
        fontsize=args.fontsize,
        gridsize=args.hexbin_gridsize,
    )
    summary["abs_error_vs_uncertainty"] = abs_error_plot

    error_norm_plot = plot_uncertainty_hexbin(
        df=lfmc_df,
        x_col="uncertainty_norm_by_pred",
        y_col="abs_error_norm_by_pred",
        save_path=os.path.join(
            plot_dir,
            "lfmc_error_vs_uncertainty_norm_by_prediction_hexbin.png",
        ),
        xlabel="Ensemble uncertainty normalized by |prediction|",
        ylabel="Absolute LFMC error normalized by |prediction|",
        fontsize=args.fontsize,
        gridsize=args.hexbin_gridsize,
    )
    summary["error_vs_uncertainty_norm_by_pred"] = error_norm_plot

    error_scaled_plot = plot_uncertainty_hexbin(
        df=lfmc_df,
        x_col="uncertainty_norm_by_pred_scaled",
        y_col="abs_error_norm_by_pred",
        save_path=os.path.join(
            plot_dir,
            "lfmc_error_vs_uncertainty_norm_by_prediction_scaled01_hexbin.png",
        ),
        xlabel="Scaled normalized uncertainty (0-1)",
        ylabel="Absolute LFMC error normalized by |prediction|",
        fontsize=args.fontsize,
        gridsize=args.hexbin_gridsize,
    )
    summary["error_vs_uncertainty_norm_by_pred_scaled01"] = error_scaled_plot

    model_entry = load_multitask_timeseries_entry(args)
    evergreen_sites_df = select_landcover_timeseries_sites(
        site_r2_df=site_r2_df,
        model_entry=model_entry,
        n_sites=args.num_evergreen_sites,
        landcover_name="evergreen_forest",
    )
    evergreen_sites_csv_path = os.path.join(
        plot_dir,
        "evergreen_timeseries_sites_random_sample.csv",
    )
    evergreen_sites_df.to_csv(evergreen_sites_csv_path, index=False)
    print(f"Wrote evergreen site selection table: {evergreen_sites_csv_path}")

    evergreen_panels, evergreen_panel_df = build_evergreen_timeseries_panels(
        selected_sites_df=evergreen_sites_df,
        model_entry=model_entry,
        args=args,
    )
    evergreen_panel_csv_path = os.path.join(
        plot_dir,
        "evergreen_timeseries_panel_metadata_random_sample.csv",
    )
    evergreen_panel_df.to_csv(evergreen_panel_csv_path, index=False)
    print(f"Wrote evergreen panel metadata: {evergreen_panel_csv_path}")

    evergreen_plot_paths = plot_evergreen_timeseries_grid(
        panels=evergreen_panels,
        save_base=os.path.join(plot_dir, "evergreen_timeseries_examples_random_sample"),
        fontsize=args.fontsize,
        dpi=300,
    )
    summary["evergreen_timeseries_examples_random_sample"] = {
        **evergreen_plot_paths,
        "site_selection_csv": evergreen_sites_csv_path,
        "panel_metadata_csv": evergreen_panel_csv_path,
        "n_sites": int(len(evergreen_panel_df)),
        "selection_mode": TIMESERIES_SELECTION_MODE,
        "selection_seed": int(TIMESERIES_SELECTION_SEED),
        "years_per_panel": int(args.timeseries_years),
        "ensemble_selection_note": context["ensemble_selection_note"],
    }

    shrubland_sites_df = select_landcover_timeseries_sites(
        site_r2_df=site_r2_df,
        model_entry=model_entry,
        n_sites=args.num_evergreen_sites,
        landcover_name="shrub",
    )
    shrubland_sites_csv_path = os.path.join(
        plot_dir,
        "shrubland_timeseries_sites_random_sample.csv",
    )
    shrubland_sites_df.to_csv(shrubland_sites_csv_path, index=False)
    print(f"Wrote shrubland site selection table: {shrubland_sites_csv_path}")

    shrubland_panels, shrubland_panel_df = build_evergreen_timeseries_panels(
        selected_sites_df=shrubland_sites_df,
        model_entry=model_entry,
        args=args,
    )
    shrubland_panel_csv_path = os.path.join(
        plot_dir,
        "shrubland_timeseries_panel_metadata_random_sample.csv",
    )
    shrubland_panel_df.to_csv(shrubland_panel_csv_path, index=False)
    print(f"Wrote shrubland panel metadata: {shrubland_panel_csv_path}")

    shrubland_plot_paths = plot_evergreen_timeseries_grid(
        panels=shrubland_panels,
        save_base=os.path.join(plot_dir, "shrubland_timeseries_examples_random_sample"),
        fontsize=args.fontsize,
        dpi=300,
    )
    summary["shrubland_timeseries_examples_random_sample"] = {
        **shrubland_plot_paths,
        "site_selection_csv": shrubland_sites_csv_path,
        "panel_metadata_csv": shrubland_panel_csv_path,
        "n_sites": int(len(shrubland_panel_df)),
        "selection_mode": TIMESERIES_SELECTION_MODE,
        "selection_seed": int(TIMESERIES_SELECTION_SEED),
        "years_per_panel": int(args.timeseries_years),
        "ensemble_selection_note": context["ensemble_selection_note"],
    }

    write_summary(summary, plot_dir)
    print("Finished dm128 test uncertainty evaluation.")


if __name__ == "__main__":
    main()
