import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(here, "../../.."))
sys.path.append(os.path.join(project_root, "lfmc_model", "utils"))
sys.path.append(os.path.join(project_root, "lfmc_model", "scripts", "paper_figures"))

from plotting import bar_plot
from eval_deep import (
    _build_row_keys,
    _format_metric_with_std,
    build_lfmc_space_time_tables,
    build_site_month_anomaly_eval_df,
    _metric_std,
    build_site_landcover_lookup,
    build_lfmc_y2y_df,
    compute_basic_metrics,
    compute_landcover_decomposition_metrics,
    compute_landcover_y2y_metrics,
    compute_monthly_y2y_metrics,
    compute_site_y2y_metrics,
    load_eval_context,
    prepare_lfmc_landcover_eval_df,
)
from paper_figure_plotting import plot_landcover_comparison_panels


LANDCOVER_CLASS_ORDER = [
    "deciduous_forest",
    "evergreen_forest",
    "shrub",
    "grass",
    "mixed_forest",
]
SINGLE_TASK_COLOR = "#6a3d9a"
MULTITASK_COLOR = "#e75480"


def get_args():
    parser = argparse.ArgumentParser(
        description="Compare two LFMC models or ensembles with deep-eval style summaries."
    )
    parser.add_argument("--model_a_name", type=str, required=True)
    parser.add_argument("--model_b_name", type=str, required=True)
    parser.add_argument("--model_a_model_dir", type=str, default=None)
    parser.add_argument("--model_b_model_dir", type=str, default=None)
    parser.add_argument("--model_a_outputs_root", type=str, default=None)
    parser.add_argument("--model_b_outputs_root", type=str, default=None)
    parser.add_argument("--model_a_ensemble_outputs_root", type=str, default=None)
    parser.add_argument("--model_b_ensemble_outputs_root", type=str, default=None)
    parser.add_argument("--model_a_ensemble_member_name_prefix", type=str, default=None)
    parser.add_argument("--model_b_ensemble_member_name_prefix", type=str, default=None)
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--fontsize", type=int, default=16)
    return parser.parse_args()


def resolve_plot_dir(model_a_name, model_b_name, plot_dir):
    if plot_dir is not None:
        return plot_dir
    scratch_root = os.environ.get("SCRATCH", "/scratch/users/trobinet")
    safe_a = str(model_a_name).replace(" ", "_")
    safe_b = str(model_b_name).replace(" ", "_")
    return os.path.join(
        scratch_root,
        "long_lfmc",
        "final_lfmc",
        "lfmc_model",
        "plots",
        "eval_compare_models",
        f"{safe_a}_vs_{safe_b}",
    )


def comparison_model_color(model_name):
    name = str(model_name).strip().lower().replace("-", "_")
    if "dm64" in name:
        return "#7ca596"
    if "dm128" in name:
        return "#d07c55"
    if name in {"lfmc", "lfmc_ens", "single_task", "singletask"}:
        return SINGLE_TASK_COLOR
    if name in {"lfmc_vh_vv", "lfmc_vv_vh", "lfmc_vh_vv_ens", "lfmc_vh_vv_ens_fullrandom", "multitask"}:
        return MULTITASK_COLOR
    return None


def _load_named_context(
    name,
    model_dir=None,
    outputs_root=None,
    ensemble_outputs_root=None,
    ensemble_member_name_prefix=None,
):
    context = load_eval_context(
        model_dir=model_dir,
        outputs_root=outputs_root,
        ensemble_outputs_root=ensemble_outputs_root,
        ascending=True,
        ensemble_member_name_prefix=ensemble_member_name_prefix,
    )
    context["name"] = name
    return context


def _lfmc_target_df(eval_df):
    out = eval_df[eval_df["target"] == "lfmc"].reset_index(drop=True).copy()
    if len(out) == 0:
        raise ValueError("No LFMC rows found for model comparison")
    out["_row_key"] = _build_row_keys(out)
    return out


def align_lfmc_frames(df_a, df_b):
    df_a = _lfmc_target_df(df_a)
    df_b = _lfmc_target_df(df_b)
    merged = df_a.merge(
        df_b[["_row_key", "pred", "obs"]],
        on="_row_key",
        how="inner",
        suffixes=("_a", "_b"),
    )
    if len(merged) == 0:
        raise ValueError("No overlapping LFMC rows found between the two models")
    if not np.allclose(
        merged["obs_a"].to_numpy(dtype=float),
        merged["obs_b"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-4,
        equal_nan=True,
    ):
        raise ValueError("Observed LFMC values do not align across the two models")
    return merged


def _member_metric_std(member_eval_dfs):
    if member_eval_dfs is None or len(member_eval_dfs) == 0:
        return {"rmse": np.nan, "r2": np.nan}
    rmse_vals = []
    r2_vals = []
    for member_eval_df in member_eval_dfs:
        member_lfmc = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
        if len(member_lfmc) == 0:
            continue
        metrics = compute_basic_metrics(member_lfmc["obs"].values, member_lfmc["pred"].values)
        rmse_vals.append(metrics["rmse"])
        r2_vals.append(metrics["r2"])
    return {
        "rmse": _metric_std(rmse_vals),
        "r2": _metric_std(r2_vals),
    }


def _target_member_metric_std(member_eval_dfs, target_name):
    if member_eval_dfs is None or len(member_eval_dfs) == 0:
        return {"rmse": np.nan, "r2": np.nan}
    rmse_vals = []
    r2_vals = []
    for member_eval_df in member_eval_dfs:
        member_target_df = member_eval_df[member_eval_df["target"] == target_name].reset_index(drop=True)
        if len(member_target_df) == 0:
            continue
        metrics = compute_basic_metrics(member_target_df["obs"].values, member_target_df["pred"].values)
        rmse_vals.append(metrics["rmse"])
        r2_vals.append(metrics["r2"])
    return {
        "rmse": _metric_std(rmse_vals),
        "r2": _metric_std(r2_vals),
    }


def compute_overall_summary(context, target_name="lfmc"):
    target_df = context["eval_df"][context["eval_df"]["target"] == target_name].reset_index(drop=True)
    metrics = compute_basic_metrics(target_df["obs"].values, target_df["pred"].values)
    if target_name == "lfmc":
        member_std = _member_metric_std(context["member_eval_dfs"])
    else:
        member_std = _target_member_metric_std(context["member_eval_dfs"], target_name)
        if context["member_eval_dfs"] is not None and len(context["member_eval_dfs"]) > 0:
            rmse_vals = []
            r2_vals = []
            for member_eval_df in context["member_eval_dfs"]:
                member_target_df = member_eval_df[member_eval_df["target"] == target_name].reset_index(drop=True)
                if len(member_target_df) == 0:
                    continue
                member_metrics = compute_basic_metrics(
                    member_target_df["obs"].values,
                    member_target_df["pred"].values,
                )
                rmse_vals.append(member_metrics["rmse"])
                r2_vals.append(member_metrics["r2"])
            finite_rmse = np.asarray(rmse_vals, dtype=float)
            finite_rmse = finite_rmse[np.isfinite(finite_rmse)]
            finite_r2 = np.asarray(r2_vals, dtype=float)
            finite_r2 = finite_r2[np.isfinite(finite_r2)]
            if finite_rmse.size > 0:
                metrics["rmse"] = float(np.mean(finite_rmse))
            if finite_r2.size > 0:
                metrics["r2"] = float(np.mean(finite_r2))
    metrics["target"] = target_name
    metrics["rmse_std_across_members"] = member_std["rmse"]
    metrics["r2_std_across_members"] = member_std["r2"]
    return metrics


def plot_error_distribution(merged_df, summary_a, summary_b, name_a, name_b, save_path, fontsize):
    err_a = np.abs(merged_df["pred_a"].to_numpy(dtype=float) - merged_df["obs_a"].to_numpy(dtype=float))
    err_b = np.abs(merged_df["pred_b"].to_numpy(dtype=float) - merged_df["obs_a"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.ecdfplot(err_a, label=name_a, ax=ax, linewidth=2.0, color=comparison_model_color(name_a))
    sns.ecdfplot(err_b, label=name_b, ax=ax, linewidth=2.0, color=comparison_model_color(name_b))
    ax.set_xlabel("Absolute LFMC error", fontsize=fontsize)
    ax.set_ylabel("Cumulative probability", fontsize=fontsize)
    ax.tick_params(labelsize=max(fontsize - 2, 8))
    stats_text = (
        f"{name_a}: R² = {_format_metric_with_std(summary_a['r2'], summary_a['r2_std_across_members'])}\n"
        f"{name_b}: R² = {_format_metric_with_std(summary_b['r2'], summary_b['r2_std_across_members'])}\n"
        f"N = {len(merged_df)}"
    )
    ax.text(
        0.97,
        0.03,
        stats_text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.4"},
        fontsize=max(fontsize - 3, 8),
    )
    ax.legend(frameon=False, fontsize=max(fontsize - 2, 8))
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def compute_variability_summary(context, min_obs=20, min_years=5):
    lfmc_y2y_df = build_lfmc_y2y_df(context["eval_df"])
    month_df = compute_monthly_y2y_metrics(
        lfmc_df=lfmc_y2y_df,
        min_obs=min_obs,
        min_years=min_years,
    )
    site_df = compute_site_y2y_metrics(
        lfmc_df=lfmc_y2y_df,
        min_obs=min_obs,
        min_years=min_years,
    )
    if len(month_df) > 0:
        overall_series = month_df["overall_pct_variability_captured_source_centered"].dropna()
        overall_pct = float(overall_series.iloc[0]) if len(overall_series) > 0 else np.nan
    else:
        overall_pct = np.nan
    overall_fraction = overall_pct / 100.0 if np.isfinite(overall_pct) else np.nan
    overall_std = np.nan
    member_eval_dfs = context["member_eval_dfs"]
    if member_eval_dfs is not None and len(member_eval_dfs) > 0:
        member_vals = []
        for member_eval_df in member_eval_dfs:
            member_lfmc_y2y_df = build_lfmc_y2y_df(member_eval_df)
            member_month_df = compute_monthly_y2y_metrics(
                lfmc_df=member_lfmc_y2y_df,
                min_obs=min_obs,
                min_years=min_years,
            )
            if len(member_month_df) == 0:
                continue
            member_overall_series = member_month_df["overall_pct_variability_captured_source_centered"].dropna()
            if len(member_overall_series) == 0:
                continue
            member_vals.append(float(member_overall_series.iloc[0]) / 100.0)
        overall_std = _metric_std(member_vals)
    return {
        "overall_fraction_yearly_variability_captured": overall_fraction,
        "overall_fraction_yearly_variability_captured_std": overall_std,
        "site_df": site_df,
    }


def plot_variability_distribution(site_df_a, site_df_b, summary_a, summary_b, name_a, name_b, save_path, fontsize):
    vals_a = site_df_a["pct_variability_captured_source_centered"].to_numpy(dtype=float) / 100.0
    vals_b = site_df_b["pct_variability_captured_source_centered"].to_numpy(dtype=float) / 100.0
    vals_a = vals_a[np.isfinite(vals_a)]
    vals_b = vals_b[np.isfinite(vals_b)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(vals_a) > 1:
        sns.ecdfplot(vals_a, label=name_a, ax=ax, linewidth=2.0, color=comparison_model_color(name_a))
    if len(vals_b) > 1:
        sns.ecdfplot(vals_b, label=name_b, ax=ax, linewidth=2.0, color=comparison_model_color(name_b))
    ax.set_xlabel("Fraction of yearly variability captured", fontsize=fontsize)
    ax.set_ylabel("Cumulative probability", fontsize=fontsize)
    ax.tick_params(labelsize=max(fontsize - 2, 8))
    stats_text = (
        f"{name_a}: {_format_metric_with_std(summary_a['overall_fraction_yearly_variability_captured'], summary_a['overall_fraction_yearly_variability_captured_std'])}\n"
        f"{name_b}: {_format_metric_with_std(summary_b['overall_fraction_yearly_variability_captured'], summary_b['overall_fraction_yearly_variability_captured_std'])}\n"
        f"N = {len(vals_a)} / {len(vals_b)} sites"
    )
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.4"},
        fontsize=max(fontsize - 3, 8),
    )
    ax.legend(frameon=False, fontsize=max(fontsize - 2, 8))
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _get_training_member_dirs(context):
    if context["member_dirs"]:
        return list(context["member_dirs"])
    return [context["model_dir"]]


def _ordered_landcover_categories(categories):
    categories = list(categories)
    ordered = []
    if "overall" in categories:
        ordered.append("overall")
    present_classes = [cls for cls in LANDCOVER_CLASS_ORDER if cls in categories]
    remaining_classes = sorted(
        [cls for cls in categories if cls not in present_classes and cls != "overall"]
    )
    return ordered + present_classes + remaining_classes


def _load_train_info_union(model_dir):
    fold_frames = []
    for fold_num in range(1, 7):
        train_info_path = os.path.join(model_dir, f"fold_{fold_num}", "train_info.csv")
        if not os.path.exists(train_info_path):
            raise FileNotFoundError(f"Missing training info: {train_info_path}")
        fold_df = pd.read_csv(train_info_path, low_memory=False)
        fold_frames.append(fold_df)
    df = pd.concat(fold_frames, ignore_index=True, sort=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _prepare_train_landcover_fraction_summary(context, plot_dir):
    member_dirs = _get_training_member_dirs(context)
    member_fraction_frames = []
    member_iter = tqdm(
        member_dirs,
        desc=f"Preparing training land-cover fractions for {context['name']}",
        unit="member",
    )
    for member_idx, member_dir in enumerate(member_iter, start=1):
        member_iter.set_postfix_str(f"{member_idx}/{len(member_dirs)}")
        train_df = _load_train_info_union(member_dir)
        required_cols = ["latitude", "longitude", "date"]
        missing_cols = [col for col in required_cols if col not in train_df.columns]
        if len(missing_cols) > 0:
            raise KeyError(
                f"Training info is missing required columns {missing_cols}: "
                f"{os.path.join(member_dir, 'fold_1', 'train_info.csv')} ... fold_6/train_info.csv"
            )
        train_df["site_key"] = (
            pd.to_numeric(train_df["latitude"], errors="coerce").round(5).astype(str) + "_" +
            pd.to_numeric(train_df["longitude"], errors="coerce").round(5).astype(str)
        )
        train_df["year"] = pd.to_datetime(train_df["date"], errors="coerce").dt.year
        lookup_path = os.path.join(plot_dir, "train_site_landcover_lookup.csv")
        requested_sites = (
            train_df[["site_key", "latitude", "longitude"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        missing_site_count = len(requested_sites)
        if os.path.exists(lookup_path):
            existing_df = pd.read_csv(lookup_path, dtype={"site_key": str})
            if "site_key" in existing_df.columns:
                missing_site_count = int(
                    len(set(requested_sites["site_key"].astype(str)) - set(existing_df["site_key"].astype(str)))
                )
        tqdm.write(
            f"[{context['name']}] member {member_idx}/{len(member_dirs)}: "
            f"{missing_site_count} site(s) need land-cover lookup"
        )
        site_lookup_df = build_site_landcover_lookup(
            train_df[["site_key", "latitude", "longitude", "year"]].copy(),
            lookup_path,
        )
        site_lookup_df["site_key"] = site_lookup_df["site_key"].astype(str)
        train_df = train_df.merge(
            site_lookup_df[["site_key", "dominant_landcover"]],
            on="site_key",
            how="left",
        )
        train_df = train_df[train_df["dominant_landcover"].notna()].copy()
        if len(train_df) == 0:
            continue
        counts = (
            train_df["dominant_landcover"]
            .value_counts(dropna=False)
            .rename_axis("dominant_landcover")
            .reset_index(name="n_train_rows_category")
        )
        counts["fraction_train_rows"] = (
            counts["n_train_rows_category"] / float(len(train_df))
        )
        counts["n_train_rows_total"] = int(len(train_df))
        member_fraction_frames.append(counts)
    if len(member_fraction_frames) == 0:
        return pd.DataFrame()
    all_landcovers = _ordered_landcover_categories(
        set().union(*[set(frame["dominant_landcover"].tolist()) for frame in member_fraction_frames])
    )
    records = []
    for landcover in all_landcovers:
        vals = []
        category_rows = []
        total_rows = []
        for frame in member_fraction_frames:
            row = frame[frame["dominant_landcover"] == landcover]
            if len(row) > 0:
                vals.append(float(row.iloc[0]["fraction_train_rows"]))
                category_rows.append(float(row.iloc[0]["n_train_rows_category"]))
            else:
                vals.append(0.0)
                category_rows.append(0.0)
            total_rows.append(float(frame["n_train_rows_total"].iloc[0]))
        records.append(
            {
                "dominant_landcover": landcover,
                "fraction_train_rows": float(np.mean(vals)),
                "fraction_train_rows_std": _metric_std(vals),
                "n_train_rows_category_mean": float(np.mean(category_rows)),
                "n_train_rows_total_mean": float(np.mean(total_rows)),
            }
        )
    out = pd.DataFrame.from_records(records)
    out["model"] = context["name"]
    return out


def _overall_metric_std(context):
    member_eval_dfs = context["member_eval_dfs"]
    if member_eval_dfs is None or len(member_eval_dfs) == 0:
        return {"rmse": np.nan, "r2": np.nan}
    vals = []
    for member_eval_df in member_eval_dfs:
        member_lfmc_df = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
        if len(member_lfmc_df) == 0:
            continue
        vals.append(
            compute_basic_metrics(
                member_lfmc_df["obs"].values,
                member_lfmc_df["pred"].values,
            )["r2"]
        )
    return {"r2": _metric_std(vals)}


def _space_time_metric_stds(context):
    member_eval_dfs = context["member_eval_dfs"]
    if member_eval_dfs is None or len(member_eval_dfs) == 0:
        nan_metric = {"rmse": np.nan, "r2": np.nan}
        return nan_metric, nan_metric
    mean_vals = []
    anomaly_vals = []
    for member_eval_df in member_eval_dfs:
        member_lfmc_df = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
        if len(member_lfmc_df) == 0:
            continue
        site_summary_df, anomaly_df = build_lfmc_space_time_tables(member_lfmc_df)
        if len(site_summary_df) > 0:
            mean_vals.append(
                compute_basic_metrics(
                    site_summary_df["obs_mean"].values,
                    site_summary_df["pred_mean"].values,
                )["r2"]
            )
        if len(anomaly_df) > 0:
            anomaly_vals.append(
                compute_basic_metrics(
                    anomaly_df["obs_anom"].values,
                    anomaly_df["pred_anom"].values,
                )["r2"]
            )
    return {"r2": _metric_std(mean_vals)}, {"r2": _metric_std(anomaly_vals)}


def _monthly_source_centered_metric_std(context, min_obs, min_years):
    member_eval_dfs = context["member_eval_dfs"]
    if member_eval_dfs is None or len(member_eval_dfs) == 0:
        return {"rmse": np.nan, "r2": np.nan}
    vals = []
    for member_eval_df in member_eval_dfs:
        month_anom_df, valid_month_groups = build_site_month_anomaly_eval_df(
            lfmc_df=build_lfmc_y2y_df(member_eval_df),
            min_obs=min_obs,
            min_years=min_years,
        )
        if len(month_anom_df) == 0 or len(valid_month_groups) == 0:
            continue
        vals.append(
            compute_basic_metrics(
                month_anom_df["obs_dev"].values,
                month_anom_df["pred_dev"].values,
            )["r2"]
        )
    return {"r2": _metric_std(vals)}


def _attach_shared_landcover_lookup(member_eval_df, site_lookup_df):
    member_lfmc_df = build_lfmc_y2y_df(member_eval_df)
    if len(member_lfmc_df) == 0:
        return pd.DataFrame()
    member_lfmc_df = member_lfmc_df.copy()
    member_lfmc_df["site_key"] = member_lfmc_df["site_key"].astype(str)
    member_lfmc_df = member_lfmc_df.merge(
        site_lookup_df[["site_key", "dominant_landcover", "dominant_landcover_frac"]],
        on="site_key",
        how="left",
    )
    member_lfmc_df = member_lfmc_df[member_lfmc_df["dominant_landcover"].notna()].copy()
    if len(member_lfmc_df) == 0:
        return pd.DataFrame()
    member_lfmc_df["site_obs_mean"] = member_lfmc_df.groupby("site_key")["obs"].transform("mean")
    member_lfmc_df["site_pred_mean"] = member_lfmc_df.groupby("site_key")["pred"].transform("mean")
    member_lfmc_df["site_obs_anom"] = member_lfmc_df["obs"] - member_lfmc_df["site_obs_mean"]
    member_lfmc_df["site_pred_anom"] = member_lfmc_df["pred"] - member_lfmc_df["site_pred_mean"]
    grp_cols = ["site_key", "month"]
    member_lfmc_df["seasonal_obs_mean"] = member_lfmc_df.groupby(grp_cols)["obs"].transform("mean")
    member_lfmc_df["seasonal_pred_mean"] = member_lfmc_df.groupby(grp_cols)["pred"].transform("mean")
    member_lfmc_df["seasonal_obs_anom"] = member_lfmc_df["obs"] - member_lfmc_df["seasonal_obs_mean"]
    member_lfmc_df["seasonal_pred_anom"] = member_lfmc_df["pred"] - member_lfmc_df["seasonal_pred_mean"]
    return member_lfmc_df


def _landcover_metric_tables(context, plot_dir, min_obs=20, min_years=5):
    lfmc_lc_df = prepare_lfmc_landcover_eval_df(context["eval_df"], plot_dir)
    if len(lfmc_lc_df) == 0:
        return pd.DataFrame()
    site_lookup_path = os.path.join(plot_dir, "lfmc_site_landcover_lookup.csv")
    site_lookup_df = build_site_landcover_lookup(
        build_lfmc_y2y_df(context["eval_df"]),
        site_lookup_path,
    )
    site_lookup_df = site_lookup_df.copy()
    site_lookup_df["site_key"] = site_lookup_df["site_key"].astype(str)
    metric_df = compute_landcover_decomposition_metrics(lfmc_lc_df)
    member_eval_dfs = context["member_eval_dfs"]
    if member_eval_dfs is not None and len(member_eval_dfs) > 0:
        member_metric_frames = []
        for member_eval_df in member_eval_dfs:
            member_lfmc_lc_df = _attach_shared_landcover_lookup(member_eval_df, site_lookup_df)
            if len(member_lfmc_lc_df) > 0:
                member_metric_frames.append(compute_landcover_decomposition_metrics(member_lfmc_lc_df))
        if len(member_metric_frames) > 0:
            for metric_name in ["overall_r2", "site_mean_r2", "site_anom_r2"]:
                std_lookup = {}
                for landcover in metric_df["dominant_landcover"].tolist():
                    vals = []
                    for member_metric_df in member_metric_frames:
                        row = member_metric_df[member_metric_df["dominant_landcover"] == landcover]
                        if len(row) > 0:
                            vals.append(float(row.iloc[0][metric_name]))
                    std_lookup[landcover] = _metric_std(vals)
                metric_df[f"{metric_name}_std"] = metric_df["dominant_landcover"].map(std_lookup)
    y2y_df, _, _ = compute_landcover_y2y_metrics(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=min_obs,
        min_years=min_years,
        plot_dir=plot_dir,
    )
    if len(y2y_df) > 0:
        metric_df = metric_df.merge(
            y2y_df[
                [
                    "dominant_landcover",
                    "pct_variability_captured_source_centered",
                    "n_groups",
                    "total_obs",
                ]
            ],
            on="dominant_landcover",
            how="left",
        )
    else:
        metric_df["pct_variability_captured_source_centered"] = np.nan
        metric_df["n_groups"] = np.nan
        metric_df["total_obs"] = np.nan
    metric_df["fraction_yearly_variability_captured"] = (
        metric_df["pct_variability_captured_source_centered"] / 100.0
    )
    if member_eval_dfs is not None and len(member_eval_dfs) > 0:
        fraction_lookup = {}
        for landcover in metric_df["dominant_landcover"].tolist():
            vals = []
            for member_eval_df in member_eval_dfs:
                member_y2y_df, _, _ = compute_landcover_y2y_metrics(
                    lfmc_df=build_lfmc_y2y_df(member_eval_df),
                    min_obs=min_obs,
                    min_years=min_years,
                    plot_dir=plot_dir,
                )
                row = member_y2y_df[member_y2y_df["dominant_landcover"] == landcover]
                if len(row) > 0:
                    vals.append(float(row.iloc[0]["pct_variability_captured_source_centered"]) / 100.0)
            fraction_lookup[landcover] = _metric_std(vals)
        metric_df["fraction_yearly_variability_captured_std"] = metric_df["dominant_landcover"].map(
            fraction_lookup
        )
    month_anom_df, valid_month_groups = build_site_month_anomaly_eval_df(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=min_obs,
        min_years=min_years,
    )
    if len(month_anom_df) > 0 and len(valid_month_groups) > 0:
        month_anom_df = month_anom_df.merge(
            site_lookup_df[["site_key", "dominant_landcover"]],
            on="site_key",
            how="left",
        )
        month_anom_df = month_anom_df[month_anom_df["dominant_landcover"].notna()].copy()
        month_r2_df = (
            month_anom_df.groupby("dominant_landcover", dropna=False)
            .apply(
                lambda df: pd.Series(
                    {
                        "monthly_dev_r2": compute_basic_metrics(
                            df["obs_dev"].values,
                            df["pred_dev"].values,
                        )["r2"],
                    }
                )
            )
            .reset_index()
        )
        metric_df = metric_df.merge(
            month_r2_df,
            on="dominant_landcover",
            how="left",
        )
        if member_eval_dfs is not None and len(member_eval_dfs) > 0:
            std_lookup = {}
            for landcover in metric_df["dominant_landcover"].tolist():
                vals = []
                for member_eval_df in member_eval_dfs:
                    member_month_anom_df, member_valid_groups = build_site_month_anomaly_eval_df(
                        lfmc_df=build_lfmc_y2y_df(member_eval_df),
                        min_obs=min_obs,
                        min_years=min_years,
                    )
                    if len(member_month_anom_df) == 0 or len(member_valid_groups) == 0:
                        continue
                    member_month_anom_df = member_month_anom_df.merge(
                        site_lookup_df[["site_key", "dominant_landcover"]],
                        on="site_key",
                        how="left",
                    )
                    member_month_anom_df = member_month_anom_df[
                        member_month_anom_df["dominant_landcover"] == landcover
                    ].copy()
                    if len(member_month_anom_df) == 0:
                        continue
                    vals.append(
                        compute_basic_metrics(
                            member_month_anom_df["obs_dev"].values,
                            member_month_anom_df["pred_dev"].values,
                        )["r2"]
                    )
                std_lookup[landcover] = _metric_std(vals)
            metric_df["monthly_dev_r2_std"] = metric_df["dominant_landcover"].map(std_lookup)
    else:
        metric_df["monthly_dev_r2"] = np.nan
        metric_df["monthly_dev_r2_std"] = np.nan
    metric_df["overall_n"] = metric_df["n_points"]
    metric_df["site_anom_n"] = metric_df["n_points"]
    metric_df["site_mean_n"] = metric_df["n_sites"]
    metric_df["monthly_dev_n"] = metric_df["total_obs"]
    return metric_df


def _prepend_overall_landcover_metrics(context, metric_df, min_obs, min_years):
    lfmc_df = context["eval_df"][context["eval_df"]["target"] == "lfmc"].reset_index(drop=True)
    site_summary_df, anomaly_df = build_lfmc_space_time_tables(lfmc_df)
    month_anom_df, _ = build_site_month_anomaly_eval_df(
        lfmc_df=build_lfmc_y2y_df(context["eval_df"]),
        min_obs=min_obs,
        min_years=min_years,
    )
    overall_std = _overall_metric_std(context)
    mean_std, anomaly_std = _space_time_metric_stds(context)
    monthly_std = _monthly_source_centered_metric_std(context, min_obs=min_obs, min_years=min_years)
    overall_metric = compute_basic_metrics(lfmc_df["obs"].values, lfmc_df["pred"].values)
    anomaly_metric = compute_basic_metrics(anomaly_df["obs_anom"].values, anomaly_df["pred_anom"].values)
    mean_metric = compute_basic_metrics(site_summary_df["obs_mean"].values, site_summary_df["pred_mean"].values)
    monthly_metric = compute_basic_metrics(month_anom_df["obs_dev"].values, month_anom_df["pred_dev"].values)
    overall_row = {column: np.nan for column in metric_df.columns}
    overall_row["dominant_landcover"] = "overall"
    overall_row["overall_r2"] = overall_metric.get("r2", np.nan)
    overall_row["site_anom_r2"] = anomaly_metric.get("r2", np.nan)
    overall_row["site_mean_r2"] = mean_metric.get("r2", np.nan)
    overall_row["monthly_dev_r2"] = monthly_metric.get("r2", np.nan)
    overall_row["overall_r2_std"] = overall_std.get("r2", np.nan)
    overall_row["site_anom_r2_std"] = anomaly_std.get("r2", np.nan)
    overall_row["site_mean_r2_std"] = mean_std.get("r2", np.nan)
    overall_row["monthly_dev_r2_std"] = monthly_std.get("r2", np.nan)
    overall_row["overall_n"] = overall_metric.get("n", np.nan)
    overall_row["site_anom_n"] = anomaly_metric.get("n", np.nan)
    overall_row["site_mean_n"] = mean_metric.get("n", np.nan)
    overall_row["monthly_dev_n"] = monthly_metric.get("n", np.nan)
    overall_row["n_points"] = overall_metric.get("n", np.nan)
    overall_row["n_sites"] = mean_metric.get("n", np.nan)
    overall_row["total_obs"] = monthly_metric.get("n", np.nan)
    overall_row["n_groups"] = np.nan
    overall_row["pct_variability_captured_source_centered"] = np.nan
    overall_row["fraction_yearly_variability_captured"] = np.nan
    overall_row["fraction_yearly_variability_captured_std"] = np.nan
    return pd.concat([pd.DataFrame([overall_row]), metric_df], ignore_index=True)


def build_landcover_comparison_df(context_a, context_b, plot_dir, min_obs=20, min_years=5):
    df_a = _prepend_overall_landcover_metrics(
        context_a,
        _landcover_metric_tables(context_a, plot_dir, min_obs=min_obs, min_years=min_years).copy(),
        min_obs=min_obs,
        min_years=min_years,
    )
    df_b = _prepend_overall_landcover_metrics(
        context_b,
        _landcover_metric_tables(context_b, plot_dir, min_obs=min_obs, min_years=min_years).copy(),
        min_obs=min_obs,
        min_years=min_years,
    )
    if len(df_a) == 0 and len(df_b) == 0:
        return pd.DataFrame()
    if len(df_a) > 0:
        df_a["model"] = context_a["name"]
    if len(df_b) > 0:
        df_b["model"] = context_b["name"]
    return pd.concat([df_a, df_b], ignore_index=True, sort=False)


def _comparison_plot_arrays(compare_df, metric_name, metric_std_name, count_col, model_names):
    categories = _ordered_landcover_categories(
        compare_df["dominant_landcover"].dropna().unique().tolist()
    )
    value_rows = []
    err_rows = []
    count_rows = []
    for landcover in categories:
        row_vals = []
        row_errs = []
        row_counts = []
        for model_name in model_names:
            row = compare_df[
                (compare_df["dominant_landcover"] == landcover) &
                (compare_df["model"] == model_name)
            ]
            if len(row) == 0:
                row_vals.append(np.nan)
                row_errs.append(np.nan)
                row_counts.append(np.nan)
            else:
                row_vals.append(float(row.iloc[0][metric_name]))
                row_errs.append(float(row.iloc[0][metric_std_name]) if metric_std_name in row.columns else np.nan)
                row_counts.append(float(row.iloc[0][count_col]))
        value_rows.append(row_vals)
        err_rows.append(row_errs)
        count_rows.append(row_counts)
    return categories, np.asarray(value_rows, dtype=float), np.asarray(err_rows, dtype=float), np.asarray(count_rows, dtype=float)


def write_summary(summary_rows, plot_dir):
    summary_csv = os.path.join(plot_dir, "model_comparison_summary.csv")
    summary_json = os.path.join(plot_dir, "model_comparison_summary.json")
    df = pd.DataFrame.from_records(summary_rows)
    df.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as file_obj:
        json.dump(summary_rows, file_obj, indent=2)
    print(f"Wrote comparison summary CSV: {summary_csv}")
    print(f"Wrote comparison summary JSON: {summary_json}")


def plot_vv_vh_metric_comparison(summary_df, model_names, save_path):
    metric_order = [
        ("vv", "r2", "VV R²"),
        ("vv", "rmse", "VV RMSE"),
        ("vh", "r2", "VH R²"),
        ("vh", "rmse", "VH RMSE"),
    ]
    categories = [label for _, _, label in metric_order]
    values = []
    errors = []
    counts = []
    for _, model_name in enumerate(model_names):
        model_vals = []
        model_errs = []
        model_counts = []
        for target_name, metric_name, _ in metric_order:
            row = summary_df[
                (summary_df["model"] == model_name) &
                (summary_df["target"] == target_name)
            ]
            if len(row) == 0:
                model_vals.append(np.nan)
                model_errs.append(np.nan)
                model_counts.append(np.nan)
            else:
                model_vals.append(float(row.iloc[0][metric_name]))
                model_errs.append(float(row.iloc[0][f"{metric_name}_std_across_members"]))
                model_counts.append(float(row.iloc[0]["n"]))
        values.append(model_vals)
        errors.append(model_errs)
        counts.append(model_counts)
    values = np.asarray(values, dtype=float).T
    errors = np.asarray(errors, dtype=float).T
    counts = np.asarray(counts, dtype=float).T
    model_colors = [comparison_model_color(name) for name in model_names]
    bar_plot(
        categories=categories,
        values=values,
        xlabel="Metric",
        ylabel="Metric value",
        save_path=save_path,
        label_with_n=True,
        sample_counts=counts,
        subcategory_labels=model_names,
        subcategory_colors=model_colors,
        errors=errors,
    )


def plot_training_landcover_fraction_comparison(compare_df, model_names, save_path):
    categories = _ordered_landcover_categories(
        compare_df["dominant_landcover"].dropna().unique().tolist()
    )
    value_rows = []
    err_rows = []
    count_rows = []
    for landcover in categories:
        row_vals = []
        row_errs = []
        row_counts = []
        for model_name in model_names:
            row = compare_df[
                (compare_df["dominant_landcover"] == landcover) &
                (compare_df["model"] == model_name)
            ]
            if len(row) == 0:
                row_vals.append(np.nan)
                row_errs.append(np.nan)
                row_counts.append(np.nan)
            else:
                row_vals.append(float(row.iloc[0]["fraction_train_rows"]))
                row_errs.append(float(row.iloc[0]["fraction_train_rows_std"]))
                row_counts.append(float(row.iloc[0]["n_train_rows_category_mean"]))
        value_rows.append(row_vals)
        err_rows.append(row_errs)
        count_rows.append(row_counts)
    bar_plot(
        categories=categories,
        values=np.asarray(value_rows, dtype=float),
        xlabel="Dominant land cover",
        ylabel="Fraction of training rows",
        save_path=save_path,
        label_with_n=True,
        sample_counts=np.asarray(count_rows, dtype=float),
        subcategory_labels=model_names,
        subcategory_colors=[comparison_model_color(name) for name in model_names],
        errors=np.asarray(err_rows, dtype=float),
    )


def plot_paper_style_landcover_comparison(compare_df, model_names, save_path, fontsize):
    model_colors = [comparison_model_color(name) for name in model_names]
    panels = []
    for title, ylabel, metric_name, metric_std_name, count_col in [
        ("Overall", "LFMC R²", "overall_r2", "overall_r2_std", "overall_n"),
        ("Anomaly", "LFMC R²", "site_anom_r2", "site_anom_r2_std", "site_anom_n"),
        ("Mean", "LFMC R²", "site_mean_r2", "site_mean_r2_std", "site_mean_n"),
        ("Monthly Anomaly", "LFMC R²", "monthly_dev_r2", "monthly_dev_r2_std", "monthly_dev_n"),
    ]:
        categories, values, errors, counts = _comparison_plot_arrays(
            compare_df,
            metric_name,
            metric_std_name,
            count_col,
            model_names,
        )
        panels.append(
            {
                "title": title,
                "ylabel": ylabel,
                "values": values,
                "errors": errors,
                "counts": counts,
            }
        )
    plot_landcover_comparison_panels(
        categories=categories,
        model_labels=model_names,
        colors=model_colors,
        panels=panels,
        save_path=save_path,
        fontsize=fontsize,
        figsize=(12, 14),
        dpi=300,
    )


def main():
    args = get_args()
    plot_dir = resolve_plot_dir(args.model_a_name, args.model_b_name, args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    context_a = _load_named_context(
        args.model_a_name,
        model_dir=args.model_a_model_dir,
        outputs_root=args.model_a_outputs_root,
        ensemble_outputs_root=args.model_a_ensemble_outputs_root,
        ensemble_member_name_prefix=args.model_a_ensemble_member_name_prefix,
    )
    context_b = _load_named_context(
        args.model_b_name,
        model_dir=args.model_b_model_dir,
        outputs_root=args.model_b_outputs_root,
        ensemble_outputs_root=args.model_b_ensemble_outputs_root,
        ensemble_member_name_prefix=args.model_b_ensemble_member_name_prefix,
    )
    merged_lfmc = align_lfmc_frames(context_a["eval_df"], context_b["eval_df"])
    aligned_a = pd.DataFrame({"obs": merged_lfmc["obs_a"], "pred": merged_lfmc["pred_a"]})
    aligned_b = pd.DataFrame({"obs": merged_lfmc["obs_a"], "pred": merged_lfmc["pred_b"]})
    summary_a = compute_basic_metrics(aligned_a["obs"].values, aligned_a["pred"].values)
    summary_b = compute_basic_metrics(aligned_b["obs"].values, aligned_b["pred"].values)
    std_a = _member_metric_std(context_a["member_eval_dfs"])
    std_b = _member_metric_std(context_b["member_eval_dfs"])
    summary_a["target"] = "lfmc"
    summary_b["target"] = "lfmc"
    summary_a["rmse_std_across_members"] = std_a["rmse"]
    summary_a["r2_std_across_members"] = std_a["r2"]
    summary_b["rmse_std_across_members"] = std_b["rmse"]
    summary_b["r2_std_across_members"] = std_b["r2"]
    print(
        f"{args.model_a_name}: R2={_format_metric_with_std(summary_a['r2'], summary_a['r2_std_across_members'])}, "
        f"RMSE={_format_metric_with_std(summary_a['rmse'], summary_a['rmse_std_across_members'])}, "
        f"N={summary_a['n']}"
    )
    print(
        f"{args.model_b_name}: R2={_format_metric_with_std(summary_b['r2'], summary_b['r2_std_across_members'])}, "
        f"RMSE={_format_metric_with_std(summary_b['rmse'], summary_b['rmse_std_across_members'])}, "
        f"N={summary_b['n']}"
    )
    vv_summary_a = compute_overall_summary(context_a, target_name="vv")
    vv_summary_b = compute_overall_summary(context_b, target_name="vv")
    vh_summary_a = compute_overall_summary(context_a, target_name="vh")
    vh_summary_b = compute_overall_summary(context_b, target_name="vh")
    for target_label, sum_a, sum_b in [
        ("VV", vv_summary_a, vv_summary_b),
        ("VH", vh_summary_a, vh_summary_b),
    ]:
        print(
            f"{args.model_a_name} {target_label}: "
            f"R2={_format_metric_with_std(sum_a['r2'], sum_a['r2_std_across_members'])}, "
            f"RMSE={_format_metric_with_std(sum_a['rmse'], sum_a['rmse_std_across_members'])}, "
            f"N={sum_a['n']}"
        )
        print(
            f"{args.model_b_name} {target_label}: "
            f"R2={_format_metric_with_std(sum_b['r2'], sum_b['r2_std_across_members'])}, "
            f"RMSE={_format_metric_with_std(sum_b['rmse'], sum_b['rmse_std_across_members'])}, "
            f"N={sum_b['n']}"
        )
    error_plot_path = os.path.join(plot_dir, "lfmc_absolute_error_distribution.png")
    plot_error_distribution(
        merged_lfmc,
        summary_a,
        summary_b,
        args.model_a_name,
        args.model_b_name,
        error_plot_path,
        args.fontsize,
    )
    print(f"Wrote error distribution plot: {error_plot_path}")
    compare_df = build_landcover_comparison_df(context_a, context_b, plot_dir)
    if len(compare_df) > 0:
        compare_csv_path = os.path.join(plot_dir, "lfmc_landcover_model_comparison.csv")
        compare_df.to_csv(compare_csv_path, index=False)
        model_names = [args.model_a_name, args.model_b_name]
        lc_categories, r2_values, r2_errors, r2_counts = _comparison_plot_arrays(
            compare_df,
            "overall_r2",
            "overall_r2_std",
            "n_points",
            model_names,
        )
        r2_plot_path = os.path.join(plot_dir, "lfmc_landcover_model_comparison_r2.png")
        bar_plot(
            categories=lc_categories,
            values=r2_values,
            xlabel="Dominant land cover",
            ylabel="LFMC R²",
            save_path=r2_plot_path,
            label_with_n=True,
            sample_counts=r2_counts,
            subcategory_labels=model_names,
            subcategory_colors=[comparison_model_color(name) for name in model_names],
            errors=r2_errors,
        )
        frac_categories, frac_values, frac_errors, frac_counts = _comparison_plot_arrays(
            compare_df,
            "fraction_yearly_variability_captured",
            "fraction_yearly_variability_captured_std",
            "n_groups",
            model_names,
        )
        frac_plot_path = os.path.join(
            plot_dir,
            "lfmc_landcover_model_comparison_fraction_yearly_variability.png",
        )
        bar_plot(
            categories=frac_categories,
            values=frac_values,
            xlabel="Dominant land cover",
            ylabel="Fraction of yearly variability captured",
            save_path=frac_plot_path,
            label_with_n=True,
            sample_counts=frac_counts,
            subcategory_labels=model_names,
            subcategory_colors=[comparison_model_color(name) for name in model_names],
            errors=frac_errors,
        )
        print(f"Wrote land-cover comparison CSV: {compare_csv_path}")
        print(f"Wrote land-cover R² comparison plot: {r2_plot_path}")
        print(f"Wrote land-cover variability comparison plot: {frac_plot_path}")
        panel_plot_path = os.path.join(
            plot_dir,
            "lfmc_landcover_model_comparison_panels.png",
        )
        plot_paper_style_landcover_comparison(
            compare_df,
            model_names,
            panel_plot_path,
            args.fontsize,
        )
        print(f"Wrote paper-style land-cover comparison plot: {panel_plot_path}")
    vv_vh_summary_df = pd.DataFrame.from_records(
        [
            {"model": args.model_a_name, **vv_summary_a},
            {"model": args.model_b_name, **vv_summary_b},
            {"model": args.model_a_name, **vh_summary_a},
            {"model": args.model_b_name, **vh_summary_b},
        ]
    )
    vv_vh_csv_path = os.path.join(plot_dir, "vv_vh_metric_comparison.csv")
    vv_vh_plot_path = os.path.join(plot_dir, "vv_vh_metric_comparison.png")
    vv_vh_summary_df.to_csv(vv_vh_csv_path, index=False)
    plot_vv_vh_metric_comparison(
        vv_vh_summary_df,
        [args.model_a_name, args.model_b_name],
        vv_vh_plot_path,
    )
    print(f"Wrote VV/VH metric comparison CSV: {vv_vh_csv_path}")
    print(f"Wrote VV/VH metric comparison plot: {vv_vh_plot_path}")
    variability_a = compute_variability_summary(context_a, min_obs=20, min_years=5)
    variability_b = compute_variability_summary(context_b, min_obs=20, min_years=5)
    variability_csv_path = os.path.join(plot_dir, "yearly_variability_site_distribution.csv")
    variability_plot_path = os.path.join(plot_dir, "yearly_variability_site_distribution.png")
    variability_site_df = pd.concat(
        [
            variability_a["site_df"].assign(model=args.model_a_name),
            variability_b["site_df"].assign(model=args.model_b_name),
        ],
        ignore_index=True,
        sort=False,
    )
    variability_site_df.to_csv(variability_csv_path, index=False)
    plot_variability_distribution(
        variability_a["site_df"],
        variability_b["site_df"],
        variability_a,
        variability_b,
        args.model_a_name,
        args.model_b_name,
        variability_plot_path,
        args.fontsize,
    )
    print(f"Wrote yearly-variability distribution CSV: {variability_csv_path}")
    print(f"Wrote yearly-variability distribution plot: {variability_plot_path}")
    train_lc_a = _prepare_train_landcover_fraction_summary(context_a, plot_dir)
    train_lc_b = _prepare_train_landcover_fraction_summary(context_b, plot_dir)
    if len(train_lc_a) > 0 or len(train_lc_b) > 0:
        train_lc_compare_df = pd.concat([train_lc_a, train_lc_b], ignore_index=True, sort=False)
        train_lc_csv_path = os.path.join(plot_dir, "training_landcover_fraction_comparison.csv")
        train_lc_plot_path = os.path.join(plot_dir, "training_landcover_fraction_comparison.png")
        train_lc_compare_df.to_csv(train_lc_csv_path, index=False)
        plot_training_landcover_fraction_comparison(
            train_lc_compare_df,
            [args.model_a_name, args.model_b_name],
            train_lc_plot_path,
        )
        print(f"Wrote training land-cover fraction CSV: {train_lc_csv_path}")
        print(f"Wrote training land-cover fraction plot: {train_lc_plot_path}")
    write_summary(
        [
            {"model": args.model_a_name, **summary_a},
            {"model": args.model_b_name, **summary_b},
            {"model": args.model_a_name, **vv_summary_a},
            {"model": args.model_b_name, **vv_summary_b},
            {"model": args.model_a_name, **vh_summary_a},
            {"model": args.model_b_name, **vh_summary_b},
        ],
        plot_dir,
    )


if __name__ == "__main__":
    main()
