import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pyproj import Transformer
from sklearn.metrics import r2_score
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(here, "../../.."))
sys.path.append(os.path.join(project_root, "lfmc_model", "utils"))

from plotting import annotated_bar_plot, bar_plot, generic_hexbin, generic_scatter, map_points


DEFAULT_OUTPUTS_ROOT = "/scratch/users/trobinet/long_lfmc/final_lfmc/lfmc_model/outputs/lfmc"
DEFAULT_SORT_METRIC = "test_insitu_rmse"
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_NLCD_FRAC_PATH = "/scratch/users/trobinet/long_lfmc/final_lfmc/nlcd/nlcd_target_grid_2000_2024.zarr"
_NLCD_FRAC_DS = None
_WGS84_TO_5070 = Transformer.from_crs("epsg:4326", "epsg:5070", always_xy=True)


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run deeper figure-oriented evaluation for a single LFMC model."
        )
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Specific model directory to analyze.",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=DEFAULT_OUTPUTS_ROOT,
        help=(
            "Model family outputs root containing transformer_* model dirs and "
            "optionally model_summary_results.csv."
        ),
    )
    parser.add_argument(
        "--model_df_index",
        type=int,
        default=None,
        help=(
            "Optional original DataFrame index from model_summary_results.csv "
            "to select a specific model row."
        ),
    )
    parser.add_argument(
        "--sort_metric",
        type=str,
        default=DEFAULT_SORT_METRIC,
        help="Metric used to choose a model from model_summary_results.csv.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending when selecting the best model from the summary CSV.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to write figures and summary tables.",
    )
    parser.add_argument(
        "--hexbin_gridsize",
        type=int,
        default=60,
        help="Hexbin grid size for predicted-vs-observed figures.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=16,
        help="Font size used for figure labels and annotations.",
    )
    parser.add_argument(
        "--ensemble_outputs_root",
        type=str,
        default=None,
        help=(
            "Optional ensemble root containing multiple completed transformer_* "
            "member directories. If provided, eval_deep aggregates ensemble-mean "
            "predictions and reports metric std across members."
        ),
    )
    return parser.parse_args()


def resolve_plot_dir(model_dir, plot_dir, outputs_root=None):
    if plot_dir is not None:
        return plot_dir
    scratch_root = os.environ.get(
        "SCRATCH",
        "/scratch/users/trobinet",
    )
    if outputs_root is not None:
        family_name = os.path.basename(outputs_root.rstrip("/"))
    else:
        family_name = os.path.basename(os.path.dirname(model_dir.rstrip("/")))
    if family_name == "":
        family_name = os.path.basename(model_dir.rstrip("/"))
    return os.path.join(
        scratch_root,
        "long_lfmc",
        "final_lfmc",
        "lfmc_model",
        "plots",
        "eval_deep",
        family_name,
    )


def get_nlcd_frac_ds():
    global _NLCD_FRAC_DS
    if _NLCD_FRAC_DS is None:
        if not os.path.exists(DEFAULT_NLCD_FRAC_PATH):
            raise FileNotFoundError(
                f"Missing NLCD fractional dataset: {DEFAULT_NLCD_FRAC_PATH}"
            )
        print(f"Opening NLCD fractional dataset: {DEFAULT_NLCD_FRAC_PATH}")
        _NLCD_FRAC_DS = xr.open_zarr(DEFAULT_NLCD_FRAC_PATH)
    return _NLCD_FRAC_DS


def is_complete_model_dir(model_dir):
    if not os.path.isdir(model_dir):
        return False
    fold_info_path = os.path.join(model_dir, "fold_info.json")
    if not os.path.exists(fold_info_path):
        return False
    with open(fold_info_path, "r") as file_obj:
        fold_info = json.load(file_obj)
    for fold in fold_info.keys():
        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        required_files = [
            os.path.join(fold_dir, "test_info.csv"),
            os.path.join(fold_dir, "test_outputs.pth"),
        ]
        if not all(os.path.exists(path) for path in required_files):
            return False
    return True


def select_ensemble_member_dirs(outputs_root):
    candidates = []
    for name in os.listdir(outputs_root):
        model_dir = os.path.join(outputs_root, name)
        if name.startswith("transformer_") and is_complete_model_dir(model_dir):
            candidates.append(model_dir)
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No complete transformer_* model dirs found under ensemble root {outputs_root}"
        )
    return sorted(candidates)


def select_model_dir(outputs_root, model_df_index=None, sort_metric=DEFAULT_SORT_METRIC, ascending=True):
    summary_csv = os.path.join(outputs_root, "model_summary_results.csv")
    if os.path.exists(summary_csv):
        print(f"Selecting model using summary CSV: {summary_csv}")
        summary_df = pd.read_csv(summary_csv)
        if "model_dir" in summary_df.columns:
            if model_df_index is not None:
                if model_df_index not in summary_df.index:
                    raise KeyError(
                        f"Requested model_df_index={model_df_index} not found in "
                        f"{summary_csv}. Available index range: "
                        f"{int(summary_df.index.min())} to {int(summary_df.index.max())}."
                    )
                model_dir = summary_df.loc[model_df_index, "model_dir"]
                if not is_complete_model_dir(model_dir):
                    raise FileNotFoundError(
                        f"Model dir at DataFrame index {model_df_index} is incomplete: {model_dir}"
                    )
                print(
                    f"Selected model {model_dir} from DataFrame index "
                    f"{model_df_index}."
                )
                return model_dir
            if sort_metric in summary_df.columns:
                summary_df[sort_metric] = pd.to_numeric(
                    summary_df[sort_metric],
                    errors="coerce",
                )
                summary_df = summary_df.dropna(subset=[sort_metric]).copy()
                summary_df = summary_df.sort_values(sort_metric, ascending=ascending)
            for model_dir in summary_df["model_dir"].tolist():
                if is_complete_model_dir(model_dir):
                    print(
                        f"Selected model {model_dir} from summary CSV using "
                        f"{sort_metric}."
                    )
                    return model_dir
            raise FileNotFoundError(
                "No complete model_dir entries found in model_summary_results.csv."
            )
    elif model_df_index is not None:
        raise FileNotFoundError(
            f"--model_df_index was provided but no model_summary_results.csv exists under {outputs_root}"
        )
    print(
        "No usable summary CSV found. Falling back to the most recently modified "
        "complete transformer_* directory."
    )
    candidates = []
    for name in os.listdir(outputs_root):
        model_dir = os.path.join(outputs_root, name)
        if name.startswith("transformer_") and is_complete_model_dir(model_dir):
            candidates.append(model_dir)
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No complete transformer_* model dirs found under {outputs_root}"
        )
    candidates = sorted(candidates, key=lambda path: os.path.getmtime(path), reverse=True)
    print(f"Selected fallback model: {candidates[0]}")
    return candidates[0]


def compute_basic_metrics(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "r2": np.nan,
        }
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    if len(obs) < 2:
        r2 = np.nan
    else:
        r2 = float(r2_score(obs, pred))
    return {
        "n": int(len(obs)),
        "rmse": rmse,
        "r2": r2,
    }


def _metric_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    return float(np.std(arr, ddof=0))


def _format_metric_with_std(value, std, fmt="{:.2f}"):
    if not np.isfinite(value):
        return "nan"
    base = fmt.format(value)
    if np.isfinite(std):
        return f"{base} +/- {fmt.format(std)}"
    return base


def _extract_target_frame(test_info, preds, true_vals, mask, target_name, fold):
    target_info = test_info.loc[mask].reset_index(drop=True).copy()
    if len(target_info) == 0:
        return pd.DataFrame()
    if preds is None or true_vals is None:
        return pd.DataFrame()
    preds = np.asarray(preds, dtype=float)
    true_vals = np.asarray(true_vals, dtype=float)
    if preds.ndim == 0 or true_vals.ndim == 0:
        return pd.DataFrame()
    aligned_n = min(len(target_info), len(preds), len(true_vals))
    if aligned_n == 0:
        return pd.DataFrame()
    if aligned_n != len(target_info) or aligned_n != len(preds) or aligned_n != len(true_vals):
        print(
            f"Warning: {target_name} alignment mismatch in fold {fold}; "
            f"using first {aligned_n} rows."
        )
    target_info = target_info.iloc[:aligned_n].copy()
    target_info["pred"] = preds[:aligned_n]
    target_info["obs"] = true_vals[:aligned_n]
    target_info["target"] = target_name
    target_info["fold"] = str(fold)
    return target_info


def load_fold_predictions(model_dir):
    fold_info_path = os.path.join(model_dir, "fold_info.json")
    with open(fold_info_path, "r") as file_obj:
        fold_info = json.load(file_obj)
    fold_frames = []
    for fold in fold_info.keys():
        print(f"Loading test outputs for fold {fold}")
        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        test_info_path = os.path.join(fold_dir, "test_info.csv")
        test_outputs_path = os.path.join(fold_dir, "test_outputs.pth")
        test_info = pd.read_csv(test_info_path, low_memory=False)
        test_outputs = torch.load(test_outputs_path, map_location="cpu", weights_only=False)
        source = test_info["source"].astype(str)
        frame_builders = [
            (
                "lfmc",
                source == "nfmd",
                test_outputs.get("lfmc_preds", []),
                test_outputs.get("lfmc_true", []),
            ),
            (
                "vv",
                source.str.startswith("vv"),
                test_outputs.get("vv_preds", []),
                test_outputs.get("vv_true", []),
            ),
            (
                "vh",
                source.str.startswith("vh"),
                test_outputs.get("vh_preds", []),
                test_outputs.get("vh_true", []),
            ),
        ]
        for target_name, mask, preds, true_vals in frame_builders:
            target_frame = _extract_target_frame(
                test_info=test_info,
                preds=preds,
                true_vals=true_vals,
                mask=mask,
                target_name=target_name,
                fold=fold,
            )
            if len(target_frame) > 0:
                fold_frames.append(target_frame)
                print(
                    f"  Added {len(target_frame)} rows for target {target_name} "
                    f"from fold {fold}"
                )
    if len(fold_frames) == 0:
        raise ValueError(f"No evaluation rows found in model dir: {model_dir}")
    eval_df = pd.concat(fold_frames, ignore_index=True)
    eval_df["date"] = pd.to_datetime(eval_df["date"], errors="coerce")
    return eval_df


def _build_row_keys(frame):
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


def aggregate_member_eval_frames(member_frames):
    if len(member_frames) == 0:
        raise ValueError("Cannot aggregate empty member frame list")
    template = member_frames[0].copy().reset_index(drop=True)
    template["_row_key"] = _build_row_keys(template)
    pred_stack = []
    for idx, frame in enumerate(member_frames):
        work = frame.copy().reset_index(drop=True)
        work["_row_key"] = _build_row_keys(work)
        if len(work) != len(template):
            raise ValueError(
                f"Member {idx} row count mismatch: {len(work)} vs {len(template)}"
            )
        if not template["_row_key"].equals(work["_row_key"]):
            raise ValueError(f"Member {idx} row alignment mismatch in ensemble evaluation frames")
        pred_stack.append(work["pred"].to_numpy(dtype=float))
    pred_stack = np.stack(pred_stack, axis=1)
    out = template.drop(columns=["_row_key"]).copy()
    out["pred"] = pred_stack.mean(axis=1)
    out["pred_std_ensemble"] = pred_stack.std(axis=1, ddof=0)
    return out


def compute_target_metrics_summary(target_df, member_target_dfs=None, aggregate_mode="prediction"):
    if aggregate_mode not in {"prediction", "member_metric"}:
        raise ValueError(f"Unsupported aggregate_mode: {aggregate_mode}")
    metrics = compute_basic_metrics(target_df["obs"].values, target_df["pred"].values)
    rmse_std = np.nan
    r2_std = np.nan
    if member_target_dfs is not None and len(member_target_dfs) > 0:
        member_metrics_list = [
            compute_basic_metrics(member_df["obs"].values, member_df["pred"].values)
            for member_df in member_target_dfs
        ]
        rmse_vals = [member_metrics["rmse"] for member_metrics in member_metrics_list]
        r2_vals = [member_metrics["r2"] for member_metrics in member_metrics_list]
        rmse_std = _metric_std(rmse_vals)
        r2_std = _metric_std(r2_vals)
        if aggregate_mode == "member_metric":
            finite_rmse = np.asarray(rmse_vals, dtype=float)
            finite_rmse = finite_rmse[np.isfinite(finite_rmse)]
            finite_r2 = np.asarray(r2_vals, dtype=float)
            finite_r2 = finite_r2[np.isfinite(finite_r2)]
            if finite_rmse.size > 0:
                metrics["rmse"] = float(np.mean(finite_rmse))
            else:
                metrics["rmse"] = np.nan
            if finite_r2.size > 0:
                metrics["r2"] = float(np.mean(finite_r2))
            else:
                metrics["r2"] = np.nan
    metrics["n"] = int(len(target_df))
    metrics["rmse_std_across_members"] = rmse_std
    metrics["r2_std_across_members"] = r2_std
    return metrics


def plot_target_hexbin(
    target_df,
    target_name,
    plot_dir,
    gridsize,
    fontsize,
    member_target_dfs=None,
    aggregate_mode="prediction",
):
    metrics = compute_target_metrics_summary(
        target_df,
        member_target_dfs=member_target_dfs,
        aggregate_mode=aggregate_mode,
    )
    if metrics["n"] == 0:
        print(f"Skipping {target_name}: no valid observations.")
        return None
    target_label_lookup = {
        "lfmc": "LFMC (%)",
        "vv": "VV (dB)",
        "vh": "VH (dB)",
    }
    target_label = target_label_lookup.get(target_name, target_name.upper())
    stats_text = (
        f"R\u00b2 = {_format_metric_with_std(metrics['r2'], metrics['r2_std_across_members'])}\n"
        f"RMSE = {_format_metric_with_std(metrics['rmse'], metrics['rmse_std_across_members'])}\n"
        f"N = {metrics['n']}"
    )
    save_path = os.path.join(plot_dir, f"{target_name}_pred_obs_hexbin.png")
    obs = target_df["obs"].to_numpy(dtype=float)
    pred = target_df["pred"].to_numpy(dtype=float)
    finite_mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[finite_mask]
    pred = pred[finite_mask]
    if len(obs) == 0:
        print(f"Skipping {target_name}: no finite observations.")
        return None
    data_min = float(min(np.min(obs), np.min(pred)))
    data_max = float(max(np.max(obs), np.max(pred)))
    generic_hexbin(
        obs,
        pred,
        save_path,
        gridsize=gridsize,
        xlabel=f"Observed {target_label}",
        ylabel=f"Predicted {target_label}",
        xlim=(data_min, data_max),
        ylim=(data_min, data_max),
        cbar_label="Count",
        fontsize=fontsize,
        line_to_plot="one_to_one",
        stats_text=stats_text,
    )
    print(f"Wrote {target_name} performance plot: {save_path}")
    metrics["plot_path"] = save_path
    return metrics


def compute_correlation_metrics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    metrics = {"n": int(len(x)), "corr": np.nan}
    if len(x) < 2:
        return metrics
    metrics["corr"] = float(np.corrcoef(x, y)[0, 1])
    return metrics


def build_cross_target_pairs(eval_df, left_target, right_target):
    left_df = eval_df[eval_df["target"] == left_target].copy()
    right_df = eval_df[eval_df["target"] == right_target].copy()
    if len(left_df) == 0 or len(right_df) == 0:
        return pd.DataFrame()
    pair_cols = [
        col for col in [
            "fold", "date", "latitude", "longitude",
            "site_id", "site_name", "fuel_type", "fuel",
        ]
        if col in left_df.columns and col in right_df.columns
    ]
    if len(pair_cols) == 0:
        print(
            f"Skipping {left_target}-{right_target} pairing: "
            "no shared metadata columns found."
        )
        return pd.DataFrame()
    left_df = left_df[pair_cols + ["obs"]].rename(columns={"obs": f"{left_target}_obs"})
    right_df = right_df[pair_cols + ["obs"]].rename(columns={"obs": f"{right_target}_obs"})
    pair_df = left_df.merge(right_df, on=pair_cols, how="inner")
    if len(pair_df) == 0:
        return pd.DataFrame()
    pair_df = pair_df.drop_duplicates().reset_index(drop=True)
    return pair_df


def plot_multitask_lfmc_vs_sar_hexbin(eval_df, plot_dir, gridsize, fontsize):
    label_lookup = {
        "vv": "VV (dB)",
        "vh": "VH (dB)",
    }
    metrics_out = {}
    for sar_target in ["vv", "vh"]:
        pair_df = build_cross_target_pairs(eval_df, "lfmc", sar_target)
        if len(pair_df) == 0:
            print(f"No paired LFMC/{sar_target.upper()} rows found; skipping.")
            continue
        lfmc_obs = pair_df["lfmc_obs"].to_numpy(dtype=float)
        sar_obs = pair_df[f"{sar_target}_obs"].to_numpy(dtype=float)
        finite_mask = np.isfinite(lfmc_obs) & np.isfinite(sar_obs)
        lfmc_obs = lfmc_obs[finite_mask]
        sar_obs = sar_obs[finite_mask]
        if len(lfmc_obs) == 0:
            print(f"No finite paired LFMC/{sar_target.upper()} rows found; skipping.")
            continue
        metrics = compute_correlation_metrics(lfmc_obs, sar_obs)
        save_path = os.path.join(plot_dir, f"lfmc_vs_{sar_target}_hexbin.png")
        generic_hexbin(
            lfmc_obs,
            sar_obs,
            save_path,
            gridsize=gridsize,
            xlabel="Observed LFMC (%)",
            ylabel=f"Observed {label_lookup[sar_target]}",
            cbar_label="Count",
            fontsize=fontsize,
            line_to_plot="correlation",
            stats_text=(
                f"r = {_format_metric_with_std(metrics['corr'], np.nan, fmt='{:.2f}')}\n"
                f"N = {metrics['n']}"
            ),
        )
        print(f"Wrote multitask LFMC-vs-{sar_target.upper()} plot: {save_path}")
        metrics["plot_path"] = save_path
        metrics_out[f"lfmc_vs_{sar_target}"] = metrics
    return metrics_out


def build_lfmc_space_time_tables(lfmc_df):
    required_cols = ["latitude", "longitude", "obs", "pred", "date"]
    missing_cols = [col for col in required_cols if col not in lfmc_df.columns]
    if len(missing_cols) > 0:
        raise KeyError(
            f"LFMC dataframe is missing required columns for space/time analysis: {missing_cols}"
        )
    work_df = lfmc_df.copy()
    work_df["latitude"] = pd.to_numeric(work_df["latitude"], errors="coerce")
    work_df["longitude"] = pd.to_numeric(work_df["longitude"], errors="coerce")
    work_df["obs"] = pd.to_numeric(work_df["obs"], errors="coerce")
    work_df["pred"] = pd.to_numeric(work_df["pred"], errors="coerce")
    work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
    work_df = work_df.dropna(subset=["latitude", "longitude", "obs", "pred", "date"]).copy()
    if len(work_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    work_df["obs_year"] = work_df["date"].dt.year
    site_cols = ["latitude", "longitude"]
    site_summary_df = (
        work_df.groupby(site_cols, dropna=False)
        .agg(
            obs_mean=("obs", "mean"),
            pred_mean=("pred", "mean"),
            n_obs=("obs", "size"),
            n_years=("obs_year", "nunique"),
        )
        .reset_index()
    )
    site_summary_df["obs_per_year"] = site_summary_df["n_obs"] / site_summary_df["n_years"]
    work_df = work_df.merge(
        site_summary_df[site_cols + ["obs_mean", "pred_mean"]],
        on=site_cols,
        how="left",
    )
    work_df["obs_anom"] = work_df["obs"] - work_df["obs_mean"]
    work_df["pred_anom"] = work_df["pred"] - work_df["pred_mean"]
    anomaly_df = work_df[
        [
            "latitude",
            "longitude",
            "date",
            "fold",
            "obs",
            "pred",
            "obs_mean",
            "pred_mean",
            "obs_anom",
            "pred_anom",
        ]
    ].copy()
    return site_summary_df, anomaly_df


def plot_hexbin_from_arrays(
    obs,
    pred,
    save_path,
    xlabel,
    ylabel,
    fontsize,
    gridsize,
    metric_std=None,
):
    metrics = compute_basic_metrics(obs, pred)
    if metrics["n"] == 0:
        return None
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    data_min = float(min(np.min(obs), np.min(pred)))
    data_max = float(max(np.max(obs), np.max(pred)))
    stats_text = (
        f"R\u00b2 = {_format_metric_with_std(metrics['r2'], metric_std.get('r2') if metric_std else np.nan)}\n"
        f"RMSE = {_format_metric_with_std(metrics['rmse'], metric_std.get('rmse') if metric_std else np.nan)}\n"
        f"N = {metrics['n']}"
    )
    generic_hexbin(
        obs,
        pred,
        save_path,
        gridsize=gridsize,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(data_min, data_max),
        ylim=(data_min, data_max),
        cbar_label="Count",
        fontsize=fontsize,
        line_to_plot="one_to_one",
        stats_text=stats_text,
    )
    metrics["plot_path"] = save_path
    return metrics


def plot_scatter_from_arrays(
    obs,
    pred,
    save_path,
    xlabel,
    ylabel,
    fontsize,
    metric_std=None,
    color_array=None,
    cbar_label="Color Value",
):
    metrics = compute_basic_metrics(obs, pred)
    if metrics["n"] == 0:
        return None
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    if color_array is not None:
        color_array = np.asarray(color_array, dtype=float)
        mask = mask & np.isfinite(color_array)
    obs = obs[mask]
    pred = pred[mask]
    color_vals = color_array[mask] if color_array is not None else None
    data_min = float(min(np.min(obs), np.min(pred)))
    data_max = float(max(np.max(obs), np.max(pred)))
    generic_scatter(
        obs,
        pred,
        save_path,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(data_min, data_max),
        ylim=(data_min, data_max),
        r2=metrics["r2"],
        r2_std=(metric_std.get("r2") if metric_std else None),
        rmse=metrics["rmse"],
        rmse_std=(metric_std.get("rmse") if metric_std else None),
        n=metrics["n"],
        alpha=0.8,
        s=28,
        fontsize=fontsize,
        line_to_plot="one_to_one",
        marker_color="#440154" if color_vals is None else None,
        color_array=color_vals,
        cbar_label=cbar_label,
    )
    metrics["r2_std_across_members"] = metric_std.get("r2") if metric_std else np.nan
    metrics["rmse_std_across_members"] = metric_std.get("rmse") if metric_std else np.nan
    metrics["plot_path"] = save_path
    return metrics


def run_lfmc_space_time_analysis(eval_df, plot_dir, gridsize, fontsize, member_eval_dfs=None):
    lfmc_df = eval_df[eval_df["target"] == "lfmc"].reset_index(drop=True)
    if len(lfmc_df) == 0:
        print("No LFMC rows found; skipping space-versus-time analysis.")
        return {}
    site_summary_df, anomaly_df = build_lfmc_space_time_tables(lfmc_df)
    if len(site_summary_df) == 0 or len(anomaly_df) == 0:
        print("Insufficient LFMC rows for space-versus-time analysis.")
        return {}
    site_summary_path = os.path.join(plot_dir, "lfmc_site_means.csv")
    anomaly_path = os.path.join(plot_dir, "lfmc_site_anomalies.csv")
    site_summary_df.to_csv(site_summary_path, index=False)
    anomaly_df.to_csv(anomaly_path, index=False)
    print(f"Wrote LFMC site-mean table: {site_summary_path}")
    print(f"Wrote LFMC anomaly table: {anomaly_path}")
    space_plot_path = os.path.join(plot_dir, "lfmc_space_mean_pred_obs_hexbin.png")
    time_plot_path = os.path.join(plot_dir, "lfmc_time_anomaly_pred_obs_hexbin.png")
    space_metric_std = None
    time_metric_std = None
    if member_eval_dfs is not None and len(member_eval_dfs) > 0:
        member_space_metrics = []
        member_time_metrics = []
        for member_eval_df in member_eval_dfs:
            member_lfmc_df = member_eval_df[member_eval_df["target"] == "lfmc"].reset_index(drop=True)
            member_site_summary_df, member_anomaly_df = build_lfmc_space_time_tables(member_lfmc_df)
            if len(member_site_summary_df) > 0:
                member_space_metrics.append(
                    compute_basic_metrics(
                        member_site_summary_df["obs_mean"].values,
                        member_site_summary_df["pred_mean"].values,
                    )
                )
            if len(member_anomaly_df) > 0:
                member_time_metrics.append(
                    compute_basic_metrics(
                        member_anomaly_df["obs_anom"].values,
                        member_anomaly_df["pred_anom"].values,
                    )
                )
        if len(member_space_metrics) > 0:
            space_metric_std = {
                "rmse": _metric_std([m["rmse"] for m in member_space_metrics]),
                "r2": _metric_std([m["r2"] for m in member_space_metrics]),
            }
        if len(member_time_metrics) > 0:
            time_metric_std = {
                "rmse": _metric_std([m["rmse"] for m in member_time_metrics]),
                "r2": _metric_std([m["r2"] for m in member_time_metrics]),
            }
    space_metrics = plot_scatter_from_arrays(
        obs=site_summary_df["obs_mean"].values,
        pred=site_summary_df["pred_mean"].values,
        save_path=space_plot_path,
        xlabel="Observed LFMC site mean (%)",
        ylabel="Predicted LFMC site mean (%)",
        fontsize=fontsize,
        metric_std=space_metric_std,
        color_array=site_summary_df["obs_per_year"].values,
        cbar_label="Obs / year",
    )
    time_metrics = plot_hexbin_from_arrays(
        obs=anomaly_df["obs_anom"].values,
        pred=anomaly_df["pred_anom"].values,
        save_path=time_plot_path,
        xlabel="Observed LFMC anomaly (%)",
        ylabel="Predicted LFMC anomaly (%)",
        fontsize=fontsize,
        gridsize=gridsize,
        metric_std=time_metric_std,
    )
    out = {}
    if space_metrics is not None:
        out["lfmc_space"] = space_metrics
        print(f"Wrote LFMC space plot: {space_plot_path}")
    if time_metrics is not None:
        out["lfmc_time"] = time_metrics
        print(f"Wrote LFMC time plot: {time_plot_path}")
    return out


def build_lfmc_y2y_df(eval_df):
    lfmc_df = eval_df[eval_df["target"] == "lfmc"].copy()
    if len(lfmc_df) == 0:
        return pd.DataFrame()
    lfmc_df["date"] = pd.to_datetime(lfmc_df["date"], errors="coerce")
    if "site_id" in lfmc_df.columns:
        site_key_raw = lfmc_df["site_id"].copy()
        site_key = site_key_raw.astype(str)
        site_key = site_key.where(site_key_raw.notna(), "")
    else:
        site_key = ""
    if not isinstance(site_key, pd.Series) or (site_key == "").all():
        lat = pd.to_numeric(lfmc_df["latitude"], errors="coerce").round(5).astype(str)
        lon = pd.to_numeric(lfmc_df["longitude"], errors="coerce").round(5).astype(str)
        site_key = lat + "_" + lon
    lfmc_df["site_key"] = pd.Series(site_key, index=lfmc_df.index).astype(str)
    lfmc_df["year"] = lfmc_df["date"].dt.year
    lfmc_df["month"] = lfmc_df["date"].dt.month
    valid_mask = (
        lfmc_df["date"].notna()
        & lfmc_df["site_key"].notna()
        & np.isfinite(pd.to_numeric(lfmc_df["obs"], errors="coerce"))
        & np.isfinite(pd.to_numeric(lfmc_df["pred"], errors="coerce"))
    )
    lfmc_df = lfmc_df.loc[valid_mask].copy()
    lfmc_df["obs"] = pd.to_numeric(lfmc_df["obs"], errors="coerce")
    lfmc_df["pred"] = pd.to_numeric(lfmc_df["pred"], errors="coerce")
    return lfmc_df


def compute_monthly_y2y_metrics(lfmc_df, min_obs, min_years):
    if len(lfmc_df) == 0:
        return pd.DataFrame()
    grp_cols = ["site_key", "month"]
    group_stats = (
        lfmc_df.groupby(grp_cols, dropna=False)
        .agg(
            n_obs=("obs", "size"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )
    valid_groups = group_stats[
        (group_stats["n_obs"] >= min_obs) &
        (group_stats["n_years"] >= min_years)
    ].copy()
    if len(valid_groups) == 0:
        return pd.DataFrame()
    eval_df = lfmc_df.merge(valid_groups, on=grp_cols, how="inner")
    eval_df["obs_mean"] = eval_df.groupby(grp_cols)["obs"].transform("mean")
    eval_df["pred_mean"] = eval_df.groupby(grp_cols)["pred"].transform("mean")
    eval_df["obs_dev"] = eval_df["obs"] - eval_df["obs_mean"]
    eval_df["pred_dev"] = eval_df["pred"] - eval_df["pred_mean"]
    eval_df["sq_err"] = np.square(eval_df["pred_dev"] - eval_df["obs_dev"])
    eval_df["sq_obs_dev"] = np.square(eval_df["obs_dev"])

    overall_denom = float(eval_df["sq_obs_dev"].sum())
    if overall_denom <= 0:
        overall_pct_captured = np.nan
    else:
        overall_pct_captured = 100.0 * (1.0 - float(eval_df["sq_err"].sum()) / overall_denom)

    month_records = []
    for month in range(1, 13):
        month_df = eval_df[eval_df["month"] == month].copy()
        month_group_stats = valid_groups[valid_groups["month"] == month].copy()
        if len(month_df) == 0 or len(month_group_stats) == 0:
            month_records.append(
                {
                    "month": month,
                    "month_label": MONTH_LABELS[month - 1],
                    "pct_variability_captured_source_centered": np.nan,
                    "overall_pct_variability_captured_source_centered": overall_pct_captured,
                    "n_groups": 0,
                    "avg_points_per_group": np.nan,
                    "total_obs": 0,
                    "min_obs": min_obs,
                    "min_years": min_years,
                }
            )
            continue
        denom = float(month_df["sq_obs_dev"].sum())
        if denom <= 0:
            pct_captured = np.nan
        else:
            pct_captured = 100.0 * (1.0 - float(month_df["sq_err"].sum()) / denom)
        month_records.append(
            {
                "month": month,
                "month_label": MONTH_LABELS[month - 1],
                "pct_variability_captured_source_centered": pct_captured,
                "overall_pct_variability_captured_source_centered": overall_pct_captured,
                "n_groups": int(len(month_group_stats)),
                "avg_points_per_group": float(month_group_stats["n_obs"].mean()),
                "total_obs": int(len(month_df)),
                "min_obs": min_obs,
                "min_years": min_years,
            }
        )
    return pd.DataFrame.from_records(month_records)


def compute_site_y2y_metrics(lfmc_df, min_obs, min_years):
    if len(lfmc_df) == 0:
        return pd.DataFrame()
    grp_cols = ["site_key", "month"]
    group_stats = (
        lfmc_df.groupby(grp_cols, dropna=False)
        .agg(
            n_obs=("obs", "size"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )
    valid_groups = group_stats[
        (group_stats["n_obs"] >= min_obs) &
        (group_stats["n_years"] >= min_years)
    ].copy()
    if len(valid_groups) == 0:
        return pd.DataFrame()
    eval_df = lfmc_df.merge(valid_groups, on=grp_cols, how="inner")
    eval_df["obs_mean"] = eval_df.groupby(grp_cols)["obs"].transform("mean")
    eval_df["pred_mean"] = eval_df.groupby(grp_cols)["pred"].transform("mean")
    eval_df["obs_dev"] = eval_df["obs"] - eval_df["obs_mean"]
    eval_df["pred_dev"] = eval_df["pred"] - eval_df["pred_mean"]
    eval_df["sq_err"] = np.square(eval_df["pred_dev"] - eval_df["obs_dev"])
    eval_df["sq_obs_dev"] = np.square(eval_df["obs_dev"])

    site_records = []
    for site_key, site_df in eval_df.groupby("site_key", dropna=False):
        denom = float(site_df["sq_obs_dev"].sum())
        if denom <= 0:
            pct_captured = np.nan
        else:
            pct_captured = 100.0 * (1.0 - float(site_df["sq_err"].sum()) / denom)
        site_records.append(
            {
                "site_key": site_key,
                "latitude": float(pd.to_numeric(site_df["latitude"], errors="coerce").iloc[0]),
                "longitude": float(pd.to_numeric(site_df["longitude"], errors="coerce").iloc[0]),
                "pct_variability_captured_source_centered": pct_captured,
                "n_valid_months": int(site_df["month"].nunique()),
                "total_obs": int(len(site_df)),
                "min_obs": min_obs,
                "min_years": min_years,
            }
        )
    return pd.DataFrame.from_records(site_records)


def build_site_landcover_lookup(lfmc_df, save_path):
    required_cols = ["site_key", "latitude", "longitude", "year"]
    missing_cols = [col for col in required_cols if col not in lfmc_df.columns]
    if len(missing_cols) > 0:
        raise KeyError(
            f"LFMC dataframe is missing required columns for land-cover lookup: {missing_cols}"
        )
    requested_sites = (
        lfmc_df[["site_key", "latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    requested_sites["site_key"] = requested_sites["site_key"].astype(str)
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, dtype={"site_key": str})
        required_lookup_cols = [
            "site_key",
            "latitude",
            "longitude",
            "dominant_landcover",
            "dominant_landcover_frac",
        ]
        if all(col in existing_df.columns for col in required_lookup_cols):
            existing_df["site_key"] = existing_df["site_key"].astype(str)
            existing_df = existing_df[required_lookup_cols].drop_duplicates(subset=["site_key"]).reset_index(drop=True)
            missing_site_keys = sorted(
                set(requested_sites["site_key"]) - set(existing_df["site_key"])
            )
            if len(missing_site_keys) == 0:
                print(f"Using cached site land-cover lookup: {save_path}")
                return existing_df
            print(
                f"Cached site land-cover lookup missing {len(missing_site_keys)} sites; "
                "computing only missing sites."
            )
            requested_sites = requested_sites[
                requested_sites["site_key"].isin(missing_site_keys)
            ].reset_index(drop=True)
        else:
            print(
                f"Existing site land-cover lookup missing required columns; rebuilding: {save_path}"
            )
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    if len(requested_sites) == 0:
        return existing_df

    nlcd_ds = get_nlcd_frac_ds()
    landcover_vars = list(nlcd_ds.data_vars)
    site_years = (
        lfmc_df.groupby("site_key")["year"]
        .apply(lambda vals: sorted(pd.Series(vals).dropna().astype(int).unique().tolist()))
        .to_dict()
    )
    records = []
    iter_sites = tqdm(
        requested_sites.iterrows(),
        total=len(requested_sites),
        desc="Computing site land cover",
        unit="site",
    )
    for _, row in iter_sites:
        site_key = row["site_key"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        x, y = _WGS84_TO_5070.transform(lon, lat)
        site_cube = nlcd_ds.sel(x=x, y=y, method="nearest").load()
        years_here = site_years.get(site_key, [])
        if "year" in site_cube.dims or "year" in site_cube.coords:
            if len(years_here) > 0:
                year_slices = []
                for year in years_here:
                    year_slices.append(site_cube.sel(year=pd.Timestamp(year, 1, 1), method="nearest"))
                site_mean = xr.concat(year_slices, dim="year_sel").mean(dim="year_sel")
            else:
                site_mean = site_cube.mean(dim="year")
        else:
            site_mean = site_cube
        frac_vals = np.array([float(site_mean[var].values) for var in landcover_vars], dtype=float)
        if np.all(~np.isfinite(frac_vals)):
            dominant_landcover = "unknown"
            dominant_frac = np.nan
        else:
            dominant_idx = int(np.nanargmax(frac_vals))
            dominant_landcover = landcover_vars[dominant_idx]
            dominant_frac = float(frac_vals[dominant_idx])
        records.append(
            {
                "site_key": site_key,
                "latitude": lat,
                "longitude": lon,
                "dominant_landcover": dominant_landcover,
                "dominant_landcover_frac": dominant_frac,
            }
        )
        partial_lookup_df = pd.DataFrame.from_records(records)
        if len(existing_df) > 0:
            checkpoint_df = pd.concat([existing_df, partial_lookup_df], ignore_index=True)
            checkpoint_df = checkpoint_df.drop_duplicates(subset=["site_key"], keep="last").reset_index(drop=True)
        else:
            checkpoint_df = partial_lookup_df
        checkpoint_df.to_csv(save_path, index=False)
        iter_sites.set_postfix_str(f"checkpointed {len(records)}/{len(requested_sites)}")
    new_lookup_df = pd.DataFrame.from_records(records)
    if len(existing_df) > 0:
        lookup_df = pd.concat([existing_df, new_lookup_df], ignore_index=True)
        lookup_df = lookup_df.drop_duplicates(subset=["site_key"], keep="last").reset_index(drop=True)
    else:
        lookup_df = new_lookup_df
    lookup_df.to_csv(save_path, index=False)
    print(f"Wrote site land-cover lookup: {save_path}")
    return lookup_df


def compute_landcover_y2y_metrics(lfmc_df, min_obs, min_years, plot_dir):
    if len(lfmc_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), np.nan
    lfmc_df = lfmc_df.copy()
    lfmc_df["site_key"] = lfmc_df["site_key"].astype(str)
    grp_cols = ["site_key", "month"]
    group_stats = (
        lfmc_df.groupby(grp_cols, dropna=False)
        .agg(
            n_obs=("obs", "size"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )
    valid_groups = group_stats[
        (group_stats["n_obs"] >= min_obs) &
        (group_stats["n_years"] >= min_years)
    ].copy()
    if len(valid_groups) == 0:
        return pd.DataFrame(), pd.DataFrame(), np.nan
    eval_df = lfmc_df.merge(valid_groups, on=grp_cols, how="inner")
    eval_df["obs_mean"] = eval_df.groupby(grp_cols)["obs"].transform("mean")
    eval_df["pred_mean"] = eval_df.groupby(grp_cols)["pred"].transform("mean")
    eval_df["obs_dev"] = eval_df["obs"] - eval_df["obs_mean"]
    eval_df["pred_dev"] = eval_df["pred"] - eval_df["pred_mean"]
    eval_df["sq_err"] = np.square(eval_df["pred_dev"] - eval_df["obs_dev"])
    eval_df["sq_obs_dev"] = np.square(eval_df["obs_dev"])
    overall_denom = float(eval_df["sq_obs_dev"].sum())
    if overall_denom <= 0:
        overall_pct_captured = np.nan
    else:
        overall_pct_captured = 100.0 * (1.0 - float(eval_df["sq_err"].sum()) / overall_denom)

    site_lookup_path = os.path.join(plot_dir, "lfmc_site_landcover_lookup.csv")
    site_lookup_df = build_site_landcover_lookup(lfmc_df, site_lookup_path)
    site_lookup_df = site_lookup_df.copy()
    site_lookup_df["site_key"] = site_lookup_df["site_key"].astype(str)
    valid_groups = valid_groups.merge(
        site_lookup_df[["site_key", "dominant_landcover"]],
        on="site_key",
        how="left",
    )
    eval_df = eval_df.merge(
        site_lookup_df[["site_key", "dominant_landcover"]],
        on="site_key",
        how="left",
    )
    eval_df = eval_df[eval_df["dominant_landcover"].notna()].copy()
    valid_groups = valid_groups[valid_groups["dominant_landcover"].notna()].copy()
    records = []
    class_order = [
        "deciduous_forest",
        "evergreen_forest",
        "shrub",
        "grass",
        "mixed_forest",
    ]
    present_classes = [cls for cls in class_order if cls in valid_groups["dominant_landcover"].unique()]
    remaining_classes = sorted(
        [cls for cls in valid_groups["dominant_landcover"].unique() if cls not in present_classes]
    )
    ordered_classes = present_classes + remaining_classes
    for lc in ordered_classes:
        class_eval_df = eval_df[eval_df["dominant_landcover"] == lc].copy()
        class_groups_df = valid_groups[valid_groups["dominant_landcover"] == lc].copy()
        if len(class_eval_df) == 0 or len(class_groups_df) == 0:
            continue
        denom = float(class_eval_df["sq_obs_dev"].sum())
        if denom <= 0:
            pct_captured = np.nan
        else:
            pct_captured = 100.0 * (1.0 - float(class_eval_df["sq_err"].sum()) / denom)
        records.append(
            {
                "dominant_landcover": lc,
                "pct_variability_captured_source_centered": pct_captured,
                "overall_pct_variability_captured_source_centered": overall_pct_captured,
                "n_groups": int(len(class_groups_df)),
                "total_obs": int(len(class_eval_df)),
                "min_obs": min_obs,
                "min_years": min_years,
            }
        )
    return pd.DataFrame.from_records(records), site_lookup_df, overall_pct_captured


def plot_landcover_y2y_metrics(landcover_df, save_path, fontsize):
    annotations = []
    overall_series = landcover_df["overall_pct_variability_captured_source_centered"].dropna()
    if len(overall_series) > 0:
        stats_text = f"Overall = {float(overall_series.iloc[0]):.2f}%"
    else:
        stats_text = "Overall = nan"
    for _, row in landcover_df.iterrows():
        annotations.append(f"N={int(row['n_groups'])}")
    annotated_bar_plot(
        categories=landcover_df["dominant_landcover"].tolist(),
        values=landcover_df["pct_variability_captured_source_centered"].to_numpy(dtype=float),
        xlabel="Dominant land cover",
        ylabel="LFMC variability captured (%)",
        save_path=save_path,
        annotations=annotations,
        fontsize=fontsize,
        bar_color="#440154",
        stats_text=stats_text,
        errors=(
            landcover_df["pct_variability_captured_source_centered_std"].to_numpy(dtype=float)
            if "pct_variability_captured_source_centered_std" in landcover_df.columns
            else None
        ),
    )


def prepare_lfmc_landcover_eval_df(eval_df, plot_dir):
    lfmc_df = build_lfmc_y2y_df(eval_df)
    if len(lfmc_df) == 0:
        return pd.DataFrame()
    lfmc_df["site_key"] = lfmc_df["site_key"].astype(str)
    site_lookup_path = os.path.join(plot_dir, "lfmc_site_landcover_lookup.csv")
    site_lookup_df = build_site_landcover_lookup(lfmc_df, site_lookup_path)
    site_lookup_df = site_lookup_df.copy()
    site_lookup_df["site_key"] = site_lookup_df["site_key"].astype(str)
    lfmc_df = lfmc_df.merge(
        site_lookup_df[["site_key", "dominant_landcover", "dominant_landcover_frac"]],
        on="site_key",
        how="left",
    )
    lfmc_df = lfmc_df[lfmc_df["dominant_landcover"].notna()].copy()
    if len(lfmc_df) == 0:
        return pd.DataFrame()
    lfmc_df["site_obs_mean"] = lfmc_df.groupby("site_key")["obs"].transform("mean")
    lfmc_df["site_pred_mean"] = lfmc_df.groupby("site_key")["pred"].transform("mean")
    lfmc_df["site_obs_anom"] = lfmc_df["obs"] - lfmc_df["site_obs_mean"]
    lfmc_df["site_pred_anom"] = lfmc_df["pred"] - lfmc_df["site_pred_mean"]
    grp_cols = ["site_key", "month"]
    lfmc_df["seasonal_obs_mean"] = lfmc_df.groupby(grp_cols)["obs"].transform("mean")
    lfmc_df["seasonal_pred_mean"] = lfmc_df.groupby(grp_cols)["pred"].transform("mean")
    lfmc_df["seasonal_obs_anom"] = lfmc_df["obs"] - lfmc_df["seasonal_obs_mean"]
    lfmc_df["seasonal_pred_anom"] = lfmc_df["pred"] - lfmc_df["seasonal_pred_mean"]
    return lfmc_df


def _safe_r2(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) < 2:
        return np.nan
    return float(r2_score(obs, pred))


def compute_landcover_decomposition_metrics(lfmc_lc_df):
    if len(lfmc_lc_df) == 0:
        return pd.DataFrame()
    site_mean_df = (
        lfmc_lc_df.groupby(["site_key", "dominant_landcover"], dropna=False)
        .agg(
            obs=("site_obs_mean", "first"),
            pred=("site_pred_mean", "first"),
        )
        .reset_index()
    )
    class_order = [
        "deciduous_forest",
        "evergreen_forest",
        "shrub",
        "grass",
        "mixed_forest",
    ]
    present_classes = [cls for cls in class_order if cls in lfmc_lc_df["dominant_landcover"].unique()]
    remaining_classes = sorted(
        [cls for cls in lfmc_lc_df["dominant_landcover"].unique() if cls not in present_classes]
    )
    ordered_classes = present_classes + remaining_classes
    records = []
    for lc in ordered_classes:
        class_obs_df = lfmc_lc_df[lfmc_lc_df["dominant_landcover"] == lc].copy()
        class_site_mean_df = site_mean_df[site_mean_df["dominant_landcover"] == lc].copy()
        if len(class_obs_df) == 0:
            continue
        records.append(
            {
                "dominant_landcover": lc,
                "overall_r2": _safe_r2(class_obs_df["obs"].values, class_obs_df["pred"].values),
                "site_mean_r2": _safe_r2(class_site_mean_df["obs"].values, class_site_mean_df["pred"].values),
                "site_anom_r2": _safe_r2(class_obs_df["site_obs_anom"].values, class_obs_df["site_pred_anom"].values),
                "seasonal_cycle_anom_r2": _safe_r2(class_obs_df["seasonal_obs_anom"].values, class_obs_df["seasonal_pred_anom"].values),
                "n_points": int(len(class_obs_df)),
                "n_sites": int(class_obs_df["site_key"].nunique()),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_landcover_fraction_bin_metrics(lfmc_lc_df, min_obs=20, min_years=5, frac_edges=None):
    if len(lfmc_lc_df) == 0:
        return {}
    work_df = lfmc_lc_df.copy()
    work_df["dominant_landcover_frac"] = pd.to_numeric(
        work_df["dominant_landcover_frac"],
        errors="coerce",
    )
    work_df = work_df.dropna(subset=["dominant_landcover", "dominant_landcover_frac"]).copy()
    if len(work_df) == 0:
        return {}
    if frac_edges is None:
        frac_edges = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], dtype=float)
    frac_edges = np.asarray(frac_edges, dtype=float)
    frac_labels = [
        f"{frac_edges[i]:.2f}-{min(frac_edges[i + 1], 1.0):.2f}"
        for i in range(len(frac_edges) - 1)
    ]
    work_df["frac_bin"] = pd.cut(
        work_df["dominant_landcover_frac"],
        bins=frac_edges,
        labels=frac_labels,
        right=False,
        include_lowest=True,
    )
    work_df = work_df[work_df["frac_bin"].notna()].copy()
    if len(work_df) == 0:
        return {}
    out = {}
    for landcover, landcover_df in work_df.groupby("dominant_landcover", dropna=False):
        records = []
        for frac_label in frac_labels:
            bin_df = landcover_df[landcover_df["frac_bin"] == frac_label].copy()
            if len(bin_df) == 0:
                continue
            overall_r2 = _safe_r2(bin_df["obs"].values, bin_df["pred"].values)
            grp_cols = ["site_key", "month"]
            group_stats = (
                bin_df.groupby(grp_cols, dropna=False)
                .agg(
                    n_obs=("obs", "size"),
                    n_years=("year", "nunique"),
                )
                .reset_index()
            )
            valid_groups = group_stats[
                (group_stats["n_obs"] >= min_obs) &
                (group_stats["n_years"] >= min_years)
            ].copy()
            if len(valid_groups) == 0:
                fraction_captured = np.nan
                n_groups = 0
            else:
                eval_df = bin_df.merge(valid_groups, on=grp_cols, how="inner")
                eval_df["obs_mean"] = eval_df.groupby(grp_cols)["obs"].transform("mean")
                eval_df["pred_mean"] = eval_df.groupby(grp_cols)["pred"].transform("mean")
                eval_df["obs_dev"] = eval_df["obs"] - eval_df["obs_mean"]
                eval_df["pred_dev"] = eval_df["pred"] - eval_df["pred_mean"]
                eval_df["sq_err"] = np.square(eval_df["pred_dev"] - eval_df["obs_dev"])
                eval_df["sq_obs_dev"] = np.square(eval_df["obs_dev"])
                denom = float(eval_df["sq_obs_dev"].sum())
                if denom <= 0:
                    fraction_captured = np.nan
                else:
                    fraction_captured = 1.0 - float(eval_df["sq_err"].sum()) / denom
                n_groups = int(len(valid_groups))
            records.append(
                {
                    "dominant_landcover": landcover,
                    "frac_bin": frac_label,
                    "overall_r2": overall_r2,
                    "fraction_yearly_variability_captured": fraction_captured,
                    "n_points": int(len(bin_df)),
                    "n_groups": int(n_groups),
                    "min_obs": min_obs,
                    "min_years": min_years,
                }
            )
        if len(records) > 0:
            out[str(landcover)] = pd.DataFrame.from_records(records)
    return out


def run_lfmc_landcover_metric_analysis(eval_df, plot_dir, fontsize, member_eval_dfs=None):
    lfmc_lc_df = prepare_lfmc_landcover_eval_df(eval_df, plot_dir)
    if len(lfmc_lc_df) == 0:
        print("No LFMC land-cover rows found; skipping land-cover metric analysis.")
        return {}
    metric_df = compute_landcover_decomposition_metrics(lfmc_lc_df)
    if len(metric_df) == 0:
        print("No land-cover metrics available; skipping land-cover metric analysis.")
        return {}
    if member_eval_dfs is not None and len(member_eval_dfs) > 0:
        member_metric_frames = []
        for member_eval_df in member_eval_dfs:
            member_lfmc_lc_df = prepare_lfmc_landcover_eval_df(member_eval_df, plot_dir)
            if len(member_lfmc_lc_df) > 0:
                member_metric_frames.append(compute_landcover_decomposition_metrics(member_lfmc_lc_df))
        if len(member_metric_frames) > 0:
            for metric_name in ["overall_r2", "site_mean_r2", "site_anom_r2", "seasonal_cycle_anom_r2"]:
                std_lookup = {}
                for lc in metric_df["dominant_landcover"].tolist():
                    vals = []
                    for member_metric_df in member_metric_frames:
                        row = member_metric_df[member_metric_df["dominant_landcover"] == lc]
                        if len(row) == 0:
                            continue
                        vals.append(float(row.iloc[0][metric_name]))
                    std_lookup[lc] = _metric_std(vals)
                metric_df[f"{metric_name}_std"] = metric_df["dominant_landcover"].map(std_lookup)
    metric_csv_path = os.path.join(plot_dir, "lfmc_landcover_r2_metrics.csv")
    metric_df.to_csv(metric_csv_path, index=False)
    simple_plot_path = os.path.join(plot_dir, "lfmc_landcover_r2_simple.png")
    grouped_plot_path = os.path.join(plot_dir, "lfmc_landcover_r2_decomposition.png")
    landcover_y2y_df, _, overall_pct_captured = compute_landcover_y2y_metrics(
        lfmc_df=build_lfmc_y2y_df(eval_df),
        min_obs=20,
        min_years=5,
        plot_dir=plot_dir,
    )
    if len(landcover_y2y_df) > 0:
        metric_df = metric_df.merge(
            landcover_y2y_df[
                [
                    "dominant_landcover",
                    "pct_variability_captured_source_centered",
                    "n_groups",
                ]
            ],
            on="dominant_landcover",
            how="left",
        )
    else:
        metric_df["pct_variability_captured_source_centered"] = np.nan
        metric_df["n_groups"] = np.nan
    metric_df["fraction_yearly_variability_captured"] = (
        metric_df["pct_variability_captured_source_centered"] / 100.0
    )
    simple_annotations = [f"N={int(v)}" for v in metric_df["n_points"].values]
    overall_r2_global = _safe_r2(lfmc_lc_df["obs"].values, lfmc_lc_df["pred"].values)
    annotated_bar_plot(
        categories=metric_df["dominant_landcover"].tolist(),
        values=metric_df["overall_r2"].to_numpy(dtype=float),
        xlabel="Dominant land cover",
        ylabel="LFMC R²",
        save_path=simple_plot_path,
        annotations=simple_annotations,
        fontsize=fontsize,
        bar_color="#440154",
        stats_text=f"Overall = {overall_r2_global:.2f}" if np.isfinite(overall_r2_global) else "Overall = nan",
        errors=(
            metric_df["overall_r2_std"].to_numpy(dtype=float)
            if "overall_r2_std" in metric_df.columns
            else None
        ),
    )
    if member_eval_dfs is not None and len(member_eval_dfs) > 0 and len(landcover_y2y_df) > 0:
        std_lookup = {}
        member_landcover_frames = []
        for member_eval_df in member_eval_dfs:
            member_lfmc_y2y_df = build_lfmc_y2y_df(member_eval_df)
            member_landcover_df, _, _ = compute_landcover_y2y_metrics(
                lfmc_df=member_lfmc_y2y_df,
                min_obs=20,
                min_years=5,
                plot_dir=plot_dir,
            )
            if len(member_landcover_df) > 0:
                member_landcover_frames.append(member_landcover_df)
        for lc in metric_df["dominant_landcover"].tolist():
            vals = []
            for member_landcover_df in member_landcover_frames:
                row = member_landcover_df[member_landcover_df["dominant_landcover"] == lc]
                if len(row) == 0:
                    continue
                vals.append(float(row.iloc[0]["pct_variability_captured_source_centered"]))
            std_lookup[lc] = _metric_std(vals)
        metric_df["pct_variability_captured_source_centered_std"] = (
            metric_df["dominant_landcover"].map(std_lookup)
        )
    if "pct_variability_captured_source_centered_std" in metric_df.columns:
        metric_df["fraction_yearly_variability_captured_std"] = (
            metric_df["pct_variability_captured_source_centered_std"] / 100.0
        )
    grouped_values = metric_df[
        ["overall_r2", "fraction_yearly_variability_captured"]
    ].to_numpy(dtype=float)
    grouped_errors = None
    if all(col in metric_df.columns for col in [
        "overall_r2_std", "fraction_yearly_variability_captured_std"
    ]):
        grouped_errors = metric_df[
            ["overall_r2_std", "fraction_yearly_variability_captured_std"]
        ].to_numpy(dtype=float)
    sample_counts = np.column_stack(
        [
            metric_df["n_points"].to_numpy(dtype=float),
            metric_df["n_groups"].to_numpy(dtype=float),
        ]
    )
    bar_plot(
        categories=metric_df["dominant_landcover"].tolist(),
        values=grouped_values,
        xlabel="Dominant land cover",
        ylabel="Metric value",
        save_path=grouped_plot_path,
        label_with_n=True,
        sample_counts=sample_counts,
        subcategory_labels=[
            "overall R²",
            "fraction of yearly variability captured",
        ],
        errors=grouped_errors,
    )
    print(f"Wrote LFMC land-cover metric CSV: {metric_csv_path}")
    print(f"Wrote LFMC land-cover simple R2 plot: {simple_plot_path}")
    print(f"Wrote LFMC land-cover decomposition plot: {grouped_plot_path}")
    fraction_bin_outputs = {}
    fraction_bin_metric_dfs = compute_landcover_fraction_bin_metrics(
        lfmc_lc_df,
        min_obs=20,
        min_years=5,
    )
    member_fraction_bin_metric_dicts = []
    if member_eval_dfs is not None and len(member_eval_dfs) > 0:
        for member_eval_df in member_eval_dfs:
            member_lfmc_lc_df = prepare_lfmc_landcover_eval_df(member_eval_df, plot_dir)
            if len(member_lfmc_lc_df) == 0:
                continue
            member_fraction_bin_metric_dicts.append(
                compute_landcover_fraction_bin_metrics(
                    member_lfmc_lc_df,
                    min_obs=20,
                    min_years=5,
                )
            )
    for landcover, bin_df in fraction_bin_metric_dfs.items():
        plot_df = bin_df.copy()
        if len(member_fraction_bin_metric_dicts) > 0:
            overall_std_lookup = {}
            fraction_std_lookup = {}
            for frac_bin in plot_df["frac_bin"].tolist():
                overall_vals = []
                fraction_vals = []
                for member_metric_dict in member_fraction_bin_metric_dicts:
                    member_bin_df = member_metric_dict.get(landcover)
                    if member_bin_df is None or len(member_bin_df) == 0:
                        continue
                    row = member_bin_df[member_bin_df["frac_bin"] == frac_bin]
                    if len(row) == 0:
                        continue
                    overall_vals.append(float(row.iloc[0]["overall_r2"]))
                    fraction_vals.append(float(row.iloc[0]["fraction_yearly_variability_captured"]))
                overall_std_lookup[frac_bin] = _metric_std(overall_vals)
                fraction_std_lookup[frac_bin] = _metric_std(fraction_vals)
            plot_df["overall_r2_std"] = plot_df["frac_bin"].map(overall_std_lookup)
            plot_df["fraction_yearly_variability_captured_std"] = plot_df["frac_bin"].map(fraction_std_lookup)
        grouped_values = plot_df[
            ["overall_r2", "fraction_yearly_variability_captured"]
        ].to_numpy(dtype=float)
        grouped_errors = None
        if all(
            col in plot_df.columns
            for col in ["overall_r2_std", "fraction_yearly_variability_captured_std"]
        ):
            grouped_errors = plot_df[
                ["overall_r2_std", "fraction_yearly_variability_captured_std"]
            ].to_numpy(dtype=float)
        sample_counts = np.column_stack(
            [
                plot_df["n_points"].to_numpy(dtype=float),
                plot_df["n_groups"].to_numpy(dtype=float),
            ]
        )
        safe_landcover = str(landcover).replace("/", "_").replace(" ", "_")
        fracbin_csv_path = os.path.join(
            plot_dir,
            f"lfmc_landcover_fracbin_{safe_landcover}.csv",
        )
        fracbin_plot_path = os.path.join(
            plot_dir,
            f"lfmc_landcover_fracbin_{safe_landcover}.png",
        )
        plot_df.to_csv(fracbin_csv_path, index=False)
        bar_plot(
            categories=plot_df["frac_bin"].tolist(),
            values=grouped_values,
            xlabel="Dominant land-cover fraction bin",
            ylabel="Metric value",
            save_path=fracbin_plot_path,
            label_with_n=True,
            sample_counts=sample_counts,
            subcategory_labels=[
                "overall R²",
                "fraction of yearly variability captured",
            ],
            errors=grouped_errors,
        )
        fraction_bin_outputs[landcover] = {
            "csv_path": fracbin_csv_path,
            "plot_path": fracbin_plot_path,
            "n_bins": int(len(plot_df)),
        }
        print(f"Wrote LFMC land-cover fraction-bin plot: {fracbin_plot_path}")
    return {
        "lfmc_landcover_r2": {
            "csv_path": metric_csv_path,
            "simple_plot_path": simple_plot_path,
            "grouped_plot_path": grouped_plot_path,
            "n_classes": int(len(metric_df)),
        },
        "lfmc_landcover_fraction_bins": fraction_bin_outputs,
    }


def plot_monthly_y2y_metrics(month_df, save_path, fontsize):
    annotations = []
    overall_series = month_df["overall_pct_variability_captured_source_centered"].dropna()
    if len(overall_series) > 0:
        overall_value = float(overall_series.iloc[0])
        stats_text = f"Overall = {overall_value:.2f}%"
    else:
        stats_text = "Overall = nan"
    for _, row in month_df.iterrows():
        if row["n_groups"] == 0:
            annotations.append("")
            continue
        annotations.append(f"N={int(row['n_groups'])}")
    annotated_bar_plot(
        categories=month_df["month_label"].tolist(),
        values=month_df["pct_variability_captured_source_centered"].to_numpy(dtype=float),
        xlabel="Month",
        ylabel="LFMC variability captured (%)",
        save_path=save_path,
        annotations=annotations,
        fontsize=fontsize,
        bar_color="#440154",
        stats_text=stats_text,
        errors=(
            month_df["pct_variability_captured_source_centered_std"].to_numpy(dtype=float)
            if "pct_variability_captured_source_centered_std" in month_df.columns
            else None
        ),
    )


def run_lfmc_monthly_y2y_analysis(eval_df, plot_dir, fontsize, member_eval_dfs=None):
    lfmc_y2y_df = build_lfmc_y2y_df(eval_df)
    if len(lfmc_y2y_df) == 0:
        print("No LFMC rows found; skipping monthly year-to-year analysis.")
        return {}
    threshold_sets = [
        (10, 2),
        (20, 3),
        (20, 5),
    ]
    out = {}
    for min_obs, min_years in threshold_sets:
        month_df = compute_monthly_y2y_metrics(
            lfmc_df=lfmc_y2y_df,
            min_obs=min_obs,
            min_years=min_years,
        )
        if len(month_df) == 0:
            print(
                f"No valid LFMC site-month groups for monthly y2y analysis with "
                f"min_obs={min_obs}, min_years={min_years}."
            )
            continue
        if member_eval_dfs is not None and len(member_eval_dfs) > 0:
            std_lookup = {}
            member_month_frames = []
            for member_eval_df in member_eval_dfs:
                member_lfmc_y2y_df = build_lfmc_y2y_df(member_eval_df)
                member_month_df = compute_monthly_y2y_metrics(
                    lfmc_df=member_lfmc_y2y_df,
                    min_obs=min_obs,
                    min_years=min_years,
                )
                if len(member_month_df) > 0:
                    member_month_frames.append(member_month_df)
            for month in month_df["month"].tolist():
                vals = []
                for member_month_df in member_month_frames:
                    row = member_month_df[member_month_df["month"] == month]
                    if len(row) == 0:
                        continue
                    vals.append(float(row.iloc[0]["pct_variability_captured_source_centered"]))
                std_lookup[month] = _metric_std(vals)
            month_df["pct_variability_captured_source_centered_std"] = month_df["month"].map(std_lookup)
        stem = f"lfmc_y2y_monthly_nobs{min_obs}_nyears{min_years}"
        csv_path = os.path.join(plot_dir, f"{stem}.csv")
        plot_path = os.path.join(plot_dir, f"{stem}.png")
        month_df.to_csv(csv_path, index=False)
        plot_monthly_y2y_metrics(
            month_df=month_df,
            save_path=plot_path,
            fontsize=fontsize,
        )
        valid_months = month_df["n_groups"] > 0
        overall_series = month_df["overall_pct_variability_captured_source_centered"].dropna()
        if valid_months.any():
            pooled_value = float(overall_series.iloc[0]) if len(overall_series) > 0 else np.nan
            total_groups = int(month_df.loc[valid_months, "n_groups"].sum())
            avg_points = float(month_df.loc[valid_months, "avg_points_per_group"].mean())
        else:
            pooled_value = np.nan
            total_groups = 0
            avg_points = np.nan
        pooled_std = _metric_std(
            month_df.loc[valid_months, "pct_variability_captured_source_centered"].values
        ) if valid_months.any() else np.nan
        key = f"lfmc_y2y_monthly_nobs{min_obs}_nyears{min_years}"
        out[key] = {
            "pct_variability_captured_source_centered_global": pooled_value
            if np.isfinite(pooled_value) else np.nan,
            "pct_variability_captured_source_centered_global_std": pooled_std
            if np.isfinite(pooled_std) else np.nan,
            "n_groups_total": total_groups,
            "avg_points_per_group_mean": avg_points,
            "csv_path": csv_path,
            "plot_path": plot_path,
        }
        print(
            f"Wrote monthly LFMC y2y outputs for min_obs={min_obs}, "
            f"min_years={min_years}: {plot_path}"
        )
        if min_obs == 20 and min_years == 5:
            landcover_df, site_lookup_df, _ = compute_landcover_y2y_metrics(
                lfmc_df=lfmc_y2y_df,
                min_obs=min_obs,
                min_years=min_years,
                plot_dir=plot_dir,
            )
            if len(landcover_df) > 0:
                if member_eval_dfs is not None and len(member_eval_dfs) > 0:
                    std_lookup = {}
                    member_landcover_frames = []
                    for member_eval_df in member_eval_dfs:
                        member_lfmc_y2y_df = build_lfmc_y2y_df(member_eval_df)
                        member_landcover_df, _, _ = compute_landcover_y2y_metrics(
                            lfmc_df=member_lfmc_y2y_df,
                            min_obs=min_obs,
                            min_years=min_years,
                            plot_dir=plot_dir,
                        )
                        if len(member_landcover_df) > 0:
                            member_landcover_frames.append(member_landcover_df)
                    for lc in landcover_df["dominant_landcover"].tolist():
                        vals = []
                        for member_landcover_df in member_landcover_frames:
                            row = member_landcover_df[member_landcover_df["dominant_landcover"] == lc]
                            if len(row) == 0:
                                continue
                            vals.append(float(row.iloc[0]["pct_variability_captured_source_centered"]))
                        std_lookup[lc] = _metric_std(vals)
                    landcover_df["pct_variability_captured_source_centered_std"] = (
                        landcover_df["dominant_landcover"].map(std_lookup)
                    )
                landcover_csv_path = os.path.join(
                    plot_dir,
                    "lfmc_y2y_landcover_nobs20_nyears5.csv",
                )
                landcover_plot_path = os.path.join(
                    plot_dir,
                    "lfmc_y2y_landcover_nobs20_nyears5.png",
                )
                landcover_df.to_csv(landcover_csv_path, index=False)
                plot_landcover_y2y_metrics(
                    landcover_df=landcover_df,
                    save_path=landcover_plot_path,
                    fontsize=fontsize,
                )
                out[f"{key}_landcover"] = {
                    "csv_path": landcover_csv_path,
                    "plot_path": landcover_plot_path,
                    "n_classes": int(len(landcover_df)),
                    "site_lookup_path": os.path.join(plot_dir, "lfmc_site_landcover_lookup.csv"),
                }
                print(
                    "Wrote stringent LFMC land-cover variability plot: "
                    f"{landcover_plot_path}"
                )
            site_df = compute_site_y2y_metrics(
                lfmc_df=lfmc_y2y_df,
                min_obs=min_obs,
                min_years=min_years,
            )
            if len(site_df) > 0:
                site_csv_path = os.path.join(
                    plot_dir,
                    "lfmc_y2y_site_map_nobs20_nyears5.csv",
                )
                site_map_path = os.path.join(
                    plot_dir,
                    "lfmc_y2y_site_map_nobs20_nyears5.png",
                )
                site_df.to_csv(site_csv_path, index=False)
                valid_site_vals = site_df["pct_variability_captured_source_centered"].to_numpy(dtype=float)
                valid_site_vals = valid_site_vals[np.isfinite(valid_site_vals)]
                if len(valid_site_vals) > 0:
                    cbar_upper = float(np.max(valid_site_vals))
                    if cbar_upper <= 0:
                        cbar_upper = 1.0
                else:
                    cbar_upper = 1.0
                map_points(
                    site_df["longitude"].values,
                    site_df["latitude"].values,
                    site_df["n_valid_months"].values,
                    site_map_path,
                    colors=site_df["pct_variability_captured_source_centered"].values,
                    cmap="viridis",
                    colorbar_label="LFMC monthly anomaly explained (%)",
                    cbar_lim=(0.0, cbar_upper),
                    s_min=12,
                    s_max=42,
                    stats_text=(
                        f"Overall = {pooled_value:.2f}%"
                        if np.isfinite(pooled_value) else "Overall = nan"
                    ),
                )
                out[f"{key}_site_map"] = {
                    "csv_path": site_csv_path,
                    "plot_path": site_map_path,
                    "n_sites": int(len(site_df)),
                    "size_count_definition": "n_valid_months",
                }
                print(
                    "Wrote stringent LFMC site monthly-anomaly map: "
                    f"{site_map_path} (point size = total contributing months)"
                )
    return out


def write_summary(outputs, plot_dir):
    metrics_json_path = os.path.join(plot_dir, "overall_metrics.json")
    metrics_csv_path = os.path.join(plot_dir, "overall_metrics.csv")
    with open(metrics_json_path, "w") as file_obj:
        json.dump(outputs, file_obj, indent=2)
    metrics_df = pd.DataFrame.from_records(
        [
            {
                "target": key,
                **value,
            }
            for key, value in outputs.items()
        ]
    )
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Wrote metric summary JSON: {metrics_json_path}")
    print(f"Wrote metric summary CSV: {metrics_csv_path}")


def _maybe_attach_target_member_frames(member_eval_dfs, target_name):
    if member_eval_dfs is None:
        return None
    out = []
    for member_eval_df in member_eval_dfs:
        target_df = member_eval_df[member_eval_df["target"] == target_name].reset_index(drop=True)
        if len(target_df) > 0:
            out.append(target_df)
    return out


def _build_ensemble_eval_df(member_eval_dfs):
    parts = []
    lfmc_member_dfs = _maybe_attach_target_member_frames(member_eval_dfs, "lfmc")
    if lfmc_member_dfs is not None and len(lfmc_member_dfs) > 0:
        parts.append(aggregate_member_eval_frames(lfmc_member_dfs))
    for target_name in ["vv", "vh"]:
        target_member_dfs = _maybe_attach_target_member_frames(member_eval_dfs, target_name)
        if target_member_dfs is None or len(target_member_dfs) == 0:
            continue
        pooled_target_df = pd.concat(target_member_dfs, ignore_index=True)
        parts.append(pooled_target_df)
    if len(parts) == 0:
        raise ValueError("No ensemble evaluation rows available after target-specific aggregation")
    eval_df = pd.concat(parts, ignore_index=True)
    if "date" in eval_df.columns:
        eval_df["date"] = pd.to_datetime(eval_df["date"], errors="coerce")
    return eval_df


def load_eval_context(model_dir=None, ensemble_outputs_root=None, outputs_root=None, model_df_index=None, sort_metric=DEFAULT_SORT_METRIC, ascending=True):
    if ensemble_outputs_root is not None:
        member_dirs = select_ensemble_member_dirs(ensemble_outputs_root)
        print(f"Using ensemble root {ensemble_outputs_root} with {len(member_dirs)} members")
        member_eval_dfs = [load_fold_predictions(member_dir) for member_dir in member_dirs]
        eval_df = _build_ensemble_eval_df(member_eval_dfs)
        return {
            "mode": "ensemble",
            "model_dir": ensemble_outputs_root,
            "member_dirs": member_dirs,
            "eval_df": eval_df,
            "member_eval_dfs": member_eval_dfs,
        }
    if model_dir is None:
        model_dir = select_model_dir(
            outputs_root=outputs_root,
            model_df_index=model_df_index,
            sort_metric=sort_metric,
            ascending=ascending,
        )
    if not is_complete_model_dir(model_dir):
        raise FileNotFoundError(
            f"Model dir is missing required test artifacts: {model_dir}"
        )
    eval_df = load_fold_predictions(model_dir)
    return {
        "mode": "single",
        "model_dir": model_dir,
        "member_dirs": [],
        "eval_df": eval_df,
        "member_eval_dfs": None,
    }


def main():
    args = get_args()
    context = load_eval_context(
        model_dir=args.model_dir,
        ensemble_outputs_root=args.ensemble_outputs_root,
        outputs_root=args.outputs_root,
        model_df_index=args.model_df_index,
        sort_metric=args.sort_metric,
        ascending=args.ascending,
    )
    model_dir = context["model_dir"]
    plot_dir = resolve_plot_dir(
        model_dir,
        args.plot_dir,
        outputs_root=(args.ensemble_outputs_root if args.ensemble_outputs_root is not None else args.outputs_root),
    )
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Analyzing model dir: {model_dir}")
    print(f"Writing outputs to: {plot_dir}")
    eval_df = context["eval_df"]
    member_eval_dfs = context["member_eval_dfs"]
    eval_table_path = os.path.join(plot_dir, "eval_observations.csv")
    eval_df.to_csv(eval_table_path, index=False)
    print(f"Wrote consolidated evaluation table: {eval_table_path}")
    metrics_out = {}
    for target_name in ["lfmc", "vv", "vh"]:
        target_df = eval_df[eval_df["target"] == target_name].reset_index(drop=True)
        if len(target_df) == 0:
            print(f"No rows found for target {target_name}; skipping.")
            continue
        aggregate_mode = "prediction"
        if context["mode"] == "ensemble" and target_name in {"vv", "vh"}:
            aggregate_mode = "member_metric"
        target_metrics = plot_target_hexbin(
            target_df=target_df,
            target_name=target_name,
            plot_dir=plot_dir,
            gridsize=args.hexbin_gridsize,
            fontsize=args.fontsize,
            member_target_dfs=_maybe_attach_target_member_frames(member_eval_dfs, target_name),
            aggregate_mode=aggregate_mode,
        )
        if target_metrics is not None:
            metrics_out[target_name] = target_metrics
    if any(target in set(eval_df["target"].astype(str)) for target in ["vv", "vh"]):
        metrics_out.update(
            plot_multitask_lfmc_vs_sar_hexbin(
                eval_df=eval_df,
                plot_dir=plot_dir,
                gridsize=args.hexbin_gridsize,
                fontsize=args.fontsize,
            )
        )
    metrics_out.update(
        run_lfmc_space_time_analysis(
            eval_df=eval_df,
            plot_dir=plot_dir,
            gridsize=args.hexbin_gridsize,
            fontsize=args.fontsize,
            member_eval_dfs=member_eval_dfs,
        )
    )
    metrics_out.update(
        run_lfmc_monthly_y2y_analysis(
            eval_df=eval_df,
            plot_dir=plot_dir,
            fontsize=args.fontsize,
            member_eval_dfs=member_eval_dfs,
        )
    )
    metrics_out.update(
        run_lfmc_landcover_metric_analysis(
            eval_df=eval_df,
            plot_dir=plot_dir,
            fontsize=args.fontsize,
            member_eval_dfs=member_eval_dfs,
        )
    )
    write_summary(metrics_out, plot_dir)
    print("Finished deep evaluation figures.")


if __name__ == "__main__":
    main()
