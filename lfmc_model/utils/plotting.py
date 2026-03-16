import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import sys
import numpy as np
import matplotlib.dates as mdates
import math
import textwrap
from typing import Sequence
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from typing import Sequence, Optional
import textwrap
import pandas as pd


def _style_legend(legend):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_alpha(0.9)
    frame.set_facecolor("white")
    frame.set_edgecolor("0.4")


def _stats_text_box(ax, stats_text, fontsize=None, loc=(0.03, 0.97)):
    if stats_text is None:
        return
    ax.text(
        loc[0],
        loc[1],
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "0.4",
        },
        fontsize=fontsize,
    )

def kde_plot(
    data, data_names, save_name, title=None,
    xlabel=None, ylabel=None, ylimit=None
):
    """
    Create a Kernel Density Estimate (KDE) plot for a specified column in the data.

    Parameters:
    - data: List of np arrays containing the data.
    - data_names: the names of the different data sets.
    - xlabel: Label for the x-axis (optional).
    - ylabel: Label for the y-axis (optional).

    Returns:
    - ax: The axes object of the plot.
    """
    plt.figure(figsize=(10, 6))
    for d,dat in enumerate(data):
        sns.kdeplot(dat, label=data_names[d])
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ylimit:
        plt.ylim(ylimit)
    plt.legend()
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def plot_multiple_timeseries(
    dates,
    vals,
    labels,
    linestyles,
    markers,
    save_path
):
    """
    Plot multiple time series on a single axis and save to disk.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for d, v, lab, ls, mk in zip(
        dates, vals, labels, linestyles, markers
    ):
        ax.plot(
            d,
            v,
            label=lab,
            linestyle=ls,
            marker=mk
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    # ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_lfmc_with_vv_vh(
    lfmc_dates,
    lfmc_vals,
    lfmc_labels,
    lfmc_linestyles,
    lfmc_markers,
    lfmc_colors,
    save_path,
    lfmc_lower_vals=None,
    lfmc_upper_vals=None,
    vv_obs_dates=None,
    vv_obs=None,
    vv_train_pred_dates=None,
    vv_train_pred=None,
    vv_pred_dates=None,
    vv_pred=None,
    vv_pred_std=None,
    vh_obs_dates=None,
    vh_obs=None,
    vh_train_pred_dates=None,
    vh_train_pred=None,
    vh_pred_dates=None,
    vh_pred=None,
    vh_pred_std=None,
    vv_dates=None,
    vh_dates=None,
    annotation_text=None,
    title_text=None,
):
    fig, ax_l = plt.subplots(figsize=(11, 4.5))
    ax_r = ax_l.twinx()
    # Ensure VV/VH (right axis) is drawn behind LFMC (left axis)
    ax_l.set_zorder(2)
    ax_r.set_zorder(1)
    ax_l.patch.set_visible(False)
    if lfmc_colors is None:
        lfmc_colors = [None] * len(lfmc_dates)
    if lfmc_lower_vals is None:
        lfmc_lower_vals = [None] * len(lfmc_dates)
    if lfmc_upper_vals is None:
        lfmc_upper_vals = [None] * len(lfmc_dates)
    for d, v, lab, ls, mk, col, lower, upper in zip(
        lfmc_dates,
        lfmc_vals,
        lfmc_labels,
        lfmc_linestyles,
        lfmc_markers,
        lfmc_colors,
        lfmc_lower_vals,
        lfmc_upper_vals,
    ):
        if (
            lower is not None and upper is not None and
            len(lower) == len(v) and len(upper) == len(v) and len(v) > 0
        ):
            fill_color = col if col is not None else "0.6"
            ax_l.fill_between(
                d,
                lower,
                upper,
                color=fill_color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        plot_kwargs = {
            "label": lab,
            "linestyle": ls,
            "marker": mk,
        }
        if lab == "lfmc_true":
            plot_kwargs["markersize"] = 10
            plot_kwargs["alpha"] = 0.9
        elif str(lab).endswith("_infer"):
            plot_kwargs["linewidth"] = 2.8
            plot_kwargs["alpha"] = 0.95
        elif "train_pred" in str(lab):
            plot_kwargs["markersize"] = 3
            plot_kwargs["alpha"] = 0.85
        if col is not None:
            plot_kwargs["color"] = col
        plot_kwargs["zorder"] = 4
        ax_l.plot(
            d,
            v,
            **plot_kwargs
        )
    ax_l.set_xlabel("Date")
    ax_l.set_ylabel("LFMC (%)")
    h_left, l_left = ax_l.get_legend_handles_labels()
    has_right = False
    # Backward compatibility: older callers used vv_dates/vh_dates for both obs and pred.
    if vv_obs_dates is None:
        vv_obs_dates = vv_dates
    if vv_pred_dates is None:
        vv_pred_dates = vv_dates
    if vh_obs_dates is None:
        vh_obs_dates = vh_dates
    if vh_pred_dates is None:
        vh_pred_dates = vh_dates
    if vv_train_pred_dates is None:
        vv_train_pred_dates = vv_dates
    if vh_train_pred_dates is None:
        vh_train_pred_dates = vh_dates
    if vv_obs_dates is not None and vv_obs is not None and len(vv_obs) > 0:
        ax_r.plot(
            vv_obs_dates,
            vv_obs,
            linestyle="",
            marker="s",
            markersize=4,
            alpha=0.6,
            color="red",
            label="vv_obs",
            zorder=1,
        )
        has_right = True
    if vv_train_pred_dates is not None and vv_train_pred is not None and len(vv_train_pred) > 0:
        ax_r.plot(
            vv_train_pred_dates,
            vv_train_pred,
            linestyle="",
            marker=".",
            markersize=4,
            alpha=0.6,
            color="red",
            label="vv_train_pred",
            zorder=1,
        )
        has_right = True
    if vv_pred_dates is not None and vv_pred is not None and len(vv_pred) > 0:
        if vv_pred_std is not None and len(vv_pred_std) == len(vv_pred):
            ax_r.fill_between(
                vv_pred_dates,
                np.asarray(vv_pred) - np.asarray(vv_pred_std),
                np.asarray(vv_pred) + np.asarray(vv_pred_std),
                color="red",
                alpha=0.12,
                linewidth=0,
                zorder=0,
            )
        ax_r.plot(
            vv_pred_dates,
            vv_pred,
            linestyle="-",
            linewidth=1.0,
            alpha=0.6,
            color="red",
            label="vv_infer",
            zorder=1,
        )
        has_right = True
    if vh_obs_dates is not None and vh_obs is not None and len(vh_obs) > 0:
        ax_r.plot(
            vh_obs_dates,
            vh_obs,
            linestyle="",
            marker="D",
            markersize=4,
            alpha=0.6,
            color="orange",
            label="vh_obs",
            zorder=1,
        )
        has_right = True
    if vh_train_pred_dates is not None and vh_train_pred is not None and len(vh_train_pred) > 0:
        ax_r.plot(
            vh_train_pred_dates,
            vh_train_pred,
            linestyle="",
            marker=".",
            markersize=4,
            alpha=0.6,
            color="orange",
            label="vh_train_pred",
            zorder=1,
        )
        has_right = True
    if vh_pred_dates is not None and vh_pred is not None and len(vh_pred) > 0:
        if vh_pred_std is not None and len(vh_pred_std) == len(vh_pred):
            ax_r.fill_between(
                vh_pred_dates,
                np.asarray(vh_pred) - np.asarray(vh_pred_std),
                np.asarray(vh_pred) + np.asarray(vh_pred_std),
                color="orange",
                alpha=0.12,
                linewidth=0,
                zorder=0,
            )
        ax_r.plot(
            vh_pred_dates,
            vh_pred,
            linestyle="-",
            linewidth=1.0,
            alpha=0.6,
            color="orange",
            label="vh_infer",
            zorder=1,
        )
        has_right = True
    if has_right:
        ax_r.set_ylabel("VV / VH (dB)")
    else:
        ax_r.set_ylabel("")
        ax_r.set_yticks([])
        ax_r.tick_params(right=False, labelright=False)
        ax_r.spines["right"].set_visible(False)
    h_right, l_right = ax_r.get_legend_handles_labels()
    all_handles = h_left + h_right
    all_labels = l_left + l_right
    if len(all_handles) > 0:
        ax_l.legend(all_handles, all_labels, loc="best")
    if title_text is not None and str(title_text).strip() != "":
        ax_l.set_title(str(title_text))
    if annotation_text is not None and str(annotation_text).strip() != "":
        ax_l.text(
            0.02,
            0.98,
            annotation_text,
            transform=ax_l.transAxes,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.88,
                "edgecolor": "0.4",
            },
        )
    fig.autofmt_xdate()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_multiple_timeseries_from_df(
    df,
    date_col,
    x_label,
    y_label,
    save_name,
    col_markers=None
):
    # get the columns that are not date
    columns = df.columns[df.columns != date_col]
    dates = df[date_col].values
    fig,ax = plt.subplots(figsize=(10, 6))
    for c,col in enumerate(columns):
        ax.plot(dates, df[col].values, label=col, marker=col_markers[c] if col_markers else None)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=45)
    plt.savefig(save_name,bbox_inches='tight')
    plt.close()

def pred_obs_scatter(
    preds,
    obs,
    plot_path,
    mae=None,
    rmse=None,
    r2=None,
    n=None
):
    plt.figure(figsize=(6, 6))
    plt.scatter(obs, preds, alpha=0.5)
    max_val = max(np.max(obs), np.max(preds))
    min_val = min(np.min(obs), np.min(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Observed')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend()
    if mae is not None:
        plt.text(0.05, 0.95, f'MAE: {mae:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
    if rmse is not None:
        plt.text(0.05, 0.90, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
    if r2 is not None:
        plt.text(0.05, 0.85, f'R²: {r2:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
    if n is not None:
        plt.text(0.05, 0.80, f'N: {n}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

def bar_plot(
    categories: Sequence[str],
    values: Sequence[float],
    xlabel: str,
    ylabel: str,
    save_path: str,
    label_with_n: bool = False,
    sample_counts: Optional[Sequence[float]] = None,
    subcategory_labels: Optional[Sequence[str]] = None,
    subcategory_colors: Optional[Sequence[str]] = None,
    errors: Optional[Sequence[float]] = None,
):
    """
    Flexible bar plot: handles 1D and 2D values.

    1D case:
        categories: length N
        values:     length N

    2D case (grouped bar plot):
        categories: length N (groups)
        values:     shape (N, M)
        subcategory_labels: length M
    """
    categories = [str(c) for c in categories]
    values_arr = np.asarray(values, dtype=float)
    errors_arr = None if errors is None else np.asarray(errors, dtype=float)

    # ---- Shape handling ----
    if values_arr.ndim == 1:
        n_cat = len(categories)
        assert values_arr.shape[0] == n_cat
        is_grouped = False
    elif values_arr.ndim == 2:
        n_cat, n_sub = values_arr.shape
        assert len(categories) == n_cat
        is_grouped = True
        if subcategory_labels is None:
            subcategory_labels = [f"sub{i}" for i in range(n_sub)]
        else:
            subcategory_labels = [str(s) for s in subcategory_labels]
            assert len(subcategory_labels) == n_sub
        if subcategory_colors is not None:
            subcategory_colors = [str(c) for c in subcategory_colors]
            assert len(subcategory_colors) == n_sub
        if errors_arr is not None:
            assert errors_arr.shape == values_arr.shape
    else:
        raise ValueError("values must be 1D or 2D for bar_plot.")
    if (not is_grouped) and errors_arr is not None:
        assert errors_arr.shape[0] == values_arr.shape[0]

    # ---- sample_counts handling ----
    if label_with_n and sample_counts is not None:
        counts_arr = np.asarray(sample_counts, dtype=float)
        if not is_grouped:
            assert counts_arr.shape[0] == values_arr.shape[0]
        else:
            assert counts_arr.shape == values_arr.shape
    else:
        counts_arr = None

    # ---- helpers ----
    def wrap_labels(labels, width):
        return [
            "\n".join(
                textwrap.wrap(lbl, width=width, break_long_words=False)
            ) if len(lbl) > width else lbl
            for lbl in labels
        ]

    def make_bar_label(v, cnt=None):
        if not np.isfinite(v):
            return ""
        if label_with_n and cnt is not None:
            return f"{v:.2f}\n(n={int(cnt)})"
        return f"{v:.2f}"

    def _compute_axis_limits(vals, errs=None, orientation="vertical"):
        vals_arr = np.asarray(vals, dtype=float)
        finite_vals = vals_arr[np.isfinite(vals_arr)]
        if finite_vals.size > 0:
            lower = float(finite_vals.min())
            upper = float(finite_vals.max())
        else:
            lower = 0.0
            upper = 1.0
        if errs is not None:
            errs_arr = np.asarray(errs, dtype=float)
            lower_vals = vals_arr - errs_arr
            upper_vals = vals_arr + errs_arr
            finite_lower = lower_vals[np.isfinite(lower_vals)]
            finite_upper = upper_vals[np.isfinite(upper_vals)]
            if finite_lower.size > 0:
                lower = min(lower, float(finite_lower.min()))
            if finite_upper.size > 0:
                upper = max(upper, float(finite_upper.max()))
        span = upper - lower
        if orientation == "horizontal":
            pad = max(0.08 * max(abs(upper), abs(lower), 1.0), 2.0 if span < 20 else 0.5)
        else:
            pad = max(0.08 * max(abs(upper), abs(lower), 1.0), 0.5 if span < 10 else 0.0)
        if span <= 0:
            pad = max(pad, 1.0)
        return lower - pad, upper + pad

    # ==================================================
    # 2D GROUPED BAR PLOT (always vertical)
    # ==================================================
    if is_grouped:
        max_label_len = max((len(c) for c in categories), default=0)
        wrap_width = max(12, min(28, 2 + int(0.45 * max_label_len)))
        xlabels = wrap_labels(categories, wrap_width)
        max_lines = max((lbl.count("\n") + 1 for lbl in xlabels), default=1)

        width = max(8.5, 0.8 * n_cat + 0.7 * n_sub)
        height = max(5.0, 4.0 + 0.40 * max_lines)

        fig, ax = plt.subplots(figsize=(width, height),
                               constrained_layout=False)

        x = np.arange(n_cat)
        bar_width = 0.8 / n_sub

        y_min, y_max = _compute_axis_limits(values_arr, errors_arr, orientation="vertical")
        ax.set_ylim(y_min, y_max)

        bars_by_sub = []
        for j in range(n_sub):
            offset = (j - (n_sub - 1) / 2.0) * bar_width
            sub_vals = values_arr[:, j]
            bars = ax.bar(
                x + offset,
                sub_vals,
                bar_width,
                label=subcategory_labels[j],
                color=(subcategory_colors[j] if subcategory_colors is not None else None),
                yerr=(errors_arr[:, j] if errors_arr is not None else None),
                capsize=(3 if errors_arr is not None else 0),
                ecolor="0.25",
            )
            bars_by_sub.append(bars)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, xlabels)
        ax.tick_params(axis="x", labelsize=9, pad=6)

        # ---- annotate each subgroup with bar_label (correct use) ----
        for j in range(n_sub):
            sub_vals = values_arr[:, j]
            if counts_arr is not None:
                sub_counts = counts_arr[:, j]
            else:
                sub_counts = [None] * n_cat

            labels = [
                make_bar_label(v, cnt)
                for v, cnt in zip(sub_vals, sub_counts)
            ]
            ax.bar_label(
                bars_by_sub[j],
                labels=labels,
                padding=2,
                fontsize=8,
            )

        ax.legend(title="", fontsize=8, frameon=False)

        bottom_margin = min(0.40, 0.16 + 0.05 * (max_lines - 1))
        fig.subplots_adjust(left=0.10, right=0.98,
                            top=0.98, bottom=bottom_margin)

        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    # ==================================================
    # 1D CASE (existing behavior)
    # ==================================================
    values_1d = values_arr.tolist()
    n = len(categories)
    max_label_len = max((len(c) for c in categories), default=0)
    use_horizontal = (max_label_len > 25) or (n > 8)

    # Horizontal
    if use_horizontal:
        wrap_width = max(20, min(42, 2 + int(0.7 * max_label_len)))
        ylabels = wrap_labels(categories, wrap_width)
        max_lines = max((lbl.count("\n") + 1 for lbl in ylabels), default=1)

        height = max(4.5, 0.45 * n + 0.25 * max_lines + 1.0)
        width = max(10.0, min(24.0, 9.0 + 0.18 * max_label_len))
        fig, ax = plt.subplots(figsize=(width, height),
                               constrained_layout=False)

        y_pos = np.arange(n)
        bars = ax.barh(
            y_pos,
            values_1d,
            color="skyblue",
            xerr=errors_arr if errors_arr is not None else None,
            capsize=(3 if errors_arr is not None else 0),
            ecolor="0.25",
        )

        ax.set_yticks(y_pos, ylabels)
        ax.tick_params(axis="y", labelsize=9, pad=6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        x_min, x_max = _compute_axis_limits(values_1d, errors_arr, orientation="horizontal")
        ax.set_xlim(x_min, x_max)

        if counts_arr is not None:
            counts_1d = counts_arr
        else:
            counts_1d = [None] * n

        labels = [
            make_bar_label(v, cnt)
            for v, cnt in zip(values_1d, counts_1d)
        ]

        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)

        ax.margins(y=0.02)
        left_margin = min(0.55, 0.18 + 0.03 * max_lines + 0.002 * max_label_len)
        fig.subplots_adjust(left=left_margin, right=0.98,
                            top=0.98, bottom=0.10)

        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    # Vertical
    else:
        wrap_width = max(12, min(28, 2 + int(0.45 * max_label_len)))
        xlabels = wrap_labels(categories, wrap_width)
        max_lines = max((lbl.count("\n") + 1 for lbl in xlabels), default=1)

        width = max(8.5, 0.75 * n + 2.5)
        height = max(5.0, 4.0 + 0.40 * max_lines)

        fig, ax = plt.subplots(figsize=(width, height),
                               constrained_layout=False)
        x = np.arange(n)
        bars = ax.bar(
            x,
            values_1d,
            color="skyblue",
            yerr=errors_arr if errors_arr is not None else None,
            capsize=(3 if errors_arr is not None else 0),
            ecolor="0.25",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, xlabels)
        ax.tick_params(axis="x", labelsize=9, pad=6)

        y_min, y_max = _compute_axis_limits(values_1d, errors_arr, orientation="vertical")
        ax.set_ylim(y_min, y_max)

        if counts_arr is not None:
            counts_1d = counts_arr
        else:
            counts_1d = [None] * n

        labels = [
            make_bar_label(v, cnt)
            for v, cnt in zip(values_1d, counts_1d)
        ]

        ax.bar_label(bars, labels=labels, padding=2, fontsize=8)

        bottom_margin = min(0.40, 0.16 + 0.05 * (max_lines - 1))
        fig.subplots_adjust(left=0.10, right=0.98,
                            top=0.98, bottom=bottom_margin)

        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

#def bar_plot(
#    categories: Sequence[str],
#    values: Sequence[float],
#    xlabel: str,
#    ylabel: str,
#    save_path: str,
#    label_with_n: bool = False,
#    sample_counts: Optional[Sequence[int]] = None,
#):
#    """
#    If label_with_n is True and sample_counts is provided,
#    each bar will be labeled like:
#
#        "<value>\n(n=XYZ)"
#    """
#    categories = [str(c) for c in categories]
#    values = list(values)
#
#    if label_with_n and sample_counts is not None:
#        sample_counts = [int(n) for n in sample_counts]
#
#    n = len(categories)
#    max_label_len = max((len(c) for c in categories), default=0)
#    use_horizontal = (max_label_len > 25) or (n > 8)
#
#    def wrap_labels(labels, width):
#        return [
#            "\n".join(
#                textwrap.wrap(
#                    lbl,
#                    width=width,
#                    break_long_words=False
#                )
#            ) if len(lbl) > width else lbl
#            for lbl in labels
#        ]
#
#    # --------------------------------------------------
#    # Horizontal bars
#    # --------------------------------------------------
#    if use_horizontal:
#        wrap_width = max(
#            20,
#            min(42, 2 + int(0.7 * max_label_len))
#        )
#        ylabels = wrap_labels(categories, wrap_width)
#        max_lines = max(
#            (lbl.count("\n") + 1 for lbl in ylabels),
#            default=1
#        )
#
#        height = max(
#            4.5,
#            0.45 * n + 0.25 * max_lines + 1.0
#        )
#        width = max(
#            10.0,
#            min(24.0, 9.0 + 0.18 * max_label_len)
#        )
#        fig, ax = plt.subplots(
#            figsize=(width, height),
#            constrained_layout=False
#        )
#
#        y_pos = np.arange(n)
#        bars = ax.barh(y_pos, values, color="skyblue")
#
#        ax.set_yticks(y_pos, ylabels)
#        ax.tick_params(axis="y", labelsize=9, pad=6)
#
#        ax.set_xlabel(xlabel)
#        ax.set_ylabel(ylabel)
#
#        vmax = max(values) if values else 1.0
#        ax.set_xlim(0, vmax * 1.12 + (2 if vmax < 20 else 0))
#
#        # ---- labels on bars ----
#        if label_with_n and sample_counts is not None:
#            labels = [
#                f"{v:g}\n(n={cnt})"
#                for v, cnt in zip(values, sample_counts)
#            ]
#        else:
#            labels = [f"{v:g}" for v in values]
#
#        ax.bar_label(
#            bars,
#            labels=labels,
#            padding=3,
#            fontsize=8
#        )
#
#        ax.margins(y=0.02)
#        left_margin = min(
#            0.55,
#            0.18 + 0.03 * max_lines + 0.002 * max_label_len
#        )
#        fig.subplots_adjust(
#            left=left_margin,
#            right=0.98,
#            top=0.98,
#            bottom=0.10
#        )
#
#        plt.savefig(save_path, bbox_inches="tight", dpi=300)
#        plt.close(fig)
#
#    # --------------------------------------------------
#    # Vertical bars
#    # --------------------------------------------------
#    else:
#        wrap_width = max(
#            12,
#            min(28, 2 + int(0.45 * max_label_len))
#        )
#        xlabels = wrap_labels(categories, wrap_width)
#        max_lines = max(
#            (lbl.count("\n") + 1 for lbl in xlabels),
#            default=1
#        )
#
#        width = max(8.5, 0.75 * n + 2.5)
#        height = max(5.0, 4.0 + 0.40 * max_lines)
#
#        fig, ax = plt.subplots(
#            figsize=(width, height),
#            constrained_layout=False
#        )
#        x = np.arange(n)
#        bars = ax.bar(x, values, color="skyblue")
#
#        ax.set_xlabel(xlabel)
#        ax.set_ylabel(ylabel)
#        ax.set_xticks(x, xlabels)
#        ax.tick_params(axis="x", labelsize=9, pad=6)
#
#        ymax = max(values) if values else 1.0
#        ax.set_ylim(0, ymax * 1.10 + (0.5 if ymax < 10 else 0))
#
#        # ---- labels on bars ----
#        if label_with_n and sample_counts is not None:
#            labels = [
#                f"{v:g}\n(n={cnt})"
#                for v, cnt in zip(values, sample_counts)
#            ]
#        else:
#            labels = [f"{v:g}" for v in values]
#
#        ax.bar_label(
#            bars,
#            labels=labels,
#            padding=2,
#            fontsize=8
#        )
#
#        bottom_margin = min(
#            0.40,
#            0.16 + 0.05 * (max_lines - 1)
#        )
#        fig.subplots_adjust(
#            left=0.10,
#            right=0.98,
#            top=0.98,
#            bottom=bottom_margin
#        )
#
#        plt.savefig(save_path, bbox_inches="tight", dpi=300)
#        plt.close(fig)


from matplotlib.colors import TwoSlopeNorm, Normalize

def map_points(
    lons,
    lats,
    counts_per_point,
    save_path,
    *,
    s_min=20,              # smallest marker area (points^2)
    s_max=300,             # largest  marker area (points^2)
    clip_quantiles=(0.00, 0.98),
    colors=None,           # numeric → colorbar
    cmap="PiYG",           # default diverging colormap
    colorbar_label="Value",
    cbar_lim=None,         # None, scalar, or (vmin, vmax)
    stats_text=None,
    show_size_legend=True,
):
    """
    Plot lon/lat points with sizes scaled by counts_per_point,
    and an optional colorbar.

    Colorbar behavior
    -----------------
    - colors is None or a single color:
        no colorbar
    - colors is numeric array and:
        * cbar_lim is None:
            symmetric around 0, +-max(abs(colors)),
            using TwoSlopeNorm (diverging)
        * cbar_lim is scalar (e.g. 0.5):
            symmetric around 0, [-0.5, 0.5],
            using TwoSlopeNorm (diverging)
        * cbar_lim is (vmin, vmax):
            - if symmetric around 0 (vmin ~ -vmax):
                use TwoSlopeNorm with center 0
            - else:
                use plain Normalize(vmin, vmax)
                (no enforced symmetry)
    """

    # -----------------------
    # Basic arrays
    # -----------------------
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    counts = np.asarray(counts_per_point, dtype=float)

    # -----------------------
    # Handle colors
    # -----------------------
    use_colorbar = False
    point_colors = "#2b83ba"   # default
    norm = None
    vmin_cbar = None
    vmax_cbar = None

    if colors is not None:
        color_arr = np.asarray(colors)

        # Scalar color (e.g. "k", "#ff0000")
        if color_arr.ndim == 0:
            point_colors = colors
            use_colorbar = False

        else:
            if color_arr.shape[0] != lons.shape[0]:
                raise ValueError(
                    "colors array must have same length "
                    "as lons/lats"
                )

            # Numeric colors → potentially use colorbar
            if color_arr.dtype.kind in ("i", "u", "f"):
                use_colorbar = True
                point_colors = color_arr

                # Determine vmin/vmax for colorbar
                if cbar_lim is None:
                    # Auto symmetric around 0
                    max_abs = np.nanmax(np.abs(color_arr))
                    if (not np.isfinite(max_abs)) or max_abs == 0:
                        vmin_cbar, vmax_cbar = -1.0, 1.0
                    else:
                        vmin_cbar, vmax_cbar = -max_abs, max_abs

                    # symmetric → diverging norm
                    norm = TwoSlopeNorm(
                        vmin=vmin_cbar,
                        vcenter=0.0,
                        vmax=vmax_cbar,
                    )

                elif np.isscalar(cbar_lim):
                    # Scalar → symmetric range ±|cbar_lim|
                    span = abs(float(cbar_lim))
                    vmin_cbar, vmax_cbar = -span, span
                    norm = TwoSlopeNorm(
                        vmin=vmin_cbar,
                        vcenter=0.0,
                        vmax=vmax_cbar,
                    )

                else:
                    # Tuple/list: (vmin, vmax)
                    if len(cbar_lim) != 2:
                        raise ValueError(
                            "cbar_lim must be scalar or (vmin, vmax)"
                        )
                    vmin_cbar, vmax_cbar = map(float, cbar_lim)
                    if vmin_cbar > vmax_cbar:
                        vmin_cbar, vmax_cbar = vmax_cbar, vmin_cbar

                    # Check if symmetric around 0
                    symmetric = (
                        vmin_cbar < 0.0 < vmax_cbar
                        and np.isclose(
                            -vmin_cbar,
                            vmax_cbar,
                            rtol=1e-6,
                            atol=1e-12,
                        )
                    )

                    if symmetric:
                        # Symmetric → diverging norm
                        norm = TwoSlopeNorm(
                            vmin=vmin_cbar,
                            vcenter=0.0,
                            vmax=vmax_cbar,
                        )
                    else:
                        # Not symmetric → plain Normalize
                        norm = Normalize(
                            vmin=vmin_cbar,
                            vmax=vmax_cbar,
                        )
            else:
                # Non-numeric array of colors (e.g. list of hex)
                point_colors = color_arr
                use_colorbar = False

    # -----------------------
    # Size scaling (counts)
    # -----------------------
    q_lo, q_hi = np.clip(clip_quantiles, 0.0, 1.0)
    vmin_counts = np.nanquantile(counts, q_lo)
    vmax_counts = np.nanquantile(counts, q_hi)

    if not np.isfinite(vmin_counts):
        vmin_counts = np.nanmin(counts)
    if not np.isfinite(vmax_counts):
        vmax_counts = np.nanmax(counts)

    if vmax_counts <= vmin_counts:
        sizes = np.full_like(
            counts,
            (s_min + s_max) / 2.0,
            dtype=float,
        )
    else:
        cclip = np.clip(counts, vmin_counts, vmax_counts)
        sizes = (
            s_min
            + (cclip - vmin_counts)
            / (vmax_counts - vmin_counts)
            * (s_max - s_min)
        )

    # -----------------------
    # Figure & map setup
    # -----------------------
    proj_data = ccrs.PlateCarree()
    proj_map = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=proj_map)

    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="#f5f5f5",
    )
    ax.add_feature(
        cfeature.OCEAN.with_scale("50m"),
        facecolor="#e8f2ff",
    )
    ax.coastlines(resolution="50m", linewidth=0.7)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.5,
    )
    try:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "cultural",
                "admin_1_states_provinces_lines",
                "50m",
            ),
            linewidth=0.4,
            edgecolor="0.6",
            facecolor="none",
        )
    except Exception:
        pass

    # Hard-coded extent; swap if needed
    ax.set_extent([-125, -100, 25, 50])

    # -----------------------
    # Scatter plot
    # -----------------------
    sc = ax.scatter(
        lons,
        lats,
        s=sizes,
        c=point_colors,
        cmap=cmap if use_colorbar else None,
        norm=norm,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.2,
        transform=proj_data,
    )

    # -----------------------
    # Colorbar
    # -----------------------
    if use_colorbar:
        cb = plt.colorbar(
            sc,
            ax=ax,
            shrink=0.7,
            pad=0.02,
        )
        cb.set_label(colorbar_label, fontsize=11)

    # -----------------------
    # Size legend
    # -----------------------
    if show_size_legend:
        if vmax_counts <= vmin_counts:
            legend_vals = [float(np.nanmean(counts))]
        else:
            levels = np.linspace(vmin_counts, vmax_counts, 4)
            rng = vmax_counts - vmin_counts
            step = 10 ** np.floor(np.log10(rng / 3.0))
            legend_vals = sorted(
                {
                    float(
                        np.maximum(
                            0.0,
                            np.round(v / step) * step,
                        )
                    )
                    for v in levels
                }
            )

        def size_for(v):
            if vmax_counts <= vmin_counts:
                return (s_min + s_max) / 2.0
            vclip = np.clip(v, vmin_counts, vmax_counts)
            return (
                s_min
                + (vclip - vmin_counts)
                / (vmax_counts - vmin_counts)
                * (s_max - s_min)
            )

        proxies = [
            plt.scatter(
                [],
                [],
                s=size_for(v),
                color="#2b83ba",
                alpha=0.7,
                edgecolor="k",
                linewidth=0.2,
            )
            for v in legend_vals
        ]
        labels = [f"{v:g}" for v in legend_vals]

        leg = ax.legend(
            proxies,
            labels,
            title="Count",
            scatterpoints=1,
            frameon=True,
            fontsize=9,
            title_fontsize=10,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.01),
        )
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("0.4")

    _stats_text_box(ax, stats_text, fontsize=12, loc=(0.03, 0.97))

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotated_bar_plot(
    categories,
    values,
    xlabel,
    ylabel,
    save_path,
    annotations=None,
    fontsize=None,
    bar_color="#440154",
    stats_text=None,
    errors=None,
    xtick_rotation=0,
):
    categories = [str(c) for c in categories]
    values = np.asarray(values, dtype=float)
    errors = None if errors is None else np.asarray(errors, dtype=float)
    if len(categories) != len(values):
        raise ValueError("categories and values must have the same length")
    if errors is not None and len(errors) != len(values):
        raise ValueError("errors and values must have the same length")
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        x,
        values,
        color=bar_color,
        width=0.75,
        yerr=errors,
        capsize=(3 if errors is not None else 0),
        ecolor="0.25",
    )
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(
        categories,
        fontsize=fontsize,
        rotation=xtick_rotation,
        ha=("right" if xtick_rotation else "center"),
    )
    if fontsize is not None:
        ax.tick_params(axis="y", labelsize=fontsize)

    finite_vals = values[np.isfinite(values)]
    yrange = 1.0
    if finite_vals.size > 0:
        ymin = float(min(0.0, finite_vals.min()))
        ymax = float(max(0.0, finite_vals.max()))
        if errors is not None:
            finite_errs = errors[np.isfinite(errors)]
            ymax += float(finite_errs.max()) if finite_errs.size > 0 else 0.0
        yrange = ymax - ymin
        if yrange <= 0:
            yrange = 1.0
        pad = 0.14 * yrange
        ax.set_ylim(ymin - 0.05 * yrange, ymax + pad)

    if annotations is not None:
        if len(annotations) != len(values):
            raise ValueError("annotations and values must have the same length")
        err_vals = np.zeros(len(values), dtype=float)
        if errors is not None:
            err_vals = np.asarray(errors, dtype=float)
            err_vals = np.where(np.isfinite(err_vals), err_vals, 0.0)
        for idx, (bar, value, annotation) in enumerate(zip(bars, values, annotations)):
            if not np.isfinite(value) or annotation in [None, ""]:
                continue
            err_here = float(err_vals[idx]) if idx < len(err_vals) else 0.0
            offset = 0.02 * yrange
            if value >= 0:
                y_text = value + max(err_here, 0.0) + offset
                va = "bottom"
            else:
                y_text = value - max(err_here, 0.0) - offset
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                annotation,
                ha="center",
                va=va,
                fontsize=max(8, fontsize - 2) if fontsize is not None else 9,
            )

    _stats_text_box(ax, stats_text, fontsize=fontsize, loc=(0.03, 0.97))

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def generic_hexbin(
    x,
    y,
    plot_path,
    gridsize=50,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    cmap="viridis",
    cbar_label="Counts",
    cbarlim=None,
    fontsize=None,
    line_to_plot=None,
    corrclip=[-np.inf, np.inf],
    title=None,
    stats_text=None,
    stats_loc=(0.03, 0.97),
):
    plt.figure(figsize=(6, 6))
    if cbarlim:
        vmin = cbarlim[0]
        vmax = cbarlim[1]
    else:
        vmin = None
        vmax = None
    hb = plt.hexbin(
        x,
        y,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=1,
        #bins='log',
        vmin=vmin,
        vmax=vmax
    )
    cbar = plt.colorbar(hb)
    cbar.set_label(cbar_label, fontsize=fontsize)
    if fontsize is not None:
        cbar.ax.tick_params(labelsize=fontsize)
    plt.xlabel(xlabel if xlabel else "X", fontsize=fontsize)
    plt.ylabel(ylabel if ylabel else "Y", fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)

    if fontsize is not None:
        plt.tick_params(axis='both', labelsize=fontsize)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if line_to_plot == 'correlation':

        mask_corr = (
            (x >= corrclip[0]) & (x <= corrclip[1]) &
            (y >= corrclip[0]) & (y <= corrclip[1])
        )
        x_corr = x[mask_corr]
        y_corr = y[mask_corr]
        corr_coef = np.corrcoef(x_corr, y_corr)[0, 1]
        m, b = np.polyfit(x_corr, y_corr, 1)

        plt.plot(
            x_corr,
            m * x_corr + b,
            color="orange",
            label=f"Best Fit Line (r={corr_coef:.2f})",
        )
        legend = plt.legend(fontsize=fontsize)
        _style_legend(legend)

    if line_to_plot == 'one_to_one':
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        min_min = min(x_min, y_min)
        max_max = max(x_max, y_max)
        plt.plot(
            [min_min, max_max],
            [min_min, max_max],
            color="orange",
            label="1:1 line",
        )
        legend = plt.legend(fontsize=fontsize)
        _style_legend(legend)

    _stats_text_box(plt.gca(), stats_text, fontsize=fontsize, loc=stats_loc)

    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()

def generic_scatter(
    x,
    y,
    plot_path,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    mae=None,
    rmse=None,
    rmse_std=None,
    r2=None,
    r2_std=None,
    n=None,
    corrclip=None,
    color_array=None,
    cmap="viridis",
    cbar_label="Color Value",
    alpha=0.5,
    s=20,
    cbar_range=None,
    cbar_scale="linear",
    fontsize=None,
    line_to_plot=None,
    marker_color=None,
    stats_text=None,
):
    # mask nans for x, y (and color_array if provided)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if color_array is not None:
        mask = mask & ~np.isnan(color_array)

    x = x[mask]
    y = y[mask]
    c = color_array[mask] if color_array is not None else None

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    norm = None
    if color_array is not None and cbar_scale == "log":
        positive_c = c[c > 0]
        if len(positive_c) > 0:
            if cbar_range is not None:
                vmin = max(float(cbar_range[0]), float(np.min(positive_c)))
                vmax = float(cbar_range[1])
            else:
                vmin = float(np.min(positive_c))
                vmax = float(np.max(positive_c))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin > 0:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # scatter
    sc = ax.scatter(
        x, y,
        alpha=alpha,
        c=c if color_array is not None else marker_color,
        cmap=cmap if color_array is not None else None,
        s=s,
        norm=norm,
        vmin=(cbar_range[0] if cbar_range and norm is None else None),
        vmax=(cbar_range[1] if cbar_range and norm is None else None),
    )

    # colorbar
    if color_array is not None:
        cbar = plt.colorbar(sc)
        cbar.set_label(cbar_label, fontsize=fontsize)
        if cbar_scale == "log" and norm is not None:
            cbar.locator = mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
            cbar.formatter = mticker.FuncFormatter(
                lambda value, pos: np.format_float_positional(
                    value,
                    trim="-",
                )
            )
            cbar.update_ticks()

        if fontsize is not None:
            cbar.ax.tick_params(labelsize=fontsize)

    # axis labels
    ax.set_xlabel(xlabel if xlabel else "X", fontsize=fontsize)
    ax.set_ylabel(ylabel if ylabel else "Y", fontsize=fontsize)

    # axis tick font sizes
    if fontsize is not None:
        ax.tick_params(axis='both', labelsize=fontsize)

    # x/y limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # correlation region
    if line_to_plot == 'correlation':
        if corrclip:
            mask_corr = (
                (x >= corrclip[0]) & (x <= corrclip[1]) &
                (y >= corrclip[0]) & (y <= corrclip[1])
            )
            x_corr = x[mask_corr]
            y_corr = y[mask_corr]
        else:
            x_corr = x
            y_corr = y

        corr_coef = np.corrcoef(x_corr, y_corr)[0, 1]
        m, b = np.polyfit(x_corr, y_corr, 1)

        ax.plot(
            x_corr,
            m * x_corr + b,
            color="orange",
            label=f"Best Fit Line (r={corr_coef:.2f})",
        )
        legend = ax.legend(fontsize=fontsize)
        _style_legend(legend)

    elif line_to_plot == 'one_to_one':
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        min_min = min(x_min, y_min)
        max_max = max(x_max, y_max)
        ax.plot(
            [min_min, max_max],
            [min_min, max_max],
            color="orange",
            label="1:1 line",
        )
        legend = ax.legend(fontsize=fontsize)
        _style_legend(legend)

    if stats_text is None:
        stats_lines = []
        if r2 is not None:
            if r2_std is not None and np.isfinite(r2_std):
                stats_lines.append(f"R² = {r2:.2f} +/- {r2_std:.2f}")
            else:
                stats_lines.append(f"R² = {r2:.2f}")
        if rmse is not None:
            if rmse_std is not None and np.isfinite(rmse_std):
                stats_lines.append(f"RMSE = {rmse:.2f} +/- {rmse_std:.2f}")
            else:
                stats_lines.append(f"RMSE = {rmse:.2f}")
        if n is not None:
            stats_lines.append(f"N = {n}")
        if mae is not None and len(stats_lines) == 0:
            stats_lines.append(f"MAE = {mae:.2f}")
        stats_text = "\n".join(stats_lines) if len(stats_lines) > 0 else None
    _stats_text_box(ax, stats_text, fontsize=fontsize, loc=(0.03, 0.97))

    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()


#def generic_scatter(
#    x,
#    y,
#    plot_path,
#    xlabel=None,
#    ylabel=None,
#    xlim=None,
#    ylim=None,
#    mae=None,
#    rmse=None,
#    r2=None,
#    n=None,
#    corrclip=None,
#    color_array=None,
#    cmap="viridis",
#    cbar_label="Color Value",
#    alpha=0.5,
#    s=20,
#    cbar_range=None,
#    fontsize=None,
#
#):
#    # mask nans for x, y (and color_array if provided)
#    mask = ~np.isnan(x) & ~np.isnan(y)
#    if color_array is not None:
#        mask = mask & ~np.isnan(color_array)
#
#    x = x[mask]
#    y = y[mask]
#    c = color_array[mask] if color_array is not None else None
#
#    plt.figure(figsize=(6, 6))
#
#    # scatter with color if provided
#    sc = plt.scatter(
#        x, y, 
#        alpha=alpha,
#        c=c,
#        cmap=cmap if color_array is not None else None,
#        s=s,
#        vmin=cbar_range[0] if cbar_range else None,
#        vmax=cbar_range[1] if cbar_range else None,
#    )
#
#    # add colorbar only if using color_array
#    if color_array is not None:
#        plt.colorbar(
#            sc,
#            label=cbar_label,
#        )
#
#    plt.xlabel(xlabel if xlabel else 'X')
#    plt.ylabel(ylabel if ylabel else 'Y')
#
#    if xlim:
#        plt.xlim(xlim)
#    if ylim:
#        plt.ylim(ylim)
#
#    # compute correlation and best-fit line
#    if corrclip:
#        mask_corr = (
#            (x >= corrclip[0]) & (x <= corrclip[1]) &
#            (y >= corrclip[0]) & (y <= corrclip[1])
#        )
#        x_corr = x[mask_corr]
#        y_corr = y[mask_corr]
#    else:
#        x_corr = x
#        y_corr = y
#
#    corr_coef = np.corrcoef(x_corr, y_corr)[0, 1]
#    m, b = np.polyfit(x_corr, y_corr, 1)
#
#    plt.plot(
#        x_corr,
#        m * x_corr + b,
#        color='orange',
#        label=f'Best Fit Line (r={corr_coef:.2f})'
#    )
#    plt.legend()
#
#    # text stats
#    ax = plt.gca()
#    if mae is not None:
#        ax.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=ax.transAxes, va='top')
#    if rmse is not None:
#        ax.text(0.05, 0.90, f'RMSE: {rmse:.3f}', transform=ax.transAxes, va='top')
#    if r2 is not None:
#        ax.text(0.05, 0.85, f'R²: {r2:.3f}', transform=ax.transAxes, va='top')
#    if n is not None:
#        ax.text(0.05, 0.80, f'N: {n}', transform=ax.transAxes, va='top')
#
#    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
#    plt.close()

#def generic_scatter(
#    x,
#    y,
#    plot_path,
#    xlabel=None,
#    ylabel=None,
#    xlim=None,
#    ylim=None,
#    mae=None,
#    rmse=None,
#    r2=None,
#    n=None,
#    corrclip=None
#):
#    # get rid of nans
#    mask = ~np.isnan(x) & ~np.isnan(y)
#    x = x[mask]
#    y = y[mask]
#    plt.figure(figsize=(6, 6))
#    plt.scatter(x, y, alpha=0.5)
#    plt.xlabel(xlabel if xlabel else 'X')
#    plt.ylabel(ylabel if ylabel else 'Y')
#    if xlim:
#        plt.xlim(xlim)
#    if ylim:
#        plt.ylim(ylim)
#    # compute correlation and best fit line
#    if corrclip:
#        mask_corr = (x >= corrclip[0]) & (x <= corrclip[1]) & (y >= corrclip[0]) & (y <= corrclip[1])
#        x_corr = x[mask_corr]
#        y_corr = y[mask_corr]
#    else:
#        x_corr = x
#        y_corr = y
#    corr_coef = np.corrcoef(x_corr, y_corr)[0, 1]
#    m, b = np.polyfit(x_corr, y_corr, 1)
#    plt.plot(x_corr, m*x_corr + b, color='orange', label=f'Best Fit Line (r={corr_coef:.2f})')
#    plt.legend()
#    if mae is not None:
#        plt.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
#    if rmse is not None:
#        plt.text(0.05, 0.90, f'RMSE: {rmse:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
#    if r2 is not None:
#        plt.text(0.05, 0.85, f'R²: {r2:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
#    if n is not None:
#        plt.text(0.05, 0.80, f'N: {n}', transform=plt.gca().transAxes, verticalalignment='top')
#    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
#    plt.close()


def heatmap(
    data,
    x_labels,
    y_labels,
    xlabel,
    ylabel,
    save_path,
    cbar_label=None,
    vmin=None,
    vmax=None,
    figsize=(8, 6),
    cmap_name="viridis",
):
    """
    Plot a 2D heatmap and save to disk.

    This version guarantees that all categories appearing in
    either x_labels or y_labels are shown on BOTH axes.
    The matrix is expanded to a square matrix with the union
    of labels on each axis, so the diagonal always represents
    'same category vs same category'.

    Parameters
    ----------
    data : 2D array-like (ny, nx)
        Values to plot.
    x_labels : sequence
        Labels for the columns (x-axis, satellite LC).
    y_labels : sequence
        Labels for the rows (y-axis, measured LC).
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    save_path : str
        Where to save the figure (e.g. '...png').
    cbar_label : str, optional
        Label for the colorbar.
    vmin, vmax : float, optional
        Color scale limits.
    figsize : tuple, optional
        Figure size in inches.
    """
    # Ensure array
    data = np.asarray(data, dtype=float)

    # Normalize labels to strings + lists
    x_labels = [str(l) for l in x_labels]
    y_labels = [str(l) for l in y_labels]

    # --------------------------------------------------
    # Expand to union-of-labels square matrix
    # --------------------------------------------------
    # Keep original y order, then append any x-only labels
    union_labels = list(y_labels)
    for lbl in x_labels:
        if lbl not in union_labels:
            union_labels.append(lbl)

    # If labels already match and matrix is square,
    # we skip the expansion step.
    need_expand = (
        (set(x_labels) != set(y_labels)) or
        (data.shape[0] != data.shape[1]) or
        (len(union_labels) != len(x_labels)) or
        (len(union_labels) != len(y_labels))
    )

    if need_expand:
        n_union = len(union_labels)
        # Start with all NaN (-> gray) for missing pairs
        new_data = np.full((n_union, n_union), np.nan,
                           dtype=float)

        union_index = {lbl: i for i, lbl in
                       enumerate(union_labels)}
        y_index = {lbl: i for i, lbl in
                   enumerate(y_labels)}
        x_index = {lbl: i for i, lbl in
                   enumerate(x_labels)}

        # Map each existing (y, x) cell into the union grid
        for y_lbl, yi_old in y_index.items():
            for x_lbl, xi_old in x_index.items():
                yi_new = union_index[y_lbl]
                xi_new = union_index[x_lbl]
                new_data[yi_new, xi_new] = data[yi_old,
                                                xi_old]

        data = new_data
        x_labels = union_labels
        y_labels = union_labels

    # --------------------------------------------------
    # Mask NaNs so they can be shown with a "bad" color
    # --------------------------------------------------
    data_masked = np.ma.masked_invalid(data)

    #cmap = plt.get_cmap("viridis").copy()
    cmap = plt.get_cmap(cmap_name).copy()
    # Gray out missing comparisons
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data_masked,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    # Tick positions
    nx = len(x_labels)
    ny = len(y_labels)
    ax.set_xticks(np.arange(nx))
    ax.set_yticks(np.arange(ny))

    # Tick labels
    ax.set_xticklabels(x_labels,
                       rotation=45,
                       ha="right")
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)

    fig.tight_layout()

    # Ensure directory exists, then save
    os.makedirs(os.path.dirname(save_path),
                exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_timeseries_by_site(
    preds_df,
    out_dir,
    data_col_name,
    y_label,
    date_col="date",
):
    os.makedirs(out_dir, exist_ok=True)

    # ensure datetime (safe even if already datetime)
    preds_df = preds_df.copy()
    preds_df[date_col] = pd.to_datetime(preds_df[date_col])

    unique_locs = preds_df[['lat', 'lon']].drop_duplicates()

    for _, loc in unique_locs.iterrows():
        lat = loc['lat']
        lon = loc['lon']

        site_data = (
            preds_df[
                (preds_df['lat'] == lat) &
                (preds_df['lon'] == lon)
            ]
            .sort_values(date_col)
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            site_data["date"],
            site_data[data_col_name],
            linewidth=1.5
        )

        ax.set_title(f"Predictions for Site ({lat:.4f}, {lon:.4f})")
        ax.set_xlabel("Date")
        ax.set_ylabel(y_label)

        # Force YYYY-MM ticks
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"timeseries_{lat:.4f}_{lon:.4f}.png"), dpi=250)
        plt.close(fig)

#def plot_timeseries_by_site(preds_df, out_dir, data_col_name,y_label):
#    # get the unique lat/lon combinations from our dataset
#    unique_locs = preds_df[['lat', 'lon']].drop_duplicates()
#    for _, loc in unique_locs.iterrows():
#        lat = loc['lat']
#        lon = loc['lon']
#        site_data = preds_df[(preds_df['lat'] == lat) & (preds_df['lon'] == lon)]
#        plt.figure()
#        plt.plot(site_data['date'], site_data[data_col_name])
#        plt.title(f"Predictions for Site ({lat}, {lon})")
#        plt.xlabel("Date")
#        plt.ylabel(y_label)
#        plt.xticks(rotation=45)
#        plt.tight_layout()
#        plt.savefig(os.path.join(out_dir, f"timeseries_{lat}_{lon}.png"))
#        plt.close()

def plot_training_progression(
    train_losses,
    val_losses,
    test_losses,
    best_epoch,
    var_name,
    out_dir
):
    plt.figure()
    epochs = np.arange(len(train_losses)) + 1
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.axvline(best_epoch, color="green", linestyle="--", label="Best Epoch")
    # if test_losses is a single value, plot vert line, otherwise plot normal line
    if len(test_losses) == 1:
        plt.axhline(test_losses[0], color="red", linestyle="--", label="Test Loss")
    else:
        plt.plot(epochs, test_losses, label="Test Loss")
    plt.title("Training Progression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"training_progression_{var_name}.png"))
    plt.close()
