import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.dates as mdates
import math
import textwrap
from typing import Sequence
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def kde_plot(
    data, data_names, save_name, title=None,
    xlabel=None, ylabel=None
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
    plt.legend()
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

def plot_multiple_timeseries_from_df(
    df,
    date_col,
    x_label,
    y_label,
    save_name
):
    # get the columns that are not date
    columns = df.columns[df.columns != date_col]
    dates = df[date_col].values
    fig,ax = plt.subplots(figsize=(10, 6))
    for c,col in enumerate(columns):
        ax.plot(dates, df[col].values, label=col)
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
        plt.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    if rmse is not None:
        plt.text(0.05, 0.90, f'RMSE: {rmse:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    if r2 is not None:
        plt.text(0.05, 0.85, f'R²: {r2:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
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
):
    categories = [str(c) for c in categories]
    values = list(values)

    n = len(categories)
    max_label_len = max((len(c) for c in categories), default=0)
    use_horizontal = (max_label_len > 25) or (n > 8)

    def wrap_labels(labels, width):
        return ["\n".join(textwrap.wrap(lbl, width=width, break_long_words=False))
                if len(lbl) > width else lbl
                for lbl in labels]

    if use_horizontal:
        # --- layout heuristics ---
        wrap_width = max(20, min(42, 2 + int(0.7 * max_label_len)))
        ylabels = wrap_labels(categories, wrap_width)
        max_lines = max((lbl.count("\n") + 1 for lbl in ylabels), default=1)

        # more height per bar to prevent overlap; extra width for long labels
        height = max(4.5, 0.45 * n + 0.25 * max_lines + 1.0)
        width  = max(10.0, min(24.0, 9.0 + 0.18 * max_label_len))
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)

        y_pos = np.arange(n)
        bars = ax.barh(y_pos, values, color="skyblue")

        # tick labels with padding so they don't collide with bars
        ax.set_yticks(y_pos, ylabels)
        ax.tick_params(axis="y", labelsize=9, pad=6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # pad x-limit so value annotations never clip
        vmax = max(values) if values else 1.0
        ax.set_xlim(0, vmax * 1.12 + (2 if vmax < 20 else 0))

        # annotate using bar_label (handles barh nicely) with padding
        ax.bar_label(bars, labels=[f"{v:g}" for v in values],
                     padding=3, fontsize=8)

        # add small outer margins + adjust to give left labels breathing room
        ax.margins(y=0.02)
        # left margin scales with wrapped label complexity
        left_margin = min(0.55, 0.18 + 0.03 * max_lines + 0.002 * max_label_len)
        fig.subplots_adjust(left=left_margin, right=0.98, top=0.98, bottom=0.10)

        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    else:
        wrap_width = max(12, min(28, 2 + int(0.45 * max_label_len)))
        xlabels = wrap_labels(categories, wrap_width)
        max_lines = max((lbl.count("\n") + 1 for lbl in xlabels), default=1)

        width  = max(8.5, 0.75 * n + 2.5)
        height = max(5.0, 4.0 + 0.40 * max_lines)

        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
        x = np.arange(n)
        bars = ax.bar(x, values, color="skyblue")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, xlabels)
        ax.tick_params(axis="x", labelsize=9, pad=6)

        # pad y-limit so labels above bars fit
        ymax = max(values) if values else 1.0
        ax.set_ylim(0, ymax * 1.10 + (0.5 if ymax < 10 else 0))

        # value labels above bars with small padding
        ax.bar_label(bars, labels=[f"{v:g}" for v in values],
                     padding=2, fontsize=8)

        # give bottom more room for wrapped ticks
        bottom_margin = min(0.40, 0.16 + 0.05 * (max_lines - 1))
        fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=bottom_margin)

        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

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

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    r2=None,
    n=None,
    corrclip=None
):
    # get rid of nans
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(xlabel if xlabel else 'X')
    plt.ylabel(ylabel if ylabel else 'Y')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    # compute correlation and best fit line
    if corrclip:
        mask_corr = (x >= corrclip[0]) & (x <= corrclip[1]) & (y >= corrclip[0]) & (y <= corrclip[1])
        x_corr = x[mask_corr]
        y_corr = y[mask_corr]
    else:
        x_corr = x
        y_corr = y
    corr_coef = np.corrcoef(x_corr, y_corr)[0, 1]
    m, b = np.polyfit(x_corr, y_corr, 1)
    plt.plot(x_corr, m*x_corr + b, color='orange', label=f'Best Fit Line (r={corr_coef:.2f})')
    plt.legend()
    if mae is not None:
        plt.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    if rmse is not None:
        plt.text(0.05, 0.90, f'RMSE: {rmse:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    if r2 is not None:
        plt.text(0.05, 0.85, f'R²: {r2:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    if n is not None:
        plt.text(0.05, 0.80, f'N: {n}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()