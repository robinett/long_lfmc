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

def map_points(
    lons,
    lats,
    counts_per_point,
    save_path,
    *,
    s_min=20,             # smallest marker area (points^2)
    s_max=300,            # largest  marker area (points^2)
    clip_quantiles=(0.00, 0.98),  # clip extreme counts for nicer scaling
):
    """
    Plot lon/lat points with sizes scaled by counts_per_point and a size legend.

    Parameters
    ----------
    lons, lats : array-like, same length
        Longitudes and latitudes (degrees). Points are paired by index.
    counts_per_point : array-like, same length
        Non-negative counts used to scale marker sizes.
    save_path : str
        Path to save the figure (PNG, PDF, etc.).
    s_min, s_max : float
        Min/max scatter 's' (area in points^2).
    clip_quantiles : tuple(float, float)
        Quantiles used to clip counts before scaling (handles outliers).
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    counts = np.asarray(counts_per_point, dtype=float)
    # Robust scaling of sizes
    q_lo, q_hi = np.clip(clip_quantiles, 0.0, 1.0)
    vmin = np.quantile(counts, q_lo)
    vmax = np.quantile(counts, q_hi)
    if not np.isfinite(vmin):
        vmin = np.nanmin(counts)
    if not np.isfinite(vmax):
        vmax = np.nanmax(counts)
    if vmax <= vmin:  # all equal
        sizes = np.full_like(counts, (s_min + s_max) / 2.0, dtype=float)
    else:
        counts_clipped = np.clip(counts, vmin, vmax)
        sizes = s_min + (counts_clipped - vmin) / (vmax - vmin) * (s_max - s_min)
    # Figure & map setup
    proj_data = ccrs.PlateCarree()
    proj_map = ccrs.PlateCarree()  # keep geographic; swap to Albers/Mercator if you prefer
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=proj_map)
    # Add a simple basemap
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f5f5")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e8f2ff")
    ax.coastlines(resolution="50m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    try:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "cultural", "admin_1_states_provinces_lines", "50m"
            ),
            linewidth=0.4, edgecolor="0.6", facecolor="none"
        )
    except Exception:
        pass  # states layer optional
    # Set extent with a little padding
    ax.set_extent([-125,-100,25,50])
    #ax.set_extent([lons.min()-1, lons.max()+1, lats.min()-1, lats.max()+1], crs=proj_data)
    # Plot points
    sc = ax.scatter(
        lons, lats,
        s=sizes,
        c="#2b83ba",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.2,
        transform=proj_data,
    )
    # Build a size legend with nice representative values
    if vmax <= vmin:
        legend_vals = [counts.mean()]
    else:
        # pick 3–4 rounded levels between vmin and vmax
        levels = np.linspace(vmin, vmax, 4)
        # round to sensible precision
        rng = vmax - vmin
        if rng > 0:
            step = 10 ** np.floor(np.log10(rng / 3.0))
            legend_vals = list(np.unique(np.maximum(0, np.round(levels / step) * step)))
        else:
            legend_vals = [vmin]
        # ensure unique and sorted
        legend_vals = sorted({float(v) for v in legend_vals})
    def size_for(v):
        if vmax <= vmin:
            return (s_min + s_max) / 2.0
        v_clipped = np.clip(v, vmin, vmax)
        return s_min + (v_clipped - vmin) / (vmax - vmin) * (s_max - s_min)
    proxies = [
        plt.scatter([], [], s=size_for(v), color="#2b83ba", alpha=0.7,
                    edgecolor="k", linewidth=0.2)
        for v in legend_vals
    ]
    labels = [f"{v:g}" for v in legend_vals]
    leg = ax.legend(
        proxies, labels, title="Count",
        scatterpoints=1, frameon=True, fontsize=9, title_fontsize=10,
        loc="lower left", bbox_to_anchor=(0.01, 0.01)
    )
    leg.get_frame().set_alpha(0.9)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)