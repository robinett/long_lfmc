#!/usr/bin/env python3

import os
from typing import Dict, List, Optional, Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch


LANDCOVER_DISPLAY = {
    "overall": "Overall",
    "shrub": "Shrub",
    "evergreen_forest": "Evergreen Forest",
    "deciduous_forest": "Deciduous Forest",
    "grass": "Grass",
    "mixed_forest": "Mixed Forest",
    "unknown": "Unknown",
}


def _paper_rc_params(fontsize: int) -> Dict[str, object]:
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


def _ensure_parent_dir(save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def _format_landcover_labels(categories: Sequence[str]) -> List[str]:
    return [LANDCOVER_DISPLAY.get(str(cat), str(cat)) for cat in categories]


def _panel_limits_from_series(series_list: Sequence[Dict[str, object]], pad_fraction: float = 0.08) -> Optional[tuple]:
    vals = []
    for series in series_list:
        values = np.asarray(series.get("values", []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size > 0:
            vals.append(values)
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


def _safe_limits(data_limits: Optional[tuple], fallback: tuple) -> tuple:
    if data_limits is None:
        return fallback
    data_min, data_max = data_limits
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return fallback
    if data_max <= data_min:
        pad = max(abs(data_min) * 0.1, 1.0)
        return data_min - pad, data_max + pad
    return data_min, data_max


def _expand_limits_about_midpoint(data_limits: Optional[tuple], factor: float) -> Optional[tuple]:
    if data_limits is None:
        return None
    data_min, data_max = _safe_limits(data_limits, (0.0, 1.0))
    midpoint = 0.5 * (data_min + data_max)
    half_span = 0.5 * (data_max - data_min) * max(float(factor), 1.0)
    if half_span <= 0:
        half_span = max(abs(midpoint) * 0.1, 1.0)
    return midpoint - half_span, midpoint + half_span


def _apply_line_artist(ax, series: Dict[str, object], idx: int):
    dates = series["dates"]
    values = series["values"]
    lower = series.get("lower")
    upper = series.get("upper")
    color = series.get("color")
    if lower is not None and upper is not None:
        ax.fill_between(
            dates,
            lower,
            upper,
            color=color,
            alpha=series.get("fill_alpha", 0.14),
            linewidth=0,
            zorder=series.get("zorder", 2) - 1,
        )
    line, = ax.plot(
        dates,
        values,
        color=color,
        linestyle=series.get("linestyle", "-"),
        linewidth=series.get("linewidth", 2.0),
        marker=series.get("marker", None),
        markersize=series.get("markersize", 5),
        alpha=series.get("alpha", 1.0),
        label=series.get("label"),
        zorder=series.get("zorder", 3),
        markerfacecolor=series.get("markerfacecolor", color),
        markeredgecolor=series.get("markeredgecolor", color),
        markeredgewidth=series.get("markeredgewidth", 0.8 if series.get("marker") else 0.0),
    )
    return line if idx == 0 else None


def _annotate_bars(
    ax,
    bars,
    labels,
    fontsize: int,
    zero_floor_for_negative: bool = False,
    tops: Optional[Sequence[float]] = None,
) -> None:
    top_values = None if tops is None else np.asarray(tops, dtype=float)
    for idx, (bar, label) in enumerate(zip(bars, labels)):
        if label == "":
            continue
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        top = height
        if top_values is not None and idx < len(top_values) and np.isfinite(top_values[idx]):
            top = float(top_values[idx])
        offset = 0.01 * max(abs(top), abs(height), 1.0)
        va = "bottom"
        if height >= 0:
            y = top + offset
        elif zero_floor_for_negative:
            y = max(0.0, top) + offset
        else:
            va = "top"
            y = top - offset
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            y,
            label,
            ha="center",
            va=va,
            fontsize=fontsize,
        )


def _format_bar_metric_label(
    value: float,
    count: float,
    uncertainty: Optional[float] = None,
) -> str:
    parts = []
    parts.append(f"{value:.2f}")
    if np.isfinite(count):
        parts.append(f"N={int(count)}")
    return "\n".join(parts)


def plot_stacked_timeseries_panels(
    panels: Sequence[Dict[str, object]],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
) -> None:
    if len(panels) == 0:
        raise ValueError("No panels provided for stacked timeseries figure")
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=tuple(figsize),
            sharex=True,
            constrained_layout=False,
        )
        if len(panels) == 1:
            axes = [axes]
        prediction_legend = []
        observation_legend = []
        uncertainty_patch = None
        use_month_aligned_axis = any(
            bool(panel.get("use_month_aligned_axis", False)) for panel in panels
        )
        for idx, (ax, panel) in enumerate(zip(axes, panels)):
            right_series = panel.get("right_series", []) or []
            series_list = panel.get("series", [])
            for series in series_list:
                if uncertainty_patch is None and series.get("lower") is not None and series.get("upper") is not None:
                    uncertainty_patch = Patch(
                        facecolor=series.get("color", "0.5"),
                        alpha=series.get("fill_alpha", 0.14),
                        edgecolor="none",
                        label="Ensemble-based uncertainty",
                    )
                line = _apply_line_artist(ax, series, idx)
                if idx == 0 and line is not None:
                    target = (
                        prediction_legend
                        if series.get("legend_group") == "predictions"
                        else observation_legend
                    )
                    target.append((line, series.get("label")))
            ax.set_ylabel(panel.get("ylabel", "LFMC (%)"))
            ax.set_title(panel.get("title", ""), loc="left", pad=4)
            ax.grid(False)
            y_limits = _panel_limits_from_series(series_list)
            if panel.get("timeseries_mode") == "banded_sar":
                y_limits = _expand_limits_about_midpoint(y_limits, factor=2.0)
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            if len(right_series) > 0:
                ax_r = ax.twinx()
                for series in right_series:
                    if uncertainty_patch is None and series.get("lower") is not None and series.get("upper") is not None:
                        uncertainty_patch = Patch(
                            facecolor=series.get("color", "0.5"),
                            alpha=series.get("fill_alpha", 0.14),
                            edgecolor="none",
                            label="Ensemble-based uncertainty",
                        )
                    line = _apply_line_artist(ax_r, series, idx)
                    if idx == 0 and line is not None:
                        target = (
                            prediction_legend
                            if series.get("legend_group") == "predictions"
                            else observation_legend
                        )
                        target.append((line, series.get("label")))
                ax_r.set_ylabel(panel.get("right_ylabel", "VV / VH (dB)"))
                right_limits = _panel_limits_from_series(right_series, pad_fraction=0.02)
                if right_limits is not None:
                    ax_r.set_ylim(*right_limits)
                ax_r.grid(False)
        if use_month_aligned_axis:
            locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
            formatter = mdates.DateFormatter("%b")
            for ax, panel in zip(axes, panels):
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                ax.tick_params(axis="x", rotation=0)
            axes[-1].set_xlabel("Month")
        else:
            axes[-1].set_xlabel("Date")
            locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
            formatter = mdates.ConciseDateFormatter(locator)
            axes[-1].xaxis.set_major_locator(locator)
            axes[-1].xaxis.set_major_formatter(formatter)
        def _dedupe_legend(entries):
            seen = set()
            handles = []
            labels = []
            for handle, label in entries:
                if label in seen:
                    continue
                seen.add(label)
                handles.append(handle)
                labels.append(label)
            return handles, labels
        pred_handles, pred_labels = _dedupe_legend(prediction_legend)
        obs_handles, obs_labels = _dedupe_legend(observation_legend)
        total_legend_items = len(pred_labels) + len(obs_labels) + (1 if uncertainty_patch is not None else 0)
        if total_legend_items <= 4:
            combined_handles = pred_handles + obs_handles
            combined_labels = pred_labels + obs_labels
            if uncertainty_patch is not None:
                combined_handles.append(uncertainty_patch)
                combined_labels.append("Ensemble-based uncertainty")
            if len(combined_handles) > 0:
                fig.legend(
                    combined_handles,
                    combined_labels,
                    loc="lower center",
                    ncol=len(combined_handles),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.008),
                )
            bottom = 0.125
        else:
            if uncertainty_patch is not None:
                pred_handles = pred_handles + [uncertainty_patch]
                pred_labels = pred_labels + ["Ensemble-based uncertainty"]
            if len(pred_handles) > 0:
                fig.legend(
                    pred_handles,
                    pred_labels,
                    loc="lower center",
                    ncol=max(1, min(len(pred_labels), 4)),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.038),
                )
            if len(obs_handles) > 0:
                fig.legend(
                    obs_handles,
                    obs_labels,
                    loc="lower center",
                    ncol=max(1, min(len(obs_labels), 4)),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.004),
                )
            bottom = 0.165
        fig.subplots_adjust(
            left=0.09,
            right=0.91,
            top=0.95,
            bottom=bottom,
            hspace=0.26,
        )
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_site_observation_map(
    site_df: pd.DataFrame,
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    gmba_basic_shapefile: Optional[str] = None,
    cmap: str = "viridis",
    marker_size: float = 34.0,
    log_color_scale: bool = False,
    cbar_vmax: Optional[float] = None,
) -> None:
    if len(site_df) == 0:
        raise ValueError("No site rows provided for site observation map")
    work = site_df.copy()
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["n_obs"] = pd.to_numeric(work["n_obs"], errors="coerce")
    work = work.dropna(subset=["longitude", "latitude", "n_obs"]).reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("No finite site rows remain after cleaning for site observation map")

    counts = work["n_obs"].to_numpy(dtype=float)
    site_gdf = gpd.GeoDataFrame(
        work.copy(),
        geometry=gpd.points_from_xy(work["longitude"], work["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    xs = site_gdf.geometry.x.to_numpy(dtype=float)
    ys = site_gdf.geometry.y.to_numpy(dtype=float)
    x_pad = max(1.5e5, 0.06 * float(np.ptp(xs)) if len(xs) > 1 else 1.5e5)
    y_pad = max(1.5e5, 0.06 * float(np.ptp(ys)) if len(ys) > 1 else 1.5e5)
    extent = [
        float(np.min(xs) - x_pad),
        float(np.max(xs) + x_pad),
        float(np.min(ys) - y_pad),
        float(np.max(ys) + y_pad),
    ]

    with plt.rc_context(_paper_rc_params(fontsize)):
        fig = plt.figure(figsize=tuple(figsize))
        proj = ccrs.AlbersEqualArea(
            central_longitude=-96,
            central_latitude=23,
            false_easting=0,
            false_northing=0,
            standard_parallels=(29.5, 45.5),
            globe=ccrs.Globe(datum="NAD83"),
        )
        ax = plt.axes(projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.add_feature(
            cfeature.LAND.with_scale("50m"),
            facecolor="#f4f1ea",
            edgecolor="none",
        )
        ax.add_feature(
            cfeature.OCEAN.with_scale("50m"),
            facecolor="#e6eef5",
            edgecolor="none",
        )
        ax.coastlines(resolution="50m", linewidth=0.6, color="#4c5c68")
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linewidth=0.45,
            edgecolor="#5f6c72",
        )
        try:
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    "cultural",
                    "admin_1_states_provinces_lines",
                    "50m",
                ),
                linewidth=0.35,
                edgecolor="#8f9398",
                facecolor="none",
            )
        except Exception:
            pass

        positive_counts = counts[counts > 0]
        norm = None
        cbar_extend = "neither"
        if log_color_scale and positive_counts.size > 0:
            vmax = (
                float(cbar_vmax)
                if cbar_vmax is not None
                else float(np.max(positive_counts))
            )
            norm = LogNorm(vmin=float(np.min(positive_counts)), vmax=vmax)
            if np.any(positive_counts > vmax):
                cbar_extend = "max"
        elif cbar_vmax is not None:
            norm = Normalize(vmin=0.0, vmax=float(cbar_vmax))
            if np.any(counts > float(cbar_vmax)):
                cbar_extend = "max"

        scatter = ax.scatter(
            xs,
            ys,
            c=counts,
            s=float(marker_size),
            cmap=cmap,
            norm=norm,
            alpha=0.92,
            edgecolor="#1f1f1f",
            linewidth=0.18,
            transform=proj,
            zorder=3,
        )
        cbar = fig.colorbar(
            scatter,
            ax=ax,
            shrink=0.82,
            pad=0.02,
            extend=cbar_extend,
        )
        cbar.set_label("Number of observations")

        stats_lines = [
            f"Sites: {len(work):,}",
            f"Total LFMC observations: {int(np.nansum(counts)):,}",
            f"Median observations/site: {int(np.nanmedian(counts)):,}",
        ]
        ax.text(
            0.985,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "0.55",
            },
            fontsize=max(fontsize - 2, 8),
        )

        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_sar_sampling_summary(
    map_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    categories: Sequence[str],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    category_order: Sequence[str],
    category_labels: Dict[str, str],
    category_colors: Dict[str, str],
    map_marker_size: float = 26.0,
) -> None:
    if len(map_df) == 0:
        raise ValueError("No map rows provided for SAR sampling summary")
    if len(bar_df) == 0:
        raise ValueError("No bar rows provided for SAR sampling summary")
    work_map = map_df.copy()
    work_map["longitude"] = pd.to_numeric(work_map["longitude"], errors="coerce")
    work_map["latitude"] = pd.to_numeric(work_map["latitude"], errors="coerce")
    work_map = work_map.dropna(subset=["longitude", "latitude", "sample_type"]).reset_index(drop=True)
    if len(work_map) == 0:
        raise ValueError("No finite map rows remain after cleaning for SAR sampling summary")
    map_gdf = gpd.GeoDataFrame(
        work_map.copy(),
        geometry=gpd.points_from_xy(work_map["longitude"], work_map["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    map_gdf["x_proj"] = map_gdf.geometry.x.to_numpy(dtype=float)
    map_gdf["y_proj"] = map_gdf.geometry.y.to_numpy(dtype=float)
    xs = map_gdf["x_proj"].to_numpy(dtype=float)
    ys = map_gdf["y_proj"].to_numpy(dtype=float)
    x_pad = max(1.5e5, 0.06 * float(np.ptp(xs)) if len(xs) > 1 else 1.5e5)
    y_pad = max(1.5e5, 0.06 * float(np.ptp(ys)) if len(ys) > 1 else 1.5e5)
    extent = [
        float(np.min(xs) - x_pad),
        float(np.max(xs) + x_pad),
        float(np.min(ys) - y_pad),
        float(np.max(ys) + y_pad),
    ]
    work_bar = bar_df.copy()
    work_bar["n_samples"] = pd.to_numeric(work_bar["n_samples"], errors="coerce").fillna(0.0)
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig = plt.figure(figsize=tuple(figsize))
        proj = ccrs.AlbersEqualArea(
            central_longitude=-96,
            central_latitude=23,
            false_easting=0,
            false_northing=0,
            standard_parallels=(29.5, 45.5),
            globe=ccrs.Globe(datum="NAD83"),
        )
        gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0])
        ax_map = fig.add_subplot(gs[0, 0], projection=proj)
        ax_bar = fig.add_subplot(gs[0, 1])

        ax_map.set_extent(extent, crs=proj)
        ax_map.add_feature(
            cfeature.LAND.with_scale("50m"),
            facecolor="#f4f1ea",
            edgecolor="none",
        )
        ax_map.add_feature(
            cfeature.OCEAN.with_scale("50m"),
            facecolor="#e6eef5",
            edgecolor="none",
        )
        ax_map.coastlines(resolution="50m", linewidth=0.6, color="#4c5c68")
        ax_map.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linewidth=0.45,
            edgecolor="#5f6c72",
        )
        try:
            ax_map.add_feature(
                cfeature.NaturalEarthFeature(
                    "cultural",
                    "admin_1_states_provinces_lines",
                    "50m",
                ),
                linewidth=0.35,
                edgecolor="#8f9398",
                facecolor="none",
            )
        except Exception:
            pass

        legend_handles = []
        stats_lines = []
        for sample_type in category_order:
            subset = map_gdf[map_gdf["sample_type"] == sample_type].copy()
            if len(subset) == 0:
                continue
            label = category_labels.get(sample_type, str(sample_type))
            color = category_colors.get(sample_type, "#4c5c68")
            handle = ax_map.scatter(
                subset["x_proj"].to_numpy(dtype=float),
                subset["y_proj"].to_numpy(dtype=float),
                s=float(map_marker_size),
                color=color,
                alpha=0.88,
                edgecolor="#1f1f1f",
                linewidth=0.22,
                transform=proj,
                zorder=3,
                label=label,
            )
            legend_handles.append(handle)
            stats_lines.append(f"{label}: {len(subset):,} locations")
        if legend_handles:
            legend = ax_map.legend(
                handles=legend_handles,
                frameon=True,
                loc="lower left",
            )
            legend.get_frame().set_alpha(0.96)
            legend.get_frame().set_edgecolor("0.55")
        ax_map.set_title("a) VV/VH sampling locations", loc="left", pad=4)
        ax_map.text(
            0.985,
            0.98,
            "\n".join(stats_lines),
            transform=ax_map.transAxes,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "0.55",
            },
            fontsize=max(fontsize - 2, 8),
        )

        x = np.arange(len(categories))
        width = 0.82 / float(max(len(category_order), 1))
        ymax = 1.0
        for sample_idx, sample_type in enumerate(category_order):
            label = category_labels.get(sample_type, str(sample_type))
            color = category_colors.get(sample_type, "#4c5c68")
            offset = (sample_idx - ((len(category_order) - 1) / 2.0)) * width
            values = np.asarray(
                [
                    work_bar.loc[
                        (work_bar["sample_type"] == sample_type)
                        & (work_bar["dominant_landcover"] == category),
                        "n_samples",
                    ].sum()
                    for category in categories
                ],
                dtype=float,
            )
            bars = ax_bar.bar(
                x + offset,
                values,
                width=width,
                label=label,
                color=color,
                zorder=2,
            )
            labels = [f"N={int(value):,}" if value > 0 else "" for value in values]
            _annotate_bars(
                ax_bar,
                bars,
                labels,
                fontsize=max(fontsize - 5, 7),
            )
            ymax = max(ymax, float(np.max(values)) if values.size > 0 else 1.0)
        ax_bar.set_xticks(x, _format_landcover_labels(categories))
        ax_bar.tick_params(axis="x", rotation=25)
        for tick in ax_bar.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax_bar.set_xlabel("Dominant land cover")
        ax_bar.set_ylabel("Number of combined VV/VH samples")
        ax_bar.set_ylim(0.0, ymax * 1.16)
        ax_bar.set_title("b) Combined VV/VH samples by land cover", loc="left", pad=4)
        legend = ax_bar.legend(frameon=True, loc="upper right")
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_edgecolor("0.4")

        fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.93, wspace=0.12)
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _scatter_stats_text(metrics: Dict[str, float]) -> str:
    r2 = metrics.get("r2", np.nan)
    rmse = metrics.get("rmse", np.nan)
    n = metrics.get("n", 0)
    parts = [
        f"R² = {r2:.2f}" if np.isfinite(r2) else "R² = nan",
        f"RMSE = {rmse:.2f}" if np.isfinite(rmse) else "RMSE = nan",
        f"N = {int(n)}",
    ]
    return "\n".join(parts)


def _panel_stats_text(panel: Dict[str, object]) -> str:
    custom = panel.get("stats_text")
    if custom is not None:
        return str(custom)
    return _scatter_stats_text(panel["metrics"])


def plot_scatter_triptych(
    panels: Sequence[Dict[str, object]],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
) -> None:
    if len(panels) not in {2, 3, 4}:
        raise ValueError("Scatter layout expects exactly 2, 3, or 4 panels")
    with plt.rc_context(_paper_rc_params(fontsize)):
        if len(panels) == 4:
            fig, axes = plt.subplots(2, 2, figsize=tuple(figsize), constrained_layout=False)
            axes = np.asarray(axes).reshape(-1)
        elif len(panels) == 2:
            fig, axes = plt.subplots(1, 2, figsize=tuple(figsize), constrained_layout=False)
            axes = np.asarray(axes).reshape(-1)
        else:
            fig, axes = plt.subplots(1, 3, figsize=tuple(figsize), constrained_layout=False)
            axes = np.asarray(axes).reshape(-1)
        for ax, panel in zip(axes, panels):
            x = np.asarray(panel["x"], dtype=float)
            y = np.asarray(panel["y"], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            color_array = panel.get("color_array")
            if color_array is not None:
                color_array = np.asarray(color_array, dtype=float)
                mask = mask & np.isfinite(color_array)
            x = x[mask]
            y = y[mask]
            if color_array is not None:
                color_array = color_array[mask]
            draw_identity = bool(panel.get("draw_identity", True))
            if draw_identity:
                default_x_min = float(min(np.min(x), np.min(y)))
                default_x_max = float(max(np.max(x), np.max(y)))
                default_y_min = default_x_min
                default_y_max = default_x_max
            else:
                default_x_min = float(np.min(x))
                default_x_max = float(np.max(x))
                default_y_min = float(np.min(y))
                default_y_max = float(np.max(y))
            xlim = panel.get("xlim")
            ylim = panel.get("ylim")
            line_min = min(default_x_min, default_y_min)
            line_max = max(default_x_max, default_y_max)
            if xlim is not None and ylim is not None:
                line_min = float(min(xlim[0], ylim[0]))
                line_max = float(max(xlim[1], ylim[1]))
            if panel.get("kind", "hexbin") == "hexbin":
                hb = ax.hexbin(
                    x,
                    y,
                    gridsize=int(panel.get("gridsize", 55)),
                    cmap=panel.get("cmap", "viridis"),
                    mincnt=1,
                    vmax=panel.get("cbar_vmax"),
                )
                cbar = fig.colorbar(
                    hb,
                    ax=ax,
                    fraction=0.046,
                    pad=0.03,
                    extend=panel.get("cbar_extend", "neither"),
                )
                cbar.set_label(panel.get("cbar_label", "Count"))
            else:
                norm = None
                if color_array is not None and panel.get("cbar_scale") == "log":
                    finite_color = color_array[np.isfinite(color_array) & (color_array > 0)]
                    if finite_color.size > 0:
                        norm = LogNorm(
                            vmin=float(np.min(finite_color)),
                            vmax=float(np.max(finite_color)),
                        )
                elif color_array is not None:
                    vmin = panel.get("cbar_vmin")
                    vmax = panel.get("cbar_vmax")
                    if vmin is None:
                        vmin = float(np.min(color_array))
                    if vmax is None:
                        vmax = float(np.max(color_array))
                    norm = Normalize(
                        vmin=float(vmin),
                        vmax=float(vmax),
                    )
                scatter = ax.scatter(
                    x,
                    y,
                    c=color_array if color_array is not None else panel.get("color", "#2f5d50"),
                    cmap=panel.get("cmap", "viridis"),
                    norm=norm,
                    s=panel.get("s", 34),
                    alpha=panel.get("alpha", 0.85),
                    edgecolor="none",
                )
                if color_array is not None:
                    cbar = fig.colorbar(
                        scatter,
                        ax=ax,
                        fraction=0.046,
                        pad=0.03,
                        extend=panel.get("cbar_extend", "neither"),
                    )
                    cbar.set_label(panel.get("cbar_label", "Color"))
            if draw_identity:
                ax.plot(
                    [line_min, line_max],
                    [line_min, line_max],
                    linestyle="--",
                    color="0.35",
                    linewidth=1.0,
                    zorder=1,
                )
            if xlim is not None:
                ax.set_xlim(*xlim)
            else:
                ax.set_xlim(default_x_min, default_x_max)
            if ylim is not None:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(default_y_min, default_y_max)
            ax.set_xlabel(panel["xlabel"])
            ax.set_ylabel(panel["ylabel"])
            ax.set_title(panel["title"], pad=4)
            ax.text(
                0.03,
                0.97,
                _panel_stats_text(panel),
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "0.5",
                },
                fontsize=max(fontsize - 2, 8),
            )
        for ax in axes[len(panels):]:
            ax.set_visible(False)
        if len(panels) == 4:
            fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.94, wspace=0.42, hspace=0.34)
        elif len(panels) == 2:
            fig.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.92, wspace=0.34)
        else:
            fig.subplots_adjust(left=0.06, right=0.985, bottom=0.14, top=0.92, wspace=0.34)
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_monthly_variability_bars(
    month_df,
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    bar_color: str,
) -> None:
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, ax = plt.subplots(figsize=tuple(figsize), constrained_layout=True)
        x = np.arange(len(month_df))
        values = month_df["pct_variability_captured_source_centered"].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=bar_color, width=0.72)
        ax.set_xticks(x, month_df["month_label"].tolist())
        ax.set_xlabel("Month")
        ax.set_ylabel("Variability explained (%)")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        labels = []
        for value, n_groups in zip(values, month_df["n_groups"].to_numpy(dtype=float)):
            if not np.isfinite(value):
                labels.append("")
                continue
            labels.append(f"{value:.1f}%\nN={int(n_groups)}")
        _annotate_bars(ax, bars, labels, fontsize=max(fontsize - 3, 8))
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_landcover_metric_grouped(
    categories: Sequence[str],
    metric_labels: Sequence[str],
    values: np.ndarray,
    counts: np.ndarray,
    errors: Optional[np.ndarray],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    colors: Sequence[str],
) -> None:
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, ax = plt.subplots(figsize=tuple(figsize), constrained_layout=True)
        x = np.arange(len(categories))
        n_metrics = values.shape[1]
        width = 0.92 / float(n_metrics)
        err_arr = None if errors is None else np.asarray(errors, dtype=float)
        all_bars = []
        for metric_idx in range(n_metrics):
            offset = (metric_idx - ((n_metrics - 1) / 2.0)) * width
            yerr = None if err_arr is None else err_arr[:, metric_idx]
            bars = ax.bar(
                x + offset,
                values[:, metric_idx],
                width=width,
                label=metric_labels[metric_idx],
                color=colors[metric_idx],
                yerr=yerr,
                ecolor="0.25",
                capsize=2.5,
                error_kw={"elinewidth": 1.2, "capthick": 1.1, "zorder": 4},
                zorder=2,
            )
            all_bars.append(bars)
        ax.set_xticks(x, _format_landcover_labels(categories))
        ax.set_xlabel("Land cover")
        ax.set_ylabel("R²")
        finite_vals = values[np.isfinite(values)]
        ymax = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
        if err_arr is not None:
            finite_combined = (values + err_arr)[np.isfinite(values + err_arr)]
            if finite_combined.size > 0:
                ymax = max(ymax, float(np.max(finite_combined)))
        ax.set_ylim(-0.1, max(ymax + 0.12, 0.35))
        for metric_idx, bars in enumerate(all_bars):
            labels = []
            top_positions = []
            for value, count in zip(values[:, metric_idx], counts[:, metric_idx]):
                if not np.isfinite(value):
                    labels.append("")
                    top_positions.append(np.nan)
                    continue
                err_val = 0.0
                if err_arr is not None:
                    err_candidate = err_arr[len(top_positions), metric_idx]
                    err_val = 0.0 if not np.isfinite(err_candidate) else float(err_candidate)
                labels.append(_format_bar_metric_label(value, count, err_val if np.isfinite(err_val) and err_val > 0 else np.nan))
                top_positions.append(float(value) + err_val)
            _annotate_bars(
                ax,
                bars,
                labels,
                fontsize=max(fontsize - 5, 7),
                zero_floor_for_negative=True,
                tops=top_positions,
            )
        legend = ax.legend(frameon=True, ncol=2)
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_edgecolor("0.4")
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_landcover_comparison_panels(
    categories: Sequence[str],
    model_labels: Sequence[str],
    colors: Sequence[str],
    panels: Sequence[Dict[str, object]],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
) -> None:
    if len(panels) == 0:
        raise ValueError("Landcover comparison plot expects at least 1 panel")
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=tuple(figsize),
            sharex=True,
            constrained_layout=False,
        )
        if len(panels) == 1:
            axes = [axes]
        x = np.arange(len(categories))
        width = 0.9 / float(len(model_labels))
        for ax, panel in zip(axes, panels):
            values = np.asarray(panel["values"], dtype=float)
            counts = np.asarray(panel.get("counts"), dtype=float) if panel.get("counts") is not None else None
            errors = np.asarray(panel.get("errors"), dtype=float) if panel.get("errors") is not None else None
            ymax = 0.35
            for model_idx, model_label in enumerate(model_labels):
                offset = (model_idx - ((len(model_labels) - 1) / 2.0)) * width
                yerr = None if errors is None else errors[:, model_idx]
                bars = ax.bar(
                    x + offset,
                    values[:, model_idx],
                    width=width,
                    label=model_label,
                    color=colors[model_idx],
                    yerr=yerr,
                    ecolor="0.25",
                    capsize=2.5,
                    error_kw={"elinewidth": 1.2, "capthick": 1.1, "zorder": 4},
                    zorder=2,
                )
                labels = []
                top_positions = []
                for row_idx, value in enumerate(values[:, model_idx]):
                    if not np.isfinite(value):
                        labels.append("")
                        top_positions.append(np.nan)
                        continue
                    count = np.nan if counts is None else counts[row_idx, model_idx]
                    err_val = 0.0
                    if errors is not None:
                        err_candidate = errors[row_idx, model_idx]
                        err_val = 0.0 if not np.isfinite(err_candidate) else float(err_candidate)
                    labels.append(
                        _format_bar_metric_label(
                            value,
                            count,
                            err_val if np.isfinite(err_val) and err_val > 0 else np.nan,
                        )
                    )
                    top_positions.append(float(value) + err_val)
                _annotate_bars(
                    ax,
                    bars,
                    labels,
                    fontsize=max(fontsize - 5, 7),
                    zero_floor_for_negative=True,
                    tops=top_positions,
                )
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size > 0:
                ymax = max(ymax, float(np.max(finite_vals)) + 0.12)
            if errors is not None:
                finite_combined = (values + errors)[np.isfinite(values + errors)]
                if finite_combined.size > 0:
                    ymax = max(ymax, float(np.max(finite_combined)) + 0.12)
            ax.set_ylim(-0.1, ymax)
            ax.set_ylabel(panel["ylabel"])
            ax.set_title(panel["title"], loc="left", pad=4)
        legend = axes[0].legend(frameon=True, ncol=len(model_labels), loc="upper right")
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_edgecolor("0.4")
        axes[-1].set_xticks(x, _format_landcover_labels(categories))
        axes[-1].set_xlabel("Land cover")
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.96, hspace=0.34)
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
