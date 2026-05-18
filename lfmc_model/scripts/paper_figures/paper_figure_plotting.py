#!/usr/bin/env python3

import os
from typing import Dict, List, Optional, Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Patch
from shapely.ops import unary_union


LANDCOVER_DISPLAY = {
    "overall": "Overall",
    "shrub": "Shrub",
    "evergreen_forest": "Evergreen Forest",
    "deciduous_forest": "Deciduous Forest",
    "grass": "Grass",
    "mixed_forest": "Mixed Forest",
    "unknown": "Unknown",
}

LFMC_BROWN_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "lfmc_brown_green",
    ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"],
)


def _paper_rc_params(fontsize: int) -> Dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Futura", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
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


def _save_figure_outputs(fig, save_path: str, dpi: int, bbox_inches: str = "tight") -> None:
    _ensure_parent_dir(save_path)
    print(f"Saving PNG: {save_path}", flush=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved PNG: {save_path}", flush=True)
    stem, _ = os.path.splitext(save_path)
    svg_path = f"{stem}.svg"
    print(f"Saving SVG: {svg_path}", flush=True)
    fig.savefig(svg_path, bbox_inches=bbox_inches)
    print(f"Saved SVG: {svg_path}", flush=True)


def _albers_equal_area_5070():
    return ccrs.AlbersEqualArea(
        central_longitude=-96,
        central_latitude=23,
        false_easting=0,
        false_northing=0,
        standard_parallels=(29.5, 45.5),
        globe=ccrs.Globe(datum="NAD83"),
    )


def _projected_extent(xs: np.ndarray, ys: np.ndarray, pad_fraction: float = 0.06) -> List[float]:
    x_pad = max(1.5e5, pad_fraction * float(np.ptp(xs)) if len(xs) > 1 else 1.5e5)
    y_pad = max(1.5e5, pad_fraction * float(np.ptp(ys)) if len(ys) > 1 else 1.5e5)
    return [
        float(np.min(xs) - x_pad),
        float(np.max(xs) + x_pad),
        float(np.min(ys) - y_pad),
        float(np.max(ys) + y_pad),
    ]


def _add_paper_map_background(ax) -> None:
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


def _prediction_region_linework():
    prediction_states = {
        "Arizona", "California", "Colorado", "Idaho", "Montana",
        "New Mexico", "Nevada", "Oregon", "Texas", "Utah", "Washington", "Wyoming",
    }
    states_shp = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_1_states_provinces",
    )
    state_geoms = [
        rec.geometry for rec in shpreader.Reader(states_shp).records()
        if rec.attributes.get("admin") == "United States of America"
        and rec.attributes.get("name") in prediction_states
    ]
    prediction_union = unary_union(state_geoms)
    coast_shp = shpreader.natural_earth(
        resolution="50m", category="physical", name="coastline",
    )
    clipped_coasts = [
        geom.intersection(prediction_union)
        for geom in shpreader.Reader(coast_shp).geometries()
        if geom.intersects(prediction_union)
    ]
    clipped_coasts = [geom for geom in clipped_coasts if not geom.is_empty]
    border_shp = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_boundary_lines_land",
    )
    clipped_borders = [
        geom.intersection(prediction_union)
        for geom in shpreader.Reader(border_shp).geometries()
        if geom.intersects(prediction_union)
    ]
    clipped_borders = [geom for geom in clipped_borders if not geom.is_empty]
    return state_geoms, clipped_coasts, clipped_borders


def _add_prediction_region_lines(
    ax,
    state_geoms,
    clipped_coasts,
    clipped_borders,
    border_color: str = "#888888",
    border_lw: float = 0.8,
) -> None:
    ax.add_feature(
        ShapelyFeature(state_geoms, ccrs.PlateCarree()),
        facecolor="none",
        edgecolor=border_color,
        linewidth=border_lw,
        zorder=2,
    )
    if clipped_coasts:
        ax.add_feature(
            ShapelyFeature(clipped_coasts, ccrs.PlateCarree()),
            facecolor="none",
            edgecolor=border_color,
            linewidth=border_lw,
            zorder=2,
        )
    if clipped_borders:
        ax.add_feature(
            ShapelyFeature(clipped_borders, ccrs.PlateCarree()),
            facecolor="none",
            edgecolor=border_color,
            linewidth=border_lw,
            zorder=2,
        )


def _format_landcover_labels(categories: Sequence[str]) -> List[str]:
    return [LANDCOVER_DISPLAY.get(str(cat), str(cat)) for cat in categories]


def _format_landcover_labels_wrapped(categories: Sequence[str]) -> List[str]:
    labels = _format_landcover_labels(categories)
    return [label.replace(" Forest", "\nForest") for label in labels]


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


def _annotate_metric_bars(
    ax,
    bars,
    values: Sequence[float],
    counts: Sequence[float],
    fontsize: int,
    tops: Optional[Sequence[float]] = None,
    value_fontsize: Optional[float] = None,
    count_fontsize: Optional[float] = None,
    value_rotation: float = 0.0,
    value_rotation_mode: str = "anchor",
    value_offset_scale: float = 0.01,
    value_ha: str = "center",
    value_va: str = "bottom",
    count_y: Optional[float] = None,
    count_transform=None,
    count_prefix: str = "N=",
    count_rotation: float = 0.0,
    count_ha: str = "center",
    count_va: str = "bottom",
    value_y_min: Optional[float] = None,
) -> None:
    top_values = None if tops is None else np.asarray(tops, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    count_arr = np.asarray(counts, dtype=float)
    resolved_value_fontsize = fontsize if value_fontsize is None else float(value_fontsize)
    resolved_count_fontsize = fontsize if count_fontsize is None else float(count_fontsize)
    for idx, bar in enumerate(bars):
        if idx >= len(value_arr) or not np.isfinite(value_arr[idx]):
            continue
        top = bar.get_height()
        if top_values is not None and idx < len(top_values) and np.isfinite(top_values[idx]):
            top = float(top_values[idx])
        offset = float(value_offset_scale) * max(abs(top), abs(bar.get_height()), 1.0)
        x_loc = bar.get_x() + (bar.get_width() / 2.0)
        value_y = top + offset
        resolved_value_va = value_va
        if value_y_min is not None and value_y < value_y_min:
            value_y = value_y_min
            resolved_value_va = "bottom"
        ax.text(
            x_loc,
            value_y,
            f"{float(value_arr[idx]):.2f}",
            ha=value_ha,
            va=resolved_value_va,
            fontsize=resolved_value_fontsize,
            rotation=value_rotation,
            rotation_mode=value_rotation_mode,
            clip_on=False,
        )
        if idx < len(count_arr) and np.isfinite(count_arr[idx]) and count_y is not None:
            count_label = f"{count_prefix}{int(round(float(count_arr[idx])))}"
            ax.text(
                x_loc,
                count_y,
                count_label,
                ha=count_ha,
                va=count_va,
                fontsize=resolved_count_fontsize,
                rotation=count_rotation,
                rotation_mode="anchor",
                transform=count_transform if count_transform is not None else ax.transData,
                clip_on=False,
            )


def plot_stacked_timeseries_panels(
    panels: Sequence[Dict[str, object]],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    locator_inset_bounds: Optional[Sequence[float]] = None,
    locator_marker_size: float = 24.0,
    uncertainty_before_observations: bool = False,
    legend_fontsize: Optional[float] = None,
    legend_ncol: Optional[int] = None,
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
        has_locator_insets = any(
            panel.get("site_latitude") is not None and panel.get("site_longitude") is not None
            for panel in panels
        )
        state_geoms, clipped_coasts, clipped_borders = _prediction_region_linework()
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
            ax.text(
                -0.06,
                1.03,
                chr(ord("a") + idx),
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontweight="bold",
                fontsize=fontsize + 6,
                clip_on=False,
            )
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
            site_lat = panel.get("site_latitude")
            site_lon = panel.get("site_longitude")
            if site_lat is not None and site_lon is not None:
                inset_bounds = list(locator_inset_bounds or [1.01, 0.56, 0.18, 0.47])
                inset_ax = ax.inset_axes(
                    inset_bounds,
                    projection=_albers_equal_area_5070(),
                )
                inset_ax.set_extent([-125, -101, 30, 50], crs=ccrs.PlateCarree())
                try:
                    inset_ax.outline_patch.set_visible(False)
                except AttributeError:
                    for spine in inset_ax.spines.values():
                        spine.set_visible(False)
                _add_prediction_region_lines(
                    inset_ax,
                    state_geoms,
                    clipped_coasts,
                    clipped_borders,
                )
                inset_ax.scatter(
                    [float(site_lon)],
                    [float(site_lat)],
                    transform=ccrs.PlateCarree(),
                    s=locator_marker_size,
                    color="#cf5c36",
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
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
        resolved_legend_fontsize = (
            float(legend_fontsize)
            if legend_fontsize is not None
            else float(mpl.rcParams.get("legend.fontsize", max(fontsize - 2, 8)))
        )
        resolved_legend_ncol = int(legend_ncol) if legend_ncol is not None else None
        if total_legend_items <= 4:
            if uncertainty_patch is not None:
                if uncertainty_before_observations:
                    combined_handles = pred_handles + [uncertainty_patch] + obs_handles
                    combined_labels = pred_labels + ["Ensemble-based uncertainty"] + obs_labels
                else:
                    combined_handles = pred_handles + obs_handles + [uncertainty_patch]
                    combined_labels = pred_labels + obs_labels + ["Ensemble-based uncertainty"]
            else:
                combined_handles = pred_handles + obs_handles
                combined_labels = pred_labels + obs_labels
            if len(combined_handles) > 0:
                fig.legend(
                    combined_handles,
                    combined_labels,
                    loc="lower center",
                    ncol=resolved_legend_ncol or len(combined_handles),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.008),
                    fontsize=resolved_legend_fontsize,
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
                    ncol=resolved_legend_ncol or max(1, min(len(pred_labels), 4)),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.038),
                    fontsize=resolved_legend_fontsize,
                )
            if len(obs_handles) > 0:
                fig.legend(
                    obs_handles,
                    obs_labels,
                    loc="lower center",
                    ncol=resolved_legend_ncol or max(1, min(len(obs_labels), 4)),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.004),
                    fontsize=resolved_legend_fontsize,
                )
            bottom = 0.165
        fig.subplots_adjust(
            left=0.12,
            right=0.80 if (has_locator_insets and locator_inset_bounds is not None) else (0.84 if has_locator_insets else 0.91),
            top=0.95,
            bottom=bottom,
            hspace=0.34,
        )
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
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
        proj = _albers_equal_area_5070()
        ax = plt.axes(projection=proj)
        ax.set_extent(extent, crs=proj)
        _add_paper_map_background(ax)

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

        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_lfmc_snapshot_quadrants(
    panels: Sequence[Dict[str, object]],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    cmap=LFMC_BROWN_GREEN_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    col_labels: Optional[Sequence[str]] = None,
    row_labels: Optional[Sequence[str]] = None,
    state_lines_only: bool = False,
    subplot_wspace: float = -0.15,
    subplot_hspace: float = 0.04,
) -> None:
    finite_values = []
    for panel in panels:
        values = np.asarray(panel["values"], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            finite_values.append(finite)
    if len(finite_values) == 0:
        raise ValueError("No finite LFMC map values available for snapshot figure")
    if col_labels is not None and row_labels is not None:
        ncols = len(col_labels)
        nrows = len(row_labels)
        expected_panels = nrows * ncols
        if len(panels) != expected_panels:
            raise ValueError(
                f"LFMC snapshot figure expected {expected_panels} panels for a "
                f"{nrows}x{ncols} layout, got {len(panels)}"
            )
    else:
        ncols = min(len(panels), 3)
        nrows = int(np.ceil(len(panels) / max(ncols, 1)))
    combined = np.concatenate(finite_values)
    if vmin is None:
        vmin = float(np.percentile(combined, 2))
    if vmax is None:
        vmax = float(np.percentile(combined, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid LFMC color limits: vmin={vmin}, vmax={vmax}")

    x_vals = np.asarray(panels[0]["x"], dtype=float)
    y_vals = np.asarray(panels[0]["y"], dtype=float)
    extent = [
        float(np.min(x_vals)),
        float(np.max(x_vals)),
        float(np.min(y_vals)),
        float(np.max(y_vals)),
    ]
    proj = _albers_equal_area_5070()
    if state_lines_only:
        state_geoms, clipped_coasts, clipped_borders = _prediction_region_linework()
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=tuple(figsize),
            subplot_kw={"projection": proj},
            constrained_layout=False,
            squeeze=False,
        )
        axes = np.asarray(axes).reshape(-1)
        label_fontsize = fontsize + 12
        mappable = None
        for idx, (ax, panel) in enumerate(zip(axes, panels)):
            row_idx = idx // ncols
            col_idx = idx % ncols
            ax.set_extent(extent, crs=proj)
            ax.patch.set_facecolor("none")
            ax.patch.set_alpha(0.0)
            try:
                ax.outline_patch.set_visible(False)
            except AttributeError:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if not state_lines_only:
                _add_paper_map_background(ax)
            values = np.asarray(panel["values"], dtype=float)
            x_panel = np.asarray(panel["x"], dtype=float)
            y_panel = np.asarray(panel["y"], dtype=float)
            mappable = ax.imshow(
                values,
                origin="upper",
                extent=[
                    float(np.min(x_panel)),
                    float(np.max(x_panel)),
                    float(np.min(y_panel)),
                    float(np.max(y_panel)),
                ],
                transform=proj,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                zorder=2,
            )
            if state_lines_only:
                _add_prediction_region_lines(ax, state_geoms, clipped_coasts, clipped_borders)
            panel_label = panel.get("panel_label")
            if panel_label in {None, ""}:
                raw_title = panel.get("title")
                if raw_title not in {None, ""}:
                    panel_label = str(raw_title).split(")", 1)[0].strip()
            if panel_label not in {None, ""}:
                ax.text(
                    -0.01,
                    1.015,
                    str(panel_label),
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontweight="bold",
                    fontsize=label_fontsize,
                    clip_on=False,
                )
            if col_labels is not None and row_labels is not None:
                if row_idx == 0:
                    ax.set_title(
                        col_labels[col_idx], loc="center", pad=22,
                        fontsize=label_fontsize + 2,
                    )
                if col_idx == 0:
                    ax.text(
                        -0.14, 0.5, row_labels[row_idx],
                        transform=ax.transAxes,
                        rotation=0,
                        va="center", ha="right",
                        fontsize=label_fontsize + 2,
                        linespacing=1.4,
                    )
            else:
                ax.set_title(str(panel["title"]), loc="left", pad=4)
        for ax in axes[len(panels):]:
            ax.set_visible(False)
        left_margin = 0.16 if row_labels is not None else 0.03
        cbar_left = 0.20 if row_labels is not None else 0.14
        cbar_width = 0.60 if ncols <= 2 else 0.64
        fig.subplots_adjust(
            left=left_margin,
            right=0.995,
            top=0.90,
            bottom=0.10,
            wspace=float(subplot_wspace),
            hspace=float(subplot_hspace),
        )
        cax = fig.add_axes([cbar_left, 0.03, cbar_width, 0.025])
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.set_label("Live Fuel Moisture Content (%)", fontsize=label_fontsize, labelpad=12)
        cbar.ax.tick_params(labelsize=label_fontsize - 2)
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_training_location_maps(
    panels: Sequence[Dict[str, object]],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    state_lines_only: bool = False,
) -> None:
    if len(panels) != 2:
        raise ValueError("Training location map figure expects exactly 2 panels")
    proj = _albers_equal_area_5070()
    all_xs = []
    all_ys = []
    gdfs = []
    for panel in panels:
        map_df = panel["map_df"].copy()
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["n_points"] = pd.to_numeric(map_df["n_points"], errors="coerce")
        map_df = map_df.dropna(subset=["longitude", "latitude", "n_points"]).reset_index(drop=True)
        if len(map_df) == 0:
            raise ValueError(f"No finite rows remain for panel '{panel['title']}'")
        gdf = gpd.GeoDataFrame(
            map_df,
            geometry=gpd.points_from_xy(map_df["longitude"], map_df["latitude"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:5070")
        gdf["x_proj"] = gdf.geometry.x.to_numpy(dtype=float)
        gdf["y_proj"] = gdf.geometry.y.to_numpy(dtype=float)
        all_xs.append(gdf["x_proj"].to_numpy(dtype=float))
        all_ys.append(gdf["y_proj"].to_numpy(dtype=float))
        gdfs.append(gdf)
    extent = _projected_extent(np.concatenate(all_xs), np.concatenate(all_ys))
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=tuple(figsize),
            subplot_kw={"projection": proj},
            constrained_layout=False,
        )
        axes = np.asarray(axes).reshape(-1)
        if state_lines_only:
            state_geoms, clipped_coasts, clipped_borders = _prediction_region_linework()
        for ax, panel, gdf in zip(axes, panels, gdfs):
            ax.set_extent(extent, crs=proj)
            try:
                ax.outline_patch.set_visible(False)
            except AttributeError:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if state_lines_only:
                _add_prediction_region_lines(ax, state_geoms, clipped_coasts, clipped_borders)
            else:
                _add_paper_map_background(ax)
            counts = gdf["n_points"].to_numpy(dtype=float)
            cmap = panel.get("cmap", "viridis")
            vmax = panel.get("cbar_vmax")
            color_vmax = float(np.nanmax(counts)) if vmax is None else float(vmax)
            if not np.isfinite(color_vmax) or color_vmax <= 0:
                color_vmax = 1.0
            norm = Normalize(vmin=0.0, vmax=color_vmax)
            extend = "neither"
            if np.any(counts > color_vmax):
                extend = "max"
            marker_defs = dict(panel.get("marker_defs", {}))
            scatter = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            scatter.set_array([])
            ordered_gdf = gdf[gdf["marker_group"].isin(marker_defs)].copy()
            ordered_gdf = ordered_gdf.sort_values(
                ["n_points", "x_proj", "y_proj"],
                ascending=[True, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            legend_handle_by_group = {}
            unique_counts = [
                float(count_val)
                for count_val in ordered_gdf["n_points"].drop_duplicates().sort_values().tolist()
            ]
            for z_idx, count_val in enumerate(unique_counts, start=3):
                count_subset = ordered_gdf[ordered_gdf["n_points"] == count_val]
                for marker_group, marker_style in marker_defs.items():
                    subset = count_subset[count_subset["marker_group"] == marker_group]
                    if len(subset) == 0:
                        continue
                    marker_size = float(marker_style.get("size", panel.get("marker_size", 34.0)))
                    marker_alpha = float(marker_style.get("alpha", 0.92))
                    marker_edgecolor = marker_style.get("edgecolor", "#1f1f1f")
                    marker_linewidth = float(marker_style.get("linewidth", 0.18))
                    handle = ax.scatter(
                        subset["x_proj"].to_numpy(dtype=float),
                        subset["y_proj"].to_numpy(dtype=float),
                        c=subset["n_points"].to_numpy(dtype=float),
                        cmap=cmap,
                        norm=norm,
                        s=marker_size,
                        marker=str(marker_style["marker"]),
                        alpha=marker_alpha,
                        edgecolor=marker_edgecolor,
                        linewidth=marker_linewidth,
                        transform=proj,
                        zorder=float(z_idx) + float(marker_style.get("zorder_offset", 0.0)),
                        label=str(marker_style["label"]) if marker_group not in legend_handle_by_group else None,
                    )
                    legend_handle_by_group.setdefault(marker_group, handle)
            legend_handles = [
                legend_handle_by_group[marker_group]
                for marker_group in marker_defs
                if marker_group in legend_handle_by_group
            ]
            if len(legend_handles) == 0:
                raise ValueError(f"No scatter points were drawn for panel '{panel['title']}'")
            cbar = fig.colorbar(
                scatter,
                ax=ax,
                shrink=0.82,
                pad=float(panel.get("cbar_pad", 0.02)),
                extend=extend,
            )
            cbar.set_label(str(panel["cbar_label"]), fontsize=fontsize)
            cbar.ax.tick_params(labelsize=max(fontsize - 1, 8))
            if len(legend_handles) > 1:
                legend = ax.legend(
                    handles=legend_handles,
                    frameon=True,
                    loc=str(panel.get("legend_loc", "lower left")),
                    bbox_to_anchor=panel.get("legend_bbox_to_anchor"),
                    fontsize=max(fontsize - 2, 8),
                )
                legend_zorder = 10000
                legend.set_zorder(legend_zorder)
                legend_frame = legend.get_frame()
                legend_frame.set_zorder(legend_zorder)
                legend_frame.set_facecolor("white")
                legend_frame.set_alpha(1.0)
                legend_frame.set_edgecolor("0.55")
                for text in legend.get_texts():
                    text.set_zorder(legend_zorder + 1)
                for handle in legend.legend_handles:
                    try:
                        handle.set_zorder(legend_zorder + 1)
                    except Exception:
                        pass
            stats_lines = [
                f"Sites: {int(gdf[['longitude', 'latitude']].drop_duplicates().shape[0]):,}",
                f"{str(panel.get('stats_total_label', 'Total points'))}: {int(np.nansum(counts)):,}",
                f"{str(panel.get('stats_mean_label', 'Mean points/site'))}: {float(np.nanmean(counts)):,.1f}",
            ]
            stats_text_zorder = 10000
            stats_text = ax.text(
                float(panel.get("stats_x", 0.985)),
                float(panel.get("stats_y", 0.98)),
                "\n".join(stats_lines),
                transform=ax.transAxes,
                ha=str(panel.get("stats_ha", "right")),
                va=str(panel.get("stats_va", "top")),
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "0.55",
                },
                fontsize=max(fontsize - 2, 8),
                zorder=stats_text_zorder,
                clip_on=False,
            )
            stats_bbox = stats_text.get_bbox_patch()
            if stats_bbox is not None:
                stats_bbox.set_zorder(stats_text_zorder)
            panel_label = panel.get("panel_label")
            if panel_label not in {None, ""}:
                ax.text(
                    -0.04,
                    1.01,
                    str(panel_label),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=fontsize + 2,
                    fontweight="bold",
                    clip_on=False,
                )
            ax.set_title(
                str(panel["title"]),
                loc=str(panel.get("title_loc", "center")),
                pad=float(panel.get("title_pad", 10)),
                fontweight=str(panel.get("title_fontweight", "bold")),
                fontsize=fontsize + int(panel.get("title_fontsize_offset", 2)),
            )
        fig.subplots_adjust(left=0.03, right=0.98, bottom=0.06, top=0.90, wspace=0.16)
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
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
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_site_r2_landcover_distribution(
    site_r2_df: pd.DataFrame,
    categories: Sequence[str],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    x_limits: Sequence[float],
    show_summary_text: bool = True,
    axis_label_fontsize: Optional[float] = None,
    tick_label_fontsize: Optional[float] = None,
    axis_label_pad: float = 4.0,
    legend_fontsize: Optional[float] = None,
    legend_loc: str = "best",
    line_width: float = 2.2,
) -> None:
    if len(site_r2_df) == 0:
        raise ValueError("No site-level R2 rows provided for landcover distribution plot")
    if len(categories) == 0:
        raise ValueError("No landcover categories provided for site-level R2 plot")
    x_min = float(x_limits[0])
    x_max = float(x_limits[1])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError(f"Invalid x_limits for site-level R2 plot: {x_limits}")
    work = site_r2_df.copy()
    work["site_r2"] = pd.to_numeric(work["site_r2"], errors="coerce")
    work = work[np.isfinite(work["site_r2"])].copy()
    work = work[work["dominant_landcover"].isin(categories)].copy()
    if len(work) == 0:
        raise ValueError("No finite site-level R2 rows remain after filtering to landcover categories")
    work["site_r2_clipped"] = work["site_r2"].clip(lower=x_min, upper=x_max)
    palette = sns.color_palette("colorblind", n_colors=max(len(categories), 1))
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, ax = plt.subplots(figsize=tuple(figsize))
        for idx, category in enumerate(categories):
            class_df = work[work["dominant_landcover"] == category].copy()
            if len(class_df) == 0:
                continue
            label = f"{LANDCOVER_DISPLAY.get(str(category), str(category))} (n={len(class_df)})"
            color = palette[idx]
            if len(class_df) >= 2:
                sns.kdeplot(
                    data=class_df,
                    x="site_r2_clipped",
                    ax=ax,
                    linewidth=float(line_width),
                    label=label,
                    color=color,
                    fill=False,
                    common_norm=False,
                    bw_adjust=0.25,
                    cut=0,
                    clip=(x_min, x_max),
                    gridsize=512,
                )
            else:
                ax.axvline(
                    float(class_df["site_r2_clipped"].iloc[0]),
                    color=color,
                    linewidth=float(line_width),
                    label=label,
                )
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(
                handles,
                labels,
                frameon=False,
                fontsize=max(fontsize - 2, 8) if legend_fontsize is None else float(legend_fontsize),
                loc=str(legend_loc),
                title=None,
            )
        ax.set_xlim(x_min, x_max)
        resolved_axis_label_fontsize = fontsize if axis_label_fontsize is None else float(axis_label_fontsize)
        resolved_tick_label_fontsize = max(fontsize - 2, 8) if tick_label_fontsize is None else float(tick_label_fontsize)
        ax.set_xlabel(
            "Site-level LFMC R²",
            fontsize=resolved_axis_label_fontsize,
            labelpad=float(axis_label_pad),
        )
        ax.set_ylabel(
            "Density",
            fontsize=resolved_axis_label_fontsize,
            labelpad=float(axis_label_pad),
        )
        ax.tick_params(axis="both", labelsize=resolved_tick_label_fontsize)
        if show_summary_text:
            ax.text(
                0.98,
                0.98,
                (
                    f"Sites = {len(work):,}\n"
                    f"Land covers = {work['dominant_landcover'].nunique()}"
                ),
                transform=ax.transAxes,
                va="top",
                ha="right",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "0.45",
                },
                fontsize=max(fontsize - 3, 8),
            )
        fig.tight_layout()
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_training_sample_landcover_comparison(
    categories: Sequence[str],
    dataset_labels: Sequence[str],
    colors: Sequence[str],
    values: np.ndarray,
    errors: Optional[np.ndarray],
    count_values: Optional[np.ndarray],
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
    note_text: Optional[str] = None,
    legend_below: bool = False,
    text_scale: float = 1.0,
    x_label_rotation: float = 25.0,
    counts_below_axis: bool = False,
    count_label_y: float = -0.12,
    value_label_fontsize: Optional[float] = None,
    count_label_fontsize: Optional[float] = None,
    value_label_rotation: float = 0.0,
    value_label_rotation_mode: str = "anchor",
    y_tick_fontsize: Optional[float] = None,
    category_label_fontsize: Optional[float] = None,
    y_label_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = None,
    legend_bbox_y: float = 0.02,
    legend_bottom: Optional[float] = None,
    x_tick_pad: Optional[float] = None,
    subplot_left: float = 0.09,
) -> None:
    if len(categories) == 0:
        raise ValueError("No landcover categories provided for training-sample comparison plot")
    if len(dataset_labels) == 0:
        raise ValueError("No dataset labels provided for training-sample comparison plot")
    val_arr = np.asarray(values, dtype=float)
    if val_arr.ndim != 2:
        raise ValueError("Training-sample comparison values must be a 2D array")
    err_arr = None if errors is None else np.asarray(errors, dtype=float)
    count_arr = None if count_values is None else np.asarray(count_values, dtype=float)
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, ax = plt.subplots(figsize=tuple(figsize))
        x = np.arange(len(categories))
        width = 0.82 / float(len(dataset_labels))
        ymax = 0.0
        label_fontsize = max(int(round((fontsize - 5) * float(text_scale))), 7)
        resolved_value_label_fontsize = label_fontsize if value_label_fontsize is None else float(value_label_fontsize)
        resolved_count_label_fontsize = label_fontsize if count_label_fontsize is None else float(count_label_fontsize)
        resolved_y_tick_fontsize = (
            max(int(round((fontsize - 1) * float(text_scale))), 8)
            if y_tick_fontsize is None else float(y_tick_fontsize)
        )
        resolved_category_label_fontsize = (
            max(int(round((fontsize - 1) * float(text_scale))), 8)
            if category_label_fontsize is None else float(category_label_fontsize)
        )
        resolved_y_label_fontsize = (
            int(round(fontsize * float(text_scale)))
            if y_label_fontsize is None else float(y_label_fontsize)
        )
        resolved_legend_fontsize = (
            max(int(round((fontsize - 1) * float(text_scale))), 8)
            if legend_fontsize is None else float(legend_fontsize)
        )
        for dataset_idx, dataset_label in enumerate(dataset_labels):
            offset = (dataset_idx - ((len(dataset_labels) - 1) / 2.0)) * width
            yerr = None if err_arr is None else err_arr[:, dataset_idx]
            bars = ax.bar(
                x + offset,
                val_arr[:, dataset_idx],
                width=width,
                label=dataset_label,
                color=colors[dataset_idx],
                yerr=yerr,
                ecolor="0.25",
                capsize=2.5,
                error_kw={"elinewidth": 1.2, "capthick": 1.1, "zorder": 4},
                zorder=2,
            )
            finite_vals = val_arr[:, dataset_idx][np.isfinite(val_arr[:, dataset_idx])]
            if finite_vals.size > 0:
                ymax = max(ymax, float(np.max(finite_vals)))
            if err_arr is not None:
                finite_tops = (val_arr[:, dataset_idx] + err_arr[:, dataset_idx])[
                    np.isfinite(val_arr[:, dataset_idx] + err_arr[:, dataset_idx])
                ]
                if finite_tops.size > 0:
                    ymax = max(ymax, float(np.max(finite_tops)))
            if count_arr is not None:
                top_positions = []
                count_col = np.asarray(count_arr[:, dataset_idx], dtype=float)
                for row_idx, value in enumerate(val_arr[:, dataset_idx]):
                    if not np.isfinite(value):
                        top_positions.append(np.nan)
                        continue
                    err_val = 0.0
                    if err_arr is not None:
                        err_candidate = err_arr[row_idx, dataset_idx]
                        err_val = 0.0 if not np.isfinite(err_candidate) else float(err_candidate)
                    top_positions.append(float(value) + err_val)
                if counts_below_axis:
                    _annotate_metric_bars(
                        ax,
                        bars,
                        values=val_arr[:, dataset_idx],
                        counts=count_col,
                        fontsize=label_fontsize,
                        value_fontsize=resolved_value_label_fontsize,
                        count_fontsize=resolved_count_label_fontsize,
                        value_rotation=float(value_label_rotation),
                        value_rotation_mode=str(value_label_rotation_mode),
                        value_ha="center",
                        tops=top_positions,
                        count_y=float(count_label_y),
                        count_transform=ax.get_xaxis_transform(),
                        count_ha="center",
                        count_va="top",
                    )
                else:
                    labels = []
                    for row_idx, value in enumerate(val_arr[:, dataset_idx]):
                        if not np.isfinite(value):
                            labels.append("")
                            continue
                        count_value = count_arr[row_idx, dataset_idx]
                        count_text = "" if not np.isfinite(count_value) else f"\nN={int(round(float(count_value))):,}"
                        labels.append(
                            f"{float(value):.2f}{count_text}"
                        )
                    _annotate_bars(
                        ax,
                        bars,
                        labels,
                        fontsize=label_fontsize,
                        zero_floor_for_negative=False,
                        tops=top_positions,
                    )
        ax.set_xticks(x, _format_landcover_labels(categories))
        x_tick_kwargs = {
            "axis": "x",
            "rotation": float(x_label_rotation),
            "labelsize": resolved_category_label_fontsize,
        }
        if x_tick_pad is not None:
            x_tick_kwargs["pad"] = float(x_tick_pad)
        ax.tick_params(**x_tick_kwargs)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right" if float(x_label_rotation) != 0.0 else "center")
        ax.set_xlabel("Land cover", fontsize=resolved_y_label_fontsize)
        ax.set_ylabel("Fraction of training samples", fontsize=resolved_y_label_fontsize)
        ax.set_ylim(0.0, min(1.0, ymax * 1.2 if ymax > 0 else 1.0))
        ax.tick_params(axis="y", labelsize=resolved_y_tick_fontsize)
        if legend_below:
            fig.legend(
                loc="lower center",
                ncol=max(1, min(len(dataset_labels), 4)),
                frameon=False,
                bbox_to_anchor=(0.5, float(legend_bbox_y)),
                fontsize=resolved_legend_fontsize,
            )
        else:
            legend = ax.legend(
                frameon=True,
                loc="upper right",
                fontsize=resolved_legend_fontsize,
            )
            legend.get_frame().set_alpha(1.0)
            legend.get_frame().set_edgecolor("0.4")
        if note_text not in {None, ""}:
            ax.text(
                0.98,
                0.98,
                str(note_text),
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.92,
                    "edgecolor": "0.45",
                },
                fontsize=max(int(round((fontsize - 3) * float(text_scale))), 8),
            )
        if legend_below:
            bottom = (
                float(legend_bottom)
                if legend_bottom is not None else (0.22 if counts_below_axis else 0.16)
            )
            fig.subplots_adjust(left=float(subplot_left), right=0.985, bottom=bottom, top=0.95)
        else:
            fig.tight_layout()
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_placeholder_figure(
    title: str,
    description: str,
    save_path: str,
    fontsize: int,
    figsize: Sequence[float],
    dpi: int,
) -> None:
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, ax = plt.subplots(figsize=tuple(figsize))
        fig.patch.set_facecolor("#f5f1e8")
        ax.set_facecolor("#f5f1e8")
        ax.axis("off")
        ax.text(
            0.5,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=fontsize + 4,
            color="#2f3b33",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.42,
            description,
            ha="center",
            va="center",
            fontsize=max(fontsize, 10),
            color="#4f5d53",
            transform=ax.transAxes,
            wrap=True,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.82,
                "edgecolor": "0.55",
            },
        )
        fig.tight_layout()
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _scatter_stats_text(metrics: Dict[str, float]) -> str:
    r2 = metrics.get("r2", np.nan)
    rmse = metrics.get("rmse", np.nan)
    bias = metrics.get("bias", np.nan)
    n = metrics.get("n", 0)
    parts = [
        f"R² = {r2:.2f}" if np.isfinite(r2) else "R² = nan",
        f"RMSE = {rmse:.2f}%" if np.isfinite(rmse) else "RMSE = nan%",
    ]
    if np.isfinite(bias):
        parts.append(f"Bias = {bias:.2f}%")
    else:
        parts.append(f"N = {int(n)}")
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
    title_fontsize: Optional[int] = None,
    axis_label_fontsize: Optional[int] = None,
    tick_label_fontsize: Optional[int] = None,
    colorbar_label_fontsize: Optional[int] = None,
    colorbar_tick_fontsize: Optional[int] = None,
    stats_fontsize: Optional[int] = None,
    panel_label_fontsize: Optional[int] = None,
) -> None:
    if len(panels) not in {2, 3, 4, 5}:
        raise ValueError("Scatter layout expects exactly 2, 3, 4, or 5 panels")
    with plt.rc_context(_paper_rc_params(fontsize)):
        if len(panels) == 5:
            fig, axes = plt.subplots(3, 2, figsize=tuple(figsize), constrained_layout=False)
            axes = np.asarray(axes).reshape(-1)
        elif len(panels) == 4:
            fig, axes = plt.subplots(2, 2, figsize=tuple(figsize), constrained_layout=False)
            axes = np.asarray(axes).reshape(-1)
        elif len(panels) == 2:
            fig, axes = plt.subplots(1, 2, figsize=tuple(figsize), constrained_layout=False)
            axes = np.asarray(axes).reshape(-1)
        else:
            fig, axes = plt.subplots(1, 3, figsize=tuple(figsize), constrained_layout=False)
            axes = np.asarray(axes).reshape(-1)
        panel_title_fontsize = int(title_fontsize) if title_fontsize is not None else fontsize + 4
        panel_label_fontsize = (
            int(panel_label_fontsize) if panel_label_fontsize is not None else fontsize + 6
        )
        axis_label_fontsize = (
            int(axis_label_fontsize) if axis_label_fontsize is not None else fontsize
        )
        tick_label_fontsize = (
            int(tick_label_fontsize) if tick_label_fontsize is not None else max(fontsize - 1, 8)
        )
        colorbar_label_fontsize = (
            int(colorbar_label_fontsize) if colorbar_label_fontsize is not None else fontsize
        )
        colorbar_tick_fontsize = (
            int(colorbar_tick_fontsize) if colorbar_tick_fontsize is not None else max(fontsize - 1, 8)
        )
        stats_fontsize = int(stats_fontsize) if stats_fontsize is not None else max(fontsize - 2, 8)

        for panel_idx, (ax, panel) in enumerate(zip(axes, panels), start=1):
            print(f"Rendering scatter panel {panel_idx}/{len(panels)}: {panel['title']}", flush=True)
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
                cbar.set_label(panel.get("cbar_label", "Count"), fontsize=colorbar_label_fontsize)
                cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
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
                    cbar.set_label(panel.get("cbar_label", "Color"), fontsize=colorbar_label_fontsize)
                    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
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
            ax.set_xlabel(panel["xlabel"], fontsize=axis_label_fontsize, labelpad=10)
            ax.set_ylabel(panel["ylabel"], fontsize=axis_label_fontsize, labelpad=10)
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)
            ax.set_title(panel["title"], pad=10, fontsize=panel_title_fontsize)
            if panel.get("panel_label") not in {None, ""}:
                ax.text(
                    -0.18,
                    1.12,
                    str(panel["panel_label"]),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontweight="bold",
                    fontsize=panel_label_fontsize,
                    clip_on=False,
                )
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
                fontsize=stats_fontsize,
            )
        for ax in axes[len(panels):]:
            ax.set_visible(False)
        if len(panels) == 5:
            fig.subplots_adjust(left=0.095, right=0.985, bottom=0.065, top=0.955, wspace=0.78, hspace=0.56)
        elif len(panels) == 4:
            fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.94, wspace=0.42, hspace=0.34)
        elif len(panels) == 2:
            fig.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.92, wspace=0.34)
        else:
            fig.subplots_adjust(left=0.06, right=0.985, bottom=0.14, top=0.92, wspace=0.34)
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
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
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
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
    legend_below: bool = False,
    group_gap_scale: float = 1.0,
    count_label_rotation: float = 0.0,
    count_values_only: bool = False,
    n_label: Optional[str] = None,
    count_label_y: float = -0.035,
    x_tick_pad: float = 36.0,
    y_tick_fontsize: Optional[float] = None,
    category_label_fontsize: Optional[float] = None,
    annotation_fontsize: Optional[float] = None,
    value_label_fontsize: Optional[float] = None,
    count_label_fontsize: Optional[float] = None,
    n_label_fontsize: Optional[float] = None,
    y_label_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = None,
    legend_ncol: Optional[int] = None,
    legend_bbox_y: float = 0.015,
    legend_bottom: float = 0.31,
    value_label_rotation: float = 0.0,
    wrap_landcover_labels: bool = False,
) -> None:
    with plt.rc_context(_paper_rc_params(fontsize)):
        fig, ax = plt.subplots(figsize=tuple(figsize), constrained_layout=False)
        n_metrics = values.shape[1]
        total_bar_width = 0.92
        group_spacing = total_bar_width + ((1.0 - total_bar_width) * float(group_gap_scale))
        x = np.arange(len(categories), dtype=float) * group_spacing
        width = total_bar_width / float(n_metrics)
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
        ax.set_xticks(x)
        category_labels = (
            _format_landcover_labels_wrapped(categories)
            if wrap_landcover_labels
            else _format_landcover_labels(categories)
        )
        ax.set_xticklabels(
            category_labels,
            fontsize=category_label_fontsize,
        )
        resolved_y_label_fontsize = (
            fontsize + 10
            if y_label_fontsize is None
            else float(y_label_fontsize)
        )
        ax.set_ylabel("R²", fontsize=resolved_y_label_fontsize, labelpad=12)
        finite_vals = values[np.isfinite(values)]
        ymax = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
        if err_arr is not None:
            finite_combined = (values + err_arr)[np.isfinite(values + err_arr)]
            if finite_combined.size > 0:
                ymax = max(ymax, float(np.max(finite_combined)))
        y_min = -0.045
        y_max = max(ymax + 0.12, 0.35)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(0.0, y_max + 0.001, 0.2))
        ax.tick_params(
            axis="y",
            which="major",
            length=5,
            width=1.0,
            labelleft=True,
            labelsize=y_tick_fontsize,
        )
        ax.tick_params(axis="x", which="major", length=5, width=1.0, pad=float(x_tick_pad))
        count_y = float(count_label_y)
        count_transform = ax.get_xaxis_transform()
        count_prefix = "" if count_values_only else "N="
        resolved_annotation_fontsize = (
            max(fontsize - 5, 7)
            if annotation_fontsize is None
            else float(annotation_fontsize)
        )
        resolved_value_label_fontsize = (
            resolved_annotation_fontsize
            if value_label_fontsize is None
            else float(value_label_fontsize)
        )
        resolved_count_label_fontsize = (
            resolved_annotation_fontsize
            if count_label_fontsize is None
            else float(count_label_fontsize)
        )
        resolved_n_label_fontsize = (
            resolved_count_label_fontsize
            if n_label_fontsize is None
            else float(n_label_fontsize)
        )
        for metric_idx, bars in enumerate(all_bars):
            top_positions = []
            for row_idx, value in enumerate(values[:, metric_idx]):
                if not np.isfinite(value):
                    top_positions.append(np.nan)
                    continue
                err_val = 0.0
                if err_arr is not None:
                    err_candidate = err_arr[row_idx, metric_idx]
                    err_val = 0.0 if not np.isfinite(err_candidate) else float(err_candidate)
                top_positions.append(float(value) + err_val)
            _annotate_metric_bars(
                ax,
                bars,
                values=values[:, metric_idx],
                counts=counts[:, metric_idx],
                fontsize=resolved_annotation_fontsize,
                tops=top_positions,
                value_fontsize=resolved_value_label_fontsize,
                count_fontsize=resolved_count_label_fontsize,
                value_rotation=float(value_label_rotation),
                value_ha="center",
                value_va="bottom",
                count_y=count_y,
                count_transform=count_transform,
                count_prefix=count_prefix,
                count_rotation=count_label_rotation,
                count_ha="center",
                count_va="top",
                value_y_min=y_min + 0.01,
            )
        if n_label is not None:
            left_bar_edge = x[0] - (total_bar_width / 2.0)
            ax.text(
                left_bar_edge - (0.12 * group_spacing),
                count_y,
                n_label,
                ha="right",
                va="top",
                fontsize=resolved_n_label_fontsize,
                transform=count_transform,
                clip_on=False,
            )
        ax.set_xlim(
            x[0] - (total_bar_width / 2.0) - (0.35 * group_spacing),
            x[-1] + (total_bar_width / 2.0) + (0.20 * group_spacing),
        )
        if legend_below:
            fig.legend(
                loc="lower center",
                ncol=legend_ncol if legend_ncol is not None else max(1, min(len(metric_labels), 4)),
                frameon=False,
                bbox_to_anchor=(0.5, float(legend_bbox_y)),
                fontsize=legend_fontsize,
            )
            fig.subplots_adjust(left=0.08, right=0.985, bottom=float(legend_bottom), top=0.94)
        else:
            legend = ax.legend(frameon=True, ncol=2)
            legend.get_frame().set_alpha(1.0)
            legend.get_frame().set_edgecolor("0.4")
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
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
    group_gap_scale: float = 1.0,
    count_label_y: float = -0.06,
    count_values_only: bool = False,
    n_label: Optional[str] = None,
    x_tick_pad: float = 28.0,
    value_label_fontsize: Optional[float] = None,
    count_label_fontsize: Optional[float] = None,
    n_label_fontsize: Optional[float] = None,
    y_tick_fontsize: Optional[float] = None,
    category_label_fontsize: Optional[float] = None,
    y_label_fontsize: Optional[float] = None,
    panel_title_fontsize: Optional[float] = None,
    panel_title_pad: float = 4.0,
    legend_fontsize: Optional[float] = None,
    legend_bbox_y: float = 0.005,
    legend_bottom: float = 0.27,
    value_label_rotation: float = 0.0,
    value_label_rotation_mode: str = "anchor",
    value_label_offset_scale: float = 0.01,
    count_label_rotation: float = 0.0,
    x_tick_length: float = 5.0,
    wrap_category_labels: bool = False,
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
        total_bar_width = 0.9
        group_spacing = total_bar_width + ((1.0 - total_bar_width) * float(group_gap_scale))
        x = np.arange(len(categories), dtype=float) * group_spacing
        width = total_bar_width / float(len(model_labels))
        count_transform = axes[-1].get_xaxis_transform()
        count_prefix = "" if count_values_only else "N="
        annotation_fontsize = max(fontsize - 5, 7)
        resolved_value_label_fontsize = (
            annotation_fontsize if value_label_fontsize is None else float(value_label_fontsize)
        )
        resolved_count_label_fontsize = (
            annotation_fontsize if count_label_fontsize is None else float(count_label_fontsize)
        )
        resolved_n_label_fontsize = fontsize if n_label_fontsize is None else float(n_label_fontsize)
        resolved_y_tick_fontsize = max(fontsize - 1, 8) if y_tick_fontsize is None else float(y_tick_fontsize)
        resolved_category_label_fontsize = (
            max(fontsize - 1, 8) if category_label_fontsize is None else float(category_label_fontsize)
        )
        resolved_y_label_fontsize = fontsize + 10 if y_label_fontsize is None else float(y_label_fontsize)
        resolved_panel_title_fontsize = (
            fontsize if panel_title_fontsize is None else float(panel_title_fontsize)
        )
        resolved_legend_fontsize = fontsize if legend_fontsize is None else float(legend_fontsize)
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
                top_positions = []
                for row_idx, value in enumerate(values[:, model_idx]):
                    if not np.isfinite(value):
                        top_positions.append(np.nan)
                        continue
                    err_val = 0.0
                    if errors is not None:
                        err_candidate = errors[row_idx, model_idx]
                        err_val = 0.0 if not np.isfinite(err_candidate) else float(err_candidate)
                    top_positions.append(float(value) + err_val)
                count_col = (
                    np.full(values.shape[0], np.nan, dtype=float)
                    if counts is None else counts[:, model_idx]
                )
                _annotate_metric_bars(
                    ax,
                    bars,
                    values=values[:, model_idx],
                    counts=count_col,
                    fontsize=annotation_fontsize,
                    value_fontsize=resolved_value_label_fontsize,
                    count_fontsize=resolved_count_label_fontsize,
                    value_rotation=float(value_label_rotation),
                    value_rotation_mode=str(value_label_rotation_mode),
                    value_offset_scale=float(value_label_offset_scale),
                    value_ha="center",
                    tops=top_positions,
                    count_y=count_label_y,
                    count_transform=ax.get_xaxis_transform(),
                    count_prefix=count_prefix,
                    count_rotation=float(count_label_rotation),
                    count_ha="center",
                    count_va="top",
                    value_y_min=0.01,
                )
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size > 0:
                ymax = max(ymax, float(np.max(finite_vals)) + 0.12)
            if errors is not None:
                finite_combined = (values + errors)[np.isfinite(values + errors)]
                if finite_combined.size > 0:
                    ymax = max(ymax, float(np.max(finite_combined)) + 0.12)
            y_min = -0.035
            ax.set_ylim(y_min, ymax)
            ax.set_yticks(np.arange(0.0, ymax + 0.001, 0.2))
            ax.tick_params(
                axis="y",
                which="major",
                length=5,
                width=1.0,
                labelleft=True,
                labelsize=resolved_y_tick_fontsize,
            )
            ax.tick_params(
                axis="x",
                which="major",
                length=float(x_tick_length),
                width=1.0,
                pad=float(x_tick_pad),
            )
            ax.set_ylabel(panel["ylabel"], fontsize=resolved_y_label_fontsize, labelpad=12)
            title = str(panel.get("title", ""))
            if title != "":
                ax.set_title(
                    title,
                    loc="left",
                    pad=float(panel_title_pad),
                    fontsize=resolved_panel_title_fontsize,
                )
        if n_label is not None:
            left_bar_edge = x[0] - (total_bar_width / 2.0)
            axes[-1].text(
                left_bar_edge - (0.12 * group_spacing),
                count_label_y,
                n_label,
                ha="right",
                va="top",
                fontsize=resolved_n_label_fontsize,
                transform=count_transform,
                clip_on=False,
            )
        axes[-1].set_xlim(
            x[0] - (total_bar_width / 2.0) - (0.35 * group_spacing),
            x[-1] + (total_bar_width / 2.0) + (0.20 * group_spacing),
        )
        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            model_labels,
            frameon=False,
            ncol=max(1, min(len(model_labels), 4)),
            loc="lower center",
            bbox_to_anchor=(0.5, float(legend_bbox_y)),
            fontsize=resolved_legend_fontsize,
        )
        category_labels = (
            _format_landcover_labels_wrapped(categories)
            if wrap_category_labels
            else _format_landcover_labels(categories)
        )
        axes[-1].set_xticks(x, category_labels)
        axes[-1].tick_params(axis="x", labelsize=resolved_category_label_fontsize)
        fig.subplots_adjust(left=0.08, right=0.985, bottom=float(legend_bottom), top=0.94, hspace=0.34)
        _save_figure_outputs(fig, save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
