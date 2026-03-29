#!/usr/bin/env python3

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import yaml


QUALITY_FLAG_VALUES = {"final": 0, "low_latency": 1}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    return value


def default_source_registry_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'source_registry.yaml')


def load_source_registry(registry_path: Optional[str] = None) -> Dict[str, object]:
    path = registry_path or default_source_registry_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing source registry: {path}')
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Source registry must parse to a dict: {path}')
    cfg['_registry_path'] = path
    return cfg


def normalize_product_tier(product_tier: str) -> str:
    tier = str(product_tier).strip().lower()
    if tier not in QUALITY_FLAG_VALUES:
        raise ValueError(
            f'product tier must be one of {sorted(QUALITY_FLAG_VALUES.keys())}; got {product_tier!r}'
        )
    return tier


def _require_path(path: str, label: str) -> str:
    if path in {None, '', 'None'}:
        raise ValueError(f'Missing path for {label}')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing {label}: {path}')
    return path


def _get_time_bounds(zarr_path: str) -> Dict[str, object]:
    ds = xr.open_zarr(zarr_path, consolidated=False)
    times = pd.to_datetime(ds['time'].values)
    return {
        'start_date': pd.Timestamp(times.min()).normalize(),
        'end_date': pd.Timestamp(times.max()).normalize(),
    }


def _merge_daymet_datasets(daymet_ds: xr.Dataset, anomaly_ds: Optional[xr.Dataset]) -> xr.Dataset:
    if anomaly_ds is None:
        return daymet_ds
    anomaly_ds = anomaly_ds.reindex(time=daymet_ds["time"])
    return xr.concat(
        [daymet_ds, anomaly_ds],
        dim="variable",
        compat="override",
        coords="minimal",
        join="exact",
    )


def _get_year_values(zarr_path: str, coord_name: str = 'year') -> List[int]:
    ds = xr.open_zarr(zarr_path)
    vals = ds[coord_name].values
    years = []
    for val in vals:
        if isinstance(val, (np.datetime64, pd.Timestamp)):
            years.append(int(pd.Timestamp(val).year))
        else:
            years.append(int(val))
    return sorted(set(years))


def _resolve_nlcd_source_year(output_year: int, available_years: Sequence[int], nlcd_mode: str) -> int:
    available_years = sorted(int(year) for year in available_years)
    if nlcd_mode == 'same_year_required':
        if output_year not in available_years:
            raise ValueError(
                f'NLCD year {output_year} is required for final mode but not available; '
                f'available years: {available_years[:3]}...{available_years[-3:]}'
            )
        return int(output_year)
    if nlcd_mode == 'latest_available_year':
        prior_years = [year for year in available_years if year <= output_year]
        if len(prior_years) > 0:
            return int(prior_years[-1])
        return int(available_years[0])
    raise ValueError(f'Unsupported nlcd_mode: {nlcd_mode}')


def resolve_inference_sources(
    registry_path: Optional[str],
    product_tier: str,
    requested_start_date,
    requested_end_date,
    output_years: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    registry = load_source_registry(registry_path)
    tier = normalize_product_tier(product_tier)
    tier_cfg = registry.get('tiers', {}).get(tier)
    if not isinstance(tier_cfg, dict):
        raise ValueError(f'Missing tier config for {tier!r} in {registry["_registry_path"]}')
    sources = registry.get('sources', {})

    modis_path = _require_path(sources.get('modis', {}).get('path'), 'MODIS zarr')
    static_path = _require_path(sources.get('static', {}).get('path'), 'static dataset')
    soils_path = _require_path(sources.get('soils', {}).get('path'), 'soils dataset')
    canopy_height_path = _require_path(
        sources.get('canopy_height', {}).get('path'),
        'canopy height dataset',
    )
    nlcd_path = _require_path(sources.get('nlcd', {}).get('annual_path'), 'annual NLCD zarr')
    archive_daymet_path = _require_path(
        sources.get('daymet', {}).get('archive_path'),
        'archive Daymet zarr',
    )
    anomaly_daymet_path = _require_path(
        sources.get('daymet', {}).get('anomalies_path'),
        'Daymet anomaly zarr',
    )
    monthly_latency_daymet_path = sources.get('daymet', {}).get('monthly_latency_path')
    low_latency_climate_path = sources.get('climate_low_latency', {}).get('path')

    start_date = pd.Timestamp(requested_start_date).normalize()
    end_date = pd.Timestamp(requested_end_date).normalize()
    archive_daymet_bounds = _get_time_bounds(archive_daymet_path)
    daymet_mode = str(tier_cfg['daymet_mode'])
    daymet_paths: List[str]
    if daymet_mode == 'archive_only':
        daymet_paths = [archive_daymet_path]
    elif daymet_mode == 'archive_then_monthly_latency':
        if end_date <= archive_daymet_bounds['end_date']:
            daymet_paths = [archive_daymet_path]
        else:
            latency_path = _require_path(
                monthly_latency_daymet_path,
                'monthly-latency Daymet zarr',
            )
            if start_date <= archive_daymet_bounds['end_date']:
                daymet_paths = [archive_daymet_path, latency_path]
            else:
                daymet_paths = [latency_path]
    elif daymet_mode == 'archive_then_low_latency_climate':
        if end_date <= archive_daymet_bounds['end_date']:
            daymet_paths = [archive_daymet_path]
        else:
            latency_path = _require_path(
                low_latency_climate_path,
                'low-latency climate zarr',
            )
            if start_date <= archive_daymet_bounds['end_date']:
                daymet_paths = [archive_daymet_path, latency_path]
            else:
                daymet_paths = [latency_path]
    else:
        raise ValueError(f'Unsupported daymet_mode: {daymet_mode}')

    nlcd_output_year_to_source_year = None
    if output_years is not None:
        available_nlcd_years = _get_year_values(nlcd_path, coord_name='year')
        nlcd_output_year_to_source_year = {
            str(int(output_year)): _resolve_nlcd_source_year(
                int(output_year),
                available_nlcd_years,
                str(tier_cfg['nlcd_mode']),
            )
            for output_year in sorted({int(year) for year in output_years})
        }

    return {
        'registry_path': registry['_registry_path'],
        'tier': tier,
        'quality_flag_value': int(tier_cfg['quality_flag']),
        'requested_start_date': str(start_date.date()),
        'requested_end_date': str(end_date.date()),
        'modis_path': modis_path,
        'static_path': static_path,
        'soils_path': soils_path,
        'canopy_height_path': canopy_height_path,
        'landcover_path': nlcd_path,
        'archive_daymet_path': archive_daymet_path,
        'anomaly_daymet_path': anomaly_daymet_path,
        'monthly_latency_daymet_path': monthly_latency_daymet_path,
        'low_latency_climate_path': low_latency_climate_path,
        'daymet_mode': daymet_mode,
        'daymet_paths': list(daymet_paths),
        'archive_daymet_bounds': {
            'start_date': str(archive_daymet_bounds['start_date'].date()),
            'end_date': str(archive_daymet_bounds['end_date'].date()),
        },
        'nlcd_mode': str(tier_cfg['nlcd_mode']),
        'nlcd_output_year_to_source_year': nlcd_output_year_to_source_year,
        'production_zarr_path': sources.get('production', {}).get('zarr_path'),
        'promotion_metadata_dir': sources.get('production', {}).get('metadata_dir'),
    }


def open_inference_datasets_from_resolution(source_resolution: Dict[str, object]) -> Dict[str, xr.Dataset]:
    daymet_paths = [str(path) for path in source_resolution.get('daymet_paths', [])]
    if len(daymet_paths) == 0:
        raise ValueError('source_resolution is missing daymet_paths')
    daymet_parts = [xr.open_zarr(path, consolidated=False) for path in daymet_paths]
    if len(daymet_parts) == 1:
        daymet_ds = daymet_parts[0]
    else:
        daymet_ds = xr.concat(daymet_parts, dim='time').sortby('time')
        time_index = pd.DatetimeIndex(pd.to_datetime(daymet_ds['time'].values))
        if time_index.has_duplicates:
            keep_mask = ~time_index.duplicated(keep='last')
            daymet_ds = daymet_ds.isel(time=np.where(keep_mask)[0])
    anomaly_daymet_ds = xr.open_zarr(
        str(source_resolution['anomaly_daymet_path']),
        consolidated=False,
    )
    daymet_ds = _merge_daymet_datasets(daymet_ds, anomaly_daymet_ds)
    return {
        'daymet': daymet_ds,
        'modis': xr.open_zarr(str(source_resolution['modis_path'])),
        'static': xr.open_dataset(str(source_resolution['static_path'])),
        'soils': xr.open_dataset(str(source_resolution['soils_path'])),
        'canopy_height': xr.open_dataset(str(source_resolution['canopy_height_path'])),
        'landcover_frac': xr.open_zarr(str(source_resolution['landcover_path'])),
    }


def json_safe_source_resolution(source_resolution: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if source_resolution is None:
        return None
    return _json_safe(source_resolution)
