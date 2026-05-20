import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import earthaccess
import pandas as pd
import xarray as xr
from earthaccess.exceptions import DownloadFailure


DEFAULT_GRID_PATH = '/scratch/users/trobinet/long_lfmc/final_lfmc/grid/epsg5070_500m_westUS_grid.nc4'
DEFAULT_OUTPUT_ROOT = '/scratch/users/trobinet/long_lfmc/final_lfmc/modis/modis_earthaccess'
DEFAULT_DOWNLOAD_THREADS = 8
DEFAULT_MAX_DOWNLOAD_ROUNDS = 3
MISSING_GRANULES_MANIFEST = '_modis_missing_granules.json'
TILES_V = [4, 4, 4, 4, 5, 5, 5, 5, 6, 6]
TILES_H = [8, 9, 10, 11, 7, 8, 9, 10, 8, 9]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--grid_path', type=str, default=DEFAULT_GRID_PATH)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--download_threads', type=int, default=DEFAULT_DOWNLOAD_THREADS)
    parser.add_argument('--max_download_rounds', type=int, default=DEFAULT_MAX_DOWNLOAD_ROUNDS)
    return parser.parse_args()


def expected_filenames_for_month(month_days, short_name):
    expected = {}
    for date in month_days:
        date_tag = f"A{date.strftime('%Y%j')}"
        for v, h in zip(TILES_V, TILES_H):
            tile_tag = f'h{h:02d}v{v:02d}'
            fname = f'{short_name}.{date_tag}.{tile_tag}.061.hdf'
            expected[fname] = (date_tag, tile_tag)
    return expected


def iter_request_months(start_date: pd.Timestamp, end_date: pd.Timestamp):
    month_starts = pd.date_range(
        start=start_date.replace(day=1),
        end=end_date.replace(day=1),
        freq='MS',
    )
    for month_start in month_starts:
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        this_start = max(month_start.normalize(), start_date)
        this_end = min(month_end, end_date)
        if this_end < this_start:
            continue
        yield this_start, this_end


def link_filename(url: str) -> str:
    return Path(urlparse(url).path).name


def modis_file_key(name: str):
    parts = Path(name).name.split('.')
    if len(parts) < 5:
        return None
    short_name, date_tag, tile_tag, version = parts[:4]
    if short_name not in {'MCD43A4', 'MCD43A2'}:
        return None
    if not date_tag.startswith('A') or len(date_tag) != 8:
        return None
    if not tile_tag.startswith('h') or 'v' not in tile_tag:
        return None
    return short_name, date_tag, tile_tag, version


def expected_file_keys_for_month(month_days, short_name):
    expected = {}
    for fname, metadata in expected_filenames_for_month(month_days, short_name).items():
        key = modis_file_key(fname)
        if key is None:
            raise ValueError(f'Could not parse expected MODIS filename: {fname}')
        expected[key] = metadata
    return expected


def existing_file_keys(out_dir: Path, short_name: str):
    keys = set()
    for path in out_dir.glob(f'{short_name}*.hdf'):
        key = modis_file_key(path.name)
        if key is not None:
            keys.add(key)
    return keys


def missing_manifest_path(output_root: Path | str) -> Path:
    return Path(output_root) / MISSING_GRANULES_MANIFEST


def key_to_dict(key, reason=None):
    short_name, date_tag, tile_tag, version = key
    record = {
        'short_name': short_name,
        'date_tag': date_tag,
        'tile_tag': tile_tag,
        'version': version,
    }
    if reason is not None:
        record['reason'] = reason
    return record


def dict_to_key(record):
    return (
        record['short_name'],
        record['date_tag'],
        record['tile_tag'],
        record['version'],
    )


def load_missing_granules_manifest(output_root: Path | str):
    manifest = missing_manifest_path(output_root)
    if not manifest.exists():
        return {}
    with open(manifest, 'r') as f:
        payload = json.load(f)
    loaded = {}
    for month, records in payload.items():
        loaded[month] = [dict(record) for record in records]
    return loaded


def write_missing_granules_manifest(output_root: Path | str, missing_by_month: dict):
    manifest = missing_manifest_path(output_root)
    if not missing_by_month:
        if manifest.exists():
            manifest.unlink()
            print(f'Removed resolved missing-granule manifest: {manifest}')
        return
    serializable = {}
    for month, records in sorted(missing_by_month.items()):
        serializable[month] = sorted(
            records,
            key=lambda record: (
                record['short_name'],
                record['date_tag'],
                record['tile_tag'],
                record['version'],
            ),
        )
    with open(manifest, 'w') as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
    print(f'Wrote missing-granule manifest: {manifest}')


def compute_missing_expected(out_dir: Path, short_name: str, month_days):
    expected = expected_file_keys_for_month(month_days, short_name)
    existing = existing_file_keys(out_dir, short_name)
    missing_expected = set(expected) - existing
    return expected, existing, missing_expected


def filter_missing_links(out_dir: Path, links, short_name: str, month_days, skip_existing: bool):
    if not skip_existing:
        return links, None, None
    expected, existing, missing_expected = compute_missing_expected(out_dir, short_name, month_days)
    if not missing_expected:
        print(f'All expected {short_name} files already exist under {out_dir}; skipping download')
        return [], existing, missing_expected
    filtered = []
    link_keys = {}
    for url in links:
        key = modis_file_key(link_filename(url))
        if key is not None:
            link_keys[url] = key
            if key in missing_expected:
                filtered.append(url)
    print(f'{short_name}: existing={len(existing)} missing_expected={len(missing_expected)} download_candidates={len(filtered)}')
    if links and not filtered:
        sample_names = sorted({link_filename(url) for url in links})[:5]
        sample_missing = [
            f'{short}.{date}.{tile}.{version}'
            for short, date, tile, version in sorted(missing_expected)[:5]
        ]
        sample_link_keys = [link_keys[url] for url in list(link_keys)[:5]]
        print(
            f'Found {len(links)} {short_name} links but none matched the missing expected filenames. '
            f'sample_link_names={sample_names} sample_link_keys={sample_link_keys} '
            f'sample_missing_expected={sample_missing}'
        )
    return filtered, existing, missing_expected


def _expected_date_tile_pairs(month_days):
    for date in month_days:
        date_tag = f"A{date.strftime('%Y%j')}"
        for v, h in zip(TILES_V, TILES_H):
            tile_tag = f'h{h:02d}v{v:02d}'
            yield date_tag, tile_tag


def _result_links_by_key(results, short_name: str):
    links_by_key = {}
    for res in results:
        for url in res.data_links():
            key = modis_file_key(link_filename(url))
            if key is None or key[0] != short_name:
                continue
            links_by_key.setdefault((key[1], key[2]), url)
    return links_by_key


def _search_exact_granule_link(short_name: str, date_tag: str, tile_tag: str):
    results = earthaccess.search_data(
        short_name=short_name,
        version='061',
        granule_name=f'{short_name}.{date_tag}.{tile_tag}.061.*',
        downloadable=True,
    )
    links_by_key = _result_links_by_key(results, short_name)
    return links_by_key.get((date_tag, tile_tag))


def collect_links(results, month_days, short_name: str, retry_missing_exact: bool = True):
    links = []
    exact_recovered = 0
    links_by_key = _result_links_by_key(results, short_name)
    expected_pairs = list(_expected_date_tile_pairs(month_days))
    desired = len(expected_pairs)
    for date_tag, tile_tag in expected_pairs:
        url = links_by_key.get((date_tag, tile_tag))
        if url is None and retry_missing_exact:
            url = _search_exact_granule_link(short_name, date_tag, tile_tag)
            if url is not None:
                exact_recovered += 1
        if url is not None:
            links.append(url)
        else:
            print(f'Warning: No {short_name} link found for {date_tag} {tile_tag}')
    if exact_recovered:
        print(
            f'Recovered {exact_recovered} {short_name} links with exact granule-name searches '
            f'after broad CMR search misses.'
        )
    print(f'Found {len(links)} {short_name} links, out of {desired} desired.')
    return links


def collect_links_for_range(short_name: str, this_start: pd.Timestamp, this_end: pd.Timestamp, bounding_box):
    results = earthaccess.search_data(
        short_name=short_name,
        version='061',
        temporal=(this_start, this_end),
        bounding_box=bounding_box,
        downloadable=True,
    )
    month_days = pd.date_range(start=this_start, end=this_end, freq='D')
    return collect_links(results, month_days, short_name)


def download_links_with_retries(
    out_dir: Path,
    short_name: str,
    this_start: pd.Timestamp,
    this_end: pd.Timestamp,
    bounding_box,
    initial_links,
    skip_existing: bool,
    download_threads: int,
    max_download_rounds: int,
):
    month_days = pd.date_range(start=this_start, end=this_end, freq='D')
    links = initial_links
    last_error = None
    for round_idx in range(1, max_download_rounds + 1):
        filtered, existing, missing_expected = filter_missing_links(
            out_dir, links, short_name, month_days, skip_existing
        )
        if missing_expected is not None:
            print(
                f'{short_name}: month={this_start:%Y-%m} '
                f'round={round_idx}/{max_download_rounds} '
                f'existing={len(existing)} missing_expected={len(missing_expected)} '
                f'download_candidates={len(filtered)}'
            )
        if not filtered:
            _, _, missing_after = compute_missing_expected(out_dir, short_name, month_days)
            if not missing_after:
                print(f'{short_name}: month={this_start:%Y-%m} already complete')
                return []
            reason = str(last_error) if last_error is not None else 'download_candidates_exhausted'
            print(
                f'{short_name}: month={this_start:%Y-%m} still missing {len(missing_after)} logical granules '
                f'but has no download candidates; carrying forward as NaN fallbacks; '
                f'sample_missing={sorted(missing_after)[:10]}'
            )
            return [
                key_to_dict(key, reason=reason)
                for key in sorted(missing_after)
            ]
        try:
            earthaccess.download(filtered, str(out_dir), threads=download_threads, show_progress=True)
            last_error = None
        except DownloadFailure as exc:
            last_error = exc
            print(
                f'{short_name}: month={this_start:%Y-%m} '
                f'round={round_idx}/{max_download_rounds} download hit DownloadFailure: {exc}'
            )

        _, _, missing_after = compute_missing_expected(out_dir, short_name, month_days)
        print(
            f'{short_name}: month={this_start:%Y-%m} '
            f'after round {round_idx}/{max_download_rounds} remaining_missing={len(missing_after)}'
        )
        if not missing_after:
            print(f'{short_name}: month={this_start:%Y-%m} download coverage complete')
            return []
        if round_idx < max_download_rounds:
            print(
                f'{short_name}: month={this_start:%Y-%m} re-querying missing logical granules '
                f'for round {round_idx + 1}/{max_download_rounds}'
            )
            links = collect_links_for_range(short_name, this_start, this_end, bounding_box)

    _, _, missing_final = compute_missing_expected(out_dir, short_name, month_days)
    if missing_final:
        reason = str(last_error) if last_error is not None else 'download_candidates_exhausted'
        print(
            f'{short_name}: month={this_start:%Y-%m} still missing {len(missing_final)} logical granules '
            f'after {max_download_rounds} download rounds; sample_missing={sorted(missing_final)[:10]}'
        )
        return [
            key_to_dict(key, reason=reason)
            for key in sorted(missing_final)
        ]
    return []


def main():
    args = parse_args()
    start_date = args.start_date
    end_date = args.end_date
    start_date_pd = pd.to_datetime(start_date)
    end_date_pd = pd.to_datetime(end_date)
    print('start date:', start_date)
    print('end date:', end_date)
    print('output_root:', args.output_root)
    print('skip_existing:', args.skip_existing)
    print('download_threads:', args.download_threads)
    print('max_download_rounds:', args.max_download_rounds)

    print('logging in')
    earthaccess.login()

    print('searching for data')
    grid = xr.open_dataset(args.grid_path)
    min_lat = grid['lat'].min().values - 1.0
    max_lat = grid['lat'].max().values + 1.0
    min_lon = grid['lon'].min().values - 1.0
    max_lon = grid['lon'].max().values + 1.0
    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    print('bounding box:', bounding_box)
    missing_by_month = {}
    write_missing_granules_manifest(args.output_root, missing_by_month)

    for this_start, this_end in iter_request_months(start_date_pd, end_date_pd):
        out_dir = Path(args.output_root) / this_end.strftime('%Y')
        month_label = this_start.strftime('%Y-%m')
        print(f'Downloading MODIS data from {this_start.date()} to {this_end.date()}')
        month_days = pd.date_range(start=this_start, end=this_end, freq='D')
        out_dir.mkdir(parents=True, exist_ok=True)

        data_links = collect_links_for_range('MCD43A4', this_start, this_end, bounding_box)
        quality_links = collect_links_for_range('MCD43A2', this_start, this_end, bounding_box)

        data_candidates, _, data_missing = filter_missing_links(
            out_dir, data_links, 'MCD43A4', month_days, args.skip_existing
        )
        quality_candidates, _, quality_missing = filter_missing_links(
            out_dir, quality_links, 'MCD43A2', month_days, args.skip_existing
        )
        if not data_candidates and not quality_candidates and not data_missing and not quality_missing:
            print('No files to download for this month, skipping download step.')
            continue
        data_missing_records = download_links_with_retries(
            out_dir=out_dir,
            short_name='MCD43A4',
            this_start=this_start,
            this_end=this_end,
            bounding_box=bounding_box,
            initial_links=data_links,
            skip_existing=args.skip_existing,
            download_threads=args.download_threads,
            max_download_rounds=args.max_download_rounds,
        )
        quality_missing_records = download_links_with_retries(
            out_dir=out_dir,
            short_name='MCD43A2',
            this_start=this_start,
            this_end=this_end,
            bounding_box=bounding_box,
            initial_links=quality_links,
            skip_existing=args.skip_existing,
            download_threads=args.download_threads,
            max_download_rounds=args.max_download_rounds,
        )
        data_missing_records = data_missing_records or []
        quality_missing_records = quality_missing_records or []
        unresolved = data_missing_records + quality_missing_records
        if unresolved:
            print(
                f'Month {month_label}: carrying forward {len(unresolved)} missing logical granules as NaN fallbacks'
            )
            missing_by_month[month_label] = unresolved
        elif month_label in missing_by_month:
            missing_by_month.pop(month_label, None)
        write_missing_granules_manifest(args.output_root, missing_by_month)


if __name__ == "__main__":
    main()
