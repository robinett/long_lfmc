import { useEffect, useRef, useState } from "react";
import proj4 from "proj4";
import Map from "ol/Map";
import View from "ol/View";
import Feature from "ol/Feature";
import Point from "ol/geom/Point";
import Polygon from "ol/geom/Polygon";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import XYZ from "ol/source/XYZ";
import TileGrid from "ol/tilegrid/TileGrid";
import { ScaleLine, defaults as defaultControls } from "ol/control";
import { getCenter } from "ol/extent";
import { get as getProjection, transform as transformCoordinate } from "ol/proj";
import { register } from "ol/proj/proj4";
import { Circle as CircleStyle, Fill, Stroke, Style } from "ol/style";

const EPSG5070_DEF =
  "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +type=crs";

proj4.defs("EPSG:5070", EPSG5070_DEF);
register(proj4);

const DEFAULT_API_BASE_URL = "https://long-lfmc-api.onrender.com";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, "");
const MAX_DOWNLOAD_YEARS = 3;
const DEFAULT_DATASET_KEY = "modis";
const SENTINEL_DATASET_KEY = "sentinel1";
const DEFAULT_TIMESERIES_MODE = "mean";
const TIMESERIES_WINDOW_DAYS = 90;
const SENTINEL_DATE_TOLERANCE_DAYS = 20;
const GLOBAL_DATE_START = "2001-01-01";
const PRODUCT_DOC_URL = "https://docs.google.com/document/d/1b8n4UQ1XYDd_llw2nO0yPj-pN8Ar0BUjXGQiM-G6CvY/edit?usp=sharing";
const DETAIL_TABS = [
  ["timeseries", "View timeseries"],
  ["download", "Download data"],
];
const DATASET_SUMMARIES = {
  [DEFAULT_DATASET_KEY]: "500m resolution, daily, begins 2001",
  [SENTINEL_DATASET_KEY]: "250m resolution, 15-day temporal resolution, begins 2016",
};

function apiUrl(pathAndQuery) {
  const normalizedPath = pathAndQuery.startsWith("/") ? pathAndQuery : `/${pathAndQuery}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

function formatValue(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "NA";
  }
  return Number(value).toFixed(digits);
}

function formatMetricValue(value, digits = 1, supported = true) {
  if (!supported) {
    return "Unavailable";
  }
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "No data";
  }
  return Number(value).toFixed(digits);
}

function formatLabel(label) {
  if (!label) {
    return "Unknown";
  }
  return String(label)
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function legendGradient(layerConfig) {
  const stops = layerConfig?.stops ?? [];
  const palette = layerConfig?.palette ?? [];
  if (!stops.length || !palette.length) {
    return "linear-gradient(to right, #6d271d 0%, #c9794a 24%, #d2c487 52%, #6d8e60 76%, #253b36 100%)";
  }
  const pieces = palette.map((color, idx) => {
    const pct = `${Math.round(Number(stops[idx]) * 100)}%`;
    return `rgb(${color.join(",")}) ${pct}`;
  });
  return `linear-gradient(to right, ${pieces.join(", ")})`;
}

function isAnomalyLayer(layerKey) {
  return layerKey === "anomaly";
}

function layerUnitLabel(layerConfig) {
  return layerConfig?.unit === "percent" ? "%" : (layerConfig?.unit ?? "");
}

function niceLegendTickStep(span, targetTickCount = 5) {
  const roughStep = span / Math.max(targetTickCount - 1, 1);
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(roughStep, 1e-6)));
  const candidates = [1, 2, 4, 5, 10].map((value) => value * magnitude);

  return candidates.reduce((best, candidate) => {
    const candidateDistance = Math.abs(candidate - roughStep);
    const bestDistance = Math.abs(best - roughStep);
    if (candidateDistance < bestDistance) {
      return candidate;
    }
    if (candidateDistance === bestDistance && candidate < best) {
      return candidate;
    }
    return best;
  }, candidates[0]);
}

function buildLegendTicks(layerConfig, layerKey) {
  if (!layerConfig) {
    return [];
  }

  const minValue = Number(layerConfig.min);
  const maxValue = Number(layerConfig.max);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || maxValue <= minValue) {
    return [];
  }

  const span = maxValue - minValue;
  const step = niceLegendTickStep(span);
  const unitLabel = layerUnitLabel(layerConfig);
  const tickValues = [minValue];
  const firstTick = Math.ceil((minValue - step * 0.001) / step) * step;
  const lastTick = maxValue + step * 0.001;
  const endpointTolerance = Math.max(Math.abs(step) * 1e-6, 1e-6);

  for (let value = firstTick; value <= lastTick; value += step) {
    const roundedValue = Number(value.toFixed(6));
    if (
      roundedValue <= minValue + endpointTolerance ||
      roundedValue >= maxValue - endpointTolerance
    ) {
      continue;
    }
    tickValues.push(roundedValue);
  }
  tickValues.push(maxValue);

  return tickValues.map((value) => {
    let label = `${formatValue(value, 0)}${unitLabel}`;
    let subLabel = "";
    if (isAnomalyLayer(layerKey)) {
      if (Math.abs(value - minValue) <= endpointTolerance) {
        subLabel = "Dry";
      } else if (Math.abs(value - maxValue) <= endpointTolerance) {
        subLabel = "Wet";
      }
    }

    return {
      label,
      subLabel,
      position: ((value - minValue) / span) * 100,
    };
  });
}

function niceTickStep(span, targetTickCount) {
  const roughStep = span / Math.max(targetTickCount - 1, 1);
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(roughStep, 1e-6)));
  const normalized = roughStep / magnitude;

  if (normalized <= 1) {
    return 1 * magnitude;
  }
  if (normalized <= 2) {
    return 2 * magnitude;
  }
  if (normalized <= 2.5) {
    return 2.5 * magnitude;
  }
  if (normalized <= 5) {
    return 5 * magnitude;
  }
  return 10 * magnitude;
}

function buildAxisTicks(minValue, maxValue, count = 4) {
  const span = maxValue - minValue;
  if (!Number.isFinite(span) || span <= 0) {
    return [{ value: minValue, label: formatValue(minValue, 0), fraction: 0.5 }];
  }

  const step = niceTickStep(span, count);
  const axisMin = Math.floor(minValue / step) * step;
  const axisMax = Math.ceil(maxValue / step) * step;
  const axisSpan = Math.max(axisMax - axisMin, step);
  const ticks = [];

  for (let value = axisMin; value <= axisMax + step * 0.001; value += step) {
    const roundedValue = Number(value.toFixed(6));
    const fraction = (axisMax - roundedValue) / axisSpan;
    ticks.push({
      value: roundedValue,
      label: formatValue(roundedValue, 0),
      fraction,
    });
  }

  return ticks;
}

function parseDateString(dateStr) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(dateStr));
  if (!match) {
    return null;
  }
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    return null;
  }
  return date;
}

function formatDateString(date) {
  return date.toISOString().slice(0, 10);
}

function dateDiffDays(left, right) {
  const leftDate = parseDateString(left);
  const rightDate = parseDateString(right);
  if (!leftDate || !rightDate) {
    return Number.POSITIVE_INFINITY;
  }
  return Math.round(Math.abs(leftDate - rightDate) / 86400000);
}

function shiftDateString(dateStr, amount, unit) {
  const date = parseDateString(dateStr);
  if (!date) {
    return dateStr;
  }
  if (unit === "month") {
    const originalDay = date.getUTCDate();
    date.setUTCDate(1);
    date.setUTCMonth(date.getUTCMonth() + amount);
    const lastDayOfTargetMonth = new Date(
      Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + 1, 0),
    ).getUTCDate();
    date.setUTCDate(Math.min(originalDay, lastDayOfTargetMonth));
  } else {
    date.setUTCDate(date.getUTCDate() + amount);
  }
  return formatDateString(date);
}

function buildDailyDateRange(startDate, endDate) {
  const start = parseDateString(startDate);
  const end = parseDateString(endDate);
  if (!start || !end || start > end) {
    return [];
  }
  const values = [];
  const cursor = new Date(start.getTime());
  while (cursor <= end) {
    values.push(formatDateString(cursor));
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return values;
}

function findDateIndex(dates, targetDate, direction = "nearest") {
  if (!dates.length) {
    return -1;
  }
  let low = 0;
  let high = dates.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (dates[mid] === targetDate) {
      return mid;
    }
    if (dates[mid] < targetDate) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  const beforeIdx = Math.max(0, high);
  const afterIdx = Math.min(dates.length - 1, low);
  if (direction === "backward") {
    return beforeIdx;
  }
  if (direction === "forward") {
    return afterIdx;
  }
  return dateDiffDays(dates[beforeIdx], targetDate) <= dateDiffDays(dates[afterIdx], targetDate)
    ? beforeIdx
    : afterIdx;
}

function surroundingDateIndices(dates, targetDate) {
  if (!dates.length || !targetDate) {
    return { beforeIdx: -1, afterIdx: -1, exactIdx: -1 };
  }
  let low = 0;
  let high = dates.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (dates[mid] === targetDate) {
      return { beforeIdx: mid, afterIdx: mid, exactIdx: mid };
    }
    if (dates[mid] < targetDate) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return {
    beforeIdx: high >= 0 ? high : -1,
    afterIdx: low < dates.length ? low : -1,
    exactIdx: -1,
  };
}

function maxDownloadEndDate(startDate) {
  const parsed = parseDateString(startDate);
  if (!parsed) {
    return null;
  }
  const targetYear = parsed.getUTCFullYear() + MAX_DOWNLOAD_YEARS;
  const month = parsed.getUTCMonth();
  const day = parsed.getUTCDate();
  const lastDayOfTargetMonth = new Date(Date.UTC(targetYear, month + 1, 0)).getUTCDate();
  return formatDateString(new Date(Date.UTC(targetYear, month, Math.min(day, lastDayOfTargetMonth))));
}

function isDownloadRangeWithinLimit(startDate, endDate) {
  const maxEnd = maxDownloadEndDate(startDate);
  return Boolean(maxEnd) && endDate <= maxEnd;
}

function minDateString(...values) {
  return values.filter(Boolean).sort()[0] ?? "";
}

function maxDateString(...values) {
  const sorted = values.filter(Boolean).sort();
  return sorted[sorted.length - 1] ?? "";
}

function createDownloadSite(startDate = "", endDate = "") {
  return {
    lat: "",
    lon: "",
    startDate,
    endDate,
  };
}

function defaultDownloadEndDate(dates, startDate) {
  return startDate
    ? minDateString(dates[dates.length - 1], maxDownloadEndDate(startDate))
    : dates[dates.length - 1] ?? "";
}

function clampDateString(value, minValue, maxValue) {
  if (!value) {
    return value;
  }
  let nextValue = value;
  if (minValue && nextValue < minValue) {
    nextValue = minValue;
  }
  if (maxValue && nextValue > maxValue) {
    nextValue = maxValue;
  }
  return nextValue;
}

function clampDownloadSiteDates(site, dates) {
  const datasetStart = dates[0] ?? "";
  const datasetEnd = dates[dates.length - 1] ?? "";
  const nextSite = { ...site };

  nextSite.startDate = clampDateString(nextSite.startDate, datasetStart, datasetEnd);
  if (nextSite.startDate && nextSite.endDate && nextSite.startDate > nextSite.endDate) {
    nextSite.endDate = nextSite.startDate;
  }

  const endMax = defaultDownloadEndDate(dates, nextSite.startDate);
  nextSite.endDate = clampDateString(nextSite.endDate, nextSite.startDate || datasetStart, endMax);
  return nextSite;
}

function configuredInitialDate(dates, initialDate) {
  if (!dates.length) {
    return "";
  }
  if (initialDate === "latest") {
    return dates[dates.length - 1];
  }
  return dates.includes(initialDate) ? initialDate : dates[dates.length - 1];
}

function startupViewResolutions(manifestPayload, extent, mapElement) {
  const baseResolutions = manifestPayload.tiles.view_resolutions ?? manifestPayload.tiles.resolutions ?? [];
  const mapPaddingPx = 48;
  const minimumFullExtentWidth = 320;
  const minimumFullExtentHeight = 360;
  const width = Math.max(mapElement?.clientWidth ?? 0, 1);
  const height = Math.max(mapElement?.clientHeight ?? 0, 1);
  const fitResolutions = [
    Math.max(
      (extent[2] - extent[0]) / Math.max(width - mapPaddingPx, 1),
      (extent[3] - extent[1]) / Math.max(height - mapPaddingPx, 1),
    ),
    Math.max(
      (extent[2] - extent[0]) / minimumFullExtentWidth,
      (extent[3] - extent[1]) / minimumFullExtentHeight,
    ),
  ].filter((resolution) => Number.isFinite(resolution) && resolution > 0);
  const resolutions = [...baseResolutions, ...fitResolutions]
    .map(Number)
    .filter((resolution) => Number.isFinite(resolution) && resolution > 0)
    .sort((a, b) => b - a);
  return resolutions.filter((resolution, index) => index === 0 || Math.abs(resolution - resolutions[index - 1]) > 1e-6);
}

function showDatePicker(event) {
  const input = event.currentTarget;
  if (typeof input.showPicker === "function") {
    input.showPicker();
  }
}

function preventManualDateEdit(event) {
  event.preventDefault();
}

function cellPolygonCoordinates(cellBounds) {
  return [[
    [cellBounds.west, cellBounds.south],
    [cellBounds.east, cellBounds.south],
    [cellBounds.east, cellBounds.north],
    [cellBounds.west, cellBounds.north],
    [cellBounds.west, cellBounds.south],
  ]];
}

function tileUrlForCoord(assetBaseUrl, tileTemplate, selectedDate, z, x, y) {
  const relpath = tileTemplate
    .replace("{date}", selectedDate)
    .replace("{z}", String(z))
    .replace("{x}", String(x))
    .replace("{y}", String(y));
  return `${assetBaseUrl.replace(/\/$/, "")}/${relpath}`;
}

function formatCoordinateInput(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "";
  }
  return Number(value).toFixed(4);
}

function parseCoordinateInput(value) {
  const parsed = Number(String(value).trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function seriesPath(points, xCoord, yCoord) {
  let lineStarted = false;
  const parts = [];
  for (const point of points) {
    if (point.value === null || point.value === undefined || Number.isNaN(point.value)) {
      lineStarted = false;
      continue;
    }
    parts.push(`${lineStarted ? "L" : "M"} ${xCoord(point.offset)} ${yCoord(Number(point.value))}`);
    lineStarted = true;
  }
  return parts.join(" ");
}

function buildTimeseriesGeometry(pointInfo, mode = DEFAULT_TIMESERIES_MODE) {
  if (!pointInfo?.timeseries) {
    return null;
  }

  const timeseries = pointInfo.timeseries;
  const dates = timeseries.dates ?? [];
  const offsets = timeseries.day_offsets ?? dates.map((_, idx) => idx);
  const isAnomalyMode = mode === "anomaly";
  const values = isAnomalyMode ? (timeseries.lfmc_anomaly ?? []) : (timeseries.lfmc_ens_mean ?? []);
  const stds = isAnomalyMode ? [] : (timeseries.lfmc_ens_std ?? []);
  const climatologyValues = timeseries.lfmc_climatology_mean ?? [];
  const currentPoints = dates.map((date, idx) => ({
    date,
    offset: Number(offsets[idx]),
    value: values[idx] === null || values[idx] === undefined ? null : Number(values[idx]),
    std: stds[idx] === null || stds[idx] === undefined ? null : Number(stds[idx]),
    year: date.slice(0, 4),
    current: true,
  }));
  const historicalSeries = (timeseries.historical_windows ?? [])
    .map((window) => {
      const windowValues = isAnomalyMode ? (window.lfmc_anomaly ?? []) : (window.lfmc_ens_mean ?? []);
      const windowDates = window.dates ?? [];
      const windowOffsets = window.day_offsets ?? windowDates.map((_, idx) => idx);
      return {
        year: window.year,
        points: windowDates.map((date, idx) => ({
          date,
          offset: Number(windowOffsets[idx]),
          value: windowValues[idx] === null || windowValues[idx] === undefined ? null : Number(windowValues[idx]),
          year: String(window.year ?? date.slice(0, 4)),
          current: false,
        })),
      };
    })
    .filter((series) => series.points.filter((point) => Number.isFinite(point.value)).length >= 2);
  const seasonalPoints = isAnomalyMode
    ? []
    : dates.map((date, idx) => ({
        date,
        offset: Number(offsets[idx]),
        value: climatologyValues[idx] === null || climatologyValues[idx] === undefined
          ? null
          : Number(climatologyValues[idx]),
        year: "Seasonal cycle",
        current: false,
      }));
  const validSeasonalPoints = seasonalPoints.filter((point) => Number.isFinite(point.value));

  const validCurrentPoints = currentPoints.filter((point) => Number.isFinite(point.value));
  if (validCurrentPoints.length < 2) {
    return null;
  }

  const width = 352;
  const height = 210;
  const padding = { left: 48, right: 24, top: 18, bottom: 38 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const validBandPoints = currentPoints.filter(
    (point) => !isAnomalyMode && Number.isFinite(point.value) && Number.isFinite(point.std),
  );
  const allYValues = [
    ...validCurrentPoints.map((point) => point.value),
    ...validSeasonalPoints.map((point) => point.value),
    ...validBandPoints.flatMap((point) => [point.value - point.std, point.value + point.std]),
    ...historicalSeries.flatMap((series) =>
      series.points.filter((point) => Number.isFinite(point.value)).map((point) => point.value),
    ),
  ];
  const dataYMin = isAnomalyMode ? Math.min(...allYValues, 0) : Math.min(...allYValues);
  const dataYMax = isAnomalyMode ? Math.max(...allYValues, 0) : Math.max(...allYValues);
  const yTicks = buildAxisTicks(dataYMin, dataYMax);
  const yMin = yTicks[0].value;
  const yMax = yTicks[yTicks.length - 1].value;
  const ySpan = Math.max(yMax - yMin, 1e-6);
  const xCoord = (offset) => padding.left + (Math.max(0, Math.min(TIMESERIES_WINDOW_DAYS - 1, offset)) / (TIMESERIES_WINDOW_DAYS - 1)) * innerWidth;
  const yCoord = (value) => padding.top + ((yMax - value) / ySpan) * innerHeight;
  const linePath = seriesPath(currentPoints, xCoord, yCoord);
  const historicalPaths = historicalSeries.map((series) => ({
    year: series.year,
    path: seriesPath(series.points, xCoord, yCoord),
  }));
  const seasonalPath = validSeasonalPoints.length >= 2 ? seriesPath(seasonalPoints, xCoord, yCoord) : "";

  const bandSegments = [];
  let currentBandSegment = [];
  for (const point of currentPoints) {
    if (Number.isFinite(point.value) && Number.isFinite(point.std)) {
      currentBandSegment.push({
        offset: point.offset,
        low: point.value - point.std,
        high: point.value + point.std,
      });
    } else if (currentBandSegment.length > 0) {
      bandSegments.push(currentBandSegment);
      currentBandSegment = [];
    }
  }
  if (currentBandSegment.length > 0) {
    bandSegments.push(currentBandSegment);
  }
  const areaPaths = bandSegments
    .filter((segment) => segment.length >= 2)
    .map((segment) =>
      [
        segment
          .map((point, idx) => `${idx === 0 ? "M" : "L"} ${xCoord(point.offset)} ${yCoord(point.high)}`)
          .join(" "),
        ...[...segment].reverse().map((point) => `L ${xCoord(point.offset)} ${yCoord(point.low)}`),
        "Z",
      ].join(" "),
    );
  const hoverPoints = [
    ...currentPoints,
    ...seasonalPoints.map((point) => ({ ...point, seasonal: true })),
    ...historicalSeries.flatMap((series) => series.points),
  ]
    .filter((point) => Number.isFinite(point.value))
    .map((point) => ({
      ...point,
      x: xCoord(point.offset),
      y: yCoord(point.value),
      label: point.seasonal
        ? `Seasonal cycle: ${formatValue(point.value, 1)}%`
        : `${point.year}: ${formatValue(point.value, 1)}%`,
    }));
  const firstDate = dates[0] ?? "";
  const lastDate = dates[dates.length - 1] ?? "";
  const middleOffset = (TIMESERIES_WINDOW_DAYS - 1) / 2;
  const middlePoint = currentPoints
    .filter((point) => point.date && Number.isFinite(point.offset))
    .reduce((best, point) => {
      if (!best) {
        return point;
      }
      return Math.abs(point.offset - middleOffset) < Math.abs(best.offset - middleOffset) ? point : best;
    }, null);
  const middleDate = middlePoint?.date ?? dates[Math.floor(dates.length / 2)] ?? "";

  return {
    width,
    height,
    linePath,
    historicalPaths,
    seasonalPath,
    areaPaths,
    axisLeft: padding.left,
    axisRight: width - padding.right,
    axisTop: padding.top,
    axisBottom: height - padding.bottom,
    yTicks: yTicks.map((tick) => ({
      ...tick,
      y: padding.top + tick.fraction * innerHeight,
    })),
    zeroLineY: isAnomalyMode ? yCoord(0) : null,
    xTicks: [
      { x: padding.left, label: firstDate, anchor: "start" },
      { x: padding.left + innerWidth / 2, label: middleDate, anchor: "middle" },
      { x: width - padding.right, label: lastDate, anchor: "end" },
    ],
    mode,
    axisLabel: isAnomalyMode ? "LFMC Anomaly (%)" : "LFMC (%)",
    lineLabel: isAnomalyMode ? "Selected-year LFMC anomaly" : "Selected-year LFMC",
    showSeasonalCycle: !isAnomalyMode && seasonalPath !== "",
    showBand: !isAnomalyMode && areaPaths.length > 0,
    hoverPoints,
  };
}

function TimeseriesChart({ pointInfo, mode, supportsAnomaly }) {
  const [hoverPoint, setHoverPoint] = useState(null);
  const activeMode = mode === "anomaly" && supportsAnomaly ? "anomaly" : DEFAULT_TIMESERIES_MODE;
  const geometry = buildTimeseriesGeometry(pointInfo, activeMode);

  if (!geometry) {
    const emptyMessage = pointInfo
      ? `No finite ${activeMode === "anomaly" ? "LFMC anomaly" : "LFMC"} values are available for this cell in the previous 90 days.`
      : "Click the map to load the previous 90 days of values.";
    return (
      <div className="timeseries-wrap">
        <p className="panel-note">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div className="timeseries-wrap">
      <div className="timeseries-chart-wrap" onMouseLeave={() => setHoverPoint(null)}>
        <svg viewBox={`0 0 ${geometry.width} ${geometry.height}`} className="timeseries-chart" role="img">
          <line
            x1={geometry.axisLeft}
            x2={geometry.axisLeft}
            y1={geometry.axisTop}
            y2={geometry.axisBottom}
            className="chart-axis"
          />
          <line
            x1={geometry.axisLeft}
            x2={geometry.axisRight}
            y1={geometry.axisBottom}
            y2={geometry.axisBottom}
            className="chart-axis"
          />
          {geometry.historicalPaths.map((historicalPath) => (
            <path key={`hist-${historicalPath.year}`} d={historicalPath.path} className="chart-line-history" />
          ))}
          {geometry.areaPaths.map((areaPath, idx) => (
            <path key={`band-${idx}`} d={areaPath} className="chart-band" />
          ))}
          {geometry.showSeasonalCycle ? (
            <path d={geometry.seasonalPath} className="chart-line-seasonal" />
          ) : null}
          {geometry.zeroLineY !== null ? (
            <line
              x1={geometry.axisLeft}
              x2={geometry.axisRight}
              y1={geometry.zeroLineY}
              y2={geometry.zeroLineY}
              className="chart-zero-line"
            />
          ) : null}
          <path d={geometry.linePath} className={`chart-line chart-line-${geometry.mode}`} />
          {geometry.yTicks.map((tick) => (
            <g key={`y-${tick.label}`}>
              <line
                x1={geometry.axisLeft - 5}
                x2={geometry.axisLeft}
                y1={tick.y}
                y2={tick.y}
                className="chart-tick"
              />
              <text x={geometry.axisLeft - 9} y={tick.y + 4} className="chart-label chart-label-y">
                {tick.label}
              </text>
            </g>
          ))}
          {geometry.xTicks.map((tick) => (
            <g key={`x-${tick.label}-${tick.x}`}>
              <line
                x1={tick.x}
                x2={tick.x}
                y1={geometry.axisBottom}
                y2={geometry.axisBottom + 5}
                className="chart-tick"
              />
              <text
                x={tick.x}
                y={geometry.axisBottom + 18}
                textAnchor={tick.anchor}
                className="chart-label chart-label-x"
              >
                {tick.label}
              </text>
            </g>
          ))}
          <text
            x="16"
            y={geometry.axisTop + (geometry.axisBottom - geometry.axisTop) / 2}
            transform={`rotate(-90 16 ${geometry.axisTop + (geometry.axisBottom - geometry.axisTop) / 2})`}
            className="chart-label chart-label-axis"
          >
            {geometry.axisLabel}
          </text>
          {geometry.hoverPoints.map((point, idx) => (
            <circle
              key={`hover-${idx}-${point.date}-${point.year}`}
              cx={point.x}
              cy={point.y}
              r="7"
              className="chart-hover-target"
              onMouseEnter={() => setHoverPoint(point)}
              onMouseMove={() => setHoverPoint(point)}
            />
          ))}
        </svg>
        {hoverPoint ? (
          <div
            className="timeseries-tooltip"
            style={{
              left: `${(hoverPoint.x / geometry.width) * 100}%`,
              top: `${(hoverPoint.y / geometry.height) * 100}%`,
            }}
          >
            {hoverPoint.label}
          </div>
        ) : null}
      </div>
      <div className="timeseries-legend" aria-label="Timeseries legend">
        <div className="timeseries-legend-item">
          <span className={`timeseries-legend-swatch timeseries-legend-swatch-line timeseries-legend-swatch-${geometry.mode}`} />
          <span className="timeseries-legend-label">{geometry.lineLabel}</span>
        </div>
        <div className="timeseries-legend-item">
          <span className="timeseries-legend-swatch timeseries-legend-swatch-history" />
          <span className="timeseries-legend-label">Other years</span>
        </div>
        {geometry.showSeasonalCycle ? (
          <div className="timeseries-legend-item">
            <span className="timeseries-legend-swatch timeseries-legend-swatch-seasonal" />
            <span className="timeseries-legend-label">Average seasonal cycle</span>
          </div>
        ) : null}
        {geometry.showBand ? (
          <div className="timeseries-legend-item">
            <span className="timeseries-legend-swatch timeseries-legend-swatch-band" />
            <span className="timeseries-legend-label">Ensemble-based uncertainty</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function App() {
  const lfmcDisplayOpacity = 0.75;
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const rasterTileLayersRef = useRef([]);
  const activeRasterLayerIndexRef = useRef(0);
  const selectionSourceRef = useRef(null);
  const manifestRef = useRef(null);
  const dateIndexRef = useRef(0);
  const selectedLayerKeyRef = useRef("mean");
  const activeDatasetKeyRef = useRef(DEFAULT_DATASET_KEY);
  const pointRef = useRef(null);
  const transitionTokenRef = useRef(0);
  const queuedDateIndexRef = useRef(null);
  const transitionInFlightRef = useRef(false);
  const playbackTimeoutRef = useRef(null);
  const activeDownloadSiteIndexRef = useRef(0);
  const noticeTimeoutRef = useRef(null);
  const pointQueryTokenRef = useRef(0);
  const selectedLocationRef = useRef(null);
  const headerDatasetSelectorRef = useRef(null);

  const [metadata, setMetadata] = useState(null);
  const [datasetManifests, setDatasetManifests] = useState({});
  const [activeDatasetKey, setActiveDatasetKey] = useState(DEFAULT_DATASET_KEY);
  const [manifest, setManifest] = useState(null);
  const [dateIndex, setDateIndex] = useState(0);
  const [selectedLayerKey, setSelectedLayerKey] = useState("mean");
  const [timeseriesMode, setTimeseriesMode] = useState(DEFAULT_TIMESERIES_MODE);
  const [isPlaying, setIsPlaying] = useState(false);
  const [locationLatInput, setLocationLatInput] = useState("");
  const [locationLonInput, setLocationLonInput] = useState("");
  const [downloadSites, setDownloadSites] = useState([createDownloadSite()]);
  const [activeDownloadSiteIndex, setActiveDownloadSiteIndex] = useState(0);
  const [isDownloadingCsv, setIsDownloadingCsv] = useState(false);
  const [pointInfo, setPointInfo] = useState(null);
  const [statusText, setStatusText] = useState("Loading viewer manifest...");
  const [noticeText, setNoticeText] = useState("");
  const [isMapLoading, setIsMapLoading] = useState(false);
  const [isPointLoading, setIsPointLoading] = useState(false);
  const [isPointHistoryLoading, setIsPointHistoryLoading] = useState(false);
  const [activeDetailTab, setActiveDetailTab] = useState("timeseries");
  const [isIntroCollapsed, setIsIntroCollapsed] = useState(false);
  const [headerDatasetSelectorHeight, setHeaderDatasetSelectorHeight] = useState(0);

  const dates = manifest?.dates ?? [];
  const selectedDate = dates[dateIndex] ?? "NA";
  const sentinelDates = datasetManifests[SENTINEL_DATASET_KEY]?.dates ?? [];
  const globalDateEnd = sentinelDates[sentinelDates.length - 1] ?? dates[dates.length - 1] ?? "";
  const globalDates = buildDailyDateRange(GLOBAL_DATE_START, globalDateEnd);
  const globalDateIndex = Math.max(findDateIndex(globalDates, selectedDate), 0);
  const datasetMeta = metadata?.datasets?.[activeDatasetKey] ?? {};
  const supportsAnomaly = Boolean(datasetMeta.supports_anomaly);
  const supportsClimatology = Boolean(datasetMeta.supports_climatology);
  const manifestLayers = manifest?.layers ?? {};
  const configuredLayerKeys = Array.isArray(datasetMeta.layer_keys) ? datasetMeta.layer_keys : [];
  const orderedLayerKeys = [
    ...configuredLayerKeys.filter((layerKey) => manifestLayers[layerKey]),
    ...Object.keys(manifestLayers).filter((layerKey) => !configuredLayerKeys.includes(layerKey)),
  ];
  const layerEntries = orderedLayerKeys.map((layerKey) => [layerKey, manifestLayers[layerKey]]);
  const activeLayer = manifest?.layers?.[selectedLayerKey] ?? null;
  const activeLayerKey = activeLayer ? selectedLayerKey : Object.keys(manifest?.layers ?? {})[0] ?? "";
  const datasetKeys = Object.keys(metadata?.datasets ?? {});

  function showNotice(message) {
    setNoticeText(message);
    if (noticeTimeoutRef.current) {
      window.clearTimeout(noticeTimeoutRef.current);
    }
    noticeTimeoutRef.current = window.setTimeout(() => {
      setNoticeText("");
      noticeTimeoutRef.current = null;
    }, 30000);
  }

  function preferredLayerForDataset(datasetKey, targetManifest) {
    if (datasetKey === SENTINEL_DATASET_KEY) {
      return targetManifest?.layers?.lfmc ? "lfmc" : Object.keys(targetManifest?.layers ?? {})[0] ?? "";
    }
    return targetManifest?.layers?.mean ? "mean" : Object.keys(targetManifest?.layers ?? {})[0] ?? "";
  }

  function dateResolutionForDataset(datasetKey, targetDate, direction = "nearest") {
    const targetManifest = datasetManifests[datasetKey];
    const targetDates = targetManifest?.dates ?? [];
    if (!targetDates.length || !targetDate) {
      return null;
    }
    const targetIndex = findDateIndex(targetDates, targetDate, direction);
    if (targetIndex < 0) {
      return null;
    }
    const resolvedDate = targetDates[targetIndex];
    return {
      datasetKey,
      manifest: targetManifest,
      index: targetIndex,
      date: resolvedDate,
      distanceDays: dateDiffDays(resolvedDate, targetDate),
    };
  }

  function sentinelResolutionForInRange(targetDate) {
    const targetManifest = datasetManifests[SENTINEL_DATASET_KEY];
    const targetDates = targetManifest?.dates ?? [];
    if (!targetDates.length || !targetDate) {
      return { resolution: null, message: "" };
    }
    const { beforeIdx, afterIdx, exactIdx } = surroundingDateIndices(targetDates, targetDate);
    if (exactIdx >= 0) {
      return {
        resolution: dateResolutionForDataset(SENTINEL_DATASET_KEY, targetDate, "nearest"),
        message: "",
      };
    }
    if (beforeIdx >= 0 && afterIdx >= 0) {
      const previousDate = targetDates[beforeIdx];
      const nextDate = targetDates[afterIdx];
      if (dateDiffDays(nextDate, previousDate) > SENTINEL_DATE_TOLERANCE_DAYS) {
        const nearestResolution = dateResolutionForDataset(SENTINEL_DATASET_KEY, targetDate, "nearest");
        return {
          resolution: nearestResolution,
          message: nearestResolution
            ? `Sentinel-1 data gap from ${previousDate} to ${nextDate}; switched to nearest available date ${nearestResolution.date}.`
            : "",
        };
      }
      return {
        resolution: dateResolutionForDataset(SENTINEL_DATASET_KEY, nextDate, "nearest"),
        message: "",
      };
    }
    return {
      resolution: dateResolutionForDataset(SENTINEL_DATASET_KEY, targetDate, "nearest"),
      message: "",
    };
  }

  function resolutionForRequest(datasetKey, targetDate, direction = "nearest") {
    const targetManifest = datasetManifests[datasetKey];
    const targetDates = targetManifest?.dates ?? [];
    if (
      datasetKey === SENTINEL_DATASET_KEY &&
      targetDates.length &&
      targetDate >= targetDates[0] &&
      targetDate <= targetDates[targetDates.length - 1]
    ) {
      return sentinelResolutionForInRange(targetDate);
    }
    return {
      resolution: dateResolutionForDataset(datasetKey, targetDate, direction),
      message: "",
    };
  }

  function alternateDatasetKey(datasetKey) {
    return datasetKey === SENTINEL_DATASET_KEY ? DEFAULT_DATASET_KEY : SENTINEL_DATASET_KEY;
  }

  function canUseResolution(resolution) {
    if (!resolution) {
      return false;
    }
    return resolution.datasetKey !== SENTINEL_DATASET_KEY || resolution.distanceDays <= SENTINEL_DATE_TOLERANCE_DAYS;
  }

  function refreshSelectedLocation(dateStr, updateDownloadSite = false) {
    const selectedLocation = selectedLocationRef.current;
    if (!selectedLocation || !dateStr) {
      return;
    }
    void loadPointAtLocation(selectedLocation.lat, selectedLocation.lon, dateStr, {
      updateDownloadSite,
      loadHistory: true,
    }).catch((error) => {
      setStatusText(`Point refresh failed: ${error.message}`);
    });
  }

  function applyResolvedSelection(resolution, message = "") {
    const nextManifest = resolution.manifest;
    const nextLayerKey = preferredLayerForDataset(resolution.datasetKey, nextManifest);
    setIsPlaying(false);
    setActiveDatasetKey(resolution.datasetKey);
    activeDatasetKeyRef.current = resolution.datasetKey;
    setManifest(nextManifest);
    manifestRef.current = nextManifest;
    setDateIndex(resolution.index);
    dateIndexRef.current = resolution.index;
    setSelectedLayerKey(nextLayerKey);
    selectedLayerKeyRef.current = nextLayerKey;
    setTimeseriesMode(DEFAULT_TIMESERIES_MODE);
    setDownloadSites((currentSites) =>
      currentSites.map((site) => clampDownloadSiteDates(site, nextManifest.dates ?? [])),
    );
    setStatusText(`Showing ${nextManifest.dataset_label} for ${resolution.date}`);
    void transitionToDateIndex(resolution.index, { force: true, layerKey: nextLayerKey });
    refreshSelectedLocation(resolution.date);
    if (message) {
      showNotice(message);
    }
  }

  function requestDatasetDate(datasetKey, targetDate, direction = "nearest", options = {}) {
    const { forceDataset = false } = options;
    const targetManifest = datasetManifests[datasetKey];
    const targetDates = targetManifest?.dates ?? [];
    if (!targetDates.length || !targetDate) {
      return;
    }
    const targetStart = targetDates[0];
    const targetEnd = targetDates[targetDates.length - 1];
    if (targetDate < targetStart || targetDate > targetEnd) {
      const rangeMessage = `${targetManifest.dataset_label} is only available from ${targetStart} to ${targetEnd}`;
      if (forceDataset) {
        const forcedResolution = dateResolutionForDataset(datasetKey, targetDate, "nearest");
        if (forcedResolution) {
          applyResolvedSelection(
            forcedResolution,
            `${rangeMessage}; switched to nearest available date ${forcedResolution.date}.`,
          );
        }
        return;
      }
      const alternateKey = alternateDatasetKey(datasetKey);
      const { resolution: alternateResolution } = resolutionForRequest(alternateKey, targetDate, "nearest");
      if (canUseResolution(alternateResolution)) {
        applyResolvedSelection(
          alternateResolution,
          `${rangeMessage}; switched to ${alternateResolution.manifest.dataset_label}.`,
        );
      }
      return;
    }
    const { resolution, message: resolvedMessage } = resolutionForRequest(datasetKey, targetDate, direction);
    if (!resolution) {
      return;
    }
    const targetIndex = resolution.index;
    const resolvedDate = resolution.date;
    const message = resolvedMessage || (resolvedDate !== targetDate && datasetKey !== SENTINEL_DATASET_KEY
      ? `${targetManifest.dataset_label} is only available on ${resolvedDate}; snapped from ${targetDate}.`
      : "");
    if (datasetKey !== activeDatasetKeyRef.current) {
      applyResolvedSelection(resolution, message);
      return;
    }
    if (message) {
      showNotice(message);
    }
    setIsPlaying(false);
    requestDateTransition(targetIndex);
    refreshSelectedLocation(resolvedDate);
  }

  async function queryPoint(params, dateStr, options = {}) {
    const {
      datasetKey = activeDatasetKeyRef.current,
      includeHistory = false,
    } = options;
    const query = new URLSearchParams({
      dataset: datasetKey,
      date: dateStr,
      include_timeseries: "true",
      include_history: includeHistory ? "true" : "false",
      timeseries_days: String(TIMESERIES_WINDOW_DAYS),
    });
    if (params.x !== undefined && params.y !== undefined) {
      query.set("x", String(params.x));
      query.set("y", String(params.y));
    } else {
      query.set("lat", String(params.lat));
      query.set("lon", String(params.lon));
    }
    const maxAttempts = 30;
    const retryDelayMs = 2000;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const response = await fetch(apiUrl(`/api/point?${query.toString()}`), { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok || payload.error) {
        const errorMessage = payload.error || `Point HTTP ${response.status}`;
        const isLoading = response.status === 503 && errorMessage.toLowerCase().includes("loading");
        if (isLoading && attempt < maxAttempts) {
          await new Promise((resolve) => window.setTimeout(resolve, retryDelayMs));
          continue;
        }
        throw new Error(errorMessage);
      }
      return payload;
    }

    throw new Error("Point query timed out while viewer dataset was loading");
  }

  function mergePointHistory(payload, token) {
    setIsPointHistoryLoading(true);
    queryPoint(
      {
        lat: payload.requested_lat,
        lon: payload.requested_lon,
      },
      payload.date,
      {
        datasetKey: payload.dataset_key,
        includeHistory: true,
      },
    )
      .then((historyPayload) => {
        if (token !== pointQueryTokenRef.current) {
          return;
        }
        setPointInfo(historyPayload);
      })
      .catch((error) => {
        if (token === pointQueryTokenRef.current) {
          setStatusText(`All-year comparison load failed: ${error.message}`);
        }
      })
      .finally(() => {
        if (token === pointQueryTokenRef.current) {
          setIsPointHistoryLoading(false);
        }
      });
  }

  async function loadPoint(params, dateStr, options = {}) {
    const { recenter = false, updateDownloadSite = true, loadHistory = true } = options;
    const token = pointQueryTokenRef.current + 1;
    pointQueryTokenRef.current = token;
    setIsPointLoading(true);
    setIsPointHistoryLoading(false);
    try {
      const payload = await queryPoint(params, dateStr, { includeHistory: false });
      if (token !== pointQueryTokenRef.current) {
        return payload;
      }
      setPointInfo(payload);
      pointRef.current = {
        x: payload.requested_grid_x,
        y: payload.requested_grid_y,
      };
      selectedLocationRef.current = {
        lat: payload.requested_lat,
        lon: payload.requested_lon,
      };
      if (updateDownloadSite) {
        setDownloadSites((currentSites) =>
          currentSites.map((site, siteIndex) =>
            siteIndex === activeDownloadSiteIndexRef.current
              ? {
                  ...site,
                  lat: formatCoordinateInput(payload.cell_center_lat),
                  lon: formatCoordinateInput(payload.cell_center_lon),
                }
              : site,
          ),
        );
      }
      updateSelectionFeatures(payload);
      if (recenter && mapRef.current) {
        mapRef.current.getView().animate({
          center: [payload.requested_grid_x, payload.requested_grid_y],
          duration: 250,
        });
      }
      if (loadHistory) {
        mergePointHistory(payload, token);
      }
      return payload;
    } finally {
      if (token === pointQueryTokenRef.current) {
        setIsPointLoading(false);
      }
    }
  }

  async function loadPointAtCoordinate(x, y, dateStr, options = {}) {
    return loadPoint({ x, y }, dateStr, options);
  }

  async function loadPointAtLocation(lat, lon, dateStr, options = {}) {
    return loadPoint({ lat, lon }, dateStr, options);
  }

  function buildTileGrid(manifestPayload) {
    return new TileGrid({
      extent: [
        manifestPayload.grid_extent.west,
        manifestPayload.grid_extent.south,
        manifestPayload.grid_extent.east,
        manifestPayload.grid_extent.north,
      ],
      origin: manifestPayload.tiles.origin,
      resolutions: manifestPayload.tiles.resolutions,
      tileSize: manifestPayload.tiles.tile_size,
    });
  }

  function visibleTileReadiness(tileGrid, tileCounts) {
    const map = mapRef.current;
    if (!map) {
      return { keys: new Set(), threshold: 0 };
    }
    const mapSize = map.getSize();
    const resolution = map.getView().getResolution();
    if (!mapSize || !resolution) {
      return { keys: new Set(), threshold: 0 };
    }

    const z = tileGrid.getZForResolution(resolution);
    const visibleExtent = map.getView().calculateExtent(mapSize);
    const tileRange = tileGrid.getTileRangeForExtentAndZ(visibleExtent, z);
    const zoomCounts = tileCounts[String(z)];
    const keys = new Set();
    if (!zoomCounts) {
      return { keys, threshold: 0 };
    }

    for (let tileX = tileRange.minX; tileX <= tileRange.maxX; tileX += 1) {
      if (tileX < 0 || tileX >= zoomCounts.x) {
        continue;
      }
      for (let tileY = tileRange.minY; tileY <= tileRange.maxY; tileY += 1) {
        if (tileY < 0 || tileY >= zoomCounts.y) {
          continue;
        }
        keys.add(`${z}/${tileX}/${tileY}`);
      }
    }

    const threshold = keys.size <= 4 ? keys.size : Math.ceil(keys.size * 0.6);
    return { keys, threshold };
  }

  function createTileSourceForDate(manifestPayload, targetDate, layerKey, requestToken, onReady) {
    const layerConfig = manifestPayload.layers[layerKey];
    const tileTemplate = layerConfig?.tile_root_template;
    const assetBaseUrl = manifestPayload.asset_base_url;
    if (!tileTemplate) {
      throw new Error(`No tile template found for ${layerKey} ${targetDate}`);
    }
    if (!assetBaseUrl) {
      throw new Error(`No asset base URL configured for ${layerKey} ${targetDate}`);
    }

    const tileGrid = buildTileGrid(manifestPayload);
    const tileCounts = layerConfig.tile_counts ?? {};
    const readiness = visibleTileReadiness(tileGrid, tileCounts);
    const loadedVisibleKeys = new Set();
    let readyCalled = false;

    const maybeReady = () => {
      if (readyCalled) {
        return;
      }
      if (readiness.threshold === 0 || loadedVisibleKeys.size >= readiness.threshold) {
        readyCalled = true;
        onReady();
      }
    };

    const tileSource = new XYZ({
      projection: manifestPayload.grid_crs,
      tileGrid,
      wrapX: false,
      transition: 0,
      interpolate: false,
      tileUrlFunction: (tileCoord) => {
        if (!tileCoord) {
          return undefined;
        }
        const [z, x, rawY] = tileCoord;
        const zoomCounts = tileCounts[String(z)];
        if (!zoomCounts) {
          return undefined;
        }
        let y = rawY;
        if (y < 0) {
          y = -rawY - 1;
        }
        if (x < 0 || y < 0 || x >= zoomCounts.x || y >= zoomCounts.y) {
          return undefined;
        }
        return tileUrlForCoord(assetBaseUrl, tileTemplate, targetDate, z, x, y);
      },
    });

    const handleTileSettled = (event) => {
      const tileCoord = event.tile?.getTileCoord?.();
      if (!tileCoord) {
        return;
      }
      const [z, x, rawY] = tileCoord;
      let y = rawY;
      if (y < 0) {
        y = -rawY - 1;
      }
      const key = `${z}/${x}/${y}`;
      if (readiness.keys.has(key)) {
        loadedVisibleKeys.add(key);
        maybeReady();
      }
    };

    tileSource.on("tileloadstart", () => {
      if (requestToken === transitionTokenRef.current) {
        setIsMapLoading(true);
        setStatusText(`Loading ${layerConfig.label} tiles for ${targetDate}`);
      }
    });
    tileSource.on("tileloadend", handleTileSettled);
    tileSource.on("tileloaderror", handleTileSettled);

    if (readiness.threshold === 0) {
      window.setTimeout(maybeReady, 0);
    }

    return tileSource;
  }

  function animateLayerFade(outLayer, inLayer, durationMs = 200) {
    return new Promise((resolve) => {
      const start = performance.now();
      inLayer.setVisible(true);
      inLayer.setOpacity(0);
      outLayer.setOpacity(lfmcDisplayOpacity);

      const step = (now) => {
        const fraction = Math.min((now - start) / durationMs, 1);
        inLayer.setOpacity(lfmcDisplayOpacity * fraction);
        outLayer.setOpacity(lfmcDisplayOpacity * (1 - fraction));
        if (fraction < 1) {
          window.requestAnimationFrame(step);
          return;
        }
        inLayer.setOpacity(lfmcDisplayOpacity);
        outLayer.setOpacity(0);
        outLayer.setVisible(false);
        outLayer.setSource(null);
        outLayer.setOpacity(lfmcDisplayOpacity);
        resolve();
      };

      window.requestAnimationFrame(step);
    });
  }

  async function transitionToDateIndex(targetIndex, options = {}) {
    const { immediate = false, force = false, layerKey = selectedLayerKeyRef.current } = options;
    const manifestPayload = manifestRef.current;
    if (!manifestPayload || !mapRef.current) {
      return false;
    }
    if (!immediate && !force && targetIndex === dateIndexRef.current) {
      return true;
    }

    const targetDate = manifestPayload.dates[targetIndex];
    if (!targetDate) {
      return false;
    }
    const layerConfig = manifestPayload.layers[layerKey];
    if (!layerConfig) {
      setStatusText(`Layer ${layerKey} is not available`);
      return false;
    }

    const requestToken = transitionTokenRef.current + 1;
    transitionTokenRef.current = requestToken;

    const currentLayerIndex = activeRasterLayerIndexRef.current;
    const nextLayerIndex = immediate ? currentLayerIndex : 1 - currentLayerIndex;
    const currentLayer = rasterTileLayersRef.current[currentLayerIndex];
    const nextLayer = rasterTileLayersRef.current[nextLayerIndex];
    if (!nextLayer) {
      return false;
    }

    return new Promise((resolve) => {
      const tileSource = createTileSourceForDate(manifestPayload, targetDate, layerKey, requestToken, async () => {
        if (requestToken !== transitionTokenRef.current) {
          resolve(false);
          return;
        }

        if (immediate) {
          nextLayer.setOpacity(lfmcDisplayOpacity);
          nextLayer.setVisible(true);
        } else {
          await animateLayerFade(currentLayer, nextLayer, 200);
          activeRasterLayerIndexRef.current = nextLayerIndex;
        }

        setDateIndex(targetIndex);
        dateIndexRef.current = targetIndex;
        setIsMapLoading(false);
        setStatusText(`Showing ${layerConfig.label} for ${targetDate}`);
        resolve(true);
      });

      nextLayer.setSource(tileSource);
      nextLayer.setVisible(true);
      nextLayer.setOpacity(immediate ? lfmcDisplayOpacity : 0);
    });
  }

  async function drainQueuedTransitions() {
    if (transitionInFlightRef.current) {
      return;
    }
    transitionInFlightRef.current = true;
    try {
      while (queuedDateIndexRef.current !== null) {
        const targetIndex = queuedDateIndexRef.current;
        queuedDateIndexRef.current = null;
        await transitionToDateIndex(targetIndex);
      }
    } finally {
      transitionInFlightRef.current = false;
    }
  }

  function requestDateTransition(targetIndex) {
    queuedDateIndexRef.current = targetIndex;
    void drainQueuedTransitions();
  }

  function handleLayerChange(layerKey) {
    const manifestPayload = manifestRef.current;
    if (!manifestPayload?.layers?.[layerKey]) {
      return;
    }
    if (isAnomalyLayer(layerKey) && !supportsAnomaly) {
      showNotice("LFMC anomaly is unavailable for the selected dataset.");
      return;
    }
    setIsPlaying(false);
    setSelectedLayerKey(layerKey);
    selectedLayerKeyRef.current = layerKey;
    if (isAnomalyLayer(layerKey)) {
      setTimeseriesMode("anomaly");
    } else {
      setTimeseriesMode(DEFAULT_TIMESERIES_MODE);
    }
    void transitionToDateIndex(dateIndexRef.current, { force: true, layerKey });
  }

  function requestDateValueTransition(targetDate, direction = "nearest") {
    if (!dates.length || !targetDate) {
      return;
    }
    requestDatasetDate(activeDatasetKeyRef.current, targetDate, direction);
  }

  function handleDateStep(amount, unit) {
    if (!dates.length || selectedDate === "NA") {
      return;
    }
    if (activeDatasetKeyRef.current === SENTINEL_DATASET_KEY && unit === "day" && Math.abs(amount) === 15) {
      const nextIndex = Math.max(0, Math.min(dates.length - 1, dateIndexRef.current + Math.sign(amount)));
      requestDateTransition(nextIndex);
      refreshSelectedLocation(dates[nextIndex]);
      return;
    }
    const targetDate = shiftDateString(selectedDate, amount, unit);
    requestDateValueTransition(targetDate, amount < 0 ? "backward" : "forward");
  }

  function updateSelectionFeatures(payload) {
    const selectionSource = selectionSourceRef.current;
    if (!selectionSource) {
      return;
    }
    selectionSource.clear();
    selectionSource.addFeature(
      new Feature({
        geometry: new Point([payload.requested_grid_x, payload.requested_grid_y]),
        role: "click_point",
      }),
    );
    selectionSource.addFeature(
      new Feature({
        geometry: new Polygon(cellPolygonCoordinates(payload.cell_bounds)),
        role: "cell_outline",
      }),
    );
  }

  useEffect(() => {
    let cancelled = false;
    const maxAttempts = 120;
    const retryDelayMs = 2000;

    async function loadManifests() {
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        try {
          setStatusText("Starting viewer...");
          const metadataResponse = await fetch(apiUrl("/api/metadata"), { cache: "no-store" });
          if (!metadataResponse.ok) {
            let metadataError = `Metadata HTTP ${metadataResponse.status}`;
            try {
              const metadataErrorPayload = await metadataResponse.json();
              metadataError = metadataErrorPayload.error || metadataError;
            } catch {
              // Keep the HTTP status message if the response is not JSON.
            }
            throw new Error(metadataError);
          }
          const metadataPayload = await metadataResponse.json();
          const nextManifests = {};
          for (const [datasetKey, datasetConfig] of Object.entries(metadataPayload.datasets ?? {})) {
            const manifestResponse = await fetch(datasetConfig.asset_manifest_url, { cache: "no-store" });
            if (!manifestResponse.ok) {
              throw new Error(`${datasetConfig.dataset_label} manifest HTTP ${manifestResponse.status}`);
            }
            const manifestPayload = await manifestResponse.json();
            nextManifests[datasetKey] = {
              ...datasetConfig,
              ...manifestPayload,
              dataset_key: datasetKey,
              dataset_label: datasetConfig.dataset_label,
              initial_date: datasetConfig.initial_date,
              asset_base_url: datasetConfig.asset_base_url,
              asset_manifest_url: datasetConfig.asset_manifest_url,
              supports_anomaly: Boolean(datasetConfig.supports_anomaly),
              supports_uncertainty: Boolean(datasetConfig.supports_uncertainty),
              supports_climatology: Boolean(datasetConfig.supports_climatology),
            };
          }
          if (cancelled) {
            return;
          }
          const defaultKey = metadataPayload.default_dataset || DEFAULT_DATASET_KEY;
          const initialManifest = nextManifests[defaultKey] ?? nextManifests[Object.keys(nextManifests)[0]];
          const initialDate = configuredInitialDate(initialManifest.dates, initialManifest.initial_date);
          const initialIndex = Math.max(initialManifest.dates.indexOf(initialDate), 0);
          const initialLayerKey = preferredLayerForDataset(defaultKey, initialManifest);
          setMetadata(metadataPayload);
          setDatasetManifests(nextManifests);
          setActiveDatasetKey(defaultKey);
          activeDatasetKeyRef.current = defaultKey;
          setManifest(initialManifest);
          manifestRef.current = initialManifest;
          setDateIndex(initialIndex);
          dateIndexRef.current = initialIndex;
          setSelectedLayerKey(initialLayerKey);
          selectedLayerKeyRef.current = initialLayerKey;
          setDownloadSites((currentSites) =>
            currentSites.map((site) => ({
              ...site,
              startDate: site.startDate || initialManifest.dates[0] || "",
              endDate:
                site.endDate ||
                defaultDownloadEndDate(initialManifest.dates, site.startDate || initialManifest.dates[0] || ""),
            })),
          );
          setStatusText(`Loaded ${initialManifest.dataset_label}`);
          return;
        } catch (error) {
          if (cancelled) {
            return;
          }
          if (attempt === maxAttempts) {
            setStatusText(`Manifest load failed: ${error.message}`);
            return;
          }
          setStatusText(`Waiting for viewer data... (${attempt}/${maxAttempts})`);
          await new Promise((resolve) => window.setTimeout(resolve, retryDelayMs));
        }
      }
    }

    loadManifests();

    return () => {
      cancelled = true;
      if (noticeTimeoutRef.current) {
        window.clearTimeout(noticeTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    manifestRef.current = manifest;
  }, [manifest]);

  useEffect(() => {
    dateIndexRef.current = dateIndex;
  }, [dateIndex]);

  useEffect(() => {
    selectedLayerKeyRef.current = selectedLayerKey;
  }, [selectedLayerKey]);

  useEffect(() => {
    activeDatasetKeyRef.current = activeDatasetKey;
  }, [activeDatasetKey]);

  useEffect(() => {
    activeDownloadSiteIndexRef.current = activeDownloadSiteIndex;
  }, [activeDownloadSiteIndex]);

  useEffect(() => {
    const node = headerDatasetSelectorRef.current;
    if (!node) {
      return undefined;
    }

    const updateHeight = () => {
      setHeaderDatasetSelectorHeight(Math.ceil(node.getBoundingClientRect().height));
    };
    updateHeight();

    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateHeight);
      return () => window.removeEventListener("resize", updateHeight);
    }

    const observer = new ResizeObserver(updateHeight);
    observer.observe(node);
    return () => observer.disconnect();
  }, [datasetKeys.length]);

  useEffect(() => {
    if (!mapRef.current) {
      return undefined;
    }
    const animationFrameId = window.requestAnimationFrame(() => {
      mapRef.current?.updateSize();
    });
    const timeoutId = window.setTimeout(() => {
      mapRef.current?.updateSize();
    }, 220);
    return () => {
      window.cancelAnimationFrame(animationFrameId);
      window.clearTimeout(timeoutId);
    };
  }, [isIntroCollapsed, headerDatasetSelectorHeight]);

  useEffect(() => {
    if (!manifest || !mapContainerRef.current) {
      return undefined;
    }

    const extent = [
      manifest.grid_extent.west,
      manifest.grid_extent.south,
      manifest.grid_extent.east,
      manifest.grid_extent.north,
    ];
    const projectionCode = manifest.grid_crs;
    const projection = getProjection(projectionCode);
    if (!projection) {
      setStatusText(`Unsupported map projection ${projectionCode}`);
      return undefined;
    }
    const isWebMercator = projectionCode === "EPSG:3857";
    if (!isWebMercator) {
      projection.setExtent(extent);
    }

    const view = new View({
      projection,
      center: getCenter(extent),
      resolutions: startupViewResolutions(manifest, extent, mapContainerRef.current),
      constrainResolution: true,
      extent,
      showFullExtent: true,
      zoom: 0,
    });

    const rasterTileLayerA = new TileLayer({
      opacity: lfmcDisplayOpacity,
      preload: 1,
    });
    const rasterTileLayerB = new TileLayer({
      opacity: 0,
      visible: false,
      preload: 1,
    });
    rasterTileLayersRef.current = [rasterTileLayerA, rasterTileLayerB];
    activeRasterLayerIndexRef.current = 0;
    const layers = [];
    if (isWebMercator) {
      layers.push(
        new TileLayer({
          source: new XYZ({
            url: "https://{a-d}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            attributions: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            maxZoom: 20,
          }),
          preload: 1,
        }),
      );
    }

    const selectionSource = new VectorSource();
    selectionSourceRef.current = selectionSource;
    const selectionLayer = new VectorLayer({
      source: selectionSource,
      style: (feature) => {
        const role = feature.get("role");
        if (role === "click_point") {
          return new Style({
            image: new CircleStyle({
              radius: 4.5,
              fill: new Fill({ color: "#f4efe3" }),
              stroke: new Stroke({ color: "#4c2b17", width: 1.5 }),
            }),
          });
        }
        return new Style({
          stroke: new Stroke({ color: "#f4efe3", width: 2.25 }),
          fill: new Fill({ color: "rgba(244, 239, 227, 0.08)" }),
        });
      },
    });

    layers.push(rasterTileLayerA, rasterTileLayerB, selectionLayer);

    const map = new Map({
      target: mapContainerRef.current,
      layers,
      view,
      controls: defaultControls().extend([new ScaleLine()]),
    });

    map.getView().fit(extent, { padding: [24, 24, 24, 24], duration: 0 });

    map.on("click", async (event) => {
      const currentManifest = manifestRef.current;
      const currentSelectedDate = currentManifest.dates[dateIndexRef.current];
      const [x, y] = event.coordinate;
      setIsPlaying(false);
      try {
        await loadPointAtCoordinate(x, y, currentSelectedDate);
      } catch (error) {
        setStatusText(`Point query failed: ${error.message}`);
      }
    });

    mapRef.current = map;
    void transitionToDateIndex(dateIndexRef.current, { immediate: true, force: true });

    return () => {
      if (playbackTimeoutRef.current) {
        window.clearTimeout(playbackTimeoutRef.current);
        playbackTimeoutRef.current = null;
      }
      map.setTarget(null);
      mapRef.current = null;
      rasterTileLayersRef.current = [];
      selectionSourceRef.current = null;
    };
  }, [Boolean(manifest)]);

  useEffect(() => {
    if (!pointRef.current || !manifest || isPlaying) {
      return;
    }

    const currentSelectedDate = manifest.dates[dateIndex];
    const selectedLocation = selectedLocationRef.current;
    if (!selectedLocation) {
      return;
    }

    loadPointAtLocation(selectedLocation.lat, selectedLocation.lon, currentSelectedDate)
      .then((payload) => {
        updateSelectionFeatures(payload);
      })
      .catch((error) => {
        setStatusText(`Point refresh failed: ${error.message}`);
      });
  }, [dateIndex, manifest, isPlaying]);

  useEffect(() => {
    if (!isPlaying || dates.length < 2) {
      return undefined;
    }
    let cancelled = false;

    async function runPlayback() {
      while (!cancelled) {
        const increment = activeDatasetKeyRef.current === SENTINEL_DATASET_KEY ? 1 : 7;
        const nextIndex = (dateIndexRef.current + increment) % dates.length;
        if (nextIndex === dateIndexRef.current) {
          break;
        }
        await transitionToDateIndex(nextIndex);
        if (cancelled) {
          break;
        }
        await new Promise((resolve) => {
          playbackTimeoutRef.current = window.setTimeout(resolve, 225);
        });
      }
    }

    void runPlayback();

    return () => {
      cancelled = true;
      if (playbackTimeoutRef.current) {
        window.clearTimeout(playbackTimeoutRef.current);
        playbackTimeoutRef.current = null;
      }
    };
  }, [isPlaying, dates.length]);

  useEffect(() => {
    if (!pointInfo) {
      return;
    }
    setLocationLatInput(formatCoordinateInput(pointInfo.cell_center_lat));
    setLocationLonInput(formatCoordinateInput(pointInfo.cell_center_lon));
  }, [pointInfo]);

  async function handleLocationSubmit(event) {
    event.preventDefault();
    if (!manifest) {
      return;
    }
    setIsPlaying(false);
    const latitude = parseCoordinateInput(locationLatInput);
    const longitude = parseCoordinateInput(locationLonInput);
    if (latitude === null || longitude === null) {
      setStatusText("Enter valid numeric latitude and longitude values");
      return;
    }
    if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
      setStatusText("Latitude must be between -90 and 90 and longitude between -180 and 180");
      return;
    }
    try {
      const [x, y] = transformCoordinate([longitude, latitude], "EPSG:4326", manifest.grid_crs);
      await loadPointAtCoordinate(x, y, selectedDate, {
        recenter: true,
      });
    } catch (error) {
      setStatusText(`Location lookup failed: ${error.message}`);
    }
  }

  async function handleDownloadCsv() {
    if (!manifest) {
      return;
    }

    const sites = [];
    for (const site of downloadSites) {
      const isBlank = String(site.lat).trim() === "" && String(site.lon).trim() === "";
      if (isBlank) {
        continue;
      }
      const latitude = parseCoordinateInput(site.lat);
      const longitude = parseCoordinateInput(site.lon);
      if (latitude === null || longitude === null) {
        setStatusText("Each site must have valid numeric latitude and longitude");
        return;
      }
      if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
        setStatusText("All site coordinates must have valid latitude/longitude ranges");
        return;
      }
      if (!site.startDate || !site.endDate) {
        setStatusText("Each site must have a start date and end date");
        return;
      }
      if (site.endDate < site.startDate) {
        setStatusText("Each site end date must be on or after its start date");
        return;
      }
      if (!isDownloadRangeWithinLimit(site.startDate, site.endDate)) {
        setStatusText(`Each site CSV range is limited to ${MAX_DOWNLOAD_YEARS} years`);
        return;
      }
      sites.push({
        lat: latitude,
        lon: longitude,
        startDate: site.startDate,
        endDate: site.endDate,
      });
    }
    if (!sites.length) {
      setStatusText("Select a site on the map or enter a site before downloading");
      return;
    }

    const queryStartDate = minDateString(...sites.map((site) => site.startDate));
    const queryEndDate = maxDateString(...sites.map((site) => site.endDate));
    const query = new URLSearchParams({
      dataset: activeDatasetKey,
      start_date: queryStartDate,
      end_date: queryEndDate,
    });
    sites.forEach((site) => {
      query.append("site", `${site.lat},${site.lon},${site.startDate},${site.endDate}`);
    });

    setIsDownloadingCsv(true);
    setStatusText(`Preparing ${manifest.dataset_label} CSV download...`);
    try {
      const response = await fetch(apiUrl(`/api/download_csv?${query.toString()}`));
      if (!response.ok) {
        const errorText = await response.text();
        let message = errorText || `HTTP ${response.status}`;
        try {
          const parsed = JSON.parse(errorText);
          if (parsed?.error) {
            message = parsed.error;
          }
        } catch {
          // Keep the raw error text if the API did not return JSON.
        }
        throw new Error(message);
      }

      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") ?? "";
      const filenameMatch = disposition.match(/filename="([^"]+)"/);
      const filename = filenameMatch?.[1] ?? "lfmc_site_download.csv";
      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
      setStatusText(`Downloaded ${manifest.dataset_label} CSV for ${sites.length} site${sites.length === 1 ? "" : "s"}`);
    } catch (error) {
      setStatusText(`CSV download failed: ${error.message}`);
    } finally {
      setIsDownloadingCsv(false);
    }
  }

  function handleAddDownloadSite() {
    const activeSite = downloadSites[activeDownloadSiteIndex];
    if (!activeSite) {
      return;
    }
    const latitude = parseCoordinateInput(activeSite.lat);
    const longitude = parseCoordinateInput(activeSite.lon);
    if (latitude === null || longitude === null) {
      setStatusText("Fill Site coordinates before adding another site");
      return;
    }
    if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
      setStatusText("Site coordinates must have valid latitude/longitude ranges");
      return;
    }
    setDownloadSites((currentSites) => {
      if (currentSites.length >= 10) {
        return currentSites;
      }
      return [
        ...currentSites,
        createDownloadSite(
          activeSite.startDate || dates[0] || "",
          activeSite.endDate || defaultDownloadEndDate(dates, activeSite.startDate || dates[0] || ""),
        ),
      ];
    });
    setActiveDownloadSiteIndex((currentValue) => Math.min(currentValue + 1, 9));
  }

  function handleUpdateDownloadSite(index, field, value) {
    setDownloadSites((currentSites) =>
      currentSites.map((site, siteIndex) => {
        if (siteIndex !== index) {
          return site;
        }
        const nextSite = {
          ...site,
          [field]: value,
        };
        if (field === "startDate" || field === "endDate") {
          return clampDownloadSiteDates(nextSite, dates);
        }
        return nextSite;
      }),
    );
  }

  function handleRemoveDownloadSite(index) {
    setDownloadSites((currentSites) => {
      const nextSites =
        currentSites.length === 1
          ? [createDownloadSite(dates[0] ?? "", defaultDownloadEndDate(dates, dates[0] ?? ""))]
          : currentSites.filter((_, siteIndex) => siteIndex !== index);
      setActiveDownloadSiteIndex((currentValue) => {
        if (nextSites.length === 1) {
          return 0;
        }
        if (index < currentValue) {
          return currentValue - 1;
        }
        if (index === currentValue) {
          return Math.max(0, currentValue - 1);
        }
        return Math.min(currentValue, nextSites.length - 1);
      });
      return nextSites;
    });
  }

  const dateStepControls = activeDatasetKey === SENTINEL_DATASET_KEY
    ? [
        [-3, "month", "-3 months"],
        [-1, "month", "-1 month"],
        [-15, "day", "-15 days"],
        [15, "day", "+15 days"],
        [1, "month", "+1 month"],
        [3, "month", "+3 months"],
      ]
    : [
        [-3, "month", "-3 months"],
        [-1, "month", "-1 month"],
        [-1, "day", "-1 day"],
        [1, "day", "+1 day"],
        [1, "month", "+1 month"],
        [3, "month", "+3 months"],
      ];

  return (
    <div className={`app-shell ${isIntroCollapsed ? "app-shell-intro-collapsed" : ""}`}>
      <header className="date-bar">
        <div className="date-bar-main">
          <div
            className={`viewer-title-block ${isIntroCollapsed ? "viewer-title-block-collapsed" : ""}`}
            style={
              isIntroCollapsed && headerDatasetSelectorHeight > 0
                ? { maxHeight: `${headerDatasetSelectorHeight}px` }
                : undefined
            }
          >
            <div className="viewer-title-row">
              <h1>Live Fuel Moisture Content Products from Stanford's Remote Sensing Ecohydrology Group</h1>
              <button
                type="button"
                className="intro-toggle-button"
                aria-controls="viewer-intro"
                aria-expanded={!isIntroCollapsed}
                onClick={() => setIsIntroCollapsed((currentValue) => !currentValue)}
              >
                {isIntroCollapsed ? "Show more" : "Minimize text"}
              </button>
            </div>
            <p id="viewer-intro" className="viewer-intro">
              Live fuel moisture content (LFMC) is the mass of water in vegetation normalized by dry biomass, and
              is an important indicator for wildland fire risk. Because no single observing system captures all
              regions and time periods equally well, this viewer presents complementary LFMC products for different
              use cases. You can choose between two datasets. First, a MODIS-based dataset provides a long
              historical record beginning in 2001 at 500 m and daily resolution, but updates only annually and can
              be uncertain in some evergreen forests. Alternatively, a Sentinel-1 based dataset provides a shorter
              historical record beginning in 2016 at 250 m and 15-day resolution, but updates with approximately
              10-day latency and is more skillful in evergreen forests. Given these performance differences, we
              present MODIS-based LFMC in this map viewer with high opacity to note that, while usable in some
              situations, it should be treated with caution. You can view absolute LFMC or LFMC anomaly, where anomaly shows
              whether vegetation is wetter or drier than typical for that calendar day. For guidance on choosing
              the appropriate dataset, citing this data, performance metrics, and download instructions, please see{" "}
              <a href={PRODUCT_DOC_URL} target="_blank" rel="noreferrer">
                this information document
              </a>
              .
            </p>
          </div>
          <section ref={headerDatasetSelectorRef} className="header-dataset-selector" aria-label="Dataset selector">
            <div className="panel-label">Dataset selection</div>
            <div className="dataset-toggle-row dataset-toggle-row-header">
              {datasetKeys.map((datasetKey) => {
                const runtimeManifest = datasetManifests[datasetKey];
                const label = metadata?.datasets?.[datasetKey]?.dataset_label ?? formatLabel(datasetKey);
                return (
                  <button
                    key={datasetKey}
                    type="button"
                    className={`toggle-button dataset-toggle-button dataset-toggle-button-header ${activeDatasetKey === datasetKey ? "toggle-button-active" : ""}`}
                    disabled={!runtimeManifest}
                    onClick={() => requestDatasetDate(datasetKey, selectedDate, "nearest", { forceDataset: true })}
                  >
                    <span className="dataset-toggle-label">{label}</span>
                    <span className="dataset-toggle-summary">{DATASET_SUMMARIES[datasetKey] ?? ""}</span>
                  </button>
                );
              })}
            </div>
          </section>
        </div>
      </header>
      <aside className="control-rail">
        <section className="rail-card rail-card-layer">
          <div className="panel-label">Map Layer</div>
          <div className="legend-wrap legend-wrap-primary">
            <div
              className="legend-bar"
              style={{ background: activeLayer ? legendGradient(activeLayer) : undefined }}
            />
            <div className="legend-axis">
              {buildLegendTicks(activeLayer, activeLayerKey).map((tick) => (
                <div
                  key={`${tick.position}-${tick.label}`}
                  className="legend-tick"
                  style={{ left: `${tick.position}%` }}
                >
                  <span className="legend-tick-mark" />
                  <span className="legend-tick-label">{tick.label}</span>
                  {tick.subLabel ? <span className="legend-tick-sub-label">{tick.subLabel}</span> : null}
                </div>
              ))}
            </div>
          </div>
          <div className="toggle-row layer-toggle-row" aria-label="Map layer selector">
            {layerEntries.map(([layerKey, layer]) => {
              const disabled = isAnomalyLayer(layerKey) && !supportsAnomaly;
              return (
                <button
                  key={layerKey}
                  type="button"
                  className={`toggle-button layer-toggle-button ${activeLayerKey === layerKey ? "toggle-button-active" : ""}`}
                  disabled={disabled}
                  onClick={() => handleLayerChange(layerKey)}
                >
                  {layer.label ?? formatLabel(layerKey)}
                </button>
              );
            })}
            {!supportsAnomaly ? (
              <button
                type="button"
                className="toggle-button layer-toggle-button"
                disabled
              >
                LFMC anomaly
              </button>
            ) : null}
          </div>
        </section>
        <section className="rail-card">
          <div className="panel-label">Date</div>
          <div className="date-slider-control">
            <label className="date-input-field">
              <input
                className="location-input date-input picker-only-date"
                type="date"
                value={selectedDate !== "NA" ? selectedDate : ""}
                min={globalDates[0] ?? undefined}
                max={globalDates[globalDates.length - 1] ?? undefined}
                disabled={!globalDates.length}
                onClick={showDatePicker}
                onBeforeInput={preventManualDateEdit}
                onChange={(event) => requestDateValueTransition(event.target.value)}
                onDrop={preventManualDateEdit}
                onKeyDown={preventManualDateEdit}
                onPaste={preventManualDateEdit}
              />
            </label>
            <div className="date-slider-stack">
              <input
                className="date-slider"
                type="range"
                min="0"
                max={Math.max(globalDates.length - 1, 0)}
                step="1"
                value={globalDateIndex}
                disabled={!globalDates.length}
                onChange={(event) => {
                  setIsPlaying(false);
                  requestDateValueTransition(globalDates[Number(event.target.value)]);
                }}
              />
              <div className="slider-extents date-slider-extents">
                <span>{globalDates[0] ?? "--"}</span>
                <span>{globalDates[globalDates.length - 1] ?? "--"}</span>
              </div>
            </div>
          </div>
          <button
            type="button"
            className={`toggle-button play-button ${isPlaying ? "toggle-button-active" : ""}`}
            disabled={dates.length < 2}
            onClick={() => setIsPlaying((currentValue) => !currentValue)}
          >
            {isPlaying ? "Pause" : "Play"}
          </button>
          <div className="date-step-controls" aria-label="Date step controls">
            {dateStepControls.map(([amount, unit, label]) => (
              <button
                type="button"
                className="toggle-button date-step-button"
                disabled={!dates.length}
                onClick={() => handleDateStep(amount, unit)}
                key={`${amount}-${unit}-${label}`}
              >
                {label}
              </button>
            ))}
          </div>
        </section>
        <section className="rail-card workflow-card">
          <div className="detail-tabs" aria-label="Viewer detail panels">
            {DETAIL_TABS.map(([tabKey, label]) => (
              <button
                key={tabKey}
                type="button"
                className={`detail-tab-button ${activeDetailTab === tabKey ? "detail-tab-button-active" : ""}`}
                onClick={() => setActiveDetailTab(tabKey)}
              >
                {label}
              </button>
            ))}
          </div>

          <section className="detail-tab-content" hidden={activeDetailTab !== "timeseries"}>
            <section className="workflow-section">
            <div className="panel-label">View timeseries</div>
            <div className="timeseries-shell">
              {isPointHistoryLoading ? (
                <div className="timeseries-history-status">Loading comparisons from other years...</div>
              ) : null}
              <TimeseriesChart
                pointInfo={pointInfo}
                mode={timeseriesMode}
                supportsAnomaly={supportsAnomaly}
              />
              {isPointLoading ? <div className="timeseries-play-overlay">loading</div> : null}
              {!isPointLoading && isPlaying ? (
                <div className="timeseries-play-overlay">will update after play</div>
              ) : null}
            </div>
            </section>

            <section className="workflow-section">
            <div className="panel-label">Clicked cell info</div>
            {pointInfo ? (
              <div className="stats-grid">
                <div>
                  <span className="stats-key">LFMC (%)</span>
                  <span className="stats-value">{formatMetricValue(pointInfo.lfmc_ens_mean, 1, true)}</span>
                </div>
                <div>
                  <span className="stats-key">Average LFMC on this day of year (%)</span>
                  <span className="stats-value">{formatMetricValue(pointInfo.lfmc_climatology_mean, 1, supportsClimatology)}</span>
                </div>
                <div>
                  <span className="stats-key">LFMC Anomaly (%)</span>
                  <span className="stats-value">{formatMetricValue(pointInfo.lfmc_anomaly, 1, supportsAnomaly)}</span>
                </div>
                <div>
                  <span className="stats-key">Dominant land cover</span>
                  <span className="stats-value">{formatLabel(pointInfo.landcover_name)}</span>
                </div>
              </div>
            ) : (
              <p className="panel-note">Click the map to query a viewer grid cell.</p>
            )}
            </section>

            <section className="workflow-section">
            <div className="panel-label">Input a location</div>
            <form onSubmit={handleLocationSubmit}>
              <div className="location-grid">
                <label className="location-field">
                  <span className="stats-key">Latitude</span>
                  <input
                    className="location-input"
                    type="text"
                    inputMode="decimal"
                    value={locationLatInput}
                    onChange={(event) => setLocationLatInput(event.target.value)}
                    placeholder="34.2206"
                  />
                </label>
                <label className="location-field">
                  <span className="stats-key">Longitude</span>
                  <input
                    className="location-input"
                    type="text"
                    inputMode="decimal"
                    value={locationLonInput}
                    onChange={(event) => setLocationLonInput(event.target.value)}
                    placeholder="-119.0504"
                  />
                </label>
              </div>
              <div className="location-actions">
                <button type="submit" className="toggle-button location-button">
                  Snap To Cell
                </button>
              </div>
            </form>
            </section>
          </section>

          <section className="workflow-section" hidden={activeDetailTab !== "download"}>
          <div className="panel-label">Download data</div>
          <p className="panel-note download-note">
            This tool downloads the currently selected LFMC dataset for up to 10 sites and up to three years at
            each site. For larger downloads, please see{" "}
            <a href={PRODUCT_DOC_URL} target="_blank" rel="noreferrer">
              this information document
            </a>
            .
          </p>
          {downloadSites.map((site, index) => (
            <div
              className={`download-site-block ${index === activeDownloadSiteIndex ? "download-site-block-active" : ""}`}
              key={`download-site-${index}`}
            >
              <div className="download-site-header">
                <span className="stats-key">Site {index + 1}</span>
                <div className="download-site-actions">
                  {index === activeDownloadSiteIndex ? (
                    <span className="download-site-active">Active</span>
                  ) : null}
                  {downloadSites.length > 1 ? (
                    <button
                      type="button"
                      className="toggle-button download-site-remove"
                      onClick={() => handleRemoveDownloadSite(index)}
                    >
                      Remove
                    </button>
                  ) : null}
                </div>
              </div>
              <div className="location-grid">
                <label className="location-field">
                  <span className="stats-key">Latitude</span>
                  <input
                    className="location-input"
                    type="text"
                    inputMode="decimal"
                    value={site.lat}
                    onChange={(event) => handleUpdateDownloadSite(index, "lat", event.target.value)}
                    onFocus={() => setActiveDownloadSiteIndex(index)}
                    placeholder="34.2206"
                  />
                </label>
                <label className="location-field">
                  <span className="stats-key">Longitude</span>
                  <input
                    className="location-input"
                    type="text"
                    inputMode="decimal"
                    value={site.lon}
                    onChange={(event) => handleUpdateDownloadSite(index, "lon", event.target.value)}
                    onFocus={() => setActiveDownloadSiteIndex(index)}
                    placeholder="-119.0504"
                  />
                </label>
              </div>
              <div className="download-date-grid">
                <label className="location-field">
                  <span className="stats-key">Start Date</span>
                  <input
                    className="location-input picker-only-date"
                    type="date"
                    value={site.startDate}
                    min={dates[0] ?? undefined}
                    max={site.endDate || dates[dates.length - 1] || undefined}
                    onClick={showDatePicker}
                    onBeforeInput={preventManualDateEdit}
                    onChange={(event) => handleUpdateDownloadSite(index, "startDate", event.target.value)}
                    onDrop={preventManualDateEdit}
                    onFocus={() => {
                      setActiveDownloadSiteIndex(index);
                    }}
                    onKeyDown={preventManualDateEdit}
                    onPaste={preventManualDateEdit}
                  />
                </label>
                <label className="location-field">
                  <span className="stats-key">End Date</span>
                  <input
                    className="location-input picker-only-date"
                    type="date"
                    value={site.endDate}
                    min={site.startDate || dates[0] || undefined}
                    max={
                      site.startDate
                        ? defaultDownloadEndDate(dates, site.startDate)
                        : dates[dates.length - 1] ?? undefined
                    }
                    onClick={showDatePicker}
                    onBeforeInput={preventManualDateEdit}
                    onChange={(event) => handleUpdateDownloadSite(index, "endDate", event.target.value)}
                    onDrop={preventManualDateEdit}
                    onFocus={() => {
                      setActiveDownloadSiteIndex(index);
                    }}
                    onKeyDown={preventManualDateEdit}
                    onPaste={preventManualDateEdit}
                  />
                </label>
              </div>
            </div>
          ))}
          <div className="location-actions">
            <button
              type="button"
              className="toggle-button location-button"
              disabled={downloadSites.length >= 10}
              onClick={handleAddDownloadSite}
            >
              Add Site
            </button>
          </div>
          <div className="location-actions">
            <button
              type="button"
              className="toggle-button location-button"
              disabled={
                isDownloadingCsv ||
                !manifest ||
                !downloadSites.some(
                  (site) => String(site.lat).trim() !== "" && String(site.lon).trim() !== "",
                )
              }
              onClick={() => {
                void handleDownloadCsv();
              }}
            >
              {isDownloadingCsv ? "Preparing CSV..." : `Download ${manifest?.dataset_label ?? "LFMC"} CSV`}
            </button>
          </div>
          </section>
        </section>
      </aside>

      <main className="map-stage">
        <div className="map-frame">
          {noticeText ? <div className="map-notice">{noticeText}</div> : null}
          {isMapLoading ? <div className="map-loading">loading</div> : null}
          <div ref={mapContainerRef} className="map-container" />
        </div>
      </main>
    </div>
  );
}

export default App;
