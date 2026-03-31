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
import { get as getProjection } from "ol/proj";
import { register } from "ol/proj/proj4";
import { Circle as CircleStyle, Fill, Stroke, Style } from "ol/style";

const EPSG5070_DEF =
  "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +type=crs";

proj4.defs("EPSG:5070", EPSG5070_DEF);
register(proj4);

function formatValue(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "NA";
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
  const stops = layerConfig.stops ?? [];
  const palette = layerConfig.palette ?? [];
  if (!stops.length || !palette.length) {
    return "linear-gradient(to right, #6d271d 0%, #c9794a 24%, #d2c487 52%, #6d8e60 76%, #253b36 100%)";
  }
  const pieces = palette.map((color, idx) => {
    const pct = `${Math.round(Number(stops[idx]) * 100)}%`;
    return `rgb(${color.join(",")}) ${pct}`;
  });
  return `linear-gradient(to right, ${pieces.join(", ")})`;
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

function buildDateTicks(dates) {
  if (!dates.length) {
    return [];
  }

  const candidateIndices = [0, Math.floor((dates.length - 1) / 2), dates.length - 1];
  const seen = new Set();

  return candidateIndices
    .filter((idx) => {
      if (seen.has(idx)) {
        return false;
      }
      seen.add(idx);
      return true;
    })
    .map((idx) => ({
      idx,
      label: dates[idx],
    }));
}

function buildTimeseriesGeometry(pointInfo, selectedDate) {
  if (!pointInfo?.timeseries) {
    return null;
  }

  const dates = pointInfo.timeseries.dates ?? [];
  const means = pointInfo.timeseries.lfmc_ens_mean ?? [];
  const stds = pointInfo.timeseries.lfmc_ens_std ?? [];
  const validPoints = [];

  for (let idx = 0; idx < dates.length; idx += 1) {
    const meanValue = means[idx];
    const stdValue = stds[idx];
    if (meanValue === null || stdValue === null || Number.isNaN(meanValue) || Number.isNaN(stdValue)) {
      continue;
    }
    validPoints.push({
      idx,
      mean: Number(meanValue),
      low: Number(meanValue) - Number(stdValue),
      high: Number(meanValue) + Number(stdValue),
    });
  }

  if (validPoints.length < 2) {
    return null;
  }

  const width = 340;
  const height = 210;
  const padding = { left: 48, right: 12, top: 12, bottom: 38 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const dataYMin = Math.min(...validPoints.map((point) => point.low));
  const dataYMax = Math.max(...validPoints.map((point) => point.high));
  const yTicks = buildAxisTicks(dataYMin, dataYMax);
  const yMin = yTicks[0].value;
  const yMax = yTicks[yTicks.length - 1].value;
  const ySpan = Math.max(yMax - yMin, 1e-6);
  const xDenominator = Math.max(dates.length - 1, 1);

  const xCoord = (index) => padding.left + (index / xDenominator) * innerWidth;
  const yCoord = (value) => padding.top + ((yMax - value) / ySpan) * innerHeight;

  const linePath = validPoints
    .map((point, idx) => `${idx === 0 ? "M" : "L"} ${xCoord(point.idx)} ${yCoord(point.mean)}`)
    .join(" ");

  const upperPath = validPoints
    .map((point, idx) => `${idx === 0 ? "M" : "L"} ${xCoord(point.idx)} ${yCoord(point.high)}`)
    .join(" ");
  const lowerPath = [...validPoints]
    .reverse()
    .map((point) => `L ${xCoord(point.idx)} ${yCoord(point.low)}`)
    .join(" ");
  const areaPath = `${upperPath} ${lowerPath} Z`;

  const selectedIndex = Math.max(dates.indexOf(selectedDate), 0);
  const selectedX = xCoord(selectedIndex);
  const xTicks = buildDateTicks(dates).map((tick) => ({
    ...tick,
    x: xCoord(tick.idx),
  }));

  return {
    width,
    height,
    padding,
    linePath,
    areaPath,
    selectedX,
    axisLeft: padding.left,
    axisRight: width - padding.right,
    axisTop: padding.top,
    axisBottom: height - padding.bottom,
    yTicks: yTicks.map((tick) => ({
      ...tick,
      y: padding.top + tick.fraction * innerHeight,
    })),
    xTicks,
  };
}

function TimeseriesChart({ pointInfo, selectedDate }) {
  const geometry = buildTimeseriesGeometry(pointInfo, selectedDate);

  if (!geometry) {
    return <p className="panel-note">Click the map to load a full time series.</p>;
  }

  return (
    <div className="timeseries-wrap">
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
        <path d={geometry.areaPath} className="chart-band" />
        <path d={geometry.linePath} className="chart-line" />
        <line
          x1={geometry.selectedX}
          x2={geometry.selectedX}
          y1={geometry.axisTop}
          y2={geometry.axisBottom}
          className="chart-marker"
        />
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
          <g key={`x-${tick.idx}`}>
            <line
              x1={tick.x}
              x2={tick.x}
              y1={geometry.axisBottom}
              y2={geometry.axisBottom + 5}
              className="chart-tick"
            />
            <text x={tick.x} y={geometry.axisBottom + 18} className="chart-label chart-label-x">
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
          LFMC (%)
        </text>
      </svg>
    </div>
  );
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

function cellBoundsFromIndex(cellIndex, manifest) {
  const dx = Number(manifest.grid_resolution.dx);
  const dy = Number(manifest.grid_resolution.dy);
  const west = Number(manifest.grid_extent.west) + cellIndex.x * dx;
  const east = west + dx;
  const north = Number(manifest.grid_extent.north) - cellIndex.y * dy;
  const south = north - dy;
  return { west, east, south, north };
}

function App() {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const rasterLayerRef = useRef(null);
  const selectionSourceRef = useRef(null);
  const manifestRef = useRef(null);
  const dateIndexRef = useRef(0);
  const pointRef = useRef(null);

  const [manifest, setManifest] = useState(null);
  const [dateIndex, setDateIndex] = useState(0);
  const [layerKey, setLayerKey] = useState("mean");
  const [pointInfo, setPointInfo] = useState(null);
  const [statusText, setStatusText] = useState("Loading viewer manifest...");
  const [isMapLoading, setIsMapLoading] = useState(false);

  async function queryPoint(x, y, dateStr) {
    const query = new URLSearchParams({
      date: dateStr,
      x: String(x),
      y: String(y),
    });
    const response = await fetch(`/api/point?${query.toString()}`);
    const payload = await response.json();
    if (payload.error) {
      throw new Error(payload.error);
    }
    return payload;
  }

  function updateSelectionFeatures(payload, manifestPayload) {
    const selectionSource = selectionSourceRef.current;
    if (!selectionSource) {
      return;
    }
    const cellBounds = cellBoundsFromIndex(payload.cell_index, manifestPayload);
    selectionSource.clear();
    selectionSource.addFeature(
      new Feature({
        geometry: new Point([payload.requested_grid_x, payload.requested_grid_y]),
        role: "click_point",
      }),
    );
    selectionSource.addFeature(
      new Feature({
        geometry: new Polygon(cellPolygonCoordinates(cellBounds)),
        role: "cell_outline",
      }),
    );
  }

  useEffect(() => {
    async function loadManifest() {
      const response = await fetch("/viewer-assets/manifest.json");
      const payload = await response.json();
      setManifest(payload);
      manifestRef.current = payload;
      const initialIndex = Math.max(payload.dates.indexOf(payload.initial_date), 0);
      setDateIndex(initialIndex);
      dateIndexRef.current = initialIndex;
      setStatusText(`Loaded ${payload.dataset_label}`);
    }

    loadManifest().catch((error) => {
      setStatusText(`Manifest load failed: ${error.message}`);
    });
  }, []);

  useEffect(() => {
    manifestRef.current = manifest;
  }, [manifest]);

  useEffect(() => {
    dateIndexRef.current = dateIndex;
  }, [dateIndex]);

  useEffect(() => {
    if (!manifest || mapRef.current || !mapContainerRef.current) {
      return;
    }

    const extent = [
      manifest.grid_extent.west,
      manifest.grid_extent.south,
      manifest.grid_extent.east,
      manifest.grid_extent.north,
    ];
    const projection = getProjection("EPSG:5070");
    projection.setExtent(extent);

    const view = new View({
      projection,
      center: getCenter(extent),
      resolutions: manifest.tiles.view_resolutions,
      constrainResolution: true,
      extent,
      showFullExtent: true,
      zoom: 0,
    });

    const rasterLayer = new TileLayer();
    rasterLayerRef.current = rasterLayer;

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

    const map = new Map({
      target: mapContainerRef.current,
      layers: [rasterLayer, selectionLayer],
      view,
      controls: defaultControls().extend([new ScaleLine()]),
    });

    map.getView().fit(extent, { padding: [24, 24, 24, 24], duration: 0 });

    map.on("click", async (event) => {
      const currentManifest = manifestRef.current;
      const selectedDate = currentManifest.dates[dateIndexRef.current];
      const [x, y] = event.coordinate;
      setStatusText(`Querying ${selectedDate} at x=${x.toFixed(0)}, y=${y.toFixed(0)}`);
      try {
        const payload = await queryPoint(x, y, selectedDate);
        setPointInfo(payload);
        pointRef.current = {
          x: payload.requested_grid_x,
          y: payload.requested_grid_y,
        };
        updateSelectionFeatures(payload, currentManifest);
        setStatusText(`Loaded exact cell query for ${selectedDate}`);
      } catch (error) {
        setStatusText(`Point query failed: ${error.message}`);
      }
    });

    mapRef.current = map;

    return () => {
      map.setTarget(null);
      mapRef.current = null;
      rasterLayerRef.current = null;
      selectionSourceRef.current = null;
    };
  }, [manifest]);

  useEffect(() => {
    if (!manifest || !mapRef.current || !rasterLayerRef.current) {
      return;
    }

    const selectedDate = manifest.dates[dateIndex];
    const layerConfig = manifest.layers[layerKey];
    const tileTemplate = layerConfig?.tile_root_template;
    if (!tileTemplate) {
      setStatusText(`No native-grid tile template found for ${layerKey} ${selectedDate}`);
      return;
    }

    const extent = [
      manifest.grid_extent.west,
      manifest.grid_extent.south,
      manifest.grid_extent.east,
      manifest.grid_extent.north,
    ];
    const tileGrid = new TileGrid({
      extent,
      origin: manifest.tiles.origin,
      resolutions: manifest.tiles.resolutions,
      tileSize: manifest.tiles.tile_size,
    });

    const tileCounts = layerConfig.tile_counts ?? {};
    let pendingTiles = 0;
    const source = new XYZ({
      projection: "EPSG:5070",
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
        const relpath = tileTemplate
          .replace("{date}", selectedDate)
          .replace("{z}", String(z))
          .replace("{x}", String(x))
          .replace("{y}", String(y));
        return `${window.location.origin}/viewer-assets/${relpath}`;
      },
    });

    source.on("tileloadstart", () => {
      pendingTiles += 1;
      setIsMapLoading(true);
      setStatusText(`Loading ${layerConfig.label} tiles for ${selectedDate}`);
    });
    const settleTile = () => {
      pendingTiles = Math.max(0, pendingTiles - 1);
      if (pendingTiles === 0) {
        setIsMapLoading(false);
        setStatusText(`Showing ${layerConfig.label} for ${selectedDate}`);
      }
    };
    source.on("tileloadend", settleTile);
    source.on("tileloaderror", settleTile);
    rasterLayerRef.current.setSource(source);
  }, [manifest, dateIndex, layerKey]);

  useEffect(() => {
    if (!pointRef.current || !manifest) {
      return;
    }

    const selectedDate = manifest.dates[dateIndex];
    queryPoint(pointRef.current.x, pointRef.current.y, selectedDate)
      .then((payload) => {
        setPointInfo(payload);
        updateSelectionFeatures(payload, manifest);
      })
      .catch((error) => {
        setStatusText(`Point refresh failed: ${error.message}`);
      });
  }, [dateIndex, manifest]);

  const dates = manifest?.dates ?? [];
  const selectedDate = dates[dateIndex] ?? "NA";
  const activeLayer = manifest?.layers?.[layerKey] ?? null;

  return (
    <div className="app-shell">
      <aside className="control-panel">
        <h1>Viewer for long-term LFMC dataset</h1>
        <div className="status-line">{statusText}</div>

        <section className="panel-card">
          <div className="panel-label">Layer</div>
          <div className="toggle-row">
            {Object.entries(manifest?.layers ?? {}).map(([key, layer]) => (
              <button
                key={key}
                type="button"
                className={`toggle-button ${key === layerKey ? "toggle-button-active" : ""}`}
                onClick={() => setLayerKey(key)}
              >
                {layer.label}
              </button>
            ))}
          </div>
        </section>

        <section className="panel-card">
          <div className="panel-label">Date</div>
          <div className="date-row">
            <div className="date-value">{selectedDate}</div>
            <div className={`pill ${isMapLoading ? "pill-loading" : ""}`}>
              {isMapLoading ? "Loading" : "Ready"}
            </div>
          </div>
          <input
            className="date-slider"
            type="range"
            min="0"
            max={Math.max(dates.length - 1, 0)}
            step="1"
            value={dateIndex}
            disabled={!dates.length}
            onChange={(event) => setDateIndex(Number(event.target.value))}
          />
          <div className="slider-extents">
            <span>{dates[0] ?? "--"}</span>
            <span>{dates[dates.length - 1] ?? "--"}</span>
          </div>
        </section>

        <section className="panel-card">
          <div className="panel-label">Legend</div>
          <div
            className="legend-bar"
            style={{ background: activeLayer ? legendGradient(activeLayer) : undefined }}
          />
          <div className="slider-extents">
            <span>{formatValue(activeLayer?.min, 0)}</span>
            <span>{formatValue(activeLayer?.max, 0)}</span>
          </div>
        </section>

        <section className="panel-card">
          <div className="panel-label">Clicked Cell</div>
          {pointInfo ? (
            <div className="stats-grid">
              <div>
                <span className="stats-key">LFMC</span>
                <span className="stats-value">{formatValue(pointInfo.lfmc_ens_mean, 1)}</span>
              </div>
              <div>
                <span className="stats-key">Uncertainty</span>
                <span className="stats-value">{formatValue(pointInfo.lfmc_ens_std, 1)}</span>
              </div>
              <div>
                <span className="stats-key">Center Lat</span>
                <span className="stats-value">{formatValue(pointInfo.cell_center_lat, 4)}</span>
              </div>
              <div>
                <span className="stats-key">Center Lon</span>
                <span className="stats-value">{formatValue(pointInfo.cell_center_lon, 4)}</span>
              </div>
              <div>
                <span className="stats-key">Land Cover</span>
                <span className="stats-value">{formatLabel(pointInfo.landcover_name)}</span>
              </div>
              <div>
                <span className="stats-key">Product Level</span>
                <span className="stats-value">{formatLabel(pointInfo.data_product_level)}</span>
              </div>
            </div>
          ) : (
            <p className="panel-note">Click the map to query an exact 500 m grid cell.</p>
          )}
        </section>

        <section className="panel-card">
          <div className="panel-label">Time Series</div>
          <TimeseriesChart pointInfo={pointInfo} selectedDate={selectedDate} />
        </section>
      </aside>

      <main className="map-stage">
        <div className="map-frame">
          <div ref={mapContainerRef} className="map-container" />
        </div>
      </main>
    </div>
  );
}

export default App;
