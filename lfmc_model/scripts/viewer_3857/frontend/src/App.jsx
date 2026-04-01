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
      anchor: idx === 0 ? "start" : idx === dates.length - 1 ? "end" : "middle",
    }));
}

function buildTimeseriesGeometry(pointInfo, selectedDate) {
  if (!pointInfo?.timeseries) {
    return null;
  }

  const dates = pointInfo.timeseries.dates ?? [];
  const means = pointInfo.timeseries.lfmc_ens_mean ?? [];
  const validPoints = [];

  for (let idx = 0; idx < dates.length; idx += 1) {
    const meanValue = means[idx];
    if (meanValue === null || Number.isNaN(meanValue)) {
      continue;
    }
    validPoints.push({
      idx,
      mean: Number(meanValue),
    });
  }

  if (validPoints.length < 2) {
    return null;
  }

  const width = 352;
  const height = 210;
  const padding = { left: 48, right: 24, top: 18, bottom: 38 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const dataYMin = Math.min(...validPoints.map((point) => point.mean));
  const dataYMax = Math.max(...validPoints.map((point) => point.mean));
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

  const selectedIndex = Math.max(dates.indexOf(selectedDate), 0);
  const selectedX = xCoord(selectedIndex);
  const todayLabelX = Math.min(Math.max(selectedX + 8, padding.left + 18), width - padding.right - 6);
  const xTicks = buildDateTicks(dates).map((tick) => ({
    ...tick,
    x: xCoord(tick.idx),
  }));

  return {
    width,
    height,
    padding,
    linePath,
    selectedX,
    todayLabelX,
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
        <path d={geometry.linePath} className="chart-line" />
        <line
          x1={geometry.selectedX}
          x2={geometry.selectedX}
          y1={geometry.axisTop}
          y2={geometry.axisBottom}
          className="chart-marker"
        />
        <text
          x={geometry.todayLabelX}
          y={geometry.axisTop - 5}
          className="chart-label chart-label-today"
        >
          Today
        </text>
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

function App() {
  const lfmcDisplayOpacity = 0.75;
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const rasterTileLayersRef = useRef([]);
  const activeRasterLayerIndexRef = useRef(0);
  const selectionSourceRef = useRef(null);
  const manifestRef = useRef(null);
  const dateIndexRef = useRef(0);
  const pointRef = useRef(null);
  const transitionTokenRef = useRef(0);
  const queuedDateIndexRef = useRef(null);
  const transitionInFlightRef = useRef(false);
  const playbackTimeoutRef = useRef(null);
  const activeDownloadSiteIndexRef = useRef(0);

  const [manifest, setManifest] = useState(null);
  const [dateIndex, setDateIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [locationLatInput, setLocationLatInput] = useState("");
  const [locationLonInput, setLocationLonInput] = useState("");
  const [downloadStartDate, setDownloadStartDate] = useState("");
  const [downloadEndDate, setDownloadEndDate] = useState("");
  const [downloadSites, setDownloadSites] = useState([{ lat: "", lon: "" }]);
  const [activeDownloadSiteIndex, setActiveDownloadSiteIndex] = useState(0);
  const [isDownloadingCsv, setIsDownloadingCsv] = useState(false);
  const [pointInfo, setPointInfo] = useState(null);
  const [statusText, setStatusText] = useState("Loading viewer manifest...");
  const [isMapLoading, setIsMapLoading] = useState(false);
  const dates = manifest?.dates ?? [];
  const selectedDate = dates[dateIndex] ?? "NA";
  const activeLayer = manifest?.layers?.mean ?? null;

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

  async function loadPointAtCoordinate(x, y, dateStr, options = {}) {
    const { recenter = false } = options;
    setStatusText("Loading clicked cell and time series...");
    const payload = await queryPoint(x, y, dateStr);
    setPointInfo(payload);
    pointRef.current = {
      x: payload.requested_grid_x,
      y: payload.requested_grid_y,
    };
    setDownloadSites((currentSites) =>
      currentSites.map((site, siteIndex) =>
        siteIndex === activeDownloadSiteIndexRef.current
          ? {
              lat: formatCoordinateInput(payload.cell_center_lat),
              lon: formatCoordinateInput(payload.cell_center_lon),
            }
          : site,
      ),
    );
    updateSelectionFeatures(payload, manifestRef.current);
    if (recenter && mapRef.current) {
      mapRef.current.getView().animate({
        center: [payload.requested_grid_x, payload.requested_grid_y],
        duration: 250,
      });
    }
    setStatusText(`Loaded clicked cell for ${dateStr}`);
    return payload;
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

  function createTileSourceForDate(manifestPayload, targetDate, requestToken, onReady) {
    const layerConfig = manifestPayload.layers.mean;
    const tileTemplate = layerConfig?.tile_root_template;
    const assetBaseUrl = manifestPayload.asset_base_url;
    if (!tileTemplate) {
      throw new Error(`No tile template found for mean ${targetDate}`);
    }
    if (!assetBaseUrl) {
      throw new Error(`No asset base URL configured for mean ${targetDate}`);
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
        setStatusText(`Loading LFMC tiles for ${targetDate}`);
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
    const { immediate = false } = options;
    const manifestPayload = manifestRef.current;
    if (!manifestPayload || !mapRef.current) {
      return false;
    }
    if (!immediate && targetIndex === dateIndexRef.current) {
      return true;
    }

    const targetDate = manifestPayload.dates[targetIndex];
    if (!targetDate) {
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
      const tileSource = createTileSourceForDate(manifestPayload, targetDate, requestToken, async () => {
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
        setStatusText(`Showing LFMC for ${targetDate}`);
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
    let cancelled = false;
    const maxAttempts = 8;
    const retryDelayMs = 1000;

    async function loadManifest() {
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        try {
          setStatusText("Starting viewer...");
          const metadataResponse = await fetch("/api/metadata", { cache: "no-store" });
          if (!metadataResponse.ok) {
            throw new Error(`Metadata HTTP ${metadataResponse.status}`);
          }
          const metadata = await metadataResponse.json();
          const manifestResponse = await fetch(metadata.asset_manifest_url, { cache: "no-store" });
          if (!manifestResponse.ok) {
            throw new Error(`Manifest HTTP ${manifestResponse.status}`);
          }
          const payload = await manifestResponse.json();
          if (cancelled) {
            return;
          }
          const runtimeManifest = {
            ...payload,
            asset_base_url: metadata.asset_base_url,
            asset_manifest_url: metadata.asset_manifest_url,
          };
          setManifest(runtimeManifest);
          manifestRef.current = runtimeManifest;
          const initialIndex = Math.max(runtimeManifest.dates.indexOf(runtimeManifest.initial_date), 0);
          setDateIndex(initialIndex);
          dateIndexRef.current = initialIndex;
          setDownloadStartDate((currentValue) => currentValue || runtimeManifest.dates[0] || "");
          setDownloadEndDate(
            (currentValue) => currentValue || runtimeManifest.dates[runtimeManifest.dates.length - 1] || "",
          );
          setStatusText(`Loaded ${runtimeManifest.dataset_label}`);
          return;
        } catch (error) {
          if (cancelled) {
            return;
          }
          if (attempt === maxAttempts) {
            setStatusText(`Manifest load failed: ${error.message}`);
            return;
          }
          await new Promise((resolve) => window.setTimeout(resolve, retryDelayMs));
        }
      }
    }

    loadManifest();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    manifestRef.current = manifest;
  }, [manifest]);

  useEffect(() => {
    dateIndexRef.current = dateIndex;
  }, [dateIndex]);

  useEffect(() => {
    activeDownloadSiteIndexRef.current = activeDownloadSiteIndex;
  }, [activeDownloadSiteIndex]);

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
    const projectionCode = manifest.grid_crs;
    const projection = getProjection(projectionCode);
    if (!projection) {
      setStatusText(`Unsupported map projection ${projectionCode}`);
      return;
    }
    const isWebMercator = projectionCode === "EPSG:3857";
    if (!isWebMercator) {
      projection.setExtent(extent);
    }

    const view = new View({
      projection,
      center: getCenter(extent),
      resolutions: manifest.tiles.view_resolutions,
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
      const selectedDate = currentManifest.dates[dateIndexRef.current];
      const [x, y] = event.coordinate;
      setIsPlaying(false);
      try {
        await loadPointAtCoordinate(x, y, selectedDate);
      } catch (error) {
        setStatusText(`Point query failed: ${error.message}`);
      }
    });

    mapRef.current = map;
    void transitionToDateIndex(dateIndexRef.current, { immediate: true });

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
  }, [manifest]);

  useEffect(() => {
    if (!pointRef.current || !manifest) {
      return;
    }

    const selectedDate = manifest.dates[dateIndex];
    loadPointAtCoordinate(pointRef.current.x, pointRef.current.y, selectedDate)
      .then((payload) => {
        updateSelectionFeatures(payload, manifest);
      })
      .catch((error) => {
        setStatusText(`Point refresh failed: ${error.message}`);
      });
  }, [dateIndex, manifest]);

  useEffect(() => {
    if (!isPlaying || dates.length < 2) {
      return undefined;
    }
    let cancelled = false;

    async function runPlayback() {
      while (!cancelled) {
        const nextIndex = (dateIndexRef.current + 7) % dates.length;
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
        statusPrefix: `Snapping ${latitude.toFixed(4)}, ${longitude.toFixed(4)} to`,
      });
    } catch (error) {
      setStatusText(`Location lookup failed: ${error.message}`);
    }
  }

  async function handleDownloadCsv() {
    if (!manifest) {
      return;
    }
    if (!downloadStartDate || !downloadEndDate) {
      setStatusText("Select a start date and end date before downloading CSV");
      return;
    }
    if (downloadEndDate < downloadStartDate) {
      setStatusText("CSV end date must be on or after the start date");
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
      sites.push({ lat: latitude, lon: longitude });
    }
    if (!sites.length) {
      setStatusText("Select a site on the map or enter a site before downloading");
      return;
    }

    const query = new URLSearchParams({
      start_date: downloadStartDate,
      end_date: downloadEndDate,
    });
    sites.forEach((site) => {
      query.append("site", `${site.lat},${site.lon}`);
    });

    setIsDownloadingCsv(true);
    setStatusText("Preparing scientific CSV download...");
    try {
      const response = await fetch(`/api/download_csv?${query.toString()}`);
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
      setStatusText(`Downloaded scientific CSV for ${downloadStartDate} to ${downloadEndDate}`);
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
      return [...currentSites, { lat: "", lon: "" }];
    });
    setActiveDownloadSiteIndex((currentValue) => Math.min(currentValue + 1, 9));
  }

  function handleUpdateDownloadSite(index, field, value) {
    setDownloadSites((currentSites) =>
      currentSites.map((site, siteIndex) =>
        siteIndex === index
          ? {
              ...site,
              [field]: value,
            }
          : site,
      ),
    );
  }

  function handleRemoveDownloadSite(index) {
    setDownloadSites((currentSites) => {
      const nextSites =
        currentSites.length === 1
          ? [{ lat: "", lon: "" }]
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

  return (
    <div className="app-shell">
      <aside className="control-panel">
        <h1>Viewer for long-term LFMC dataset</h1>
        <div className="status-line">{statusText}</div>

        <section className="panel-card">
          <div className="panel-label">LFMC (%)</div>
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
          <div className="panel-label">Date</div>
          <div className="date-row">
            <div className="date-value">{selectedDate}</div>
            <div className={`pill ${isMapLoading ? "pill-loading" : ""}`}>
              {isMapLoading ? "Loading" : "Ready"}
            </div>
          </div>
          <div className="date-toolbar">
            <button
              type="button"
              className={`toggle-button play-button ${isPlaying ? "toggle-button-active" : ""}`}
              disabled={dates.length < 2}
              onClick={() => setIsPlaying((currentValue) => !currentValue)}
            >
              {isPlaying ? "Pause" : "Play"}
            </button>
            <input
              className="date-slider"
              type="range"
              min="0"
              max={Math.max(dates.length - 1, 0)}
              step="1"
              value={dateIndex}
              disabled={!dates.length}
              onChange={(event) => {
                setIsPlaying(false);
                requestDateTransition(Number(event.target.value));
              }}
            />
          </div>
          <div className="slider-extents">
            <span>{dates[0] ?? "--"}</span>
            <span>{dates[dates.length - 1] ?? "--"}</span>
          </div>
        </section>

        <section className="panel-card">
          <div className="panel-label">Location</div>
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

        <section className="panel-card">
          <div className="panel-label">Clicked Cell</div>
          {pointInfo ? (
            <div className="stats-grid">
              <div>
                <span className="stats-key">LFMC</span>
                <span className="stats-value">{formatValue(pointInfo.lfmc_ens_mean, 1)}</span>
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
            <p className="panel-note">Click the map to query a viewer grid cell.</p>
          )}
        </section>

        <section className="panel-card">
          <div className="panel-label">Time Series</div>
          <TimeseriesChart pointInfo={pointInfo} selectedDate={selectedDate} />
        </section>

        <section className="panel-card">
          <div className="panel-label">Download CSV</div>
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
            </div>
          ))}
          <div className="location-grid">
            <label className="location-field">
              <span className="stats-key">Start Date</span>
              <input
                className="location-input"
                type="date"
                value={downloadStartDate}
                min={dates[0] ?? undefined}
                max={dates[dates.length - 1] ?? undefined}
                onChange={(event) => setDownloadStartDate(event.target.value)}
              />
            </label>
            <label className="location-field">
              <span className="stats-key">End Date</span>
              <input
                className="location-input"
                type="date"
                value={downloadEndDate}
                min={dates[0] ?? undefined}
                max={dates[dates.length - 1] ?? undefined}
                onChange={(event) => setDownloadEndDate(event.target.value)}
              />
            </label>
          </div>
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
              {isDownloadingCsv ? "Preparing CSV..." : "Download .CSVs For These Sites"}
            </button>
          </div>
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
