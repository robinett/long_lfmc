import os,time,json,pathlib,requests,math,shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import ee
import rasterio
from rasterio.merge import merge as rio_merge

def img_basename(img, idx):
    """Prefer system:index, fallback to time_start or sequence."""
    im = ee.Image(img)
    try:
        sid = im.get("system:index").getInfo()
        if sid:
            return sid.replace("/", "_")
    except Exception:
        pass
    try:
        ts = im.get("system:time_start").getInfo()
        if ts:
            import datetime as dt
            d = dt.datetime.utcfromtimestamp(ts / 1000)
            return d.strftime("%Y%m%d")
    except Exception:
        pass
    return f"image_{idx:04d}"

def native_scale(img):
    """Return native nominal scale (meters) for the image’s default band."""
    proj = ee.Image(img).projection()
    return proj.nominalScale().getInfo()

def img_bbox(img):
    """Return (minx, miny, maxx, maxy) degrees for the image footprint."""
    geom = ee.Image(img).geometry()
    b = ee.Feature(ee.Geometry(geom.bounds(1))).geometry().bounds(1)
    coords = b.coordinates().getInfo()[0]
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))

def tiles_from_bbox(minx, miny, maxx, maxy, step_deg):
    """Yield tiles (row, col, ee.Geometry.Rectangle) covering bbox."""
    step = abs(float(step_deg))
    cols = max(1, math.ceil((maxx - minx) / step))
    rows = max(1, math.ceil((maxy - miny) / step))
    for r in range(rows):
        y0 = miny + r * step
        y1 = min(y0 + step, maxy)
        for c in range(cols):
            x0 = minx + c * step
            x1 = min(x0 + step, maxx)
            rect = ee.Geometry.Rectangle([x0, y0, x1, y1], geodesic=False)
            yield r, c, rect

def tile_output_path(base_dir, name, r, c):
    return os.path.join(base_dir, f"{name}__tile_r{r}_c{c}.tif")

def download_tile(img, name, r, c, region_geom, scale_m, out_path,
                  max_retries=5, max_pixels=1_000_000_000_000, request_timeout=600):
    """Download one tile GeoTIFF for the given image and region."""
    if os.path.exists(out_path):
        return "exists"

    params = {
        # Preserve native: we just ask EE to render at the image's nominal scale.
        "scale": scale_m,
        "region": json.dumps(region_geom.getInfo()),
        "format": "GEO_TIFF",
        "filePerBand": False,
        "maxPixels": max_pixels,
    }

    url = ee.Image(img).getDownloadURL(params)

    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=request_timeout) as r_:
                r_.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r_.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            return "ok"
        except Exception as e:
            if attempt == max_retries:
                return f"fail: {e}"
            time.sleep(backoff)
            backoff *= 1.7

def mosaic_tiles_to_single(tile_paths, out_file):
    """Mosaic all tile GeoTIFFs into a single file, then close/done."""
    # Filter existing tiles
    tile_paths = [p for p in tile_paths if os.path.exists(p)]
    if not tile_paths:
        raise RuntimeError("No tile files found to mosaic.")

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, out_transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        })
    finally:
        for s in srcs:
            s.close()

    # Write the mosaic
    with rasterio.open(out_file, "w", **meta) as dst:
        dst.write(mosaic)

def process_image(img, idx, out_dir, tile_deg, max_workers=4):
    """Download all tiles for one image, then stitch to a single file."""
    name = img_basename(img, idx)
    final_out = os.path.join(out_dir, f"{name}.tif")
    if os.path.exists(final_out):
        print(f"{name}: final exists, skipping")
        return name, "final_exists"
    scale_m = native_scale(img)
    minx, miny, maxx, maxy = img_bbox(img)
    # Temp dir for this image's tiles
    img_tmp_dir = os.path.join(out_dir, f"._tiles_{name}")
    pathlib.Path(img_tmp_dir).mkdir(parents=True, exist_ok=True)
    # Fan out tile downloads
    tile_paths = []
    futures = []
    ok = exists = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r, c, rect in tiles_from_bbox(minx, miny, maxx, maxy, tile_deg):
            #print(r)
            #print(c)
            #print(rect)
            tpath = tile_output_path(img_tmp_dir, name, r, c)
            tile_paths.append(tpath)
            futures.append(
                ex.submit(download_tile, img, name, r, c, rect, scale_m, tpath)
            )
        for fut in as_completed(futures):
            status = fut.result()
            if status == "ok": ok += 1
            elif status == "exists": exists += 1
            else: fail += 1
    print(f"{name}: tiles ok={ok}, exists={exists}, fail={fail}")
    if ok == 0 and exists == 0:
        # Nothing downloaded; clean temp dir and abort
        shutil.rmtree(img_tmp_dir, ignore_errors=True)
        raise RuntimeError(f"{name}: no tiles downloaded")
    # Stitch to final single GeoTIFF
    mosaic_tiles_to_single(tile_paths, final_out)
    print(f"{name}: stitched to {final_out}")
    # Optional: cleanup tile dir to save space
    shutil.rmtree(img_tmp_dir, ignore_errors=True)
    return name, "done"

def main():
    ee.Authenticate()
    ee.Initialize(project='long-lfmc')
    start_date = '2016-01-01'
    end_date = '2022-01-01'
    scale = 250
    out_dir = '/scratch/users/trobinet/long_lfmc/trent_datasets/krishna/krishna_raw_from_gee_api'
    col_asset = (
        '/users/kkraoj/lfm-mapper/lfmc_col_25_may_2021'
    )
    tile_deg = 5
    # get the image collection
    col = ee.ImageCollection(
        'users/kkraoj/lfm-mapper/lfmc_col_25_may_2021'
    ).filterDate(start_date, end_date)
    n = col.size().getInfo()
    imgs = col.toList(n)
    print(f"Found {n} images in collection.")
    done = 0
    skipped = 0
    errors = 0
    for i in range(n):
        try:
            name,status = process_image(imgs.get(i), i, out_dir, tile_deg)
            if status == "done":
                done += 1
            elif status == "final_exists":
                skipped += 1
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            errors += 1
    print(f"All done: done={done}, skipped={skipped}, errors={errors}")

if __name__ == "__main__":
    main()