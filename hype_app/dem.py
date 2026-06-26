"""Fetch a USGS 3DEP DEM covering the drawn domain and save it as a GeoTIFF.

3DEP DEMs are hydro-flattened, so by default this same DEM serves as BOTH the terrain
and the water-surface elevation in run_hyporheic (the caller may override WSE later).
"""
from __future__ import annotations

from pathlib import Path

import py3dep
from shapely.geometry import box

# Keep the fetch in step with hype_app.estimate.estimate_cells (same buffer).
BUFFER_FRAC = 0.12
_RES_PRIORITY = (1, 3, 5, 10, 30)     # finest-first; 3DEP metres (1/3/5 m are lidar-only)
_MAX_PIXELS = 7_000_000               # stay under py3dep's ~8 M dynamic-service cap


def _candidate_resolutions(aoi_bounds, requested) -> list:
    """Ordered 3DEP resolutions (m) to try. ``requested='auto'`` → finest available first (via
    ``check_3dep_availability``); an explicit value → just that. 10 m then 30 m are always appended
    as nationwide last-resorts, so a fetch always has something to fall back to."""
    order = []
    if requested in (None, "auto", "Auto"):
        try:
            avail = py3dep.check_3dep_availability(tuple(float(b) for b in aoi_bounds))
        except Exception:  # noqa: BLE001 — service hiccup → just try finest-first
            avail = {}
        order = [r for r in _RES_PRIORITY if avail.get(f"{r}m") is True]
        if not order:                 # availability unknown/failed → try everything, finest-first
            order = list(_RES_PRIORITY)
    else:
        order = [int(requested)]
    for r in (10, 30):                # nationwide safety net
        if r not in order:
            order.append(r)
    return order


def fetch_dem(domain_gdf_4326, out_path, resolution="auto", buffer_frac: float = BUFFER_FRAC):
    """Download the finest available USGS 3DEP DEM over the domain bbox (+buffer) and write it to
    `out_path` as a GeoTIFF. `resolution` is ``"auto"`` (finest 3DEP available — 1 m where lidar
    exists) or an explicit metre value (1/3/5/10/30). Resolutions whose request would exceed the
    3DEP pixel cap are skipped, and the fetch falls back to coarser data on any failure. Returns
    ``{"path", "resolution_m", "source"}``."""
    import geopandas as gpd

    minx, miny, maxx, maxy = (float(v) for v in domain_gdf_4326.total_bounds)
    dx, dy = (maxx - minx) * buffer_frac, (maxy - miny) * buffer_frac
    aoi = box(minx - dx, miny - dy, maxx + dx, maxy + dy)
    mb = gpd.GeoSeries([aoi], crs=4326).to_crs(5070).iloc[0].bounds       # metric size for the budget
    w_m, h_m = (mb[2] - mb[0]), (mb[3] - mb[1])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for res in _candidate_resolutions(aoi.bounds, resolution):
        if (w_m / res) * (h_m / res) > _MAX_PIXELS:
            errors.append((res, "exceeds the 3DEP pixel budget")); continue
        try:
            dem = py3dep.get_dem(aoi, res)
            if dem.rio.nodata is None:
                dem = dem.rio.write_nodata(-9999.0)
            dem.rio.to_raster(out_path)
            return {"path": str(out_path), "resolution_m": int(res), "source": "USGS 3DEP"}
        except Exception as e:  # noqa: BLE001
            errors.append((res, str(e)[:140])); continue
    raise RuntimeError(f"3DEP DEM fetch failed at all candidate resolutions: {errors}")


def clip_dem_to_polygon(dem_path, polygon_gdf, out_path, nodata: float = -9999.0):
    """Write a WSE raster = DEM elevations INSIDE the polygon, `nodata` everywhere else.

    The "draw the wetted extent" workflow: the user draws the water-surface extent and we hand
    the engine a WSE raster (DEM values inside the polygon) exactly as if it had been uploaded.
    Cells outside the polygon — and any non-finite DEM pixels inside it — become `nodata`
    (-9999, matching the engine's WSE `!= -9999` / `>= 0` filters) so they never become CHD
    cells (and never re-introduce a NaN). Returns the output path.
    """
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask

    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float32")
        poly = polygon_gdf.to_crs(src.crs)
        inside = geometry_mask(
            [geom.__geo_interface__ for geom in poly.geometry],
            out_shape=arr.shape, transform=src.transform, invert=True,
        )
        out = np.where(inside & np.isfinite(arr), arr, np.float32(nodata)).astype("float32")
        meta = src.meta.copy()
    meta.update(driver="GTiff", dtype="float32", count=1, nodata=float(nodata))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out, 1)
    return str(out_path)


def load_raster_4326(path, *, max_dim: int = 1024):
    """Reproject a single-band raster to EPSG:4326 and return (z, xs, ys, dx_m, dy_m): a 2-D
    float array (NaN at nodata), 1-D lon/lat cell centres (north-up), and the cell spacing in
    metres. Shared by the DEM hillshade and the result-raster overlays."""
    import math

    import numpy as np
    import rioxarray  # noqa: F401 — registers the .rio accessor

    da = rioxarray.open_rasterio(path, masked=True).squeeze().rio.reproject("EPSG:4326")
    z = np.asarray(da.values, dtype=float)
    xs = np.asarray(da["x"].values, dtype=float)
    ys = np.asarray(da["y"].values, dtype=float)
    if z.ndim != 2 or xs.size < 2 or ys.size < 2:
        raise ValueError("Unexpected raster shape for overlay.")
    if ys[0] < ys[-1]:                      # north-up: row 0 = northernmost (overlay's top edge)
        z = z[::-1, :]; ys = ys[::-1]
    step = max(1, math.ceil(max(z.shape) / max_dim))   # keep the data-URI small
    if step > 1:
        z = z[::step, ::step]; xs = xs[::step]; ys = ys[::step]
    lat0 = float((ys.min() + ys.max()) / 2)
    dx = abs(float(xs[1] - xs[0])) * 111320.0 * max(math.cos(math.radians(lat0)), 1e-6)
    dy = abs(float(ys[1] - ys[0])) * 110540.0
    return z, xs, ys, dx, dy


def rgba_to_overlay(rgba, xs, ys) -> dict:
    """Encode an RGBA float array (0..1) as a base64 PNG data URI + EPSG:4326 bounds for an
    ipyleaflet ImageOverlay: {"url": ..., "bounds": [[s, w], [n, e]]}."""
    import base64
    import io

    from matplotlib import image as mpimg

    buf = io.BytesIO()
    mpimg.imsave(buf, rgba, format="png")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return {"url": url,
            "bounds": [[float(ys.min()), float(xs.min())], [float(ys.max()), float(xs.max())]]}


def dem_overlay(dem_path, *, max_dim: int = 1024) -> dict:
    """Render the DEM as a hillshade + elevation-tint RGBA PNG (transparent at nodata) for an
    ipyleaflet ImageOverlay — lets the user toggle terrain vs. aerial while tracing the wetted
    extent. Returns {"url": <base64 data URI>, "bounds": [[s, w], [n, e]]} in EPSG:4326."""
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import LightSource, Normalize

    z, xs, ys, dx, dy = load_raster_4326(dem_path, max_dim=max_dim)
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("DEM has no valid pixels to render.")
    lo, hi = (float(v) for v in np.nanpercentile(z[valid], [2, 98]))
    if not hi > lo:
        hi = lo + 1.0
    ls = LightSource(azdeg=315, altdeg=45)
    rgba = ls.shade(np.where(valid, z, lo), cmap=cm.terrain, blend_mode="soft",
                    norm=Normalize(vmin=lo, vmax=hi), vert_exag=2.0, dx=dx, dy=dy)
    rgba[..., 3] = valid.astype(float)     # transparent at nodata
    return rgba_to_overlay(rgba, xs, ys)


def dem_summary(out_path) -> dict:
    """Quick stats for the fetched DEM (for the UI)."""
    import rasterio
    import numpy as np
    with rasterio.open(out_path) as src:
        a = src.read(1, masked=True)
        return {
            "width": src.width, "height": src.height,
            "min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
            "crs": str(src.crs),
        }
