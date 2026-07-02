"""Post-process HEC-RAS 2025 outputs into the shapes the app consumes.

Three products from the last-timestep depth/WSE GeoTIFFs that `ras map` writes:
  - wetted_extent_feature: the water-surface-extent polygon (EPSG:4326 GeoJSON Feature)
    that used to be hand-drawn on the Boundaries step
  - wse_on_dem_grid: the modeled water-surface elevations resampled onto the app DEM's
    exact grid, nodata -9999 outside the wetted area — the same contract as
    dem.clip_dem_to_polygon, so it slots straight into the groundwater run's wse_path
  - depth_overlay: an ipyleaflet ImageOverlay payload of the depth field

Plus cells_to_rasters_fallback: builds equivalent depth/WSE rasters directly from the
result HDF5 (last-profile cell depths + cell WSE interpolated over the terrain grid) for
environments where `ras map` is unavailable (e.g. a GDAL problem on Linux).
"""
from __future__ import annotations

from pathlib import Path

DEPTH_THRESH_M = 0.01     # "wet" = deeper than 1 cm; filters numerical film

DEPTH_CMAP = "hype_depth"     # cyan-blue -> dark blue; shallow water must read as water,
                              # never as white (registered below at import time)


def _register_depth_cmap():
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        DEPTH_CMAP, ["#7fe3e8", "#1e88c9", "#08306b"])
    try:
        matplotlib.colormaps.register(cmap, name=DEPTH_CMAP)
    except ValueError:        # already registered (module re-import)
        pass


_register_depth_cmap()


def _count_vertices(geom) -> int:
    parts = getattr(geom, "geoms", [geom])
    n = 0
    for p in parts:
        n += len(p.exterior.coords) + sum(len(r.coords) for r in p.interiors)
    return n


def wetted_extent_feature(depth_tif, thresh: float = DEPTH_THRESH_M,
                          simplify_px: float = 0.3, min_part_px: float = 4.0,
                          min_part_m2: float | None = None,
                          max_vertices: int = 25_000):
    """Wet pixels (depth > thresh) -> dissolved polygon -> EPSG:4326 GeoJSON Feature.

    High fidelity by design: the outline must visually match the depth raster, so
    pixels are connected DIAGONALLY too (connectivity=8 — a thin channel crossing a
    pixel corner stays one polygon instead of splitting into pools) and simplification
    is sub-pixel (0.3 px keeps the pixel stair-steps from reading as jagged without
    deviating from the raster). Isolated parts smaller than `min_part_m2` (typically
    ~half a mesh cell — features BELOW the solver's resolving power, so dropping them
    is honest, not lossy) or `min_part_px` pixels are removed. If the outline still
    exceeds `max_vertices`, simplification is stepped up (0.75 px, then 1.5 px) purely
    as a payload guard.

    The Feature carries quality metrics in `properties`: `n_parts` (parts kept in the
    polygon) and `main_frac` (largest connected region's share of the TOTAL wetted
    area, measured before size filtering) — a low main_frac means the water surface
    itself is broken up, the real red flag for coarse terrain/mesh."""
    import rasterio
    import rasterio.features
    import shapely.geometry as sg
    import shapely.ops
    from pyproj import Transformer
    from shapely.geometry import mapping

    with rasterio.open(depth_tif) as ds:
        a = ds.read(1, masked=True)
        wet = (~a.mask) & (a.data > thresh)
        if not wet.any():
            return None
        shapes = rasterio.features.shapes(wet.astype("uint8"), mask=wet,
                                          transform=ds.transform, connectivity=8)
        poly = shapely.ops.unary_union([sg.shape(s) for s, v in shapes if v == 1])
        px = abs(ds.transform.a)
        crs = ds.crs
    if poly.is_empty:
        return None
    parts = list(getattr(poly, "geoms", [poly]))
    total_area = sum(p.area for p in parts)
    main_frac = max(p.area for p in parts) / total_area if total_area > 0 else 1.0
    min_area = max(min_part_px * px * px, float(min_part_m2 or 0.0))
    keep = [p for p in parts if p.area >= min_area]
    if keep:
        poly = shapely.ops.unary_union(keep)
    n_parts = len(list(getattr(poly, "geoms", [poly])))
    for factor in (simplify_px, 0.75, 1.5):
        simplified = poly.simplify(factor * px, preserve_topology=True)
        if _count_vertices(simplified) <= max_vertices:
            break
    poly = simplified

    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    poly4326 = shapely.ops.transform(tr.transform, poly)
    return {"type": "Feature",
            "properties": {"source": "HEC-RAS 2025", "n_parts": n_parts,
                           "main_frac": round(float(main_frac), 4)},
            "geometry": mapping(poly4326)}


def wetted_area_m2(depth_tif, thresh: float = DEPTH_THRESH_M) -> float:
    import rasterio

    with rasterio.open(depth_tif) as ds:
        a = ds.read(1, masked=True)
        wet = int(((~a.mask) & (a.data > thresh)).sum())
        return wet * abs(ds.transform.a) * abs(ds.transform.e)


def wse_on_dem_grid(wse_tif, depth_tif, dem_path, out_path, thresh: float = DEPTH_THRESH_M,
                    nodata: float = -9999.0) -> str:
    """Resample the modeled WSE onto the app DEM's grid, masked to the wetted area.

    Output matches dem.clip_dem_to_polygon's contract (float32, nodata -9999, exact DEM
    grid/CRS) so _wse_path() can hand it to the groundwater engine unchanged — except the
    values are real modeled water-surface elevations rather than bare terrain.
    """
    import numpy as np
    import rioxarray  # noqa: F401 — .rio accessor
    from rioxarray.exceptions import NoDataInBounds  # noqa: F401

    wse = rioxarray.open_rasterio(wse_tif, masked=True).squeeze()
    depth = rioxarray.open_rasterio(depth_tif, masked=True).squeeze()
    wse = wse.where(depth > thresh)                       # same grid: ras map wrote both
    dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
    out = wse.rio.reproject_match(dem)                    # bilinear default is fine for WSE
    out = out.astype("float32").fillna(nodata).rio.write_nodata(nodata)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.rio.to_raster(out_path, compress="deflate")
    return str(out_path)


def result_overlay(tif, kind: str, *, max_dim: int = 1400) -> dict | None:
    """Colorized result raster as an ipyleaflet ImageOverlay payload, plus the color
    scale for a legend: {"url","bounds","vmin","vmax","cmap","label"}.
    kind="depth" -> Blues from 0 to p98(depth); kind="wse" -> viridis over p2..p98."""
    import numpy as np
    import rasterio

    from .results import raster_overlay

    with rasterio.open(tif) as ds:
        a = ds.read(1, masked=True)
        if a.count() == 0:
            return None
        vals = a.compressed()
    if kind == "depth":
        vmin, vmax = 0.0, max(float(np.percentile(vals, 98)), 0.1)
        cmap, label = DEPTH_CMAP, "Water depth (m)"
    else:
        lo, hi = (float(v) for v in np.percentile(vals, [2, 98]))
        vmin, vmax = lo, (hi if hi > lo else lo + 0.1)
        cmap, label = "viridis", "Water surface elevation (m)"
    ov = raster_overlay(str(tif), vmin=vmin, vmax=vmax, cmap=cmap,
                        max_dim=max_dim, smooth_to=0)
    ov.update(vmin=vmin, vmax=vmax, cmap=cmap, label=label)
    return ov


def depth_overlay(depth_tif, *, max_dim: int = 1400) -> dict | None:
    """Back-compat wrapper: the depth flavor of result_overlay."""
    return result_overlay(depth_tif, "depth", max_dim=max_dim)


def mesh_overlay(segments_4326, *, max_dim: int = 1800,
                 color=(0.2, 0.2, 0.2, 0.85), linewidth: float = 0.7) -> dict | None:
    """Rasterize mesh face segments into a transparent PNG ImageOverlay payload.

    A vector GeoJSON of thousands of face edges makes Leaflet's SVG renderer crawl, so
    the triangular mesh is drawn server-side (same approach as results.grid_overlay for
    the structured MODFLOW grid). `segments_4326` is an (N, 2, 2) array-like of
    [[lon_a, lat_a], [lon_b, lat_b]] pairs.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    from .dem import rgba_to_overlay

    segs = np.asarray(segments_4326, dtype=float)
    if segs.size == 0:
        return None
    lons = segs[..., 0]
    lats = segs[..., 1]
    west, east = float(lons.min()), float(lons.max())
    south, north = float(lats.min()), float(lats.max())
    if not (east > west and north > south):
        return None
    # pixel aspect ~ metric aspect at this latitude (lon degrees are compressed)
    import math
    kx = math.cos(math.radians((south + north) / 2.0))
    w_m, h_m = (east - west) * kx, (north - south)
    if w_m >= h_m:
        w_px, h_px = max_dim, max(64, int(round(max_dim * h_m / w_m)))
    else:
        h_px, w_px = max_dim, max(64, int(round(max_dim * w_m / h_m)))

    dpi = 100
    fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0.0)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.axis("off")
    ax.add_collection(LineCollection(segs, colors=[color], linewidths=linewidth,
                                     capstyle="round"))
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=float) / 255.0
    xs = np.array([west, east])
    ys = np.array([south, north])
    return rgba_to_overlay(rgba, xs, ys)


def cells_to_rasters_fallback(result_h5, geometry_h5, terrain_tif, depth_out, wse_out,
                              thresh: float = DEPTH_THRESH_M, nodata: float = -9999.0):
    """No-`ras map` fallback: rasterize the last profile straight from the result HDF5.

    Cell WSE = cell depth + cell minimum elevation (RAS's per-cell datum). WSE is
    interpolated between wet-cell centers (linear griddata, nearest fill), evaluated on
    the terrain grid, and depth = WSE - terrain, masked where depth <= thresh. Coarser
    than RAS Mapper's face-aware rendering but faithful enough for extent + wse_path.
    """
    import h5py
    import numpy as np
    import rasterio
    from scipy.interpolate import griddata

    with h5py.File(result_h5, "r") as f:
        depth_cells = f["Results/Output Blocks/Base Output/2D Flow Areas/Mesh/Cell Depth"][-1, :]
    with h5py.File(geometry_h5, "r") as f:
        centers = f["Geometry/2D Flow Areas/Mesh/Cell Coordinates"][...]
        try:
            zmin = f["Geometry/2D Flow Areas/Mesh/Property Tables/Cell Minimum Elevation"][...]
        except KeyError:
            zmin = None

    with rasterio.open(terrain_tif) as ds:
        terrain = ds.read(1, masked=True).filled(np.nan)
        transform, crs, shape = ds.transform, ds.crs, ds.shape
        meta = ds.meta.copy()

    if zmin is None:                                     # sample terrain at cell centers
        rows, cols = rasterio.transform.rowcol(transform, centers[:, 0], centers[:, 1])
        rows = np.clip(rows, 0, shape[0] - 1)
        cols = np.clip(cols, 0, shape[1] - 1)
        zmin = terrain[rows, cols]
    wse_cells = np.asarray(depth_cells, dtype=float) + np.asarray(zmin, dtype=float)
    wet = np.asarray(depth_cells) > thresh
    if not wet.any():
        raise RuntimeError("No wet cells at the last timestep — nothing to rasterize.")

    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    gx, gy = rasterio.transform.xy(transform, ys.ravel(), xs.ravel(), offset="center")
    gx = np.asarray(gx).reshape(shape)
    gy = np.asarray(gy).reshape(shape)
    pts = centers[wet]
    vals = wse_cells[wet]
    wse = griddata(pts, vals, (gx, gy), method="linear")
    near = griddata(pts, vals, (gx, gy), method="nearest")
    wse = np.where(np.isfinite(wse), wse, near)
    depth = wse - terrain
    ok = np.isfinite(depth) & (depth > thresh)
    depth_arr = np.where(ok, depth, nodata).astype("float32")
    wse_arr = np.where(ok, wse, nodata).astype("float32")

    meta.update(driver="GTiff", dtype="float32", count=1, nodata=nodata, compress="deflate")
    for path, arr in ((depth_out, depth_arr), (wse_out, wse_arr)):
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(arr, 1)
    return str(depth_out), str(wse_out)
