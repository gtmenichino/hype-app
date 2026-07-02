"""Build MODFLOW-grid geometry for the browser 3D mesh viewer — runs the REAL engine grid, NO vtk.

The Mesh tab's "Compute mesh" button calls :func:`build_grid_geometry` to turn the domain polygon
+ terrain DEM + (cell_size, model depth, layer thickness) into a **decimated** set of active
hexahedral cells (VTK_HEXAHEDRON) that ``www/mesh3d.js`` renders with vtk.js.

To match what the model actually builds, this reuses the engine's own discretization
(``hypetool.functions.my_utils.build_model_domain``): a **flat bottom** at ``bed = min(DEM)`` with
uniform layers of thickness ``z`` stacked down to a flat base, and only the **top** layer's top
stretched to the terrain (``tops[0] = DEM`` per cell, ``botm[0] = bed`` flat, ``botm[k] = bed − k·z``;
``nlay = int(depth / z)``). Each cell gets a **single, flat top and bottom elevation** — the mesh is
blocky/stair-stepped, never interpolated within a cell — so the preview is the real grid, not a
smooth terrain-following slab.

Points are emitted in **local metres** (SW corner = origin, z above the flat model base) so WebGL's
float32 coordinates stay precise; the client applies vertical exaggeration + clipping.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np


def _grid_extent(domain_gdf_proj, cell_size: float, buffer_frac: float):
    """Buffered domain bbox + cell count (mirrors hype_app.estimate.estimate_cells)."""
    minx, miny, maxx, maxy = (float(v) for v in domain_gdf_proj.total_bounds)
    dx, dy = (maxx - minx) * buffer_frac, (maxy - miny) * buffer_frac
    minx, miny, maxx, maxy = minx - dx, miny - dy, maxx + dx, maxy + dy
    ncol = max(1, math.ceil((maxx - minx) / cell_size))
    nrow = max(1, math.ceil((maxy - miny) / cell_size))
    return minx, miny, maxx, maxy, ncol, nrow


def _reproject_dem_to_grid(dem_path, crs, minx, maxy, cell_size, nrow, ncol, out_tif):
    """Reproject the DEM into the exact preview grid (proj_crs, north-up, `cell_size`) and write a
    grid-aligned GeoTIFF at `out_tif`. Aligning the raster to `from_origin(minx, maxy, cell_size)`×
    (nrow, ncol) makes the engine re-derive the identical grid, so `build_model_domain` reproduces the
    same cells. Gaps → nodata (-9999) so the engine masks them exactly as it does a fetched DEM."""
    import rasterio
    from rasterio.crs import CRS as RioCRS
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    dst_transform = from_origin(minx, maxy, float(cell_size), float(cell_size))
    top = np.full((nrow, ncol), np.nan, dtype="float32")
    with rasterio.open(dem_path) as src:
        band = src.read(1, masked=True).filled(np.nan).astype("float32")
        reproject(source=band, destination=top,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs=crs,
                  src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    if not np.isfinite(top).any():
        raise ValueError("DEM does not cover the domain grid.")
    nodata = -9999.0
    arr = np.where(np.isfinite(top), top, np.float32(nodata)).astype("float32")
    meta = dict(driver="GTiff", height=nrow, width=ncol, count=1, dtype="float32",
                crs=RioCRS.from_user_input(crs), transform=dst_transform, nodata=nodata)
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(arr, 1)


BOUNDARY_STYLE = {                     # matches the app's 2-D map colors (app.py *_STYLE)
    "up": ("Upstream", "#f08c00"),
    "left": ("Left FPL", "#1f6feb"),
    "right": ("Right FPL", "#d83933"),
    "down": ("Downstream", "#9b59b6"),
}


def preview_cell_cap() -> int:
    """FULL-grid cell cap for the 3-D preview build. The engine discretization allocates
    per-layer float64 arrays for the whole (buffered) bbox, so an over-fine cell size can
    OOM the app process — refuse anything the run itself would refuse (same red band)."""
    from . import estimate
    return int(os.environ.get("HYPE_MESH_PREVIEW_MAX_CELLS", estimate.AMBER_MAX))


def build_grid_geometry(domain_feat, dem_path, crs, cell_size, depth, z, *,
                        sides: dict | None = None, want_basemap: bool = True,
                        max_cells: int = 40_000, max_layers: int = 30,
                        buffer_frac: float = 0.12, log=print) -> dict:
    """Domain Feature (4326) + DEM + (cell_size, depth, z) → JSON-safe geometry for vtk.js:
    ``{points, cells, cellLayer, cellElev, elevRange, nHex, nPoints, dims, previewDims, decimation,
    layerStride, nActiveFull, bounds, boundaries, basemap}``. ``cellElev`` colours the top layer by
    real terrain elevation (deeper layers get a below-``elevRange`` sentinel → gray). ``cells`` is a
    flat list of 8 point-indices per hexahedron (the client
    adds the VTK cell-size/type framing). Runs the real engine grid (``build_model_domain``); each
    hexahedron has a flat per-cell top/bottom (blocky). Decimated so ``nHex ≤ max_cells``.

    ``sides`` (optional) = the four oriented boundary LineString Features (EPSG:4326,
    keys up/left/right/down) → per-side marker polylines along the top of the boundary's
    cells, for on-mesh orientation labels. ``want_basemap`` fetches a USGS aerial image
    over the preview extent for the client to drape on the top surface.
    """
    from shapely.ops import unary_union

    from .geometry import single_feature_gdf

    dom = single_feature_gdf(domain_feat).to_crs(crs)
    minx, miny, maxx, maxy, ncol0, nrow0 = _grid_extent(dom, float(cell_size), buffer_frac)

    # hard safeguard BEFORE the engine allocates anything (a too-fine cell size here used
    # to OOM the whole app process; the build now also runs in a child process, but there
    # is no point burning a core on a grid the run itself would refuse)
    nlay_est = max(1, math.ceil(float(depth) / float(z)))
    cap = preview_cell_cap()
    if ncol0 * nrow0 * nlay_est > cap:
        need = float(cell_size) * math.sqrt(ncol0 * nrow0 * nlay_est / cap)
        raise ValueError(
            f"Grid would be {ncol0}×{nrow0}×{nlay_est} = {ncol0 * nrow0 * nlay_est:,} cells — "
            f"over the {cap:,}-cell preview limit. Try a cell size of ~{need:.0f} m, a shallower "
            f"model, or thicker layers.")

    tmpdir = tempfile.mkdtemp(prefix="hype_mesh_")
    try:
        tmp_tif = os.path.join(tmpdir, "terrain_projcrs.tif")
        _reproject_dem_to_grid(dem_path, crs, minx, maxy, float(cell_size), nrow0, ncol0, tmp_tif)

        # --- run the REAL engine discretization (flat bottom + uniform layers + terrain-following top) ---
        from hypetool.functions.my_utils import build_model_domain
        cfg = SimpleNamespace(terrain_output_raster=tmp_tif,
                              cell_size_x=float(cell_size), cell_size_y=float(cell_size),
                              gw_mod_depth=float(depth), z=float(z))
        dm = build_model_domain(cfg)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    tops, botm = dm["tops"], dm["botm"]                    # each a list of nlay (nrow, ncol) arrays
    nlay, nrow, ncol = int(dm["nlay"]), int(dm["nrow"]), int(dm["ncol"])

    # --- active cells: engine cell-centres inside the domain (exact to build_model_domain's grid) ---
    dom_geom = unary_union(list(dom.geometry))
    inside = np.asarray(dm["grid_points"].within(dom_geom)).reshape(nrow, ncol)
    n_active2d = int(inside.sum())
    if n_active2d == 0:
        raise ValueError("No grid cells fall inside the domain.")

    # --- decimate to the budget: layer stride lf, then row/col stride f ---
    lf = max(1, math.ceil(nlay / max_layers))
    nlay_d = max(1, math.ceil(nlay / lf))
    f = 1
    while f < max(nrow, ncol) and int(inside[::f, ::f].sum()) * nlay_d > max_cells:
        f += 1
    inside_d = inside[::f, ::f]
    nrow_d, ncol_d = inside_d.shape
    delr_d = float(cell_size) * f

    # --- emit blocky hexahedra: one flat top (tops[s]) + flat bottom (botm[s]) per cell, no interp ---
    z_ref = float(np.nanmin(botm[nlay - 1]))               # flat model base (the deepest, uniform bottom)
    tvals = np.asarray(tops[0])[inside]                    # real terrain elevations over active cells
    elev_lo, elev_hi = float(np.nanmin(tvals)), float(np.nanmax(tvals))
    if elev_hi - elev_lo < 1e-6:                           # flat terrain → give the legend a usable span
        elev_hi = elev_lo + 1.0
    sentinel = elev_lo - 1000.0                            # deeper layers → below-range → gray body in the viewer
    points: list = []
    cells: list = []
    cell_layer: list = []
    cell_elev: list = []                                   # per-hex colour scalar: top layer = terrain elev, else sentinel
    zt_max = 0.0
    for s in range(nlay_d):
        kt = s * lf                                        # merged preview layer s spans real layers kt..kb
        kb = min((s + 1) * lf, nlay) - 1
        top_a = np.asarray(tops[kt])[::f, ::f]
        bot_a = np.asarray(botm[kb])[::f, ::f]
        for R in range(nrow_d):
            y0, y1 = R * delr_d, (R + 1) * delr_d
            for C in range(ncol_d):
                if not inside_d[R, C]:
                    continue
                zt = float(top_a[R, C]) - z_ref
                zb = float(bot_a[R, C]) - z_ref
                if zt - zb <= 1e-6:                        # skip zero-thickness cells (e.g. top layer at min-DEM)
                    continue
                x0, x1 = C * delr_d, (C + 1) * delr_d
                b = len(points) // 3
                points.extend((x0, y0, zb, x1, y0, zb, x1, y1, zb, x0, y1, zb,     # bottom face 0..3
                               x0, y0, zt, x1, y0, zt, x1, y1, zt, x0, y1, zt))    # top face 4..7
                cells.extend((b, b + 1, b + 2, b + 3, b + 4, b + 5, b + 6, b + 7))
                cell_layer.append(s)
                cell_elev.append(float(top_a[R, C]) if s == 0 else sentinel)   # terrain colour on the top layer only
                if zt > zt_max:
                    zt_max = zt

    n_hex = len(cell_layer)
    log(f"[mesh] engine grid {ncol}x{nrow}x{nlay}; preview x{f} (layers /{lf}) -> "
        f"{n_hex} hexes, {len(points) // 3} points")

    # Local-coordinate anchor: cell [R, C] sits at local (C·delr, R·delr) with row 0 = SOUTH
    # (build_model_domain's grid_y runs ymin→ymax), i.e. local = (x − dm.xmin, y − dm.ymin).
    x_anchor, y_anchor = float(dm["xmin"]), float(dm["ymin"])
    top0_d = np.asarray(tops[0])[::f, ::f]
    boundaries = _boundary_markers(sides, crs, x_anchor, y_anchor, delr_d, inside_d,
                                   top0_d, z_ref) if sides else []
    basemap = None
    if want_basemap:
        basemap = _fetch_basemap(crs, x_anchor, y_anchor,
                                 float(ncol_d * delr_d), float(nrow_d * delr_d), log=log)

    return {
        "points": points, "cells": cells, "cellLayer": cell_layer,
        "cellElev": cell_elev, "elevRange": [elev_lo, elev_hi],
        "nHex": n_hex, "nPoints": len(points) // 3,
        "dims": {"nlay": nlay, "nrow": nrow, "ncol": ncol},
        "previewDims": {"nlay": nlay_d, "nrow": nrow_d, "ncol": ncol_d},
        "decimation": f, "layerStride": lf, "nActiveFull": n_active2d * nlay,
        "bounds": [0.0, float(ncol_d * delr_d), 0.0, float(nrow_d * delr_d), 0.0, float(zt_max)],
        "boundaries": boundaries, "basemap": basemap,
    }


def _boundary_markers(sides, crs, x_anchor, y_anchor, delr_d, inside_d, top0_d, z_ref,
                      lift: float = 0.6) -> list:
    """Per-boundary marker polylines along the TOP of the boundary's preview cells.

    Each of the four oriented sides (EPSG:4326 LineStrings) is sampled at sub-cell spacing;
    every sample maps to its decimated preview cell (nearest active cell within 2 cells),
    and consecutive distinct cells become a polyline through the cell-top centres (z lifted
    slightly so the line never z-fights the top faces). Coordinates are preview-local.
    """
    from pyproj import Transformer
    from shapely.geometry import shape

    nrow_d, ncol_d = inside_d.shape
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    out = []
    for key, (name, color) in BOUNDARY_STYLE.items():
        feat = (sides or {}).get(key)
        if not feat:
            continue
        try:
            line = shape(feat["geometry"])
            len_m = line.length * 111_000.0              # degrees → rough metres (sampling only)
            n_samp = int(np.clip(len_m / (delr_d * 0.75), 8, 400))
            pts4326 = [line.interpolate(i / (n_samp - 1), normalized=True) for i in range(n_samp)]
            xs, ys = tr.transform([p.x for p in pts4326], [p.y for p in pts4326])
        except Exception:  # noqa: BLE001 — a malformed side just loses its marker
            continue
        path, last_rc = [], None
        for x, y in zip(xs, ys):
            cd = int((x - x_anchor) // delr_d)
            rd = int((y - y_anchor) // delr_d)
            best = None
            for dr in range(-2, 3):                      # nearest ACTIVE preview cell (≤ 2 cells off)
                for dc in range(-2, 3):
                    r2, c2 = rd + dr, cd + dc
                    if 0 <= r2 < nrow_d and 0 <= c2 < ncol_d and inside_d[r2, c2]:
                        d2 = dr * dr + dc * dc
                        if best is None or d2 < best[0]:
                            best = (d2, r2, c2)
            if best is None or (best[1], best[2]) == last_rc:
                continue
            _, r2, c2 = best
            last_rc = (r2, c2)
            zt = float(top0_d[r2, c2]) - z_ref
            if not np.isfinite(zt):
                continue
            path.extend(((c2 + 0.5) * delr_d, (r2 + 0.5) * delr_d, zt + lift))
        if len(path) >= 6:                               # at least 2 points
            out.append({"key": key, "name": name, "color": color, "points": path})
    return out


def _fetch_basemap(crs, x_anchor, y_anchor, width_m, height_m, *, max_px: int = 1024,
                   timeout_s: float = 30.0, log=print):
    """USGS aerial imagery over the preview extent as a base64 JPEG for the 3-D drape:
    {"url", "x0", "y0", "x1", "y1"} in preview-local metres (y0 = south edge; the image's
    top row is the NORTH edge). None on any failure — the drape is a nice-to-have."""
    import base64
    import urllib.parse
    import urllib.request

    from pyproj import CRS

    try:
        epsg = CRS.from_user_input(crs).to_epsg()
        if epsg is None:
            return None
        aspect = height_m / width_m if width_m > 0 else 1.0
        if aspect <= 1.0:
            w_px, h_px = max_px, max(64, int(round(max_px * aspect)))
        else:
            w_px, h_px = max(64, int(round(max_px / aspect))), max_px
        params = urllib.parse.urlencode({
            "bbox": f"{x_anchor},{y_anchor},{x_anchor + width_m},{y_anchor + height_m}",
            "bboxSR": epsg, "imageSR": epsg, "size": f"{w_px},{h_px}",
            "format": "jpg", "transparent": "false", "f": "image"})
        url = ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/"
               "MapServer/export?" + params)
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            data = r.read()
        if not data or len(data) < 1000:                 # error page / empty tile
            return None
        log(f"[mesh] basemap drape fetched ({len(data) // 1024} KB, {w_px}x{h_px} px)")
        return {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
                "x0": 0.0, "y0": 0.0, "x1": float(width_m), "y1": float(height_m)}
    except Exception as e:  # noqa: BLE001
        log(f"[mesh] basemap drape unavailable: {e}")
        return None


def child_build(payload: dict, q) -> None:
    """Run the preview build in a spawned child process (crash/OOM isolation + hard cancel):
    puts ('log', line)… then ('result', geometry) or ('error', message) on `q`. Top-level and
    picklable for the 'spawn' start method."""
    try:
        g = build_grid_geometry(
            payload["domain"], payload["dem"], payload["crs"],
            float(payload["cell_size"]), float(payload["depth"]), float(payload["z"]),
            sides=payload.get("sides"), want_basemap=payload.get("want_basemap", True),
            log=lambda m: q.put(("log", str(m))),
        )
        q.put(("result", g))
    except Exception as e:  # noqa: BLE001
        q.put(("error", str(e)))
