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


def build_grid_geometry(domain_feat, dem_path, crs, cell_size, depth, z, *,
                        max_cells: int = 40_000, max_layers: int = 30,
                        buffer_frac: float = 0.12, log=print) -> dict:
    """Domain Feature (4326) + DEM + (cell_size, depth, z) → JSON-safe geometry for vtk.js:
    ``{points, cells, cellLayer, cellElev, elevRange, nHex, nPoints, dims, previewDims, decimation,
    layerStride, nActiveFull, bounds}``. ``cellElev`` colours the top layer by real terrain elevation
    (deeper layers get a below-``elevRange`` sentinel → gray). ``cells`` is a flat list of 8 point-indices per hexahedron (the client
    adds the VTK cell-size/type framing). Runs the real engine grid (``build_model_domain``); each
    hexahedron has a flat per-cell top/bottom (blocky). Decimated so ``nHex ≤ max_cells``.
    """
    from shapely.ops import unary_union

    from .geometry import single_feature_gdf

    dom = single_feature_gdf(domain_feat).to_crs(crs)
    minx, miny, maxx, maxy, ncol0, nrow0 = _grid_extent(dom, float(cell_size), buffer_frac)

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
    return {
        "points": points, "cells": cells, "cellLayer": cell_layer,
        "cellElev": cell_elev, "elevRange": [elev_lo, elev_hi],
        "nHex": n_hex, "nPoints": len(points) // 3,
        "dims": {"nlay": nlay, "nrow": nrow, "ncol": ncol},
        "previewDims": {"nlay": nlay_d, "nrow": nrow_d, "ncol": ncol_d},
        "decimation": f, "layerStride": lf, "nActiveFull": n_active2d * nlay,
        "bounds": [0.0, float(ncol_d * delr_d), 0.0, float(nrow_d * delr_d), 0.0, float(zt_max)],
    }
