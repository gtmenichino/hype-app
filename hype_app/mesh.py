"""Build MODFLOW-grid geometry for the browser 3D mesh viewer — pure NumPy/rasterio, NO vtk.

The Mesh tab's "Compute mesh" button calls :func:`build_grid_geometry` to turn the domain polygon
+ terrain DEM + (cell_size, model depth, layer thickness) into a **decimated, de-duplicated** set
of active hexahedral cells (VTK_HEXAHEDRON) that ``www/mesh3d.js`` renders with vtk.js. It mirrors
the engine's grid math (top = DEM, ``botm[k] = top − k·z``, idomain = cells inside the domain)
closely enough for a discretization *preview*, without importing vtk/pyvista or running MODFLOW.

Points are emitted in **local metres** (SW corner = origin, z above the model base) so WebGL's
float32 coordinates stay precise; the client applies vertical exaggeration + clipping.
"""
from __future__ import annotations

import math

import numpy as np


def _grid_extent(domain_gdf_proj, cell_size: float, buffer_frac: float):
    """Buffered domain bbox + cell count (mirrors hype_app.estimate.estimate_cells)."""
    minx, miny, maxx, maxy = (float(v) for v in domain_gdf_proj.total_bounds)
    dx, dy = (maxx - minx) * buffer_frac, (maxy - miny) * buffer_frac
    minx, miny, maxx, maxy = minx - dx, miny - dy, maxx + dx, maxy + dy
    ncol = max(1, math.ceil((maxx - minx) / cell_size))
    nrow = max(1, math.ceil((maxy - miny) / cell_size))
    return minx, miny, maxx, maxy, ncol, nrow


def build_grid_geometry(domain_feat, dem_path, crs, cell_size, depth, z, *,
                        max_cells: int = 40_000, max_layers: int = 30,
                        buffer_frac: float = 0.12, log=print) -> dict:
    """Domain Feature (4326) + DEM + (cell_size, depth, z) → JSON-safe geometry for vtk.js:
    ``{points, cells, cellLayer, nHex, nPoints, dims, previewDims, decimation, layerStride,
    nActiveFull, bounds}``. ``cells`` is a flat list of 8 point-indices per hexahedron (the client
    adds the VTK cell-size/type framing). Decimated so ``nHex ≤ max_cells``.
    """
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    from .geometry import single_feature_gdf

    dom = single_feature_gdf(domain_feat).to_crs(crs)
    minx, miny, maxx, maxy, ncol, nrow = _grid_extent(dom, float(cell_size), buffer_frac)
    nlay = max(1, math.ceil(float(depth) / float(z)))
    dst_transform = from_origin(minx, maxy, float(cell_size), float(cell_size))   # north-up

    # --- top: reproject the DEM onto the target grid ---
    top = np.full((nrow, ncol), np.nan, dtype="float32")
    with rasterio.open(dem_path) as src:
        band = src.read(1, masked=True).filled(np.nan).astype("float32")
        reproject(source=band, destination=top,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs=crs,
                  src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    finite = np.isfinite(top)
    if not finite.any():
        raise ValueError("DEM does not cover the domain grid.")
    top = np.where(finite, top, np.float32(np.nanmin(top[finite])))   # fill gaps so botm is finite

    # --- idomain (2D, all layers equal — matches the engine) ---
    inside = geometry_mask([g.__geo_interface__ for g in dom.geometry],
                           out_shape=(nrow, ncol), transform=dst_transform, invert=True)
    n_active2d = int(inside.sum())
    if n_active2d == 0:
        raise ValueError("No grid cells fall inside the domain.")

    # --- decimate to the budget: layer stride lf, then row/col stride f ---
    lf = max(1, math.ceil(nlay / max_layers))
    nlay_d = max(1, math.ceil(nlay / lf))
    f = 1
    while f < max(nrow, ncol) and int(inside[::f, ::f].sum()) * nlay_d > max_cells:
        f += 1
    top_d = top[::f, ::f]
    inside_d = inside[::f, ::f]
    nrow_d, ncol_d = top_d.shape
    delr_d = float(cell_size) * f
    dz = float(z) * lf

    # --- per-surface elevations (cell centres → corners), local z above the model base ---
    z_ref = float(top_d.min() - dz * nlay_d)
    surf = [(top_d - s * dz) - z_ref for s in range(nlay_d + 1)]      # s=0 top … s=nlay_d base

    def _to_corners(a):
        p = np.pad(a, 1, mode="edge")
        return 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])   # (nrow_d+1, ncol_d+1)

    surf_c = [_to_corners(s) for s in surf]
    xs = np.arange(ncol_d + 1) * delr_d
    yc = (nrow_d * delr_d) - np.arange(nrow_d + 1) * delr_d           # row 0 = north (larger y)

    # --- emit active hexahedra, remapping only the corner points actually used ---
    stride = (nrow_d + 1) * (ncol_d + 1)
    remap: dict = {}
    points: list = []
    cells: list = []
    cell_layer: list = []

    def _pt(s, j, i):
        key = s * stride + j * (ncol_d + 1) + i
        idx = remap.get(key)
        if idx is None:
            idx = len(points) // 3
            points.extend((float(xs[i]), float(yc[j]), float(surf_c[s][j, i])))
            remap[key] = idx
        return idx

    for R in range(nrow_d):
        for C in range(ncol_d):
            if not inside_d[R, C]:
                continue
            for s in range(nlay_d):
                lo, hi = s + 1, s                     # surface s+1 = bottom (lower z), s = top
                cells.extend((
                    _pt(lo, R, C), _pt(lo, R, C + 1), _pt(lo, R + 1, C + 1), _pt(lo, R + 1, C),
                    _pt(hi, R, C), _pt(hi, R, C + 1), _pt(hi, R + 1, C + 1), _pt(hi, R + 1, C)))
                cell_layer.append(s)

    n_hex = len(cell_layer)
    log(f"[mesh] grid {ncol}x{nrow}x{nlay}; preview x{f} (layers /{lf}) -> "
        f"{n_hex} hexes, {len(points) // 3} points")
    zmin = float(surf_c[-1].min())
    zmax = float(surf_c[0].max())
    return {
        "points": points, "cells": cells, "cellLayer": cell_layer,
        "nHex": n_hex, "nPoints": len(points) // 3,
        "dims": {"nlay": nlay, "nrow": nrow, "ncol": ncol},
        "previewDims": {"nlay": nlay_d, "nrow": nrow_d, "ncol": ncol_d},
        "decimation": f, "layerStride": lf, "nActiveFull": n_active2d * nlay,
        "bounds": [0.0, float(xs[-1]), float(yc.min()), float(yc.max()), zmin, zmax],
    }
