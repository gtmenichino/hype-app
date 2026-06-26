# my_utils.py
# ----------------
# Business-logic and helper functions for the hyporheic workflow.

from __future__ import annotations

import os
import zipfile
from pathlib import Path, PurePath
from typing import Any, Sequence, Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from shapely.geometry import (
    box, Point, LineString, Polygon, shape as shp_shape
)
from shapely.ops import linemerge, unary_union, nearest_points

import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.mask import mask
from rasterio.features import shapes as rio_shapes
from rasterio.transform import rowcol, xy as rio_xy

import flopy
from flopy.modpath import Modpath7, ParticleGroup, ParticleData
from flopy.utils import CellBudgetFile
from flopy.utils.binaryfile import HeadFile
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ----------------------------
# Light-weight package installer
# ----------------------------
def install_missing_packages(required: list[str]) -> None:
    import importlib, subprocess, sys, logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("deps")
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except Exception:
            logger.info(f"Installing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


# ----------------------------
# Step 1 – Pre-processing
# ----------------------------
def preprocess_data(cfg) -> dict:
    """
    Mirrors original Step 1 using cfg helpers:
      - load CRS from projection_file
      - reproject terrain raster to HEC-RAS CRS
      - reproject + crop water-surface raster to terrain extent
      - load & reproject vectors
    Populates cfg fields in-place and returns a summary dict.
    """
    cfg.setup_workspace(clean=False)
    cfg.setup_projection()
    cfg.setup_terrain(cfg.hec_ras_crs)
    cfg.setup_water_surface(cfg.hec_ras_crs)
    cfg.setup_vectors()

    return dict(
        hec_ras_crs=cfg.hec_ras_crs,
        terrain_output_raster=cfg.terrain_output_raster,
        water_surface_output_raster=cfg.cropped_water_surface_raster,
        ground_water_domain=cfg.ground_water_domain,
        left_boundary=cfg.left_boundary,
        right_boundary=cfg.right_boundary,
    )


# ----------------------------
# Step 3 – Model domain
# ----------------------------
def interpolate_na(terrain: np.ma.MaskedArray) -> np.ndarray:
    """Nearest-neighbor fill for masked terrain values (as in original)."""
    from scipy.interpolate import griddata
    valid_mask = ~terrain.mask
    valid_coords = np.array(np.nonzero(valid_mask)).T
    valid_values = terrain[valid_mask]
    invalid_mask = terrain.mask
    if not np.any(invalid_mask):
        return terrain.filled(terrain)
    invalid_coords = np.array(np.nonzero(invalid_mask)).T
    interpolated_values = griddata(valid_coords, valid_values, invalid_coords, method="nearest")
    terrain[invalid_mask] = interpolated_values
    return terrain


def build_model_domain(cfg) -> dict:
    """
    Read reprojected terrain raster and construct grid parameters/arrays.
    """
    terrain_raster = cfg.terrain_output_raster
    if not terrain_raster or not Path(terrain_raster).exists():
        raise FileNotFoundError("Reprojected terrain raster not found; run preprocess_data first.")

    with rasterio.open(terrain_raster) as src:
        raster_array = src.read(1)
        raster_transform = src.transform
        raster_crs = src.crs
        raster_bounds_box = box(*src.bounds)
        terrain_elevation = np.ma.masked_equal(raster_array, src.nodata)

    # nan-safe: mask both the nodata sentinel and any non-finite pixels (a fetched 3DEP DEM
    # uses nodata=NaN, which masked_equal cannot catch, so np.min would return NaN and poison
    # every botm/strt downstream).
    bed_elevation = float(np.nanmin(np.ma.filled(terrain_elevation, np.nan)))
    if not np.isfinite(bed_elevation):
        raise ValueError("Terrain raster has no valid elevations (all nodata/NaN). "
                         "Check the fetched DEM and the drawn AOI.")

    xmin = raster_transform.c
    ymax = raster_transform.f
    xmax = xmin + (terrain_elevation.shape[1] * raster_transform.a)
    ymin = ymax + (terrain_elevation.shape[0] * raster_transform.e)

    ncol = int((xmax - xmin) / cfg.cell_size_x)
    nrow = int((ymax - ymin) / cfg.cell_size_y)

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin + cfg.cell_size_x / 2, xmax - cfg.cell_size_x / 2, ncol),
        np.linspace(ymin + cfg.cell_size_y / 2, ymax - cfg.cell_size_y / 2, nrow),
    )

    grid_points = gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())]},
        crs=raster_crs,
    )

    # Sample terrain at cell centers
    top = np.full((nrow, ncol), np.nan, dtype=float)
    for r in range(nrow):
        for c in range(ncol):
            x, y = grid_x[r, c], grid_y[r, c]
            col, row = ~raster_transform * (x, y)
            col, row = int(col), int(row)
            if 0 <= row < terrain_elevation.shape[0] and 0 <= col < terrain_elevation.shape[1]:
                top[r, c] = terrain_elevation[row, col]
    top = interpolate_na(np.ma.masked_invalid(top))

    nlay = int(cfg.gw_mod_depth / cfg.z)
    tops = [top]
    botm = [np.full_like(top, bed_elevation)]
    for _ in range(1, nlay):
        next_top = botm[-1]
        next_bot = next_top - cfg.z
        tops.append(next_top)
        botm.append(next_bot)

    cfg.terrain_elevation = terrain_elevation
    cfg.raster_transform = raster_transform
    cfg.raster_crs = raster_crs
    cfg.raster_bounds_box = raster_bounds_box
    cfg.bed_elevation = bed_elevation
    cfg.ncol = ncol
    cfg.nrow = nrow
    cfg.nlay = nlay
    cfg.top = top
    cfg.tops = tops
    cfg.botm = botm
    cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax = xmin, xmax, ymin, ymax
    cfg.grid_x, cfg.grid_y = grid_x, grid_y
    cfg.grid_points = grid_points
    cfg.xorigin, cfg.yorigin = xmin, ymin

    return dict(
        bed_elevation=bed_elevation, ncol=ncol, nrow=nrow, nlay=nlay,
        top=top, tops=tops, botm=botm, grid_x=grid_x, grid_y=grid_y, grid_points=grid_points,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, raster_crs=raster_crs
    )


# ----------------------------
# Step 4 – Model boundaries & idomain
# ----------------------------
def define_floodplain_boundaries(left_boundary: gpd.GeoDataFrame, right_boundary: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    left_start = left_boundary.geometry.iloc[0].coords[0]
    left_end = left_boundary.geometry.iloc[-1].coords[-1]
    right_start = right_boundary.geometry.iloc[0].coords[0]
    right_end = right_boundary.geometry.iloc[-1].coords[-1]

    upstream_line = LineString([left_start, right_start])
    downstream_line = LineString([left_end, right_end])

    upstream = gpd.GeoDataFrame(geometry=[upstream_line], crs=left_boundary.crs)
    downstream = gpd.GeoDataFrame(geometry=[downstream_line], crs=left_boundary.crs)
    return upstream, downstream


def make_idomain(cfg, ground_water_domain: gpd.GeoDataFrame) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    nlay, nrow, ncol = cfg.nlay, cfg.nrow, cfg.ncol
    cell_size_x, cell_size_y = cfg.cell_size_x, cfg.cell_size_y
    grid_x, grid_y = cfg.grid_x, cfg.grid_y

    grid_cells = []
    for r in range(nrow):
        for c in range(ncol):
            x_min = grid_x[r, c] - (cell_size_x / 2)
            x_max = grid_x[r, c] + (cell_size_x / 2)
            y_min = grid_y[r, c] - (cell_size_y / 2)
            y_max = grid_y[r, c] + (cell_size_y / 2)
            grid_cells.append(Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]))
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=ground_water_domain.crs)

    inside = grid_gdf.geometry.intersects(ground_water_domain.unary_union)
    grid_gdf["inside_domain"] = inside

    idomain = np.zeros((nlay, nrow, ncol), dtype=int)
    for idx, ok in enumerate(inside):
        r, c = divmod(idx, ncol)
        if ok:
            idomain[:, r, c] = 1
    return idomain, grid_gdf


# ----------------------------
# Step 5 – Boundary conditions
# ----------------------------
def identify_boundary_cells(idomain: np.ndarray) -> list[tuple[int, int, int]]:
    """Find cells on the edge of the active domain (first layer based)."""
    boundary = set()
    nlay, nrow, ncol = idomain.shape
    for r in range(nrow):
        for c in range(ncol):
            if idomain[0, r, c] == 1:
                if (
                    (r > 0 and idomain[0, r - 1, c] == 0) or
                    (r < nrow - 1 and idomain[0, r + 1, c] == 0) or
                    (c > 0 and idomain[0, r, c - 1] == 0) or
                    (c < ncol - 1 and idomain[0, r, c + 1] == 0)
                ):
                    for k in range(nlay):
                        boundary.add((k, r, c))
    return list(boundary)

def classify_boundary_cells_faster(
    boundary_cells,
    grid_gdf,
    left_boundary,
    right_boundary,
    upstream_boundary,
    downstream_boundary,
    ncol
):
    left_u       = left_boundary.geometry.union_all() if hasattr(left_boundary.geometry, "union_all") else left_boundary.unary_union
    right_u      = right_boundary.geometry.union_all() if hasattr(right_boundary.geometry, "union_all") else right_boundary.unary_union
    upstream_u   = upstream_boundary.geometry.union_all() if hasattr(upstream_boundary.geometry, "union_all") else upstream_boundary.unary_union
    downstream_u = downstream_boundary.geometry.union_all() if hasattr(downstream_boundary.geometry, "union_all") else downstream_boundary.unary_union

    left_cells, right_cells, up_cells, down_cells = [], [], [], []
    rc_to_side = {}

    geoms = grid_gdf.geometry

    for k, r, c in boundary_cells:
        rc = (r, c)
        if rc not in rc_to_side:
            g = geoms.iat[r * ncol + c]
            d_left = g.distance(left_u)
            d_right = g.distance(right_u)
            d_up = g.distance(upstream_u)
            d_down = g.distance(downstream_u)

            side = min(
                (("left", d_left), ("right", d_right), ("upstream", d_up), ("downstream", d_down)),
                key=lambda t: t[1]
            )[0]
            rc_to_side[rc] = side

        side = rc_to_side[rc]
        (left_cells if side == "left" else
         right_cells if side == "right" else
         up_cells if side == "upstream" else
         down_cells).append((k, r, c))

    return left_cells, right_cells, up_cells, down_cells


def parse_fraction_gradient_profile(profile: str) -> list[tuple[float, float]]:
    """
    Parse a space/comma-delimited profile string into sorted (fraction, gradient) pairs.

    Example input:
        "0,0.01 0.5,0.05 1,0.1"

    Returns
    -------
    list[(float, float)] sorted by fraction ascending.
    Requires fractions 0 and 1 be present (within 1e-6 tolerance).
    """
    import re
    if profile is None or not str(profile).strip():
        raise ValueError("Gradient profile string is empty.")

    pairs: list[tuple[float, float]] = []
    for m in re.finditer(r'([0-9]*\.?[0-9]+)\s*,\s*([+-]?[0-9]*\.?[0-9]+)', str(profile)):
        f = float(m.group(1))
        g = float(m.group(2))
        if not (0.0 - 1e-9 <= f <= 1.0 + 1e-9):
            raise ValueError(f"Profile fraction {f} is outside [0,1].")
        pairs.append((f, g))

    if not pairs:
        raise ValueError("No valid 'fraction,gradient' pairs parsed from profile string.")

    # sort & coalesce duplicates by keeping last occurrence
    pairs.sort(key=lambda t: (t[0],))  # ascending fraction
    uniq: dict[float, float] = {}
    for f, g in pairs:
        uniq[round(f, 12)] = g  # round fraction to stabilize keys
    fracs = sorted(uniq.keys())
    grads = [uniq[f] for f in fracs]

    if (abs(fracs[0] - 0.0) > 1e-6) or (abs(fracs[-1] - 1.0) > 1e-6):
        raise ValueError("Profile must include fractions 0 and 1.")

    return list(zip(fracs, grads))

def compute_boundary_heads_from_profile(
    first_layer_cells: list[tuple[int, int, int]],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    line: LineString | None,
    profile_string: str,
    wse_edge_index: dict,
    *,
    log: callable = print,
) -> tuple[list[float], float, float]:
    """
    From a 'fraction,gradient' profile for a boundary LineString:
      1) Build anchor heads at the specified fractions using
         head = WSE_edge + gradient * distance_to_WSE_edge.
      2) Interpolate those anchor heads to each boundary cell (first layer).
      3) Return (heads_for_cells, head_at_f0, head_at_f1).

    Notes
    -----
    * 'line' should be the merged boundary LineString (longest segment if multipart).
    * 'first_layer_cells' must be sorted by distance along line (caller should sort).
    * Uses nearest edge of *valid* WSE area for distance/value via
      `nearest_wse_edge_distance_and_value`.
    """
    if not first_layer_cells or line is None:
        return [], float("nan"), float("nan")

    # 1) Parse profile
    pairs = parse_fraction_gradient_profile(profile_string)  # [(f, grad), ...]
    fracs = np.asarray([f for f, _ in pairs], dtype=float)
    grads = np.asarray([g for _, g in pairs], dtype=float)

    # 2) Build anchor heads at the specified fractions
    L = float(line.length) if line.length else 0.0
    if L <= 0:
        raise ValueError("Boundary line has zero length; cannot evaluate fractions.")
    anchor_heads: list[float] = []
    for f, g in zip(fracs, grads):
        s = float(np.clip(f, 0.0, 1.0)) * L
        pt = line.interpolate(s)
        dist, wse, _edge_xy, _border_xy = nearest_wse_edge_distance_and_value(wse_edge_index, pt)
        head = float(wse) + float(g) * float(dist)
        anchor_heads.append(head)
        log(f"    profile f={f:g}: dist_to_WSE={dist:.3f}, WSE={wse:.3f}, grad={g:g} → head={head:.3f}")

    anchor_heads = np.asarray(anchor_heads, dtype=float)
    head_f0 = float(anchor_heads[0])
    head_f1 = float(anchor_heads[-1])

    # 3) Interpolate to each boundary cell
    heads_for_cells: list[float] = []
    for (k, r, c) in first_layer_cells:
        p = Point(float(grid_x[r, c]), float(grid_y[r, c]))
        t = float(line.project(p) / L)  # fraction [0,1]
        h = float(np.interp(t, fracs, anchor_heads))  # clamps at 0/1
        heads_for_cells.append(h)

    return heads_for_cells, head_f0, head_f1

# def compile_chd_data(
#     river_cells: list[tuple[int, int, int, float]],
#     left_cells_0: list[tuple[int, int, int]], left_heads: list[float],
#     right_cells_0: list[tuple[int, int, int]], right_heads: list[float],
#     up_cells_0: list[tuple[int, int, int]], up_heads: list[float],
#     down_cells_0: list[tuple[int, int, int]], down_heads: list[float],
#     *,
#     nlay: int = 1,
#     copy_boundary_heads_to_all_layers: bool = True,
# ) -> tuple[list[list[float]], int, int]:
#     """
#     Build CHD stress period data.

#     Changes:
#     --------
#     - Boundary heads (left/right/up/down) are **copied down all layers** when
#       copy_boundary_heads_to_all_layers=True (constant head with depth).
#     - River cells remain on the top layer only (as before).
#     """
#     chd_data: list[list[float]] = []
#     unique: set[tuple[int, int, int]] = set()
#     dupes: set[tuple[int, int, int]] = set()

#     # River cells (top layer only; unchanged)
#     for (k, j, i, head) in river_cells:
#         if (k, j, i) not in unique:
#             chd_data.append([k, j, i, float(head)])
#             unique.add((k, j, i))
#         else:
#             dupes.add((k, j, i))

#     def _add_boundary(cells0, heads0):
#         for idx, (_k0, j, i) in enumerate(cells0):
#             head_val = float(heads0[idx])
#             if copy_boundary_heads_to_all_layers and nlay and nlay > 1:
#                 ks = range(nlay)
#             else:
#                 ks = (0,)
#             for k in ks:
#                 key = (int(k), int(j), int(i))
#                 if key not in unique:
#                     chd_data.append([int(k), int(j), int(i), head_val])
#                     unique.add(key)
#                 else:
#                     dupes.add(key)

#     _add_boundary(left_cells_0, left_heads)
#     _add_boundary(right_cells_0, right_heads)
#     _add_boundary(up_cells_0, up_heads)
#     _add_boundary(down_cells_0, down_heads)

#     return chd_data, len(unique), len(dupes)

def compile_chd_data(
    river_cells: list[tuple[int, int, int, float]],
    left_cells_0: list[tuple[int, int, int]], left_heads: list[float],
    right_cells_0: list[tuple[int, int, int]], right_heads: list[float],
    up_cells_0: list[tuple[int, int, int]], up_heads: list[float],
    down_cells_0: list[tuple[int, int, int]], down_heads: list[float],
    *,
    nlay: int = 1,
    copy_boundary_heads_to_all_layers: bool = True,
) -> tuple[list[list[float]], int, int]:
    """
    Build CHD stress period data as a *flat list* of [k, j, i, head] rows.
    - River cells are included on the *top* layer only (k as provided, usually 0).
    - Side boundaries (left/right/up/down) can be copied to all layers when
      `copy_boundary_heads_to_all_layers=True`.

    IMPORTANT (Option B):
    - We do *not* set IFACE here. In `build_gwf_model(...)` we split the CHD
      into two packages and set IFACE=6 **only** for the river package.
    """
    chd_data: list[list[float]] = []
    unique: set[tuple[int, int, int]] = set()
    dupes: set[tuple[int, int, int]] = set()

    # 1) River (top layer only)
    for (k, j, i, head) in river_cells:
        key = (int(k), int(j), int(i))
        if key not in unique:
            chd_data.append([int(k), int(j), int(i), float(head)])
            unique.add(key)
        else:
            dupes.add(key)

    # 2) Side boundaries
    def _add_boundary(cells0, heads0):
        for idx, (_k0, j, i) in enumerate(cells0):
            head_val = float(heads0[idx])
            ks = range(nlay) if (copy_boundary_heads_to_all_layers and nlay and nlay > 1) else (0,)
            for k in ks:
                key = (int(k), int(j), int(i))
                if key not in unique:
                    chd_data.append([int(k), int(j), int(i), head_val])
                    unique.add(key)
                else:
                    dupes.add(key)

    _add_boundary(left_cells_0,  left_heads)
    _add_boundary(right_cells_0, right_heads)
    _add_boundary(up_cells_0,    up_heads)
    _add_boundary(down_cells_0,  down_heads)

    return chd_data, len(unique), len(dupes)


# ----------------------------
# NEW — WSE edge/endpoint helpers (valid WSE only)
# ----------------------------
def endpoints_and_line(line_gdf: gpd.GeoDataFrame) -> tuple[Optional[Point], Optional[Point], Optional[LineString]]:
    """
    Merge a (multi)line GeoDataFrame into a single representative LineString
    (longest if multiple), and return (first_point, last_point, merged_line).
    """
    if line_gdf is None or line_gdf.empty:
        return None, None, None
    try:
        merged = linemerge(line_gdf.unary_union)
    except Exception:
        merged = line_gdf.geometry.unary_union

    line = None
    try:
        if isinstance(merged, LineString):
            line = merged
        elif getattr(merged, "geom_type", "") == "MultiLineString":
            line = max(list(merged.geoms), key=lambda g: g.length)
    except Exception:
        pass

    if line is None:
        try:
            g0 = line_gdf.geometry.iloc[0]
            if g0.geom_type == "LineString":
                line = g0
            elif g0.geom_type == "MultiLineString":
                line = max(list(g0.geoms), key=lambda g: g.length)
        except Exception:
            return None, None, None

    coords = list(line.coords)
    return Point(coords[0]), Point(coords[-1]), line


def sort_cells_along_line(cells: list[tuple[int, int, int]], grid_x: np.ndarray, grid_y: np.ndarray, line: LineString) -> list[tuple[int, int, int]]:
    """Sort first-layer boundary cells by projected distance along the given line."""
    if not cells or line is None:
        return cells

    def _key(cell):
        _, r, c = cell
        p = Point(float(grid_x[r, c]), float(grid_y[r, c]))
        try:
            return float(line.project(p))
        except Exception:
            return 0.0

    return sorted(cells, key=_key)


def build_wse_valid_edge_index(
    wse_raster_path: str | Path,
    *,
    extra_nodata: tuple[float, ...] = (-9999.0,),
) -> dict:
    """
    Build an index describing the *valid* WSE area and its *edge* pixels:
      - valid_polygon : union of polygons that correspond to valid (non-NoData) WSE
      - edge_x/edge_y : centers of valid pixels that touch NoData (the "edge")
      - edge_values   : WSE at those edge pixels
      - transform/crs : raster georeferencing

    Implementation detail:
      Avoids rasterio.features.shapes() (and thus GDAL MEM:::DATAPOINTER) by:
        1) Detecting the edge with morphological erosion
        2) Extracting boundary lines with marching squares (scikit-image)
        3) Polygonizing those lines with Shapely
        4) Keeping only polygons whose representative point lies in a valid pixel
    """
    from scipy.ndimage import binary_erosion
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol, xy as rio_xy
    from shapely.geometry import LineString, Polygon, Point
    from shapely.ops import unary_union, polygonize
    from pathlib import Path
    import math

    # scikit-image is a declared dependency; import directly. Do NOT pip-install at
    # runtime — that fails on read-only deploy filesystems (e.g. Posit Connect Cloud).
    try:
        from skimage import measure
    except Exception as e:
        raise ImportError(
            "scikit-image is required for WSE edge extraction (marching squares). "
            "Install it with: pip install scikit-image"
        ) from e

    wse_raster_path = Path(wse_raster_path)
    if not wse_raster_path.exists():
        raise FileNotFoundError(f"WSE raster not found: {wse_raster_path}")

    with rasterio.open(wse_raster_path) as src:
        arr = src.read(1)
        tfm = src.transform
        crs = src.crs
        nod = src.nodata

        # --- Build boolean valid mask (north-up or rotated—doesn't matter)
        invalid = ~np.isfinite(arr)
        if nod is not None:
            invalid |= np.isclose(arr, nod)
        for v in extra_nodata:
            invalid |= np.isclose(arr, float(v))
        invalid |= (arr <= -1.0e20)
        valid = ~invalid

        if not np.any(valid):
            raise ValueError("No valid WSE pixels found in the raster.")

        # --- Edge mask: valid pixels whose 8-neighborhood touches invalid
        try:
            interior = binary_erosion(valid, structure=np.ones((3, 3), dtype=bool), border_value=0)
            edge_mask = valid & (~interior)
        except Exception:
            # Conservative fallback
            edge_mask = valid.copy()

        # --- Edge pixel centers (+ values) — unchanged behavior
        rr, cc = np.nonzero(edge_mask)
        if rr.size == 0:
            rr, cc = np.nonzero(valid)  # degenerate case: mark all valids as edge
        xs, ys = rio_xy(tfm, rr, cc, offset="center")
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        edge_vals = arr[rr, cc].astype(float)

        # --- Build boundary lines via marching squares (no GDAL MEM)
        #     find_contours returns arrays of [row, col] vertices in array coords.
        contours = measure.find_contours(valid.astype(np.uint8), level=0.5, fully_connected="high")
        lines: list[LineString] = []
        if contours:
            # Vectorized transform: (col,row) -> world (x,y)
            a, b, c = tfm.a, tfm.b, tfm.c
            d, e, f = tfm.d, tfm.e, tfm.f
            for cont in contours:
                # cont[:, 0] -> rows (y-index), cont[:, 1] -> cols (x-index)
                rows = cont[:, 0].astype(float)
                cols = cont[:, 1].astype(float)
                xw = a * cols + b * rows + c
                yw = d * cols + e * rows + f
                if xw.size >= 2:
                    # Close tiny loops may degenerate; enforce minimum length
                    try:
                        ln = LineString(np.column_stack([xw, yw]))
                        if ln.length > 0:
                            lines.append(ln)
                    except Exception:
                        continue

        # --- Polygonize the boundary graph and keep only faces that are truly "valid"
        valid_polygon = None
        if lines:
            polys = list(polygonize(unary_union(lines)))
            kept: list[Polygon] = []
            for poly in polys:
                if not (poly and poly.is_valid and (poly.area > 0)):
                    continue
                # Use representative_point() (guaranteed interior) to test membership in the valid mask
                rp = poly.representative_point()
                try:
                    r, c_ = rowcol(tfm, rp.x, rp.y)
                except Exception:
                    # If inverse transform fails, skip this candidate face
                    continue
                if (0 <= r < src.height) and (0 <= c_ < src.width) and bool(valid[r, c_]):
                    kept.append(poly)

            if kept:
                valid_polygon = unary_union(kept)

        # --- Robust fallback if polygonization produced nothing (e.g., valid region touches dataset edge)
        if valid_polygon is None or valid_polygon.is_empty:
            # Buffer the traced boundary by ~half a pixel to create an areal shell.
            # This keeps downstream "distance to boundary" logic meaningful.
            # Pixel sizes (account for potential rotation terms):
            px = math.hypot(tfm.a, tfm.d) if (tfm.d != 0 or tfm.a != 0) else abs(tfm.a)
            py = math.hypot(tfm.b, tfm.e) if (tfm.b != 0 or tfm.e != 0) else abs(tfm.e)
            buf = 0.5 * max(px if px > 0 else 1.0, py if py > 0 else 1.0)

            if lines:
                edge_boundary = unary_union(lines)
                valid_polygon = edge_boundary.buffer(buf, cap_style=2, join_style=2)
            else:
                # Last-resort: buffer edge pixel centers; union may be heavy on giant rasters,
                # but this path is rare and keeps behavior safe without GDAL.
                pts = [Point(x, y) for x, y in zip(xs.tolist(), ys.tolist())]
                valid_polygon = unary_union([p.buffer(buf, cap_style=1) for p in pts])

        return {
            "valid_polygon": valid_polygon,
            "edge_x": xs,
            "edge_y": ys,
            "edge_values": edge_vals,
            "transform": tfm,
            "crs": crs,
        }


def nearest_wse_edge_distance_and_value(
    wse_edge_index: dict,
    pt: Point
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """
    For a given point (e.g., an endpoint on a GW boundary line), compute:
      - distance to the *boundary of valid WSE region* (in raster/map units)
      - WSE value at the *nearest valid edge pixel* to that boundary point
      - (x,y) of the nearest edge pixel center
      - (x,y) of the nearest point on the valid boundary

    Returns: (distance, wse_value, (edge_x, edge_y), (boundary_x, boundary_y))
    """
    border = wse_edge_index["valid_polygon"].boundary
    nearest_on_border = nearest_points(pt, border)[1]
    dist = float(pt.distance(border))

    ex = wse_edge_index["edge_x"]
    ey = wse_edge_index["edge_y"]
    ev = wse_edge_index["edge_values"]

    if ex.size == 0:
        return dist, float("nan"), (float("nan"), float("nan")), (nearest_on_border.x, nearest_on_border.y)

    dx = ex - nearest_on_border.x
    dy = ey - nearest_on_border.y
    idx = int(np.argmin(dx * dx + dy * dy))
    return (
        dist,
        float(ev[idx]),
        (float(ex[idx]), float(ey[idx])),
        (float(nearest_on_border.x), float(nearest_on_border.y)),
    )


# ----------------------------
# Raster sampling helpers
# ----------------------------
def csv_points_elevation(points_gdf: gpd.GeoDataFrame,
                         raster_path: str | Path,
                         out_csv: str | Path | None = None,
                         nodata_values: tuple[int | float, ...] = (-9999,)) -> pd.DataFrame:
    """
    Sample *raster_path* at each point in *points_gdf* and return a clean
    DataFrame with x, y, elevation.
    """
    from rasterio.transform import rowcol

    raster_path = Path(raster_path)
    vals: list[float] = []

    with rasterio.open(raster_path) as src:
        band = src.read(1)
        nodat = src.nodata
        nodata_set = {nodat} if nodat is not None else set()
        nodata_set.update(nodata_values)

        for pt in points_gdf.geometry:
            r, c = rowcol(src.transform, pt.x, pt.y)
            if 0 <= r < src.height and 0 <= c < src.width:
                vals.append(band[r, c])
            else:
                vals.append(np.nan)

    df = points_gdf.copy()
    df["x"] = df.geometry.x
    df["y"] = df.geometry.y
    df["elevation"] = vals

    df = df.dropna(subset=["elevation"])

    drop_mask = np.zeros(len(df), dtype=bool)
    for nd in nodata_set:
        drop_mask |= np.isclose(df["elevation"], nd)

    df = df.loc[~drop_mask, ["x", "y", "elevation"]]

    if out_csv is not None:
        out_csv = Path(out_csv).with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"Elevation CSV written → {out_csv}")

    return df


def sample_surface_elevations_to_grid_points(surface_raster: str, grid_points: gpd.GeoDataFrame, out_csv: Path) -> pd.DataFrame:
    coords = [(pt.x, pt.y) for pt in grid_points.geometry]
    vals: list[float | None] = []
    with rasterio.open(surface_raster) as src:
        for x, y in coords:
            row, col = rasterio.transform.rowcol(src.transform, x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                vals.append(src.read(1)[row, col])
            else:
                vals.append(None)
    df = pd.DataFrame({"x": [c[0] for c in coords], "y": [c[1] for c in coords], "elevation": vals})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def sample_surface_elevations_to_grid_points_new(
    surface_raster: str,
    grid_points: gpd.GeoDataFrame,
    out_csv: Path
) -> pd.DataFrame:
    df_valid = csv_points_elevation(
        points_gdf=grid_points,
        raster_path=surface_raster,
        out_csv=None,
        nodata_values=(-9999,)
    )

    base = pd.DataFrame({
        "x": grid_points.geometry.x.to_numpy(),
        "y": grid_points.geometry.y.to_numpy(),
    })
    out = base.merge(df_valid, on=["x", "y"], how="left")

    if "elevation" not in out.columns:
        out["elevation"] = np.nan
    out["elevation"] = out["elevation"].astype(object)
    out.loc[out["elevation"].isna(), "elevation"] = None

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    return out


# ----------------------------
# Misc helpers
# ----------------------------
def coerce_to_float(val, default: float = 0.0) -> float:
    """
    Robustly coerce Flopy MFScalar / numpy scalars / arrays / Python numbers to float.
    Falls back to `default` on failure.
    """
    try:
        # Already numeric?
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

        # Flopy MFScalar and similar may expose get_data()
        if hasattr(val, "get_data"):
            data = val.get_data()
            return float(np.asarray(data).squeeze())

        # Common attributes
        for attr in ("data", "array", "value"):
            if hasattr(val, attr):
                data = getattr(val, attr)
                # If it's a method, call it
                if callable(data):
                    data = data()
                return float(np.asarray(data).squeeze())

        # Try generic numpy coercion
        return float(np.asarray(val).squeeze())
    except Exception:
        return float(default)


def get_max_elevation(boundary_cells_coords: list[tuple[float, float]], grid_points_df: pd.DataFrame) -> float | None:
    elevations = []
    for x, y in boundary_cells_coords:
        sel = grid_points_df.loc[(grid_points_df['x'] == x) & (grid_points_df['y'] == y), 'elevation'].values
        if sel.size > 0:
            elevations.append(sel[0])
    return max(elevations) if elevations else None


def calculate_gw_elevation(boundary_cells, top_elevation: float, offset: float) -> list[float]:
    return [float(top_elevation + offset) for _ in boundary_cells]


def get_boundary_first_last(boundary_cells):
    if not boundary_cells:
        return (None, None)
    return boundary_cells[0], boundary_cells[-1]


def interpolate_gw_elevation_first_layer_only(first_layer_cells: list[tuple[int, int, int]], head_first: float, head_last: float) -> list[float]:
    n = len(first_layer_cells)
    if n <= 1:
        return [head_first] * n
    return [head_first + (head_last - head_first) * (i / (n - 1)) for i in range(n)]


def fit_csv_to_grid(df: pd.DataFrame, ncol: int, nrow: int, xmin: float, ymin: float, xmax: float, ymax: float) -> pd.DataFrame:
    x_spacing = (xmax - xmin) / ncol
    y_spacing = (ymax - ymin) / nrow
    out = df.copy()
    out['x_transformed'] = ((out['x'] - xmin) / x_spacing).astype(int)
    out['y_transformed'] = ((out['y'] - ymin) / y_spacing).astype(int)
    return out


def extract_river_cells(df: pd.DataFrame, idomain: np.ndarray) -> list[tuple[int, int, int, float]]:
    river = []
    for _, row in df.iterrows():
        x, y = int(row['x_transformed']), int(row['y_transformed'])
        elev = float(row['elevation'])
        if 0 <= x < idomain.shape[2] and 0 <= y < idomain.shape[1]:
            if idomain[0, y, x] == 1:
                river.append((0, y, x, elev))
    return river



# ----------------------------
# Step 6 – Build & run models
# ----------------------------
# def build_gwf_model(cfg, chd_data: Sequence[Sequence[float]], idomain: np.ndarray) -> tuple[flopy.mf6.MFSimulation, flopy.mf6.ModflowGwf]:
#     """
#     Build a steady-state GWF model. Writes large arrays as external *binary* files for faster I/O:
#       - DIS: top, botm, (idomain if present)
#       - IC : strt
#       - NPF: k (horizontal K) and k33 (vertical K)  <-- added

#     Notes
#     -----
#     * If KH/KV polygons are not provided, uniform cfg.kh/cfg.kv are expanded to full
#       (nlay, nrow, ncol) arrays so that NPF K/K33 are still written as binary per-layer files.
#     """
#     if idomain.shape != (cfg.nlay, cfg.nrow, cfg.ncol):
#         raise ValueError("`idomain` dimensions do not match cfg grid.")

#     # exe_name: use explicit path if provided, otherwise rely on 'mf6' on PATH
#     mf6_exe = str(cfg.md6_exe_path) if getattr(cfg, "md6_exe_path", None) else "mf6"

#     sim = flopy.mf6.MFSimulation(
#         sim_name=cfg.sim_name,
#         exe_name=mf6_exe,
#         sim_ws=str(cfg.gwf_ws),
#     )
#     flopy.mf6.ModflowTdis(
#         sim, time_units=cfg.time_units.upper(),
#         nper=cfg.nper, perioddata=[(cfg.perlen, cfg.nstp, cfg.tsmult)]
#     )
#     gwf = flopy.mf6.ModflowGwf(sim, modelname=cfg.gwf_name, save_flows=True)

#     # DIS
#     flopy.mf6.ModflowGwfdis(
#         gwf, nlay=cfg.nlay, nrow=cfg.nrow, ncol=cfg.ncol,
#         delr=cfg.cell_size_x, delc=cfg.cell_size_y,
#         top=cfg.tops[0], botm=cfg.botm, idomain=idomain,
#         xorigin=cfg.xmin, yorigin=cfg.ymin
#     )
#     # Keep CRS only; DIS.xorigin/yorigin control placement.
#     gwf.modelgrid.crs = cfg.raster_crs

#     # IC
#     strt = np.full((cfg.nlay, cfg.nrow, cfg.ncol), cfg.bed_elevation, dtype=float)
#     flopy.mf6.ModflowGwfic(gwf, strt=strt)

#     # Build KH / KV arrays (from polygons if present; else uniform)
#     k_array, k33_array = _kh_arrays_from_polygon(cfg, gwf, idomain)
#     if k_array is None:
#         k_array = np.full((cfg.nlay, cfg.nrow, cfg.ncol), float(cfg.kh), dtype=float)
#     if k33_array is None:
#         k33_array = np.full((cfg.nlay, cfg.nrow, cfg.ncol), float(cfg.kv), dtype=float)

#     # NPF (pass arrays; we'll also set external binary records below)
#     flopy.mf6.ModflowGwfnpf(
#         gwf,
#         icelltype=2,
#         k=k_array,
#         k33=k33_array,
#         save_flows=True,
#         save_saturation=True,
#         save_specific_discharge=True,
#     )

#     # # CHD
#     # if chd_data:
#     #     flopy.mf6.ModflowGwfchd(
#     #         gwf, maxbound=len(chd_data),
#     #         stress_period_data={0: chd_data}, save_flows=True
#     #     )

#     # CHD (per-cell IFACE provided)
#     if chd_data:
#         flopy.mf6.ModflowGwfchd(
#             gwf,
#             maxbound=len(chd_data),
#             stress_period_data={0: chd_data},  # list of ((k,j,i), head, IFACE)
#             save_flows=True,
#             auxiliary=["IFACE"],
#         )

#     # OC
#     flopy.mf6.ModflowGwfoc(
#         gwf,
#         saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
#         head_filerecord=[cfg.headfile],
#         budget_filerecord=[cfg.budgetfile],
#         printrecord=[("HEAD", "LAST")]
#     )

#     # IMS
#     flopy.mf6.ModflowIms(
#         sim, print_option="SUMMARY",
#         outer_dvclose=1e-4, outer_maximum=200,
#         inner_maximum=500, inner_dvclose=1e-4, rcloserecord=1e-4,
#         linear_acceleration="BICGSTAB", relaxation_factor=0.97
#     )

#     # ---------- Write big arrays as external binary ----------
#     external_dir = Path(gwf.model_ws) / "arrays"
#     external_dir.mkdir(exist_ok=True)

#     dis = gwf.get_package("DIS")
#     ic = gwf.get_package("IC")
#     npf = gwf.get_package("NPF")

#     def _layered_records(array: np.ndarray, basename: str) -> list[dict]:
#         """Build FloPy 'set_record' entries for a 3D array (nlay,nrow,ncol), one file per layer."""
#         return [{
#             "filename": str(PurePath("arrays") / f"{basename}_L{lay + 1}.bin"),
#             "binary": True,
#             "data": np.asarray(array[lay]),
#             "iprn": 0,
#             "factor": 1.0,
#         } for lay in range(array.shape[0])]

#     # DIS: top/botm/idomain → binary (as you had)
#     if dis is not None:
#         dis.top.set_record({
#             "filename": str(PurePath("arrays") / "top.bin"),
#             "binary": True, "data": np.asarray(dis.top.array), "iprn": 0, "factor": 1.0
#         })
#         dis.botm.set_record(_layered_records(dis.botm.array, "botm"))
#         if hasattr(dis, "idomain") and dis.idomain.array is not None:
#             dis.idomain.set_record(_layered_records(dis.idomain.array, "idomain"))

#     # IC: strt → binary (as you had)
#     if ic is not None:
#         ic.strt.set_record(_layered_records(ic.strt.array, "strt"))

#     # NPF: NEW — write k and k33 per layer as binary
#     # Ensure we use the arrays we constructed above (k_array/k33_array)
#     if npf is not None:
#         # Horizontal K
#         if getattr(npf, "k", None) is not None:
#             npf.k.set_record(_layered_records(k_array, "k"))
#         # Vertical K
#         if getattr(npf, "k33", None) is not None:
#             npf.k33.set_record(_layered_records(k33_array, "k33"))

#     return sim, gwf
def build_gwf_model(
    cfg,
    chd_data: Sequence[Sequence[float]],
    idomain: np.ndarray,
    river_cells: list[tuple[int, int, int, float]] | None = None,
) -> tuple[flopy.mf6.MFSimulation, flopy.mf6.ModflowGwf]:
    """
    Build a steady-state GWF model. Writes big arrays externally and
    installs CHD as **two packages** (Option B):

      • CHD_RIVER : river/top-surface cells with IFACE=6 (top) via auxiliary.
      • CHD_SIDES : all other CHD cells (no IFACE supplied).

    Notes
    -----
    * 'chd_data' must be plain rows [k, j, i, head].
    * 'river_cells' must be (k, j, i, stage). Typically k=0 for top layer.
    """
    if idomain.shape != (cfg.nlay, cfg.nrow, cfg.ncol):
        raise ValueError("`idomain` dimensions do not match cfg grid.")

    mf6_exe = str(cfg.md6_exe_path) if getattr(cfg, "md6_exe_path", None) else "mf6"

    sim = flopy.mf6.MFSimulation(
        sim_name=cfg.sim_name,
        exe_name=mf6_exe,
        sim_ws=str(cfg.gwf_ws),
    )
    flopy.mf6.ModflowTdis(
        sim, time_units=cfg.time_units.upper(),
        nper=cfg.nper, perioddata=[(cfg.perlen, cfg.nstp, cfg.tsmult)]
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=cfg.gwf_name, save_flows=True)

    # DIS
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=cfg.nlay, nrow=cfg.nrow, ncol=cfg.ncol,
        delr=cfg.cell_size_x, delc=cfg.cell_size_y,
        top=cfg.tops[0], botm=cfg.botm, idomain=idomain,
        xorigin=cfg.xmin, yorigin=cfg.ymin
    )
    gwf.modelgrid.crs = cfg.raster_crs

    # IC
    strt = np.full((cfg.nlay, cfg.nrow, cfg.ncol), cfg.bed_elevation, dtype=float)
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # NPF
    k_array, k33_array = _kh_arrays_from_polygon(cfg, gwf, idomain)
    if k_array is None:
        k_array = np.full((cfg.nlay, cfg.nrow, cfg.ncol), float(cfg.kh), dtype=float)
    if k33_array is None:
        k33_array = np.full((cfg.nlay, cfg.nrow, cfg.ncol), float(cfg.kv), dtype=float)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=2,
        k=k_array,
        k33=k33_array,
        save_flows=True,
        save_saturation=True,
        save_specific_discharge=True,
    )

    # === CHD (Option B): split into "river with IFACE" vs "sides without IFACE" ===
    spd_river: list[tuple[tuple[int, int, int], float, int]] = []
    spd_sides: list[tuple[int, int, int, float]] = []

    river_set = {(int(k), int(j), int(i)) for (k, j, i, _stage) in (river_cells or [])}

    # Build river CHD from 'river_cells' (IFACE = 6 = top)
    if river_cells:
        for (k, j, i, stage) in river_cells:
            spd_river.append(((int(k), int(j), int(i)), float(stage), 6))

    # Sides = all CHD rows that are not river
    if chd_data:
        for rec in chd_data:
            k, j, i, head = int(rec[0]), int(rec[1]), int(rec[2]), float(rec[3])
            if (k, j, i) in river_set:
                # skip; already handled in spd_river (ensures no duplicates)
                continue
            spd_sides.append((k, j, i, head))

    # Install the two packages
    if spd_river:
        flopy.mf6.ModflowGwfchd(
            gwf,
            maxbound=len(spd_river),
            stress_period_data={0: spd_river},  # ((k,j,i), head, IFACE)
            auxiliary=["IFACE"],                 # expose IFACE
            save_flows=True,
            pname="CHD_RIVER",
        )
    if spd_sides:
        flopy.mf6.ModflowGwfchd(
            gwf,
            maxbound=len(spd_sides),
            stress_period_data={0: spd_sides},  # (k, j, i, head) tuples
            save_flows=True,
            pname="CHD_SIDES",
        )

    # OC
    flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        head_filerecord=[cfg.headfile],
        budget_filerecord=[cfg.budgetfile],
        printrecord=[("HEAD", "LAST")]
    )

    # IMS
    flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY",
        outer_dvclose=1e-4, outer_maximum=200,
        inner_maximum=500, inner_dvclose=1e-4, rcloserecord=1e-4,
        linear_acceleration="BICGSTAB", relaxation_factor=0.97
    )

    # ---------- Externalize large arrays (as in your original) ----------
    external_dir = Path(gwf.model_ws) / "arrays"
    external_dir.mkdir(exist_ok=True)

    dis = gwf.get_package("DIS")
    ic = gwf.get_package("IC")
    npf = gwf.get_package("NPF")

    def _layered_records(array: np.ndarray, basename: str) -> list[dict]:
        return [{
            "filename": str(PurePath("arrays") / f"{basename}_L{lay + 1}.bin"),
            "binary": True,
            "data": np.asarray(array[lay]),
            "iprn": 0,
            "factor": 1.0,
        } for lay in range(array.shape[0])]

    if dis is not None:
        dis.top.set_record({
            "filename": str(PurePath("arrays") / "top.bin"),
            "binary": True, "data": np.asarray(dis.top.array), "iprn": 0, "factor": 1.0
        })
        dis.botm.set_record(_layered_records(dis.botm.array, "botm"))
        if hasattr(dis, "idomain") and dis.idomain.array is not None:
            dis.idomain.set_record(_layered_records(dis.idomain.array, "idomain"))

    if ic is not None:
        ic.strt.set_record(_layered_records(ic.strt.array, "strt"))

    if npf is not None:
        if getattr(npf, "k", None) is not None:
            npf.k.set_record(_layered_records(k_array, "k"))
        if getattr(npf, "k33", None) is not None:
            npf.k33.set_record(_layered_records(k33_array, "k33"))

    return sim, gwf


def _kh_arrays_from_polygon(cfg, gwf, idomain: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Construct 3D arrays for KH (and optionally KV) from a polygon shapefile.

    Expected polygon attributes (case-insensitive):
      ZONE_ID, KH, KV, TOP_ELEV, BOT_ELEV, LABEL

    Rules:
      - For each horizontal cell, assign the KH/KV of the polygon with the
        largest area overlap (dominant polygon).
      - If TOP_ELEV/BOT_ELEV are present, only apply values to layers whose
        [layer_top, layer_bot] interval intersects [BOT_ELEV, TOP_ELEV].
      - Cells with no polygon coverage use defaults cfg.kh / cfg.kv.
    """
    shp_path = getattr(cfg, "kh_polygon_shapefile", None)
    if shp_path is None:
        gdf_poly = getattr(cfg, "kh_polygon_gdf", None)
    else:
        gdf_poly = None
    # Load if not already loaded
    if gdf_poly is None and shp_path is not None and Path(shp_path).exists():
        gdf_poly = gpd.read_file(shp_path)

    if gdf_poly is None or gdf_poly.empty:
        return None, None

    # Reproject to model grid CRS
    try:
        mg = gwf.modelgrid
        crs = getattr(mg, "crs", None)
        if crs is not None:
            gdf_poly = gdf_poly.to_crs(crs)
    except Exception:
        pass

    # Normalize attribute column names to uppercase for robust access, but
    # DO NOT rename the active geometry column (renaming it breaks GeoPandas state)
    try:
        geom_col = gdf_poly.geometry.name
    except Exception:
        geom_col = 'geometry' if 'geometry' in gdf_poly.columns else None
    col_map = {c: c.upper() for c in gdf_poly.columns if c != geom_col}
    if col_map:
        gdf_poly = gdf_poly.rename(columns=col_map)
    if geom_col and gdf_poly.geometry.name != geom_col:
        try:
            gdf_poly = gdf_poly.set_geometry(geom_col, inplace=False)
        except Exception:
            pass

    has_kh = "KH" in gdf_poly.columns
    has_kv = "KV" in gdf_poly.columns
    has_top = "TOP_ELEV" in gdf_poly.columns
    has_bot = "BOT_ELEV" in gdf_poly.columns

    if not has_kh:
        print("[WARN] KH polygon shapefile missing 'KH' attribute; using uniform KH.")
        return None, None

    dis = gwf.get_package("DIS")
    nlay, nrow, ncol = idomain.shape
    Xv = gwf.modelgrid.xvertices
    Yv = gwf.modelgrid.yvertices

    rows = []
    cols = []
    geoms = []
    for r in range(nrow):
        for c in range(ncol):
            poly = Polygon([
                (Xv[r, c],     Yv[r, c]),
                (Xv[r, c+1],   Yv[r, c+1]),
                (Xv[r+1, c+1], Yv[r+1, c+1]),
                (Xv[r+1, c],   Yv[r+1, c]),
            ])
            geoms.append(poly)
            rows.append(r)
            cols.append(c)
    g_cells = gpd.GeoDataFrame({"row": rows, "col": cols}, geometry=geoms, crs=getattr(gwf.modelgrid, "crs", None))

    inter = gpd.overlay(g_cells, gdf_poly, how="intersection")
    if inter is None or inter.empty:
        return None, None

    inter["_area"] = inter.geometry.area
    inter_sorted = inter.sort_values(["row", "col", "_area"], ascending=[True, True, False])
    inter_dominant = inter_sorted.drop_duplicates(subset=["row", "col"], keep="first")

    k = np.full((nlay, nrow, ncol), float(cfg.kh), dtype=float)
    k33 = np.full((nlay, nrow, ncol), float(cfg.kv), dtype=float)

    top2d = np.asarray(dis.top.array, dtype=float)
    bot3d = np.asarray(dis.botm.array, dtype=float)  # shape (nlay, nrow, ncol)

    for _, rec in inter_dominant.iterrows():
        r = int(rec["row"]); c = int(rec["col"])
        try:
            kh_val = float(rec["KH"]) if pd.notna(rec["KH"]) else float(cfg.kh)
        except Exception:
            kh_val = float(cfg.kh)
        if has_kv:
            try:
                kv_val = float(rec["KV"]) if pd.notna(rec["KV"]) else float(cfg.kv)
            except Exception:
                kv_val = float(cfg.kv)
        else:
            kv_val = float(cfg.kv)

        if has_top and has_bot:
            try:
                z_top = float(rec["TOP_ELEV"]) if pd.notna(rec["TOP_ELEV"]) else None
                z_bot = float(rec["BOT_ELEV"]) if pd.notna(rec["BOT_ELEV"]) else None
            except Exception:
                z_top = z_bot = None
        else:
            z_top = z_bot = None

        if z_top is None or z_bot is None:
            k[:, r, c] = kh_val
            k33[:, r, c] = kv_val
        else:
            for lay in range(nlay):
                lay_top = float(top2d[r, c]) if lay == 0 else float(bot3d[lay - 1, r, c])
                lay_bot = float(bot3d[lay, r, c])
                if (lay_top > z_bot) and (lay_bot < z_top):
                    k[lay, r, c] = kh_val
                    k33[lay, r, c] = kv_val

    return k, k33


def build_particle_models(
    sim_name: str,
    gwf: "flopy.mf6.ModflowGwf",
    river_cells: list[tuple[int, int, int, float]],
    *,
    mp7_ws: Path | str | None = None,
    exe_path: str | Path | None = None,
    nx: int = 1,                  # number of particles across X per river cell
    ny: int = 1,                  # number of particles across Y per river cell
    place_wse: bool = True,       # drop a particle at the water table (WSE)
    place_center: bool = False,   # drop a particle at zloc=0.5 (only if WSE >= 0.5 in that cell)
    place_bottom: bool = False,   # drop a particle at zloc=0.0 (cell bottom)
) -> tuple[Modpath7, Modpath7]:
    """
    Build MODPATH 7 forward/backward models with particles released in the **topmost
    wet cell** (the topmost layer that contains the river stage), using an (nx × ny)
    evenly spaced grid in local X/Y for each river cell.

    Vertical placement per XY point (toggles):
      - place_wse=True     → particle at the water table (WSE) inside host cell.
      - place_center=True  → particle at zloc=0.5 **only if** WSE zloc >= 0.5 in that cell; otherwise skipped.
      - place_bottom=True  → particle at zloc=0.0 (cell bottom).

    Notes
    -----
    - If all three toggles are False, we fall back to place_wse=True.
    - We do NOT set mpbas.defaultiface; CHD IFACEs are carried in the CHD package.
    - By default we leave weak source/sink behavior as pass_through (more stable).
    """
    # Ensure at least one vertical option
    if not (place_wse or place_center or place_bottom):
        place_wse = True

    # Sanitize counts
    nx = max(1, int(nx))
    ny = max(1, int(ny))

    mp7_ws = Path(mp7_ws or (Path(gwf.simulation.sim_path).parent / "mp7_workspace")).absolute()
    mp7_ws.mkdir(parents=True, exist_ok=True)
    exe_path = str(exe_path or "mp7")

    # Grid geometry
    dis = gwf.get_package("DIS")
    top2d  = np.asarray(dis.top.array,  dtype=float)        # (nrow, ncol)
    botm3d = np.asarray(dis.botm.array, dtype=float)        # (nlay, nrow, ncol)
    idomain = np.asarray(dis.idomain.array, dtype=int) if hasattr(dis, "idomain") else None
    nlay, nrow, ncol = botm3d.shape

    TOL = 1.0e-6
    TOL_CENTER = 1.0e-9

    def _host_layer_for_stage(irow: int, icol: int, stage: float) -> tuple[int, float, float]:
        """
        Return (k, lay_top, lay_bot) for the TOPMOST layer whose vertical interval
        contains 'stage'. If stage is above the column, clamp to top layer; if below,
        clamp to bottom layer.
        """
        for k in range(nlay):
            lay_top = top2d[irow, icol] if k == 0 else botm3d[k - 1, irow, icol]
            lay_bot = botm3d[k, irow, icol]
            if (stage <= lay_top + TOL) and (stage >= lay_bot - TOL):
                if idomain is None or idomain[k, irow, icol] == 1:
                    return k, lay_top, lay_bot
        if stage > (top2d[irow, icol] + TOL):
            return 0, top2d[irow, icol], botm3d[0, irow, icol]
        k = nlay - 1
        return k, (botm3d[-2, irow, icol] if nlay > 1 else top2d[irow, icol]), botm3d[-1, irow, icol]

    # Precompute evenly spaced local coordinates inside a cell (bin midpoints)
    xgrid = ((np.arange(nx, dtype=float) + 0.5) / float(nx)).tolist()
    ygrid = ((np.arange(ny, dtype=float) + 0.5) / float(ny)).tolist()

    def _make(direction: str) -> Modpath7:
        mp = Modpath7.create_mp7(
            modelname=f"{sim_name}_mp_{direction}",
            trackdir=direction,
            flowmodel=gwf,
            model_ws=mp7_ws,
            exe_name=exe_path,
            rowcelldivisions=1,
            columncelldivisions=1,
            layercelldivisions=1,
        )

        partlocs: list[tuple[int, int, int]] = []
        localx: list[float] = []
        localy: list[float] = []
        localz: list[float] = []

        for (_k_ignore, irow, icol, stage) in river_cells:
            if not (0 <= irow < nrow and 0 <= icol < ncol):
                continue
            if stage is None or not np.isfinite(stage):
                continue

            k_host, lay_top, lay_bot = _host_layer_for_stage(irow, icol, float(stage))
            if idomain is not None and idomain[k_host, irow, icol] != 1:
                continue

            dz = max(lay_top - lay_bot, TOL)
            zloc_wse = float(np.clip((stage - lay_bot) / dz, 0.0, 1.0))

            # Assemble the vertical positions requested for *this* cell
            z_candidates: list[float] = []
            if place_wse:
                z_candidates.append(zloc_wse)
            if place_center and (zloc_wse + TOL_CENTER >= 0.5):
                z_candidates.append(0.5)
            if place_bottom:
                z_candidates.append(0.0)

            if not z_candidates:
                # All requested options were skipped (e.g., only center requested but WSE < center)
                # Fall back to WSE in this cell to ensure at least one particle here.
                z_candidates.append(zloc_wse)

            # Create particles on the (nx × ny) XY grid at each requested Z
            for yloc in ygrid:
                for xloc in xgrid:
                    for zloc in z_candidates:
                        partlocs.append((int(k_host), int(irow), int(icol)))
                        localx.append(float(xloc))
                        localy.append(float(yloc))
                        localz.append(float(zloc))

        # Global fallback if nothing qualified at all
        if not partlocs:
            # Choose a reasonable fallback vertical list based on toggles
            fallback_zs: list[float] = []
            if place_wse:    fallback_zs.append(1.0)  # top face proxy
            if place_center: fallback_zs.append(0.5)
            if place_bottom: fallback_zs.append(0.0)
            if not fallback_zs:
                fallback_zs = [1.0]  # at least something

            for (_k_ignore, irow, icol, _stage) in river_cells:
                if 0 <= irow < nrow and 0 <= icol < ncol and (idomain is None or idomain[0, irow, icol] == 1):
                    for yloc in ygrid:
                        for xloc in xgrid:
                            for zloc in fallback_zs:
                                partlocs.append((0, int(irow), int(icol)))
                                localx.append(float(xloc))
                                localy.append(float(yloc))
                                localz.append(float(zloc))

        # Build particle group with local coordinates
        pg = ParticleGroup(
            particledata=ParticleData(
                partlocs,
                structured=True,
                localx=localx,
                localy=localy,
                localz=localz,
                drape=0,  # do not drape; we set z explicitly
            )
        )
        mpsim = mp.get_package("MPSIM")
        mpsim.particlegroups.clear()
        mpsim.particlegroups.append(pg)

        mpsim.weaksinkoption = "2"    # 2 is stop at
        mpsim.weaksourceoption = "2"  # 2 is stop at, 1 is pass through

        return mp

    return _make("forward"), _make("backward")


def write_models(*sims, silent=False):
    for sim in sims:
        if isinstance(sim, flopy.mf6.MFSimulation):
            sim.write_simulation(silent=silent)
        else:
            sim.write_input()


def run_models(*sims, silent=False):
    for sim in sims:
        if isinstance(sim, flopy.mf6.MFSimulation):
            print(f"Running simulation: {sim.name}")
            success, buff = sim.run_simulation(silent=silent, report=True)
        else:
            print(f"Running model: {sim.name}")
            success, buff = sim.run_model(silent=silent, report=True)
        if not success:
            print(f"Simulation {sim.name} failed.")
            print(buff)
            raise RuntimeError(f"{sim.name} failed")
        else:
            print(f"Simulation {sim.name} succeeded.")


# =========================
# NEW — Hydraulic head exports (netCDF + per-layer rasters + mosaic)
# =========================
def export_hydraulic_head_layers(*, cfg, gwf, log=print) -> dict:
    """
    Create:
      1) Per-layer head GeoTIFFs (rotation-aware placement)
      2) A mosaic dataset containing those rasters (tagged with layer index 'k')
      3) A single netCDF (z,y,x) when grid is unrotated & uniformly spaced

    Returns dict with keys: 'netcdf', 'mosaic_gdb', 'mosaic_dataset', 'geotiffs'
    """
    out = {"netcdf": None, "mosaic_gdb": None, "mosaic_dataset": None, "geotiffs": []}

    # --- Find a head file reliably ---
    from pathlib import Path
    from flopy.utils.binaryfile import HeadFile
    import numpy as np
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS as RioCRS

    gwf_ws = Path(cfg.gwf_ws)
    candidates = []

    # 1) YAML/Settings-specified headfile (if present)
    try:
        if getattr(cfg, "headfile", None):
            hf = Path(cfg.headfile)
            if not hf.is_absolute():
                hf = gwf_ws / hf
            candidates.append(hf)
    except Exception:
        pass

    # 2) Conventional names under the GWF workspace
    candidates.append(gwf_ws / f"{cfg.gwf_name}.hds")
    candidates.append(gwf_ws / "gwf_model.hds")

    # 3) Any *.hds in the workspace
    for p in sorted(gwf_ws.glob("*.hds")):
        candidates.append(p)

    hds_path = next((p for p in candidates if p.exists()), None)
    if not hds_path:
        log(f"[WARN] Head file not found (checked: {', '.join(str(p) for p in candidates)})")
        return out

    # --- Read steady-state head ---
    hobj = HeadFile(str(hds_path))
    head3d = hobj.get_data(totim=hobj.get_times()[-1])
    if head3d.ndim == 4:
        head3d = head3d[-1]

    # Mask inactive cells
    dis = gwf.get_package("DIS")
    idomain = dis.idomain.array
    head3d = np.where(idomain == 1, head3d, np.nan).astype("float32")

    delr = np.asarray(dis.delr.array, dtype=float)
    delc = np.asarray(dis.delc.array, dtype=float)
    nlay, nrow, ncol = head3d.shape
    angrot = float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0)

    # Robust origin extraction
    def _coerce_float(obj, default=0.0):
        try:
            if hasattr(obj, "get_data"):
                return float(np.asarray(obj.get_data()).squeeze())
            return float(np.asarray(obj).squeeze())
        except Exception:
            return float(default)

    xorigin = _coerce_float(getattr(dis, "xorigin", getattr(gwf.modelgrid, "xoffset", 0.0)),
                            getattr(gwf.modelgrid, "xoffset", 0.0))
    yorigin = _coerce_float(getattr(dis, "yorigin", getattr(gwf.modelgrid, "yoffset", 0.0)),
                            getattr(gwf.modelgrid, "yoffset", 0.0))

    uniform_x = np.allclose(delr, delr.flat[0])
    uniform_y = np.allclose(delc, delc.flat[0])
    dx = float(delr.flat[0]) if uniform_x else float(np.mean(delr))
    dy = float(delc.flat[0]) if uniform_y else float(np.mean(delc))

    # Output folders
    base_dir = Path(cfg.output_directory) / "summary" / "head"
    base_dir.mkdir(parents=True, exist_ok=True)
    tifs_dir = base_dir / "per_layer_tif"
    tifs_dir.mkdir(parents=True, exist_ok=True)

    # --- (A) Write per-layer GeoTIFFs (rotation-aware transform) ---
    theta = np.deg2rad(angrot)
    a = dx * np.cos(theta)
    b = -dy * np.sin(theta)
    d = dx * np.sin(theta)
    e = dy * np.cos(theta)
    transform = Affine(a, b, xorigin, d, e, yorigin)

    # Try to coerce CRS into something rasterio accepts (and keep it for WKT fallback)
    crs_rio = None
    try:
        if getattr(cfg, "projection_file", None) and Path(cfg.projection_file).exists():
            wkt = Path(cfg.projection_file).read_text().strip()
            if wkt:
                crs_rio = RioCRS.from_wkt(wkt)
        if crs_rio is None and getattr(cfg, "hec_ras_crs", None):
            crs_rio = RioCRS.from_user_input(cfg.hec_ras_crs)
    except Exception:
        crs_rio = None

    nodata = np.float32(-9999.0)
    geotiffs = []
    try:
        for k in range(nlay):
            arr = head3d[k, :, :].astype("float32")
            mask_bad = (~np.isfinite(arr)) | (arr <= np.float32(-1e20)) | (np.isclose(arr, np.float32(-9999.0)))
            arr = np.where(mask_bad, nodata, arr).astype("float32")

            tif = tifs_dir / f"head_L{(k+1):02d}.tif"
            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "nodata": nodata,
                "width": ncol,
                "height": nrow,
                "count": 1,
                "transform": transform,
                "compress": "lzw",
            }
            if crs_rio is not None:
                profile["crs"] = crs_rio
            with rasterio.open(tif, "w", **profile) as dst:
                dst.write(arr, 1)
            geotiffs.append(str(tif))
        out["geotiffs"] = geotiffs
        print(f"Wrote {len(geotiffs)} head GeoTIFFs → {tifs_dir}")
    except Exception as e:
        print(f"[WARN] Could not write head GeoTIFFs: {e}")

    # --- (B) Mosaic dataset (unchanged from your version) ---
    try:
        import arcpy
        arcpy.env.overwriteOutput = True

        try:
            lvl = "Unknown"
            info = arcpy.GetInstallInfo()
            for key in ("LicenseLevel", "License Level", "Edition"):
                v = info.get(key)
                if v:
                    low = str(v).lower()
                    lvl = {"arcview": "Basic", "arceditor": "Standard", "arcinfo": "Advanced"}.get(low, str(v))
                    break
        except Exception:
            try:
                prod = arcpy.ProductInfo()
                lvl = {"ArcView": "Basic", "ArcEditor": "Standard", "ArcInfo": "Advanced"}.get(prod, "Unknown")
            except Exception:
                lvl = "Unknown"

        if lvl not in ("Standard", "Advanced"):
            print(f"[INFO] Pro license level is {lvl}; skipping mosaic dataset build (requires Standard or Advanced).")
        else:
            gdb = base_dir / "head_md.gdb"
            if not gdb.exists():
                arcpy.management.CreateFileGDB(str(base_dir), gdb.name)

            sr = None
            try:
                if getattr(cfg, "projection_file", None) and Path(cfg.projection_file).exists():
                    wkt = Path(cfg.projection_file).read_text().strip()
                    if wkt:
                        sr = arcpy.SpatialReference()
                        sr.loadFromString(wkt)
                elif getattr(cfg, "hec_ras_crs", None):
                    sr = arcpy.SpatialReference(cfg.hec_ras_crs)
            except Exception:
                sr = None

            md_name = "head_md"
            md_path = str(gdb / md_name)
            if arcpy.Exists(md_path):
                arcpy.management.Delete(md_path)

            arcpy.management.CreateMosaicDataset(
                in_workspace=str(gdb),
                in_mosaicdataset_name=md_name,
                coordinate_system=(sr if sr else ""),
                num_bands=1,
                pixel_type="32_BIT_FLOAT",
                product_definition="NONE"
            )

            arcpy.management.AddRastersToMosaicDataset(
                in_mosaic_dataset=md_path,
                raster_type="Raster Dataset",
                input_path=str(tifs_dir),
                filter="*.tif",
                update_cellsize_ranges="UPDATE_CELL_SIZES",
                update_boundary="UPDATE_BOUNDARY",
                update_overviews="NO_OVERVIEWS"
            )

            try:
                if "k" not in [f.name for f in arcpy.ListFields(md_path)]:
                    arcpy.management.AddField(md_path, "k", "SHORT")
                arcpy.management.CalculateField(
                    in_table=md_path,
                    field="k",
                    expression="parse(!Name!)",
                    expression_type="PYTHON3",
                    code_block="""
def parse(name):
    import os, re
    base = os.path.basename(name or "")
    m = re.search(r"_L(\\d+)", base, re.IGNORECASE)
    return int(m.group(1)) if m else None
"""
                )
            except Exception:
                pass

            try:
                import arcpy.md as md
                md.BuildMultidimensionalInfo(
                    in_mosaic_dataset=md_path,
                    variable="head",
                    dimension_definitions=[["StdZ", "k", "", "", ""]]
                )
            except Exception:
                pass

            out["mosaic_gdb"] = str(gdb)
            out["mosaic_dataset"] = md_path
            print(f"Built mosaic dataset → {md_path}")

    except Exception as e:
        try:
            gp = arcpy.GetMessages(2)
            if gp:
                print(f"[WARN] Mosaic dataset step raised: {e}\n[GP]: {gp}")
            else:
                print(f"[WARN] Mosaic dataset step raised: {e}")
        except Exception:
            print(f"[WARN] Mosaic dataset step raised: {e}")

    # --- Helper: resolve an ESRI-friendly WKT for netCDF spatial_ref ---
    def _resolve_esri_wkt() -> str:
        # 1) explicit WKT from YAML
        w = (str(getattr(cfg, "projection_wkt", "") or "")).strip()
        if w:
            return w
        # 2) .prj file
        try:
            prj = getattr(cfg, "projection_file", None)
            if prj and Path(prj).exists():
                txt = Path(prj).read_text().strip()
                if txt:
                    return txt
        except Exception:
            pass
        # 3) hec_ras_crs / modelgrid CRS / GeoTIFF CRS
        for cand in (getattr(cfg, "hec_ras_crs", None),
                     getattr(getattr(gwf, "modelgrid", None), "crs", None),
                     (crs_rio.to_wkt() if crs_rio is not None else None)):
            if not cand:
                continue
            try:
                from pyproj import CRS
                try:
                    crs = CRS.from_user_input(cand)
                except Exception:
                    # If we were handed a WKT-like string already
                    try:
                        crs = CRS.from_wkt(str(cand))
                    except Exception:
                        crs = CRS.from_user_input(str(cand))
                # Prefer ESRI WKT for ArcGIS
                try:
                    from pyproj.enums import WKTVersion
                    return crs.to_wkt(WKTVersion.WKT1_ESRI)
                except Exception:
                    return crs.to_wkt()
            except Exception:
                try:
                    return str(cand)
                except Exception:
                    pass
        return ""

    # Optional: units for x/y axes in netCDF
    xy_units = str(getattr(cfg, "xy_units", "")) or ""

    # --- (C) netCDF head cube (unrotated & uniform spacing only) ---
    can_write_netcdf = (abs(angrot) < 1e-9) and uniform_x and uniform_y
    if can_write_netcdf:
        try:
            from netCDF4 import Dataset
            FILL = np.float32(-9999.0)
            BAD  = np.float32(1.0e20)

            # 1-D cell-center coordinates
            x_coords = xorigin + dx * (np.arange(ncol, dtype=float) + 0.5)
            y_coords = yorigin + dy * (np.arange(nrow, dtype=float) + 0.5)

            wkt_esri = _resolve_esri_wkt()

            nc_path = base_dir / "head_zyx.nc"
            with Dataset(nc_path, "w", format="NETCDF4") as ds:
                # Dimensions
                ds.createDimension("x", ncol)
                ds.createDimension("y", nrow)
                ds.createDimension("z", nlay)

                # Coord variables
                xvar = ds.createVariable("x", "f8", ("x",))
                yvar = ds.createVariable("y", "f8", ("y",))
                zvar = ds.createVariable("z", "f8", ("z",))

                xvar[:] = x_coords
                yvar[:] = y_coords
                zvar[:] = np.arange(1, nlay + 1, dtype=float)

                xvar.standard_name = "projection_x_coordinate"
                xvar.long_name = "x coordinate of projection"
                xvar.axis = "X"
                if xy_units:
                    xvar.units = xy_units

                yvar.standard_name = "projection_y_coordinate"
                yvar.long_name = "y coordinate of projection"
                yvar.axis = "Y"
                yvar.positive = "up"
                if xy_units:
                    yvar.units = xy_units

                zvar.long_name = "Layer index (top=1)"
                zvar.axis = "Z"

                # Spatial reference (write both 'spatial_ref' and 'esri_pe_string')
                sref = ds.createVariable("spatial_ref", "i4")
                if wkt_esri:
                    try:
                        sref.spatial_ref = wkt_esri
                    except Exception:
                        pass
                    try:
                        sref.esri_pe_string = wkt_esri
                    except Exception:
                        pass
                try:
                    sref.long_name = "CRS definition"
                except Exception:
                    pass

                # Head variable
                v = ds.createVariable("head", "f4", ("z", "y", "x"), zlib=True, complevel=4, fill_value=FILL)
                v.long_name = "steady_state_head"
                v.units = str(getattr(cfg, "length_units", "") or "")
                v.coordinates = "z y x"
                v.grid_mapping = "spatial_ref"
                v.missing_value = FILL  # ArcGIS understands both _FillValue and missing_value

                data = head3d.astype("float32")
                mask = (~np.isfinite(data)) | (np.abs(data) >= BAD) | np.isclose(data, FILL)
                data = np.where(mask, FILL, data).astype("float32")
                v[:, :, :] = data

            out["netcdf"] = str(nc_path)
            print(f"Wrote netCDF head cube → {nc_path}")
        except Exception as e:
            print(f"[WARN] netCDF export skipped: {e}")
    else:
        print("[INFO] Skipping netCDF: requires uniform cell sizes and zero rotation. Mosaic dataset was created.")

    return out


# ----------------------------
# MODPATH7 export + figures
# ----------------------------
def process_and_export_modpath7_results(
    workspace: str | Path,
    workspace_gwf: str | Path,
    sim_name: str,
    gwf_model_name: str,
    hec_ras_crs=None,
    bed_elevation: float | None = None,
    ncol: int | None = None,
    nrow: int | None = None,
    z: float | None = None,
    nlay: int | None = None,
    river_cells: list[tuple[int, int, int, float]] | None = None,
    gwf=None,
    xorigin_value: float | None = None,
    yorigin_value: float | None = None,
    output_folder: str | Path | None = None,
    direction: str = "Forward",
    endpoints_file: str | Path | None = None,
    pathline_file: str | Path | None = None,
    *,
    projection_file: str | Path | None = None,
    export_csv: bool = True,
    export_shp: bool = True,
    export_shp_3d: bool = True,
    export_shp_wgs84: bool = True,
    export_kml: bool = True,
    export_kmz: bool = True,
    export_gpkg: bool = True,
    export_results_txt: bool = True,
    include_pngs_in_return: bool = True,
    export_pngs: bool = True,
    plots_dpi: int = 300,
    plots_show: bool = False,
    budget_term: str = "CHD",             # e.g., "FLOW-JA-FACE", "CHD", "RIV", etc.
    budget_kstpkper: tuple[int, int] | None = None,  # e.g., (kstp, kper). If None, use last.
) -> dict:
    """
    Export full MODPATH7 results (tables, vectors, figures) and write
    publication-ready pathline statistics to a .txt file.

    **Revisions:**
      - Adds a Zone-Budget-based hyporheic throughflow calculation.
      - Corrects particle-level hydraulic gradient to Δh / straight-line (3D) distance.
    """
    # --- helper: safe save ---
    def _maybe_save(fig, outpath: Path):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=plots_dpi, bbox_inches="tight")
        if plots_show:
            plt.show()
        plt.close(fig)
        return str(outpath)

    # --- helper: publication-ready block text builder ---
    def _block(lines: list[str], title: str, series: pd.Series, units: str = ""):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return
        def fmt(v: float) -> str:
            av = abs(v)
            if (av > 1e5) or (0 < av < 1e-3):
                f = f"{v:.3e}"
            else:
                f = f"{v:,.3f}"
            return f + (f" {units}" if units else "")
        lines.append(f"{title}")
        lines.append(f"  count: {len(s)}")
        lines.append(f"  mean: {fmt(s.mean())}")
        lines.append(f"  median: {fmt(s.median())}")
        lines.append(f"  min: {fmt(s.min())}")
        lines.append(f"  max: {fmt(s.max())}")
        if len(s) >= 10:
            lines.append(f"  p10: {fmt(s.quantile(0.10))}")
            lines.append(f"  p90: {fmt(s.quantile(0.90))}")
        lines.append("")

    # --- helper: pick first existing col name (makes stats/plots robust) ---
    def _col(df: pd.DataFrame, *candidates: str) -> pd.Series:
        for name in candidates:
            if name in df.columns:
                return df[name]
        raise KeyError(
            f"Expected one of columns {candidates} but DataFrame has: {list(df.columns)}"
        )

    from pyproj import CRS as _CRS
    workspace = Path(workspace)
    workspace_gwf = Path(workspace_gwf)
    output_folder = Path(output_folder) if output_folder is not None else Path(".")
    os.makedirs(output_folder, exist_ok=True)

    # CRS resolution (kept)
    if projection_file is not None and Path(projection_file).exists():
        wkt = Path(projection_file).read_text().strip()
        try:
            hec_crs = _CRS.from_wkt(wkt)
        except Exception:
            hec_crs = _CRS.from_string(wkt)
    elif hec_ras_crs is not None:
        hec_crs = _CRS.from_user_input(hec_ras_crs)
    else:
        raise ValueError("Provide either `projection_file` (.prj) or `hec_ras_crs`.")

    # Open GWF (if not passed)
    if gwf is None:
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(workspace_gwf))
        gwf = sim.get_model(gwf_model_name)

    dis = gwf.get_package("DIS")
    idomain = dis.idomain.array

    # robust MFScalar → float
    def _f(obj, default=0.0):
        return coerce_to_float(obj, default)

    xorigin = _f(getattr(dis, "xorigin", xorigin_value if xorigin_value is not None else 0.0),
                 default=(xorigin_value if xorigin_value is not None else 0.0))
    yorigin = _f(getattr(dis, "yorigin", yorigin_value if yorigin_value is not None else 0.0),
                 default=(yorigin_value if yorigin_value is not None else 0.0))

    delr = np.asarray(dis.delr.array, dtype=float)
    delc = np.asarray(dis.delc.array, dtype=float)

    nlay_m, nrow_m, ncol_m = gwf.modelgrid.shape
    nlay = int(nlay if nlay is not None else nlay_m)
    nrow = int(nrow if nrow is not None else nrow_m)
    ncol = int(ncol if ncol is not None else ncol_m)
    default_z_cell_size = float(z) if z is not None else 0.5

    # HEADS (kept)
    hds = flopy.utils.HeadFile(workspace_gwf / f"{gwf_model_name}.hds")
    head_array = hds.get_data()
    if head_array.ndim == 4:
        head_array = head_array[-1]

    # locate MP7 files
    def _first_existing(candidates: list[Path]) -> Path:
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    if pathline_file is None:
        pathline_file = _first_existing([
            workspace / f"{sim_name}_mp_{direction.lower()}.mppth",
            workspace / f"{sim_name}_{direction.lower()}.mppth",
            workspace / f"{sim_name}.mppth",
        ])
    else:
        pathline_file = Path(pathline_file)
    if endpoints_file is None:
        endpoints_file = _first_existing([
            workspace / f"{sim_name}_mp_{direction.lower()}.mpend",
            workspace / f"{sim_name}_{direction.lower()}.mpend",
            workspace / f"{sim_name}.mpend",
        ])
    else:
        endpoints_file = Path(endpoints_file)

    ep_reader = flopy.utils.EndpointFile(str(endpoints_file))
    endpoints = ep_reader.get_alldata()

    if bed_elevation is None:
        raise ValueError("`bed_elevation` is required to filter endpoints.")

    # filter endpoints (kept)
    Z_TOL = 1.0e-6
    filtered_particles = [
        ep for ep in endpoints
        if (float(ep["z"]) >= float(bed_elevation) - Z_TOL)
        and (abs(float(ep["z"]) - float(ep["z0"])) > Z_TOL)
    ]
    # NOTE: We intentionally DO NOT require (x,y) to change; vertical-only paths are valid.
    if len(filtered_particles) == 0:
        raise RuntimeError("No particles left after endpoint filtering.")
    filtered_particle_ids = [int(ep["particleid"]) for ep in filtered_particles]

    # read pathlines for filtered ids
    pl_reader = flopy.utils.PathlineFile(str(pathline_file))
    cols_pl = ['particleid', 'particlegroup', 'sequencenumber', 'particleidloc', 'time',
               'x', 'y', 'z', 'k', 'node', 'xloc', 'yloc', 'zloc', 'stressperiod', 'timestep']
    pathrecs: list = []
    for pid in filtered_particle_ids:
        pathrecs.extend(pl_reader.get_data(partid=pid))
    df_pl = pd.DataFrame.from_records(pathrecs, columns=cols_pl)
    if df_pl.empty:
        raise RuntimeError("No pathline records loaded for filtered particles.")

    # keep "long" pathlines (kept)
    mean_cell = float((np.mean(delr) + np.mean(delc)) / 2.0)
    thr_feet = 3.0 * mean_cell
    lengths_2d = (
        df_pl.sort_values(["particleid", "time"])
            .groupby("particleid")
            .apply(lambda g: np.sqrt(np.diff(g["x"])**2 + np.diff(g["y"])**2).sum())
    )
    keep_ids = lengths_2d[lengths_2d >= thr_feet].index
    df_long = df_pl[df_pl["particleid"].isin(keep_ids)].copy()
    if df_long.empty:
        # Short/shallow domains (e.g. a sub-km reach with a wide floodplain) can have every
        # hyporheic path shorter than the 3x-cell horizontal filter — those paths are still valid,
        # so keep them rather than abort a solved model.
        kept = lengths_2d[lengths_2d > 1e-6].index
        df_long = df_pl[df_pl["particleid"].isin(kept)].copy()
        if df_long.empty:
            df_long = df_pl.copy()
        print(f"[WARN] No pathlines exceeded the {thr_feet:.1f}-unit length filter; keeping "
              f"{df_long['particleid'].nunique()} shorter path(s) (short/shallow domain).", flush=True)

    # model→world transform (kept)
    angrot = float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0)
    theta = np.deg2rad(angrot)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_local = df_long["x"].to_numpy(dtype=float)
    y_local_top = df_long["y"].to_numpy(dtype=float)
    total_y = float(np.sum(delc))
    y_local = total_y - y_local_top
    df_long["x_abs"] = xorigin + (x_local * cos_t - y_local * sin_t)
    df_long["y_abs"] = yorigin + (x_local * sin_t + y_local * cos_t)

    # index cells (row/col/layer) (kept)
    x_edges = np.concatenate(([0.0], np.cumsum(delr)))
    y_edges = np.concatenate(([0.0], np.cumsum(delc)))
    cols_idx = np.searchsorted(x_edges, df_long["x"].to_numpy(dtype=float), side="right") - 1
    rows_idx_td = np.searchsorted(y_edges, total_y - df_long["y"].to_numpy(dtype=float), side="right") - 1

    ncol_mg = gwf.modelgrid.ncol
    nrow_mg = gwf.modelgrid.nrow
    oob = (cols_idx < 0) | (cols_idx >= ncol_mg) | (rows_idx_td < 0) | (rows_idx_td >= nrow_mg)
    cols_idx = cols_idx.astype(float); cols_idx[oob] = np.nan
    rows_idx = rows_idx_td.astype(float); rows_idx[oob] = np.nan

    top = np.asarray(gwf.modelgrid.top)
    botm = np.asarray(gwf.modelgrid.botm)
    nlay_mg = gwf.modelgrid.nlay
    layers = np.full_like(cols_idx, np.nan, dtype=float)
    valid = (~oob) & (~np.isnan(df_long["z"].to_numpy(dtype=float)))
    valid_idx = np.flatnonzero(valid)
    CHUNK = 300_000
    for st in range(0, valid_idx.size, CHUNK):
        sel = valid_idx[st:st + CHUNK]
        jj = rows_idx[sel].astype(int)
        ii = cols_idx[sel].astype(int)
        zz = df_long["z"].to_numpy(dtype=float)[sel]
        bots = botm[:, jj, ii]
        tops = np.empty_like(bots)
        tops[0] = top[jj, ii]
        if nlay_mg > 1:
            tops[1:] = bots[:-1]
        in_layer = (zz[None, :] >= bots) & (zz[None, :] < tops)
        has = in_layer.any(axis=0)
        kk = np.argmax(in_layer, axis=0).astype(float)
        kk[~has] = np.nan
        layers[sel] = kk
    df_long["layer"] = layers
    df_long["row"] = rows_idx
    df_long["col"] = cols_idx

    # Hyporheic space presence & per-cell volumes (kept)
    binary_presence = np.zeros((nlay, nrow, ncol), dtype=int)
    for k, j, i in zip(df_long["layer"].to_numpy(), df_long["row"].to_numpy(), df_long["col"].to_numpy()):
        if (not np.isnan(k)) and (not np.isnan(j)) and (not np.isnan(i)):
            ki, ji, ii = int(k), int(j), int(i)
            if 0 <= ki < nlay and 0 <= ji < nrow and 0 <= ii < ncol:
                binary_presence[ki, ji, ii] = 1

    delr_mean = float(np.mean(delr))
    delc_mean = float(np.mean(delc))
    cell_volumes = np.zeros((nlay, nrow, ncol), dtype=float)

    river_stage_array = np.zeros((nrow, ncol), dtype=float)
    if river_cells:
        for (k, j, i, stage) in river_cells:
            if k == 0 and 0 <= j < nrow and 0 <= i < ncol:
                river_stage_array[j, i] = float(stage)
    top_arr = gwf.modelgrid.top
    bot0 = gwf.modelgrid.botm[0, :, :]
    use_river = river_stage_array > 0.0
    thickness_top = np.where(use_river, np.maximum(river_stage_array - bot0, 0.0), 0.0)
    cell_volumes[0, :, :] = delr_mean * delc_mean * thickness_top
    for kk in range(1, nlay):
        cell_volumes[kk, :, :] = delr_mean * delc_mean * (float(z) if z is not None else default_z_cell_size)

    total_volume_per_cell = np.nansum(cell_volumes, axis=0)
    hyporheic_volumes = cell_volumes * binary_presence

    # segment-wise deltas (kept)
    df = df_long.sort_values(["particleid", "time"]).reset_index(drop=True)
    df["dx"] = df.groupby("particleid")["x"].diff()
    df["dy"] = df.groupby("particleid")["y"].diff()
    df["dz"] = df.groupby("particleid")["z"].diff()
    df["dt"] = df.groupby("particleid")["time"].diff()
    df["segment_length"] = np.sqrt(df["dx"]**2 + df["dy"]**2 + df["dz"]**2)
    df["particle_velocity"] = np.where(df["dt"] > 0, df["segment_length"] / df["dt"], np.nan)

    # per-segment hyporheic volume + head/gradient (kept for per-cell summaries)
    vol_vals = []
    for k, j, i in zip(df["layer"].to_numpy(), df["row"].to_numpy(), df["col"].to_numpy()):
        if (not np.isnan(k)) and (not np.isnan(j)) and (not np.isnan(i)):
            vol_vals.append(hyporheic_volumes[int(k), int(j), int(i)])
        else:
            vol_vals.append(0.0)
    df["hyporheic_volume"] = np.asarray(vol_vals, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["flow_rate"] = df["hyporheic_volume"] / df["dt"]
    df.loc[(df["dt"] <= 0) | (~np.isfinite(df["flow_rate"])), "flow_rate"] = np.nan

    def _get_head(row):
        try:
            return head_array[int(row["layer"]), int(row["row"]), int(row["col"])]
        except Exception:
            return np.nan

    df["head"] = df.apply(_get_head, axis=1)
    df["head_start"] = df.groupby("particleid")["head"].shift(1)
    df["head_end"] = df["head"]
    # NOTE: keep segment-based gradient for cell-level aggregation (unchanged)
    df["hydraulic_gradient"] = np.where(
        df["segment_length"] > 0,
        (df["head_start"] - df["head_end"]) / df["segment_length"],
        np.nan
    )
    df["cumulative_segment_length"] = df.groupby("particleid")["segment_length"].cumsum()

    # condense per cell (kept)
    df_clean = df.dropna(subset=["row", "col", "particleid"]).copy()
    df_clean["row_int"] = df_clean["row"].astype(int)
    df_clean["col_int"] = df_clean["col"].astype(int)

    agg_dict = {
        "z": "min",
        "hyporheic_volume": "sum",
        "dt": "sum",
        "time": "max",
        "segment_length": "sum",
        "cumulative_segment_length": "max",
        "particle_velocity": "mean",
        "flow_rate": "sum",
        "hydraulic_gradient": "mean",
        "head": "mean",
        "head_start": "mean",
        "head_end": "mean",
    }

    df_prc = (
        df_clean.groupby(["particleid", "row_int", "col_int", "time"], as_index=False)
            .agg(agg_dict)
            .rename(columns={
                "row_int": "row",
                "col_int": "col",
                "z": "z_elevation",
                "hyporheic_volume": "hyporheic_volume_cubic_ft",
                "dt": "cell_residence_time_days",
                "time": "total_residence_time_days",
                "segment_length": "length_in_cell_ft",
                "cumulative_segment_length": "total_length_ft",
                "particle_velocity": "particle_velocity_ft_per_day",
                "flow_rate": "flow_rate_cubic_ft_per_day",
                "head": "hydraulic_head",
                "head_start": "starting_hydraulic_head",
                "head_end": "ending_hydraulic_head",
            })
    )

    def _cell_total_vol(row):
        r, c = int(row["row"]), int(row["col"])
        if 0 <= r < total_volume_per_cell.shape[0] and 0 <= c < total_volume_per_cell.shape[1]:
            return float(total_volume_per_cell[r, c])
        return np.nan

    df_prc["total_volume_cubic_ft"] = df_prc.apply(_cell_total_vol, axis=1)

    # assign world coords per cell center (for points) (kept)
    x_edges_abs = xorigin + np.concatenate(([0.0], np.cumsum(delr)))
    y_edges_abs = yorigin + np.concatenate(([0.0], np.cumsum(delc)))
    x_centers_model = 0.5 * (x_edges_abs[:-1] + x_edges_abs[1:])
    y_centers_model = 0.5 * (y_edges_abs[:-1] + y_edges_abs[1:])

    def _rc_to_world(j: int, i: int):
        xl = (x_centers_model[i] - xorigin)
        yl = (y_centers_model[j] - yorigin)
        return (
            xorigin + (xl * cos_t - yl * sin_t),
            yorigin + (xl * sin_t + yl * cos_t),
        )

    xcoords, ycoords = [], []
    for r, c in zip(df_prc["row"].astype(int).to_numpy(), df_prc["col"].astype(int).to_numpy()):
        if 0 <= r < nrow and 0 <= c < ncol:
            xw, yw = _rc_to_world(r, c)
        else:
            xw, yw = (np.nan, np.nan)
        xcoords.append(xw); ycoords.append(yw)
    df_prc["x_coord"] = xcoords
    df_prc["y_coord"] = ycoords
    df_prc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prc = df_prc.dropna(subset=["x_coord", "y_coord", "row", "col", "particleid"])

    # ---------- PARTICLE-LEVEL (revised head + gradient) ----------
    # True start/end positions and heads for each particle
    #g_sorted = df_long.sort_values(["particleid", "time"])
    g_sorted = df.sort_values(["particleid", "time"])
    first_pts = g_sorted.groupby("particleid").first().reset_index()
    last_pts  = g_sorted.groupby("particleid").last().reset_index()

    # Straight-line distances (3D and plan-view)
    dxx = last_pts["x_abs"].to_numpy(float) - first_pts["x_abs"].to_numpy(float)
    dyy = last_pts["y_abs"].to_numpy(float) - first_pts["y_abs"].to_numpy(float)
    dzz = last_pts["z"].to_numpy(float)     - first_pts["z"].to_numpy(float)
    sl_2d = np.sqrt(dxx**2 + dyy**2)
    sl_3d = np.sqrt(dxx**2 + dyy**2 + dzz**2)

    straight_df = pd.DataFrame({
        "particleid": first_pts["particleid"].astype(int).to_numpy(),
        "straight_line_length_2d_ft": sl_2d,
        "straight_line_length_3d_ft": sl_3d,
        "x_start": first_pts["x_abs"].to_numpy(float),
        "y_start": first_pts["y_abs"].to_numpy(float),
        "z_start": first_pts["z"].to_numpy(float),
        "x_end":   last_pts["x_abs"].to_numpy(float),
        "y_end":   last_pts["y_abs"].to_numpy(float),
        "z_end":   last_pts["z"].to_numpy(float),
    })

    # Starting/ending heads from actual first/last records
    # heads_first = g_sorted.groupby("particleid")["head"].first().reset_index(name="starting_hydraulic_head")
    # heads_last  = g_sorted.groupby("particleid")["head"].last().reset_index(name="ending_hydraulic_head")
    heads_first = (
        g_sorted.groupby("particleid", as_index=False)["head"]
                .first()
                .rename(columns={"head": "starting_hydraulic_head"})
    )
    heads_last = (
        g_sorted.groupby("particleid", as_index=False)["head"]
                .last()
                .rename(columns={"head": "ending_hydraulic_head"})
    )

    # Per-particle summaries kept from df_prc
    total_volume_available_along_path = df_prc.groupby("particleid")["total_volume_cubic_ft"].sum().reset_index()
    total_hyporheic_volume_per_particle = df_prc.groupby("particleid")["hyporheic_volume_cubic_ft"].sum().reset_index()
    total_residence_time_per_particle = df_prc.groupby("particleid")["total_residence_time_days"].max().reset_index()
    total_length_per_particle = df_prc.groupby("particleid")["total_length_ft"].max().reset_index()
    average_particle_velocity_per_particle = df_prc.groupby("particleid")["particle_velocity_ft_per_day"].mean().reset_index()
    average_flow_rate_per_particle = df_prc.groupby("particleid")["flow_rate_cubic_ft_per_day"].mean().reset_index()

    # Build particle summary (merge new heads/lengths)
    df_particle_summary = (
        straight_df
        .merge(total_volume_available_along_path, on="particleid", how="left")
        .merge(total_hyporheic_volume_per_particle, on="particleid", how="left")
        .merge(total_residence_time_per_particle, on="particleid", how="left")
        .merge(total_length_per_particle, on="particleid", how="left")
        .merge(average_particle_velocity_per_particle, on="particleid", how="left")
        .merge(average_flow_rate_per_particle, on="particleid", how="left")
        .merge(heads_first, on="particleid", how="left")
        .merge(heads_last,  on="particleid", how="left")
    )

    # Corrected hydraulic gradient: Δh / straight-line 3D distance
    with np.errstate(divide="ignore", invalid="ignore"):
        df_particle_summary["hydraulic_gradient"] = (
            (df_particle_summary["starting_hydraulic_head"] - df_particle_summary["ending_hydraulic_head"])
            / df_particle_summary["straight_line_length_3d_ft"]
        )

    df_particle_summary["x_cell_size_ft"] = delr_mean
    df_particle_summary["y_cell_size_ft"] = delc_mean
    df_particle_summary["z_cell_size_ft"] = float(z) if z is not None else default_z_cell_size

    # ---------------- I/O artifacts scaffold ----------------
    artifacts = {
        "results": None,
        "csv": None,
        "points_csv": None,
        "csv_summary": None,
        "points_shp": None,
        "lines_shp": None,           # 2D
        "lines_shp_3d": None,        # 3D shapefile (preferred for hyporheic/filtered)
        "lines_fc_3d": None,         # reserved
        "points_shp_wgs84": None,
        "lines_shp_wgs84": None,
        "points_kml": None,
        "lines_kml": None,
        "points_kmz": None,
        "lines_kmz": None,
        "points_gpkg": None,
        "lines_gpkg": None,
        "plots": [],
        "plot_paths": {},
        "zone_budget": None,
    }

    # ---------------- CSV outputs ----------------
    if export_csv:
        csv_path_pathlines_filtered = output_folder / f"{direction}_pathlines_filtered.csv"
        df_long.to_csv(csv_path_pathlines_filtered, index=False)
        artifacts["csv"] = str(csv_path_pathlines_filtered)

        csv_points_table = output_folder / f"{direction}_points_table.csv"
        df_prc.to_csv(csv_points_table, index=False)
        artifacts["points_csv"] = str(csv_points_table)

        csv_summary = output_folder / f"{direction}_particle_summary_table.csv"
        df_particle_summary.to_csv(csv_summary, index=False)
        artifacts["csv_summary"] = str(csv_summary)

    # ---------------- Spatial exports (kept) ----------------
    # (points + 2D/3D lines) — identical to your previous version ...
    if any([export_shp, export_shp_wgs84, export_kml, export_kmz, export_gpkg, export_shp_3d]):
        # points (cell centers)
        custom_map_points = {
            "row": "row", "col": "col",
            "z_elevation": "zelev",
            "hyporheic_volume_cubic_ft": "hypvol_ft",
            "cell_residence_time_days": "cel_res_t",
            "total_residence_time_days": "res_time",
            "length_in_cell_ft": "Len_ft",
            "total_length_ft": "tot_len_ft",
            "particle_velocity_ft_per_day": "vel_ftday",
            "flow_rate_cubic_ft_per_day": "flow_ftday",
            "hydraulic_gradient": "hyd_grad",
            "hydraulic_head": "hyd_head",
            "starting_hydraulic_head": "head_strt",
            "ending_hydraulic_head": "head_end",
            "x_coord": "xcoord",
            "y_coord": "ycoord",
            "total_volume_cubic_ft": "tot_vol_ft",
            "particleid": "particleid",
        }
        df_prc_short = df_prc[list(custom_map_points.keys())].rename(columns=custom_map_points)
        points_geom = [Point(xy) for xy in zip(df_prc_short["xcoord"], df_prc_short["ycoord"])]
        points_gdf = gpd.GeoDataFrame(df_prc_short, geometry=points_geom, crs=hec_crs)

        # lines 2D
        line_rows_2d, line_geoms_2d = [], []
        for pid, g in df_long.sort_values(["particleid", "time"]).groupby("particleid", sort=False):
            pts = list(zip(g["x_abs"].to_numpy(), g["y_abs"].to_numpy()))
            if len(pts) > 1:
                line_geoms_2d.append(LineString(pts))
                line_rows_2d.append({"particleid": int(pid)})
        lines_gdf_2d = gpd.GeoDataFrame(line_rows_2d, geometry=line_geoms_2d, crs=hec_crs)

        # per-particle summary for lines (rename for clarity on attributes)
        df_particle_summary_for_lines = df_particle_summary.rename(columns={
            "z_elevation": "min_elevation_per_particle",
            "total_residence_time_days": "total_residence_time_per_particle",
            "total_length_ft": "total_length_per_particle",
            "particle_velocity_ft_per_day": "average_particle_velocity_per_particle",
            "flow_rate_cubic_ft_per_day": "average_flow_rate_per_particle",
        })
        summary_map = {
            "particleid": "particleid",
            "total_volume_cubic_ft": "tot_vol_ft",
            "hyporheic_volume_cubic_ft": "hypvol_ft",
            "starting_hydraulic_head": "head_strt",
            "ending_hydraulic_head": "head_end",
            "min_elevation_per_particle": "min_elev",
            "total_residence_time_per_particle": "res_time",
            "total_length_per_particle": "tot_len_ft",
            "average_particle_velocity_per_particle": "vel_ftday",
            "average_flow_rate_per_particle": "flow_ftday",
            "hydraulic_gradient": "hyd_grad",
            "x_cell_size_ft": "x_cell_ft",
            "y_cell_size_ft": "y_cell_ft",
            "z_cell_size_ft": "z_cell_ft",
        }
        df_particle_summary_short = df_particle_summary_for_lines.rename(columns=summary_map)

        lines_gdf_2d = lines_gdf_2d.merge(df_particle_summary_short, on="particleid", how="left")

        # 3D polylines
        line_rows_3d, line_geoms_3d = [], []
        for pid, g in df_long.sort_values(["particleid", "time"]).groupby("particleid", sort=False):
            xs = g["x_abs"].to_numpy(float)
            ys = g["y_abs"].to_numpy(float)
            zs = g["z"].to_numpy(float)
            valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
            if valid.sum() > 1:
                line_geoms_3d.append(LineString(list(zip(xs[valid], ys[valid], zs[valid]))))
                line_rows_3d.append({"particleid": int(pid)})
        lines_gdf_3d = gpd.GeoDataFrame(line_rows_3d, geometry=line_geoms_3d, crs=hec_crs)
        lines_gdf_3d = lines_gdf_3d.merge(df_particle_summary_short, on="particleid", how="left")

        # write formats
        if export_shp:
            p_shp = output_folder / f"{direction}_hyporheic_points_HECRAS_CRS.shp"
            l_shp = output_folder / f"{direction}_hyporheic_pathlines_2D_HECRAS_CRS.shp"
            points_gdf.to_file(p_shp, driver="ESRI Shapefile")
            lines_gdf_2d.to_file(l_shp, driver="ESRI Shapefile")
            artifacts["points_shp"] = str(p_shp)
            artifacts["lines_shp"] = str(l_shp)

        # 3D shapefile
        if export_shp_3d:
            try:
                shp3d = output_folder / f"{direction}_hyporheic_pathlines_3D_HECRAS_CRS.shp"
                lines_gdf_3d.to_file(shp3d, driver="ESRI Shapefile")
                artifacts["lines_shp_3d"] = str(shp3d)
            except Exception as e:
                print(f"[WARN] GeoPandas 3D shapefile export failed ({e}).")

        if export_shp_wgs84:
            points_wgs = points_gdf.to_crs(epsg=4326)
            lines_wgs = lines_gdf_2d.to_crs(epsg=4326)
            p_shp_wgs = output_folder / f"{direction}_hyporheic_points.shp"
            l_shp_wgs = output_folder / f"{direction}_hyporheic_pathlines.shp"
            points_wgs.to_file(p_shp_wgs, driver="ESRI Shapefile")
            lines_wgs.to_file(l_shp_wgs, driver="ESRI Shapefile")
            artifacts["points_shp_wgs84"] = str(p_shp_wgs)
            artifacts["lines_shp_wgs84"] = str(l_shp_wgs)

        if export_kml or export_kmz:
            points_wgs = points_gdf.to_crs(epsg=4326)
            lines_wgs = lines_gdf_2d.to_crs(epsg=4326)
            if export_kml:
                p_kml = output_folder / f"{direction}_hyporheic_points.kml"
                l_kml = output_folder / f"{direction}_hyporheic_pathlines.kml"
                points_wgs.to_file(p_kml, driver="KML")
                lines_wgs.to_file(l_kml, driver="KML")
                artifacts["points_kml"] = str(p_kml)
                artifacts["lines_kml"] = str(l_kml)
            if export_kmz:
                p_kmz = output_folder / f"{direction}_hyporheic_points.kmz"
                l_kmz = output_folder / f"{direction}_hyporheic_pathlines.kmz"
                with zipfile.ZipFile(p_kmz, "w", zipfile.ZIP_DEFLATED) as kmz:
                    kmz.write(output_folder / f"{direction}_hyporheic_points.kml",
                              f"{direction}_hyporheic_points.kml")
                with zipfile.ZipFile(l_kmz, "w", zipfile.ZIP_DEFLATED) as kmz:
                    kmz.write(output_folder / f"{direction}_hyporheic_pathlines.kml",
                              f"{direction}_hyporheic_pathlines.kml")
                artifacts["points_kmz"] = str(p_kmz)
                artifacts["lines_kmz"] = str(l_kmz)

        if export_gpkg:
            p_gpkg = output_folder / f"{direction}_hyporheic_points.gpkg"
            l_gpkg = output_folder / f"{direction}_hyporheic_pathlines_2D.gpkg"
            try:
                points_gdf.to_file(p_gpkg, layer="points", driver="GPKG")
                lines_gdf_2d.to_file(l_gpkg, layer="lines_2d", driver="GPKG")
                artifacts["points_gpkg"] = str(p_gpkg)
                artifacts["lines_gpkg"] = str(l_gpkg)
            except Exception:
                pass

    # ---------------- Publication‑ready stats TXT ----------------
    # First, compute Zone-Budget hyporheic throughflow and prepend that section.
    zone_lines: list[str] = []
    zone_report: dict[str, float | int | str] = {}

    from flopy.utils import CellBudgetFile

    try:
        # --- locate a budget file we can open ---
        candidates = [
            workspace_gwf / f"{gwf_model_name}.cbb",
            workspace_gwf / f"{gwf_model_name}.cbc",
            workspace_gwf / f"{gwf_model_name}.bud",
            workspace_gwf / "gwf_model.cbb",
            workspace_gwf / "gwf_model.cbc",
            workspace_gwf / "gwf_model.bud",
            *sorted(Path(workspace_gwf).glob("*.cbb")),
            *sorted(Path(workspace_gwf).glob("*.cbc")),
            *sorted(Path(workspace_gwf).glob("*.bud")),
        ]
        candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
        if not candidates:
            raise FileNotFoundError("No budget file (*.cbb|*.cbc|*.bud) found in GWF workspace.")

        # choose first viable file
        used_path = candidates[0]
        cbc = CellBudgetFile(str(used_path), precision="double")

        # choose timestep — use caller's choice if given, otherwise last record
        if budget_kstpkper is None:
            kkp = cbc.get_kstpkper() if hasattr(cbc, "get_kstpkper") else None
            budget_kstpkper_eff = kkp[-1] if kkp else None
        else:
            budget_kstpkper_eff = budget_kstpkper

        # pull full‑3D per‑cell net budget for the specified term (default FLOW‑JA‑FACE)
        def _get_flow3d():
            if budget_kstpkper_eff is None:
                a = cbc.get_data(text=budget_term, full3D=True)
                if not a:
                    raise RuntimeError(f"No records for '{budget_term}' in {used_path.name}")
                arr = a[-1]
            else:
                a = cbc.get_data(kstpkper=budget_kstpkper_eff, text=budget_term, full3D=True)
                if not a:
                    raise RuntimeError(
                        f"No records for '{budget_term}' at kstpkper={budget_kstpkper_eff} in {used_path.name}"
                    )
                arr = a[0]

            arr = np.asarray(arr)
            shp = gwf.modelgrid.shape  # (nlay, nrow, ncol)
            if arr.ndim == 3 and arr.shape == shp:
                return arr
            if arr.ndim == 1 and arr.size == gwf.modelgrid.nnodes:
                # some builds may return flattened ncells vector
                return arr.reshape(shp)
            # One more try: sometimes full3D returns a 3D array but reshaping is needed
            if arr.ndim == 3 and arr.size == int(np.prod(shp)):
                return arr.reshape(shp)
            raise RuntimeError(
                f"Unexpected shape for '{budget_term}' (got {arr.shape}, expected {shp} or (nnodes,))."
            )

        flow3d = _get_flow3d()

        # ---------- build "zones" from your filtered endpoints (route from the working example) ----------
        # Use robust cell-edge mapping (works with variable DELR/DELC), then infer layer from top/botm.
        x_edges = np.concatenate(([0.0], np.cumsum(delr)))
        y_edges = np.concatenate(([0.0], np.cumsum(delc)))
        total_y = float(np.sum(delc))
        nlay_mg, nrow_mg, ncol_mg = gwf.modelgrid.shape

        def _infer_kji_from_xyz(x_model: float, y_model_top_axis: float, z_model: float,
                                *, ztol: float = 1.0e-6) -> tuple[int, int, int] | None:
            # i/j from model-axis x (left→right) and y (top→bottom) with clamping
            i = int(np.clip(np.searchsorted(x_edges, x_model, side="right") - 1, 0, ncol_mg - 1))
            j = int(np.clip(np.searchsorted(y_edges, total_y - y_model_top_axis, side="right") - 1, 0, nrow_mg - 1))

            # layer from top/botm at (j,i)
            bots = botm[:, j, i]
            tops = np.empty_like(bots)
            tops[0] = top[j, i]
            if nlay_mg > 1:
                tops[1:] = botm[:-1, j, i]

            # Treat z == top as inside the top layer; clamp above/below column
            if z_model >= (tops[0] - ztol):
                return (0, j, i)
            if z_model <= (bots[-1] + ztol):
                return (nlay_mg - 1, j, i)

            in_layer = (z_model >= (bots - ztol)) & (z_model <= (tops + ztol))
            if in_layer.any():
                return (int(np.argmax(in_layer)), j, i)

            # Fallback: choose nearest layer center (very rare)
            centers = 0.5 * (tops + bots)
            return (int(np.argmin(np.abs(centers - z_model))), j, i)


        # Start/End "zones": unique cells at particle starts/ends (from filtered endpoints)
        start_cells: set[tuple[int, int, int]] = set()
        end_cells:   set[tuple[int, int, int]] = set()

        for ep in filtered_particles:
            s = _infer_kji_from_xyz(float(ep["x0"]), float(ep["y0"]), float(ep["z0"]))
            e = _infer_kji_from_xyz(float(ep["x"]),  float(ep["y"]),  float(ep["z"]))
            if s is not None:
                start_cells.add(s)
            if e is not None:
                end_cells.add(e)


        if not end_cells:
            try:
                # Use the already-opened PathlineFile (pl_reader). If not available, open it here.
                for pid in filtered_particle_ids:
                    g = pl_reader.get_data(partid=pid)
                    if not g:
                        continue
                    last = g[-1]
                    e = _infer_kji_from_xyz(float(last["x"]), float(last["y"]), float(last["z"]))
                    if e is not None:
                        end_cells.add(e)
            except Exception:
                pass


        if (not start_cells) and (not end_cells):
            raise RuntimeError("Could not infer any start/end cells from endpoints for zone budget.")

        # ---------- Sum per‑cell net budgets in each zone ----------
        def _vals(cells: set[tuple[int, int, int]]) -> np.ndarray:
            if not cells:
                return np.array([], dtype=float)
            return np.array([float(flow3d[k, j, i]) for (k, j, i) in cells], dtype=float)

        start_vals = _vals(start_cells)
        end_vals   = _vals(end_cells)

        total_start = float(np.nansum(start_vals))                # signed into‑cell
        total_end   = float(np.nansum(end_vals))                  # signed into‑cell
        abs_start   = float(np.nansum(np.abs(start_vals)))        # magnitude
        abs_end     = float(np.nansum(np.abs(end_vals)))          # magnitude

        # Throughflow magnitude (average of magnitudes of entry/exit sums).
        # This mirrors your earlier idea but follows the "route" of sampling a single full3D budget slice.
        #through_avg_mag = 0.5 * (abs_start + abs_end)
        through_avg_mag = max(abs_start, abs_end) #corrected because the averaging doesn't seem to work
        net_balance     = total_start + total_end                 # should be ~0 for steady runs

        # Assemble report + pretty lines
        zone_report = {
            "budget_path": str(used_path),
            "budget_term": str(budget_term),
            "kstpkper": tuple(budget_kstpkper_eff) if budget_kstpkper_eff is not None else None,
            "flow3d_shape": tuple(int(x) for x in np.asarray(flow3d).shape),
            "n_start_cells": int(len(start_cells)),
            "n_end_cells": int(len(end_cells)),
            "total_start_flow_ft3_d": total_start,
            "total_end_flow_ft3_d": total_end,
            "abs_start_ft3_d": abs_start,
            "abs_end_ft3_d": abs_end,
            "throughflow_avg_abs_ft3_d": through_avg_mag,
            "net_flow_balance_ft3_d": net_balance,
        }

        zone_lines.append("Zone Budget — Hyporheic Throughflow (ft³/d)")
        zone_lines.append(f"  budget file: {Path(used_path).name}")
        zone_lines.append(f"  term: {budget_term}")
        zone_lines.append(
            f"  kstpkper: {budget_kstpkper_eff if budget_kstpkper_eff is not None else 'last record'}"
        )
        zone_lines.append(f"  start cells: {len(start_cells)}")
        zone_lines.append(f"  end cells: {len(end_cells)}")
        zone_lines.append(f"  total start flow (signed): {total_start:,.3f} ft³/d")
        zone_lines.append(f"  total end flow (signed):   {total_end:,.3f} ft³/d")
        zone_lines.append(f"  |start|: {abs_start:,.3f} ft³/d")
        zone_lines.append(f"  |end|:   {abs_end:,.3f} ft³/d")
        zone_lines.append(f"  hyporheic throughflow (avg(|start|,|end|)): {through_avg_mag:,.3f} ft³/d")
        zone_lines.append(f"  net flow balance (start+end): {net_balance:,.3f} ft³/d")
        zone_lines.append("")

    except Exception as e:
        zone_lines.append("Zone Budget — Hyporheic Throughflow (ft³/d)")
        zone_lines.append(f"  (no results — {e})")
        zone_lines.append("")
        zone_report = {"error": str(e)}

    artifacts["zone_budget"] = zone_report


    # Other publication stats (kept, but gradients now reflect corrected particle-level values)
    spans = df_long.groupby("particleid").apply(
        lambda g: pd.Series({
            "span_x": g["x_abs"].max() - g["x_abs"].min(),
            "span_y": g["y_abs"].max() - g["y_abs"].min(),
            "span_xy_diag": np.hypot(g["x_abs"].max() - g["x_abs"].min(),
                                     g["y_abs"].max() - g["y_abs"].min()),
            "vertical_excursion": g["z"].max() - g["z"].min(),
        })
    ).reset_index()

    df_pub = df_particle_summary.merge(spans, on="particleid", how="left")

    stats_lines: list[str] = []
    # Prepend zone-budget section
    stats_lines.extend(zone_lines)

    # 3D path length (ft)
    _block(stats_lines, "Particle Path — 3D Length (ft)",
           _col(df_pub, "total_length_per_particle", "total_length_ft"), "ft")
    # Plan‑view width (ft)
    _block(stats_lines, "Plan‑View Excursion (ft)", df_pub["span_xy_diag"], "ft")
    # Straight‑line distance (3D) — informative
    _block(stats_lines, "Straight‑Line Distance (3D, ft)", df_pub["straight_line_length_3d_ft"], "ft")
    # Vertical excursion (ft)
    _block(stats_lines, "Vertical Excursion (ft)", df_pub["vertical_excursion"], "ft")
    # Residence time (days)
    _block(stats_lines, "Residence Time (days)",
           _col(df_pub, "total_residence_time_per_particle", "total_residence_time_days"), "days")
    # Velocity (ft/day)
    _block(stats_lines, "Average Particle Velocity (ft/day)",
           _col(df_pub, "average_particle_velocity_per_particle", "particle_velocity_ft_per_day"), "ft/day")
    # Hydraulic gradient (–) — corrected Δh / straight-line length
    _block(stats_lines, "Hydraulic Gradient (–)", df_pub["hydraulic_gradient"], "")
    # Hyporheic volume (ft³)
    _block(stats_lines, "Hyporheic Volume per Particle (ft³)", df_pub["hyporheic_volume_cubic_ft"], "ft³")

    if export_results_txt:
        results_txt = output_folder / f"{direction}_pathline_stats.txt"
        results_txt.write_text("\n".join(stats_lines), encoding="utf-8")
        artifacts["results"] = str(results_txt)

    # ---------------- PNGs — Distributions (kept) ----------------
    if export_pngs:
        # (1) Length/Width/Depth combined figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # length 3D
        s_len = pd.to_numeric(_col(df_pub, "total_length_per_particle", "total_length_ft"),
                              errors="coerce").dropna()
        sns.histplot(s_len, kde=True, ax=axes[0])
        axes[0].set_xlabel("3D Path Length (ft)"); axes[0].set_ylabel("Count")
        axes[0].set_title("Path Length")
        # width (plan‑view excursion as diag)
        s_w = pd.to_numeric(df_pub["span_xy_diag"], errors="coerce").dropna()
        sns.histplot(s_w, kde=True, ax=axes[1])
        axes[1].set_xlabel("Plan‑View Excursion (ft)"); axes[1].set_ylabel("Count")
        axes[1].set_title("Plan‑View Width")
        # depth (vertical excursion)
        s_d = pd.to_numeric(df_pub["vertical_excursion"], errors="coerce").dropna()
        sns.histplot(s_d, kde=True, ax=axes[2])
        axes[2].set_xlabel("Vertical Excursion (ft)"); axes[2].set_ylabel("Count")
        axes[2].set_title("Vertical Range")
        plt.tight_layout()
        p_len_w_d = output_folder / f"{direction}_pathline_length_width_depth_distributions.png"
        artifacts["plot_paths"]["length_width_depth"] = _maybe_save(fig, p_len_w_d)
        artifacts["plots"].append(str(p_len_w_d))

        # (2) Residence time distribution
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        rt = pd.to_numeric(_col(df_pub, "total_residence_time_per_particle", "total_residence_time_days"),
                           errors="coerce").dropna()
        if not rt.empty:
            sns.histplot(rt, kde=True, ax=ax)
        ax.set_xlabel("Residence Time (days)"); ax.set_ylabel("Count")
        ax.set_title("Residence Time Distribution")
        plt.tight_layout()
        p_rt = output_folder / f"{direction}_residence_time_distribution.png"
        artifacts["plot_paths"]["residence_time"] = _maybe_save(fig, p_rt)
        artifacts["plots"].append(str(p_rt))

        # (3) Velocity distribution
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        vv = pd.to_numeric(_col(df_pub, "average_particle_velocity_per_particle", "particle_velocity_ft_per_day"),
                           errors="coerce").dropna()
        if not vv.empty:
            sns.histplot(vv, kde=True, ax=ax)
        ax.set_xlabel("Average Velocity (ft/day)"); ax.set_ylabel("Count")
        ax.set_title("Particle Velocity Distribution")
        plt.tight_layout()
        p_vv = output_folder / f"{direction}_velocity_distribution.png"
        artifacts["plot_paths"]["velocity"] = _maybe_save(fig, p_vv)
        artifacts["plots"].append(str(p_vv))
    else:
        if include_pngs_in_return:
            artifacts["plots"].extend([
                str(output_folder / f"{direction}_pathline_length_width_depth_distributions.png"),
                str(output_folder / f"{direction}_residence_time_distribution.png"),
                str(output_folder / f"{direction}_velocity_distribution.png"),
            ])

    return artifacts


def export_full_modpath7_pathlines_3d_shp(
    *,
    workspace: str | Path,
    workspace_gwf: str | Path,
    sim_name: str,
    gwf_model_name: str,
    hec_ras_crs=None,
    projection_file: str | Path | None = None,
    output_folder: str | Path | None = None,
    direction: str = "Forward",
) -> str | None:
    """
    Build true-3D pathlines (POLYLINE Z) for *all* MODPATH7 particles (no hyporheic filtering).
    Writes a shapefile in `output_folder` and returns its path, or None if no features were created.

    Parameters
    ----------
    workspace : MODPATH7 workspace folder (where *.mppth/*.mpend live)
    workspace_gwf : MODFLOW 6 GWF workspace (to read DIS/grid + origins/rotation)
    sim_name : Simulation name (used to locate default MP7 files)
    gwf_model_name : GWF model name (used to open the head/grid model)
    hec_ras_crs : Any CRS value GeoPandas/pyproj accepts (EPSG, WKT, etc.)
    projection_file : Optional .prj file (used if hec_ras_crs is None)
    output_folder : Destination folder (defaults to `workspace` if None)
    direction : "Forward" or "Backward" (file name prefixes only)

    Returns
    -------
    str | None : Path to the created shapefile (POLYLINE Z) or None.
    """
    import os
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import LineString
    from pathlib import Path
    from pyproj import CRS as _CRS
    import flopy

    def _first_existing(paths: list[Path]) -> Path:
        for p in paths:
            if p.exists():
                return p
        # return the first candidate even if non-existent (caller will fail/readably)
        return paths[0]

    # ---- Resolve IO paths
    workspace = Path(workspace)
    workspace_gwf = Path(workspace_gwf)
    output_folder = Path(output_folder) if output_folder else workspace
    output_folder.mkdir(parents=True, exist_ok=True)

    pl_file = _first_existing([
        workspace / f"{sim_name}_mp_{direction.lower()}.mppth",
        workspace / f"{sim_name}_{direction.lower()}.mppth",
        workspace / f"{sim_name}.mppth",
    ])
    ep_file = _first_existing([
        workspace / f"{sim_name}_mp_{direction.lower()}.mpend",
        workspace / f"{sim_name}_{direction.lower()}.mpend",
        workspace / f"{sim_name}.mpend",
    ])

    # ---- Load GWF model to get DIS + grid -> transform to world XY
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(workspace_gwf))
    gwf = sim.get_model(gwf_model_name)
    dis = gwf.get_package("DIS")

    # robust coercion (handles MFScalar / numpy scalars)
    xorigin = coerce_to_float(getattr(dis, "xorigin", getattr(gwf.modelgrid, "xoffset", 0.0)),
                              default=getattr(gwf.modelgrid, "xoffset", 0.0))
    yorigin = coerce_to_float(getattr(dis, "yorigin", getattr(gwf.modelgrid, "yoffset", 0.0)),
                              default=getattr(gwf.modelgrid, "yoffset", 0.0))

    delr = np.asarray(dis.delr.array, dtype=float)
    delc = np.asarray(dis.delc.array, dtype=float)
    total_y = float(np.sum(delc))
    angrot = float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0)
    theta = np.deg2rad(angrot)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # ---- Build CRS
    hec_crs = None
    if hec_ras_crs is not None:
        hec_crs = _CRS.from_user_input(hec_ras_crs)
    elif projection_file is not None:
        prj = Path(projection_file)
        if prj.exists():
            wkt = prj.read_text().strip()
            try:
                hec_crs = _CRS.from_wkt(wkt)
            except Exception:
                hec_crs = _CRS.from_string(wkt)

    # ---- Read all particle IDs, then pull all pathline records
    ep = flopy.utils.EndpointFile(str(ep_file))
    endpoints = ep.get_alldata()
    if endpoints is None or len(endpoints) == 0:
        print(f"[WARN] No endpoints found in: {ep_file}")
        return None
    all_pids = [int(rec["particleid"]) for rec in endpoints]

    pl = flopy.utils.PathlineFile(str(pl_file))
    cols = ['particleid', 'particlegroup', 'sequencenumber', 'particleidloc', 'time',
            'x', 'y', 'z', 'k', 'node', 'xloc', 'yloc', 'zloc', 'stressperiod', 'timestep']

    pathrecs_all: list = []
    for pid in all_pids:
        try:
            pathrecs_all.extend(pl.get_data(partid=pid))
        except Exception:
            # It’s ok if some particles have no pathline records
            continue

    if not pathrecs_all:
        print(f"[WARN] No pathline records found in: {pl_file}")
        return None

    df = pd.DataFrame.from_records(pathrecs_all, columns=cols)

    # ---- Model->World XY transform (MODFLOW Y is top-down; switch to bottom-up)
    x_local = df["x"].to_numpy(float)
    y_local_top = df["y"].to_numpy(float)
    y_local_bottom = total_y - y_local_top

    x_abs = xorigin + (x_local * cos_t - y_local_bottom * sin_t)
    y_abs = yorigin + (x_local * sin_t + y_local_bottom * cos_t)

    df["x_abs"] = x_abs
    df["y_abs"] = y_abs

    # ---- Build 3D lines
    rows, geoms = [], []
    for pid, g in (df.sort_values(["particleid", "time"]).groupby("particleid", sort=False)):
        xs = g["x_abs"].to_numpy(float)
        ys = g["y_abs"].to_numpy(float)
        zs = g["z"].to_numpy(float)
        ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
        if ok.sum() > 1:
            geoms.append(LineString(list(zip(xs[ok], ys[ok], zs[ok]))))
            rows.append({"particleid": int(pid)})

    if not geoms:
        print("[WARN] No valid 3D polylines could be built.")
        return None

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=hec_crs)

    # ---- Write to shapefile (prefer ArcPy POLYLINE Z; fall back to GeoPandas)
    shp_name = f"{direction}_full_pathlines_3D.shp"
    out_shp = output_folder / shp_name
    try:
        import arcpy
        # Build SpatialReference from CRS
        sr = None
        try:
            epsg = hec_crs.to_epsg() if hec_crs is not None and hasattr(hec_crs, "to_epsg") else None
            if epsg:
                sr = arcpy.SpatialReference(epsg)
        except Exception:
            sr = None
        if sr is None:
            try:
                wkt = hec_crs.to_wkt() if hec_crs is not None and hasattr(hec_crs, "to_wkt") else None
                if wkt:
                    sr = arcpy.SpatialReference(); sr.loadFromString(wkt)
            except Exception:
                sr = None

        if arcpy.Exists(str(out_shp)):
            arcpy.management.Delete(str(out_shp))
        arcpy.management.CreateFeatureclass(
            out_path=str(out_shp.parent),
            out_name=out_shp.name,
            geometry_type="POLYLINE",
            template="",
            has_m="DISABLED",
            has_z="ENABLED",
            spatial_reference=sr if sr else ""
        )
        # Minimal schema: particleid
        arcpy.management.AddField(str(out_shp), "particleid", "LONG")

        with arcpy.da.InsertCursor(str(out_shp), ["SHAPE@", "particleid"]) as cur:
            for rec, geom in zip(rows, geoms):
                # Rebuild with arcpy geometry for safety
                pts = getattr(geom, "coords", geom.__geo_interface__["coordinates"])
                arr = arcpy.Array([arcpy.Point(float(x), float(y), float(z)) for (x, y, z) in pts])
                cur.insertRow([arcpy.Polyline(arr, sr, has_z=True), rec["particleid"]])

        return str(out_shp)

    except Exception as e:
        # GeoPandas fallback (some GDAL builds preserve Z properly)
        try:
            gdf.to_file(out_shp, driver="ESRI Shapefile")
            return str(out_shp)
        except Exception as e2:
            print(f"[ERROR] Could not write full 3D pathlines shapefile via ArcPy ({e}) or GeoPandas ({e2}).")
            return None


# ---------- Vertical exaggeration helpers ----------
def _snap_to_nice(v: float,
                  palette=(0.5, 0.75, 1.0, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8,
                           10, 12, 15, 20, 25, 30, 40, 50, 60)) -> float:
    arr = np.asarray(palette, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(v)))])


def auto_vertical_exaggeration(
    Xg, Yg, top, botm,
    *,
    mode: str = "isometric",
    xy_screen_scale: tuple[float, float] = (1.0, 1.0),
    clip: tuple[float, float] = (2.0, 98.0),
    target_ratio_iso: float = 0.60,
    target_ratio_longitudinal: float = 0.35,
    min_ve: float = 0.75,
    max_ve: float = 60.0,
    nice: bool = True,
) -> tuple[float, dict]:
    x_span = float(np.nanmax(Xg) - np.nanmin(Xg))
    y_span = float(np.nanmax(Yg) - np.nanmin(Yg))

    z_low = float(np.nanpercentile(botm[-1], clip[0]))
    z_high = float(np.nanpercentile(top, clip[1]))
    z_span = max(z_high - z_low, 1e-9)

    x_eff = x_span * float(xy_screen_scale[0])
    y_eff = y_span * float(xy_screen_scale[1])

    if mode.lower().startswith("iso"):
        lateral_ref = (x_eff * y_eff) ** 0.5
        target = target_ratio_iso
    else:
        lateral_ref = max(x_eff, y_eff)
        target = target_ratio_longitudinal

    ve = (target * lateral_ref) / z_span
    ve = float(np.clip(ve, min_ve, max_ve))
    if nice:
        ve = _snap_to_nice(ve)

    return ve, {"x_span": x_span, "y_span": y_span, "z_span": z_span}


# ----------------------------
# Figure set (incl. alignment debug & 3D fixes)
# ----------------------------
def plot_hyporheic_workflow(
    *,
    sim_name: str,
    sim_path: str,
    exe_name: str,
    sat_image_path: str,
    projection_file: str,
    gw_domain_shapefile_path: str,
    particle_points_shp: str,
    pathlines_shp: str,
    river_cells,  # DataFrame-like with [layer,row,col,river_stage]
    particle_data_csv: str,
    pathline_stats_txt: str,
    direction: str,
    output_folder: str,
    plot_layer: int = 1,
    dpi: int = 300,
    show: bool = False,
    save_fig_head_overlay: bool = True,
    save_fig_head_overlay_w_paths_points: bool = True,
    save_fig_paths_points_only: bool = True,
    save_fig_isometric: bool = True,
    save_fig_longitudinal: bool = True,
    save_stats_csv: bool = True,
    save_stats_md: bool = True,
    export_head_geotiff: bool = False,
    export_head_mask_geotiff: bool = False,
    export_start_end_points_shp: bool = False,
    export_start_end_points_csv: bool = False,
    export_cropped_lines_shp: bool = False,
    ve_isometric: float | None = None,
    ve_longitudinal: float | None = None,
    ve_clip: tuple[float, float] = (2.0, 98.0),
    ve_target_ratio_isometric: float = 0.60,
    ve_target_ratio_longitudinal: float = 0.35,
    ve_min: float = 0.75,
    ve_max: float = 60.0,
    ve_snap_nice: bool = True,
) -> dict:
    """
    Generate the figure set (1–5) and optionally export GeoTIFFs + start/end SHPs/CSVs.
    """
    from shapely.ops import unary_union
    from shapely.geometry import Point as _Pt
    from rasterio.transform import Affine
    import matplotlib.lines as mlines

    def _maybe_save(fig, path: Path, enabled: bool):
        if not enabled:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        return str(path)

    def _write_gtiff(array, crs, extent, out_path: Path, nodata: float = -9999.0):
        """
        array: 2D np.ndarray or np.ma.MaskedArray (nrow, ncol)
        extent: [left, right, bottom, top] in same CRS
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(array)
        if isinstance(array, np.ma.MaskedArray):
            data = array.filled(nodata).astype(float)
        else:
            data = arr.astype(float)
        nrow, ncol = data.shape
        dx = (extent[1] - extent[0]) / ncol
        dy = (extent[3] - extent[2]) / nrow
        transform = Affine(dx, 0, extent[0], 0, -dy, extent[3])  # north-up
        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": nodata,
            "width": ncol,
            "height": nrow,
            "count": 1,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data.astype("float32"), 1)
        return str(out_path)

    outdir = Path(output_folder) / f"plots_{direction.lower()}"
    outdir.mkdir(parents=True, exist_ok=True)

    out = {
        "fig_head_overlay": None,
        "fig_head_overlay_w_paths_points": None,
        "fig_paths_points_only": None,
        "fig_isometric": None,
        "fig_longitudinal": None,
        "stats_csv": None,
        "stats_md": None,
        "head_geotiff": None,
        "mask_geotiff": None,
        "start_points_shp": None,
        "end_points_shp": None,
        "start_points_csv": None,
        "end_points_csv": None,
        "lines_shp_cropped": None,
        "output_dir": str(outdir),
        "ve_isometric_used": None,
        "ve_longitudinal_used": None,
    }

    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path, exe_name=exe_name)
    gwf = sim.get_model(sim.model_names[0])
    dis = gwf.get_package("DIS")

    candidate_hds = [
        Path(sim_path) / f"{gwf.name}.hds",
        Path(sim_path) / "gwf_model.hds",
    ]
    head_file = next((p for p in candidate_hds if p.exists()), None)
    if head_file is None:
        hds = list(Path(sim_path).glob("*.hds"))
        if not hds:
            raise FileNotFoundError("Could not find a head file (*.hds) in sim_path.")
        head_file = hds[0]

    hobj = flopy.utils.HeadFile(str(head_file))
    head3d = hobj.get_data(totim=hobj.get_times()[-1])

    # Load satellite and domain
    with rasterio.open(sat_image_path) as src:
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        img_crs = src.crs
        if src.count >= 3:
            img = np.dstack([src.read(i) for i in [1, 2, 3]]).astype(float)
            for i in range(3):
                b = img[:, :, i]
                rng = (b.max() - b.min()) or 1.0
                img[:, :, i] = (b - b.min()) / rng
        else:
            img = src.read(1)

    hec_crs = img_crs  # use the image CRS consistently in this function

    gdf_domain = gpd.read_file(gw_domain_shapefile_path)
    if gdf_domain.crs != img_crs:
        gdf_domain = gdf_domain.to_crs(img_crs)
    polygon = unary_union(gdf_domain.geometry)

    idomain = dis.idomain.array
    head_layer = head3d[plot_layer, :, :]
    idomain_layer = idomain[plot_layer, :, :]
    head_layer_masked = np.ma.masked_where(idomain_layer == 0, head_layer)

    nrow, ncol = head_layer.shape
    x = np.linspace(extent[0], extent[1], ncol)
    y = np.linspace(extent[2], extent[3], nrow)
    Xc, Yc = np.meshgrid(x, y)

    mask_shape = np.zeros_like(head_layer_masked, dtype=bool)
    for i in range(nrow):
        for j in range(ncol):
            mask_shape[i, j] = polygon.contains(_Pt(Xc[i, j], Yc[i, j]))
    head_layer_clipped = np.ma.masked_where(~mask_shape, head_layer_masked)

    finite_vals = np.asarray(head_layer_clipped.filled(np.nan)).ravel()
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    levels = np.linspace(finite_vals.min(), finite_vals.max(), 20) if finite_vals.size else np.linspace(0.0, 1.0, 20)

    # gdf_lines = gpd.read_file(pathlines_shp) if Path(pathlines_shp).exists() else gpd.GeoDataFrame()
    # after (robust to ""/None and directories)
    gdf_lines = (
        gpd.read_file(pathlines_shp)
        if (pathlines_shp and Path(pathlines_shp).is_file())
        else gpd.GeoDataFrame()
    )
    if not gdf_lines.empty and gdf_lines.crs != img_crs:
        gdf_lines = gdf_lines.to_crs(img_crs)

    # gdf_points = gpd.read_file(particle_points_shp) if Path(particle_points_shp).exists() else gpd.GeoDataFrame()
    gdf_points = (
        gpd.read_file(particle_points_shp)
        if (particle_points_shp and Path(particle_points_shp).is_file())
        else gpd.GeoDataFrame()
    )
    if not gdf_points.empty and gdf_points.crs != img_crs:
        gdf_points = gdf_points.to_crs(img_crs)

    if not gdf_points.empty:
        sort_keys = [k for k in ("particleid", "res_time", "time", "t") if k in gdf_points.columns]
        gdf_points_sorted = gdf_points.sort_values(sort_keys) if sort_keys else gdf_points.copy()
        pid_col = "particleid" if "particleid" in gdf_points.columns else gdf_points.columns[0]
        start_points = gdf_points_sorted.groupby(pid_col).first()
        end_points = gdf_points_sorted.groupby(pid_col).last()
    else:
        start_points = gpd.GeoDataFrame(geometry=[], crs=img_crs)
        end_points = gpd.GeoDataFrame(geometry=[], crs=img_crs)

    # ---------- Alignment Debug (grid vs pathlines/points) ----------
    def _as_float_or_default(v, default=0.0):
        return coerce_to_float(v, default)

    angrot = float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0)
    xorigin = _as_float_or_default(getattr(dis, "xorigin", None), getattr(gwf.modelgrid, "xoffset", 0.0))
    yorigin = _as_float_or_default(getattr(dis, "yorigin", None), getattr(gwf.modelgrid, "yoffset", 0.0))

    delr = np.asarray(dis.delr.array, dtype=float)
    delc = np.asarray(dis.delc.array, dtype=float)

    def _model_to_world(x, y_topdown, total_y, x0, y0, ang_deg):
        theta = np.deg2rad(float(ang_deg))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        y_bottom = total_y - y_topdown
        xw = x0 + (x * cos_t - y_bottom * sin_t)
        yw = y0 + (x * sin_t + y_bottom * cos_t)
        return xw, yw

    # Debug directory
    debug_dir = Path(outdir) / f"alignment_debug_{direction.lower()}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # === Use vertices from flopy (authoritative world placement)
    grid_poly, grid_edges_gdf, grid_w, grid_h = _grid_footprint_from_vertices(gwf.modelgrid)
    grid_gdf = gpd.GeoDataFrame(
        [{
            "name": "grid_box",
            "xorigin": coerce_to_float(getattr(dis, "xorigin", 0.0), 0.0),
            "yorigin": coerce_to_float(getattr(dis, "yorigin", 0.0), 0.0),
            "angrot_deg": float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0),
            "width": grid_w, "height": grid_h
        }],
        geometry=[grid_poly],
        crs=gwf.modelgrid.crs or hec_crs
    )

    grid_box_shp = debug_dir / "grid_box_HECRAS_CRS.shp"
    grid_edges_shp = debug_dir / "grid_edges_HECRAS_CRS.shp"
    grid_gdf.to_file(grid_box_shp, driver="ESRI Shapefile")
    grid_edges_gdf.to_file(grid_edges_shp, driver="ESRI Shapefile")

    lines_debug_shp = None
    points_debug_shp = None
    if not gdf_lines.empty:
        lines_debug_shp = debug_dir / "pathlines_debug.shp"
        gdf_lines.to_file(lines_debug_shp, driver="ESRI Shapefile")
    if not gdf_points.empty:
        points_debug_shp = debug_dir / "points_debug.shp"
        gdf_points.to_file(points_debug_shp, driver="ESRI Shapefile")

    report = {}
    report["grid_box_shp"] = str(grid_box_shp)
    report["grid_edges_shp"] = str(grid_edges_shp)
    if lines_debug_shp:
        report["pathlines_shp"] = str(lines_debug_shp)
    if points_debug_shp:
        report["points_shp"] = str(points_debug_shp)

    try:
        if not gdf_lines.empty:
            total_len = float(gdf_lines.length.sum())
            inter_len = float(gdf_lines.intersection(grid_poly).length.sum())
            report["pathline_length_fraction_inside"] = inter_len / total_len if total_len > 0 else None
        if not gdf_points.empty:
            inside_pts = gdf_points.within(grid_poly).sum()
            report["points_fraction_inside"] = float(inside_pts) / float(len(gdf_points)) if len(gdf_points) else None
    except Exception:
        report["pathline_length_fraction_inside"] = None
        report["points_fraction_inside"] = None

    gp_cent = grid_gdf.geometry.centroid.iloc[0]
    report["grid_centroid"] = (float(gp_cent.x), float(gp_cent.y))
    report["grid_bounds"] = tuple(map(float, grid_gdf.total_bounds))

    if not gdf_lines.empty:
        lp_cent = gdf_lines.unary_union.centroid
        report["pathlines_centroid"] = (float(lp_cent.x), float(lp_cent.y))
        report["pathlines_bounds"] = tuple(map(float, gdf_lines.total_bounds))
    else:
        report["pathlines_centroid"] = None
        report["pathlines_bounds"] = None

    if not gdf_points.empty:
        pp_cent = gdf_points.unary_union.centroid
        report["points_centroid"] = (float(pp_cent.x), float(pp_cent.y))
        report["points_bounds"] = tuple(map(float, gdf_points.total_bounds))
    else:
        report["points_centroid"] = None
        report["points_bounds"] = None

    report["grid_rotation_deg"] = float(angrot % 180.0)

    try:
        part_bbox_model = None
        if isinstance(particle_data_csv, (str, Path)) and Path(particle_data_csv).exists():
            _df_part = pd.read_csv(particle_data_csv, nrows=200000)
            if "m_xcoord" in _df_part.columns and "m_ycoord" in _df_part.columns:
                xm, ym = _df_part["m_xcoord"].to_numpy(float), _df_part["m_ycoord"].to_numpy(float)
            elif {"x", "y"} <= set(_df_part.columns):
                xm, ym = _df_part["x"].to_numpy(float), _df_part["y"].to_numpy(float)
            else:
                xm = ym = None
            if xm is not None:
                total_x = float(np.sum(delr))
                total_y = float(np.sum(delc))
                inside = (xm >= 0) & (xm <= total_x) & (ym >= 0) & (ym <= total_y)
                report["model_axis_fraction_inside_rect_0_total"] = float(np.nanmean(inside.astype(float)))
                part_bbox_model = (float(np.nanmin(xm)), float(np.nanmin(ym)),
                                   float(np.nanmax(xm)), float(np.nanmax(ym)))
        report["model_axis_bbox_pathpoints"] = part_bbox_model
    except Exception:
        report["model_axis_fraction_inside_rect_0_total"] = None
        report["model_axis_bbox_pathpoints"] = None

    print("\n=== ALIGNMENT DEBUG (HEC‑RAS CRS) ===")
    print(f"Grid box shapefile: {report['grid_box_shp']}")
    print(f"Grid edges shapefile: {report['grid_edges_shp']}")
    if lines_debug_shp:
        print(f"Pathlines shapefile: {report['pathlines_shp']}")
    if points_debug_shp:
        print(f"Points shapefile: {report['points_shp']}")
    print(f"Grid centroid: {report['grid_centroid']}, bounds: {report['grid_bounds']}")
    if report.get('pathlines_bounds'):
        print(f"Pathlines centroid: {report['pathlines_centroid']}, bounds: {report['pathlines_bounds']}")
    if report.get('points_bounds'):
        print(f"Points centroid: {report['points_centroid']}, bounds: {report['points_bounds']}")
    print(f"Grid rotation (deg): {report['grid_rotation_deg']}")
    print(f"Pathline length fraction inside grid: {report.get('pathline_length_fraction_inside')}")
    print(f"Points fraction inside grid: {report.get('points_fraction_inside')}")
    print(f"Model-axis fraction inside [0..Σdelr]×[0..Σdelc]: {report.get('model_axis_fraction_inside_rect_0_total')}")
    print("=====================================\n")

    out["alignment_debug_report"] = report

    # ---------- FIGURE 1 — Head overlay ----------
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, extent=extent, origin="upper", zorder=1)
    contourf = ax.contourf(Xc, Yc, head_layer_clipped, levels=levels, cmap="viridis", alpha=1.0, zorder=0)
    ax.contour(Xc, Yc, head_layer_clipped, levels=levels, colors="k", linewidths=0.25, zorder=2)
    gdf_domain.boundary.plot(ax=ax, edgecolor="white", linewidth=2, zorder=6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="5%", pad=1.0)
    cb = fig.colorbar(contourf, cax=cax)
    cax.yaxis.set_ticks_position("left")
    cb.set_label("Groundwater Head (ft)")
    ax.set_xlabel("Easting (UTM)")
    ax.set_ylabel("Northing (UTM)")
    ax.set_title("")
    plt.tight_layout()
    out["fig_head_overlay"] = _maybe_save(fig, outdir / f"01_head_overlay_{direction}_layer{plot_layer}.png", save_fig_head_overlay)
    if show:
        plt.show()
    plt.close(fig)

    # ---------- FIGURE 2 — Overlay + pathlines/points ----------
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, extent=extent, origin="upper", zorder=1)
    contourf = ax.contourf(Xc, Yc, head_layer_clipped, levels=levels, cmap="viridis", alpha=1.0, zorder=0)
    ax.contour(Xc, Yc, head_layer_clipped, levels=levels, cmap="viridis", zorder=2)
    if not gdf_lines.empty:
        gdf_lines.plot(ax=ax, color="blue", linewidth=0.5, label="_nolegend_", zorder=3)
    if not start_points.empty:
        ax.scatter(start_points.geometry.x, start_points.geometry.y, color="lime", s=20, label="Start Points", zorder=5)
    if not end_points.empty:
        ax.scatter(end_points.geometry.x, end_points.geometry.y, color="red", s=20, label="End Points", zorder=6)
    gdf_domain.boundary.plot(ax=ax, edgecolor="white", linewidth=2, zorder=6)
    xmin, ymin, xmax, ymax = gdf_domain.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    start_handle = mlines.Line2D([], [], color="lime", marker="o", linestyle="None", markersize=5, label="Start Points")
    end_handle = mlines.Line2D([], [], color="red", marker="o", linestyle="None", markersize=5, label="End Points")
    line_handle = mlines.Line2D([], [], color="blue", linewidth=2, label="Pathlines")
    ax.legend(handles=[start_handle, end_handle, line_handle], loc="upper right", fontsize=12, frameon=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="5%", pad=1.0)
    cb = fig.colorbar(contourf, cax=cax)
    cax.yaxis.set_ticks_position("left")
    cb.set_label("Groundwater Head (ft)")
    ax.set_xlabel("Easting (UTM)")
    ax.set_ylabel("Northing (UTM)")
    ax.set_title("")
    plt.tight_layout()
    out["fig_head_overlay_w_paths_points"] = _maybe_save(
        fig, outdir / f"02_head_overlay_w_paths_points_{direction}_layer{plot_layer}.png",
        save_fig_head_overlay_w_paths_points
    )
    if show:
        plt.show()
    plt.close(fig)

    # ---------- FIGURE 3 — Paths/points only ----------
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, extent=extent, origin="upper")
    gdf_domain.boundary.plot(ax=ax, edgecolor="white", linewidth=2, zorder=2)
    if not gdf_lines.empty:
        gdf_lines.plot(ax=ax, color="blue", linewidth=0.2, label="Pathlines", zorder=3)
    if not start_points.empty:
        start_points.plot(ax=ax, marker="o", color="lime", markersize=2, label="Start Points", zorder=5)
    if not end_points.empty:
        end_points.plot(ax=ax, marker="x", color="red", markersize=10, label="End Points", zorder=4)
    plt.legend()
    plt.xlabel("Easting (UTM)")
    plt.ylabel("Northing (UTM)")
    plt.title("")
    plt.tight_layout()
    out["fig_paths_points_only"] = _maybe_save(
        fig, outdir / f"03_paths_points_only_{direction}.png", save_fig_paths_points_only
    )
    if show:
        plt.show()
    plt.close(fig)

    # ---------- 3D PREP ----------
    def _real_world_grid_coords(dis):
        delr_ = dis.delr.array
        delc_ = dis.delc.array
        x_ = np.cumsum(np.insert(delr_, 0, 0))[:-1]
        y_ = np.cumsum(np.insert(delc_, 0, 0))[:-1]
        X_, Y_ = np.meshgrid(x_, y_)
        return X_, Y_, delr_, delc_

    top = dis.top.array.copy()
    botm = dis.botm.array
    Xg, Yg, delr3d, delc3d = _real_world_grid_coords(dis)

    try:
        rc_df = pd.DataFrame(river_cells).copy()
        for r in rc_df.itertuples(index=False):
            k = getattr(r, "layer", getattr(r, "k", 0))
            j = getattr(r, "row", getattr(r, "j", None))
            i = getattr(r, "col", getattr(r, "i", None))
            stage = getattr(r, "river_stage", getattr(r, "stage", None))
            if k == 0 and stage is not None and j is not None and i is not None:
                top[j, i] = stage
    except Exception:
        pass

    def _load_pathline_vertices_for_3d(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if "particleid" not in df.columns:
            for alt in ("particle_id", "pid", "partid", "particleidloc"):
                if alt in df.columns:
                    df["particleid"] = df[alt]
                    break
        if "particleid" not in df.columns:
            df["particleid"] = df.groupby(["x", "y"]).ngroup() if {"x", "y"} <= set(df.columns) else np.arange(len(df))

        if not {"x", "y"}.issubset(df.columns):
            raise ValueError(
                "Pathline CSV must contain model-axis columns 'x' and 'y'. "
                "Pass the *_pathlines_filtered.csv written by process_and_export_modpath7_results."
            )
        zcol = "z"
        if zcol not in df.columns:
            for altz in ("zelev", "zloc", "elev", "elevation"):
                if altz in df.columns:
                    zcol = altz
                    break
        if zcol not in df.columns:
            raise ValueError("Pathline CSV must contain a z-like column ('z', 'zelev', 'zloc', 'elev', or 'elevation').")

        if "res_time" in df.columns:
            df["res_time"] = df["res_time"]
        elif "time" in df.columns:
            df["res_time"] = df["time"]
        else:
            df["res_time"] = df.groupby("particleid").cumcount()

        out_df = df.rename(columns={zcol: "zelev"}).loc[:, ["particleid", "res_time", "x", "y", "zelev"]]
        out_df = out_df.rename(columns={"x": "m_xcoord", "y": "m_ycoord"})
        return out_df

    part = _load_pathline_vertices_for_3d(particle_data_csv)

    start_df = pd.DataFrame(columns=["m_xcoord", "m_ycoord", "zelev", "particleid", "res_time"])
    end_df = start_df.copy()
    try:
        if "particleid" in part.columns:
            if "res_time" in part.columns:
                s_idx = part.groupby("particleid")["res_time"].idxmin()
                e_idx = part.groupby("particleid")["res_time"].idxmax()
                start_df = part.loc[s_idx]
                end_df = part.loc[e_idx]
            else:
                tmp = part.sort_values("particleid")
                start_df = tmp.groupby("particleid").first().reset_index()
                end_df = tmp.groupby("particleid").last().reset_index()
    except Exception:
        pass

    # ---------- FIGURE 4 — 3D Isometric ----------
    fig = plt.figure(figsize=(18, 18))
    try:
        ax = fig.add_subplot(111, projection="3d")

        ve_iso_calc, spans_iso = auto_vertical_exaggeration(
            Xg, Yg, top, botm,
            mode="isometric",
            clip=ve_clip,
            target_ratio_iso=ve_target_ratio_isometric,
            target_ratio_longitudinal=ve_target_ratio_longitudinal,
            min_ve=ve_min, max_ve=ve_max, nice=ve_snap_nice,
        )
        ve_iso = float(ve_iso_calc if ve_isometric is None else np.clip(ve_isometric, ve_min, ve_max))
        out["ve_isometric_used"] = ve_iso

        rs = max(1, top.shape[0] // 300)
        cs = max(1, top.shape[1] // 300)
        Xd, Yd = Xg[::rs, ::cs], Yg[::rs, ::cs]
        top_d = top[::rs, ::cs] * ve_iso
        bot_d = botm[-1, ::rs, ::cs] * ve_iso

        ax.plot_surface(Xd, Yd, top_d, edgecolor="black", linewidth=0.1, color="white", alpha=0.05, antialiased=True)
        ax.plot_surface(Xd, Yd, bot_d, edgecolor="black", linewidth=0.1, color="white", alpha=0.05, antialiased=True)

        y_side, x_side = Yg[:, 0], Xg[0, :]
        z2 = np.column_stack([botm[-1, :, 0], top[:, 0]]) * ve_iso
        ax.plot_surface(np.zeros((top.shape[0], 2)) + x_side[0], np.repeat(y_side[:, None], 2, axis=1),
                        z2, edgecolor="black", linewidth=0.1, color="white", alpha=0.05)
        z2 = np.column_stack([botm[-1, :, -1], top[:, -1]]) * ve_iso
        ax.plot_surface(np.full((top.shape[0], 2), x_side[-1]), np.repeat(y_side[:, None], 2, axis=1),
                        z2, edgecolor="black", linewidth=0.1, color="white", alpha=0.05)
        z2 = np.column_stack([botm[-1, 0, :], top[0, :]]) * ve_iso
        ax.plot_surface(np.repeat(x_side[:, None], 2, axis=1), np.zeros((top.shape[1], 2)) + y_side[0],
                        z2, edgecolor="black", linewidth=0.1, color="white", alpha=0.05)
        z2 = np.column_stack([botm[-1, -1, :], top[-1, :]]) * ve_iso
        ax.plot_surface(np.repeat(x_side[:, None], 2, axis=1), np.full((top.shape[1], 2), y_side[-1]),
                        z2, edgecolor="black", linewidth=0.1, color="white", alpha=0.05)

        # Pathlines (flip Y to bottom-up)
        if {"m_xcoord", "m_ycoord", "zelev", "particleid", "res_time"} <= set(part.columns):
            total_y_plot = float(np.sum(delc3d))
            first_line = True
            for pid, g in part.groupby("particleid", sort=False):
                g = g.sort_values("res_time")
                if len(g) > 4000:
                    step = int(np.ceil(len(g) / 4000))
                    g = g.iloc[::step, :]
                y_plot = total_y_plot - g["m_ycoord"]
                ax.plot(g["m_xcoord"], y_plot, g["zelev"] * ve_iso,
                        linewidth=0.6 if first_line else 0.4, alpha=0.8, color="blue",
                        label="Pathline" if first_line else None)
                first_line = False
        if not end_df.empty:
            total_y_plot = float(np.sum(delc3d))
            ax.scatter(end_df["m_xcoord"], total_y_plot - end_df["m_ycoord"], end_df["zelev"] * ve_iso,
                       s=5, color="red", label="End Points")
        if not start_df.empty:
            total_y_plot = float(np.sum(delc3d))
            ax.scatter(start_df["m_xcoord"], total_y_plot - start_df["m_ycoord"], start_df["zelev"] * ve_iso,
                       s=5, color="lime", label="Start Points")

        x_span, y_span, z_span = spans_iso["x_span"], spans_iso["y_span"], spans_iso["z_span"]
        ax.set_box_aspect([x_span, y_span, z_span * ve_iso])

        ax.set_xlabel("Easting (m)", labelpad=15, fontsize=12)
        ax.set_ylabel("Northing (m)", labelpad=15, fontsize=12)
        ax.set_zlabel(f"Elevation (ft)  (VE ×{ve_iso:g})")
        ax.set_title(f"Isometric View — VE ×{ve_iso:g}")
        if ax.get_legend_handles_labels()[1]:
            ax.legend()
    finally:
        out["fig_isometric"] = _maybe_save(fig, outdir / f"04_isometric_{direction}.png", save_fig_isometric)
        if show:
            plt.show()
        plt.close(fig)

    # ---------- FIGURE 5 — 3D Longitudinal ----------
    fig = plt.figure(figsize=(20, 18))
    try:
        ax = fig.add_subplot(111, projection="3d")
        yfac, xfac = 8.0, 2.0

        ve_long_calc, spans_long = auto_vertical_exaggeration(
            Xg, Yg, top, botm,
            mode="longitudinal",
            xy_screen_scale=(xfac, yfac),
            clip=ve_clip,
            target_ratio_iso=ve_target_ratio_isometric,
            target_ratio_longitudinal=ve_target_ratio_longitudinal,
            min_ve=ve_min, max_ve=ve_max, nice=ve_snap_nice,
        )
        ve_long = float(ve_long_calc if ve_longitudinal is None else np.clip(ve_longitudinal, ve_min, ve_max))
        out["ve_longitudinal_used"] = ve_long

        TARGET = 300
        rs = max(1, top.shape[0] // TARGET)
        cs = max(1, top.shape[1] // TARGET)
        Xd, Yd = Xg[::rs, ::cs], Yg[::rs, ::cs]
        top_d = top[::rs, ::cs] * ve_long
        bot_d = botm[-1, ::rs, ::cs] * ve_long

        ax.plot_surface(Xd, Yd, top_d, edgecolor="black", linewidth=0.1, color="white", alpha=0.05, antialiased=True)
        ax.plot_surface(Xd, Yd, bot_d, edgecolor="black", linewidth=0.1, color="white", alpha=0.05, antialiased=True)

        y_side, x_side = Yg[:, 0], Xg[0, :]
        step_y = max(1, top.shape[0] // TARGET)
        step_x = max(1, top.shape[1] // TARGET)
        z2 = np.column_stack([botm[-1, ::step_y, 0], top[::step_y, 0]]) * ve_long
        ax.plot_surface(np.zeros((y_side[::step_y].size, 2)) + x_side[0],
                        np.repeat(y_side[::step_y, None], 2, axis=1),
                        z2, edgecolor="black", linewidth=0.1, color="white", alpha=0.05)
        z2 = np.column_stack([botm[-1, ::step_y, -1], top[::step_y, -1]]) * ve_long
        ax.plot_surface(np.full((y_side[::step_y].size, 2), x_side[-1]),
                        np.repeat(y_side[::step_y, None], 2, axis=1),
                        z2, edgecolor="black", linewidth=0.1, color="white", alpha=0.05)
        z2 = np.column_stack([botm[-1, 0, ::step_x], top[0, ::step_x]]) * ve_long
        ax.plot_surface(np.repeat(x_side[::step_x, None], 2, axis=1),
                        np.zeros((x_side[::step_x].size, 2)) + y_side[0],
                        z2, edgecolor="black", linewidth=0.1, color="white", alpha=0.05)
        z2 = np.column_stack([botm[-1, -1, ::step_x], top[-1, ::step_x]]) * ve_long
        ax.plot_surface(np.repeat(x_side[::step_x, None], 2, axis=1),
                        np.full((x_side[::step_x].size, 2), y_side[-1]),
                        z2, edgecolor="black", linewidth=0.1, color="white", alpha=0.05)

        # Pathlines (flip Y)
        if {"m_xcoord", "m_ycoord", "zelev", "particleid", "res_time"} <= set(part.columns):
            total_y_plot = float(np.sum(delc3d))
            first_line = True
            for pid, g in part.groupby("particleid", sort=False):
                g = g.sort_values("res_time")
                if len(g) > 4000:
                    step = int(np.ceil(len(g) / 4000))
                    g = g.iloc[::step, :]
                y_plot = total_y_plot - g["m_ycoord"]
                ax.plot(g["m_xcoord"], y_plot, g["zelev"] * ve_long,
                        linewidth=0.6 if first_line else 0.4, alpha=0.8, color="blue",
                        label="Pathline" if first_line else None)
                first_line = False
        if not end_df.empty:
            total_y_plot = float(np.sum(delc3d))
            ax.scatter(end_df["m_xcoord"], total_y_plot - end_df["m_ycoord"], end_df["zelev"] * ve_long,
                       s=5, color="red", label="End Points")
        if not start_df.empty:
            total_y_plot = float(np.sum(delc3d))
            ax.scatter(start_df["m_xcoord"], total_y_plot - start_df["m_ycoord"], start_df["zelev"] * ve_long,
                       s=5, color="lime", label="Start Points")

        x_span, y_span, z_span = spans_long["x_span"], spans_long["y_span"], spans_long["z_span"]
        ax.set_box_aspect([x_span * xfac, y_span * yfac, z_span * ve_long])

        ax.set_xlabel("Easting (m)", labelpad=15, fontsize=12)
        ax.set_ylabel("Northing (m)", labelpad=40, fontsize=12)
        ax.set_zlabel(f"Elevation (ft)  (VE ×{ve_long:g})")
        ax.set_title(f"Longitudinal View — VE ×{ve_long:g}")
        ax.view_init(elev=1, azim=180)
        if ax.get_legend_handles_labels()[1]:
            ax.legend()
    finally:
        out["fig_longitudinal"] = _maybe_save(fig, outdir / f"05_longitudinal_{direction}.png", save_fig_longitudinal)
        if show:
            plt.show()
        plt.close(fig)

    # ---------- Optional GeoTIFF exports ----------
    if export_head_geotiff:
        out["head_geotiff"] = _write_gtiff(
            head_layer_clipped, img_crs, extent, outdir / f"{direction}_head_layer{plot_layer}_clipped.tif"
        )
    if export_head_mask_geotiff:
        mask01 = mask_shape.astype("float32")
        out["mask_geotiff"] = _write_gtiff(
            mask01, img_crs, extent, outdir / f"{direction}_domain_mask_01.tif", nodata=-1.0
        )

    if export_start_end_points_shp and not start_points.empty:
        p = outdir / f"{direction}_start_points.shp"
        start_points.to_file(p, driver="ESRI Shapefile")
        out["start_points_shp"] = str(p)
    if export_start_end_points_shp and not end_points.empty:
        p = outdir / f"{direction}_end_points.shp"
        end_points.to_file(p, driver="ESRI Shapefile")
        out["end_points_shp"] = str(p)

    if export_start_end_points_csv and not start_points.empty:
        p = outdir / f"{direction}_start_points.csv"
        start_points.assign(x=start_points.geometry.x, y=start_points.geometry.y).drop(columns="geometry").to_csv(p, index=False)
        out["start_points_csv"] = str(p)
    if export_start_end_points_csv and not end_points.empty:
        p = outdir / f"{direction}_end_points.csv"
        end_points.assign(x=end_points.geometry.x, y=end_points.geometry.y).drop(columns="geometry").to_csv(p, index=False)
        out["end_points_csv"] = str(p)

    if export_cropped_lines_shp and not gdf_lines.empty:
        try:
            cropped = gpd.clip(gdf_lines, gdf_domain)
            p = outdir / f"{direction}_pathlines_cropped.shp"
            cropped.to_file(p, driver="ESRI Shapefile")
            out["lines_shp_cropped"] = str(p)
        except Exception:
            pass

    def _parse_stats_file(filepath: str) -> pd.DataFrame:
        rows, group = [], None
        with open(filepath, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                is_group = (
                    line and
                    not line.startswith(("mean:", "median:", "min:", "max:", "range:")) and
                    ":" not in line and
                    not line.startswith("(")
                ) or line.startswith(("Per Cell:", "Particle Path:"))
                if is_group:
                    group = line
                    continue
                if group and ":" in line:
                    stat, val = line.split(":", 1)
                    stat = stat.strip()
                    try:
                        value = float(val.strip())
                    except ValueError:
                        value = val.strip()
                    rows.append({"Group": group, "Stat": stat, "Value": value})
        return pd.DataFrame(rows)

    def _stats_markdown(df: pd.DataFrame, float_format: str = "{:,.3f}") -> str:
        lines, prev = [], None
        for _, r in df.iterrows():
            grp, stat, val = r["Group"], r["Stat"], r["Value"]
            if isinstance(val, float):
                if abs(val) > 1e5 or (abs(val) < 1e-3 and val != 0):
                    v = f"{val:.3e}"
                else:
                    v = float_format.format(val)
            else:
                v = str(val)
            if grp != prev:
                lines.append(f"\n**{grp}**")
                prev = grp
            lines.append(f"  - {stat}: {v}")
        return "\n".join(lines)

    if pathline_stats_txt and Path(pathline_stats_txt).exists():
        df_stats = _parse_stats_file(pathline_stats_txt)
        if save_stats_csv:
            p = outdir / f"pathline_stats_{direction}.csv"
            df_stats.to_csv(p, index=False)
            out["stats_csv"] = str(p)
        if save_stats_md:
            p = outdir / f"pathline_stats_{direction}.md"
            p.write_text(_stats_markdown(df_stats))
            out["stats_md"] = str(p)

    return out


# ----------------------------
# Diagnostics exporter (grid walls + pathlines/points)
# ----------------------------
def export_fig45_diagnostics(
    *,
    gwf,
    dis,
    idomain: np.ndarray,
    part_df: pd.DataFrame,           # columns: particleid, res_time, m_xcoord, m_ycoord, zelev
    start_df: pd.DataFrame,          # same columns subset
    end_df: pd.DataFrame,            # same columns subset
    output_folder: str | Path,
    hec_crs,                         # pyproj CRS or anything gpd can accept
    direction: str = "Forward",
    write_world: bool = True,
    write_model_axes: bool = True,
    write_top_raster_if_uniform: bool = True,
) -> dict:
    """
    Create shapefiles to verify whether 3D grid vs pathlines/points are aligned.
    """
    from rasterio.transform import Affine

    outdir = Path(output_folder) / f"diagnostics_{direction.lower()}"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def _is_uniform(arr) -> bool:
        return np.allclose(arr, arr.flat[0])

    def _model_vertices(delr, delc):
        x_edges = np.concatenate(([0.0], np.cumsum(delr)))
        y_edges = np.concatenate(([0.0], np.cumsum(delc)))
        Xv = np.tile(x_edges[None, :], (y_edges.size, 1))
        Yv = np.tile(y_edges[:, None], (x_edges.size, 1))
        return Xv, Yv

    def _world_vertices():
        Xv = gwf.modelgrid.xvertices
        Yv = gwf.modelgrid.yvertices
        return Xv, Yv

    def _write_cells_as_polygons(Xv, Yv, crs, label: str):
        polys, top_vals, bot0_vals, rows, cols = [], [], [], [], []
        top = dis.top.array
        bot0 = dis.botm.array[0, :, :]
        nrow, ncol = top.shape
        for j in range(nrow):
            for i in range(ncol):
                if idomain[0, j, i] != 1:
                    continue
                poly = Polygon([
                    (Xv[j, i], Yv[j, i]),
                    (Xv[j, i + 1], Yv[j, i + 1]),
                    (Xv[j + 1, i + 1], Yv[j + 1, i + 1]),
                    (Xv[j + 1, i], Yv[j + 1, i]),
                ])
                polys.append(poly)
                top_vals.append(float(top[j, i]))
                bot0_vals.append(float(bot0[j, i]))
                rows.append(j)
                cols.append(i)
        gdf = gpd.GeoDataFrame(
            {"row": rows, "col": cols, "top": top_vals, "bot0": bot0_vals},
            geometry=polys, crs=crs
        )
        p = outdir / f"grid_cells_{label}.shp"
        gdf.to_file(p, driver="ESRI Shapefile")
        return str(p)

    def _write_outer_walls(Xv, Yv, crs, label: str):
        nrow, ncol = dis.top.array.shape
        lines, side_list, z_top, z_bot, rows, cols = [], [], [], [], [], []
        top = dis.top.array
        bot0 = dis.botm.array[0, :, :]

        def _seg(j0, i0, j1, i1):
            return LineString([(Xv[j0, i0], Yv[j0, i0]), (Xv[j1, i1], Yv[j1, i1])])

        for j in range(nrow):
            for i in range(ncol):
                if idomain[0, j, i] != 1:
                    continue
                if i == 0 or idomain[0, j, i - 1] != 1:
                    lines.append(_seg(j, i, j + 1, i))
                    side_list.append("west"); z_top.append(float(top[j, i])); z_bot.append(float(bot0[j, i])); rows.append(j); cols.append(i)
                if i == ncol - 1 or idomain[0, j, i + 1] != 1:
                    lines.append(_seg(j, i + 1, j + 1, i + 1))
                    side_list.append("east"); z_top.append(float(top[j, i])); z_bot.append(float(bot0[j, i])); rows.append(j); cols.append(i)
                if j == 0 or idomain[0, j - 1, i] != 1:
                    lines.append(_seg(j, i, j, i + 1))
                    side_list.append("north"); z_top.append(float(top[j, i])); z_bot.append(float(bot0[j, i])); rows.append(j); cols.append(i)
                if j == nrow - 1 or idomain[0, j + 1, i] != 1:
                    lines.append(_seg(j + 1, i, j + 1, i + 1))
                    side_list.append("south"); z_top.append(float(top[j, i])); z_bot.append(float(bot0[j, i])); rows.append(j); cols.append(i)

        gdf = gpd.GeoDataFrame(
            {"row": rows, "col": cols, "side": side_list, "z_top": z_top, "z_bot": z_bot},
            geometry=lines, crs=crs
        )
        p = outdir / f"grid_walls_{label}.shp"
        gdf.to_file(p, driver="ESRI Shapefile")
        return str(p)

    def _write_pathlines_points(label: str, use_world: bool):
        line_rows, line_geoms = [], []
        df = part_df.copy()

        if use_world:
            angrot = float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0)
            theta = np.deg2rad(angrot)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x0 = coerce_to_float(getattr(dis, "xorigin", 0.0), 0.0)
            y0 = coerce_to_float(getattr(dis, "yorigin", 0.0), 0.0)
            xw = x0 + (df["m_xcoord"].to_numpy() * cos_t - df["m_ycoord"].to_numpy() * sin_t)
            yw = y0 + (df["m_xcoord"].to_numpy() * sin_t + df["m_ycoord"].to_numpy() * cos_t)
            df["X"] = xw; df["Y"] = yw
            crs = hec_crs
        else:
            df["X"] = df["m_xcoord"]; df["Y"] = df["m_ycoord"]
            crs = None

        for pid, g in df.sort_values(["particleid", "res_time"]).groupby("particleid", sort=False):
            pts = list(zip(g["X"].to_numpy(), g["Y"].to_numpy()))
            if len(pts) > 1:
                line_geoms.append(LineString(pts))
                line_rows.append({"particleid": int(pid)})

        lines_gdf = gpd.GeoDataFrame(line_rows, geometry=line_geoms, crs=crs)
        p_lines = outdir / f"fig45_pathlines_{label}.shp"
        lines_gdf.to_file(p_lines, driver="ESRI Shapefile")

        def _pts(df_pts, name):
            if df_pts is None or df_pts.empty:
                return None
            if use_world:
                angrot = float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0)
                theta = np.deg2rad(angrot)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                x0 = coerce_to_float(getattr(dis, "xorigin", 0.0), 0.0)
                y0 = coerce_to_float(getattr(dis, "yorigin", 0.0), 0.0)
                X = x0 + (df_pts["m_xcoord"].to_numpy() * cos_t - df_pts["m_ycoord"].to_numpy() * sin_t)
                Y = y0 + (df_pts["m_xcoord"].to_numpy() * sin_t + df_pts["m_ycoord"].to_numpy() * cos_t)
                crsp = hec_crs
            else:
                X = df_pts["m_xcoord"].to_numpy()
                Y = df_pts["m_ycoord"].to_numpy()
                crsp = None
            gdfp = gpd.GeoDataFrame(
                {"particleid": df_pts["particleid"].astype(int).to_numpy(), "zelev": df_pts["zelev"].to_numpy()},
                geometry=[Point(x, y) for x, y in zip(X, Y)],
                crs=crsp
            )
            p = outdir / f"fig45_points_{name}_{label}.shp"
            gdfp.to_file(p, driver="ESRI Shapefile")
            return str(p)

        p_start = _pts(start_df, "start")
        p_end = _pts(end_df, "end")
        return str(p_lines), p_start, p_end

    def _write_top_raster_rotated_if_uniform():
        delr = np.asarray(dis.delr.array, dtype=float)
        delc = np.asarray(dis.delc.array, dtype=float)
        if not (_is_uniform(delr) and _is_uniform(delc)):
            return None

        dx = float(delr.flat[0])
        dy = float(delc.flat[0])
        angrot = float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0)
        theta = np.deg2rad(angrot)
        a = dx * np.cos(theta)
        b = -dy * np.sin(theta)
        d = dx * np.sin(theta)
        e = dy * np.cos(theta)
        x0 = coerce_to_float(getattr(dis, "xorigin", 0.0), 0.0)
        y0 = coerce_to_float(getattr(dis, "yorigin", 0.0), 0.0)
        transform = Affine(a, b, x0, d, e, y0)

        top = dis.top.array.astype("float32")
        nodata = -9999.0
        p = outdir / "top_surface_rotated.tif"
        with rasterio.open(
            p, "w",
            driver="GTiff",
            height=top.shape[0],
            width=top.shape[1],
            count=1,
            dtype="float32",
            crs=hec_crs,
            transform=transform,
            nodata=nodata,
            compress="lzw",
        ) as dst:
            dst.write(top, 1)
        return str(p)

    Xvw, Yvw = _world_vertices()
    delr_arr = np.asarray(dis.delr.array, dtype=float)
    delc_arr = np.asarray(dis.delc.array, dtype=float)
    Xvm, Yvm = _model_vertices(delr_arr, delc_arr)

    if write_world:
        paths["grid_cells_world"] = _write_cells_as_polygons(Xvw, Yvw, hec_crs, "world")
        paths["grid_walls_world"] = _write_outer_walls(Xvw, Yvw, hec_crs, "world")
        pl, ps, pe = _write_pathlines_points("world", use_world=True)
        paths["fig45_pathlines_world"] = pl
        paths["fig45_points_start_world"] = ps
        paths["fig45_points_end_world"] = pe

    if write_model_axes:
        paths["grid_cells_model_axes"] = _write_cells_as_polygons(Xvm, Yvm, None, "model_axes")
        paths["grid_walls_model_axes"] = _write_outer_walls(Xvm, Yvm, None, "model_axes")
        pl, ps, pe = _write_pathlines_points("model_axes", use_world=False)
        paths["fig45_pathlines_model_axes"] = pl
        paths["fig45_points_start_model_axes"] = ps
        paths["fig45_points_end_model_axes"] = pe

    topo_tif = _write_top_raster_rotated_if_uniform() if write_top_raster_if_uniform else None
    if topo_tif is None:
        paths["top_surface_polygons"] = paths.get("grid_cells_world")

    try:
        xmin = float(np.nanmin(Xvw)); xmax = float(np.nanmax(Xvw))
        ymin = float(np.nanmin(Yvw)); ymax = float(np.nanmax(Yvw))
        print(f"[Diagnostics/{direction}] World grid extent: [{xmin:.2f}, {ymin:.2f}] – [{xmax:.2f}, {ymax:.2f}]")
        print(f"[Diagnostics/{direction}] Model‑axis extent X: 0 – {np.sum(delr_arr):.2f}; Y: 0 – {np.sum(delc_arr):.2f}")
    except Exception:
        pass

    return paths


def _grid_footprint_from_vertices(mg) -> tuple[Polygon, gpd.GeoDataFrame, float, float]:
    Xv = mg.xvertices
    Yv = mg.yvertices
    ring = []
    ring += [(Xv[0, i], Yv[0, i]) for i in range(Xv.shape[1])]
    ring += [(Xv[j, -1], Yv[j, -1]) for j in range(1, Xv.shape[0])]
    ring += [(Xv[-1, i], Yv[-1, i]) for i in range(Xv.shape[1]-2, -1, -1)]
    ring += [(Xv[j, 0], Yv[j, 0]) for j in range(Xv.shape[0]-2, 0, -1)]
    poly = Polygon(ring)

    UL = (Xv[0, 0],    Yv[0, 0])
    UR = (Xv[0, -1],   Yv[0, -1])
    LR = (Xv[-1, -1],  Yv[-1, -1])
    LL = (Xv[-1, 0],   Yv[-1, 0])
    edges = [
        ("north", LineString([UL, UR])),
        ("east",  LineString([UR, LR])),
        ("south", LineString([LR, LL])),
        ("west",  LineString([LL, UL])),
    ]
    edges_gdf = gpd.GeoDataFrame({"name": [e[0] for e in edges], "geometry": [e[1] for e in edges]}, crs=mg.crs)
    width  = edges[2][1].length
    height = edges[3][1].length
    return poly, edges_gdf, width, height


# ----------------------------
# Orchestrator
# ----------------------------
# def scenario(
#     cfg, idomain: np.ndarray, chd_data: list[list[float]], river_cells: list[tuple[int, int, int, float]],
#     write: bool = True, run: bool = True, plot: bool = True, silent: bool = False
# ) -> tuple[flopy.mf6.MFSimulation, flopy.mf6.ModflowGwf]:
#     gwfsim, gwf = build_gwf_model(cfg, chd_data, idomain)
    
#     # Set additional simulation options
#     #gwfsim.set_all_data_external(binary=True)... this doesn't seem to work in current flopy version
#     gwfsim.simulation_data.auto_set_sizes = False #Shortcut
#     gwfsim.simulation_data.verify_data = False #Shortcut
#     gwfsim.simulation_data.lazy_io = True #Shortcut

#     if write:
#         write_models(gwfsim, silent=silent)
#     if run:
#         print("Running MODFLOW 6...")
#         run_models(gwfsim, silent=False)
#         print("Building MODPATH 7 forward")
#         mp_fwd, _mp_bwd = build_particle_models(cfg.sim_name, 
#                                                 gwf, river_cells, 
#                                                 mp7_ws=cfg.mp7_ws, 
#                                                 exe_path=cfg.md7_exe_path,
#                                                 nxy=3,
#                                                 nz_per_cell=3,
#                                                 include_stage_depth=True,
#                                                 layers_below=2,
#                                                 xy_margin=None,
#                                                 z_margin=0.05,
#                                                 z_scheme="stage_then_below")
#         if write:
#             write_models(mp_fwd, silent=silent)
#         print("Running MODPATH 7 forward...")
#         run_models(mp_fwd, silent=silent)
#     return gwfsim, gwf
def scenario(
    cfg, idomain: np.ndarray, chd_data: list[list[float]], river_cells: list[tuple[int, int, int, float]],
    write: bool = True, run: bool = True, plot: bool = True, silent: bool = False
) -> tuple[flopy.mf6.MFSimulation, flopy.mf6.ModflowGwf]:
    # Build GWF with Option B CHD split
    gwfsim, gwf = build_gwf_model(cfg, chd_data, idomain, river_cells=river_cells)

    # (optional) reduce Flopy overhead as you had
    gwfsim.simulation_data.auto_set_sizes = False
    gwfsim.simulation_data.verify_data = False
    gwfsim.simulation_data.lazy_io = True

    if write:
        write_models(gwfsim, silent=silent)
    if run:
        print("Running MODFLOW 6...")
        run_models(gwfsim, silent=False)

        print("Building MODPATH 7 forward")
        mp_fwd, _mp_bwd = build_particle_models(
            cfg.sim_name, gwf, river_cells, mp7_ws=cfg.mp7_ws, exe_path=cfg.md7_exe_path
        )
        if write:
            write_models(mp_fwd, silent=silent)
        print("Running MODPATH 7 forward...")
        run_models(mp_fwd, silent=silent)

    return gwfsim, gwf
