from __future__ import annotations

"""Reusable helpers for raster↔grid operations used across the Hyporheic project.

The functions here are intentionally **side‑effect‑free**: they return data
rather than mutate the global ``cfg`` object.  Your *__main__.py* (or the
pipeline step module) can then decide what to do with those results.
"""

from pathlib import Path
from typing import Tuple, Any

import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely.geometry import box, Point
import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
from numpy.ma import MaskedArray
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from typing import Any, Literal

__all__ = [
    "load_raster",
    "mask_nodata",
    "raster_extent",
    "grid_dimensions",
    "generate_grid_centres",
    "grid_to_geodataframe",
    "interpolate_na",
]

# ---------------------------------------------------------------------------
# 1. Raster loading / masking
# ---------------------------------------------------------------------------

def load_raster(path: str | Path):
    """Open *path* and return (array, transform, crs, nodata, bounds_box)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        array = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds_box = box(*src.bounds)

    return array, transform, crs, nodata, bounds_box

def mask_nodata(array: np.ndarray, nodata: float | int | None):
    """Return a NumPy *masked array* with *nodata* values masked out."""
    if nodata is None:
        return np.ma.array(array)  # nothing to mask
    return np.ma.masked_equal(array, nodata)

def interpolate_na(terrain: MaskedArray, *, method: str = "nearest") -> MaskedArray:
    """Fill *NA* (masked) cells in a masked raster array via spatial interpolation.

    Parameters
    ----------
    terrain : numpy.ma.MaskedArray
        Two‑dimensional masked array representing the raster. Cells marked as
        masked (``terrain.mask``) are treated as missing values to be
        interpolated.
    method : str, optional
        Interpolation method passed to :pyfunc:`scipy.interpolate.griddata`.
        Accepts ``'nearest'`` (default), ``'linear'`` or ``'cubic'``.

    Returns
    -------
    numpy.ma.MaskedArray
        A *new* masked array where previously masked cells have been replaced
        by interpolated values and the mask cleared (all values now valid).
    """

    # If there is no mask, just return the array unchanged
    if terrain.mask is np.ma.nomask:
        return terrain

    # 1. Collect coordinates & values for valid pixels
    valid_mask = ~terrain.mask
    valid_coords = np.column_stack(np.nonzero(valid_mask))
    valid_values = terrain[valid_mask]

    # 2. Collect coordinates for invalid pixels (to be filled)
    invalid_coords = np.column_stack(np.nonzero(terrain.mask))

    # 3. Interpolate values at invalid pixel locations
    interpolated_values = griddata(valid_coords, valid_values, invalid_coords, method=method)

    # 4. Build a *copy* so the original array isn't modified in‑place
    filled = terrain.copy()
    filled[terrain.mask] = interpolated_values
    filled.mask = np.zeros_like(filled, dtype=bool)  # clear mask

    return filled

# ---------------------------------------------------------------------------
# 2. Grid helpers
# ---------------------------------------------------------------------------

def raster_extent(transform: Affine, width: int, height: int):
    """Compute (xmin, ymin, xmax, ymax) from *transform*, *width*, *height*."""
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + width * transform.a
    ymin = ymax + height * transform.e  # transform.e is negative (pixel size in Y)
    return xmin, ymin, xmax, ymax


def grid_dimensions(width_ft: float, height_ft: float, dx: float, dy: float):
    """Return (ncol, nrow) given raster *width/height* and cell sizes *dx*, *dy*."""
    ncol = int(width_ft / dx)
    nrow = int(height_ft / dy)
    return ncol, nrow


def generate_grid_centres(
    ncol: int,
    nrow: int,
    dx: float,
    dy: float,
    xmin: float,
    ymin: float,
    *,
    origin: Literal["lower", "upper"] = "upper",   # ← new keyword
):
    """
    Return (grid_x, grid_y) 2‑D arrays with **cell‑centre** coordinates.

    Parameters
    ----------
    origin : {"upper", "lower"}, default "upper"
        "upper"  → row 0 is the *top* of the grid (common GIS convention)  
        "lower"  → row 0 is the *bottom* (the old behaviour)
    """
    # --- x direction is the same either way ---------------------------------
    x = np.arange(ncol) * dx + (dx / 2) + xmin        # shape (ncol,)

    # --- y axis depends on origin -------------------------------------------
    if origin == "upper":
        ymax = ymin + nrow * dy                       # topmost edge
        y = ymax - (np.arange(nrow) * dy + dy / 2)    # flip: top → bottom
    else:                                             # "lower" (old)
        y = np.arange(nrow) * dy + (dy / 2) + ymin

    return np.meshgrid(x, y)                          # each shape (nrow, ncol)


def grid_to_geodataframe(grid_x: np.ndarray,
                         grid_y: np.ndarray,
                         crs: Any) -> gpd.GeoDataFrame:
    """Convert centre arrays to a GeoDataFrame of Points."""
    points = [Point(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
    return gpd.GeoDataFrame({"geometry": points}, crs=crs)



