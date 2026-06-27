"""Pre-run grid-size estimate + guardrail bands.

The MODFLOW grid is derived from the terrain-raster extent, and we fetch/clip the DEM to
the domain bbox + the same buffer (see hype_app.dem.BUFFER_FRAC), so estimating from the
buffered domain bbox closely matches the real grid the engine builds.
"""
from __future__ import annotations

import math
import os

from .dem import BUFFER_FRAC

# Cell-count guardrail bands. Anchored to observed runs (≈1.65M cells solved in ~4 s locally,
# ~2.67M crashed mid-setup) and the 8 GB Connect Cloud target (don't max it out). Override per
# environment with HYPE_GREEN_CELLS / HYPE_MAX_CELLS — the local dev preview sets a lower
# HYPE_MAX_CELLS so testing is bounded by what this machine reliably runs.
GREEN_MAX = int(os.environ.get("HYPE_GREEN_CELLS", 1_500_000))   # < this: fast (≈ proven 1.65M)
AMBER_MAX = int(os.environ.get("HYPE_MAX_CELLS", 4_000_000))     # < this: allowed (warn); >= this: blocked


def estimate_cells(domain_gdf_proj, cell_size: float, gw_mod_depth: float, z: float,
                   buffer_frac: float = BUFFER_FRAC) -> dict:
    """Estimate ncol*nrow*nlay from the (buffered) projected domain bbox."""
    minx, miny, maxx, maxy = (float(v) for v in domain_gdf_proj.total_bounds)
    dx, dy = (maxx - minx) * buffer_frac, (maxy - miny) * buffer_frac
    w = (maxx - minx) + 2 * dx
    h = (maxy - miny) + 2 * dy
    ncol = max(1, math.ceil(w / float(cell_size)))
    nrow = max(1, math.ceil(h / float(cell_size)))
    nlay = max(1, math.ceil(float(gw_mod_depth) / float(z)))
    return {"ncol": ncol, "nrow": nrow, "nlay": nlay, "n_cells": ncol * nrow * nlay,
            "cell_size": float(cell_size),
            "dom_w": maxx - minx, "dom_h": maxy - miny}   # raw (unbuffered) domain footprint, m


def band(n_cells: int) -> str:
    """'green' | 'amber' | 'red' guardrail band for a cell count."""
    if n_cells < GREEN_MAX:
        return "green"
    if n_cells < AMBER_MAX:
        return "amber"
    return "red"


def band_message(est: dict) -> str:
    b = band(est["n_cells"])
    grid = f'{est["ncol"]}×{est["nrow"]}×{est["nlay"]} = {est["n_cells"]:,} cells'
    cs = est.get("cell_size")
    # Cell count scales ~1/cell_size² (layers fixed); suggest a size that lands in the green band.
    sugg = math.ceil(cs * math.sqrt(est["n_cells"] / GREEN_MAX)) if cs else None
    tip = f"For a fast run, try ~{sugg} m cells" if sugg else "Try a coarser cell size"
    if b == "green":
        return f"Grid ≈ {grid}. Good to run."
    if b == "amber":
        return (f"Grid ≈ {grid}. Large — slower and more memory-hungry. "
                f"{tip}, or increase the layer thickness.")
    return (f"Grid ≈ {grid}. Too large — over the {AMBER_MAX:,}-cell limit. "
            f"{tip}, reduce the model depth, or increase the layer thickness.")
