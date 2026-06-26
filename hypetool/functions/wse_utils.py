"""Derive a channel-only water-surface-elevation (WSE) raster from a (hydro-flattened) DEM.

3DEP DEMs are hydro-flattened: the wetted channel is rendered as a near-flat surface at the
water elevation. This detects those flat, connected regions (optionally restricted to the
model domain) and writes a WSE raster valid ONLY over the channel — restoring the original
tool's "WSE covers the wetted channel" semantics (vs. using the full DEM as WSE, which makes
every cell a constant-head/river cell).

Returns the output path, or None when no plausible channel is found — so the caller can fall
back to using the full DEM as the WSE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio


def derive_channel_wse(dem_path, out_path, *,
                       domain_gdf=None,
                       relief_thresh: float = 0.2,
                       window: int = 3,
                       min_area_frac: float = 0.002,
                       max_area_frac: float = 0.6,
                       log: Callable[[str], None] = print) -> Optional[str]:
    """
    Parameters
    ----------
    dem_path : the (hydro-flattened) DEM to read.
    out_path : where to write the channel-only WSE GeoTIFF.
    domain_gdf : optional polygon(s); detection is restricted to inside it.
    relief_thresh : a cell is "flat" if the local elevation range (over a `window`x`window`
        neighborhood, valid cells only) is below this (DEM units, e.g. metres).
    min_area_frac / max_area_frac : sanity bounds on the detected channel as a fraction of
        valid domain cells; outside these we give up (return None) and the caller uses the DEM.
    """
    from scipy.ndimage import label, maximum_filter, minimum_filter
    from rasterio.features import geometry_mask

    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        profile = src.profile

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= ~np.isclose(arr, nodata)
    valid &= arr > -1.0e20

    if domain_gdf is not None and len(domain_gdf):
        try:
            dom = domain_gdf.to_crs(crs) if domain_gdf.crs is not None else domain_gdf
            inside = geometry_mask(list(dom.geometry), out_shape=arr.shape,
                                   transform=transform, invert=True)
            valid &= inside
        except Exception as e:  # noqa: BLE001
            log(f"[WARN] channel-WSE: could not rasterize domain ({e}); using full DEM extent.")

    n_valid = int(valid.sum())
    if n_valid == 0:
        log("[WARN] channel-WSE: no valid DEM pixels; falling back to DEM-as-WSE.")
        return None

    # Local elevation range over valid cells only (invalid -> -inf in hi / +inf in lo, so
    # they never widen the range; a window with no valid cells yields a non-finite relief).
    hi = np.where(valid, arr, -np.inf)
    lo = np.where(valid, arr, np.inf)
    relief = maximum_filter(hi, size=window) - minimum_filter(lo, size=window)
    flat = valid & np.isfinite(relief) & (relief < float(relief_thresh))
    if not flat.any():
        log(f"[WARN] channel-WSE: no flat (<{relief_thresh} m) cells; falling back to DEM-as-WSE.")
        return None

    # Largest connected flat component = the main channel.
    lbl, n = label(flat)
    if n == 0:
        return None
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    channel = lbl == int(sizes.argmax())
    n_channel = int(channel.sum())
    frac = n_channel / n_valid

    if frac > max_area_frac:
        log(f"[WARN] channel-WSE: flat region is {frac:.0%} of the domain (relief threshold "
            f"too loose); falling back to DEM-as-WSE.")
        return None
    if n_channel < max(4, int(min_area_frac * n_valid)):
        log(f"[WARN] channel-WSE: detected channel too small ({n_channel} px); "
            f"falling back to DEM-as-WSE.")
        return None

    fill = np.float32(-9999.0)
    out = np.where(channel, arr, fill).astype("float32")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=float(fill))
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    log(f"[OK] channel-WSE: {n_channel} channel px ({frac:.1%} of domain) -> {out_path.name}")
    return str(out_path)
