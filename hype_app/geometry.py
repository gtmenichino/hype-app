"""Drawn-geometry handling: ipyleaflet DrawControl GeoJSON -> projected GeoDataFrames.

The map hands us EPSG:4326 features. The MODFLOW grid math works in the projected CRS's
linear units, so we reproject to a UTM zone (metres) chosen from the domain centroid; the
model therefore runs in metres (length_units='meters').
"""
from __future__ import annotations

from math import hypot
from typing import Iterable, Optional

import geopandas as gpd
from shapely.geometry import LineString, Polygon, mapping, shape


def features_to_gdf(features: Iterable[dict], crs="EPSG:4326") -> gpd.GeoDataFrame:
    """A list of GeoJSON Feature dicts -> GeoDataFrame in EPSG:4326."""
    geoms = [shape(f["geometry"]) for f in features if f and f.get("geometry")]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def single_feature_gdf(feature: dict, crs="EPSG:4326") -> gpd.GeoDataFrame:
    """A single GeoJSON Feature dict -> 1-row GeoDataFrame in EPSG:4326."""
    return gpd.GeoDataFrame(geometry=[shape(feature["geometry"])], crs=crs)


def pick_projected_crs(domain_gdf_4326: gpd.GeoDataFrame):
    """UTM (metre) CRS appropriate for the domain centroid. The model works in metres."""
    return domain_gdf_4326.estimate_utm_crs()


def _feat(geom) -> dict:
    """Wrap a shapely geometry as a GeoJSON Feature dict (matches delineate._feat)."""
    return {"type": "Feature", "properties": {}, "geometry": mapping(geom)}


def _coords_of(feature) -> list:
    """First LineString's (x, y) coords from a Feature / geometry / FeatureCollection."""
    g = feature
    if isinstance(g, dict) and g.get("type") == "FeatureCollection":
        g = (g.get("features") or [{}])[0].get("geometry") or {}
    elif isinstance(g, dict) and g.get("type") == "Feature":
        g = g.get("geometry") or {}
    coords = list((g or {}).get("coordinates") or [])
    if coords and isinstance(coords[0][0], (list, tuple)):   # MultiLineString → first part
        coords = list(coords[0])
    return [tuple(c[:2]) for c in coords]


def assemble_domain_from_sides(up, left, right, down) -> Optional[dict]:
    """Stitch four boundary LineString Features — Upstream, Left FPL, Right FPL, Downstream — into a
    closed domain Polygon, snapping the four shared corners.

    Returns ``{"domain", "left", "right", "up", "down"}`` of GeoJSON Features with **left/right
    oriented upstream→downstream and up/down oriented left→right** (the orientation the engine's
    gradient interpolation expects), or ``None`` if any side is missing or the ring can't be built.
    Works in lon/lat (EPSG:4326); corners are matched by nearest endpoints. This is the inverse of
    ``delineate._sides_from_ring``.
    """
    coords = {}
    for k, f in (("up", up), ("left", left), ("right", right), ("down", down)):
        c = _coords_of(f) if f else []
        if len(c) < 2:
            return None
        coords[k] = c

    def _ends(k):
        return (coords[k][0], coords[k][-1])

    def _corner(a_ends, b_ends):
        """Mean of the closest endpoint pair between two sides (the snapped shared corner)."""
        best = None
        for pa in a_ends:
            for pb in b_ends:
                d = hypot(pa[0] - pb[0], pa[1] - pb[1])
                if best is None or d < best[0]:
                    best = (d, pa, pb)
        _, pa, pb = best
        return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)

    ul = _corner(_ends("up"), _ends("left"))      # upstream-left
    ur = _corner(_ends("up"), _ends("right"))     # upstream-right
    dl = _corner(_ends("down"), _ends("left"))    # downstream-left
    dr = _corner(_ends("down"), _ends("right"))   # downstream-right

    def _oriented(k, start, end):
        """Side k as coords running start→end (flip if stored backwards), with both endpoints
        replaced by the snapped corners."""
        cs = list(coords[k])
        d0 = hypot(cs[0][0] - start[0], cs[0][1] - start[1])
        d1 = hypot(cs[-1][0] - start[0], cs[-1][1] - start[1])
        if d0 > d1:
            cs = cs[::-1]
        cs[0], cs[-1] = start, end
        return cs

    # Walk the ring: up(UL→UR) → right(UR→DR) → down(DR→DL) → left(DL→UL), dropping each repeated corner.
    ring = list(_oriented("up", ul, ur))
    for seg in (_oriented("right", ur, dr), _oriented("down", dr, dl), _oriented("left", dl, ul)):
        ring.extend(seg[1:])
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon" or poly.area <= 0:
            return None
    except Exception:  # noqa: BLE001
        return None

    return {
        "domain": _feat(poly),
        "left": _feat(LineString(_oriented("left", ul, dl))),   # upstream→downstream
        "right": _feat(LineString(_oriented("right", ur, dr))),  # upstream→downstream
        "up": _feat(LineString(_oriented("up", ul, ur))),        # left→right
        "down": _feat(LineString(_oriented("down", dl, dr))),    # left→right
    }
