"""Auto-delineate the GW domain, floodplain boundary lines, and wetted-extent polygon
from a reach line + DEM, sized by Bieger bankfull geometry.

The floodplain is measured at only TWO cross-sections — one at the upstream end (s=0) and
one at the downstream end (s=L). On each, the DEM is sampled across a line perpendicular to
the reach: find the thalweg (min elevation, anchored to a channel window so it can't snap to
a far tributary), then walk outward each side to the first point where the DEM rises to a
target elevation (thalweg + X * bankfull_depth for the domain; + 1 * bankfull_depth for the
wetted channel). The resulting left/right offsets are then linearly INTERPOLATED along the
reach, with each intermediate cross-section kept perpendicular to the reach — so the ribbon
follows the channel curve with a smoothly varying width (no per-transect jumping/self-cross).
Connecting the interpolated edge points yields the domain ribbon, the left/right boundary
lines, and the wetted-extent polygon.
"""
from __future__ import annotations

from typing import Optional

CRS_ALBERS = 5070  # work in metres (USGS CONUS Albers)


def _normal(line, s, ds=5.0):
    """Unit vector perpendicular to `line` at station `s` (metres), via a centred
    difference so it's stable at the endpoints. Returns (point, (nx, ny))."""
    L = line.length
    s0 = max(0.0, min(float(s), L))
    a = line.interpolate(max(0.0, s0 - ds))
    b = line.interpolate(min(L, s0 + ds))
    dx, dy = b.x - a.x, b.y - a.y
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    return line.interpolate(s0), (-dy / n, dx / n)


def _edge_offsets(dem, line, s, *, half, n_samp, rel_height, depth_bf, chan_half):
    """At station `s`: sample the DEM across the perpendicular transect and return the signed
    offsets ``(lo, ro)`` along the unit normal (``lo <= 0`` left, ``ro >= 0`` right) where the
    surface first rises ``rel_height * depth_bf`` above the thalweg on each side. The thalweg
    search is anchored to the channel window ``|t| <= chan_half`` so it can't snap to a far-off
    tributary; if the rise isn't reached on a side, fall back to that side's local elevation-max
    (the valley shoulder) rather than the far ±half edge. Returns ``(lo, ro)`` or ``None``."""
    import numpy as np
    import xarray as xr

    p, (nx, ny) = _normal(line, s)
    ts = np.linspace(-half, half, n_samp)
    zx = p.x + nx * ts
    zy = p.y + ny * ts
    z = np.asarray(dem.interp(x=xr.DataArray(zx, dims="t"),
                              y=xr.DataArray(zy, dims="t")).values, dtype=float)
    ok = np.isfinite(z)
    if ok.sum() < 5:
        return None
    vi = np.where(ok)[0]
    centre = ok & (np.abs(ts) <= chan_half)
    pool = np.where(centre)[0] if centre.any() else vi
    k = int(pool[int(np.argmin(z[pool]))])             # thalweg, anchored to the channel window
    thresh = z[k] + float(rel_height) * float(depth_bf)
    li = next((i for i in range(k, -1, -1) if ok[i] and z[i] >= thresh), None)
    ri = next((i for i in range(k, len(ts)) if ok[i] and z[i] >= thresh), None)
    if li is None:                                     # shoulder fallback: highest ground left of k
        seg = [i for i in range(0, k + 1) if ok[i]]
        li = int(seg[int(np.argmax(z[seg]))]) if seg else int(vi[0])
    if ri is None:                                     # highest ground right of k
        seg = [i for i in range(k, len(ts)) if ok[i]]
        ri = int(seg[int(np.argmax(z[seg]))]) if seg else int(vi[-1])
    if li == ri:                                       # degenerate: nudge a couple samples
        li = max(int(vi[0]), k - 2); ri = min(int(vi[-1]), k + 2)
        if li == ri:
            return None
    return float(ts[li]), float(ts[ri])


def _interp_sides(dem, line, L, stations, *, rel_height, depth_bf, half, n_samp, chan_half):
    """Measure the floodplain offsets only at the upstream (s=0) and downstream (s=L) cross-
    sections, then build per-station edge points by linearly interpolating the left/right
    offsets along the reach and placing them perpendicular to it — so the ribbon follows the
    channel curve with a smoothly varying width (no per-transect jumping). Returns
    ``(dleft, dright)`` coordinate lists, or ``None`` if neither end could be sampled."""
    def _end(s):
        e = _edge_offsets(dem, line, s, half=half, n_samp=n_samp, rel_height=rel_height,
                          depth_bf=depth_bf, chan_half=chan_half)
        if e is None and L > 0:                        # retry ~2% in from the reach end
            s2 = min(max(s + (0.02 * L if s < L / 2.0 else -0.02 * L), 0.0), L)
            e = _edge_offsets(dem, line, s2, half=half, n_samp=n_samp, rel_height=rel_height,
                              depth_bf=depth_bf, chan_half=chan_half)
        return e

    up = _end(0.0)
    dn = _end(L)
    if up is None and dn is None:
        return None
    up = up or dn                                      # if one end failed, reuse the other
    dn = dn or up
    lo_u, ro_u = up
    lo_d, ro_d = dn
    dleft, dright = [], []
    for s in stations:
        f = (s / L) if L > 0 else 0.0
        lo = lo_u + f * (lo_d - lo_u)
        ro = ro_u + f * (ro_d - ro_u)
        p, (nx, ny) = _normal(line, s)
        dleft.append((p.x + nx * lo, p.y + ny * lo))
        dright.append((p.x + nx * ro, p.y + ny * ro))
    return dleft, dright


def _ribbon(dleft, dright):
    """A single valid simple Polygon from left+right side coordinates. Uses the direct ring when
    the sides don't cross (the common case); otherwise unions per-panel quads — which provably
    contain both side-lines, so the boundary lines can never fall outside the domain — and keeps
    the largest piece. Returns the Polygon or ``None``."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    try:
        from shapely import make_valid
    except Exception:  # noqa: BLE001
        make_valid = None

    n = min(len(dleft), len(dright))
    if n < 2:
        return None
    ring = Polygon(dleft + dright[::-1])
    if ring.is_valid and ring.is_simple and ring.area > 0:
        return ring
    quads = []
    for i in range(n - 1):
        q = Polygon([dleft[i], dleft[i + 1], dright[i + 1], dright[i]])
        q = make_valid(q) if make_valid is not None else q.buffer(0)
        if (not q.is_empty) and q.area > 0:
            quads.append(q.buffer(0))
    if not quads:
        return None
    merged = unary_union(quads)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    merged = merged.buffer(0)
    return merged if (not merged.is_empty and merged.area > 0) else None


def _resample_line(line, spacing):
    """Re-distribute the centerline's vertices at even `spacing` (m) so the perpendiculars don't
    wobble on jagged input — same idea as RAS's fixed along-channel resampling."""
    import numpy as np
    from shapely.geometry import LineString
    L = float(line.length)
    if L <= 0:
        return line
    n = max(2, int(L / max(spacing, 1.0)) + 1)
    pts = [line.interpolate(float(s)) for s in np.linspace(0.0, L, n)]
    return LineString([(p.x, p.y) for p in pts])


def _sides_from_ring(domain, up_left, up_right, dn_left, dn_right):
    """Derive the left/right boundary lines + flat upstream/downstream caps as slices of the clean
    domain boundary ring — the shapely equivalent of RAS's SnapToRing + SubRing, so the four sides
    are simple, lie exactly on the domain, and meet at the caps (instead of using the raw offset
    points, which can self-cross). Returns ``(left, right, up_cap, dn_cap)`` LineStrings (left/right
    oriented upstream→downstream), or ``None`` if the four corners aren't in a clean cyclic order
    (the caller then falls back to the raw offset lines)."""
    from shapely.geometry import LineString, Point
    from shapely.ops import substring
    try:
        ring = LineString(domain.exterior.coords)
        total = float(ring.length)
        if total <= 0:
            return None
        corners = {"ul": tuple(up_left), "ur": tuple(up_right),
                   "dl": tuple(dn_left), "dr": tuple(dn_right)}
        pos = {k: float(ring.project(Point(v))) for k, v in corners.items()}
        order = sorted(corners, key=lambda k: pos[k])
        arc_by_pair = {}
        for i in range(4):
            a, b = order[i], order[(i + 1) % 4]
            if i < 3:
                seg = substring(ring, pos[a], pos[b])
            else:                                          # last arc wraps past the closure point
                s1 = substring(ring, pos[a], total)
                s2 = substring(ring, 0.0, pos[b])
                seg = LineString(list(s1.coords) + list(s2.coords)[1:])
            arc_by_pair[frozenset((a, b))] = seg
        need = [("ul", "dl"), ("dl", "dr"), ("dr", "ur"), ("ur", "ul")]
        if any(frozenset(p) not in arc_by_pair for p in need):
            return None                                    # corners not in a clean cyclic order

        def _oriented(start_key, end_key):
            seg = arc_by_pair[frozenset((start_key, end_key))]
            sp = ring.interpolate(pos[start_key])
            if seg is None or seg.is_empty or len(seg.coords) < 2:
                return LineString([sp, ring.interpolate(pos[end_key])])
            cs = list(seg.coords)
            if Point(cs[0]).distance(sp) > Point(cs[-1]).distance(sp):
                cs = cs[::-1]                              # orient start → end
            return LineString(cs)

        sides = (_oriented("ul", "dl"), _oriented("ur", "dr"),
                 _oriented("ul", "ur"), _oriented("dl", "dr"))
        for g in sides:
            if not (g.is_simple and g.length > 0):
                return None
            if max(Point(c).distance(ring) for c in g.coords) > 5.0:
                return None             # a side degenerated to an off-boundary chord (corner swallowed
            #                             inside the union on a very tight curve) — let the caller fall back
        return sides
    except Exception:  # noqa: BLE001
        return None


def _line_coords(reach_geojson) -> list:
    """Pull the first LineString's coordinates from a FeatureCollection / Feature / geometry."""
    g = reach_geojson
    if isinstance(g, dict) and g.get("type") == "FeatureCollection":
        g = (g.get("features") or [{}])[0].get("geometry") or {}
    elif isinstance(g, dict) and g.get("type") == "Feature":
        g = g.get("geometry") or {}
    coords = (g or {}).get("coordinates") or []
    if coords and isinstance(coords[0][0], (list, tuple)):  # MultiLineString → first part
        coords = coords[0]
    return coords


def _feat(geom):
    from shapely.geometry import mapping
    return {"type": "Feature", "properties": {}, "geometry": mapping(geom)}


SIMPLIFY_TOL_M = 2.0  # metres; Douglas-Peucker tolerance. << the 10 m model cells, so the
#                       generated linework keeps its shape while shedding redundant vertices.


def _simplify(geom, tol=SIMPLIFY_TOL_M):
    """Drop duplicate + near-collinear vertices (Douglas-Peucker; first/last point always kept).
    preserve_topology keeps polygons valid / non-self-intersecting; buffer(0) is a safety net."""
    if geom is None or geom.is_empty:
        return geom
    s = geom.simplify(tol, preserve_topology=True)
    if s is None or s.is_empty:
        return geom                                    # never collapse to nothing
    if s.geom_type in ("Polygon", "MultiPolygon"):
        s = s.buffer(0)
        if s.geom_type == "MultiPolygon" and not s.is_empty:
            s = max(s.geoms, key=lambda g: g.area)     # keep the largest piece (single Polygon)
    return s if (s is not None and not s.is_empty) else geom


def _nverts(geom):
    """Vertex count of a Polygon ring / LineString (for the delineation readout)."""
    if geom is None or getattr(geom, "is_empty", True):
        return 0
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords)
    if geom.geom_type == "LineString":
        return len(geom.coords)
    return 0


def auto_delineate(reach_geojson, dem_path, *, da_sqkm, lat=None, lon=None,
                   x_mult=2.0, n_domain=10, log=print) -> dict:
    """Build {domain, left, right, wse_extent} GeoJSON Features (EPSG:4326) + meta."""
    import geopandas as gpd
    import numpy as np
    import rioxarray  # noqa: F401 — .rio accessor
    from shapely.geometry import LineString

    from . import bieger

    coords = _line_coords(reach_geojson)
    if len(coords) < 2:
        raise ValueError("Reach line has too few vertices to delineate.")
    line = gpd.GeoSeries([LineString(coords)], crs=4326).to_crs(CRS_ALBERS).iloc[0]
    L = float(line.length)
    if L < 5.0:
        raise ValueError("Reach is too short to delineate.")
    line = _resample_line(line, max(5.0, L / 40.0))     # even vertices → stable perpendiculars

    bf = bieger.bankfull_geometry(da_sqkm, lat, lon)
    depth_bf = max(float(bf["depth_m"]), 0.05)
    w_bf = max(float(bf["width_m"]), 1.0)
    half = min(max(8.0 * w_bf, 250.0), 800.0)          # search half-width (m); matches DEM buffer
    n_samp = int(2 * half / 5.0) + 1                    # ~5 m spacing across the transect
    chan_half = min(half, max(4.0 * w_bf, 100.0))       # channel window for the thalweg anchor

    dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze().rio.reproject(CRS_ALBERS)

    # --- domain + boundaries: measure floodplain offsets at the two end cross-sections only,
    #     then interpolate the left/right widths along the reach (perpendicular at each station). ---
    n_dom = max(12, min(60, int(L / 20.0)))
    dom_stations = list(np.linspace(0.0, L, n_dom))
    sides = _interp_sides(dem, line, L, dom_stations, rel_height=float(x_mult),
                          depth_bf=depth_bf, half=half, n_samp=n_samp, chan_half=chan_half)
    if sides is None or len(sides[0]) < 3:
        raise ValueError("Could not sample valid cross-sections at the reach ends (DEM gaps?).")
    dleft, dright = sides
    domain = _simplify(_ribbon(dleft, dright))
    if domain is None or domain.is_empty or domain.area <= 0:
        raise ValueError("Delineated domain is degenerate; try a different reach or X.")
    # Derive the left/right boundaries + flat upstream/downstream caps as slices of the clean
    # domain ring (RAS ChannelTopologyBuilder approach) — simple, on the domain, and meeting at
    # the caps — instead of the raw offset points, which self-cross when the floodplain is wide.
    split = _sides_from_ring(domain, dleft[0], dright[0], dleft[-1], dright[-1])
    if split is not None:
        left_line, right_line, up_cap, down_cap = split
    else:                                               # fallback: raw offset lines + straight caps
        left_line = _simplify(LineString(dleft))
        right_line = _simplify(LineString(dright))
        up_cap = LineString([dleft[0], dright[0]])
        down_cap = LineString([dleft[-1], dright[-1]])

    # --- wetted extent: same two-XS interpolation at the bankfull-channel threshold ---
    n_wse = max(20, min(80, int(L / 15.0)))
    wsides = _interp_sides(dem, line, L, list(np.linspace(0.0, L, n_wse)), rel_height=1.0,
                           depth_bf=depth_bf, half=half, n_samp=n_samp, chan_half=chan_half)
    wse = None
    if wsides is not None and len(wsides[0]) >= 3:
        wse = _simplify(_ribbon(wsides[0], wsides[1]))
        if wse is None or wse.is_empty or wse.area <= 0:
            wse = None

    def to4326(geom):
        return gpd.GeoSeries([geom], crs=CRS_ALBERS).to_crs(4326).iloc[0]

    def to4326_poly(geom):
        """Reproject + guarantee a single valid Polygon (reprojection can introduce a self-touch
        on a thin ribbon, and very short reaches with huge floodplains can still fold). Keeps the
        largest polygonal piece of whatever make_valid returns (Polygon/Multi/GeometryCollection)."""
        g = to4326(geom)
        if not g.is_valid:
            try:
                from shapely import make_valid
                g = make_valid(g)
            except Exception:  # noqa: BLE001
                g = g.buffer(0)
        polys = []
        for part in (g.geoms if hasattr(g, "geoms") else [g]):
            if part.geom_type == "Polygon":
                polys.append(part)
            elif part.geom_type == "MultiPolygon":
                polys.extend(part.geoms)
        if polys:
            g = max(polys, key=lambda p: p.area)
        return g

    out = {
        "domain": _feat(to4326_poly(domain)),
        "left": _feat(to4326(left_line)),
        "right": _feat(to4326(right_line)),
        "up_cap": _feat(to4326(up_cap)),
        "down_cap": _feat(to4326(down_cap)),
        "wse_extent": _feat(to4326_poly(wse)) if wse is not None else None,
        "meta": {
            "da_sqkm": round(float(da_sqkm or 0.0), 3),
            "bankfull_depth_m": depth_bf, "bankfull_width_m": w_bf,
            "division": bf["division_name"], "x_mult": float(x_mult),
            "reach_len_m": round(L, 1), "n_domain_xs": len(dleft),
            "wse_vertices": _nverts(wse), "boundary_vertices": _nverts(left_line),
        },
    }
    log(f"Delineated: {len(dleft)} domain XS, reach {L:.0f} m, "
        f"bankfull depth {depth_bf:.2f} m ({bf['division_name']}), X={x_mult}.")
    return out


def cross_section_lines(reach_geojson, dem_path, *, da_sqkm, lat=None, lon=None,
                        x_mult=2.0, n=10) -> Optional[dict]:
    """The domain cross-section transects as a GeoJSON FeatureCollection (for display)."""
    import geopandas as gpd
    import numpy as np
    import rioxarray  # noqa: F401
    from shapely.geometry import LineString

    from . import bieger

    coords = _line_coords(reach_geojson)
    if len(coords) < 2:
        return None
    line = gpd.GeoSeries([LineString(coords)], crs=4326).to_crs(CRS_ALBERS).iloc[0]
    L = float(line.length)
    bf = bieger.bankfull_geometry(da_sqkm, lat, lon)
    depth_bf = max(float(bf["depth_m"]), 0.05)
    w_bf = max(float(bf["width_m"]), 1.0)
    half = min(max(8.0 * w_bf, 250.0), 800.0)
    n_samp = int(2 * half / 5.0) + 1
    chan_half = min(half, max(4.0 * w_bf, 100.0))
    dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze().rio.reproject(CRS_ALBERS)
    sides = _interp_sides(dem, line, L, list(np.linspace(0.0, L, max(3, int(n)))),
                          rel_height=float(x_mult), depth_bf=depth_bf, half=half,
                          n_samp=n_samp, chan_half=chan_half)
    if sides is None:
        return None
    dleft, dright = sides
    segs = [LineString([dleft[i], dright[i]]) for i in range(len(dleft))]
    if not segs:
        return None
    gj = gpd.GeoSeries(segs, crs=CRS_ALBERS).to_crs(4326)
    return {"type": "FeatureCollection",
            "features": [_feat(g) for g in gj.geometry]}
