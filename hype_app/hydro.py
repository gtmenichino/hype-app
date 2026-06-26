"""NHD hydrography helpers via the USGS **NHDPlus HR ArcGIS REST** service
(`hydro.nationalmap.gov`) — the same host as the NHD basemap tiles, so it resolves
wherever the app runs (incl. the restricted preview sandbox, unlike pynhd's
`api.water.usgs.gov`). One service gives flowline geometry + total drainage area
(`totdasqkm`), so we fetch, snap, read drainage area, and trace the reach without NLDI.
Every call fails soft (returns None / raises a clear ValueError).
"""
from __future__ import annotations

import functools
import json
import urllib.parse
import urllib.request
from typing import Optional

CRS_WGS84 = 4326
CRS_ALBERS = 5070            # USGS CONUS Albers, metres
FT_PER_M = 3.280839895
MAX_REACH_M = 1609.344       # 1 mile

# NHDPlus HR — layer 3 = NetworkNHDFlowline (the connected stream network).
_FLOW_URL = ("https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/"
             "MapServer/3/query")
_OUT_FIELDS = "nhdplusid,totdasqkm,gnis_name,lengthkm"


_FLOW_CACHE: dict = {}        # success-only cache (don't cache transient failures)


def _fetch(w: float, s: float, e: float, n: float):
    """NHDPlus-HR flowline query for a (rounded) bbox → GeoJSON FeatureCollection (EPSG:4326,
    LineStrings with nhdplusid/totdasqkm) or None. Geometry is simplified server-side for a fast,
    small response; retries a few times because the service is occasionally slow. Caches only
    successes so a transient timeout doesn't poison the result."""
    key = (w, s, e, n)
    if key in _FLOW_CACHE:
        return _FLOW_CACHE[key]
    params = {
        "where": "1=1",
        "geometry": f"{w},{s},{e},{n}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": _OUT_FIELDS,
        "returnGeometry": "true",
        "maxAllowableOffset": "0.0001",      # ~11 m: simplify server-side → faster/smaller
        "resultRecordCount": "2000",
        "f": "geojson",
    }
    url = _FLOW_URL + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            feats = (data or {}).get("features") or []
            print(f"[flowlines] {len(feats)} flowlines for {key} (try {attempt + 1})", flush=True)
            if feats:
                _FLOW_CACHE[key] = data
            return data if feats else None
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[flowlines] attempt {attempt + 1} failed for {key}: {exc!r}", flush=True)
    print(f"[flowlines] gave up for {key}: {last!r}", flush=True)
    return None


def flowlines_bbox(w: float, s: float, e: float, n: float, *, max_area_deg2: float = 1.0):
    """GeoJSON FeatureCollection of NHD flowlines in the bbox (map layer + snapping), or None
    if the view is too large (guard) or empty."""
    if e <= w or n <= s or (e - w) * (n - s) > max_area_deg2:
        return None
    return _fetch(round(w, 4), round(s, 4), round(e, 4), round(n, 4))


def _prop(row, names):
    for c in names:
        try:
            if c in row.index and row.get(c) is not None:
                return row[c]
        except Exception:  # noqa: BLE001
            continue
    return None


def snap(lat: float, lon: float, flowlines_gdf=None) -> Optional[dict]:
    """Snap (lat, lon) to the nearest NHD flowline → {lat, lon, comid, da_sqkm, dist_ft}.
    Uses `flowlines_gdf` if given; else queries a local box around the point."""
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    gdf = flowlines_gdf
    if gdf is None or getattr(gdf, "empty", True):
        d = 0.01
        gj = flowlines_bbox(lon - d, lat - d, lon + d, lat + d, max_area_deg2=2.0)
        if not gj or not gj.get("features"):
            return None
        gdf = gpd.GeoDataFrame.from_features(gj["features"], crs=CRS_WGS84)
    if gdf is None or gdf.empty:
        return None
    g = gdf.to_crs(CRS_ALBERS)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    if g.empty:
        return None
    click = gpd.GeoSeries([Point(lon, lat)], crs=CRS_WGS84).to_crs(CRS_ALBERS).iloc[0]
    idx = g.geometry.distance(click).idxmin()
    snapped = nearest_points(g.geometry.loc[idx], click)[0]
    dist_ft = float(click.distance(snapped) * FT_PER_M)
    back = gpd.GeoSeries([snapped], crs=CRS_ALBERS).to_crs(CRS_WGS84).iloc[0]
    row = g.loc[idx]
    da = _prop(row, ("totdasqkm", "TotDASqKm", "TotDASqKM"))
    comid = _prop(row, ("nhdplusid", "NHDPlusID", "comid", "COMID"))
    return {"lat": float(back.y), "lon": float(back.x),
            "comid": (int(comid) if comid is not None else None),
            "da_sqkm": (float(da) if da is not None else None),
            "dist_ft": dist_ft}


def reach_between(up: dict, dn: dict) -> dict:
    """Trace the reach along the NHD network between two snapped points by merging the flowlines
    in their bounding box and taking the substring between the two projections. Orders the points
    by drainage area (larger = downstream). Raises ValueError if they aren't on one connected
    mainstem or the reach exceeds 1 mile. Returns {reach (4326 Feature), length_m, da_sqkm, lat,
    lon, warnings}."""
    import geopandas as gpd
    import networkx as nx
    from shapely.geometry import LineString, Point, mapping, shape
    from shapely.ops import substring

    da_up = up.get("da_sqkm") or 0.0
    da_dn = dn.get("da_sqkm") or 0.0
    if da_dn < da_up:                                  # ensure dn is downstream
        up, dn = dn, up
        da_up, da_dn = da_dn, da_up

    pad = 0.01
    lons, lats = [up["lon"], dn["lon"]], [up["lat"], dn["lat"]]
    gj = flowlines_bbox(min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad,
                        max_area_deg2=4.0)
    if not gj or not gj.get("features"):
        raise ValueError("Could not fetch the stream network for these points.")
    raw = gpd.GeoSeries([shape(f["geometry"]) for f in gj["features"] if f.get("geometry")],
                        crs=CRS_WGS84).to_crs(CRS_ALBERS)
    parts = []                                         # explode to simple LineStrings
    for g in raw.geometry:
        if g is None or g.is_empty:
            continue
        parts.extend(list(g.geoms) if g.geom_type == "MultiLineString" else [g])
    if not parts:
        raise ValueError("No usable flowline geometry for the reach.")
    sp_up = gpd.GeoSeries([Point(up["lon"], up["lat"])], crs=CRS_WGS84).to_crs(CRS_ALBERS).iloc[0]
    sp_dn = gpd.GeoSeries([Point(dn["lon"], dn["lat"])], crs=CRS_WGS84).to_crs(CRS_ALBERS).iloc[0]

    iu = min(range(len(parts)), key=lambda i: parts[i].distance(sp_up))
    idn = min(range(len(parts)), key=lambda i: parts[i].distance(sp_dn))
    if parts[iu].distance(sp_up) > 80.0 or parts[idn].distance(sp_dn) > 80.0:
        raise ValueError("Couldn't match the points to NHD flowlines — click directly on a cyan line.")

    def nd(xy):
        return (round(xy[0], 1), round(xy[1], 1))      # 0.1 m node snap (NHD shares exact endpoints)

    if iu == idn:                                       # both points on one segment
        full = parts[iu]
    else:
        G = nx.Graph()
        for ln in parts:
            cs = ln.coords
            G.add_edge(nd(cs[0]), nd(cs[-1]), weight=ln.length, geom=ln)
        # start/end at the segment endpoints farther from the other point → both end
        # segments are traversed, so the substring covers the full reach.
        un = max((parts[iu].coords[0], parts[iu].coords[-1]), key=lambda c: Point(c).distance(sp_dn))
        vn = max((parts[idn].coords[0], parts[idn].coords[-1]), key=lambda c: Point(c).distance(sp_up))
        try:
            path = nx.shortest_path(G, nd(un), nd(vn), weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise ValueError("The two points aren't on the same connected stream — pick points "
                             "along one reach, or use Draw manually.")
        coords = []
        for k in range(len(path) - 1):
            cs = list(G[path[k]][path[k + 1]]["geom"].coords)
            if nd(cs[0]) != path[k]:
                cs = cs[::-1]                           # orient start→end along the path
            coords.extend(cs[1:] if coords else cs)
        if len(coords) < 2:
            raise ValueError("Could not assemble the reach geometry.")
        full = LineString(coords)

    a, b = sorted([full.project(sp_up), full.project(sp_dn)])
    seg = substring(full, a, b)
    length_m = float(seg.length)
    if length_m > MAX_REACH_M:
        raise ValueError(f"Reach is {length_m / MAX_REACH_M:.2f} miles — the maximum length is "
                         f"1 mile. Pick points closer together.")
    if length_m < 5.0:
        raise ValueError("The two points are essentially the same — pick points farther apart.")

    reach = gpd.GeoSeries([seg], crs=CRS_ALBERS).to_crs(CRS_WGS84).iloc[0]
    mid = reach.interpolate(0.5, normalized=True)
    return {
        "reach": {"type": "Feature", "properties": {}, "geometry": mapping(reach)},
        "length_m": length_m,
        "da_sqkm": float(da_dn or da_up or 0.0),
        "lat": float(mid.y), "lon": float(mid.x),
        "warnings": [],
    }
