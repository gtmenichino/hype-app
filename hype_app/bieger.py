"""Bieger et al. (2015) regional bankfull hydraulic geometry (ported from EASI).

Bankfull width, mean depth, and cross-sectional area as power functions of drainage
area, stratified by the eight Fenneman physiographic *divisions* of the conterminous
U.S. (Bieger, Rathjens, Allen & Arnold, 2015, JAWRA 51(4):842-858, Table 3):

    y = a * DA^b        DA in km^2;  width & depth in m, area in m^2.

The division is looked up from the analysis point against bundled USGS physiographic-
division polygons (data/physio_divisions.geojson). Outside CONUS / unknown → national
("USA") curve. No exception escapes ``bankfull_geometry`` — it always returns a usable
estimate.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

# Division abbr -> (a, b) for bankfull width [m], mean depth [m], area [m^2]; DA in km^2.
# Bieger et al. 2015, Table 3.  LUP and IHI are tentative (n < 10).
COEF: dict[str, dict[str, tuple[float, float]]] = {
    "LUP": {"width": (4.15, 0.308), "depth": (0.31, 0.202), "area": (1.27, 0.509)},
    "APL": {"width": (2.22, 0.363), "depth": (0.24, 0.323), "area": (0.52, 0.680)},
    "AHI": {"width": (3.12, 0.415), "depth": (0.26, 0.287), "area": (0.82, 0.704)},
    "IPL": {"width": (2.56, 0.351), "depth": (0.38, 0.191), "area": (1.28, 0.472)},
    "IHI": {"width": (23.23, 0.121), "depth": (0.27, 0.267), "area": (6.28, 0.387)},
    "RMS": {"width": (1.24, 0.435), "depth": (0.23, 0.225), "area": (0.20, 0.688)},
    "IMP": {"width": (1.11, 0.415), "depth": (0.07, 0.329), "area": (0.07, 0.751)},
    "PMS": {"width": (2.76, 0.399), "depth": (0.23, 0.294), "area": (0.87, 0.652)},
    "USA": {"width": (2.70, 0.352), "depth": (0.30, 0.213), "area": (0.95, 0.540)},
}

_DIV_ABBR = {
    "LAURENTIAN UPLAND": "LUP", "ATLANTIC PLAIN": "APL",
    "APPALACHIAN HIGHLANDS": "AHI", "INTERIOR PLAINS": "IPL",
    "INTERIOR HIGHLANDS": "IHI", "ROCKY MOUNTAIN SYSTEM": "RMS",
    "INTERMONTANE PLATEAUS": "IMP", "PACIFIC MOUNTAIN SYSTEM": "PMS",
}
DIV_NAME = {abbr: name.title() for name, abbr in _DIV_ABBR.items()}
DIV_NAME["USA"] = "National curve"

_GEOJSON = Path(__file__).resolve().parent / "data" / "physio_divisions.geojson"


@functools.lru_cache(maxsize=1)
def _divisions():
    """Bundled physiographic-division polygons (EPSG:4326), loaded once."""
    import geopandas as gpd

    gdf = gpd.read_file(_GEOJSON)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)


def division_at(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """Physiographic-division abbreviation containing (lat, lon), else None (nearest
    within ~0.25 deg for coarse coastline/border polygons; truly off-grid → None)."""
    if lat is None or lon is None:
        return None
    try:
        from shapely.geometry import Point

        gdf = _divisions()
        pt = Point(float(lon), float(lat))
        hit = gdf[gdf.geometry.intersects(pt)]
        if not hit.empty:
            name = hit.iloc[0]["DIVISION"]
        else:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dist = gdf.geometry.distance(pt)
            if float(dist.min()) > 0.25:
                return None
            name = gdf.loc[dist.idxmin(), "DIVISION"]
        return _DIV_ABBR.get(str(name).strip().upper())
    except Exception:  # noqa: BLE001 — resilience by design (fall back to national)
        return None


def _power(coef: tuple[float, float], da: float) -> float:
    a, b = coef
    return a * (da ** b)


def bankfull_geometry(da_sqkm: float, lat: Optional[float] = None,
                      lon: Optional[float] = None) -> dict:
    """Regional bankfull geometry for a drainage area at a location (Bieger 2015
    physiographic-division curve; national fallback). Returns metres / m²."""
    da = max(float(da_sqkm or 0.0), 0.01)
    abbr = division_at(lat, lon)
    key = abbr if abbr in COEF else "USA"
    c = COEF[key]
    return {
        "width_m": round(_power(c["width"], da), 2),
        "depth_m": round(_power(c["depth"], da), 3),
        "area_m2": round(_power(c["area"], da), 2),
        "division": key,
        "division_name": DIV_NAME.get(key, "National curve"),
        "regional": key != "USA",
    }
