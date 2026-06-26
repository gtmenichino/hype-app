"""Drawn-geometry handling: ipyleaflet DrawControl GeoJSON -> projected GeoDataFrames.

The map hands us EPSG:4326 features. The MODFLOW grid math works in the projected CRS's
linear units, so we reproject to a UTM zone (metres) chosen from the domain centroid; the
model therefore runs in metres (length_units='meters').
"""
from __future__ import annotations

from typing import Iterable

import geopandas as gpd
from shapely.geometry import shape


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
