"""Place / address / stream-name → (lat, lon) for the Reach-tab search box.

OSM-based geocoders that resolve place names, street addresses, and natural
features (streams/rivers): **Photon** (Komoot) first — which also powers the
client-side type-ahead in ``www/geocode.js`` — then **Nominatim** as a fallback.
CONUS-biased. Never raises — returns None on failure / no match. Uses urllib
(no extra dependency), matching hype_app.hydro.

Data © OpenStreetMap contributors (via Photon / Nominatim).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Optional, Tuple

_UA = "HYPE-hyporheic/1.0 (https://github.com/gtmenichino/hype-app)"
_PHOTON = "https://photon.komoot.io/api/"
_NOMINATIM = "https://nominatim.openstreetmap.org/search"
_CONUS = (39.5, -98.35)     # bias results toward the lower-48


def _get(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _photon(q: str, timeout: float) -> Optional[Tuple[float, float]]:
    url = _PHOTON + "?" + urllib.parse.urlencode(
        {"q": q, "limit": 5, "lang": "en", "lat": _CONUS[0], "lon": _CONUS[1]})
    for feat in (_get(url, timeout) or {}).get("features", []):
        if (feat.get("properties") or {}).get("countrycode") == "US":  # keep results in CONUS
            lon, lat = feat["geometry"]["coordinates"][:2]
            return float(lat), float(lon)
    return None


def _nominatim(q: str, timeout: float) -> Optional[Tuple[float, float]]:
    url = _NOMINATIM + "?" + urllib.parse.urlencode(
        {"q": q, "format": "jsonv2", "limit": 1, "countrycodes": "us"})
    data = _get(url, timeout)
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def geocode_address(address: str, timeout: float = 15.0) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for a US place / address / stream name, or None. Tries Photon then
    Nominatim. Backs the "Find on map" button; the as-you-type dropdown queries Photon from
    the browser (www/geocode.js)."""
    if not address or not address.strip():
        return None
    q = address.strip()
    for fn in (_photon, _nominatim):
        try:
            hit = fn(q, timeout)
        except Exception:  # noqa: BLE001 — try the next provider / give up
            hit = None
        if hit:
            return hit
    return None
