"""Turn run_hyporheic artifacts into map-ready GeoJSON (EPSG:4326) + summary text."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd


def _to_geojson_4326(path, max_features: int = 4000):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    gdf = gpd.read_file(p)
    if gdf.empty:
        return None
    gdf = gdf.to_crs(4326)
    if len(gdf) > max_features:                      # down-sample dense particle sets
        gdf = gdf.sample(max_features, random_state=0).sort_index()
    return json.loads(gdf.to_json())


def pathlines_geojson(result: dict):
    return _to_geojson_4326(result.get("pathlines_fc"))


def points_geojson(result: dict):
    return _to_geojson_4326(result.get("points_fc"), max_features=6000)


def bounds_latlon(result: dict):
    """[[south, west], [north, east]] over the result vectors, for map.fit_bounds."""
    for key in ("pathlines_fc", "points_fc", "pathlines_fc_3d"):
        p = result.get(key)
        if p and Path(p).exists():
            gdf = gpd.read_file(p).to_crs(4326)
            if not gdf.empty:
                minx, miny, maxx, maxy = (float(v) for v in gdf.total_bounds)
                return [[miny, minx], [maxy, maxx]]
    return None


def summary_text(result: dict, out_dir) -> str:
    """Best-effort: the engine's publication-ready stats .txt, else a short fallback."""
    stats = Path(out_dir) / "summary" / "Forward_pathline_stats.txt"
    if stats.exists():
        try:
            return stats.read_text(encoding="utf-8-sig")
        except Exception:  # noqa: BLE001
            pass
    grid = result.get("grid") or {}
    return (f"Grid: {grid.get('ncol')}×{grid.get('nrow')}×{grid.get('nlay')} "
            f"({grid.get('n_cells_total'):,} cells)\n"
            f"Pathlines: {result.get('pathlines_fc')}\n"
            f"Points: {result.get('points_fc')}")


# ---- hydraulic-head + grid visualization (per-layer GeoTIFFs the engine already exports) ----

def head_rasters(work_dir, result: dict | None = None) -> list[str]:
    """Sorted per-layer head GeoTIFFs (index 0 = head_L01 = top layer)."""
    import glob
    d = Path(work_dir) / "summary" / "head" / "per_layer_tif"
    tifs = sorted(glob.glob(str(d / "head_L*.tif")))
    if tifs:
        return tifs
    head = (result or {}).get("head") or {}            # fallback to engine-reported paths
    for key in ("geotiffs", "tifs", "per_layer_tif"):
        v = head.get(key) if isinstance(head, dict) else None
        if isinstance(v, (list, tuple)) and v:
            return sorted(str(p) for p in v)
    return []


def _valid_mask(a, nodata):
    import numpy as np
    m = np.isfinite(a)
    if nodata is not None:
        m &= (a != nodata)
    return m & (a > -9000.0)                            # guard -9999 sentinel / HDRY


def head_value_range(paths) -> tuple[float, float]:
    """Global (vmin, vmax) of head across all layers, ignoring nodata — keeps colors comparable."""
    import numpy as np
    import rasterio
    lo, hi = np.inf, -np.inf
    for f in paths:
        with rasterio.open(f) as s:
            a = s.read(1).astype("float64"); nod = s.nodata
        m = _valid_mask(a, nod)
        if m.any():
            lo = min(lo, float(a[m].min())); hi = max(hi, float(a[m].max()))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return (0.0, 1.0)
    return (lo, hi)


def raster_overlay(path, *, vmin, vmax, cmap="viridis", max_dim: int = 1024,
                   smooth_to: int = 700) -> dict:
    """Colorize a single-band raster (e.g. a head layer) → ipyleaflet ImageOverlay payload
    {"url","bounds"} in EPSG:4326, transparent at nodata. NaN-aware upsampling renders the
    otherwise blocky ~180 px per-cell field as a smooth, crisp overlay."""
    import matplotlib
    import numpy as np
    from matplotlib.colors import Normalize

    from .dem import load_raster_4326, rgba_to_overlay
    z, xs, ys, _dx, _dy = load_raster_4326(path, max_dim=max_dim)
    valid = np.isfinite(z)
    f = int(smooth_to // max(z.shape)) if max(z.shape) else 1
    if f >= 2 and valid.any():
        try:                                            # smooth head within valid area, crisp edges
            from scipy.ndimage import zoom
            z = zoom(np.where(valid, z, float(np.nanmean(z[valid]))), f, order=1)
            valid = zoom(valid.astype(np.float32), f, order=0) > 0.5
        except Exception:  # noqa: BLE001 — scipy missing / zoom failure: keep native resolution
            pass
    cmap_obj = (matplotlib.colormaps[cmap] if hasattr(matplotlib, "colormaps")
                else matplotlib.cm.get_cmap(cmap))
    rgba = cmap_obj(Normalize(vmin=vmin, vmax=vmax)(np.where(valid, z, vmin)))
    rgba[..., 3] = valid.astype(float)
    return rgba_to_overlay(rgba, xs, ys)


def head_contours_geojson(path, *, levels):
    """Hydraulic-head contour lines (EPSG:4326 GeoJSON LineStrings, each with a `level`)."""
    import numpy as np
    import rasterio
    from contourpy import contour_generator
    from pyproj import Transformer

    with rasterio.open(path) as s:
        a = s.read(1).astype("float64"); nod = s.nodata; transform = s.transform; crs = s.crs
    z = np.where(_valid_mask(a, nod), a, np.nan)        # NaN → contourpy masks it (no edge artifacts)
    if not np.isfinite(z).any():
        return None
    cg = contour_generator(z=z)
    tr = Transformer.from_crs(crs, 4326, always_xy=True)
    feats = []
    for lv in levels:
        for seg in cg.lines(float(lv)):                 # seg: (N,2) array of (col, row) index coords
            if len(seg) < 2:
                continue
            xs_, ys_ = rasterio.transform.xy(transform, seg[:, 1], seg[:, 0], offset="center")
            lon, lat = tr.transform(np.asarray(xs_), np.asarray(ys_))
            coords = [[float(x), float(y)] for x, y in zip(np.atleast_1d(lon), np.atleast_1d(lat))]
            feats.append({"type": "Feature", "properties": {"level": round(float(lv), 3)},
                          "geometry": {"type": "LineString", "coordinates": coords}})
    return {"type": "FeatureCollection", "features": feats} if feats else None


def head_contour_labels(gj, *, max_labels: int = 40):
    """One (lat, lon, "NNN.N") label per contour line (at its midpoint), decimated to max_labels."""
    if not gj:
        return []
    out = []
    for f in gj.get("features", []):
        coords = f.get("geometry", {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = coords[len(coords) // 2]
        out.append((float(lat), float(lon), f"{float(f['properties']['level']):.1f}"))
    if len(out) > max_labels:
        s = len(out) / max_labels
        out = [out[int(i * s)] for i in range(max_labels)]
    return out


def grid_overlay(paths, *, max_dim: int = 1400):
    """Render ONLY the actively-modeled cells as a grid-line overlay (EPSG:4326), aligned with
    the head overlay. Active mask ≈ idomain = union of (head ≠ nodata) across all layers (the
    deep layers fill the whole domain). Returns {"url","bounds"} or None."""
    import numpy as np
    import rasterio
    import rioxarray  # noqa: F401 — .rio accessor
    from rasterio.enums import Resampling

    from .dem import rgba_to_overlay
    mask = None
    for f in paths:
        with rasterio.open(f) as s:
            a = s.read(1).astype("float64"); nod = s.nodata
        m = _valid_mask(a, nod)
        mask = m if mask is None else (mask | m)
    if mask is None or not mask.any():
        return None
    da = rioxarray.open_rasterio(paths[0], masked=True).squeeze()
    da_m = da.copy(data=mask.astype("float32")).rio.write_nodata(0.0)
    da_m = da_m.rio.reproject("EPSG:4326", resampling=Resampling.nearest)
    M = np.asarray(da_m.values, dtype=float) > 0.5
    xs = np.asarray(da_m["x"].values, dtype=float); ys = np.asarray(da_m["y"].values, dtype=float)
    if ys[0] < ys[-1]:
        M = M[::-1, :]; ys = ys[::-1]
    nr, nc = M.shape
    u = max(1, min(6, int(max_dim // max(nr, nc))))
    rgba = np.zeros((nr * u, nc * u, 4), dtype=float)
    up = np.repeat(np.repeat(M, u, axis=0), u, axis=1)
    if u >= 2:                                          # paint each active cell's edges as a line
        rr = np.arange(nr * u) % u; cc = np.arange(nc * u) % u
        edge = (rr[:, None] == 0) | (rr[:, None] == u - 1) | (cc[None, :] == 0) | (cc[None, :] == u - 1)
        rgba[up & edge] = [0.16, 0.16, 0.16, 0.9]
    else:                                               # very large grid: faint fill instead of lines
        rgba[up] = [0.30, 0.30, 0.30, 0.25]
    return rgba_to_overlay(rgba, xs, ys)


def colorbar_datauri(vmin, vmax, *, cmap="viridis", label="Hydraulic head") -> str:
    """A small horizontal colorbar PNG (base64 data URI) for the Results legend."""
    import base64
    import io

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cmap_obj = (matplotlib.colormaps[cmap] if hasattr(matplotlib, "colormaps")
                else matplotlib.cm.get_cmap(cmap))
    fig, ax = plt.subplots(figsize=(3.2, 0.55))
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap_obj, norm=Normalize(vmin, vmax),
                                          orientation="horizontal")
    cb.set_label(label, fontsize=8)
    ax.tick_params(labelsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
