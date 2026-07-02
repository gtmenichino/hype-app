"""Standalone end-to-end smoke test for the HEC-RAS 2025 surface-model pipeline.

Drives hype_app.ras.run_surface_model exactly the way app.py will, but outside Shiny:
four boundary lines (derived from the bundled template's arcs, so they sit over real
terrain at Mink Brook, NH) -> assemble_domain_from_sides -> DEM -> pipeline.

Usage (Windows dev):
  set HYPE_RAS_BIN=D:\\Code\\Work\\hype-app\\reference\\HEC-RAS_2025\\HEC-RAS 2025 Alpha
  py -3.12 tools\\ras_smoke.py [work_dir] [--dem path.tif] [--geographic-dem]

With --dem, that GeoTIFF is used as the app DEM. Otherwise a ~1 m 3DEP DEM is fetched
from the USGS ImageServer over the domain (in EPSG:26918, or EPSG:4326 with
--geographic-dem to exercise the estimate_utm_crs fallback).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402

from hype_app import geometry, ras  # noqa: E402

TEMPLATE_GEOM = ras.TEMPLATE_DIR / "Geometries" / "Geometry.h5"


def template_lines_4326() -> dict:
    """The template's four arcs as EPSG:4326 LineString Features (up/left/right/down)."""
    with h5py.File(TEMPLATE_GEOM, "r") as f:
        mt = f["Geometry/Mesh Topology"]
        nodes = mt["Nodes"][...]
        ipts = mt["Arc Internal Points"][...]
        icnt = mt["Arc Internal Points"].attrs["Count"]
        ist = mt["Arc Internal Points"].attrs["Start"]
        wkt = f.attrs["Project Projection"].decode()
    tr = Transformer.from_crs(CRS.from_wkt(wkt), "EPSG:4326", always_xy=True)

    def arc(i, a, b):
        pts = np.vstack([nodes[a], ipts[ist[i]:ist[i] + icnt[i]], nodes[b]])
        lon, lat = tr.transform(pts[:, 0], pts[:, 1])
        return {"type": "Feature", "properties": {},
                "geometry": {"type": "LineString",
                             "coordinates": [[float(x), float(y)] for x, y in zip(lon, lat)]}}

    # template arcs: 0 = up (N0->N1), 1 = left bank (N1->N2), 2 = down (N2->N3), 3 = right (N3->N0)
    return {"up": arc(0, 0, 1), "left": arc(1, 1, 2), "down": arc(2, 2, 3), "right": arc(3, 3, 0)}


def fetch_3dep(bounds4326, out_path, epsg: int, res_m: float = 1.0) -> str:
    """Minimal 3DEP fetch via the USGS ImageServer (py3dep-free)."""
    import shutil
    import urllib.parse
    import urllib.request

    minx, miny, maxx, maxy = bounds4326
    dx, dy = (maxx - minx) * ras.BUFFER_FRAC * 2, (maxy - miny) * ras.BUFFER_FRAC * 2
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform([minx - dx, maxx + dx, minx - dx, maxx + dx],
                          [miny - dy, miny - dy, maxy + dy, maxy + dy])
    bxmin, bymin, bxmax, bymax = min(xs), min(ys), max(xs), max(ys)
    if epsg == 4326:
        w = max(64, min(4000, round((bxmax - bxmin) * 111320 / res_m)))
        h = max(64, min(4000, round((bymax - bymin) * 110540 / res_m)))
    else:
        w = max(64, min(4000, round((bxmax - bxmin) / res_m)))
        h = max(64, min(4000, round((bymax - bymin) / res_m)))
    params = urllib.parse.urlencode({
        "bbox": f"{bxmin},{bymin},{bxmax},{bymax}", "bboxSR": epsg, "imageSR": epsg,
        "size": f"{w},{h}", "format": "tiff", "pixelType": "F32", "noData": -9999,
        "interpolation": "RSP_BilinearInterpolation", "f": "image"})
    url = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/"
           "ImageServer/exportImage?" + params)
    with urllib.request.urlopen(url, timeout=180) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir", nargs="?", default=None)
    ap.add_argument("--dem", default=None, help="use this GeoTIFF instead of fetching 3DEP")
    ap.add_argument("--geographic-dem", action="store_true",
                    help="fetch the DEM in EPSG:4326 (tests the UTM-fallback CRS path)")
    ap.add_argument("--cell", type=float, default=20.0)
    ap.add_argument("--flow", type=float, default=28.3, help="m3/s")
    ap.add_argument("--slope", type=float, default=0.001)
    ap.add_argument("--n", type=float, default=0.06)
    ap.add_argument("--hours", type=float, default=6.0)
    args = ap.parse_args()

    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="ras_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {work}")
    if not ras.ras_available():
        print("ERROR: no RAS CLI found — set HYPE_RAS_BIN or add bin/ras2025.")
        return 2

    lines = template_lines_4326()
    build = geometry.assemble_domain_from_sides(lines["up"], lines["left"],
                                                lines["right"], lines["down"])
    assert build, "assemble_domain_from_sides failed on the template arcs"

    dem = args.dem
    if not dem:
        dom = geometry.single_feature_gdf(build["domain"])
        epsg = 4326 if args.geographic_dem else 26918
        dem = fetch_3dep(tuple(float(b) for b in dom.total_bounds),
                         work / f"dem_{epsg}.tif", epsg)
        print(f"DEM fetched: {dem}")

    slope = ras.default_friction_slope(dem, build["up"], build["down"]) or args.slope
    print(f"friction slope: {slope}")
    est = ras.estimate_cell_count(geometry.single_feature_gdf(build["domain"]), args.cell)
    print(f"estimated cells @ {args.cell} m: {est:,}")

    payload = {
        "up": build["up"], "left": build["left"], "right": build["right"],
        "down": build["down"], "domain": build["domain"], "dem": str(dem),
        "flow_cms": args.flow, "friction_slope": slope, "manning_n": args.n,
        "cell_size_m": args.cell, "duration_hr": args.hours,
        "timestep_s": 10.0, "output_interval_s": args.hours * 3600 / 24,
        "work_dir": str(work),
    }
    # exercise the mesh-preview path first (fast; also proves Face Data decoding)
    prev = ras.build_mesh_preview_safe(payload, log=lambda m: print(f"[mesh] {m}"))
    if "error" in prev:
        print("MESH PREVIEW FAILED:\n" + prev["error"])
        return 1
    ov = prev.get("overlay")
    print(f"mesh preview: {prev['cell_count']:,} cells, {prev['n_faces']:,} faces, "
          f"overlay {'%d KB' % (len(ov['url']) // 1024) if ov else 'MISSING'}, "
          f"too_big={prev['too_big']}")
    assert prev["too_big"] or ov, "mesh preview produced no overlay"

    def _progress(stage, pct):
        if pct is None or pct % 25 == 0:
            print(f"[progress] {stage}: {pct if pct is not None else '…'}")

    t0 = time.time()
    result = ras.run_surface_model_safe(payload, log=lambda m: print(f"[ras] {m}"),
                                        progress=_progress)
    if "error" in result:
        print("FAILED:\n" + result["error"])
        return 1
    print(f"\nOK in {time.time() - t0:.0f}s")
    for k in ("n_cells", "profiles", "max_depth_m", "wetted_area_m2", "epsg", "runtime_s"):
        print(f"  {k}: {result[k]}")
    for k in ("depth_tif", "wse_tif", "wse_for_gw"):
        print(f"  {k}: {result[k]}")

    # fidelity regression guards: results must be mapped at the 1 m target (finer if the
    # DEM+mesh allow), the extent must carry real vertex density, and on fine-DEM runs the
    # wetted area must come out (nearly) continuous — fragmentation means quality loss
    import rasterio
    with rasterio.open(result["depth_tif"]) as ds:
        map_res = abs(ds.transform.a)
    with rasterio.open(dem) as ds:
        dem_res = abs(ds.transform.a) if ds.crs and ds.crs.is_projected else None
    print(f"  map raster: {map_res:g} m px (dem {dem_res if dem_res else '4326'} px)")
    if args.cell / 2.0 >= 1.0:
        assert map_res <= 1.001, f"expected <=1 m result mapping, got {map_res} m"
    ext = result["extent_feat"]
    geoms = (ext["geometry"]["coordinates"] if ext["geometry"]["type"] == "MultiPolygon"
             else [ext["geometry"]["coordinates"]])
    nvert = sum(len(ring) for g in geoms for ring in g)
    n_parts = int(result.get("n_parts") or len(geoms))
    main_frac = float(result.get("main_frac") or 1.0)
    print(f"  extent polygon: {ext['geometry']['type']}, {n_parts} part(s), "
          f"main {main_frac:.1%}, {nvert:,} vertices")
    assert nvert >= 200, f"extent suspiciously coarse: {nvert} vertices"
    if dem_res is not None and dem_res <= 2.0:
        assert main_frac >= 0.9, (f"water surface fragmented (main part only "
                                  f"{main_frac:.0%}) on a {dem_res:g} m DEM — "
                                  "raster-quality regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
