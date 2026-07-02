"""Run the HEC-RAS 2025 2D surface-water model on a repacked copy of the bundled template.

Pipeline (run_surface_model, called inside a worker thread):
  1. copy hype_app/data/ras_template -> <work_dir>/ras  (fresh every run)
  2. reproject+clip the app's 3DEP DEM to the model CRS  -> terrain_src.tif
  3. `ras createterrain`  -> Terrains/Terrain.h5 (+ normalized companion tif)
  4. h5py/XML repack (hype_app.ras_h5): arcs from the 4 boundary lines, BC polylines,
     constant Flow / Normal Depth, Manning n, time window, SI + model-CRS metadata
  5. `ras mesh`  -> regenerate the computational mesh; enforce the cell-count cap
  6. `ras solve <project>.ras --solver CPU`  -> Results/... (Result).h5
  7. `ras map` depth + watersurface at the LAST timestep -> GeoTIFFs
  8. post-process (hype_app.ras_results): wetted-extent polygon + WSE-on-DEM-grid raster
     that feeds the groundwater run's wse_path

The RAS CLI is resolved like the MODFLOW binaries (hype_app/run.py): the env override
HYPE_RAS_BIN (Windows dev -> the installed "HEC-RAS 2025 Alpha" folder or ras.exe) wins;
otherwise the bundled Linux runtime at bin/ras2025 is used via its own dotnet.

CLI quirks learned the hard way (Phase 0):
  - createterrain -o must be an ABSOLUTE ("rooted") path, and must not already exist
  - `ras map -o` needs a directory component (bare filename crashes the tif writer)
  - the CLI writes results to "Results/<Plan> (Result).h5" (the GUI uses "<Plan>.h5")
  - run duration comes from the BC file's Start/End window (Time Window Mode=BoundaryCondition)
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from . import ras_h5

_APP_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "data" / "ras_template"
PROJECT_NAME = "MinkBrook"          # template project/plan names stay as-is
BUFFER_FRAC = 0.12                  # terrain clip buffer, matches hype_app.dem.BUFFER_FRAC

# All model times are anchored at the template's epoch (arbitrary for a constant-flow run).
MODEL_START = _dt.datetime(2099, 1, 1, tzinfo=_dt.timezone.utc)

_NOISE = ("Environment does not allow", "Quads:", "Triangles:", "Cartesian:",
          "Postprocessing:", "Global pre/postprocessing:")
_PROGRESS_RE = re.compile(r"^\s*Progress:\s*(\d{1,3})%")


def ras_bundle_dir() -> Path:
    return _APP_ROOT / "bin" / "ras2025"


def ras_cmd() -> tuple[list[str], dict]:
    """(argv prefix, env) for invoking the RAS CLI on this platform.

    HYPE_RAS_BIN may point at ras.exe itself or its folder (Windows dev — GDAL is found
    automatically next to the exe). Otherwise the bundled linux-x64 runtime is used.
    """
    env = dict(os.environ)
    override = os.environ.get("HYPE_RAS_BIN")
    if override:
        exe = Path(override)
        if exe.is_dir():
            exe = exe / ("ras.exe" if sys.platform.startswith("win") else "ras")
        return [str(exe)], env

    bundle = ras_bundle_dir()
    dotnet = bundle / "dotnet" / "dotnet"
    ras_dll = bundle / "app" / "ras.dll"
    env.update(_linux_env(bundle))
    return [str(dotnet), str(ras_dll)], env


# The HDF5 natives ship ONLY under their bare names (libhdf5.so); the SONAME names
# (libhdf5.so.320) are created here as runtime symlinks pointing at those same files.
# CRITICAL: they must be symlinks to the SAME inode, never file copies — with copies the
# dynamic loader maps two independent libhdf5 instances (the .NET binding opens datasets in
# one while libhdf5_hl's internal calls land in the other) and every H5Dwrite_chunk fails
# with "invalid dataset ID", silently NaN-filling all results.
_HDF5_SONAME_LINKS = {"libhdf5.so.320": "libhdf5.so", "libhdf5_hl.so.320": "libhdf5_hl.so"}


def _hdf5_links_dir() -> Path:
    return Path(tempfile.gettempdir()) / "hype_ras_hdf5links"


def _linux_env(bundle: Path) -> dict:
    """Env vars for the bundled linux runtime: native lib resolution + GDAL data."""
    natives = bundle / "natives"
    gdal = bundle / "GDAL"
    ld = os.pathsep.join(str(p) for p in (_hdf5_links_dir(), natives, gdal / "lib")
                         if p.is_dir())
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    return {
        "DOTNET_ROOT": str(bundle / "dotnet"),
        "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_NOLOGO": "1",
        "RAS_GDAL": str(gdal),
        "LD_LIBRARY_PATH": (ld + os.pathsep + prev) if prev else ld,
        "HOME": os.environ.get("HOME") or "/tmp",
    }


def prepare_linux_bundle() -> None:
    """One-time Linux setup: chmod +x the bundled dotnet host (git from Windows loses the
    exec bit) and create the HDF5 SONAME symlinks. No-op on Windows / with HYPE_RAS_BIN."""
    if sys.platform.startswith("win") or os.environ.get("HYPE_RAS_BIN"):
        return
    import stat
    dotnet = ras_bundle_dir() / "dotnet" / "dotnet"
    if dotnet.exists():
        try:
            dotnet.chmod(dotnet.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:  # noqa: BLE001
            pass
    natives = ras_bundle_dir() / "natives"
    links = _hdf5_links_dir()
    try:
        links.mkdir(parents=True, exist_ok=True)
        for soname, bare in _HDF5_SONAME_LINKS.items():
            target = natives / bare
            link = links / soname
            if target.exists() and not link.exists():
                link.symlink_to(target)
    except Exception:  # noqa: BLE001 — surfaced later by the solver if HDF5 can't load
        pass


def ras_available() -> bool:
    """Is a RAS CLI plausibly runnable here (dev override or bundled runtime present)?"""
    override = os.environ.get("HYPE_RAS_BIN")
    if override:
        p = Path(override)
        return (p / "ras.exe").exists() if p.is_dir() else p.exists()
    return (ras_bundle_dir() / "app" / "ras.dll").exists()


# ---------------------------------------------------------------- CRS + terrain helpers

def model_crs_for(dem_path, domain_gdf_4326):
    """The RAS model CRS: the DEM's own projection when it is projected in metres
    (e.g. a UTM-tile 3DEP download), else the app's estimated UTM zone."""
    import rasterio
    from pyproj import CRS

    with rasterio.open(dem_path) as ds:
        crs = CRS.from_user_input(ds.crs) if ds.crs else None
    if crs is not None and crs.is_projected:
        try:
            unit = crs.axis_info[0].unit_name.lower()
        except Exception:  # noqa: BLE001
            unit = ""
        if unit in ("metre", "meter", "m"):
            return crs
    return CRS.from_user_input(domain_gdf_4326.estimate_utm_crs())


def prepare_terrain_tif(dem_path, domain_gdf_4326, model_crs, out_path) -> dict:
    """Reproject the app DEM to the model CRS, clipped to the domain bbox + buffer,
    float32 / nodata -9999 — the input handed to `ras createterrain`."""
    import numpy as np
    import rioxarray  # noqa: F401 — registers .rio
    from rioxarray.exceptions import NoDataInBounds

    minx, miny, maxx, maxy = (float(v) for v in domain_gdf_4326.total_bounds)
    dx, dy = (maxx - minx) * BUFFER_FRAC, (maxy - miny) * BUFFER_FRAC

    da = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
    try:
        da = da.rio.clip_box(minx - dx, miny - dy, maxx + dx, maxy + dy, crs="EPSG:4326")
    except NoDataInBounds as e:
        raise RuntimeError("The terrain DEM does not cover the drawn domain.") from e
    da = da.rio.reproject(model_crs)
    da = da.astype("float32").rio.write_nodata(-9999.0, encoded=False)
    da = da.fillna(-9999.0).rio.write_nodata(-9999.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(out_path, compress="deflate", tiled=True)
    res = abs(float(da.rio.resolution()[0]))
    arr = np.asarray(da.values)
    valid = arr[arr != -9999.0]
    if valid.size == 0:
        raise RuntimeError("The terrain DEM has no valid pixels over the drawn domain.")
    return {"path": str(out_path), "resolution_m": res,
            "zmin": float(valid.min()), "zmax": float(valid.max())}


# ---------------------------------------------------------------- size guardrail

# The mesher refines along the boundary arcs, so real cell counts run well above the
# naive area/cell² figure (observed 811 vs 347 @10 m, 3,220 vs 1,386 @5 m → ~2.3×).
MESH_REFINEMENT_FACTOR = 2.3


def estimate_cell_count(domain_gdf_4326, cell_size_m: float) -> int:
    area = float(domain_gdf_4326.to_crs(domain_gdf_4326.estimate_utm_crs()).area.iloc[0])
    return max(1, int(round(MESH_REFINEMENT_FACTOR * area / max(cell_size_m, 0.1) ** 2)))


def _map_resolution(dem_res_m: float, cell_size_m: float, bbox, max_pixels: float = 12e6) -> float:
    """Resolution for `ras map` output rasters: target 1 m (or finer when both the DEM and
    the mesh are finer). RAS interpolates faces/terrain bilinearly, so mapping finer than
    the DEM stays smooth — 1 m pixels are what keep a thin channel CONNECTED in the wet/dry
    raster instead of breaking into pools across coarse pixels, and what let the extent
    polygon resolve the water edge with high accuracy. Floored at 0.25 m and coarsened only
    as needed to keep the raster under `max_pixels` for the domain bbox."""
    res = max(0.25, min(dem_res_m, cell_size_m / 2.0, 1.0))
    w = max(bbox[2] - bbox[0], 1.0)
    h = max(bbox[3] - bbox[1], 1.0)
    if (w / res) * (h / res) > max_pixels:
        res = ((w * h) / max_pixels) ** 0.5
    return round(res, 3)


def cell_budget() -> tuple[int, int]:
    """(green ceiling, hard cap) — tune via env after Connect Cloud timings.
    Empirical: ~14 k cells × 6 hr ≈ 9 min on a desktop CPU (explicit SWE scales with
    cells × timesteps), so "quick" is a few thousand cells at the default duration."""
    green = int(os.environ.get("HYPE_RAS_GREEN_CELLS", 4_000))
    cap = int(os.environ.get("HYPE_RAS_MAX_CELLS", 60_000))
    return green, cap


def default_friction_slope(dem_path, up_feat, down_feat) -> float | None:
    """Prefill for the Normal Depth slope: mean DEM elevation drop from the upstream to the
    downstream boundary line over the distance between their midpoints. None if unsampleable."""
    try:
        import numpy as np
        import rasterio
        from pyproj import Transformer

        def _mid_and_pts(feat):
            cs = np.asarray(feat["geometry"]["coordinates"], dtype=float)
            return cs[len(cs) // 2], cs

        with rasterio.open(dem_path) as ds:
            tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)

            def _mean_elev(pts):
                xs, ys = tr.transform(pts[:, 0], pts[:, 1])
                vals = np.array([v[0] for v in ds.sample(zip(xs, ys))], dtype=float)
                vals = vals[np.isfinite(vals) & (vals > -9000)]
                return float(vals.mean()) if vals.size else None

            up_mid, up_pts = _mid_and_pts(up_feat)
            dn_mid, dn_pts = _mid_and_pts(down_feat)
            zu, zd = _mean_elev(up_pts), _mean_elev(dn_pts)
        if zu is None or zd is None:
            return None
        # distance between midpoints, metres (local equirectangular is plenty here)
        import math
        kx = 111320.0 * math.cos(math.radians((up_mid[1] + dn_mid[1]) / 2))
        dist = math.hypot((up_mid[0] - dn_mid[0]) * kx, (up_mid[1] - dn_mid[1]) * 110540.0)
        if dist <= 0:
            return None
        return max(round((zu - zd) / dist, 5), 0.0001)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- subprocess plumbing

class RasError(RuntimeError):
    pass


def _run_ras(args, *, cwd, env, log, cancel_evt, proc_holder, timeout_s, label,
             on_progress=None):
    """Run one RAS CLI verb, streaming de-noised output lines to `log`. `Progress: N%`
    lines (the engine emits one per percent of simulated time during the compute, and
    0-100 sweeps in other stages) are routed to `on_progress(pct)` instead of the log.
    Kills the process on cancel or when the wall-clock deadline passes; raises RasError
    on any failure."""
    cmd, base_env = ras_cmd()
    full = cmd + args
    log(f"$ ras {' '.join(args[:3])}{' ...' if len(args) > 3 else ''}")
    merged = dict(base_env)
    merged.update(env or {})
    proc = subprocess.Popen(
        full, cwd=str(cwd), env=merged,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    proc_holder["proc"] = proc
    deadline = time.monotonic() + timeout_s
    timed_out = {"v": False}

    def _watchdog():
        while proc.poll() is None:
            if cancel_evt is not None and cancel_evt.is_set():
                proc.kill()
                return
            if time.monotonic() > deadline:
                timed_out["v"] = True
                proc.kill()
                return
            time.sleep(0.5)

    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()
    tail = []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            m = _PROGRESS_RE.match(line)
            if m:
                if on_progress is not None:
                    try:
                        on_progress(min(100, int(m.group(1))))
                    except Exception:  # noqa: BLE001 — progress display must never kill a run
                        pass
                continue
            if any(k in line for k in _NOISE):
                continue
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
            log(f"  {line}")
    finally:
        proc.stdout.close()
        rc = proc.wait()
        proc_holder["proc"] = None
    if cancel_evt is not None and cancel_evt.is_set():
        raise RasError("Run cancelled.")
    if timed_out["v"]:
        raise RasError(f"{label} exceeded the {timeout_s:.0f}s time limit and was stopped.")
    if rc != 0:
        raise RasError(f"{label} failed (exit {rc}). Last output:\n" + "\n".join(tail[-15:]))


# ---------------------------------------------------------------- the pipeline

def run_surface_model(payload: dict, log=print, cancel_evt=None, proc_holder=None,
                      progress=None) -> dict:
    """Repack + mesh + solve + map + post-process. Returns the result dict; raises on failure.

    payload keys:
      up/left/right/down : oriented EPSG:4326 LineString Features (assemble_domain_from_sides)
      domain             : EPSG:4326 Polygon Feature
      dem                : path to the app's terrain GeoTIFF
      flow_cms           : upstream constant flow, m3/s
      friction_slope     : downstream Normal Depth slope
      manning_n          : constant Manning's n
      cell_size_m        : nominal mesh cell size, metres
      duration_hr        : simulated time window, hours
      timestep_s         : base compute timestep, seconds
      output_interval_s  : output profile spacing, seconds
      work_dir           : run scratch dir (project goes to <work_dir>/ras)

    `progress(stage: str, pct: int | None)` is called from this worker thread as the run
    advances; pct is None for stages without percent reporting (indeterminate).
    """
    import numpy as np
    from pyproj import Transformer

    from . import geometry, ras_results

    proc_holder = proc_holder if proc_holder is not None else {}
    timeout_s = float(os.environ.get("HYPE_RAS_TIMEOUT_S", 1800))
    t0 = time.monotonic()

    def _stage(name):
        if progress is not None:
            try:
                progress(name, None)
            except Exception:  # noqa: BLE001
                pass

        def _pct(p, _name=name):
            if progress is not None:
                try:
                    progress(_name, p)
                except Exception:  # noqa: BLE001
                    pass
        return _pct

    _stage("Preparing")
    prepare_linux_bundle()

    proj = Path(payload["work_dir"]) / "ras"
    if proj.exists():
        shutil.rmtree(proj)
    shutil.copytree(TEMPLATE_DIR, proj)
    for sub in ("Terrains", "Results"):          # ensure empty dirs exist (git drops them)
        (proj / sub).mkdir(exist_ok=True)
        (proj / sub / ".gitkeep").unlink(missing_ok=True)
    geometry_h5 = proj / "Geometries" / "Geometry.h5"
    bc_h5 = proj / "Boundary Conditions" / "Boundary Condition.h5"
    plan_h5 = proj / "Plans" / "Plan.h5"
    nvalues_h5 = proj / "Surface Layers" / "N Values.h5"
    ras_project = proj / f"{PROJECT_NAME}.ras"

    # -- model CRS + terrain
    dom_gdf = geometry.single_feature_gdf(payload["domain"])
    crs = model_crs_for(payload["dem"], dom_gdf)
    epsg = crs.to_epsg()
    if epsg is None:
        raise RasError("Could not resolve an EPSG code for the model CRS.")
    log(f"Model CRS: EPSG:{epsg} | SI units")
    _stage("Preparing terrain")
    terr = prepare_terrain_tif(payload["dem"], dom_gdf, crs, proj / "terrain_src.tif")
    log(f"Terrain ready: {terr['resolution_m']:.2f} m px, z {terr['zmin']:.1f}-{terr['zmax']:.1f} m")
    if terr["resolution_m"] > 3.0:
        log(f"NOTE: the terrain is {terr['resolution_m']:.0f} m — water-surface detail is "
            "limited by the DEM. Re-fetch the DEM at 1 m (DEM step) if lidar is available.")

    _run_ras(["createterrain", "-f", str(proj / "terrain_src.tif"),
              "-o", str(proj / "Terrains" / "Terrain.h5"), "-j", f"EPSG:{epsg}"],
             cwd=proj, env=None, log=log, cancel_evt=cancel_evt, proc_holder=proc_holder,
             timeout_s=timeout_s, label="Terrain creation",
             on_progress=_stage("Building terrain"))

    # -- repack (h5py + XML)
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def _xy(feat):
        cs = np.asarray(feat["geometry"]["coordinates"], dtype=float)
        x, y = tr.transform(cs[:, 0], cs[:, 1])
        return np.column_stack([x, y])

    sides_xy = {k: _xy(payload[k]) for k in ("up", "left", "right", "down")}
    wkt = crs.to_wkt("WKT1_GDAL")
    duration_s = float(payload["duration_hr"]) * 3600.0

    topo = ras_h5.write_geometry_topology(geometry_h5, sides_xy, float(payload["cell_size_m"]))
    log(f"Topology written: arc cells {topo['arc_cell_counts']}, "
        f"{topo['n_internal_points']} interior vertices")
    ras_h5.write_bc_lines(bc_h5, up_xy=sides_xy["up"], down_xy=sides_xy["down"],
                          flow_cms=float(payload["flow_cms"]),
                          friction_slope=float(payload["friction_slope"]),
                          start=MODEL_START, duration_s=duration_s)
    ras_h5.write_plan_attrs(plan_h5, start=MODEL_START, duration_s=duration_s,
                            timestep_s=float(payload["timestep_s"]),
                            output_interval_s=float(payload["output_interval_s"]))
    ras_h5.write_n_values(nvalues_h5, float(payload["manning_n"]))
    for h5 in (geometry_h5, bc_h5, plan_h5, nvalues_h5, proj / "Terrains" / "Terrain.h5"):
        ras_h5.write_root_attrs(h5, wkt=wkt, units="SI")
    ras_h5.rewrite_ras_xml(ras_project, epsg)

    # -- mesh + guardrail
    _run_ras(["mesh", "--source", str(geometry_h5)],
             cwd=proj, env=None, log=log, cancel_evt=cancel_evt, proc_holder=proc_holder,
             timeout_s=timeout_s, label="Meshing", on_progress=_stage("Meshing"))
    mesh = ras_h5.read_mesh_summary(geometry_h5)
    _green, cap = cell_budget()
    log(f"Mesh: {mesh['cell_count']:,} cells")
    if mesh["cell_count"] > cap:
        need = float(payload["cell_size_m"]) * (mesh["cell_count"] / cap) ** 0.5
        raise RasError(
            f"The mesh has {mesh['cell_count']:,} cells — above the {cap:,} cap for this server. "
            f"Increase the cell size to ~{need:.0f} m or shrink the domain.")

    # -- solve
    _run_ras(["solve", str(ras_project), "--solver", "CPU", "--core-count", "-1", "--force"],
             cwd=proj, env=None, log=log, cancel_evt=cancel_evt, proc_holder=proc_holder,
             timeout_s=timeout_s, label="HEC-RAS solve", on_progress=_stage("Computing"))
    results = sorted((proj / "Results").glob("*.h5"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise RasError("The solver finished but produced no result file.")
    result_h5 = results[-1]
    summary = ras_h5.read_compute_summary(result_h5)
    if not summary["success"]:
        raise RasError("HEC-RAS reported an unsuccessful compute.\n" + summary["log"][-2000:])
    log(f"Solve OK: {summary['profiles']} output profiles, "
        f"max depth {summary['max_depth_last']:.2f} m at the last timestep")

    # -- extract last-timestep rasters
    terrain_tifs = sorted((proj / "Terrains").glob("Terrain.*.tif"))
    if not terrain_tifs:
        raise RasError("createterrain did not leave a normalized terrain tif.")
    depth_tif = proj / "depth_last.tif"
    wse_tif = proj / "wse_last.tif"
    mapcell = _map_resolution(terr["resolution_m"], float(payload["cell_size_m"]),
                              mesh["perimeter_bbox"])
    log(f"Mapping results at {mapcell:g} m")
    try:
        for mt, out in (("depth", depth_tif), ("watersurface", wse_tif)):
            _run_ras(["map", "-r", str(result_h5), "-m", mt, "-p", "^1", "-o", str(out),
                      "-t", str(terrain_tifs[0]), "-c", f"{mapcell:g}", "--overwrite"],
                     cwd=proj, env=None, log=log, cancel_evt=cancel_evt,
                     proc_holder=proc_holder, timeout_s=timeout_s, label=f"Mapping {mt}",
                     on_progress=_stage(f"Mapping {'depth' if mt == 'depth' else 'water surface'}"))
    except RasError:
        if cancel_evt is not None and cancel_evt.is_set():
            raise
        log("`ras map` failed — falling back to rasterizing cell results with h5py.")
        ras_results.cells_to_rasters_fallback(result_h5, geometry_h5, terr["path"],
                                              depth_tif, wse_tif)

    # -- post-process for the app
    _stage("Post-processing")
    cell = float(payload["cell_size_m"])
    extent_feat = ras_results.wetted_extent_feature(depth_tif, min_part_m2=0.5 * cell * cell)
    props = (extent_feat or {}).get("properties") or {}
    n_parts = int(props.get("n_parts", 0))
    main_frac = float(props.get("main_frac", 1.0))
    wse_for_gw = Path(payload["work_dir"]) / "inputs" / "wse_ras.tif"
    ras_results.wse_on_dem_grid(wse_tif, depth_tif, payload["dem"], wse_for_gw)
    wetted_area = ras_results.wetted_area_m2(depth_tif)
    runtime_s = time.monotonic() - t0
    log(f"Surface model complete in {runtime_s:.0f}s — wetted area {wetted_area:,.0f} m² "
        f"({n_parts} part{'s' if n_parts != 1 else ''}, main {main_frac:.0%}).")
    if main_frac < 0.9:
        if terr["resolution_m"] > 2.0:
            log(f"WARNING: the water surface is fragmented (largest connected area only "
                f"{main_frac:.0%}) — the {terr['resolution_m']:.0f} m terrain is likely too "
                "coarse to resolve the channel. Re-fetch the DEM at 1 m if available.")
        else:
            log(f"WARNING: the water surface is fragmented (largest connected area only "
                f"{main_frac:.0%}). On fine terrain this can be real shallow/braided flow — "
                "try a smaller mesh cell size or a higher flow if you expect a continuous "
                "surface.")
    return {
        "project_dir": str(proj),
        "result_h5": str(result_h5),
        "depth_tif": str(depth_tif),
        "wse_tif": str(wse_tif),
        "wse_for_gw": str(wse_for_gw),
        "extent_feat": extent_feat,
        "n_parts": n_parts,
        "main_frac": main_frac,
        "terrain_res_m": terr["resolution_m"],
        "n_cells": mesh["cell_count"],
        "profiles": summary["profiles"],
        "max_depth_m": summary["max_depth_last"],
        "wetted_area_m2": wetted_area,
        "epsg": epsg,
        "runtime_s": runtime_s,
    }


def run_surface_model_safe(payload: dict, log=print, cancel_evt=None, proc_holder=None,
                           progress=None) -> dict:
    """extended_task-friendly wrapper: never raises; returns {"error": ...} on failure."""
    try:
        return run_surface_model(payload, log=log, cancel_evt=cancel_evt,
                                 proc_holder=proc_holder, progress=progress)
    except RasError as e:
        return {"error": str(e)}
    except Exception:  # noqa: BLE001
        return {"error": traceback.format_exc()}


# ---------------------------------------------------------------- mesh preview

MESH_PREVIEW_MAX_FACES = 150_000    # beyond this even the rasterized preview isn't useful


def build_mesh_preview(payload: dict, log=print, cancel_evt=None, proc_holder=None) -> dict:
    """Run ONLY the meshing (template copy -> topology repack -> `ras mesh`, ~1 s; no
    terrain needed) and return the triangular mesh rasterized as a transparent PNG
    ImageOverlay payload (a vector layer with thousands of edges would choke Leaflet),
    plus the real cell count.

    payload: up/left/right/down + domain Features, dem (CRS pick only), cell_size_m,
    work_dir. Returns {"cell_count", "n_faces", "overlay"|None, "too_big": bool,
    "cell_size_m"}; raises RasError on failure.
    """
    import h5py
    import numpy as np
    from pyproj import Transformer

    from . import geometry, ras_results

    proc_holder = proc_holder if proc_holder is not None else {}
    timeout_s = float(os.environ.get("HYPE_RAS_TIMEOUT_S", 1800))
    prepare_linux_bundle()

    proj = Path(payload["work_dir"]) / "ras_mesh"
    if proj.exists():
        shutil.rmtree(proj)
    shutil.copytree(TEMPLATE_DIR, proj)
    geometry_h5 = proj / "Geometries" / "Geometry.h5"

    dom_gdf = geometry.single_feature_gdf(payload["domain"])
    crs = model_crs_for(payload["dem"], dom_gdf)
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def _xy(feat):
        cs = np.asarray(feat["geometry"]["coordinates"], dtype=float)
        x, y = tr.transform(cs[:, 0], cs[:, 1])
        return np.column_stack([x, y])

    sides_xy = {k: _xy(payload[k]) for k in ("up", "left", "right", "down")}
    cell = float(payload["cell_size_m"])
    ras_h5.write_geometry_topology(geometry_h5, sides_xy, cell)
    _run_ras(["mesh", "--source", str(geometry_h5)],
             cwd=proj, env=None, log=log, cancel_evt=cancel_evt, proc_holder=proc_holder,
             timeout_s=timeout_s, label="Meshing")

    with h5py.File(geometry_h5, "r") as f:
        mesh = f["Geometry/2D Flow Areas/Mesh"]
        att = f["Geometry/2D Flow Areas/Attributes"][...]
        cell_count = int(att["Cell Count"][0])
        nodes = mesh["Node Coordinates"][...]
        faces = mesh["Face Data"][...]            # columns: [CellA, CellB, NodeA, NodeB]
    a, b = faces[:, 2], faces[:, 3]
    ok = (a >= 0) & (b >= 0) & (a < len(nodes)) & (b < len(nodes))
    a, b = a[ok], b[ok]
    n_faces = int(len(a))
    if n_faces > MESH_PREVIEW_MAX_FACES:
        return {"cell_count": cell_count, "n_faces": n_faces, "overlay": None,
                "too_big": True, "cell_size_m": cell}

    back = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_a, lat_a = back.transform(nodes[a, 0], nodes[a, 1])
    lon_b, lat_b = back.transform(nodes[b, 0], nodes[b, 1])
    segs = np.stack([np.column_stack([lon_a, lat_a]),
                     np.column_stack([lon_b, lat_b])], axis=1)
    overlay = ras_results.mesh_overlay(segs)
    log(f"Mesh preview: {cell_count:,} cells / {n_faces:,} faces at {cell:g} m")
    return {"cell_count": cell_count, "n_faces": n_faces, "overlay": overlay,
            "too_big": False, "cell_size_m": cell}


def build_mesh_preview_safe(payload: dict, log=print, cancel_evt=None,
                            proc_holder=None) -> dict:
    try:
        return build_mesh_preview(payload, log=log, cancel_evt=cancel_evt,
                                  proc_holder=proc_holder)
    except RasError as e:
        return {"error": str(e)}
    except Exception:  # noqa: BLE001
        return {"error": traceback.format_exc()}
