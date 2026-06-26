# src/hypetool/core/run_headless.py
"""
Headless, in-memory entry point for the hyporheic engine.

`run_hyporheic()` drives the *same* pipeline as `run_from_yaml()` (it calls the shared
`_run_pipeline`), but takes already-loaded geometries (GeoDataFrames) and raster paths
instead of a YAML config plus shapefiles on disk. It is designed for the Shiny / Posit
Connect Cloud web app and for any programmatic caller.

Key differences from the file-based path:
  * No `.prj` is read; the caller passes a `crs` and we set `cfg.hec_ras_crs` directly.
  * The domain polygon and the left/right floodplain boundary lines are injected as
    in-memory GeoDataFrames (no shapefiles required).
  * By default the terrain DEM is *also* used as the water-surface elevation (3DEP DEMs
    are hydro-flattened and capture the water surface at flight time). Pass `wse_path`
    to override with a dedicated WSE raster.
  * The ArcGIS Pro map-group step is skipped (`add_to_map=False`); no arcpy is required.

Returns the same artifact dict as `run_from_yaml` plus a `"grid"` sub-dict
(`ncol, nrow, nlay, n_cells_total`) for UI display / guardrails.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

from hypetool.inputs import Settings
from hypetool.functions import my_utils as myu
from hypetool.core.run_from_yaml import _ensure_executables, _run_pipeline


# --------------------------------------------------------------------------- #
# Geometry injection helpers
# --------------------------------------------------------------------------- #
def _as_single_line_gdf(gdf: "gpd.GeoDataFrame", target_crs: Any, name: str) -> "gpd.GeoDataFrame":
    """
    Reproject a (possibly multi-row / MultiLineString) line GeoDataFrame to `target_crs`
    and merge it into a single clean ``LineString`` so the downstream consumers
    (``define_floodplain_boundaries`` etc., which read ``geometry.iloc[0].coords``) never
    trip on hand-drawn geometry. Robust to MultiLineString and disjoint segments.
    """
    if gdf is None or len(gdf) == 0:
        raise ValueError(f"{name} is empty; a single boundary line is required.")
    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS set. Call .set_crs(...) before run_hyporheic().")

    g = gdf.to_crs(target_crs)
    geoms = [ge for ge in g.geometry if ge is not None and not ge.is_empty]
    if not geoms:
        raise ValueError(f"{name} has no usable geometry.")
    if len(geoms) == 1 and isinstance(geoms[0], LineString):
        # Single drawn/loaded line: use as-is to preserve vertex order and direction
        # (upstream/downstream sense matters downstream). linemerge() rejects a lone
        # LineString, so never route it through there.
        merged = geoms[0]
    else:
        u = unary_union(geoms)
        if isinstance(u, MultiLineString):
            lm = linemerge(u)  # stitch contiguous segments into one line
            merged = lm if isinstance(lm, LineString) else max(lm.geoms, key=lambda ls: ls.length)
        else:
            merged = u
    if not isinstance(merged, LineString):
        raise ValueError(f"{name} did not resolve to a LineString (got {type(merged).__name__}).")
    return gpd.GeoDataFrame(geometry=[merged], crs=target_crs)


def _inject_vectors(cfg: Settings,
                    domain_gdf: "gpd.GeoDataFrame",
                    left_gdf: "gpd.GeoDataFrame",
                    right_gdf: "gpd.GeoDataFrame",
                    kh_gdf: "gpd.GeoDataFrame | None" = None) -> None:
    """
    Replacement for ``Settings.setup_vectors`` that takes in-memory GeoDataFrames instead
    of reading shapefiles. Reprojects everything to the canonical target CRS so the vectors
    align with the reprojected rasters.
    """
    target = cfg.hec_ras_crs or cfg.project_crs
    cfg.project_crs = target

    if domain_gdf is None or len(domain_gdf) == 0:
        raise ValueError("domain_gdf is empty; a groundwater-domain polygon is required.")
    if domain_gdf.crs is None:
        raise ValueError("domain_gdf has no CRS set. Call .set_crs(...) before run_hyporheic().")

    cfg.ground_water_domain = domain_gdf.to_crs(target)
    cfg.left_boundary = _as_single_line_gdf(left_gdf, target, "left_line_gdf")
    cfg.right_boundary = _as_single_line_gdf(right_gdf, target, "right_line_gdf")

    if kh_gdf is not None and len(kh_gdf) > 0:
        if kh_gdf.crs is None:
            raise ValueError("kh_polygon_gdf has no CRS set.")
        cfg.kh_polygon_gdf = kh_gdf.to_crs(target)
    else:
        cfg.kh_polygon_gdf = None


def estimate_grid(domain_gdf: "gpd.GeoDataFrame",
                  crs: Any,
                  *,
                  cell_size_x: float = 10.0,
                  cell_size_y: float = 10.0,
                  gw_mod_depth: float = 20.0,
                  z: float = 0.5) -> Dict[str, Any]:
    """
    Cheap, raster-free grid-size estimate from the domain polygon's bounding box. Mirrors
    the engine's own grid math (``build_model_domain``: ncol = (xmax-xmin)/cell_size_x,
    nrow = (ymax-ymin)/cell_size_y, nlay = gw_mod_depth/z) so the web UI can give live
    feedback as the user drags the cell-size slider, without fetching a DEM or running MF6.

    Note: this uses the bbox, so ``n_cells_total`` is an *upper bound* on the number of
    active cells (the polygon clip in ``make_idomain`` reduces it).
    """
    g = domain_gdf.to_crs(crs)
    xmin, ymin, xmax, ymax = (float(v) for v in g.total_bounds)
    ncol = max(1, int((xmax - xmin) / float(cell_size_x)))
    nrow = max(1, int((ymax - ymin) / float(cell_size_y)))
    nlay = max(1, int(float(gw_mod_depth) / float(z)))
    return {
        "ncol": ncol, "nrow": nrow, "nlay": nlay,
        "n_cells_total": ncol * nrow * nlay,
        "bbox": (xmin, ymin, xmax, ymax),
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_hyporheic(*,
                  # geometry (in-memory; any CRS — reprojected internally)
                  domain_gdf: "gpd.GeoDataFrame",
                  left_line_gdf: "gpd.GeoDataFrame",
                  right_line_gdf: "gpd.GeoDataFrame",
                  crs: Any,
                  # rasters
                  dem_path: str | Path,
                  wse_path: str | Path | None = None,
                  wse_mode: str = "dem",
                  wse_relief_thresh: float = 0.2,
                  aerial_path: str | Path | None = None,
                  # optional K zones
                  kh_polygon_gdf: "gpd.GeoDataFrame | None" = None,
                  # numeric params (engine defaults mirror Settings)
                  cell_size_x: float = 10.0,
                  cell_size_y: float = 10.0,
                  gw_mod_depth: float = 20.0,
                  z: float = 0.5,
                  kh: float = 10.0,
                  kv: float = 1.0,
                  porosity: float = 0.3,
                  length_units: str = "feet",
                  time_units: str = "days",
                  nper: int = 1,
                  nstp: int = 1,
                  perlen: float = 1.0,
                  tsmult: float = 1.0,
                  contour_interval: float = 0.5,
                  # boundary conditions
                  boundary_condition_mode: str = "4 Corner Gradients",
                  upstream_left_fpl_gw_gradient: float = 0.01,
                  upstream_right_fpl_gw_gradient: float = 0.01,
                  downstream_left_fpl_gw_gradient: float = 0.01,
                  downstream_right_fpl_gw_gradient: float = 0.01,
                  left_boundary_gradient_profile: str | None = None,
                  right_boundary_gradient_profile: str | None = None,
                  # runtime
                  work_dir: str | Path,
                  sim_name: str = "hyporheic",
                  modflow_bin_dir: str | Path | None = None,
                  log: Callable[[str], None] = print,
                  make_figures: bool = False,
                  estimate_only: bool = False) -> Dict[str, Optional[Any]]:
    """
    Build and run a MODFLOW 6 + MODPATH 7 hyporheic model from in-memory geometries and a
    terrain DEM. See module docstring for the input model. Returns the artifact-path dict
    from the shared pipeline plus a ``"grid"`` sub-dict.

    If ``estimate_only=True``, only the workspace + terrain reprojection + grid sizing are
    computed (no MODFLOW run); the return value is ``{"grid": {...}}``.
    """
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    dem_path = Path(dem_path).expanduser().resolve()
    aerial_resolved = Path(aerial_path).expanduser().resolve() if aerial_path else None
    bin_resolved = Path(modflow_bin_dir).expanduser().resolve() if modflow_bin_dir else None

    # Resolve the water-surface-elevation raster:
    #   * an explicit upload (wse_path) always wins;
    #   * wse_mode="channel" derives a channel-only WSE from the (hydro-flattened) DEM;
    #   * wse_mode="dem" (default) uses the full DEM as the WSE.
    if wse_path:
        wse_resolved = Path(wse_path).expanduser().resolve()
    elif wse_mode == "channel" and not estimate_only:
        from hypetool.functions.wse_utils import derive_channel_wse
        ch_out = work_dir / "inputs" / "wse_channel.tif"
        try:
            derived = derive_channel_wse(dem_path, ch_out, domain_gdf=domain_gdf,
                                         relief_thresh=wse_relief_thresh, log=log)
        except Exception as e:  # noqa: BLE001
            log(f"[WARN] channel-WSE derivation failed ({e}); using the DEM as WSE.")
            derived = None
        wse_resolved = Path(derived) if derived else dem_path
    else:
        wse_resolved = dem_path

    # Normalize enum-like strings to the patterns Settings enforces.
    length_units = str(length_units).strip().lower()
    time_units = str(time_units).strip().lower()

    crs_obj = CRS.from_user_input(crs)

    # Build Settings programmatically. All path args are absolute, so the path validator
    # short-circuits (no YAML cfg_dir context needed).
    cfg = Settings(
        output_directory=work_dir,
        terrain_elevation_raster=dem_path,
        water_surface_elevation_raster=wse_resolved,
        aerial_raster=aerial_resolved,
        projection_file=None,
        sim_name=sim_name,
        length_units=length_units,
        time_units=time_units,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        gw_mod_depth=gw_mod_depth,
        z=z,
        kh=kh,
        kv=kv,
        porosity=porosity,
        nper=nper,
        nstp=nstp,
        perlen=perlen,
        tsmult=tsmult,
        contour_interval=contour_interval,
        boundary_condition_mode=boundary_condition_mode,
        upstream_left_fpl_gw_gradient=upstream_left_fpl_gw_gradient,
        upstream_right_fpl_gw_gradient=upstream_right_fpl_gw_gradient,
        downstream_left_fpl_gw_gradient=downstream_left_fpl_gw_gradient,
        downstream_right_fpl_gw_gradient=downstream_right_fpl_gw_gradient,
        left_boundary_gradient_profile=left_boundary_gradient_profile,
        right_boundary_gradient_profile=right_boundary_gradient_profile,
        kh_polygon=bool(kh_polygon_gdf is not None and len(kh_polygon_gdf) > 0),
        modflow_bin_dir=bin_resolved,
    )

    # Workspace + executables (no-op-safe on Linux via platform-aware resolution).
    cfg.setup_workspace(clean=False)
    _ensure_executables(cfg, log)

    # CRS is supplied directly (replaces setup_projection / no .prj file).
    cfg.hec_ras_crs = crs_obj
    cfg.project_crs = crs_obj

    log(f"Headless run - work_dir={work_dir}")
    log(f"Target CRS: {crs_obj.to_string()}")
    log(f"Terrain DEM: {dem_path.name}")
    if wse_path:
        log(f"WSE source: uploaded ({wse_resolved.name})")
    elif wse_mode == "channel" and wse_resolved != dem_path:
        log(f"WSE source: channel mask from DEM ({wse_resolved.name})")
    else:
        log("WSE source: full DEM")

    # Terrain reprojection is needed for both estimate and full run.
    cfg.setup_terrain(crs_obj)

    if estimate_only:
        dom = myu.build_model_domain(cfg)
        ncol, nrow, nlay = int(dom["ncol"]), int(dom["nrow"]), int(dom["nlay"])
        return {"grid": {
            "ncol": ncol, "nrow": nrow, "nlay": nlay,
            "n_cells_total": ncol * nrow * nlay,
        }}

    # Water-surface raster (defaults to the DEM) + in-memory vectors.
    cfg.setup_water_surface(crs_obj)
    _inject_vectors(cfg, domain_gdf, left_line_gdf, right_line_gdf, kh_polygon_gdf)

    # Run the shared STEP 2+ pipeline; skip the ArcGIS Pro map-group step.
    result = _run_pipeline(cfg, log=log, make_figures=make_figures, add_to_map=False)

    try:
        ncol, nrow, nlay = int(cfg.ncol), int(cfg.nrow), int(cfg.nlay)
        result["grid"] = {
            "ncol": ncol, "nrow": nrow, "nlay": nlay,
            "n_cells_total": ncol * nrow * nlay,
        }
    except Exception:
        pass
    return result


__all__ = ["run_hyporheic", "estimate_grid"]
