# Unified entry point (toolbox + CLI). Now also exports MODFLOW head layers,
# builds per-layer contour feature classes (ALL layers by default), and
# adds pathlines, points, contours, and netCDF/mosaic to a single Group Layer.

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, List
from shapely.geometry import Point
import numpy as np

# Optional writers for netCDF (xarray preferred; fallback to netCDF4)
try:
    import xarray as xr
    _HAS_XARRAY = True
except Exception:
    _HAS_XARRAY = False

try:
    from netCDF4 import Dataset as _NC4
    _HAS_NETCDF4 = True
except Exception:
    _HAS_NETCDF4 = False

from hypetool.inputs import load, Settings
from hypetool.functions import my_utils as myu
from hypetool.functions import path_utils as pu


# ------------------------- Executables -------------------------

def _ensure_executables(cfg: Settings, log: Callable[[str], None]) -> None:
    """Ensure mf6/mp7 are available; set paths; prepend to PATH."""
    if cfg.md6_exe_path and Path(cfg.md6_exe_path).exists():
        pu.add_modflow_executables(Path(cfg.md6_exe_path).parent)
        log(f"Using MODFLOW 6 exe: {cfg.md6_exe_path}")
    if cfg.md7_exe_path and Path(cfg.md7_exe_path).exists():
        pu.add_modflow_executables(Path(cfg.md7_exe_path).parent)
        log(f"Using MODPATH 7 exe: {cfg.md7_exe_path}")
    if (cfg.md6_exe_path and Path(cfg.md6_exe_path).exists()) or (cfg.md7_exe_path and Path(cfg.md7_exe_path).exists()):
        return

    if cfg.modflow_bin_dir and Path(cfg.modflow_bin_dir).exists():
        add_dir = Path(cfg.modflow_bin_dir)
        pu.add_modflow_executables(add_dir)
        exes = pu.detect_modflow_exes(add_dir)
        if not cfg.md6_exe_path and exes["mf6"]:
            cfg.md6_exe_path = exes["mf6"]
        if not cfg.md7_exe_path and exes["mp7"]:
            cfg.md7_exe_path = exes["mp7"]
        log(f"Using MODFLOW bin directory: {add_dir}")
        return

    packaged = pu.default_modflow_bin()
    if packaged.exists():
        pu.add_modflow_executables(packaged)
        exes = pu.detect_modflow_exes(packaged)
        if not cfg.md6_exe_path and exes["mf6"]:
            cfg.md6_exe_path = exes["mf6"]
        if not cfg.md7_exe_path and exes["mp7"]:
            cfg.md7_exe_path = exes["mp7"]
        log(f"Using packaged MODFLOW bin: {packaged}")
        return

    log("[WARN] Could not find packaged or YAML-specified mf6/mp7. "
        "Will rely on system PATH. Set 'modflow_bin_dir' or 'md6_exe_path'/'md7_exe_path' in YAML if this fails.")


# ------------------------- ArcGIS helpers -------------------------

def _results_group_name(sim_name: str | None) -> str:
    """e.g., 'Hyporheic Results 2025-09-10 17:01'"""
    sname = (sim_name or "Hyporheic").strip()
    ts = time.strftime("%Y-%m-%d %H:%M")
    return f"{sname} Results {ts}"

def _get_or_create_group_layer(*, aprx, map_, name: str):
    """Return an ArcGIS Pro Group Layer with the given name, creating it if needed."""
    for lyr in map_.listLayers():
        try:
            if lyr.isGroupLayer and lyr.name == name:
                return lyr
        except Exception:
            pass

    try:
        grp = map_.createGroupLayer(name)
    except Exception:
        import arcpy
        res = arcpy.management.CreateGroupLayer(name)
        grp = res.getOutput(0)
        map_.addLayer(grp)
    return grp


def _add_layer_to_group(map_, group_lyr, lyr_obj):
    """Add a layer object to a group (with backward-compatible API)."""
    try:
        map_.addLayerToGroup(group_lyr, lyr_obj, "BOTTOM")
    except Exception:
        map_.addLayerToGroup(group_lyr, lyr_obj)


def _safe_make_feature_layer(path: str, out_name: str):
    import arcpy
    return arcpy.management.MakeFeatureLayer(path, out_name).getOutput(0)


def _safe_add_data_from_path(map_, path: str):
    """Best-effort way to get a Layer object from a dataset path (mosaic datasets, etc.)."""
    import arcpy
    try:
        if path.lower().endswith(".gdb") or ".gdb" in path:
            try:
                md_layer = arcpy.management.MakeMosaicLayer(path, Path(path).name).getOutput(0)
                return md_layer
            except Exception:
                pass
        lyr = map_.addDataFromPath(path)
        return lyr
    except Exception:
        return None


# ---------- Build contours from per-layer GeoTIFFs and add to map ----------

def _build_head_contours_and_add_to_map(
    geotiffs: List[str],
    base_dir: Path,
    *,
    contour_interval: float = 0.5,
    max_layers: Optional[int] = None,
    group_name: Optional[str] = None,
    units_label: str = "ft",
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """
    Create contour feature classes for each GeoTIFF slice and add them to a Group Layer.
    Returns dict with keys: gdb, layers (list of FC paths actually created), group_name.
    """
    out = {"gdb": None, "layers": [], "group_name": None}

    if not geotiffs:
        log("[INFO] No per-layer GeoTIFFs found; skipping contour generation.")
        return out

    try:
        import arcpy
        from arcpy.sa import Contour
        arcpy.CheckOutExtension("Spatial")
        arcpy.env.overwriteOutput = True
    except Exception as e:
        log(f"[WARN] ArcPy/Spatial Analyst not available; skipping contour creation. ({e})")
        return out

    # Output gdb
    gdb = base_dir / "head_contours.gdb"
    if not gdb.exists():
        arcpy.management.CreateFileGDB(str(base_dir), gdb.name)
    out["gdb"] = str(gdb)

    # How many layers to contour?
    total = len(geotiffs)
    if max_layers is None:
        try:
            max_layers = int(os.environ.get("HYP_MAX_CONTOUR_LAYERS", str(total)))
        except Exception:
            max_layers = total
    if int(max_layers) <= 0:
        log("Skipping contour generation (HYP_MAX_CONTOUR_LAYERS=0).")
        return out

    n = min(total, max(1, int(max_layers)))
    log(f"Generating contours for first {n} of {total} layers (set HYP_MAX_CONTOUR_LAYERS to change).")

    # Build one FC per slice, with human-readable names including the interval and units
    fc_paths: List[str] = []
    for idx, tif in enumerate(geotiffs[:n], start=1):
        log(f"  [{idx}/{n}] Contouring {Path(tif).name} â€¦")
        fc = gdb / f"contours_L{idx:02d}"
        if arcpy.Exists(str(fc)):
            arcpy.management.Delete(str(fc))
        Contour(
            in_raster=str(tif),
            out_polyline_features=str(fc),
            contour_interval=float(contour_interval)
        )
        fc_paths.append(str(fc))
    out["layers"] = fc_paths

    # Add all contour FCs under a single group layer
    try:
        import arcpy
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        m = aprx.activeMap

        group_name = group_name or os.environ.get("HYP_GROUP_NAME", "Hyporheic Results")
        group_lyr = _get_or_create_group_layer(aprx=aprx, map_=m, name=group_name)
        out["group_name"] = group_name

        for idx, fc in enumerate(fc_paths, start=1):
            child_name = f"Head Contours L{idx:02d} ({contour_interval} {units_label})"
            lyr_obj = _safe_make_feature_layer(fc, child_name)
            _add_layer_to_group(m, group_lyr, lyr_obj)

        try:
            arcpy.RefreshActiveView()
        except Exception:
            pass
    except Exception as e:
        log(f"[WARN] Contours created, but could not add them to the current map: {e}")

    return out


def _add_products_to_group(*,
                           cfg: Settings,
                           artifacts: Dict[str, str],
                           head_info: Dict[str, str],
                           group_name: str,
                           log: Callable[[str], None] = print) -> None:
    """
    Add pathlines, points, and rasters into a single group layer.
    Preference order for rasters:
        1) Combined multidimensional netCDF with variables head, K, K33 (head_info['netcdf_multi'])
        2) Head-only netCDF (head_info['netcdf'])
        3) Mosaic dataset (head_info['mosaic_dataset'])
    Also adds expected shapefiles (by exact expected name) and any artifacts provided.
    """
    def _norm(p: str) -> str:
        return os.path.normpath(p) if p else p

    try:
        import arcpy
        arcpy.env.overwriteOutput = True
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        m = aprx.activeMap
        grp = _get_or_create_group_layer(aprx=aprx, map_=m, name=group_name)

        def _exists_any(path: str) -> bool:
            try:
                return bool(path) and (arcpy.Exists(path) or Path(path).exists())
            except Exception:
                return bool(path) and Path(path).exists()

        # Deduplicate by normalized path
        added_paths = set()

        def _add_feature(path: Optional[str], label: Optional[str] = None):
            if not path:
                return
            p = _norm(path)
            if p in added_paths:
                return
            if _exists_any(p):
                lyr = _safe_make_feature_layer(p, label or Path(p).stem)
                _add_layer_to_group(m, grp, lyr)
                added_paths.add(p)

        # ------------- Expected named shapefiles (search if not present in artifacts) -------------
        summary_dir = Path(cfg.output_directory) / "summary"

        expected_names = [
            "Forward_hyporheic_points_HECRAS_CRS.shp",
            "Forward_full_pathlines_3D.shp",
            "Forward_hyporheic_pathlines_3D_HECRAS_CRS.shp",
            "Forward_hyporheic_pathlines_2D_HECRAS_CRS.shp",
        ]

        def _search_by_name(root: Path, fname: str) -> Optional[str]:
            try:
                for hit in root.rglob(fname):
                    if hit.suffix.lower() == ".shp":
                        return str(hit)
            except Exception:
                return None
            return None

        # Use artifacts when possible, but also try exact filename search to match your request
        # Artifacts keys from processing step
        art_candidates = [
            artifacts.get("lines_fc_3d"),
            artifacts.get("lines_shp_3d"),
            artifacts.get("lines_fc_3d_full"),
            artifacts.get("lines_shp_3d_full"),
            artifacts.get("lines_shp"),
            artifacts.get("lines_shp_wgs84"),
            artifacts.get("points_shp"),
            artifacts.get("points_shp_wgs84"),
        ]
        for cand in art_candidates:
            _add_feature(cand)  # add with dataset stem as label

        for fname in expected_names:
            # If any already-added path endswith this filename, skip
            if any(Path(p).name.lower() == fname.lower() for p in added_paths):
                continue
            # Search under summary dir
            found = _search_by_name(summary_dir, fname)
            if found:
                _add_feature(found, Path(found).stem)

        # -------------------- Raster products (netCDFs/mosaic) --------------------
        added_raster = False

        # 1) Combined multidimensional netCDF (head, K, K33) if available
        nc_multi = (head_info or {}).get("netcdf_multi") if isinstance(head_info, dict) else None
        if nc_multi and _exists_any(nc_multi):
            try:
                try:
                    lyr_head = arcpy.md.MakeMultidimensionalRasterLayer(
                        nc_multi, "Head (netCDF)", variables="head"
                    ).getOutput(0)
                    _add_layer_to_group(m, grp, lyr_head)
                except Exception as e:
                    log(f"[WARN] Could not add Head variable from netCDF: {e}")

                try:
                    lyr_k = arcpy.md.MakeMultidimensionalRasterLayer(
                        nc_multi, "K (netCDF)", variables="K"
                    ).getOutput(0)
                    _add_layer_to_group(m, grp, lyr_k)
                except Exception as e:
                    log(f"[WARN] Could not add K variable from netCDF: {e}")

                try:
                    lyr_k33 = arcpy.md.MakeMultidimensionalRasterLayer(
                        nc_multi, "K33 (netCDF)", variables="K33"
                    ).getOutput(0)
                    _add_layer_to_group(m, grp, lyr_k33)
                except Exception:
                    pass

                added_raster = True
            except Exception as e:
                log(f"[WARN] Could not add combined netCDF; will try head-only netCDF/mosaic. ({e})")

        # 2) Head-only netCDF (original behavior)
        if not added_raster:
            nc = (head_info or {}).get("netcdf") if isinstance(head_info, dict) else None
            if nc and _exists_any(nc):
                try:
                    try:
                        nc_vars = (head_info or {}).get("netcdf_variables", ["head"]) or ["head"]
                        var_str = ";".join(nc_vars)
                        md_layer = arcpy.md.MakeMultidimensionalRasterLayer(
                            nc, "Head (netCDF)", variables=var_str
                        ).getOutput(0)
                    except Exception:
                        md_layer = None
                        try:
                            md_layer = arcpy.md.MakeNetCDFRasterLayer(
                                nc, "head", "x", "y", "Head (netCDF)", band_dimension="z"
                            ).getOutput(0)
                        except Exception:
                            md_layer = arcpy.md.MakeNetCDFRasterLayer(
                                nc, "head", "x", "y", "Head (netCDF)", band_dimension="", z_dimension="z"
                            ).getOutput(0)
                    _add_layer_to_group(m, grp, md_layer)
                    added_raster = True
                except Exception as e:
                    log(f"[WARN] Could not add netCDF layer; will try mosaic dataset instead. ({e})")

        # 3) Mosaic dataset
        if not added_raster:
            md_path = (head_info or {}).get("mosaic_dataset") if isinstance(head_info, dict) else None
            if md_path:
                try:
                    md_lyr = arcpy.management.MakeMosaicLayer(md_path, "Head (Mosaic)").getOutput(0)
                    _add_layer_to_group(m, grp, md_lyr)
                    added_raster = True
                except Exception:
                    md_lyr = _safe_add_data_from_path(m, md_path)
                    if md_lyr:
                        _add_layer_to_group(m, grp, md_lyr)
                        added_raster = True

        # Iterate through the list of layers in the group and set visibility to False
        for lyr in grp.listLayers():
            try:
                lyr.visible = False
            except Exception as e:
                log(f"[WARN] Could not set visibility for layer {lyr.name}: {e}")

        try:
            arcpy.RefreshActiveView()
        except Exception:
            pass

    except Exception as e:
        log(f"[WARN] Could not add outputs to map group: {e}")


# ------------------------- NEW: helpers to build combined netCDF -------------------------

def _expand_to_3d(arr_like, nlay: int, nrow: int, ncol: int) -> np.ndarray:
    """
    Expand FloPy-style scalar / per-layer / 2D inputs into a 3-D (nlay,nrow,ncol) array.
    Accepts:
        - scalar
        - 1D list/array of length nlay (each item can be scalar or 2D (nrow,ncol))
        - 2D array (nrow,ncol) -> broadcast to all layers
        - 3D array (nlay,nrow,ncol) -> returned as-is
    """
    if arr_like is None:
        return None
    a = arr_like
    # Unwrap FloPy DataInterface (sometimes has .array)
    try:
        if hasattr(a, "array"):
            a = a.array
    except Exception:
        pass

    a = np.asarray(a)

    if a.ndim == 0:  # scalar
        return np.full((nlay, nrow, ncol), float(a))
    if a.ndim == 2 and a.shape == (nrow, ncol):
        return np.repeat(a[np.newaxis, :, :], nlay, axis=0)
    if a.ndim == 3 and a.shape == (nlay, nrow, ncol):
        return a

    # Handle list/1D "layered"
    if a.ndim == 1 and a.size == nlay:
        out = np.zeros((nlay, nrow, ncol), dtype=float)
        for k in range(nlay):
            v = a[k]
            if hasattr(v, "array"):
                v = v.array
            v = np.asarray(v)
            if v.ndim == 0:
                out[k, :, :] = float(v)
            elif v.ndim == 2 and v.shape == (nrow, ncol):
                out[k, :, :] = v
            else:
                raise ValueError(f"Unsupported layer spec at layer {k}: shape {v.shape}")
        return out

    raise ValueError(f"Unsupported array-like shape for expansion: {a.shape}")


def _find_headfile(gwf_ws: Path) -> Optional[Path]:
    """Find a *.hds (or *.hdf) head file in the GWF workspace."""
    for ext in ("*.hds", "*.hdf", "*.bhd"):
        hits = list(Path(gwf_ws).glob(ext))
        if hits:
            return hits[0]
    return None

def _write_combined_nc_with_xarray(
    out_nc: Path,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    head: Optional[np.ndarray],
    K: Optional[np.ndarray],
    K33: Optional[np.ndarray],
    attrs: Dict[str, str],
) -> None:
    import xarray as xr
    import numpy as np

    FILL = np.float32(-9999.0)
    BAD  = np.float32(1.0e20)  # treat any |v| >= BAD as NoData (e.g., -1e30)

    def _prep(a):
        if a is None:
            return None
        b = np.asarray(a, dtype="float32")
        mask = (~np.isfinite(b)) | (np.abs(b) >= BAD) | np.isclose(b, FILL)
        b = np.where(mask, FILL, b).astype("float32")
        return b

    head_f = _prep(head)
    K_f    = _prep(K)
    K33_f  = _prep(K33)

    coords = {
        "x": ("x", x_coords.astype("float64")),
        "y": ("y", y_coords.astype("float64")),
        "z": ("z", z_coords.astype("float64")),
    }
    ds_vars = {}
    if head_f is not None:
        ds_vars["head"] = (("z", "y", "x"), head_f)
    if K_f is not None:
        ds_vars["K"] = (("z", "y", "x"), K_f)
    if K33_f is not None:
        ds_vars["K33"] = (("z", "y", "x"), K33_f)

    ds = xr.Dataset(ds_vars, coords=coords, attrs=attrs or {})

    # Variable attributes
    if "head" in ds:
        ds["head"].attrs.update({
            "units": attrs.get("length_units", ""),
            "long_name": "Hydraulic head",
            "coordinates": "z y x",
            "grid_mapping": "spatial_ref",
            "missing_value": FILL,
        })
    if "K" in ds:
        ds["K"].attrs.update({
            "units": attrs.get("k_units", "L/T"),
            "long_name": "Hydraulic conductivity (K11)",
            "coordinates": "z y x",
            "grid_mapping": "spatial_ref",
            "missing_value": FILL,
        })
    if "K33" in ds:
        ds["K33"].attrs.update({
            "units": attrs.get("k_units", "L/T"),
            "long_name": "Vertical hydraulic conductivity (K33)",
            "coordinates": "z y x",
            "grid_mapping": "spatial_ref",
            "missing_value": FILL,
        })

    # Axis metadata
    xy_units = attrs.get("xy_units", "")
    ds["x"].attrs.update({
        "standard_name": "projection_x_coordinate",
        "long_name": "x coordinate of projection",
        "axis": "X",
        "units": xy_units,
    })
    ds["y"].attrs.update({
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of projection",
        "axis": "Y",
        "positive": "up",
        "units": xy_units,
    })
    ds["z"].attrs.update({"axis": "Z", "long_name": "Layer index (top=1)"})

    # Grid mapping (write both 'spatial_ref' and 'esri_pe_string' for ArcGIS/GDAL)
    wkt = (attrs.get("crs_wkt", "") or "").strip()
    ds["spatial_ref"] = xr.DataArray(0)
    if wkt:
        ds["spatial_ref"].attrs["spatial_ref"] = wkt
        ds["spatial_ref"].attrs["esri_pe_string"] = wkt
    ds["spatial_ref"].attrs["long_name"] = "CRS definition"

    # Encoding: set _FillValue + compression
    enc = {}
    for v in ("head", "K", "K33"):
        if v in ds:
            enc[v] = {"zlib": True, "complevel": 4, "_FillValue": FILL, "dtype": "float32"}

    ds.to_netcdf(str(out_nc), engine="netcdf4", encoding=enc)

def _write_combined_nc_with_netCDF4(
    out_nc: Path,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    head: Optional[np.ndarray],
    K: Optional[np.ndarray],
    K33: Optional[np.ndarray],
    attrs: Dict[str, str],
) -> None:
    import numpy as np

    FILL = np.float32(-9999.0)
    BAD  = np.float32(1.0e20)

    with _NC4(str(out_nc), "w") as nc:
        # Dimensions
        nc.createDimension("x", x_coords.size)
        nc.createDimension("y", y_coords.size)
        nc.createDimension("z", z_coords.size)

        # Coord vars
        xvar = nc.createVariable("x", "f8", ("x",))
        yvar = nc.createVariable("y", "f8", ("y",))
        zvar = nc.createVariable("z", "f8", ("z",))
        xvar[:] = x_coords
        yvar[:] = y_coords
        zvar[:] = z_coords

        xvar.standard_name = "projection_x_coordinate"
        xvar.long_name = "x coordinate of projection"
        xvar.axis = "X"
        xvar.units = attrs.get("xy_units", "")

        yvar.standard_name = "projection_y_coordinate"
        yvar.long_name = "y coordinate of projection"
        yvar.axis = "Y"
        yvar.positive = "up"
        yvar.units = attrs.get("xy_units", "")

        zvar.long_name = "Layer index (top=1)"
        zvar.axis = "Z"

        # Scalar grid-mapping variable with WKT for ArcGIS/GDAL
        sref = nc.createVariable("spatial_ref", "i4")
        wkt = (attrs.get("crs_wkt", "") or "").strip()
        if wkt:
            try:
                sref.spatial_ref = wkt
            except Exception:
                pass
            try:
                sref.esri_pe_string = wkt
            except Exception:
                pass
        try:
            sref.long_name = "CRS definition"
        except Exception:
            pass

        def _write_var(name, data, units, long_name):
            v = nc.createVariable(
                name, "f4", ("z", "y", "x"),
                zlib=True, complevel=4, fill_value=FILL
            )
            arr = np.asarray(data, dtype="float32")
            mask = (~np.isfinite(arr)) | (np.abs(arr) >= BAD) | np.isclose(arr, FILL)
            arr = np.where(mask, FILL, arr).astype("float32")
            v[:, :, :] = arr
            v.long_name = long_name
            v.units = units
            v.coordinates = "z y x"
            v.grid_mapping = "spatial_ref"
            v.missing_value = FILL  # extra compatibility
            return v

        if head is not None:
            _write_var("head", head, attrs.get("length_units", ""), "Hydraulic head")
        if K is not None:
            _write_var("K", K, attrs.get("k_units", "L/T"), "Hydraulic conductivity (K11)")
        if K33 is not None:
            _write_var("K33", K33, attrs.get("k_units", "L/T"), "Vertical hydraulic conductivity (K33)")

        # Globals (unchanged)
        for k, v in (attrs or {}).items():
            try:
                setattr(nc, k, str(v))
            except Exception:
                pass



def _load_reference_xy_head(cfg: Settings) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    If the head-only netCDF (summary/head/head_zyx.nc) exists, return:
      (x_ref, y_ref, head_ref) as float arrays; else (None, None, None).
    Using the head reference guarantees our combined netCDF aligns in ArcGIS Pro.
    """
    ref_nc = Path(cfg.output_directory) / "summary" / "head" / "head_zyx.nc"
    if not ref_nc.exists():
        return None, None, None

    x_ref = y_ref = head_ref = None
    try:
        if _HAS_XARRAY:
            with xr.open_dataset(ref_nc) as ds:
                if "x" in ds: x_ref = np.asarray(ds["x"].values, dtype=float)
                if "y" in ds: y_ref = np.asarray(ds["y"].values, dtype=float)
                if "head" in ds: head_ref = np.asarray(ds["head"].values, dtype=float)  # (z,y,x)
        elif _HAS_NETCDF4:
            with _NC4(str(ref_nc), "r") as ds:
                if "x" in ds.variables: x_ref = np.asarray(ds.variables["x"][:], dtype=float)
                if "y" in ds.variables: y_ref = np.asarray(ds.variables["y"][:], dtype=float)
                if "head" in ds.variables: head_ref = np.asarray(ds.variables["head"][:], dtype=float)
    except Exception:
        # Fall through to safer defaults if anything goes wrong
        x_ref = y_ref = head_ref = None

    return x_ref, y_ref, head_ref

def _build_combined_head_K_nc(
    *,
    cfg: Settings,
    gwf,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    nlay: int,
    nrow: int,
    ncol: int,
    log: Callable[[str], None] = print,
) -> Optional[Path]:
    from flopy.utils import HeadFile
    import numpy as np

    out_dir = Path(cfg.output_directory) / "summary" / "head"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / "model_vars.nc"

    # ---------- resolve WKT (robust) ----------
    def _resolve_wkt() -> str:
        # 1) explicit WKT from YAML
        w = (getattr(cfg, "projection_wkt", "") or "").strip()
        if w:
            return w
        # 2) .prj file
        try:
            prj = getattr(cfg, "projection_file", None)
            if prj and Path(prj).exists():
                txt = Path(prj).read_text().strip()
                if txt:
                    return txt
        except Exception:
            pass
        # 3) hec_ras_crs / raster CRS / modelgrid CRS via pyproj/rasterio
        for cand in (getattr(cfg, "hec_ras_crs", None),
                     getattr(cfg, "raster_crs", None),
                     getattr(getattr(gwf, "modelgrid", None), "crs", None)):
            if not cand:
                continue
            try:
                # try pyproj
                from pyproj import CRS
                try:
                    crs = CRS.from_user_input(cand)
                except Exception:
                    # rasterio or string fallback
                    try:
                        import rasterio
                        from rasterio.crs import CRS as RioCRS
                        crs = CRS.from_wkt(RioCRS.from_user_input(cand).to_wkt())
                    except Exception:
                        crs = CRS.from_user_input(str(cand))
                # prefer ESRI-style WKT when available; else default WKT
                try:
                    from pyproj.enums import WKTVersion
                    return crs.to_wkt(WKTVersion.WKT1_ESRI)  # best for ArcGIS
                except Exception:
                    return crs.to_wkt()
            except Exception:
                # last resort: str()
                try:
                    return str(cand)
                except Exception:
                    pass
        return ""

    wkt_str = _resolve_wkt()
    xy_units = str(getattr(cfg, "xy_units", "")) or ""  # optional cosmetic

    # ---------- 1) Reference axes + head from head_zyx.nc, if present ----------
    x_ref, y_ref, head_ref = _load_reference_xy_head(cfg)

    # ---------- 2) Fallback x/y from DIS (1-D cell-center coords) ----------
    try:
        dis = gwf.get_package("DIS")
    except Exception:
        dis = None

    def _as_float(v, default=0.0) -> float:
        try:
            if hasattr(v, "get_data"):
                return float(np.asarray(v.get_data()).squeeze())
            return float(np.asarray(v).squeeze())
        except Exception:
            return float(default)

    x_coords_fb = y_coords_fb = None
    if dis is not None:
        delr = np.asarray(dis.delr.array, dtype=float)
        delc = np.asarray(dis.delc.array, dtype=float)
        uniform_x = np.allclose(delr, delr.flat[0])
        uniform_y = np.allclose(delc, delc.flat[0])
        dx = float(delr.flat[0]) if uniform_x else float(np.mean(delr))
        dy = float(delc.flat[0]) if uniform_y else float(np.mean(delc))

        xorigin = _as_float(getattr(dis, "xorigin", getattr(gwf.modelgrid, "xoffset", 0.0)),
                            getattr(gwf.modelgrid, "xoffset", 0.0))
        yorigin = _as_float(getattr(dis, "yorigin", getattr(gwf.modelgrid, "yoffset", 0.0)),
                            getattr(gwf.modelgrid, "yoffset", 0.0))

        x_coords_fb = xorigin + dx * (np.arange(ncol, dtype=float) + 0.5)
        y_coords_fb = yorigin + dy * (np.arange(nrow, dtype=float) + 0.5)

        angrot = float(getattr(gwf.modelgrid, "angrot", 0.0) or 0.0)
        if (abs(angrot) > 1e-9) or (not uniform_x) or (not uniform_y):
            log("[WARN] Grid has rotation and/or non-uniform spacing. Combined netCDF will use "
                "1D cell-center coordinates (no rotation). For exact placement in Pro, use GeoTIFFs/mosaic.")
    else:
        if grid_x.size >= 2 and grid_y.size >= 2:
            dx = float(grid_x[0, 1] - grid_x[0, 0])
            dy = float(grid_y[1, 0] - grid_y[0, 0])
        else:
            dx = dy = 1.0
        xorigin = float(grid_x[0, 0] - 0.5 * dx)
        yorigin = float(grid_y[0, 0] - 0.5 * dy)
        x_coords_fb = xorigin + dx * (np.arange(ncol, dtype=float) + 0.5)
        y_coords_fb = yorigin + dy * (np.arange(nrow, dtype=float) + 0.5)

    x_coords = x_ref if x_ref is not None else x_coords_fb
    y_coords = y_ref if y_ref is not None else y_coords_fb
    if x_coords is None or y_coords is None:
        log("[WARN] Could not determine x/y coordinates for combined netCDF.")
        return None

    # ---------- 3) Read/expand K and K33 ----------
    K_3d = K33_3d = None
    try:
        npf = getattr(gwf, "npf", None) or gwf.get_package("npf")
    except Exception:
        npf = None
    if npf is not None:
        try:
            K_3d = _expand_to_3d(getattr(npf, "k", None), nlay, nrow, ncol)
        except Exception as e:
            log(f"[WARN] Could not expand NPF.k to 3D: {e}")
        try:
            K33_3d = _expand_to_3d(getattr(npf, "k33", None), nlay, nrow, ncol)
        except Exception:
            K33_3d = None

    # ---------- 4) Head to write ----------
    head_to_write = head_ref
    if head_to_write is None:
        try:
            hds_path = _find_headfile(Path(cfg.gwf_ws))
            if hds_path and Path(hds_path).exists():
                from flopy.utils import HeadFile
                hf = HeadFile(str(hds_path))
                times = hf.get_times()
                head_to_write = np.asarray(hf.get_data(totim=times[-1] if times else None), dtype=float)
        except Exception as e:
            log(f"[WARN] Could not read head file for combined netCDF: {e}")
            head_to_write = None

    # ---------- 5) Align K/K33 on Y with ascending y-axis (head untouched) ----------
    y_is_ascending = bool(y_coords[0] < y_coords[-1])
    if y_is_ascending:
        if K_3d is not None:
            K_3d = K_3d[:, ::-1, :]
        if K33_3d is not None:
            K33_3d = K33_3d[:, ::-1, :]

    # ---------- 6) z coords ----------
    try:
        zc = np.asarray(getattr(cfg, "z", None))
        if zc is not None and zc.ndim == 1 and zc.size == nlay:
            z_coords = zc.astype(float)
        else:
            z_coords = np.arange(1, nlay + 1, dtype=float)
    except Exception:
        z_coords = np.arange(1, nlay + 1, dtype=float)

    # ---------- 7) write ----------
    attrs = {
        "title": f"{cfg.sim_name} â€” Heads and Hydraulic Conductivity",
        "source": "hypetool run_from_yaml.py",
        "Conventions": "CF-1.7",
        "length_units": str(getattr(cfg, "length_units", "") or ""),
        "k_units": str(getattr(cfg, "k_units", "L/T") or "L/T"),
        "crs_wkt": (lambda: (
            (getattr(cfg, "projection_wkt", "") or "").strip() or
            (Path(getattr(cfg, "projection_file", "")).read_text().strip() if getattr(cfg, "projection_file", None) and Path(getattr(cfg, "projection_file")).exists() else "")
        ))(),
        "xy_units": str(getattr(cfg, "xy_units", "") or ""),
    }

    try:
        if _HAS_XARRAY:
            _write_combined_nc_with_xarray(
                out_nc,
                x_coords=np.asarray(x_coords, dtype=float),
                y_coords=np.asarray(y_coords, dtype=float),
                z_coords=np.asarray(z_coords, dtype=float),
                head=head_to_write,
                K=K_3d,
                K33=K33_3d,
                attrs=attrs,
            )
        elif _HAS_NETCDF4:
            _write_combined_nc_with_netCDF4(
                out_nc,
                x_coords=np.asarray(x_coords, dtype=float),
                y_coords=np.asarray(y_coords, dtype=float),
                z_coords=np.asarray(z_coords, dtype=float),
                head=head_to_write,
                K=K_3d,
                K33=K33_3d,
                attrs=attrs,
            )
        else:
            raise RuntimeError("Neither xarray nor netCDF4 is installed. Install one to write netCDF.")
        log(f"[OK] Wrote combined netCDF with head/K/K33: {out_nc}")
        return out_nc
    except Exception as e:
        log(f"[WARN] Failed to write combined netCDF: {e}")
        return None


# ------------------------- Driver -------------------------
def run_from_yaml(yaml_path: str | Path,
                  out_folder: str | Path | None = None,
                  *,
                  log: Callable[[str], None] = print,
                  dry_run: bool = False,
                  make_figures: bool = False,
                  build_contours_in_driver: Optional[bool] = None) -> Dict[str, Optional[str]]:
    """
    Unified driver used by ArcGIS Pro toolbox and the CLI.
    """
    yaml_path = Path(yaml_path).expanduser().resolve()
    log(f"Loading configuration from {yaml_path} â€¦")
    cfg: Settings = load(yaml_path)

    if out_folder:
        cfg.output_directory = Path(out_folder).expanduser().resolve()

    if not cfg.output_directory:
        raise ValueError("`output_directory` must be set (in YAML or via out_folder).")

    # Ensure executables
    _ensure_executables(cfg, log)

    cfg.setup_workspace(clean=False)
    log(f"Workspace: {cfg.workspace_path}")
    log(f"GWF workspace: {cfg.gwf_ws}")
    log(f"MP7 workspace: {cfg.mp7_ws}")

    if dry_run:
        return {}

    # -------- Step 1: Preprocessing --------
    log("STEP 1 â€” Preprocessing rasters/vectors â€¦")
    myu.preprocess_data(cfg)

    return _run_pipeline(
        cfg,
        log=log,
        make_figures=make_figures,
        add_to_map=True,
        build_contours_in_driver=build_contours_in_driver,
    )


def _run_pipeline(cfg: Settings,
                  *,
                  log: Callable[[str], None] = print,
                  make_figures: bool = False,
                  add_to_map: bool = True,
                  build_contours_in_driver: Optional[bool] = None) -> Dict[str, Optional[str]]:
    """
    Shared pipeline (STEP 2 onward) for run_from_yaml (file-based) and run_hyporheic
    (headless/web). Assumes cfg is already preprocessed (terrain & water-surface
    reprojected; ground_water_domain/left_boundary/right_boundary populated).
    add_to_map=False skips the ArcGIS Pro map-group step (the web/headless path
    passes False; it is also a no-op without arcpy).
    """
    pre = dict(
        hec_ras_crs=cfg.hec_ras_crs,
        terrain_output_raster=cfg.terrain_output_raster,
        water_surface_output_raster=cfg.cropped_water_surface_raster,
        ground_water_domain=cfg.ground_water_domain,
        left_boundary=cfg.left_boundary,
        right_boundary=cfg.right_boundary,
    )

    # -------- Step 3: Domain --------
    log("STEP 2 â€” Building model domain â€¦")
    dom = myu.build_model_domain(cfg)
    grid_x, grid_y = dom["grid_x"], dom["grid_y"]
    grid_points = dom["grid_points"]

    # -------- Step 4: Boundaries & idomain --------
    log("STEP 3 â€” Defining boundaries & active domain â€¦")
    # Make upstream/downstream placeholders from left/right endpoints (used for classification)
    upstream_boundary, downstream_boundary = myu.define_floodplain_boundaries(
        pre["left_boundary"], pre["right_boundary"]
    )
    idomain, grid_gdf = myu.make_idomain(cfg, pre["ground_water_domain"])

    # Identify boundary cell sets
    all_boundary = myu.identify_boundary_cells(idomain)
    left_cells, right_cells, up_cells, down_cells = myu.classify_boundary_cells_faster(
        all_boundary, grid_gdf,
        pre["left_boundary"], pre["right_boundary"],
        upstream_boundary, downstream_boundary,
        cfg.ncol
    )

    # Use only the first layer cells
    left_cells_0  = [c for c in left_cells  if c[0] == 0]
    right_cells_0 = [c for c in right_cells if c[0] == 0]
    up_cells_0    = [c for c in up_cells    if c[0] == 0]
    down_cells_0  = [c for c in down_cells  if c[0] == 0]

    # --- Robust endpoints + lines for each side
    l_first_pt, l_last_pt, left_line   = myu.endpoints_and_line(pre["left_boundary"])
    r_first_pt, r_last_pt, right_line  = myu.endpoints_and_line(pre["right_boundary"])
    from shapely.geometry import LineString
    upstream_line   = LineString([l_first_pt, r_first_pt])   if l_first_pt and r_first_pt else None
    downstream_line = LineString([l_last_pt,  r_last_pt])    if l_last_pt and r_last_pt  else None

    # Sort boundary cells along their respective lines for correct interpolation
    left_cells_0  = myu.sort_cells_along_line(left_cells_0,  grid_x, grid_y, left_line)
    right_cells_0 = myu.sort_cells_along_line(right_cells_0, grid_x, grid_y, right_line)
    up_cells_0    = myu.sort_cells_along_line(up_cells_0,    grid_x, grid_y, upstream_line)
    down_cells_0  = myu.sort_cells_along_line(down_cells_0,  grid_x, grid_y, downstream_line)

    # -------- Step 4b: Boundary heads (mode-dependent) --------
    log("STEP 4 â€” Computing boundary heads from WSE edge + gradients â€¦")
    # Build WSE-edge index (edge of *valid* pixels, not raster footprint)
    wse_edge_idx = myu.build_wse_valid_edge_index(cfg.cropped_water_surface_raster)

    # Normalize mode
    mode_raw = (getattr(cfg, "boundary_condition_mode", "") or "").strip()
    mode = mode_raw.lower().replace("_", " ").replace("-", " ")
    if mode in {"four corner gradients", "4 corner gradients", "corner gradients", "corners"}:
        bc_mode = "corner"
    elif mode in {"spatially varying gradient", "spatially varying gradients", "spatial varying gradient"}:
        bc_mode = "profile"
    else:
        log(f"[WARN] Unknown boundary_condition_mode='{mode_raw}'. Falling back to '4 Corner Gradients'.")
        bc_mode = "corner"

    if bc_mode == "corner":
        # ----- Existing behavior: corner gradients -----
        def _edge_info(pt: Point | None, tag: str):
            if pt is None:
                return {"dist": 0.0, "wse": float("nan")}
            d, w, edge_xy, border_xy = myu.nearest_wse_edge_distance_and_value(wse_edge_idx, pt)
            log(f"  {tag}: distance to WSE-edge = {d:.3f}, edge WSE = {w:.3f}")
            return {"dist": d, "wse": w}

        UL = _edge_info(l_first_pt, "Upstream-Left")
        UR = _edge_info(r_first_pt, "Upstream-Right")
        DL = _edge_info(l_last_pt,  "Downstream-Left")
        DR = _edge_info(r_last_pt,  "Downstream-Right")

        # Read gradients (Length/Length); positive â†’ flow toward stream
        g_UL = float(getattr(cfg, "upstream_left_fpl_gw_gradient",  0.0))
        g_UR = float(getattr(cfg, "upstream_right_fpl_gw_gradient", 0.0))
        g_DL = float(getattr(cfg, "downstream_left_fpl_gw_gradient", 0.0))
        g_DR = float(getattr(cfg, "downstream_right_fpl_gw_gradient",0.0))

        # Corner heads = WSE_edge + gradient * distance_to_edge
        gw_left_first   = (UL["wse"] + g_UL * UL["dist"]) if l_first_pt else None
        gw_right_first  = (UR["wse"] + g_UR * UR["dist"]) if r_first_pt else None
        gw_left_last    = (DL["wse"] + g_DL * DL["dist"]) if l_last_pt  else None
        gw_right_last   = (DR["wse"] + g_DR * DR["dist"]) if r_last_pt  else None

        # Build per-side head arrays by linear interpolation between corner heads
        gw_left  = myu.interpolate_gw_elevation_first_layer_only(left_cells_0,  gw_left_first,  gw_left_last)   if left_cells_0  else []
        gw_right = myu.interpolate_gw_elevation_first_layer_only(right_cells_0, gw_right_first, gw_right_last)  if right_cells_0 else []
        gw_up    = myu.interpolate_gw_elevation_first_layer_only(up_cells_0,    gw_left_first,  gw_right_first) if up_cells_0    else []
        gw_down  = myu.interpolate_gw_elevation_first_layer_only(down_cells_0,  gw_left_last,   gw_right_last)  if down_cells_0  else []

    else:
        # ----- NEW behavior: spatially varying gradient profiles -----
        left_profile  = getattr(cfg, "left_boundary_gradient_profile",  None)
        right_profile = getattr(cfg, "right_boundary_gradient_profile", None)
        if not left_profile or not right_profile:
            raise ValueError("When boundary_condition_mode='Spatially Varying Gradient', both "
                             "'left_boundary_gradient_profile' and 'right_boundary_gradient_profile' must be provided.")

        log("  Left boundary profile:")
        gw_left, left_head_f0, left_head_f1 = myu.compute_boundary_heads_from_profile(
            left_cells_0, grid_x, grid_y, left_line, left_profile, wse_edge_idx, log=log
        ) if left_cells_0 else ([], float("nan"), float("nan"))

        log("  Right boundary profile:")
        gw_right, right_head_f0, right_head_f1 = myu.compute_boundary_heads_from_profile(
            right_cells_0, grid_x, grid_y, right_line, right_profile, wse_edge_idx, log=log
        ) if right_cells_0 else ([], float("nan"), float("nan"))

        # Upstream (interpolate between left/right at f=0); Downstream (f=1)
        gw_up = myu.interpolate_gw_elevation_first_layer_only(
            up_cells_0, left_head_f0, right_head_f0
        ) if up_cells_0 else []
        gw_down = myu.interpolate_gw_elevation_first_layer_only(
            down_cells_0, left_head_f1, right_head_f1
        ) if down_cells_0 else []

    # -------- Step 5: River stage cells (unchanged) --------
    out_csv = Path(cfg.output_directory) / "model" / "grid_points_elevation.csv"
    df_gp = myu.sample_surface_elevations_to_grid_points(cfg.cropped_water_surface_raster, grid_points, out_csv)

    df = df_gp.dropna(subset=["elevation"]).copy()
    df = df[df["elevation"] != -9999]
    df_fit = myu.fit_csv_to_grid(df, cfg.ncol, cfg.nrow, cfg.xmin, cfg.ymin, cfg.xmax, cfg.ymax)
    df_fit = df_fit[df_fit["elevation"] >= 0]
    river_cells = myu.extract_river_cells(df_fit, idomain)

    chd_data, n_unique, n_dupes = myu.compile_chd_data(
        river_cells, left_cells_0,  gw_left, right_cells_0, gw_right,
        up_cells_0, gw_up, down_cells_0, gw_down,
        nlay=int(cfg.nlay), copy_boundary_heads_to_all_layers=True
    )
    log(f"CHD cells prepared: {len(chd_data)} (unique={n_unique}, dupes={n_dupes})")

    # -------- Step 6â€“7: Build/run + postprocess --------
    if getattr(cfg, "kh_polygon", False):
        log("KH polygon mapping enabled: per-cell KH (and KV if present) will be used.")
    log("STEP 5 â€“ Building & running models (MF6 + MP7) â€¦")
    gwfsim, gwf = myu.scenario(
        cfg, idomain, chd_data, river_cells,
        write=(cfg.write or True), run=(cfg.run or True), plot=False, silent=False
    )

    log("STEP 6 â€” Postâ€‘processing MODPATH7 (Forward) â€¦")
    summary_dir = Path(cfg.output_directory) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    artifacts = myu.process_and_export_modpath7_results(
        workspace=cfg.mp7_ws,
        workspace_gwf=cfg.gwf_ws,
        sim_name=cfg.sim_name,
        gwf_model_name=cfg.gwf_name,
        hec_ras_crs=cfg.hec_ras_crs,
        bed_elevation=cfg.bed_elevation,
        ncol=cfg.ncol, nrow=cfg.nrow, z=cfg.z, nlay=cfg.nlay,
        river_cells=river_cells, gwf=gwf,
        xorigin_value=cfg.xmin, yorigin_value=cfg.ymin,
        output_folder=summary_dir, direction="Forward",
        export_csv=True,
        export_shp=True,            # 2D
        export_shp_3d=True,         # 3D (filtered/hyporheic)
        export_shp_wgs84=False,
        export_kml=False, export_kmz=False, export_gpkg=False,
        export_results_txt=True,
        include_pngs_in_return=True, export_pngs=True, plots_dpi=150
    )

    # --- NEW: Echo the publication-ready stats to the toolbox/console ---
    try:
        stats_txt = artifacts.get("results")
        if stats_txt and Path(stats_txt).exists():
            stats_text = Path(stats_txt).read_text(encoding="utf-8")
            log("")  # spacer
            log("=== Publicationâ€‘Ready Pathline Statistics ===")
            # Emit line-by-line to avoid UI truncation in Pro's Messages pane
            for line in stats_text.splitlines():
                log(line)
            log("=== end statistics ===")
    except Exception as e:
        log(f"[WARN] Could not echo statistics to output window: {e}")


    # -------- NEW: Export FULL (unfiltered) 3D pathlines shapefile --------
    try:
        full3d = myu.export_full_modpath7_pathlines_3d_shp(
            workspace=cfg.mp7_ws,
            workspace_gwf=cfg.gwf_ws,
            sim_name=cfg.sim_name,
            gwf_model_name=cfg.gwf_name,
            hec_ras_crs=cfg.hec_ras_crs,
            projection_file=getattr(cfg, "projection_file", None),
            output_folder=summary_dir,
            direction="Forward",
        )
        if full3d:
            artifacts["lines_shp_3d_full"] = full3d
    except Exception as e:
        log(f"[WARN] Full (unfiltered) 3D pathline export skipped: {e}")

    # -------- Head exports (GeoTIFFs + netCDF; mosaic) --------
    log("STEP 7 â€” Exporting hydraulic head layers â€¦")
    head_info = myu.export_hydraulic_head_layers(cfg=cfg, gwf=gwf, log=log)

    # -------- NEW: Combined netCDF (head,K,K33) --------
    try:
        log("STEP 7b â€” Writing combined netCDF (head, K, K33) and preparing map layers â€¦")
        nc_multi = _build_combined_head_K_nc(
            cfg=cfg,
            gwf=gwf,
            grid_x=grid_x,
            grid_y=grid_y,
            nlay=int(cfg.nlay),
            nrow=int(cfg.nrow),
            ncol=int(cfg.ncol),
            log=log
        )
        if isinstance(head_info, dict) and nc_multi:
            head_info["netcdf_multi"] = str(nc_multi)
    except Exception as e:
        log(f"[WARN] Could not create combined netCDF: {e}")

    # ---------------- Dynamic Results Group Name ----------------
    group_name = _results_group_name(getattr(cfg, "sim_name", None))
    os.environ["HYP_GROUP_NAME"] = group_name  # used by contour creator fallback

    # -------- Optional: SA contours + add to map --------
    contours_report = {}
    _contours_toggle = getattr(cfg, "build_contours_in_driver", None)
    if _contours_toggle is None:
        _contours_toggle = bool(build_contours_in_driver) if build_contours_in_driver is not None else False
    if _contours_toggle:
        try:
            geotiffs = head_info.get("geotiffs", []) if isinstance(head_info, dict) else []
        except Exception:
            geotiffs = []
        try:
            ci_cfg = getattr(cfg, "contour_interval", None)
            contour_interval = float(ci_cfg) if ci_cfg is not None else float(os.environ.get("HYP_CONTOUR_INTERVAL_FT", "0.5"))
        except Exception:
            contour_interval = float(os.environ.get("HYP_CONTOUR_INTERVAL_FT", "0.5"))
        units_label = str(getattr(cfg, "length_units", "ft")) or "ft"

        try:
            _max_layers_val = getattr(cfg, "max_layers", None)
            max_layers_int = int(_max_layers_val) if _max_layers_val is not None else None
        except Exception:
            max_layers_int = None

        contours_report = _build_head_contours_and_add_to_map(
            geotiffs=geotiffs,
            base_dir=(Path(cfg.output_directory) / "summary" / "head"),
            contour_interval=contour_interval,
            max_layers=max_layers_int,
            group_name=group_name,
            units_label=units_label,
            log=log
        )

    # -------- Add products to the (dynamic) group (ArcGIS Pro only) --------
    if add_to_map:
        _add_products_to_group(cfg=cfg, artifacts=artifacts, head_info=head_info,
                               group_name=group_name, log=log)

    if make_figures:
        log("STEP 8 â€” Figures â€¦")
        myu.plot_hyporheic_workflow(
            sim_name=cfg.sim_name,
            sim_path=str(cfg.gwf_ws),
            exe_name=str(cfg.md6_exe_path) if cfg.md6_exe_path else "mf6",
            sat_image_path=str(cfg.aerial_raster) if cfg.aerial_raster else str(cfg.terrain_elevation_raster),
            projection_file=str(cfg.projection_file),
            gw_domain_shapefile_path=str(cfg.ground_water_domain_shapefile),
            particle_points_shp=str(artifacts.get("points_shp") or ""),
            pathlines_shp=str(artifacts.get("lines_shp") or ""),
            river_cells=river_cells,
            particle_data_csv=str(artifacts.get("csv") or ""),
            pathline_stats_txt=str(artifacts.get("results") or ""),
            direction="Forward",
            output_folder=str(summary_dir),
            plot_layer=1, dpi=300, show=False,
            save_fig_head_overlay=True,
            save_fig_head_overlay_w_paths_points=True,
            save_fig_paths_points_only=True,
            save_fig_isometric=True,
            save_fig_longitudinal=True,
            save_stats_csv=True,
            save_stats_md=False,
            export_head_geotiff=False, export_head_mask_geotiff=False,
            export_start_end_points_shp=False, export_start_end_points_csv=False,
            export_cropped_lines_shp=False,
        )

    pathlines_best = (
        artifacts.get("lines_fc_3d") or
        artifacts.get("lines_shp_3d") or
        artifacts.get("lines_shp") or
        artifacts.get("lines_shp_wgs84")
    )

    return {
        "points_fc": artifacts.get("points_shp") or artifacts.get("points_shp_wgs84"),
        "pathlines_fc": artifacts.get("lines_shp") or artifacts.get("lines_shp_wgs84"),
        "pathlines_fc_3d": artifacts.get("lines_shp_3d"),
        "pathlines_fc_3d_full": artifacts.get("lines_shp_3d_full"),
        "head": head_info,
        "contours": contours_report,
        "group_name": group_name,
    }

