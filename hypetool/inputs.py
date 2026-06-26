# src/hypetool/inputs.py
"""
inputs.py â€“ centralized configuration loaded from inputs.yaml

Notes
-----
- The user edits *inputs.yaml*, not this file.
- Relative paths in inputs.yaml are resolved relative to the YAML's folder.
- Environment variables WRITE/RUN/PLOT/PLOT_SHOW/PLOT_SAVE override flags.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, List, Union, IO
import os
import shutil
import io
import yaml

# Light dependencies (only imported here)
import geopandas as gpd
import rasterio
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, field_validator, ValidationInfo
from pyproj import CRS
from shapely.geometry import box

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
SOURCE = Path(__file__).resolve().parent  # hypetool
DEFAULT_INPUTS = SOURCE / "inputs.yaml"   # keep your current default

# ------------------------------------------------------------------
# Pydantic v2 Settings model
# ------------------------------------------------------------------
class Settings(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"   # permit extra keys in YAML

    # Spatial input paths
    water_surface_elevation_raster: Optional[Path] = None
    terrain_elevation_raster:       Optional[Path] = None
    ground_water_domain_shapefile:  Optional[Path] = None
    left_boundary_floodplain:       Optional[Path] = None
    right_boundary_floodplain:      Optional[Path] = None
    projection_file:                Optional[Path] = None  # text CRS (.prj)
    output_directory:               Optional[Path] = None  # for model results
    aerial_raster:                  Optional[Path] = None  # Aerial imagery for plotting

    # Executables (overrides)
    modflow_bin_dir: Optional[Path] = None      # folder containing mf6/mp7 (preferred override)
    md6_exe_path:    Optional[Path] = None      # explicit file path to mf6(.exe)
    md7_exe_path:    Optional[Path] = None      # explicit file path to mp7(.exe)

    # Simulation meta & units
    sim_name: str = "hyporheic"
    workspace: str = "model"
    length_units: str = Field("feet", pattern="^(feet|meters)$")
    time_units: str = Field("days", pattern="^(days|seconds)$")

    # Grid / domain
    cell_size_x: PositiveFloat = 10.0
    cell_size_y: PositiveFloat = 10.0
    gw_mod_depth: PositiveFloat = 20.0
    z: PositiveFloat = 0.5

    # Hydraulic parameters
    kh: PositiveFloat = 10.0
    kv: PositiveFloat = 1.0
    # Optional: spatially varying K from polygon shapefile
    kh_polygon: bool = False
    kh_polygon_shapefile: Optional[Path] = None
    gw_offset: PositiveFloat = 0.5
    porosity: PositiveFloat = Field(0.3, le=0.6)

    # Stress period / time stepping
    nper: PositiveInt = 1
    nstp: PositiveInt = 1
    perlen: PositiveFloat = 1.0
    tsmult: PositiveFloat = 1.0

    # Derived / runtime
    hec_ras_crs: Optional[Any] = None
    gwf_name: Optional[str] = None
    mp7_name: Optional[str] = None
    gwf_ws: Optional[str] = None
    mp7_ws: Optional[str] = None
    headfile: Optional[str] = None
    head_filerecord: Optional[List[str]] = None
    budgetfile: Optional[str] = None
    budget_filerecord: Optional[List[str]] = None

    # Terrain / grid attributes
    terrain_elevation: Optional[Any] = None
    raster_transform: Optional[Any] = None
    raster_crs: Optional[Any] = None
    raster_bounds_box: Optional[Any] = None
    transform: Optional[Any] = None
    bed_elevation: Optional[Any] = None
    raster_width: Optional[float] = None
    raster_height: Optional[float] = None
    ncol: Optional[int] = None
    nrow: Optional[int] = None
    top: Optional[Any] = None
    nlay: Optional[int] = None
    terrain_output_raster: Optional[str] = None
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    grid_rotation_degrees: Optional[float] = None

    # Grid arrays & points
    grid_x: Optional[Any] = None
    grid_y: Optional[Any] = None
    grid_points: Optional[gpd.GeoDataFrame] = None
    intersecting_points: Optional[gpd.GeoDataFrame] = None
    xorigin: Optional[float] = None
    yorigin: Optional[float] = None
    tops: Optional[List[Any]] = None
    botm: Optional[List[Any]] = None

    # Optional KH polygon zones
    kh_polygon_gdf: Optional[gpd.GeoDataFrame] = None

    # Water-surface attrs
    surface_elevation: Optional[Any] = None
    ws_transform: Optional[Any] = None
    ws_raster_crs: Optional[Any] = None
    water_surface_output_raster: Optional[str] = None
    cropped_water_surface_raster: Optional[str] = None

    # Vectors
    project_crs: Any = "EPSG:4326"
    ground_water_domain: Optional[gpd.GeoDataFrame] = None
    left_boundary: Optional[gpd.GeoDataFrame] = None
    right_boundary: Optional[gpd.GeoDataFrame] = None

    # ------------------------------
    # Boundary condition configuration
    # ------------------------------
    # Dropdown-like string that controls how BCs are derived.
    # Allowed (case-insensitive): "4 Corner Gradients" | "Spatially Varying Gradient"
    boundary_condition_mode: str = "4 Corner Gradients"

    # Corner-gradient (existing behavior) â€” positive â†’ flow toward stream
    upstream_left_fpl_gw_gradient:  float = 0.010
    upstream_right_fpl_gw_gradient: float = 0.010
    downstream_left_fpl_gw_gradient: float = 0.010
    downstream_right_fpl_gw_gradient: float = 0.010

    # NEW: spatially varying gradient profiles (only used when boundary_condition_mode is "Spatially Varying Gradient")
    # Each is a string of space-separated "fraction,gradient" pairs. Fractions must include 0 and 1.
    # Example: "0,0.01 0.5,0.05 1,0.1"
    left_boundary_gradient_profile: Optional[str] = None
    right_boundary_gradient_profile: Optional[str] = None

    # Flags
    write: bool = False
    run: bool = False
    plot: bool = False
    plot_show: bool = False
    plot_save: bool = False
    # Contour/map options
    build_contours_in_driver: bool = False
    # Number of GeoTIFF layers to contour (None = all; <=0 = skip)
    max_layers: Optional[int] = None
    # Contour interval (units follow length_units)
    contour_interval: PositiveFloat = 0.5

    # Helper attrs
    workspace_path: Optional[Path] = None
    results_ready: bool = False
    inputs_yaml_file: Optional[Path] = None

    # --- Validators ---
    @field_validator(
        "water_surface_elevation_raster",
        "terrain_elevation_raster",
        "ground_water_domain_shapefile",
        "left_boundary_floodplain",
        "right_boundary_floodplain",
        "projection_file",
        "modflow_bin_dir",
        "md6_exe_path",
        "md7_exe_path",
        "output_directory",
        "aerial_raster",
        "kh_polygon_shapefile",
        mode="before",
    )
    @classmethod
    def _resolve_rel_paths(cls, v, info: ValidationInfo):
        if v is None:
            return v
        p = Path(v) if not isinstance(v, Path) else v
        if p.is_absolute():
            return p
        cfg_dir: Path = info.context.get("cfg_dir", Path.cwd())
        return (cfg_dir / p).resolve()

    # --- Methods ---
    def setup_workspace(self, clean: bool = False) -> None:
        """Create workspace and subfolders; set flags from env."""
        if not self.output_directory:
            raise ValueError("`output_directory` is not set.")
        self.workspace_path = Path(self.output_directory / self.workspace).resolve()
        if clean and self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Model names (â‰¤16 chars)
        self.gwf_name = self.gwf_name or "gwf_model"
        self.mp7_name = self.mp7_name or "mp7_model"

        # Sub-workspaces
        gwf_ws_path = Path(self.gwf_ws) if self.gwf_ws else self.workspace_path / "gwf_workspace"
        mp7_ws_path = Path(self.mp7_ws) if self.mp7_ws else self.workspace_path / "mp7_workspace"
        gwf_ws_path.mkdir(parents=True, exist_ok=True)
        mp7_ws_path.mkdir(parents=True, exist_ok=True)
        self.gwf_ws = str(gwf_ws_path)
        self.mp7_ws = str(mp7_ws_path)

        # Output file names
        self.headfile = self.headfile or f"{self.gwf_name}.hds"
        self.head_filerecord = self.head_filerecord or [self.headfile]
        self.budgetfile = self.budgetfile or f"{self.gwf_name}.cbb"
        self.budget_filerecord = self.budget_filerecord or [self.budgetfile]

        # Flags from env (override YAML)
        def get_env(name: str, default: bool | str):
            raw = os.getenv(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

        for flag in ("write", "run", "plot", "plot_show", "plot_save"):
            setattr(self, flag, get_env(flag.upper(), getattr(self, flag)))

    def setup_projection(self) -> None:
        if not self.projection_file or not Path(self.projection_file).exists():
            raise FileNotFoundError("projection_file path is missing or does not exist.")
        self.hec_ras_crs = CRS.from_string(Path(self.projection_file).read_text(encoding="utf-8-sig").strip())
        print(f"Loaded HECâ€‘RAS CRS: {self.hec_ras_crs}")

    # --- raster/vector helpers (unchanged) ---
    def setup_terrain(self, target_crs: Any, output_name: str | None = None) -> None:
        if not self.terrain_elevation_raster or not Path(self.terrain_elevation_raster).exists():
            raise FileNotFoundError("terrain_elevation_raster path is missing or does not exist.")

        import numpy as np
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        output_path = Path(self.workspace_path or ".") / (output_name or "reprojected_terrain_raster.tif")

        with rasterio.open(self.terrain_elevation_raster) as src:
            self.terrain_elevation = src.read(1)
            self.raster_transform = src.transform
            self.raster_crs = src.crs

            dst_transform, width, height = calculate_default_transform(
                self.raster_crs, target_crs, src.width, src.height, *src.bounds
            )
            self.transform = dst_transform

            new_meta = src.meta.copy()
            new_meta.update({"crs": target_crs, "transform": dst_transform, "width": width, "height": height})

            with rasterio.open(output_path, "w", **new_meta) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=self.raster_transform,
                    src_crs=self.raster_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )

        self.terrain_output_raster = str(output_path)
        print(f"Reprojected terrain raster saved as {output_path}")

    def setup_water_surface(self, target_crs: Any, output_name: str | None = None) -> None:
        if not self.water_surface_elevation_raster or not Path(self.water_surface_elevation_raster).exists():
            raise FileNotFoundError("water_surface_elevation_raster is missing or cannot be found.")
        if self.transform is None or not self.terrain_output_raster:
            raise RuntimeError("Run setup_terrain() first so terrain extent is available.")

        from rasterio.warp import calculate_default_transform, reproject, Resampling
        from rasterio.windows import from_bounds, Window

        ws_output = Path(self.workspace_path or ".") / (output_name or "reprojected_water_surface_raster.tif")
        cropped_output = Path(self.workspace_path or ".") / "cropped_water_surface_raster.tif"

        with rasterio.open(self.water_surface_elevation_raster) as src:
            self.surface_elevation = src.read(1)
            self.ws_transform = src.transform
            self.ws_raster_crs = src.crs

            dst_transform, width, height = calculate_default_transform(
                self.ws_raster_crs, target_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({"crs": target_crs, "transform": dst_transform, "width": width, "height": height})

            with rasterio.open(ws_output, "w", **meta) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=self.ws_transform,
                    src_crs=self.ws_raster_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )

        self.water_surface_output_raster = str(ws_output)
        print(f"Reprojected water-surface raster saved as {ws_output}")

        # Crop to terrain bounds (no mask to keep geotransform neat)
        from rasterio.windows import from_bounds
        with rasterio.open(self.terrain_output_raster) as terrain_src:
            terrain_bounds = terrain_src.bounds

        with rasterio.open(ws_output) as src:
            window = from_bounds(*terrain_bounds, transform=src.transform)
            col_off = max(0, int(window.col_off))
            row_off = max(0, int(window.row_off))
            width   = min(src.width  - col_off, int(round(window.width)))
            height  = min(src.height - row_off, int(round(window.height)))
            window  = rasterio.windows.Window(col_off, row_off, width, height)

            out_transform = src.window_transform(window)
            out_image     = src.read(window=window)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width":  out_image.shape[2],
                "transform": out_transform,
            })

            with rasterio.open(cropped_output, "w", **out_meta) as dst:
                dst.write(out_image)

        self.cropped_water_surface_raster = str(cropped_output)
        print(f"Cropped water-surface raster saved as {cropped_output}")

    def setup_vectors(self) -> None:
        # Pick a CRS if not set yet
        if self.hec_ras_crs:
            self.project_crs = self.hec_ras_crs
        elif self.water_surface_elevation_raster and Path(self.water_surface_elevation_raster).exists():
            try:
                with rasterio.open(self.water_surface_elevation_raster) as src:
                    self.project_crs = src.crs
            except Exception:
                self.project_crs = "EPSG:4326"
        else:
            self.project_crs = "EPSG:4326"

        def _load(path: Optional[Path | str]):
            if path and Path(path).exists():
                return gpd.read_file(path).to_crs(self.project_crs)
            return gpd.GeoDataFrame()

        self.ground_water_domain = _load(self.ground_water_domain_shapefile)
        self.left_boundary = _load(self.left_boundary_floodplain)
        self.right_boundary = _load(self.right_boundary_floodplain)
        # Optional KH polygon zones
        self.kh_polygon_gdf = None
        if getattr(self, "kh_polygon", False) and self.kh_polygon_shapefile and Path(self.kh_polygon_shapefile).exists():
            try:
                self.kh_polygon_gdf = gpd.read_file(self.kh_polygon_shapefile).to_crs(self.project_crs)
            except Exception:
                self.kh_polygon_gdf = None
        print("Vector layers loaded and re-projected to", self.project_crs)

# ------------------------------------------------------------------
# Loader helpers
# ------------------------------------------------------------------
def _load_from_mapping(data: dict, cfg_dir: Path) -> Settings:
    cfg = Settings.model_validate(data, context={"cfg_dir": cfg_dir})
    cfg.inputs_yaml_file = cfg_dir / "inputs.yaml"
    cfg.setup_workspace(clean=False)
    return cfg

def load(source: Union[str, Path, IO[str]] | None = None, *, inputs_yaml_file: str | None = None) -> Settings:
    if inputs_yaml_file is not None:
        source = Path(inputs_yaml_file)

    # file-like
    if isinstance(source, io.IOBase) and not isinstance(source, (str, Path)):
        text = source.read()
        data = yaml.safe_load(text) or {}
        cfg_dir = Path(".").resolve()
        return _load_from_mapping(data, cfg_dir)

    # raw YAML text
    if isinstance(source, str) and "\n" in source and not Path(source).exists():
        data = yaml.safe_load(source) or {}
        cfg_dir = Path(".").resolve()
        return _load_from_mapping(data, cfg_dir)

    # path-like
    if source is None:
        raise ValueError("Source cannot be None. Please provide a valid path or input.")
    else:
        path = Path(source).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Provided path '{path}' does not exist or is not a valid file.")
    cfg_dir = path.parent  # resolve relatives from the YAML's folder
    # utf-8-sig: read UTF-8 robustly and transparently strip a BOM if present
    # (default encoding is cp1252 on Windows, which corrupts a UTF-8/BOM YAML).
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    return _load_from_mapping(data, cfg_dir)

# Eager-load cfg so `from inputs import cfg` is safe, but don't explode if default missing.
try:
    cfg: Settings = load(DEFAULT_INPUTS)
except Exception:
    cfg = Settings()
__all__ = ["cfg", "load", "Settings"]

