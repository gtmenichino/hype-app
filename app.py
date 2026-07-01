"""Hyporheic web app — a StreamStats-style Shiny app that builds and runs a MODFLOW 6 +
MODPATH 7 hyporheic model from a map-drawn domain, and shows the pathlines/heads.

Flow: draw the groundwater-domain polygon + the left/right floodplain boundary lines
(+ optional K-zone polygons) → auto-fetch the 3DEP terrain DEM → choose how the water
surface is derived (full DEM / channel-only / uploaded) → set model parameters with a live
grid-size guardrail → run on the bundled Linux MODFLOW binaries → view pathlines + particle
points on the map and download the results.

Modeled on the EASI app (D:\\Code\\Work\\easi_claude). Deploy smoke_app.py first.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import re
import shutil
import tempfile
import time
from pathlib import Path

# 3DEP/HyRiver cache -> ephemeral /tmp (set before py3dep import, which happens in hype_app.dem)
os.environ.setdefault("HYRIVER_CACHE_NAME", os.path.join(tempfile.gettempdir(), "hype_hyriver.sqlite"))
os.environ.setdefault("HYRIVER_CACHE_EXPIRE", str(7 * 24 * 3600))

# Quiet two harmless, environment-emitted startup messages on the headless server (set before
# matplotlib / shinywidgets load below): matplotlib scanning the non-scalable Noto color-emoji
# font while building its cache, and shinywidgets' own internal use of the deprecated
# ipywidgets `Widget.widgets` API.
import logging  # noqa: E402
import warnings  # noqa: E402
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*Widget\.widgets is deprecated.*")

import anyio  # noqa: E402
from shiny import App, reactive, render, ui  # noqa: E402

from hype_app import (bieger, bundle, delineate, dem, estimate, geocode, geometry, hydro,  # noqa: E402
                      mesh, results)
from hype_app import run as runner  # noqa: E402

try:
    from ipyleaflet import (DivIcon, DrawControl, GeoJSON, ImageOverlay, LayerGroup, LayersControl,
                            Map, Marker, ScaleControl, TileLayer, ZoomControl)
    from ipywidgets import Layout
    from shinywidgets import output_widget, reactive_read, render_widget
    _HAS_MAP = True
except Exception:  # pragma: no cover
    _HAS_MAP = False

USGS_IMAGERY = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
USGS_TOPO = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
USGS_HYDRO = "https://hydro.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer/tile/{z}/{y}/{x}"
USGS_ATTR = "USGS The National Map"

PATH_STYLE = {"color": "#00a06b", "weight": 1, "opacity": 0.7}
POINT_STYLE = {"color": "#0b3d91", "fillColor": "#3399ff", "fillOpacity": 0.85, "weight": 1, "radius": 2}
GRID_STYLE = {"color": "#555555", "weight": 0.5, "opacity": 0.5, "fillOpacity": 0.0}
CONTOUR_STYLE = {"color": "#11161c", "weight": 1, "opacity": 0.85, "fillOpacity": 0.0}
# drawn inputs — thin outlines / minimal fill so they never hide the head raster underneath
DOMAIN_STYLE = {"color": "#caa700", "weight": 2, "opacity": 0.95, "fill": False}
WSE_STYLE = {"color": "#1aa6a6", "weight": 2, "opacity": 0.95, "fillColor": "#1aa6a6", "fillOpacity": 0.12}
LEFT_STYLE = {"color": "#1f6feb", "weight": 3, "opacity": 0.95}      # Left FPL (blue)
RIGHT_STYLE = {"color": "#d83933", "weight": 3, "opacity": 0.95}     # Right FPL (red)
UP_STYLE = {"color": "#f08c00", "weight": 3, "opacity": 0.95}        # Upstream boundary (orange)
DOWN_STYLE = {"color": "#9b59b6", "weight": 3, "opacity": 0.95}      # Downstream boundary (purple)
KZONE_STYLE = {"color": "#7b3fa0", "weight": 2, "opacity": 0.95, "fill": False}
NHD_STYLE = {"color": "#00c2ff", "weight": 3.5, "opacity": 0.95}     # clickable NHD flowlines (bold)
REACH_STYLE = {"color": "#ff2d95", "weight": 5, "opacity": 0.95}     # the analysis reach (magenta — pops on USGS topo, distinct from cyan NHD)
CAP_STYLE = {"color": "#333333", "weight": 2, "opacity": 0.9, "dashArray": "6 5", "fill": False}

STEP_REACH, STEP_DEM, STEP_BOUNDARIES, STEP_K, STEP_MESH, STEP_RUN, STEP_RESULTS = (
    "reach", "dem", "boundaries", "k", "mesh", "run", "results")
STEP_LABELS = [(STEP_REACH, "Reach"), (STEP_DEM, "DEM"), (STEP_BOUNDARIES, "Boundaries"),
               (STEP_K, "K"), (STEP_MESH, "Mesh"), (STEP_RUN, "Run"), (STEP_RESULTS, "Results")]
# Steps where the user draws/edits shapes in the DrawControl (vs. static mirrored layers).
EDIT_STEPS = (STEP_REACH, STEP_BOUNDARIES, STEP_K)

BC_CORNER = "4 Corner Gradients"
BC_PROFILE = "Spatially Varying Gradient"

# Progress labels keyed by the driver's "STEP N" log markers (the headless run emits 2–7).
RUN_TOTAL = 7
RUN_STEPS = {0: "Preparing terrain & geometry…", 1: "Preprocessing",
             2: "Building model domain", 3: "Boundaries & active domain",
             4: "Computing boundary heads", 5: "Running MODFLOW 6 + MODPATH 7",
             6: "Post-processing pathlines", 7: "Exporting head layers"}

_WWW = Path(__file__).parent / "www"


def _asset(name: str) -> str:
    """Append the file's mtime as a cache-busting ?v= so browsers re-fetch our static assets after
    any edit. Shiny serves styles.css / *.js with no version, so browsers cache them hard and keep
    using the stale copy across server restarts — which is why a fixed CSS/JS silently didn't apply
    (a restarted server serves the new file, but the browser never re-requests it)."""
    try:
        v = int(_WWW.joinpath(name).stat().st_mtime)
    except OSError:
        v = 0
    return f"{name}?v={v}"


app_ui = ui.page_fillable(
    ui.head_content(
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
        ui.tags.link(rel="stylesheet",
                     href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap"),
        ui.tags.link(rel="stylesheet", href=_asset("styles.css")),
        ui.tags.script(src=_asset("geocode.js")),
        ui.tags.script(src=_asset("reach_draw.js")),
        ui.tags.script(src=_asset("mesh3d.js")),     # lazy-loads vtk.js from a CDN on first Compute
    ),
    ui.div(
        ui.div(
            ui.span("HYPE", ui.tags.small("Hyporheic Exchange Explorer"), class_="hype-brand"),
            ui.div(ui.input_action_link("nav_new", "New run"),
                   ui.input_action_link("nav_help", "Help"), class_="hype-nav"),
            class_="hype-header",
        ),
        ui.div(
            output_widget("map", height="100%") if _HAS_MAP
            else ui.div("Map requires ipyleaflet + shinywidgets.", class_="p-3"),
            class_="hype-map-wrap",
        ),
        ui.div(ui.output_ui("leftpane"), class_="hype-leftpane"),
        ui.output_ui("readout"),
        ui.output_ui("flow_loading"),
        ui.output_ui("map_edit_style"),
        ui.div(id="hype-mesh3d", class_="hype-mesh3d"),     # 3D mesh viewer overlay (vtk.js)
        ui.output_ui("mesh3d_style"),
        class_="hype-shell",
    ),
    title="HYPE — Hyporheic Exchange Explorer",
    padding=0,
    fillable=True,
)


def _stepper(active, reachable):
    items = []
    for key, label in STEP_LABELS:
        cls = "hype-step" + (" active" if key == active else "") + (
            "" if key in reachable else " disabled")
        items.append(ui.input_action_link(f"go_{key}", label, class_=cls))
    return ui.div(*items, class_="hype-steps")


def server(input, output, session):
    work_dir = Path(tempfile.mkdtemp(prefix="hype_session_"))

    current_step = reactive.value(STEP_REACH)
    # Four named boundary lines (4326) that close into the domain (the domain is DERIVED from them
    # via geometry.assemble_domain_from_sides — see the domain_feat calc below).
    up_feat = reactive.value(None)         # Upstream boundary LineString Feature
    left_feat = reactive.value(None)       # Left FPL boundary LineString Feature
    right_feat = reactive.value(None)      # Right FPL boundary LineString Feature
    down_feat = reactive.value(None)       # Downstream boundary LineString Feature
    bnd_slot = reactive.value(None)        # boundary being drawn/edited: up|left|right|down|wse|None
    kz_adding = reactive.value(False)      # True while a guided "Add K-zone" polygon draw is armed
    mesh_geom = reactive.value(None)       # last computed 3D mesh geometry (for status + viewer)
    kzone_feats = reactive.value([])       # list of GeoJSON polygon features (4326)
    wse_extent_feat = reactive.value(None)  # drawn water-surface (wetted) extent polygon (4326)
    wse_mode_v = reactive.value("draw")     # mirror of the WSE-mode radio; persists across steps
    delineate_mode = reactive.value("auto")  # "auto" (pick 2 NHD points) | "manual" (draw)
    pick_pts = reactive.value([])           # snapped points: [{lat,lon,comid,dist_ft}, ...]
    reach_feat = reactive.value(None)       # traced reach LineString Feature (4326)
    auto_meta = reactive.value(None)        # {da_sqkm, length_m, bankfull_depth_m, division, ...}
    last_click = reactive.value(None)       # (lat, lon) from Map.on_interaction
    nhd_status = reactive.value("")         # NHD-streams loading/status message
    _flow = {"gdf": None}                   # cached NHD flowlines GDF (for snapping)
    proj_crs = reactive.value(None)
    dem_path = reactive.value(None)
    dem_meta = reactive.value(None)        # {"resolution_m", "source"} of the fetched 3DEP DEM
    run_result = reactive.value(None)
    head_tifs = reactive.value([])          # per-layer head GeoTIFF paths (index 0 = top layer)
    head_rng = reactive.value(None)         # global (vmin, vmax) for consistent head coloring
    head_layer_v = reactive.value(1)        # persisted slider state (survives pane re-renders)
    head_opacity_v = reactive.value(0.85)   # persisted slider state (survives pane re-renders)
    _head_cache: dict = {}                  # layer idx -> overlay payload (avoid re-render)
    _contour_cache: dict = {}               # layer idx -> contour GeoJSON
    stage = reactive.value("")
    log_lines: list[str] = []
    log_tick = reactive.value(0)
    run_t0 = reactive.value(0.0)           # monotonic start of the current run
    elapsed_v = reactive.value(0)          # seconds elapsed (updated by the poller)
    step_v = reactive.value(0)             # current STEP number parsed from the log
    _proc: dict = {"p": None}              # handle to the running child process (for cancel)

    def _terminate_child():
        p = _proc.get("p")
        if p is not None:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:  # noqa: BLE001
                pass

    def _on_session_end():
        _terminate_child()
        shutil.rmtree(work_dir, ignore_errors=True)

    session.on_ended(_on_session_end)

    def _safe(name, default):
        """Read an input that may be hidden (conditional panel) or unmounted (another step);
        return `default` if it never received a value — avoids Shiny's SilentException
        silently halting the Run handler before the task launches."""
        try:
            v = input[name]()
        except Exception:  # noqa: BLE001
            return default
        return default if v is None else v

    _layers: dict = {}
    _draw_ctl: dict = {}                     # holds the DrawControl so effects can clear it
    _bnd_shown: dict = {}                    # Boundaries step: per-layer signature (see _bnd_show)
    _map_ui: dict = {}                       # small map-view bookkeeping (last step seen, …)
    _MISSING = object()                      # sentinel: "layer not tracked yet" vs "tracked as None"

    def _set_layer(key, layer):
        old = _layers.get(key)
        if old is not None:
            try:
                _MAP.remove(old)
            except Exception:  # noqa: BLE001
                pass
        if layer is not None:
            _MAP.add(layer)
        _layers[key] = layer

    def _render_head_layer(idx: int):
        """Draw the hydraulic-head color overlay + contours for layer `idx` (1 = top), cached."""
        tifs = head_tifs(); rng = head_rng()
        if not tifs or rng is None:
            return
        idx = max(1, min(int(idx), len(tifs))); k = idx - 1
        if k not in _head_cache:
            _head_cache[k] = results.raster_overlay(tifs[k], vmin=rng[0], vmax=rng[1])
        ov = _head_cache[k]
        with reactive.isolate():                 # current opacity, without subscribing this caller
            op = float(head_opacity_v())
        _set_layer("head", ImageOverlay(url=ov["url"], bounds=ov["bounds"],
                                        name=f"Hydraulic head — L{idx}", opacity=op))
        if k not in _contour_cache:
            import numpy as _np
            levels = list(_np.linspace(rng[0], rng[1], 9))[1:-1]   # ~7 interior levels
            gj = results.head_contours_geojson(tifs[k], levels=levels)
            _contour_cache[k] = (gj, results.head_contour_labels(gj))
        gj, labels = _contour_cache[k]
        if gj:
            lines = GeoJSON(data=gj, style=CONTOUR_STYLE, name="Head contours")
            marks = [Marker(location=(la, lo), draggable=False, icon=DivIcon(
                        html=("<div style=\"font:600 11px/1 system-ui,sans-serif;color:#11161c;"
                              "white-space:nowrap;text-shadow:0 0 2px #fff,0 0 2px #fff,"
                              f"0 0 2px #fff\">{txt}</div>"), icon_anchor=[14, 8]))
                     for la, lo, txt in labels]
            _set_layer("head_contours", LayerGroup(layers=[lines, *marks], name="Head contours"))
        else:
            _set_layer("head_contours", None)

    @reactive.calc
    def _domain_build():
        """Assemble the domain (+ normalized left/right upstream→downstream) from the four boundary
        lines, or None until all four exist / if they can't close into a valid ring."""
        return geometry.assemble_domain_from_sides(up_feat(), left_feat(), right_feat(), down_feat())

    @reactive.calc
    def domain_feat():
        """The domain polygon Feature, DERIVED from the four boundary lines (None until buildable)."""
        b = _domain_build()
        return b["domain"] if b else None

    def _domain_gdf_4326():
        f = domain_feat()
        return geometry.single_feature_gdf(f) if f else None

    @reactive.effect
    def _set_proj_crs():
        g = _domain_gdf_4326()
        proj_crs.set(g.estimate_utm_crs() if g is not None else None)

    @reactive.effect
    def _sync_wse_mode():
        # Mirror the WSE-mode radio into a reactive.value so the (non-reactive) draw callback and
        # the run handler can read it. Only update while the input is mounted (Define step); when
        # it unmounts on step change, keep the last value instead of clobbering it with a default.
        try:
            wse_mode_v.set(input.wse_mode())
        except Exception:  # noqa: BLE001
            pass

    @reactive.effect
    def _sync_delineate_mode():
        try:
            delineate_mode.set(input.delineate_mode())
        except Exception:  # noqa: BLE001
            pass

    def _fc(feat):
        return feat if (feat or {}).get("type") == "FeatureCollection" else {
            "type": "FeatureCollection", "features": [feat]}

    _MIRROR_NAMES = ("Domain", "Water-surface extent", "Left boundary", "Right boundary",
                     "Upstream boundary", "Downstream boundary", "K-zones", "Reach")
    # Boundary slot → (map-layer name, style). The active slot lives in the DrawControl; the rest
    # render as static colored layers so all four sides stay visible while you edit one.
    _BND_STATIC = {"up": ("Upstream boundary", UP_STYLE), "left": ("Left boundary", LEFT_STYLE),
                   "right": ("Right boundary", RIGHT_STYLE), "down": ("Downstream boundary", DOWN_STYLE)}

    def _slot_value(slot):
        return {"up": up_feat, "left": left_feat, "right": right_feat, "down": down_feat,
                "wse": wse_extent_feat}.get(slot)

    def _mirror_features_as_layers():
        """Show the geometry as named, toggleable, thin static layers (features read isolated)."""
        with reactive.isolate():
            dom, wse, lf, rf, uf, df, kz, rch = (
                domain_feat(), wse_extent_feat(), left_feat(), right_feat(),
                up_feat(), down_feat(), list(kzone_feats()), reach_feat())
        for nm, feat, style in (("Domain", dom, DOMAIN_STYLE),
                                ("Water-surface extent", wse, WSE_STYLE),
                                ("Left boundary", lf, LEFT_STYLE), ("Right boundary", rf, RIGHT_STYLE),
                                ("Upstream boundary", uf, UP_STYLE),
                                ("Downstream boundary", df, DOWN_STYLE)):
            _set_layer(nm, GeoJSON(data=_fc(feat), style=style, name=nm) if feat else None)
        _set_layer("K-zones", GeoJSON(data={"type": "FeatureCollection", "features": kz},
                                      style=KZONE_STYLE, name="K-zones") if kz else None)
        _set_layer("Reach", GeoJSON(data=rch, style=REACH_STYLE, name="Reach") if rch else None)

    def _clear_mirror_layers():
        for nm in _MIRROR_NAMES:
            _set_layer(nm, None)
        _bnd_shown.clear()

    def _bnd_show(nm, feat, style):
        # Idempotent layer set for the Boundaries step: only rebuild `nm` when the shown feature
        # actually changed (tracked by object identity — features are replaced wholesale, never
        # mutated). This keeps *selecting* a boundary from removing + re-adding every other overlay
        # on each click; that churn (and the all-cleared transient) is what briefly blanked the map.
        sig = id(feat) if feat is not None else None
        if _bnd_shown.get(nm, _MISSING) == sig:
            return
        _bnd_shown[nm] = sig
        _set_layer(nm, GeoJSON(data=_fc(feat), style=style, name=nm) if feat is not None else None)

    def _render_boundaries(active):
        """Boundaries-step display: each side except the `active` one (which is in the DrawControl)
        as a static colored layer, plus the derived domain, the WSE (unless active), and the reach.
        Idempotent via `_bnd_show`, so re-running on a slot change only touches what actually
        changed (the active line moving in/out of the DrawControl) — never a full clear + re-add."""
        with reactive.isolate():
            feats = {"up": up_feat(), "left": left_feat(), "right": right_feat(), "down": down_feat()}
            wse = wse_extent_feat(); dom = domain_feat(); rch = reach_feat()
        for slot, (nm, style) in _BND_STATIC.items():
            _bnd_show(nm, feats[slot] if slot != active else None, style)
        _bnd_show("Domain", dom, DOMAIN_STYLE)
        _bnd_show("Water-surface extent", wse if active != "wse" else None, WSE_STYLE)
        _bnd_show("Reach", rch, REACH_STYLE)
        _bnd_show("K-zones", None, KZONE_STYLE)

    @reactive.effect
    def _sync_map_shapes():
        # Fires on STEP change (features isolated). Reach/K load their shapes into the DrawControl;
        # Boundaries is driven per-active-slot by _sync_bnd_slot; other steps clear + mirror statics.
        if not _HAS_MAP:
            return
        step = current_step()
        dc = _draw_ctl.get("dc")
        if step != STEP_REACH:                      # the auto-pick markers are Reach-only
            _set_layer("pick1", None); _set_layer("pick2", None)
        with reactive.isolate():
            kz = list(kzone_feats()); rch = reach_feat(); mode = delineate_mode()
        if step == STEP_REACH:
            _clear_mirror_layers()
            _set_layer("Reach", GeoJSON(data=rch, style=REACH_STYLE, name="Reach")
                       if (mode == "auto" and rch) else None)
            _load_into_drawcontrol([rch] if (mode == "manual" and rch) else [])
        elif step == STEP_K:
            _clear_mirror_layers()
            _set_layer("Reach", GeoJSON(data=rch, style=REACH_STYLE, name="Reach") if rch else None)
            _load_into_drawcontrol(kz)
        elif step == STEP_BOUNDARIES:
            pass                                   # _sync_bnd_slot owns the Boundaries display
        else:
            if dc is not None:
                try:
                    dc.clear(); dc.data = []
                except Exception:  # noqa: BLE001
                    pass
            _mirror_features_as_layers()

    @reactive.effect
    def _dem_backdrop_by_step():
        # The DEM hillshade (light terrain colormap, opacity 0.8) is a DEM-tab backdrop only. Off the
        # DEM step it washes the basemap pale and — stacking above the early-added DrawControl —
        # buries the boundary line you select to edit. Hide it everywhere except DEM (live opacity
        # trait, same as the head overlay; no rebuild, and opacity 0 also neutralizes the z-order).
        step = current_step()          # read FIRST so the effect subscribes even before the DEM
        lyr = _layers.get("dem")       # overlay exists (else the early return skips the dependency
        if lyr is None:                # and it never re-runs when the hillshade later appears).
            return
        try:
            lyr.opacity = 0.8 if step == STEP_DEM else 0.0
        except Exception:  # noqa: BLE001
            pass

    @reactive.effect
    def _sync_bnd_slot():
        # Owns the Boundaries-step map: load ONLY the active boundary into the DrawControl (so
        # Leaflet.draw never has to disambiguate four similar lines) and mirror the rest as statics.
        # `_render_boundaries` is idempotent, so selecting a slot only swaps the one active line in/
        # out of the DrawControl — it never clears + re-adds every overlay (which blanked the map).
        if not _HAS_MAP:
            return
        step = current_step()
        slot = bnd_slot()
        if step != STEP_BOUNDARIES:
            if slot is not None:
                with reactive.isolate():
                    bnd_slot.set(None)             # reset when leaving so re-entry starts clean
            _bnd_shown.clear()                     # rebuild cleanly on the next entry
            return
        with reactive.isolate():
            sv = _slot_value(slot)
            active_feat = sv() if sv is not None else None
        _load_into_drawcontrol([active_feat] if active_feat else [])
        _render_boundaries(slot)

    @reactive.effect
    def _refresh_domain_outline():
        # Keep the derived-domain outline live as sides are edited (the calc tracks the four slots).
        # Route through _bnd_show so the Domain layer's tracked signature stays in sync with the
        # rest of the Boundaries overlays (idempotent — no churn when the domain is unchanged).
        if not _HAS_MAP or current_step() != STEP_BOUNDARIES:
            return
        _bnd_show("Domain", domain_feat(), DOMAIN_STYLE)

    @reactive.effect
    def _frame_boundaries_on_entry():
        # Frame the derived domain when the user *lands* on the Boundaries step (step transition
        # only — not on every slot change), so the lines aren't hidden behind the left panel and it
        # never fights the user's pan/zoom mid-edit.
        step = current_step()
        prev = _map_ui.get("step")
        _map_ui["step"] = step
        if _HAS_MAP and step == STEP_BOUNDARIES and prev not in (None, STEP_BOUNDARIES):
            _fit_domain()

    def _features_of(gj):
        """Feature dicts from an on_draw `geo_json` payload (Feature / FeatureCollection / bare
        geometry). On an EDIT, ipyleaflet hands the fresh edited geometry here but does NOT update
        dc.data at the same time (that trait syncs via a separate, unordered message), so this is the
        reliable source for a just-committed shape — reading dc.data would re-save the old geometry."""
        if not isinstance(gj, dict):
            return []
        t = gj.get("type")
        if t == "FeatureCollection":
            return [f for f in (gj.get("features") or []) if isinstance(f, dict)]
        if t == "Feature":
            return [gj]
        if gj.get("coordinates") is not None:              # bare geometry
            return [{"type": "Feature", "properties": {}, "geometry": gj}]
        return []

    def _snap_boundary_endpoints(slot, feat):
        """Snap the committed boundary line's two endpoints onto the nearest endpoint of the OTHER
        three boundaries when within a zoom-scaled tolerance (~16 px), so shared corners actually
        meet. Returns the (possibly snapped) Feature — unchanged if there's no projected CRS yet or
        nothing is close. Reuses the px→m metric from _bnd_pick_on_click."""
        geom = (feat or {}).get("geometry") or {}
        if geom.get("type") != "LineString":
            return feat
        coords = [list(c) for c in (geom.get("coordinates") or [])]
        crs = proj_crs()
        if len(coords) < 2 or crs is None:
            return feat
        neighbours = []
        for k, v in {"up": up_feat, "left": left_feat, "right": right_feat, "down": down_feat}.items():
            if k == slot:
                continue
            c = (((v() or {}).get("geometry") or {}).get("coordinates")) or []
            if len(c) >= 2:
                neighbours.append(tuple(c[0][:2])); neighbours.append(tuple(c[-1][:2]))
        if not neighbours:
            return feat
        try:
            import math
            import geopandas as gpd
            from shapely.geometry import Point
            ep_idx = [0, len(coords) - 1]
            lonlat = [tuple(coords[i][:2]) for i in ep_idx] + neighbours
            proj = list(gpd.GeoSeries([Point(lo, la) for lo, la in lonlat], crs=4326).to_crs(crs))
            z = getattr(_MAP, "zoom", None) or 16   # read the trait directly — on_draw isn't reactive,
            mpp = 156543.03 * math.cos(math.radians(float(coords[0][1]))) / (2 ** int(z))  # so _view() (a
            tol = 28.0 * mpp                        # reactive.calc) could raise here and silently no-op
            n = len(ep_idx)
            for j, i in enumerate(ep_idx):
                p = proj[j]
                best_d, best_k = None, None
                for m in range(len(neighbours)):
                    d = p.distance(proj[n + m])
                    if best_d is None or d < best_d:
                        best_d, best_k = d, m
                if best_d is not None and best_d <= tol:
                    coords[i] = list(neighbours[best_k])       # snap endpoint onto the neighbour
            return {"type": "Feature", "properties": (feat or {}).get("properties") or {},
                    "geometry": {"type": "LineString", "coordinates": coords}}
        except Exception:  # noqa: BLE001
            return feat

    def _reclassify_drawn(action=None, geo_json=None):
        """Re-derive feature values from the just-drawn/edited shape, routed by step: Reach (manual) →
        the drawn line is the reach centerline; K → polygons are K-zones; Boundaries → the single
        shape goes to the active boundary slot (up/left/right/down = line, wse = polygon). Prefer the
        fresh `geo_json` from the draw event; dc.data is stale on edits (see _features_of)."""
        dc = _draw_ctl.get("dc")
        data_feats = list(getattr(dc, "data", None) or [])
        fresh = _features_of(geo_json)
        step = current_step()
        if step == STEP_REACH:
            src = fresh or data_feats
            lines = [f for f in src if (f.get("geometry") or {}).get("type") == "LineString"]
            if delineate_mode() == "manual" and lines:
                reach_feat.set(lines[0])
            return
        if step == STEP_K:
            kzone_feats.set([f for f in data_feats if (f.get("geometry") or {}).get("type") == "Polygon"])
            kz_adding.set(False)             # a guided "Add K-zone" draw just completed
            return
        if step != STEP_BOUNDARIES:
            return
        slot = bnd_slot()
        if not slot:
            return
        want = "Polygon" if slot == "wse" else "LineString"
        src = fresh or data_feats
        match = next((f for f in src if (f.get("geometry") or {}).get("type") == want), None)
        sv = _slot_value(slot)
        if match is not None and sv is not None:
            if slot != "wse":
                match = _snap_boundary_endpoints(slot, match)   # snap ends onto nearby neighbour ends
            sv.set(match)
            bnd_slot.set(None)          # commit done → deselect (line becomes a clickable static)

    def _load_into_drawcontrol(feats):
        """Put generated GeoJSON Features into the DrawControl so the user can edit them."""
        dc = _draw_ctl.get("dc")
        if dc is None:
            return
        try:
            dc.data = [f for f in feats if f]
        except Exception:  # noqa: BLE001
            pass

    # ---- persistent map + draw control ----
    if _HAS_MAP:
        def _build_map():
            m = Map(center=(39.5, -98.35), zoom=4, scroll_wheel_zoom=True,
                    zoom_control=False, max_zoom=19, layout=Layout(height="100%"))
            m.clear()
            m.add(ZoomControl(position="topright"))
            # USGS basemap caches stop at zoom 16 — cap max_native_zoom so Leaflet upscales the
            # deepest real tiles past 16 instead of showing blank tiles.
            m.add(TileLayer(url=USGS_IMAGERY, name="USGS Imagery", base=True, attribution=USGS_ATTR,
                            max_native_zoom=16, max_zoom=19))
            m.add(TileLayer(url=USGS_TOPO, name="USGS Topo", base=True, attribution=USGS_ATTR,
                            max_native_zoom=16, max_zoom=19))
            m.add(TileLayer(url=USGS_HYDRO, name="NHD Hydrography", base=False, opacity=0.85,
                            attribution=USGS_ATTR, max_native_zoom=16, max_zoom=19))
            dc = DrawControl(
                position="topright",
                polygon={"shapeOptions": {"color": "#caa700", "fillColor": "#fdf24a",
                                          "fillOpacity": 0.1}},
                polyline={"shapeOptions": {"color": "#ff2d95", "weight": 4}},
                rectangle={}, circle={}, circlemarker={}, marker={},
            )
            _draw_ctl["dc"] = dc

            def _on_draw(target, action, geo_json):
                _reclassify_drawn(action=action, geo_json=geo_json)  # re-derive from the drawn shape

            dc.on_draw(_on_draw)
            m.add(dc)
            m.add(LayersControl(position="topright"))
            m.add(ScaleControl(position="bottomleft"))

            def _on_interaction(**kw):     # capture map clicks for upstream/downstream picking
                if kw.get("type") == "click":
                    c = kw.get("coordinates") or [None, None]
                    if c[0] is not None:
                        last_click.set((float(c[0]), float(c[1])))
            m.on_interaction(_on_interaction)
            return m

        _MAP = _build_map()

        @render_widget
        def map():  # noqa: A001
            return _MAP

        @reactive.calc
        def _view():
            return reactive_read(_MAP, "zoom"), reactive_read(_MAP, "center")

        def _fit_domain():
            """Frame the derived domain in the *visible* map area. The left ~336 px is the app panel,
            so a small domain can otherwise sit hidden behind it. ipyleaflet's fit_bounds has no
            padding arg, so pad the bounds manually and bias west (extra left padding) to push the
            domain toward the right, visible half of the map."""
            with reactive.isolate():
                g = _domain_gdf_4326()
            if g is None:
                return
            minx, miny, maxx, maxy = (float(v) for v in g.total_bounds)
            dx = (maxx - minx) or 1e-4
            dy = (maxy - miny) or 1e-4
            b = [[miny - 0.25 * dy, minx - 0.9 * dx], [maxy + 0.25 * dy, maxx + 0.25 * dx]]
            try:
                _MAP.fit_bounds(b)
            except Exception:  # noqa: BLE001
                pass

    # ---- DEM fetch ----
    @reactive.extended_task
    async def dem_task(domain_geojson: dict, out_path: str, resolution) -> dict:
        def _work():
            g = geometry.single_feature_gdf(domain_geojson)
            info = dem.fetch_dem(g, out_path, resolution=resolution)
            return {"path": info["path"], "resolution_m": info["resolution_m"],
                    "source": info["source"], "summary": dem.dem_summary(info["path"])}
        return await anyio.to_thread.run_sync(_work)

    def _reach_meta():
        """Drainage area + midpoint + Bieger bankfull geometry for the current reach. AUTO reads
        the NHD-derived auto_meta; MANUAL derives it from the drawn centerline + the user's
        Drainage-area input. Returns None if there's no reach yet."""
        if delineate_mode() != "manual":
            return auto_meta()
        rf = reach_feat()
        if rf is None:
            return None
        import geopandas as gpd
        from shapely.geometry import shape as _shape
        da = float(_safe("manual_da", 1.0))
        line = _shape(rf["geometry"])
        mid = line.interpolate(0.5, normalized=True)
        try:
            length_m = float(gpd.GeoSeries([line], crs=4326).to_crs(5070).length.iloc[0])
        except Exception:  # noqa: BLE001
            length_m = 0.0
        bf = bieger.bankfull_geometry(da, mid.y, mid.x)
        return {"da_sqkm": da, "length_m": length_m, "lat": float(mid.y),
                "lon": float(mid.x), **bf}

    @reactive.effect
    @reactive.event(input.fetch_dem)
    def _fetch_dem():
        rf = reach_feat()
        if rf is None:
            ui.notification_show("Define a reach first (Reach tab).", type="warning", duration=5)
            return
        import geopandas as gpd
        from shapely.geometry import mapping, shape as _shape
        meta = _reach_meta() or {}
        half = min(max(8.0 * max(meta.get("width_m", 1.0), 1.0), 250.0), 800.0)
        buf = (gpd.GeoSeries([_shape(rf["geometry"])], crs=4326).to_crs(5070)
               .buffer(half + 60.0).to_crs(4326).iloc[0])
        stage.set("Downloading 3DEP terrain for the reach…")
        dem_task({"type": "Feature", "properties": {}, "geometry": mapping(buf)},
                 str(work_dir / "inputs" / "dem.tif"), _safe("dem_res", "auto"))

    @reactive.effect
    def _dem_done():
        if dem_task.status() in ("initial", "running"):
            return
        stage.set("")
        if dem_task.status() == "error":
            ui.notification_show("DEM fetch failed at all 3DEP resolutions — try a smaller area.",
                                 type="error", duration=8)
            return
        try:
            res = dem_task.result()
        except Exception:
            return
        dem_path.set(res["path"])
        dem_meta.set({"resolution_m": res.get("resolution_m"), "source": res.get("source")})
        if _HAS_MAP:                       # hillshade backdrop
            try:
                ov = dem.dem_overlay(res["path"])
                _set_layer("dem", ImageOverlay(url=ov["url"], bounds=ov["bounds"],
                                               name="DEM (hillshade)", opacity=0.8))
            except Exception as e:  # noqa: BLE001
                ui.notification_show(f"DEM loaded; overlay render issue: {e}", duration=5)
        # Terrain only — boundary delineation happens on the Boundaries tab ("Generate boundaries").
        ui.notification_show("Terrain ready — continue to Boundaries.", duration=4)

    # ---- auto-delineation: NHD streams → pick 2 points → reach → cross-sections ----
    @reactive.extended_task
    async def flow_task(bbox: tuple) -> dict:
        return await anyio.to_thread.run_sync(lambda: hydro.flowlines_bbox(*bbox) or {})

    def _do_flow_fetch(force=False):
        # Fetch box from the map CENTER + zoom-scaled radius (the viewport `bounds` trait is
        # unreliable per EASI; center/zoom always update via _view).
        if not _HAS_MAP:
            return
        z, c = _view()
        if not c or z is None or int(z) < 12:
            nhd_status.set("Zoom in to load streams.")
            return
        lat, lon = float(c[0]), float(c[1])
        delta = min(0.08, 0.03 * (2 ** (15 - int(z))))   # half-box in degrees
        bbox = (round(lon - delta, 3), round(lat - delta, 3),
                round(lon + delta, 3), round(lat + delta, 3))
        if not force and _flow.get("bbox") == bbox:      # already fetched this view
            return
        _flow["bbox"] = bbox
        nhd_status.set("")                 # the bottom "Loading streams…" spinner shows progress
        flow_task(bbox)

    @reactive.effect
    def _load_flowlines():
        if delineate_mode() != "auto" or current_step() != STEP_REACH:
            return
        _do_flow_fetch()                                 # reads _view() → fires on pan/zoom

    @reactive.effect
    @reactive.event(input.address_pick)
    def _on_address_pick():
        # A suggestion was chosen in the type-ahead dropdown (coords come from the client-side
        # Photon query in www/geocode.js) — recenter the map; _load_flowlines then auto-fetches
        # the NHD streams at the new view.
        if not _HAS_MAP:
            return
        p = input.address_pick() or {}
        lat, lon = p.get("lat"), p.get("lon")
        if lat is None or lon is None:
            return
        _MAP.center = (float(lat), float(lon))
        _MAP.zoom = 15

    @reactive.effect
    @reactive.event(input.find_address)
    def _find_address():
        # Button fallback: geocode server-side (Photon → Nominatim) and recenter.
        if not _HAS_MAP:
            return
        hit = geocode.geocode_address(_safe("address", ""))
        if hit:
            _MAP.center = (float(hit[0]), float(hit[1]))
            _MAP.zoom = 15
        else:
            ui.notification_show("Place not found — try a city, address, or stream name.",
                                 type="warning", duration=5)

    @reactive.effect
    def _flow_done():
        if flow_task.status() in ("initial", "running"):
            return
        if flow_task.status() == "error":
            nhd_status.set("Couldn't load streams — try again.")
            try:
                flow_task.result()
            except Exception as e:  # noqa: BLE001
                ui.notification_show(f"NHD streams failed: {e}", type="warning", duration=8)
            return
        try:
            gj = flow_task.result()
        except Exception:  # noqa: BLE001
            return
        n = len(gj.get("features", [])) if gj else 0
        if n:
            _set_layer("NHD streams", GeoJSON(data=gj, style=NHD_STYLE, name="NHD streams"))
            try:
                import geopandas as gpd
                _flow["gdf"] = gpd.GeoDataFrame.from_features(gj["features"], crs=4326)
            except Exception:  # noqa: BLE001
                _flow["gdf"] = None
            nhd_status.set(f"✓ {n} streams — click the upstream + downstream points.")
        else:
            nhd_status.set("No streams here — pan to a stream, or draw manually.")

    @reactive.extended_task
    async def snap_task(lat: float, lon: float) -> dict:
        return await anyio.to_thread.run_sync(lambda: hydro.snap(lat, lon, _flow.get("gdf")) or {})

    @reactive.effect
    @reactive.event(last_click)
    def _on_click_pick():
        if delineate_mode() != "auto" or reach_feat() is not None:
            return
        if len(pick_pts()) >= 2 or snap_task.status() == "running":
            return
        c = last_click()
        if not c:
            return
        stage.set("Snapping to the nearest stream…")
        snap_task(c[0], c[1])

    @reactive.effect
    def _snap_done():
        if snap_task.status() in ("initial", "running"):
            return
        stage.set("")
        if snap_task.status() == "error":
            ui.notification_show("Couldn't reach the NHD service — try again, or use manual drawing.",
                                 type="warning", duration=6)
            return
        try:
            sp = snap_task.result()
        except Exception:  # noqa: BLE001
            return
        if not sp or sp.get("comid") is None:
            ui.notification_show("No NHD stream near that click — zoom in and click a cyan "
                                 "flowline.", type="warning", duration=6)
            return
        # Read+write pick_pts inside isolate() so this effect doesn't depend on pick_pts —
        # otherwise the .set() below would re-trigger it, appending the same point forever.
        with reactive.isolate():
            pts = list(pick_pts())
            same_as_last = (pts and pts[-1].get("comid") == sp.get("comid")
                            and abs(pts[-1]["lat"] - sp["lat"]) < 1e-7
                            and abs(pts[-1]["lon"] - sp["lon"]) < 1e-7)
            if len(pts) >= 2 or same_as_last:
                return                                   # already have two, or a duplicate re-fire
            pts.append(sp)
            pick_pts.set(pts)
        n = len(pts)
        _set_layer(f"pick{n}", Marker(location=(sp["lat"], sp["lon"]), draggable=False,
                   title=("Upstream point" if n == 1 else "Downstream point")))
        if n == 2:
            stage.set("Tracing the reach along the NHD…")
            reach_task(pts[0], pts[1])
        else:
            ui.notification_show("Upstream point set — now click the downstream point.", duration=4)

    @reactive.extended_task
    async def reach_task(up: dict, dn: dict) -> dict:
        return await anyio.to_thread.run_sync(lambda: hydro.reach_between(up, dn))

    @reactive.effect
    def _reach_done():
        if reach_task.status() in ("initial", "running"):
            return
        stage.set("")
        if reach_task.status() == "error":
            try:
                reach_task.result()
            except Exception as e:  # noqa: BLE001
                print(f"[reach] error: {e!r}", flush=True)
                ui.notification_show(str(e), type="error", duration=10)
            pick_pts.set([]); _set_layer("pick1", None); _set_layer("pick2", None)
            return
        try:
            r = reach_task.result()
        except Exception:  # noqa: BLE001
            return
        bf = bieger.bankfull_geometry(r["da_sqkm"], r["lat"], r["lon"])
        auto_meta.set({"da_sqkm": r["da_sqkm"], "length_m": r["length_m"],
                       "lat": r["lat"], "lon": r["lon"], **bf})
        reach_feat.set(r["reach"])
        _set_layer("Reach", GeoJSON(data=r["reach"], style=REACH_STYLE, name="Reach"))
        _set_layer("pick1", None); _set_layer("pick2", None)   # drop the transient pick markers
        for w in r.get("warnings", []):
            ui.notification_show(w, duration=5)
        ui.notification_show(f"Reach {r['length_m']/1609.344:.2f} mi · drainage area "
                             f"{r['da_sqkm']:.1f} km² · bankfull depth {bf['depth_m']:.2f} m "
                             f"({bf['division_name']}). Fetching terrain…", duration=7)

    @reactive.extended_task
    async def delineate_task(reach, dem_p, da, lat, lon, x_mult) -> dict:
        return await anyio.to_thread.run_sync(
            lambda: delineate.auto_delineate(reach, dem_p, da_sqkm=da, lat=lat, lon=lon,
                                             x_mult=x_mult))

    @reactive.effect
    def _delineate_done():
        if delineate_task.status() in ("initial", "running"):
            return
        stage.set("")
        if delineate_task.status() == "error":
            try:
                delineate_task.result()
            except Exception as e:  # noqa: BLE001
                ui.notification_show(f"Auto-delineation failed: {e}. You can switch to manual "
                                     f"drawing.", type="error", duration=10)
            return
        try:
            d = delineate_task.result()
        except Exception:  # noqa: BLE001
            return
        # Fill the four named boundary slots (domain derives from them); up/down are now first-class
        # editable boundaries, not static caps.
        up_feat.set(d.get("up_cap")); left_feat.set(d["left"])
        right_feat.set(d["right"]); down_feat.set(d.get("down_cap"))
        wse_extent_feat.set(d["wse_extent"]); wse_mode_v.set("draw")
        with reactive.isolate():
            on_boundaries = current_step() == STEP_BOUNDARIES
            bnd_slot.set(None)                 # deselect; nothing armed until a boundary button is clicked
        if on_boundaries:
            _load_into_drawcontrol([])
            _render_boundaries(None)
        else:                                  # generated before reaching Boundaries → show statics
            _mirror_features_as_layers()
        if _HAS_MAP:
            _fit_domain()                      # frame the fresh domain clear of the left panel
        ui.notification_show("Domain, boundaries & wetted extent generated — open the Boundaries tab "
                             "to review/edit each boundary.", duration=8)

    @reactive.effect
    @reactive.event(input.regen)
    def _regenerate():
        if reach_feat() is None or dem_path() is None:
            ui.notification_show("Define a reach and fetch the DEM first.", type="warning", duration=5)
            return
        meta = _reach_meta() or {}
        stage.set("Building cross-sections…")
        delineate_task(reach_feat(), dem_path(), meta.get("da_sqkm", 0.0),
                       meta.get("lat"), meta.get("lon"), float(_safe("fp_mult", 10)))

    # ---- parameters + estimate ----
    @reactive.calc
    def params():
        bc = _safe("bc_mode", BC_CORNER)
        base = dict(
            cell_size_x=float(_safe("cell_size", 10.0)), cell_size_y=float(_safe("cell_size", 10.0)),
            gw_mod_depth=float(_safe("gw_mod_depth", 6.0)), z=float(_safe("z", 0.25)),
            kh=float(_safe("kh", 10.0)), kv=float(_safe("kv", 1.0)),
            porosity=float(_safe("porosity", 0.3)),
            length_units="meters", time_units="days",
            # steady hyporheic screening defaults — no stress-period fields in the UI
            nper=1, nstp=1, perlen=1.0, tsmult=1.0, sim_name="hyporheic",
            boundary_condition_mode=bc,
        )
        if bc == BC_PROFILE:
            base["left_boundary_gradient_profile"] = _safe("g_left_profile", "0,0.005 0.5,0.005 1,0.005")
            base["right_boundary_gradient_profile"] = _safe("g_right_profile", "0,0.005 0.5,0.005 1,0.005")
        else:
            base.update(
                upstream_left_fpl_gw_gradient=float(_safe("g_ul", 0.005)),
                upstream_right_fpl_gw_gradient=float(_safe("g_ur", 0.005)),
                downstream_left_fpl_gw_gradient=float(_safe("g_dl", 0.005)),
                downstream_right_fpl_gw_gradient=float(_safe("g_dr", 0.005)),
            )
        return base

    @reactive.calc
    def grid_estimate():
        g = _domain_gdf_4326()
        crs = proj_crs()
        if g is None or crs is None:
            return None
        try:
            return estimate.estimate_cells(g.to_crs(crs), float(input.cell_size()),
                                           float(input.gw_mod_depth()), float(input.z()))
        except Exception:  # noqa: BLE001
            return None

    # ---- 3D mesh preview (server builds geometry in pure numpy → vtk.js renders client-side) ----
    @reactive.extended_task
    async def mesh_task(domain_f, dem_p, crs, cell_size, depth, z) -> dict:
        return await anyio.to_thread.run_sync(
            lambda: mesh.build_grid_geometry(domain_f, dem_p, crs, cell_size, depth, z))

    @reactive.effect
    @reactive.event(input.compute_mesh)
    def _compute_mesh():
        build = _domain_build()
        if not (build and dem_path() and proj_crs() is not None):
            ui.notification_show("Need the four boundaries and terrain first.",
                                 type="warning", duration=5)
            return
        stage.set("Building the 3D mesh…")
        mesh_task(build["domain"], dem_path(), proj_crs(), float(_safe("cell_size", 10.0)),
                  float(_safe("gw_mod_depth", 6.0)), float(_safe("z", 0.25)))

    @reactive.effect
    async def _mesh_done():
        if mesh_task.status() in ("initial", "running"):
            return
        stage.set("")
        if mesh_task.status() == "error":
            try:
                mesh_task.result()
            except Exception as e:  # noqa: BLE001
                ui.notification_show(f"Mesh build failed: {e}", type="error", duration=8)
            return
        try:
            g = mesh_task.result()
        except Exception:  # noqa: BLE001
            return
        mesh_geom.set(g)
        await session.send_custom_message("hype_mesh", g)

    def _wse_path():
        """Resolve the WSE raster the engine will use: the uploaded raster, or the DEM clipped
        to the drawn wetted-extent polygon. Returns None if neither is available yet."""
        if wse_mode_v() == "upload":
            up = _safe("wse_upload", None)
            if not up:
                return None
            dst = work_dir / "inputs" / "wse_upload.tif"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(up[0]["datapath"], dst)
            return str(dst)
        feat = wse_extent_feat()
        if not feat or dem_path() is None:
            return None
        out = work_dir / "inputs" / "wse_extent.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        return dem.clip_dem_to_polygon(dem_path(), geometry.single_feature_gdf(feat), str(out))

    # ---- run (in a spawned child process so a Cancel can hard-kill MODFLOW) ----
    @reactive.extended_task
    async def run_task(payload: dict) -> dict:
        def _work():
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(target=runner.child_run, args=(payload, q), daemon=True)
            _proc["p"] = p
            p.start()
            result = error = None

            def _consume(item):
                nonlocal result, error
                kind, data = item
                if kind == "log":
                    log_lines.append(data)
                elif kind == "result":
                    result = data
                elif kind == "error":
                    error = data

            while True:
                try:
                    _consume(q.get(timeout=0.3))
                except _queue.Empty:
                    if not p.is_alive():
                        break
            while True:                       # drain whatever was queued right before exit
                try:
                    _consume(q.get_nowait())
                except _queue.Empty:
                    break
            p.join(timeout=5)
            _proc["p"] = None
            if error is not None:
                raise RuntimeError(error)
            if result is None:
                raise RuntimeError("Run produced no result (it may have been cancelled).")
            return result
        return await anyio.to_thread.run_sync(_work)

    @reactive.effect
    @reactive.event(input.run_model)
    def _start_run():
        build = _domain_build()                 # assembled domain + left/right oriented upstream→downstream
        if not (build and dem_path()):
            ui.notification_show("Need all four boundaries (Upstream/Left/Right/Downstream) that close "
                                 "into a domain, plus terrain.", type="warning", duration=6)
            return
        est = grid_estimate()
        if est and estimate.band(est["n_cells"]) == "red":
            ui.notification_show(estimate.band_message(est), type="error", duration=10)
            return
        wse = _wse_path()
        if wse is None:
            ui.notification_show("Draw the water-surface extent (or choose Upload and select a "
                                 "WSE raster).", type="warning", duration=6)
            return
        try:
            crs = proj_crs()
            crs_id = crs.to_epsg() or crs.to_wkt()      # picklable for the child process
            use_kz = bool(_safe("use_kzones", False))
            payload = {
                "crs": crs_id, "domain": build["domain"], "left": build["left"],
                "right": build["right"], "dem": dem_path(), "params": params(),
                "work_dir": str(work_dir),
                "wse_mode": "dem",          # fallback only; wse_path (below) always wins
                "wse_path": wse,
                "wse_relief_thresh": float(_safe("wse_relief", 0.2)),
                "kzones": (kzone_feats() if use_kz else []),
                "kzone_kh": float(_safe("kzone_kh", 50.0)),
                "kzone_kv": float(_safe("kzone_kv", 5.0)),
            }
        except Exception as e:  # noqa: BLE001
            ui.notification_show(f"Could not start the run: {type(e).__name__}: {e}",
                                 type="error", duration=8)
            return
        log_lines.clear()
        log_tick.set(0)
        step_v.set(0)
        run_t0.set(time.monotonic())
        elapsed_v.set(0)
        stage.set("Running MODFLOW 6 + MODPATH 7…")
        current_step.set(STEP_RUN)
        run_task(payload)

    @reactive.effect
    def _run_poll():
        if run_task.status() != "running":
            return
        reactive.invalidate_later(0.4)
        log_tick.set(len(log_lines))
        elapsed_v.set(int(time.monotonic() - run_t0()))
        for line in reversed(log_lines[-80:]):       # newest STEP marker wins
            m = re.search(r"STEP\s+(\d+)", line)
            if m:
                step_v.set(int(m.group(1)))
                break

    @reactive.effect
    def _run_done():
        status = run_task.status()
        if status in ("initial", "running"):
            return
        stage.set("")
        if status == "cancelled":
            return  # the Cancel handler already reset the UI
        if status == "error":
            msg = "Model run failed."
            try:
                run_task.result()
            except Exception as e:  # noqa: BLE001
                detail = str(e)
                if "No particles" in detail:
                    msg = ("No hyporheic pathlines were produced (all particles exited at the "
                           "boundaries). Try a stronger floodplain gradient, a different "
                           "water-surface option, or a larger domain.")
                else:
                    msg = f"Model run failed: {detail}"
            log_lines.append(msg); log_tick.set(len(log_lines))
            ui.notification_show(msg, type="error", duration=12)
            current_step.set(STEP_MESH)
            return
        try:
            res = run_task.result()
        except Exception:
            ui.notification_show("Model run failed.", type="error", duration=8)
            return
        run_result.set(res)
        if _HAS_MAP:
            try:
                pls = results.pathlines_geojson(res)
                pts = results.points_geojson(res)
                _set_layer("paths", GeoJSON(data=pls, style=PATH_STYLE, name="Pathlines") if pls else None)
                _set_layer("points", GeoJSON(data=pts, point_style=POINT_STYLE,
                                             name="Particle points") if pts else None)
                b = results.bounds_latlon(res)
                if b:
                    _MAP.fit_bounds(b)
                tifs = results.head_rasters(work_dir, res)   # per-layer head color map + grid
                _head_cache.clear(); _contour_cache.clear()
                head_tifs.set(tifs)
                if tifs:
                    head_rng.set(results.head_value_range(tifs))
                    grid = results.grid_overlay(tifs)            # active cells only (≈ idomain)
                    _set_layer("grid", ImageOverlay(url=grid["url"], bounds=grid["bounds"],
                                                    name="Model grid", opacity=0.7) if grid else None)
                    _render_head_layer(1)
            except Exception as e:  # noqa: BLE001
                ui.notification_show(f"Results computed; map render issue: {e}", duration=6)
        current_step.set(STEP_RESULTS)
        ui.notification_show("Run complete.", duration=4)

    @reactive.effect
    def _update_head_layer():
        try:
            idx = input.head_layer()        # slider exists only on the Results step
        except Exception:  # noqa: BLE001
            return
        if idx is None or not head_tifs():
            return
        head_layer_v.set(int(idx))          # persist so the slider survives pane re-renders
        try:
            _render_head_layer(idx)
        except Exception as e:  # noqa: BLE001
            ui.notification_show(f"Head layer render issue: {e}", duration=5)

    @reactive.effect
    def _head_opacity():
        try:
            op = input.head_opacity()       # mutable ImageOverlay.opacity → live, no re-render
        except Exception:  # noqa: BLE001
            return
        if op is None:
            return
        head_opacity_v.set(float(op))       # persist so the slider survives pane re-renders
        lyr = _layers.get("head")
        if lyr is not None:
            try:
                lyr.opacity = float(op)
            except Exception:  # noqa: BLE001
                pass

    @reactive.effect
    @reactive.event(input.cancel_run)
    def _cancel_run():
        _terminate_child()
        log_lines.append("[run cancelled by user]")
        log_tick.set(len(log_lines))
        try:
            run_task.cancel()
        except Exception:  # noqa: BLE001
            pass
        stage.set("")
        current_step.set(STEP_MESH)
        ui.notification_show("Run cancelled.", type="warning", duration=4)

    # ---- navigation ----
    def _reachable():
        r = {STEP_REACH}
        if reach_feat() is not None:
            r.add(STEP_DEM)
        if dem_path() is not None:
            r.add(STEP_BOUNDARIES)
        if _domain_build() is not None:          # all four boundaries close into a valid domain
            r.update({STEP_K, STEP_MESH})
        if run_result() is not None:
            r.update({STEP_RUN, STEP_RESULTS})
        return r

    @reactive.effect
    def _stepper_nav():
        for key, _ in STEP_LABELS:
            try:
                n = input[f"go_{key}"]()
            except Exception:
                n = 0
            if n and key in _reachable():
                current_step.set(key)

    _nav_seen: dict = {}

    @reactive.effect
    def _continue_nav():
        # Per-tab "Continue" buttons use DISTINCT ids + a strict-increment guard so a button-count
        # reset on a leftpane re-render can't spuriously re-fire navigation (the shared-go_next
        # footgun: @reactive.event fired on the 1→0 reset and tried to advance an extra step).
        reach = _reachable()
        for bid, tgt in (("next_reach", STEP_DEM), ("next_dem", STEP_BOUNDARIES),
                         ("next_boundaries", STEP_K), ("next_k", STEP_MESH)):
            try:
                n = int(input[bid]() or 0)
            except Exception:  # noqa: BLE001
                n = 0
            last = _nav_seen.get(bid, 0)
            if n != last:
                _nav_seen[bid] = n
                if n > last:                          # a real click (not a re-render reset)
                    if tgt in reach:
                        current_step.set(tgt)
                    else:
                        ui.notification_show("Finish this step first.", type="warning", duration=4)

    @reactive.effect
    @reactive.event(last_click)
    def _bnd_pick_on_click():
        # Boundaries editing is map-driven: click on/near a boundary line to select + edit it.
        # Only when nothing is being edited (else clicks add vertices via Leaflet.draw). Picks the
        # nearest boundary within a zoom-scaled pixel tolerance (forgiving on thin lines).
        if current_step() != STEP_BOUNDARIES or bnd_slot() is not None:
            return
        c = last_click(); crs = proj_crs()
        if not c or crs is None:
            return
        cands = {"up": up_feat(), "left": left_feat(), "right": right_feat(),
                 "down": down_feat(), "wse": wse_extent_feat()}
        cands = {k: v for k, v in cands.items() if v}
        if not cands:
            return
        import math
        import geopandas as gpd
        from shapely.geometry import Point, shape as _shape
        try:
            pt = gpd.GeoSeries([Point(float(c[1]), float(c[0]))], crs=4326).to_crs(crs).iloc[0]
            best, best_d = None, None
            for slot, f in cands.items():
                g = gpd.GeoSeries([_shape(f["geometry"])], crs=4326).to_crs(crs).iloc[0]
                d = pt.distance(g.boundary if g.geom_type == "Polygon" else g)
                if best_d is None or d < best_d:
                    best, best_d = slot, d
            z = _view()[0] or 16
            mpp = 156543.03 * math.cos(math.radians(float(c[0]))) / (2 ** int(z))
            if best is not None and best_d <= 14 * mpp:           # ~14 px tolerance
                bnd_slot.set(best)
        except Exception:  # noqa: BLE001
            return

    @reactive.effect
    @reactive.event(input.bnd_done)
    def _bnd_done():
        bnd_slot.set(None)              # "Done" → deselect; boundaries become clickable statics again

    @reactive.effect
    @reactive.event(input.bnd_clear)
    def _bnd_clear():
        with reactive.isolate():
            sv = _slot_value(bnd_slot())
        if sv is not None:
            sv.set(None)                # "Clear & redraw" → empty the slot; _push then arms a draw
            dc = _draw_ctl.get("dc")    # drop the old shape now so the fresh draw starts from empty
            if dc is not None:          # (else it lingers and _reclassify picks it, not the new one)
                try:
                    dc.clear(); dc.data = []
                except Exception:  # noqa: BLE001
                    pass

    @reactive.effect
    def _bnd_draw_links():
        # The legend's "Draw" links (only shown on empty rows) select that slot → _push arms a draw.
        # Strict-increment guard (legend re-renders, resetting link counts) — like _continue_nav.
        for slot in ("up", "left", "right", "down", "wse"):
            bid = f"bnd_draw_{slot}"
            try:
                n = int(input[bid]() or 0)
            except Exception:  # noqa: BLE001
                n = 0
            if n != _nav_seen.get(bid, 0):
                up = n > _nav_seen.get(bid, 0)
                _nav_seen[bid] = n
                if up:
                    bnd_slot.set(slot)

    @reactive.effect
    def _snap_corners():
        # "Snap corners together" (open-domain warning) → write the assembled snapped sides back so the
        # four corners coincide and the domain closes. Strict-increment guard (button lives in the
        # re-rendered domain_warning, so a plain @reactive.event would re-fire on the count reset).
        try:
            n = int(input.snap_corners() or 0)
        except Exception:  # noqa: BLE001
            n = 0
        if n != _nav_seen.get("snap_corners", 0):
            up = n > _nav_seen.get("snap_corners", 0)
            _nav_seen["snap_corners"] = n
            if up:
                with reactive.isolate():
                    b = _domain_build()
                if b:
                    up_feat.set(b["up"]); left_feat.set(b["left"])
                    right_feat.set(b["right"]); down_feat.set(b["down"])

    @reactive.effect
    def _kz_buttons():
        # K-zone list management (same strict-increment guard as _continue_nav so leftpane
        # re-render resets don't fire): Add → arm a guided polygon draw; Remove last / Clear all.
        def _clicked(bid):
            try:
                n = int(input[bid]() or 0)
            except Exception:  # noqa: BLE001
                n = 0
            last = _nav_seen.get(bid, 0)
            if n != last:
                _nav_seen[bid] = n
                return n > last
            return False
        if _clicked("kz_add"):
            kz_adding.set(True)
        if _clicked("kz_rmlast"):
            kz = list(kzone_feats())
            if kz:
                kz.pop()
                kzone_feats.set(kz)
                _load_into_drawcontrol(kz)
            kz_adding.set(False)
        if _clicked("kz_clear"):
            kzone_feats.set([])
            _load_into_drawcontrol([])
            kz_adding.set(False)

    @reactive.effect
    def _reset_kz_adding():
        if current_step() != STEP_K:           # disarm a pending Add when leaving the K step
            with reactive.isolate():
                if kz_adding():
                    kz_adding.set(False)

    def _clear_auto_picks():
        pick_pts.set([]); reach_feat.set(None); auto_meta.set(None); last_click.set(None)
        for nm in ("pick1", "pick2", "Reach", "Upstream cap", "Downstream cap"):
            _set_layer(nm, None)
        dc = _draw_ctl.get("dc")
        if dc is not None:
            try:
                dc.clear()          # actually removes drawn shapes (dc.data=[] alone doesn't)
                dc.data = []
            except Exception:  # noqa: BLE001
                pass

    @reactive.effect
    @reactive.event(input.clear_points)
    def _clear_points():
        _clear_auto_picks()
        up_feat.set(None); left_feat.set(None); right_feat.set(None); down_feat.set(None)
        wse_extent_feat.set(None); bnd_slot.set(None)
        dem_path.set(None); dem_meta.set(None)   # also drop the downloaded DEM + its overlay
        _set_layer("dem", None)
        ui.notification_show("Cleared points, linework, and DEM — pick a new upstream and "
                             "downstream point.", duration=4)

    @reactive.effect
    @reactive.event(input.clear_draw)
    def _clear_draw():
        up_feat.set(None); left_feat.set(None); right_feat.set(None); down_feat.set(None)
        kzone_feats.set([]); wse_extent_feat.set(None); bnd_slot.set(None)
        dem_path.set(None); dem_meta.set(None)
        _set_layer("dem", None)
        _clear_auto_picks()
        ui.notification_show("Cleared.", duration=3)

    @reactive.effect
    @reactive.event(input.nav_new)
    def _reset():
        up_feat.set(None); left_feat.set(None); right_feat.set(None); down_feat.set(None)
        kzone_feats.set([]); wse_extent_feat.set(None); bnd_slot.set(None)
        dem_path.set(None); dem_meta.set(None)
        run_result.set(None); stage.set("")
        head_tifs.set([]); head_rng.set(None); _head_cache.clear(); _contour_cache.clear()
        pick_pts.set([]); reach_feat.set(None); auto_meta.set(None); last_click.set(None)
        dc = _draw_ctl.get("dc")
        if dc is not None:
            try:
                dc.data = []
            except Exception:  # noqa: BLE001
                pass
        for k in list(_layers):
            _set_layer(k, None)
        current_step.set(STEP_REACH)

    @reactive.effect
    @reactive.event(input.nav_help)
    def _help():
        ui.modal_show(ui.modal(
            ui.markdown(
                "**How to use**\n\n"
                "1. **Reach** — **Auto** (default): click the **upstream** then **downstream** "
                "point on a blue NHD stream to trace the reach (≤ 1 mile). Or **Manual**: draw the "
                "reach centerline (double-click the line to edit it) and enter the drainage area.\n"
                "2. **DEM** — pick a 3DEP resolution and **Fetch terrain** over the reach.\n"
                "3. **Boundaries** — **Generate boundaries** builds the four sides (Upstream, Left "
                "FPL, Right FPL, Downstream — floodplain = X × bankfull depth) + the wetted extent, "
                "which close into the domain. Click a boundary to draw/edit it (double-click the line "
                "to edit); set the boundary-condition gradients here.\n"
                "4. **K** — horizontal/vertical conductivity & porosity; optionally draw K-zone "
                "polygons.\n"
                "5. **Mesh** — cell size, model depth & layer thickness (a live estimate keeps the "
                "grid in bounds), then **Run model**.\n"
                "6. **Run** → **Results**: pathlines + heads draw on the map; download the bundle.\n\n"
                "The water-surface extent becomes the constant-head (CHD) top boundary, using the "
                "DEM elevations inside it. Results live in temporary storage — **download before you leave**."),
            title="Help", easy_close=True))

    # ---- downloads ----
    @render.download(filename="hyporheic_results.zip")
    def dl_zip():
        if run_result():
            yield bundle.zip_dir(work_dir)

    # ---- left pane (state machine) ----
    @render.ui
    def leftpane():
        step = current_step()
        if step == STEP_REACH:
            body = ui.TagList(
                ui.input_text("address", "Address, place, or stream",
                              placeholder="e.g. Atlanta, GA  ·  Utoy Creek"),
                ui.div(ui.input_action_button("find_address", "Find on map",
                                              class_="btn-sm btn-outline-secondary"),
                       class_="hype-actions"),
                ui.div("Type to search — suggestions from OpenStreetMap / Photon.",
                       class_="hype-ac-credit"),
                ui.input_radio_buttons(
                    "delineate_mode", "Define the reach",
                    {"auto": "Auto — pick 2 points on a stream",
                     "manual": "Manual — draw the centerline"}, selected=delineate_mode()),
                ui.panel_conditional(
                    "input.delineate_mode === 'auto'",
                    ui.div("Zoom to your stream and pick the upstream + downstream points.",
                           class_="hype-instr"),
                    ui.output_ui("nhd_status_ui"),
                    ui.output_ui("auto_readout"),
                    ui.div(ui.input_action_button("clear_points", "Clear",
                                                  class_="btn-sm btn-outline-secondary"),
                           class_="hype-actions")),
                ui.panel_conditional(
                    "input.delineate_mode === 'manual'",
                    ui.div("Draw the reach centerline, then enter its drainage area. "
                           "Double-click the line to edit it.", class_="hype-instr"),
                    ui.input_numeric("manual_da", "Drainage area (km²)", value=1.0, min=0.01,
                                     step=0.5),
                    ui.output_ui("manual_reach_status"),
                    ui.div(ui.input_action_button("clear_draw", "Clear",
                                                  class_="btn-sm btn-outline-secondary"),
                           class_="hype-actions")),
                ui.div(ui.input_action_button("next_reach", "Continue → DEM", class_="btn-primary"),
                       class_="hype-actions"),
            )
        elif step == STEP_DEM:
            body = ui.TagList(
                ui.div("Pick a resolution, then fetch 3DEP terrain over the reach.",
                       class_="hype-instr"),
                ui.input_select("dem_res", "DEM resolution",
                                {"auto": "Auto — finest (1 m where available)", "1": "1 m",
                                 "3": "3 m", "5": "5 m", "10": "10 m"}, selected="auto"),
                ui.div(ui.input_action_button("fetch_dem", "Fetch terrain", class_="btn-primary"),
                       class_="hype-actions"),
                ui.output_ui("busy"),
                ui.output_ui("dem_status"),
                ui.div(ui.input_action_button("next_dem", "Continue → Boundaries",
                                              class_="btn-primary"), class_="hype-actions"),
            )
        elif step == STEP_BOUNDARIES:
            body = ui.TagList(
                ui.div("Generate the four boundaries from the reach, then click a boundary on the map "
                       "to edit it — drag vertices, click midpoints to add.", class_="hype-instr"),
                ui.input_select("fp_mult", "Floodplain extent = X × bankfull depth",
                                {"2": "2×", "5": "5×", "10": "10× (default)"}, selected="10"),
                ui.div(ui.input_action_button("regen", "Generate boundaries", class_="btn-primary"),
                       class_="hype-actions"),
                ui.output_ui("draw_status"),
                ui.output_ui("domain_warning"),
                ui.input_radio_buttons(
                    "wse_mode", "Water surface (top boundary)",
                    {"draw": "Wetted extent (auto / drawn)", "upload": "Upload a WSE raster"},
                    selected="draw"),
                ui.panel_conditional(
                    "input.wse_mode === 'upload'",
                    ui.input_file("wse_upload", "WSE GeoTIFF", accept=[".tif", ".tiff"],
                                  multiple=False)),
                ui.input_select("bc_mode", "Boundary condition",
                                {BC_CORNER: "4 corner gradients", BC_PROFILE: "Spatially varying"},
                                selected=BC_CORNER),
                ui.panel_conditional(
                    f"input.bc_mode === '{BC_CORNER}'",
                    ui.input_numeric("g_ul", "Upstream-left gradient", value=0.005, step=0.001),
                    ui.input_numeric("g_ur", "Upstream-right gradient", value=0.005, step=0.001),
                    ui.input_numeric("g_dl", "Downstream-left gradient", value=0.005, step=0.001),
                    ui.input_numeric("g_dr", "Downstream-right gradient", value=0.005, step=0.001)),
                ui.panel_conditional(
                    f"input.bc_mode === '{BC_PROFILE}'",
                    ui.input_text("g_left_profile", "Left profile", value="0,0.005 0.5,0.005 1,0.005"),
                    ui.input_text("g_right_profile", "Right profile", value="0,0.005 0.5,0.005 1,0.005"),
                    ui.div("Format: 'fraction,gradient …' along each boundary (must include 0 and 1).",
                           class_="hype-instr")),
                ui.div(ui.input_action_button("next_boundaries", "Continue → K", class_="btn-primary"),
                       class_="hype-actions"),
            )
        elif step == STEP_K:
            body = ui.TagList(
                ui.div("Hydraulic conductivity. Optionally draw K-zone polygons.",
                       class_="hype-instr"),
                ui.input_numeric("kh", "Horizontal K (m/d)", value=10.0, min=0.0001, step=1.0),
                ui.input_numeric("kv", "Vertical K (m/d)", value=1.0, min=0.0001, step=0.5),
                ui.input_numeric("porosity", "Porosity", value=0.3, min=0.01, max=0.6, step=0.05),
                ui.input_checkbox("use_kzones", "Use hydraulic-conductivity zones", value=False),
                ui.panel_conditional(
                    "input.use_kzones === true",
                    ui.div("Add one or more K-zone polygons (each uses these values); "
                           "double-click a zone to edit it.", class_="hype-instr"),
                    ui.input_numeric("kzone_kh", "Zone KH (m/d)", value=50.0, min=0.0001, step=1.0),
                    ui.input_numeric("kzone_kv", "Zone KV (m/d)", value=5.0, min=0.0001, step=0.5),
                    ui.div(
                        ui.input_action_button("kz_add", "Add K-zone", class_="btn-sm btn-primary"),
                        ui.input_action_button("kz_rmlast", "Remove last",
                                               class_="btn-sm btn-outline-secondary"),
                        ui.input_action_button("kz_clear", "Clear all",
                                               class_="btn-sm btn-outline-secondary"),
                        class_="hype-bnd-row"),
                    ui.output_ui("kzone_status")),
                ui.div(ui.input_action_button("next_k", "Continue → Mesh", class_="btn-primary"),
                       class_="hype-actions"),
            )
        elif step == STEP_MESH:
            body = ui.TagList(
                ui.div("Model grid — the live estimate below keeps the run in bounds.",
                       class_="hype-instr"),
                ui.input_numeric("cell_size", "Cell size (m) — smaller = finer grid", value=10.0,
                                 min=1.0, step=1.0),
                ui.input_numeric("gw_mod_depth", "Model depth below water surface (m)", value=6.0,
                                 min=1.0, step=0.5),
                ui.input_numeric("z", "Layer thickness (m) — depth ÷ thickness = layers", value=0.25,
                                 min=0.05, step=0.05),
                ui.output_ui("estimate_box"),
                ui.div(ui.input_action_button("compute_mesh", "Compute mesh (3D preview)",
                                              class_="btn-sm btn-outline-secondary"),
                       class_="hype-actions"),
                ui.output_ui("mesh_status"),
                ui.div(ui.input_action_button("run_model", "Run model", class_="btn-primary"),
                       class_="hype-actions"),
            )
        elif step == STEP_RUN:
            body = ui.TagList(
                ui.output_ui("run_status"),
                ui.tags.pre(ui.output_text("run_log"), class_="hype-log"),
                ui.div(ui.input_action_button("cancel_run", "Cancel run",
                                              class_="btn-sm btn-outline-danger"),
                       class_="hype-actions"),
            )
        else:
            tifs = head_tifs()
            head_ctrls = []
            if tifs:
                with reactive.isolate():     # persisted slider state, re-read on each pane re-run
                    _ly = max(1, min(int(head_layer_v()), len(tifs)))
                    _op = float(head_opacity_v())
                head_ctrls = [
                    ui.input_slider("head_layer", "Head layer (1 = top)", min=1, max=len(tifs),
                                    value=_ly, step=1),
                    ui.input_slider("head_opacity", "Head opacity", min=0.0, max=1.0,
                                    value=_op, step=0.05),
                    ui.output_ui("head_legend"),
                ]
            body = ui.TagList(
                ui.div("Run complete — adjust the sliders and toggle layers (top-right).",
                       class_="hype-instr"),
                *head_ctrls,
                ui.output_ui("result_summary"),
                ui.div("Results are in temporary storage — download before you leave.",
                       class_="hype-warn"),
                ui.div(ui.download_button("dl_zip", "Download results (.zip)", class_="btn-primary"),
                       class_="hype-actions"),
            )
        return ui.TagList(
            ui.div(f"HYPE — {dict(STEP_LABELS).get(step)}", class_="hype-pane-head"),
            ui.div(ui.output_ui("stepper_ui"), body, class_="hype-pane-body"),
        )

    @render.ui
    def stepper_ui():
        # Kept separate from leftpane() so step-reachability changes (which depend on the drawn
        # features) re-render only the stepper — not the whole pane. Otherwise re-rendering the
        # pane would reset inputs like fp_mult back to their hard-coded defaults.
        return _stepper(current_step(), _reachable())

    @render.ui
    def nhd_status_ui():
        s = nhd_status()
        return ui.div(s, class_="hype-instr") if s else None

    @render.ui
    def auto_readout():
        n = len(pick_pts()); m = auto_meta()
        rows = [ui.div(("✓ " if n >= 1 else "➤ ") + "Upstream point",
                       class_="hype-chk ok" if n >= 1 else "hype-chk"),
                ui.div(("✓ " if n >= 2 else ("➤ " if n == 1 else "○ ")) + "Downstream point",
                       class_="hype-chk ok" if n >= 2 else "hype-chk")]
        if m:
            rows.append(ui.div(
                f"Reach {m['length_m'] / 1609.344:.2f} mi · drainage area {m['da_sqkm']:.1f} km² · "
                f"bankfull depth {m['depth_m']:.2f} m ({m['division_name']})",
                class_="hype-estimate green"))
        return ui.div(*rows)

    @render.ui
    def manual_reach_status():
        ok = delineate_mode() == "manual" and reach_feat() is not None
        return ui.div(("✓ Reach centerline drawn" if ok else "○ Draw the reach centerline"),
                      class_="hype-chk ok" if ok else "hype-chk")

    @render.ui
    def draw_status():
        # Compact color legend — the boundaries are edited by clicking their lines on the map, so
        # this is informational (swatch + name + ✓/○, active row highlighted); empty rows get a
        # small "Draw" link as the only entry point when there's no line on the map to click.
        active = bnd_slot()
        defs = [("up", "Upstream", UP_STYLE["color"], up_feat()),
                ("left", "Left FPL", LEFT_STYLE["color"], left_feat()),
                ("right", "Right FPL", RIGHT_STYLE["color"], right_feat()),
                ("down", "Downstream", DOWN_STYLE["color"], down_feat())]
        if wse_mode_v() != "upload":
            defs.append(("wse", "Water surface", WSE_STYLE["color"], wse_extent_feat()))
        rows = []
        for slot, label, color, feat in defs:
            present = feat is not None
            inner = [ui.span(class_="hype-leg-swatch", style=f"background:{color};"),
                     ui.span(label, class_="hype-leg-name"),
                     ui.span("✓" if present else "○",
                             class_="hype-leg-mark ok" if present else "hype-leg-mark")]
            if not present and active is None:
                inner.append(ui.input_action_link(f"bnd_draw_{slot}", "Draw", class_="hype-leg-draw"))
            rows.append(ui.div(*inner, class_="hype-leg-row" + (" active" if slot == active else "")))
        if active:
            hint = "Editing on the map — drag vertices, or use the bar to Clear & redraw / Done."
        elif _domain_build() is not None:
            hint = "Click a boundary line on the map to edit it."
        elif any(f is not None for *_, f in defs):
            hint = "Click a boundary on the map to edit, or Generate boundaries."
        else:
            hint = "Click Generate boundaries to build the four sides."
        return ui.div(ui.div(hint, class_="hype-instr"), ui.div(*rows, class_="hype-legend"))

    @render.ui
    def domain_warning():
        # Warn when the four boundaries don't meet at a corner. The derived domain still force-closes
        # for the model run, but a big gap means the user's lines are disconnected — guide them to fix
        # it. (Snapping auto-connects near endpoints; this catches the ones too far apart to snap.)
        if not _HAS_MAP or current_step() != STEP_BOUNDARIES:
            return None
        gap = geometry.corner_gaps_m(up_feat(), left_feat(), right_feat(), down_feat())
        if gap is None or gap <= 25.0:
            return None
        return ui.div(
            ui.div(f"⚠ Boundaries don't meet at a corner (gap ≈ {gap:.0f} m). Drag an endpoint onto the "
                   "neighbouring line to connect them, or:"),
            ui.input_action_button("snap_corners", "Snap corners together", class_="hype-warn-btn"),
            class_="hype-warn")

    @render.ui
    def kzone_status():
        kn = len(kzone_feats())
        if kz_adding():
            msg = "Drawing a K-zone — click on the map to place vertices."
        elif kn:
            msg = f"✓ {kn} K-zone polygon(s) — double-click one to edit."
        else:
            msg = "No K-zones yet — click Add K-zone."
        return ui.p(msg, class_="hype-chk ok" if kn else "hype-chk")

    @render.ui
    def dem_status():
        if dem_path() is None:
            return None
        m = dem_meta() or {}
        res, src = m.get("resolution_m"), m.get("source", "USGS 3DEP")
        tag = f"{res} m ({src})" if res else src
        try:
            s = dem.dem_summary(dem_path())
            return ui.p(f"✓ DEM — {tag} · {s['width']}×{s['height']} px · "
                        f"elev {s['min']:.1f}–{s['max']:.1f} m", class_="hype-chk ok")
        except Exception:  # noqa: BLE001
            return ui.p(f"✓ DEM — {tag}", class_="hype-chk ok")

    @render.ui
    def estimate_box():
        est = grid_estimate()
        if not est:
            return None
        facts = (f"Domain ≈ {est['dom_w']:,.0f} × {est['dom_h']:,.0f} m · {est['nlay']} layers "
                 f"({est['ncol']}×{est['nrow']} cells/layer)")
        return ui.TagList(
            ui.div(facts, class_="hype-chk"),
            ui.div(estimate.band_message(est),
                   class_=f"hype-estimate {estimate.band(est['n_cells'])}"))

    @render.ui
    def mesh_status():
        if mesh_task.status() == "running":
            return ui.div(ui.div(class_="hype-spinner"), ui.span("Building 3D mesh…"),
                          class_="hype-busy")
        g = mesh_geom()
        if not g:
            return ui.p("Click Compute mesh to preview the grid in 3D.", class_="hype-chk")
        f = g.get("decimation", 1)
        note = "" if f == 1 else f" · shown at 1/{f} resolution"
        return ui.p(f"✓ {g.get('nActiveFull', 0):,} active cells{note} — drag to orbit, "
                    f"slider to slice.", class_="hype-chk ok")

    @render.ui
    def mesh3d_style():
        # Reveal the vtk.js viewer overlay (over the map) only on the Mesh step.
        if current_step() == STEP_MESH:
            return ui.tags.style(".hype-mesh3d{display:block;}")
        return None

    @render.ui
    def busy():
        s = stage()
        running = dem_task.status() == "running" or run_task.status() == "running"
        return ui.div(ui.div(class_="hype-spinner"), ui.span(s), class_="hype-busy") if (s and running) else None

    @render.ui
    def run_status():
        log_tick()
        if run_task.status() != "running":
            return ui.div(ui.div(class_="hype-spinner"), ui.span("Starting…"), class_="hype-busy")
        n = step_v()
        secs = int(elapsed_v()); mm, ss = secs // 60, secs % 60
        label = RUN_STEPS.get(n, RUN_STEPS[0])
        head = f"Step {n} of {RUN_TOTAL} — {label}" if n else label
        if n:
            pct = max(6, min(100, int(round(n / RUN_TOTAL * 100))))
            bar = ui.div(ui.div(class_="hype-prog-bar", style=f"width:{pct}%;"), class_="hype-prog")
        else:
            bar = ui.div(ui.div(class_="hype-prog-bar indet"), class_="hype-prog")
        return ui.div(
            ui.div(ui.div(class_="hype-spinner"), ui.span(head, class_="hype-run-label"),
                   ui.span(f"{mm}:{ss:02d}", class_="hype-elapsed"), class_="hype-run-head"),
            bar,
            class_="hype-run-status",
        )

    @render.text
    def run_log():
        log_tick()
        return "\n".join(log_lines[-200:]) or "Starting… preparing terrain and model inputs."

    @render.ui
    def result_summary():
        res = run_result()
        if not res:
            return None
        txt = results.summary_text(res, work_dir)
        m = dem_meta() or {}
        if m.get("resolution_m"):
            txt = f"{txt}\nDEM: {m['resolution_m']} m ({m.get('source', 'USGS 3DEP')})"
        return ui.tags.pre(txt, class_="hype-log")

    @render.ui
    def head_legend():
        rng = head_rng()
        if rng is None:
            return None
        try:
            uri = results.colorbar_datauri(rng[0], rng[1], cmap="viridis",
                                           label="Hydraulic head (m)")
        except Exception:  # noqa: BLE001
            return None
        return ui.img(src=uri, style="width:100%; max-width:320px; margin:2px 0 6px;")

    @render.ui
    def readout():
        if not _HAS_MAP:
            return None
        z, c = _view()
        if not c:
            return ui.div("Search or zoom to a stream to begin", class_="hype-readout")
        crs = proj_crs()
        crs_txt = f" · CRS {crs.to_epsg()}" if crs is not None and crs.to_epsg() else ""
        return ui.div(f"Zoom {int(z)} · {float(c[0]):.4f}, {float(c[1]):.4f}{crs_txt}",
                      class_="hype-readout")

    @render.ui
    def flow_loading():
        # Bottom-center cue that the clickable NHD stream vectors are being fetched — only on the
        # Reach step, zoomed in enough for them to appear, while a fetch is actually in flight.
        if not _HAS_MAP or current_step() != STEP_REACH:
            return None
        z, _c = _view()
        if z is None or int(z) < 12 or flow_task.status() != "running":
            return None
        return ui.div(ui.div(class_="hype-spinner"), ui.span("Loading streams…"),
                      class_="hype-flow-loading")

    @render.ui
    def map_edit_style():
        # On the Reach + Boundaries steps the draw tool is auto-driven (www/reach_draw.js), so hide
        # the Leaflet.draw toolbar (the control stays in the DOM — we click its anchors; its mouse
        # tooltip lives in the popup pane, so it still shows). Add a crosshair only while a pick or a
        # fresh draw is actually possible. Mirrors EASI's cursor_style pattern.
        step = current_step()
        if not _HAS_MAP or step not in (STEP_REACH, STEP_BOUNDARIES, STEP_K):
            return None
        css = ".hype-map-wrap .leaflet-draw{display:none !important;}"
        if step == STEP_REACH:
            z, _c = _view()
            no_reach = reach_feat() is None
            armed = delineate_mode() == "manual" and no_reach
            picking = (delineate_mode() == "auto" and no_reach and z is not None
                       and int(z) >= 12 and len(pick_pts()) < 2)
            crosshair = armed or picking
        elif step == STEP_BOUNDARIES:               # crosshair while drawing a fresh side
            slot = bnd_slot()
            sv = _slot_value(slot) if slot else None
            crosshair = bool(slot) and (sv is None or sv() is None)
        else:                                       # STEP_K — crosshair while adding a K-zone
            crosshair = kz_adding()
        if crosshair:
            css += (".hype-map-wrap .leaflet-grab{cursor:crosshair !important;}"
                    ".hype-map-wrap .leaflet-container.leaflet-dragging,"
                    ".hype-map-wrap .leaflet-container.leaflet-dragging .leaflet-grab"
                    "{cursor:grabbing !important;}")
        return ui.tags.style(css)

    @reactive.effect
    async def _push_reach_state():
        # Tell the client (www/reach_draw.js) how to guide the map: the follow-cursor pick tooltip
        # (Reach auto), auto-arm a fresh draw (`armShape` = line/polygon), and/or allow
        # double-click-to-edit. Covers the Reach centerline and the four Boundaries slots + WSE.
        if not _HAS_MAP:
            return
        step = current_step()
        z, _c = _view()
        picking = arm = can_edit = auto_edit = False
        arm_shape = "line"
        slot_id = None
        if step == STEP_REACH:
            mode = delineate_mode()
            no_reach = reach_feat() is None
            picking = (mode == "auto" and no_reach and z is not None and int(z) >= 12
                       and len(pick_pts()) < 2)
            arm = mode == "manual" and no_reach
            can_edit = mode == "manual" and not no_reach          # double-click to edit the centerline
            slot_id = "reach" if mode == "manual" else None
        elif step == STEP_BOUNDARIES:
            slot = bnd_slot()
            slot_id = slot
            if slot:
                sv = _slot_value(slot)
                has = sv() is not None if sv is not None else False
                arm_shape = "polygon" if slot == "wse" else "line"
                arm = not has                                     # empty slot → draw it
                auto_edit = has                                   # selected existing line → edit now
                can_edit = has                                    # + double-click fallback
        elif step == STEP_K:
            slot_id = "kzone"
            arm_shape = "polygon"                 # K-zones are polygons; Add arms a fresh draw
            arm = bool(kz_adding())
            can_edit = (not kz_adding()) and len(kzone_feats()) > 0
        await session.send_custom_message("hype_reach", {
            "step": step, "slot": slot_id, "picking": bool(picking), "arm": bool(arm),
            "canEdit": bool(can_edit), "autoEdit": bool(auto_edit), "armShape": arm_shape,
            "slotName": {"up": "Upstream", "left": "Left FPL", "right": "Right FPL",
                         "down": "Downstream", "wse": "Water surface"}.get(slot_id, ""),
        })


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")
