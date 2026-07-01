/* HYPE 3D mesh viewer (vtk.js, client-side).
 *
 * The server (app.py `_mesh_done`) sends a "hype_mesh" custom message with the decimated grid
 * geometry built in pure NumPy (hype_app/mesh.py) — de-duplicated corner points + active
 * hexahedra (8 point-indices each) + a per-cell layer scalar, all in local metres. This module
 * renders it with vtk.js: orbit/zoom (trackball), a clip-plane slider to slice through and reveal
 * interior layers (X/Y/Z axis), and a vertical-exaggeration slider (groundwater grids are thin).
 * Everything here is client-side — no server round-trip for interaction, and no server rendering.
 *
 * vtk.js is loaded as the monolithic UMD bundle (global `vtk`) from a CDN in app.py's head.
 */
(function () {
  "use strict";

  var CID = "hype-mesh3d";
  var S = { grw: null, ren: null, rw: null, mappers: [], actors: [], plane: null,
            clipping: false, axis: 0, t: 0, vexag: 1, bounds: null, bar: null, hint: null,
            scalarBar: null, omw: null };

  function container() { return document.getElementById(CID); }
  function V() { return window.vtk; }

  function axisExtent(b, axis) {           // [min,max] of the (vexag-scaled) bounds along an axis
    var lo = b[axis * 2], hi = b[axis * 2 + 1];
    if (axis === 2) { lo *= S.vexag; hi *= S.vexag; }
    return [lo, hi];
  }

  function applyClip() {
    if (!S.mappers || !S.mappers.length || !S.plane || !S.bounds) return;
    var ext = axisExtent(S.bounds, S.axis);
    var origin = [(S.bounds[0] + S.bounds[1]) / 2,
                  (S.bounds[2] + S.bounds[3]) / 2,
                  (S.bounds[4] + S.bounds[5]) / 2 * S.vexag];
    origin[S.axis] = ext[0] + S.t * (ext[1] - ext[0]);
    var normal = [0, 0, 0]; normal[S.axis] = 1;
    S.plane.setOrigin(origin);
    S.plane.setNormal(normal);
    var want = S.t > 0.001;
    if (want && !S.clipping) { S.mappers.forEach(function (m) { m.addClippingPlane(S.plane); }); S.clipping = true; }
    else if (!want && S.clipping) { S.mappers.forEach(function (m) { m.removeClippingPlane(S.plane); }); S.clipping = false; }
  }

  function applyVexag() {
    (S.actors || []).forEach(function (a) { a.setScale(1, 1, S.vexag); });
    applyClip();
  }

  function render() { if (S.rw) S.rw.render(); }

  function buildBar() {
    if (S.bar) return;
    var bar = document.createElement("div");
    bar.className = "hype-mesh3d-bar";
    bar.innerHTML =
      '<label>Slice <select data-k="axis"><option value="0">X</option>' +
      '<option value="1">Y</option><option value="2">Z</option></select></label>' +
      '<label><input type="range" data-k="clip" min="0" max="1" step="0.01" value="0"></label>' +
      '<label>Vert × <input type="range" data-k="vexag" min="1" max="5" step="1" value="1">' +
      '<span data-k="vexagval">1</span></label>' +
      '<button data-k="reset">Reset view</button>';
    bar.addEventListener("input", function (e) {
      var k = e.target.getAttribute("data-k");
      if (k === "axis") { S.axis = parseInt(e.target.value, 10); S.t = 0;
                          bar.querySelector('[data-k="clip"]').value = 0; applyClip(); }
      else if (k === "clip") { S.t = parseFloat(e.target.value); applyClip(); }
      else if (k === "vexag") { S.vexag = parseFloat(e.target.value);
                                bar.querySelector('[data-k="vexagval"]').textContent = S.vexag;
                                applyVexag(); if (S.ren) S.ren.resetCameraClippingRange(); }
      render();
    });
    bar.addEventListener("click", function (e) {
      if (e.target.getAttribute("data-k") !== "reset") return;
      S.axis = 0; S.t = 0; S.vexag = 1;                    // reset slice + vertical-exaggeration state
      var q = function (k) { return bar.querySelector('[data-k="' + k + '"]'); };
      if (q("axis")) q("axis").value = "0";
      if (q("clip")) q("clip").value = 0;
      if (q("vexag")) q("vexag").value = 1;
      if (q("vexagval")) q("vexagval").textContent = "1";
      applyVexag();                                        // re-scale z→1 + applyClip() (t=0 drops the plane)
      if (S.ren) { S.ren.resetCamera(); S.ren.resetCameraClippingRange(); }
      render();
    });
    container().appendChild(bar);
    S.bar = bar;
  }

  function showHint(text) {
    var el = container();
    if (!el) return;
    if (!S.hint) {
      S.hint = document.createElement("div");
      S.hint.className = "hype-mesh3d-hint";
      el.appendChild(S.hint);
    }
    S.hint.textContent = text || "";
    S.hint.style.display = text ? "block" : "none";
  }

  // Guidance in the (otherwise blank dark) overlay on the Mesh step before anything is computed.
  function idleHint() {
    if (container() && !S.grw) {
      showHint('Set the grid above, then click "Compute mesh" to build the 3D preview.');
    }
  }

  function initOnce() {
    if (S.grw) return true;
    var vtk = V(), el = container();
    if (!vtk || !el) return false;
    S.grw = vtk.Rendering.Misc.vtkGenericRenderWindow.newInstance({ background: [0.05, 0.07, 0.09] });
    S.grw.setContainer(el);
    S.ren = S.grw.getRenderer();
    S.rw = S.grw.getRenderWindow();
    buildBar();
    try {
      var ro = new ResizeObserver(function () { try { S.grw.resize(); render(); } catch (e) { /**/ } });
      ro.observe(el);
    } catch (e) { window.addEventListener("resize", function () { try { S.grw.resize(); } catch (_) {} }); }
    return true;
  }

  function buildScene(msg) {
    if (!initOnce()) { showHint("3D viewer failed to load."); return; }
    var vtk = V();
    showHint("");
    (S.actors || []).forEach(function (a) { S.ren.removeActor(a); });   // clear any prior mesh
    S.actors = []; S.mappers = []; S.clipping = false;

    // This vtk.js build ships vtkPolyData (not vtkUnstructuredGrid), so expand each active hex
    // (8 corner ids: 0-3 = bottom face, 4-7 = top face) to its 6 quad faces — clips cleanly on a cut.
    // Split into two meshes: the TOP layer (terrain-coloured by elevation) and the deeper BODY (a
    // neutral-gray block), so the ground surface shows topography while the block below stays quiet.
    var src = msg.cells, elevArr = msg.cellElev, layArr = msg.cellLayer, nHex = msg.nHex;
    var FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
    var topPolys = [], topElev = [], bodyPolys = [];
    for (var h = 0; h < nHex; h++) {
      var base = h * 8, ev = elevArr[h], isTop = layArr[h] === 0;
      for (var f = 0; f < 6; f++) {
        var fc = FACES[f];
        var q = [4, src[base + fc[0]], src[base + fc[1]], src[base + fc[2]], src[base + fc[3]]];
        if (isTop) { topPolys.push(q[0], q[1], q[2], q[3], q[4]); topElev.push(ev); }
        else { bodyPolys.push(q[0], q[1], q[2], q[3], q[4]); }
      }
    }
    var ptsData = Float32Array.from(msg.points);

    // Terrain colour ramp over the elevation range (full resolution; applied to the top mesh only).
    var rng = msg.elevRange || [0, 1], elo = rng[0], ehi = rng[1], ed = (ehi - elo) || 1;
    var ctf = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
    ctf.addRGBPoint(elo, 0.27, 0.45, 0.29);                 // low  — green
    ctf.addRGBPoint(elo + 0.40 * ed, 0.55, 0.60, 0.32);     //      — yellow-green
    ctf.addRGBPoint(elo + 0.65 * ed, 0.80, 0.74, 0.46);     //      — tan
    ctf.addRGBPoint(elo + 0.85 * ed, 0.58, 0.44, 0.32);     //      — brown
    ctf.addRGBPoint(ehi, 0.95, 0.95, 0.92);                 // high — near-white

    function addMesh(polysArr, scalars, rgb) {
      if (!polysArr.length) return;
      var pd = vtk.Common.DataModel.vtkPolyData.newInstance();
      pd.getPoints().setData(ptsData, 3);
      pd.getPolys().setData(Uint32Array.from(polysArr));
      var mapper = vtk.Rendering.Core.vtkMapper.newInstance();
      mapper.setInputData(pd);
      if (scalars) {
        pd.getCellData().setScalars(vtk.Common.Core.vtkDataArray.newInstance(
          { name: "elev", values: Float32Array.from(scalars), numberOfComponents: 1 }));
        mapper.setScalarVisibility(true);
        if (mapper.setScalarModeToUseCellData) mapper.setScalarModeToUseCellData();
        mapper.setLookupTable(ctf);
        mapper.setScalarRange(elo, ehi);
      } else {
        mapper.setScalarVisibility(false);
      }
      var actor = vtk.Rendering.Core.vtkActor.newInstance();
      actor.setMapper(mapper);
      var prop = actor.getProperty();
      if (rgb) prop.setColor(rgb[0], rgb[1], rgb[2]);
      if (prop.setEdgeVisibility) { prop.setEdgeVisibility(true); prop.setEdgeColor(0.16, 0.18, 0.22); }
      S.ren.addActor(actor);
      S.actors.push(actor); S.mappers.push(mapper);
    }
    addMesh(bodyPolys, null, [0.56, 0.58, 0.61]);           // neutral-gray body
    addMesh(topPolys, topElev, null);                       // terrain-coloured top surface

    // Elevation legend (scalar bar) — created once; its colour map tracks the current mesh.
    if (!S.scalarBar && vtk.Rendering.Core.vtkScalarBarActor) {
      S.scalarBar = vtk.Rendering.Core.vtkScalarBarActor.newInstance();
      if (S.scalarBar.setAxisLabel) S.scalarBar.setAxisLabel("Elevation (m)");
      S.ren.addActor(S.scalarBar);
    }
    if (S.scalarBar && S.scalarBar.setScalarsToColors) S.scalarBar.setScalarsToColors(ctf);

    // X/Y/Z orientation gizmo (bottom-right), created once.
    if (!S.omw && vtk.Interaction.Widgets.vtkOrientationMarkerWidget && S.rw.getInteractor) {
      var axes = vtk.Rendering.Core.vtkAxesActor.newInstance();
      S.omw = vtk.Interaction.Widgets.vtkOrientationMarkerWidget.newInstance(
        { actor: axes, interactor: S.rw.getInteractor() });
      S.omw.setEnabled(true);
      var Corners = vtk.Interaction.Widgets.vtkOrientationMarkerWidget.Corners;
      S.omw.setViewportCorner(Corners ? Corners.BOTTOM_RIGHT : 1);
      S.omw.setViewportSize(0.15);
      if (S.omw.setMinPixelSize) S.omw.setMinPixelSize(80);
    }

    S.bounds = msg.bounds;
    S.plane = vtk.Common.DataModel.vtkPlane.newInstance({ normal: [1, 0, 0], origin: [0, 0, 0] });
    S.t = 0; if (S.bar) S.bar.querySelector('[data-k="clip"]').value = 0;
    applyVexag();
    try { S.grw.resize(); } catch (e) { /**/ }
    S.ren.resetCamera();
    render();
  }

  // vtk.js is the monolithic UMD at the package root (modern builds dropped dist/vtk.js). Load it on
  // demand with the page's AMD loader (jupyter-widgets' RequireJS) temporarily disabled, so the UMD
  // exports to window.vtk instead of registering as an anonymous AMD module.
  var VTK_URL = "https://cdn.jsdelivr.net/npm/vtk.js@36.2.1/vtk.js";
  function loadVtk(cb) {
    if (window.vtk) { cb(); return; }
    S.loadCbs = S.loadCbs || [];
    S.loadCbs.push(cb);
    if (S.loading) return;
    S.loading = true;
    showHint("Loading 3D viewer…");
    var savedDefine = window.define;
    try { window.define = undefined; } catch (e) { /**/ }
    var s = document.createElement("script");
    s.src = VTK_URL;
    s.onload = function () {
      window.define = savedDefine; S.loading = false;
      var cbs = S.loadCbs; S.loadCbs = [];
      cbs.forEach(function (f) { try { f(); } catch (e) { console.error(e); } });
    };
    s.onerror = function () {
      window.define = savedDefine; S.loading = false; S.loadCbs = [];
      showHint("Could not load the 3D library (vtk.js) from the CDN.");
    };
    document.head.appendChild(s);
  }

  function onMessage(msg) {
    loadVtk(function () {
      try { buildScene(msg); }
      catch (e) { console.error("[mesh3d] build failed", e); showHint("3D render error — see console."); }
    });
  }

  function register() {
    if (window.Shiny && Shiny.addCustomMessageHandler) {
      Shiny.addCustomMessageHandler("hype_mesh", onMessage);
      return true;
    }
    return false;
  }
  if (!register()) document.addEventListener("shiny:connected", register);
  document.addEventListener("shiny:connected", idleHint);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", idleHint);
  else idleHint();
})();
