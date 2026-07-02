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
  window.__hypeMesh3d = null;                              // debug/QA handle (set to S below)
  var S = { grw: null, ren: null, rw: null, mappers: [], actors: [], plane: null,
            clipping: false, axis: 0, t: 0, vexag: 1, bounds: null, bar: null, hint: null,
            scalarBar: null, omw: null,
            drapeActor: null, drapeOpacity: 0.55,          // basemap drape (aerial on the top faces)
            labelEls: [], labelAnchors: [], labelPts: null,   // floating boundary labels
            ctf: null, topMapper: null,                       // elevation coloring (re-rangeable)
            topCellPts: null, topCellElev: null };            // top-cell centers, for range-from-view
  window.__hypeMesh3d = S;

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
    // Label anchors are projected from DATASET coords (the pixel-space mapper ignores actor
    // transforms), so bake the exaggeration into the anchor points themselves.
    if (S.labelPts && S.labelAnchors.length) {
      var arr = new Float32Array(S.labelAnchors.length * 3);
      S.labelAnchors.forEach(function (p, i) {
        arr[i * 3] = p[0]; arr[i * 3 + 1] = p[1]; arr[i * 3 + 2] = p[2] * S.vexag;
      });
      S.labelPts.setData(arr, 3);
    }
    applyClip();
  }

  function render() { if (S.rw) S.rw.render(); }

  // Camera bindings: the stock trackball style keeps left-drag = orbit and wheel = zoom;
  // MIDDLE- and RIGHT-drag panning is implemented here directly (translate the camera and
  // focal point along the view plane by the pixel delta). Handlers run in the CAPTURE phase
  // and stop propagation, because the interactor rotates on ANY move whose `buttons` bit is
  // set — it must never see these drags. (vtkInteractorStyleManipulator would be the tidy
  // way, but in this UMD build it silently swallows all interaction.)
  function setupInteraction(vtk, el) {
    var drag = null;                                     // {x, y} of the last pan position

    function panBy(dx, dy) {
      if (!S.ren || !S.rw) return;
      var cam = S.ren.getActiveCamera();
      var canvas = el.querySelector("canvas");
      var hPx = (canvas && canvas.clientHeight) || el.clientHeight || 700;
      // world units per screen pixel at the focal distance (perspective camera)
      var upp = 2 * cam.getDistance() *
                Math.tan((cam.getViewAngle() * Math.PI / 180) / 2) / hPx;
      var dop = cam.getDirectionOfProjection();
      var up = cam.getViewUp();
      var right = [dop[1] * up[2] - dop[2] * up[1],      // dop × up
                   dop[2] * up[0] - dop[0] * up[2],
                   dop[0] * up[1] - dop[1] * up[0]];
      var n = Math.hypot(right[0], right[1], right[2]) || 1;
      var mx = -dx * upp, my = dy * upp;
      var move = [right[0] / n * mx + up[0] * my,
                  right[1] / n * mx + up[1] * my,
                  right[2] / n * mx + up[2] * my];
      var fp = cam.getFocalPoint(), pos = cam.getPosition();
      cam.setFocalPoint(fp[0] + move[0], fp[1] + move[1], fp[2] + move[2]);
      cam.setPosition(pos[0] + move[0], pos[1] + move[1], pos[2] + move[2]);
      S.ren.resetCameraClippingRange();
      render();
    }

    el.addEventListener("pointerdown", function (e) {
      if (e.button !== 1 && e.button !== 2) return;      // middle / right only
      if (e.target.closest && e.target.closest(".hype-mesh3d-bar")) return;
      drag = { x: e.clientX, y: e.clientY };
      try { el.setPointerCapture(e.pointerId); } catch (err) { /**/ }
      e.stopPropagation(); e.preventDefault();
    }, true);
    el.addEventListener("pointermove", function (e) {
      if (!drag) return;
      panBy(e.clientX - drag.x, e.clientY - drag.y);
      drag = { x: e.clientX, y: e.clientY };
      e.stopPropagation(); e.preventDefault();
    }, true);
    function endPan(e) {
      if (!drag) return;
      drag = null;
      try { el.releasePointerCapture(e.pointerId); } catch (err) { /**/ }
      e.stopPropagation();
    }
    el.addEventListener("pointerup", endPan, true);
    el.addEventListener("pointercancel", endPan, true);
    // right-drag must not pop the browser menu over the 3D view
    el.addEventListener("contextmenu", function (e) { e.preventDefault(); });
  }

  function applyDrapeOpacity() {
    if (!S.drapeActor) return;
    var v = S.drapeOpacity;
    S.drapeActor.setVisibility(v > 0.01);
    S.drapeActor.getProperty().setOpacity(v);
  }

  // Terrain color ramp over [lo, hi] (values outside clamp to the end colors).
  function rampCtf(ctf, lo, hi) {
    var d = (hi - lo) || 1;
    ctf.removeAllPoints();
    ctf.addRGBPoint(lo, 0.27, 0.45, 0.29);                 // low  — green
    ctf.addRGBPoint(lo + 0.40 * d, 0.55, 0.60, 0.32);      //      — yellow-green
    ctf.addRGBPoint(lo + 0.65 * d, 0.80, 0.74, 0.46);      //      — tan
    ctf.addRGBPoint(lo + 0.85 * d, 0.58, 0.44, 0.32);      //      — brown
    ctf.addRGBPoint(hi, 0.95, 0.95, 0.92);                 // high — near-white
  }

  function setElevInputs(lo, hi) {
    if (!S.bar) return;
    var mn = S.bar.querySelector('[data-k="emin"]');
    var mx = S.bar.querySelector('[data-k="emax"]');
    if (mn) mn.value = Math.round(lo * 10) / 10;
    if (mx) mx.value = Math.round(hi * 10) / 10;
  }

  // Re-range the elevation legend: recolor the top surface + scalar bar over [lo, hi].
  function applyElevRange(lo, hi) {
    if (!S.ctf || !S.topMapper) return;
    if (!(hi > lo)) hi = lo + 0.1;
    rampCtf(S.ctf, lo, hi);
    S.topMapper.setScalarRange(lo, hi);
    if (S.scalarBar && S.scalarBar.setScalarsToColors) S.scalarBar.setScalarsToColors(S.ctf);
    setElevInputs(lo, hi);
    render();
  }

  // Elevation range of the top-layer cells currently VISIBLE: project each top-cell center
  // to normalized display coords (vertical exaggeration + active slice plane respected;
  // occlusion ignored) and range over those inside the viewport. Null when none land
  // on screen — e.g. hovering over an inactive part of the grid.
  function visibleElevRange() {
    if (!S.topCellPts || !S.ren || typeof S.ren.worldToNormalizedDisplay !== "function") {
      return null;
    }
    var lo = Infinity, hi = -Infinity;
    var n = S.topCellElev.length;
    for (var i = 0; i < n; i++) {
      var x = S.topCellPts[i * 3], y = S.topCellPts[i * 3 + 1];
      var z = S.topCellPts[i * 3 + 2] * S.vexag;
      if (S.clipping && S.plane && S.plane.evaluateFunction([x, y, z]) < 0) continue;
      var d = S.ren.worldToNormalizedDisplay(x, y, z);
      if (!d || d[0] < 0 || d[0] > 1 || d[1] < 0 || d[1] > 1 || d[2] < 0 || d[2] > 1) continue;
      var e = S.topCellElev[i];
      if (e < lo) lo = e;
      if (e > hi) hi = e;
    }
    return lo <= hi ? [lo, hi] : null;
  }

  function clearLabels() {
    (S.labelEls || []).forEach(function (el) { if (el.parentNode) el.parentNode.removeChild(el); });
    S.labelEls = []; S.labelAnchors = []; S.labelPts = null;
  }

  // Floating name chips anchored to each boundary line's midpoint, projected to screen space
  // every render by vtkPixelSpaceCallbackMapper (dataset coords; vexag baked in by applyVexag).
  function buildLabels(vtk, boundaries) {
    clearLabels();
    if (!boundaries.length || !vtk.Rendering.Core.vtkPixelSpaceCallbackMapper) return;
    var el = container();
    var pts = [];
    boundaries.forEach(function (b) {
      var n = b.points.length / 3, m = Math.floor(n / 2) * 3;
      S.labelAnchors.push([b.points[m], b.points[m + 1], b.points[m + 2] + 1.5]);
      pts.push(b.points[m], b.points[m + 1], b.points[m + 2] + 1.5);
      var chip = document.createElement("div");
      chip.className = "hype-mesh3d-label";
      chip.textContent = b.name;
      chip.style.color = b.color;
      chip.style.borderColor = b.color;
      el.appendChild(chip);
      S.labelEls.push(chip);
    });
    var pd = vtk.Common.DataModel.vtkPolyData.newInstance();
    pd.getPoints().setData(Float32Array.from(pts), 3);
    S.labelPts = pd.getPoints();
    var mapper = vtk.Rendering.Core.vtkPixelSpaceCallbackMapper.newInstance();
    mapper.setInputData(pd);
    mapper.setCallback(function (coords) {
      if (!coords || !S.labelEls.length) return;
      var canvas = el.querySelector("canvas");
      if (!canvas) return;
      var sx = canvas.clientWidth / (canvas.width || 1);
      var sy = canvas.clientHeight / (canvas.height || 1);
      for (var i = 0; i < S.labelEls.length && i < coords.length; i++) {
        var c = coords[i];
        S.labelEls[i].style.left = (c[0] * sx) + "px";
        S.labelEls[i].style.top = (canvas.clientHeight - c[1] * sy) + "px";
      }
    });
    var actor = vtk.Rendering.Core.vtkActor.newInstance();
    actor.setMapper(mapper);
    S.ren.addActor(actor);
    S.actors.push(actor);                    // removed with the mesh on rebuild (scale is a no-op)
  }

  // Colored polylines riding the top of each boundary's cells (data z pre-lifted server-side).
  function buildBoundaryLines(vtk, boundaries) {
    boundaries.forEach(function (b) {
      var n = b.points.length / 3;
      if (n < 2) return;
      var pd = vtk.Common.DataModel.vtkPolyData.newInstance();
      pd.getPoints().setData(Float32Array.from(b.points), 3);
      var line = new Uint32Array(n + 1);
      line[0] = n;
      for (var i = 0; i < n; i++) line[i + 1] = i;
      pd.getLines().setData(line);
      var mapper = vtk.Rendering.Core.vtkMapper.newInstance();
      mapper.setInputData(pd);
      var actor = vtk.Rendering.Core.vtkActor.newInstance();
      actor.setMapper(mapper);
      var rgb = [parseInt(b.color.slice(1, 3), 16) / 255,
                 parseInt(b.color.slice(3, 5), 16) / 255,
                 parseInt(b.color.slice(5, 7), 16) / 255];
      actor.getProperty().setColor(rgb[0], rgb[1], rgb[2]);
      actor.getProperty().setLineWidth(4);
      if (actor.getProperty().setLighting) actor.getProperty().setLighting(false);
      S.ren.addActor(actor);
      S.actors.push(actor); S.mappers.push(mapper);
    });
  }

  // Drape the aerial basemap onto the TOP faces: same face quads, points lifted slightly and
  // given texture coordinates spanning the basemap's local extent. vtk.js uploads DOM images
  // with WebGL's Y-flip, so v runs south→north ((y-y0)/Ly).
  function buildDrape(vtk, msg, topPolys, ptsData) {
    S.drapeActor = null;
    var bm = msg.basemap;
    if (!bm || !topPolys.length) return;
    var lift = 0.25;
    var remap = {}, pts2 = [], tc = [], polys2 = [];
    var lx = (bm.x1 - bm.x0) || 1, ly = (bm.y1 - bm.y0) || 1;
    for (var i = 0; i < topPolys.length; i += 5) {
      polys2.push(4);
      for (var j = 1; j <= 4; j++) {
        var pid = topPolys[i + j];
        var nid = remap[pid];
        if (nid === undefined) {
          nid = pts2.length / 3;
          remap[pid] = nid;
          var x = ptsData[pid * 3], y = ptsData[pid * 3 + 1], z = ptsData[pid * 3 + 2];
          pts2.push(x, y, z + lift);
          tc.push((x - bm.x0) / lx, (y - bm.y0) / ly);
        }
        polys2.push(nid);
      }
    }
    var pd = vtk.Common.DataModel.vtkPolyData.newInstance();
    pd.getPoints().setData(Float32Array.from(pts2), 3);
    pd.getPolys().setData(Uint32Array.from(polys2));
    pd.getPointData().setTCoords(vtk.Common.Core.vtkDataArray.newInstance(
      { name: "tc", values: Float32Array.from(tc), numberOfComponents: 2 }));
    var mapper = vtk.Rendering.Core.vtkMapper.newInstance();
    mapper.setInputData(pd);
    mapper.setScalarVisibility(false);
    var actor = vtk.Rendering.Core.vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(1, 1, 1);
    if (actor.getProperty().setLighting) actor.getProperty().setLighting(false);
    var texture = vtk.Rendering.Core.vtkTexture.newInstance();
    texture.setInterpolate(true);
    var img = new Image();
    img.onload = function () {
      try { texture.setImage(img); render(); }
      catch (e) { console.error("[mesh3d] drape texture failed", e); }
    };
    img.src = bm.url;
    actor.addTexture(texture);
    S.ren.addActor(actor);
    S.actors.push(actor); S.mappers.push(mapper);
    S.drapeActor = actor;
    applyDrapeOpacity();
  }

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
      '<label>Basemap <input type="range" data-k="bmop" min="0" max="1" step="0.05" value="0.55"></label>' +
      '<label>Elev <input type="number" data-k="emin" step="0.5" title="Legend minimum (m)"> – ' +
      '<input type="number" data-k="emax" step="0.5" title="Legend maximum (m)"> m</label>' +
      '<button data-k="efromview" title="Range the legend over the terrain currently on screen">' +
      'Set min/max from view</button>' +
      '<button data-k="reset">Reset view</button>';
    bar.addEventListener("change", function (e) {
      var k = e.target.getAttribute("data-k");
      if (k !== "emin" && k !== "emax") return;
      var mn = parseFloat(bar.querySelector('[data-k="emin"]').value);
      var mx = parseFloat(bar.querySelector('[data-k="emax"]').value);
      if (isFinite(mn) && isFinite(mx)) applyElevRange(mn, mx);
    });
    bar.addEventListener("input", function (e) {
      var k = e.target.getAttribute("data-k");
      if (k === "axis") { S.axis = parseInt(e.target.value, 10); S.t = 0;
                          bar.querySelector('[data-k="clip"]').value = 0; applyClip(); }
      else if (k === "clip") { S.t = parseFloat(e.target.value); applyClip(); }
      else if (k === "vexag") { S.vexag = parseFloat(e.target.value);
                                bar.querySelector('[data-k="vexagval"]').textContent = S.vexag;
                                applyVexag(); if (S.ren) S.ren.resetCameraClippingRange(); }
      else if (k === "bmop") { S.drapeOpacity = parseFloat(e.target.value); applyDrapeOpacity(); }
      render();
    });
    // The bar sits INSIDE the vtk container, whose interactor handles pointer events for the
    // trackball camera. Stopping the press alone is NOT enough: vtk.js rotates on any
    // pointermove whose `buttons` bit is set, no prior pointerdown needed — and a native
    // slider drag implicitly captures the pointer, so every drag move retargets to the
    // slider and bubbles up through the bar. Stop the whole pointer conversation here.
    ["pointerdown", "pointermove", "pointerup", "mousedown", "mousemove", "mouseup",
     "touchstart", "touchmove", "touchend", "wheel", "dblclick"].forEach(function (t) {
      bar.addEventListener(t, function (e) { e.stopPropagation(); });
    });
    bar.addEventListener("click", function (e) {
      if (e.target.getAttribute("data-k") === "efromview") {
        var rng = visibleElevRange();
        if (rng) { applyElevRange(rng[0], rng[1]); }
        else {
          showHint("No terrain in view — pan/zoom over the mesh, then try again.");
          setTimeout(function () { showHint(""); }, 2500);
        }
        return;
      }
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
    setupInteraction(vtk, el);
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
    S.drapeActor = null;
    clearLabels();

    // This vtk.js build ships vtkPolyData (not vtkUnstructuredGrid), so expand each active hex
    // (8 corner ids: 0-3 = bottom face, 4-7 = top face) to its 6 quad faces — clips cleanly on a cut.
    // Split into two meshes: the TOP layer (terrain-coloured by elevation) and the deeper BODY (a
    // neutral-gray block), so the ground surface shows topography while the block below stays quiet.
    var src = msg.cells, elevArr = msg.cellElev, layArr = msg.cellLayer, nHex = msg.nHex;
    var FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
    var topPolys = [], topElev = [], bodyPolys = [];
    var topCellPts = [], topCellElev = [];                  // one center + elev per TOP CELL
    for (var h = 0; h < nHex; h++) {
      var base = h * 8, ev = elevArr[h], isTop = layArr[h] === 0;
      if (isTop) {                                          // top-face center (corner ids 4..7)
        var sx = 0, sy = 0, sz = 0;
        for (var c = 4; c < 8; c++) {
          var pid = src[base + c] * 3;
          sx += msg.points[pid]; sy += msg.points[pid + 1]; sz += msg.points[pid + 2];
        }
        topCellPts.push(sx / 4, sy / 4, sz / 4);
        topCellElev.push(ev);
      }
      for (var f = 0; f < 6; f++) {
        var fc = FACES[f];
        var q = [4, src[base + fc[0]], src[base + fc[1]], src[base + fc[2]], src[base + fc[3]]];
        if (isTop) { topPolys.push(q[0], q[1], q[2], q[3], q[4]); topElev.push(ev); }
        else { bodyPolys.push(q[0], q[1], q[2], q[3], q[4]); }
      }
    }
    var ptsData = Float32Array.from(msg.points);
    S.topCellPts = Float32Array.from(topCellPts);
    S.topCellElev = Float32Array.from(topCellElev);

    // Terrain colour ramp over the elevation range (full resolution; applied to the top mesh
    // only). Kept in S so the bar's min/max inputs + "set from view" can re-range it live.
    var rng = msg.elevRange || [0, 1], elo = rng[0], ehi = rng[1];
    var ctf = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
    rampCtf(ctf, elo, ehi);
    S.ctf = ctf;
    S.topMapper = null;                                     // set by addMesh (scalars mesh)

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
        S.topMapper = mapper;                               // legend re-ranging hooks in here
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
    buildDrape(vtk, msg, topPolys, ptsData);                // aerial basemap on the top faces
    buildBoundaryLines(vtk, msg.boundaries || []);          // Upstream / FPL / Downstream lines
    buildLabels(vtk, msg.boundaries || []);                 // floating name chips

    // Elevation legend (scalar bar) — created once; its colour map tracks the current mesh.
    // Pinned to the middle-upper right (automated layout spans the full right edge, where its
    // NaN swatch used to collide with the bottom-right orientation gizmo).
    if (!S.scalarBar && vtk.Rendering.Core.vtkScalarBarActor) {
      S.scalarBar = vtk.Rendering.Core.vtkScalarBarActor.newInstance();
      if (S.scalarBar.setAxisLabel) S.scalarBar.setAxisLabel("Elevation (m)");
      if (S.scalarBar.setDrawNanAnnotation) S.scalarBar.setDrawNanAnnotation(false);
      if (S.scalarBar.setAutomated) S.scalarBar.setAutomated(false);
      if (S.scalarBar.setBoxPosition) S.scalarBar.setBoxPosition([0.86, -0.38]);   // NDC (-1..1)
      if (S.scalarBar.setBoxSize) S.scalarBar.setBoxSize([0.13, 1.16]);
      S.ren.addActor(S.scalarBar);
    }
    if (S.scalarBar && S.scalarBar.setScalarsToColors) S.scalarBar.setScalarsToColors(ctf);
    setElevInputs(elo, ehi);                                // a fresh mesh resets the legend range

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
