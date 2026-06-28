/* HYPE map draw guidance (Reach + Boundaries steps).
 *
 * The server (app.py `_push_reach_state`) sends a "hype_reach" custom message describing how the
 * map should behave, and this module realizes it on the client:
 *
 *   • picking  → show a crosshair-follow tooltip ("Click to select a point on a stream") while the
 *                user picks the 2 auto-delineation points (Reach auto; the snap happens server-side).
 *   • arm      → auto-start Leaflet.draw's polyline/polygon tool (`armShape`) so the active shape
 *                shows the real "Click to start drawing…" crosshair/tooltip with no button to hunt.
 *   • canEdit  → double-click the loaded shape to toggle vertex editing (and save).
 *   • slot     → which thing is being edited (reach / up / left / right / down / wse); a change
 *                commits/cancels any in-progress draw or edit so switching boundaries stays clean.
 *
 * The toolbar is hidden by CSS (app.py `map_edit_style`); the control stays in the DOM, so we drive
 * it by clicking its (hidden) anchors — the supported Leaflet.draw path. The draw/edit "actions"
 * bar's inline display:block is our source of truth for whether a draw/edit is currently active,
 * since Leaflet.draw sets it even while the toolbar is hidden.
 */
(function () {
  "use strict";

  var WRAP = ".hype-map-wrap";
  var PICK_TEXT = "Click to select a point on a stream.";
  var state = { picking: false, arm: false, canEdit: false, autoEdit: false,
                step: null, slot: null, slotName: "", armShape: "line" };
  var tip = null, bar = null;

  // ---- DOM helpers (all scoped to the map wrapper) ----
  function wrap() { return document.querySelector(WRAP); }
  function mapEl() { return document.querySelector(WRAP + " .leaflet-container"); }
  function q(sel) { var w = wrap(); return w ? w.querySelector(sel) : null; }
  function sectionOf(btnSel) { var b = q(btnSel); return b ? b.closest(".leaflet-draw-section") : null; }
  function actionsOf(sec) { return sec ? sec.querySelector(".leaflet-draw-actions") : null; }
  function isActive(sec) { var a = actionsOf(sec); return !!(a && a.style.display === "block"); }
  function isDrawing() { return isActive(sectionOf(".leaflet-draw-draw-polyline")); }
  function isEditing() { return isActive(sectionOf(".leaflet-draw-edit-edit")); }
  function withinMap(node) { var m = mapEl(); return !!(m && node && m.contains(node)); }

  function actionLink(sec, re) {
    var a = actionsOf(sec);
    if (!a) return null;
    var links = a.querySelectorAll("a");
    for (var i = 0; i < links.length; i++) {
      if (re.test(links[i].textContent || "")) return links[i];
    }
    return null;
  }
  function saveLink() {
    var sec = sectionOf(".leaflet-draw-edit-edit");
    return actionLink(sec, /save/i) || (actionsOf(sec) ? actionsOf(sec).querySelector("a") : null);
  }

  function click(el) { if (el) { try { el.click(); } catch (e) { /* ignore */ } } }

  // ---- draw arming (line or polygon per armShape; retries while the view is still mounting) ----
  function drawAnchor() {
    return state.armShape === "polygon" ? q(".leaflet-draw-draw-polygon")
                                        : q(".leaflet-draw-draw-polyline");
  }
  function arm(tries) {
    var b = drawAnchor();
    if (b) { if (!isDrawing()) click(b); return; }
    if ((tries || 0) < 20) setTimeout(function () { arm((tries || 0) + 1); }, 100);
  }
  // Re-clicking the toolbar button never cancels an active Leaflet.draw draw (it only ever calls
  // enable()), so cancel via the draw section's "Cancel" action link instead.
  function cancelDraw() {
    if (isDrawing()) click(actionLink(sectionOf(".leaflet-draw-draw-polyline"), /cancel/i));
  }

  // ---- follow-cursor pick tooltip ----
  function ensureTip() {
    if (tip) return tip;
    tip = document.createElement("div");
    tip.className = "hype-pick-tooltip";
    tip.style.display = "none";
    tip.textContent = PICK_TEXT;
    document.body.appendChild(tip);
    return tip;
  }
  function hideTip() { if (tip) tip.style.display = "none"; }
  function moveTip(e) {
    if (!state.picking) { hideTip(); return; }
    if (!withinMap(e.target)) { hideTip(); return; }
    var t = ensureTip();
    t.style.display = "block";
    t.style.left = (e.clientX + 16) + "px";
    t.style.top = (e.clientY - 10) + "px";
  }

  // ---- double-click to edit the drawn centerline (captures before Leaflet's dbl-click zoom) ----
  function onDblCapture(e) {
    if (!state.canEdit || isDrawing() || !withinMap(e.target)) return;
    e.preventDefault();
    e.stopPropagation();
    if (!isEditing()) click(q(".leaflet-draw-edit-edit"));   // enter vertex editing
    else click(saveLink());                                  // commit edits
  }

  function setInput(name) {
    if (window.Shiny && Shiny.setInputValue) Shiny.setInputValue(name, Date.now(), { priority: "event" });
  }

  // Auto-enter Leaflet.draw edit for a just-selected boundary. A short delay lets the server's
  // dc.data load arrive first (so the layer is in the edit featureGroup); double-click is the
  // fallback if the timing is missed.
  function scheduleEnterEdit() {
    setTimeout(function () {
      if (state.autoEdit && !isEditing()) click(q(".leaflet-draw-edit-edit"));
    }, 400);
  }

  // ---- floating boundary edit bar (Clear & redraw / Done) ----
  function ensureBar() {
    if (bar) return bar;
    bar = document.createElement("div");
    bar.className = "hype-edit-bar";
    bar.style.display = "none";
    bar.innerHTML = '<span class="hype-edit-label"></span>' +
      '<button type="button" data-k="clear" class="hype-edit-btn">Clear &amp; redraw</button>' +
      '<button type="button" data-k="done" class="hype-edit-btn primary">Done</button>';
    bar.addEventListener("click", function (e) {
      var k = e.target && e.target.getAttribute("data-k");
      if (k === "done") { if (isEditing()) click(saveLink()); cancelDraw(); setInput("bnd_done"); }
      else if (k === "clear") { setInput("bnd_clear"); }
    });
    (wrap() || document.body).appendChild(bar);
    return bar;
  }
  function showBar(mode, name) {
    var b = ensureBar();
    b.querySelector(".hype-edit-label").textContent = (mode === "draw" ? "Drawing " : "Editing ") + name;
    b.querySelector('[data-k="clear"]').style.display = mode === "draw" ? "none" : "";
    b.querySelector('[data-k="done"]').textContent = mode === "draw" ? "Cancel" : "Done";
    b.style.display = "flex";
  }
  function hideBar() { if (bar) bar.style.display = "none"; }

  // ---- apply the latest server state (idempotent; guards keep redundant messages harmless) ----
  function reconcile() {
    if (!state.picking) hideTip();
    if (state.arm) arm(0); else cancelDraw();
    if (!state.canEdit && !state.autoEdit && isEditing()) click(saveLink());  // commit before leaving
    var onBnd = state.step === "boundaries" && !!state.slot;
    if (onBnd && state.autoEdit) showBar("edit", state.slotName || "boundary");
    else if (onBnd && state.arm) showBar("draw", state.slotName || "boundary");
    else hideBar();
  }

  function onMessage(s) {
    var nextSlot = s.slot || null;
    var slotChanged = nextSlot !== state.slot;
    if (slotChanged) {                    // switching target → commit/cancel any in-progress work
      if (isEditing()) click(saveLink());
      if (isDrawing()) cancelDraw();
    }
    state.slot = nextSlot;
    state.slotName = s.slotName || "";
    state.armShape = s.armShape || "line";
    state.picking = !!s.picking;
    state.arm = !!s.arm;
    state.canEdit = !!s.canEdit;
    state.autoEdit = !!s.autoEdit;
    state.step = s.step;
    reconcile();
    if (slotChanged && state.autoEdit) scheduleEnterEdit();   // single-click select → edit now
  }

  // ---- wiring ----
  document.addEventListener("mousemove", moveTip);
  document.addEventListener("dblclick", onDblCapture, true);   // capture phase → beats dbl-click zoom

  function register() {
    if (window.Shiny && Shiny.addCustomMessageHandler) {
      Shiny.addCustomMessageHandler("hype_reach", onMessage);
      return true;
    }
    return false;
  }
  if (!register()) document.addEventListener("shiny:connected", register);
})();
