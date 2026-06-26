/* HYPE Reach-tab draw guidance.
 *
 * The server (app.py `_push_reach_state`) sends a "hype_reach" custom message describing how the
 * Reach-tab map should behave, and this module realizes it on the client:
 *
 *   • picking  → show a crosshair-follow tooltip ("Click to select a point on a stream") while the
 *                user picks the 2 auto-delineation points (the actual snap happens server-side).
 *   • arm      → auto-start Leaflet.draw's polyline tool so manual mode shows the real
 *                "Click to start drawing line" crosshair/tooltip with no toolbar button to hunt for.
 *   • canEdit  → double-click the drawn centerline to toggle vertex editing (and save).
 *
 * The Reach-tab toolbar itself is hidden by CSS (app.py `reach_map_style`); the control stays in
 * the DOM, so we drive it by clicking its (hidden) anchors — the supported Leaflet.draw path. The
 * draw/edit "actions" bar's inline display:block is our source of truth for whether a draw/edit is
 * currently active, since Leaflet.draw sets it even while the toolbar is hidden.
 */
(function () {
  "use strict";

  var WRAP = ".hype-map-wrap";
  var PICK_TEXT = "Click to select a point on a stream.";
  var state = { picking: false, arm: false, canEdit: false, step: null };
  var tip = null;

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

  // ---- manual-draw arming (retries while the widget view is still mounting) ----
  function arm(tries) {
    var b = q(".leaflet-draw-draw-polyline");
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

  // ---- apply the latest server state (idempotent; guards keep redundant messages harmless) ----
  function reconcile() {
    if (!state.picking) hideTip();
    if (state.arm) arm(0); else cancelDraw();
    if (!state.canEdit && isEditing()) click(saveLink());    // commit before leaving the step/mode
  }

  function onMessage(s) {
    state.picking = !!s.picking;
    state.arm = !!s.arm;
    state.canEdit = !!s.canEdit;
    state.step = s.step;
    reconcile();
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
