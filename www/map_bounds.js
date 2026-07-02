// Report the Leaflet map's view bounds to Shiny as input `map_bounds`
// ({west, south, east, north}, EPSG:4326), refreshed on every moveend/zoomend.
//
// Why this exists: ipyleaflet's server-side `Map.bounds` trait arrives DEGENERATE in this
// stack — ((center, center)) instead of the real extent — so server features that need the
// current view (e.g. the DEM step's "Recalculate legend from view") can't use it. The map
// instance lives inside the jupyter-leaflet bundle with no global registry, so it is
// captured here via L.Map.addInitHook (public Leaflet API), with a late-capture fallback
// (a transient L.Evented.fire hook + synthetic mousemove) in case the map was constructed
// before the hook registered.
(function () {
  "use strict";

  function report(map) {
    try {
      // NOT map.getBounds(): in this widget layout the container's clientWidth/Height
      // are 0 (absolutely-positioned children), so Leaflet's cached _size — and with it
      // getBounds() and ipyleaflet's bounds trait — collapse to the center point. The
      // rendered box from getBoundingClientRect() is correct, and containerPointToLatLng
      // only needs the pane offset, not the cached size.
      var r = map.getContainer().getBoundingClientRect();
      var w = r.width, h = r.height;
      if (!(w > 0 && h > 0)) {           // hidden/backgrounded tab: lumino layout is 0×0
        w = window.innerWidth || 1000;   // → approximate the view with the window size
        h = window.innerHeight || 700;
      }
      var a = map.containerPointToLatLng([0, 0]);
      var b = map.containerPointToLatLng([w, h]);
      if (window.Shiny && window.Shiny.setInputValue) {
        window.Shiny.setInputValue("map_bounds", {
          west: Math.min(a.lng, b.lng), south: Math.min(a.lat, b.lat),
          east: Math.max(a.lng, b.lng), north: Math.max(a.lat, b.lat)
        });
      }
    } catch (e) { /* map not ready — next moveend will report */ }
  }

  function attach(map) {
    if (window.__hypeMap === map) return;
    window.__hypeMap = map;
    map.on("moveend zoomend", function () { report(map); });
    report(map);
  }

  function lateCapture() {
    if (window.__hypeMap || !window.L || !window.L.Evented) return;
    var orig = window.L.Evented.prototype.fire;
    window.L.Evented.prototype.fire = function () {
      if (this instanceof window.L.Map) attach(this);
      return orig.apply(this, arguments);
    };
    var cont = document.querySelector(".leaflet-container");
    if (cont) {
      cont.dispatchEvent(new MouseEvent("mousemove",
        {bubbles: true, clientX: 5, clientY: 5}));
    }
    window.L.Evented.prototype.fire = orig;
  }

  var hooked = false;
  var tries = 0;
  var t = setInterval(function () {
    tries += 1;
    if (!hooked && window.L && window.L.Map && window.L.Map.addInitHook) {
      hooked = true;
      window.L.Map.addInitHook(function () { attach(this); });
      lateCapture();                       // in case the map beat the hook
    }
    if (hooked && !window.__hypeMap) lateCapture();
    if (window.__hypeMap || tries > 150) clearInterval(t);   // give up after ~30 s
  }, 200);
})();
