/* HYPE place/stream search — client-side type-ahead via Photon (Komoot, OSM).
 *
 * Attaches (by event delegation, since the Reach pane re-renders) to the #address
 * text input. As the user types it queries Photon directly from the browser (free,
 * no key, CORS-enabled) and shows a dropdown of US places/streams. Picking one fills
 * the box and posts the coordinates to Shiny via Shiny.setInputValue("address_pick",
 * {...}); the server then recenters the map (and the NHD streams auto-load).
 */
(function () {
  "use strict";

  var CONUS = { lat: 39.5, lon: -98.35 };   // bias suggestions to the lower-48
  var MIN_CHARS = 3, DEBOUNCE_MS = 250;
  var menu = null, timer = null, activeInput = null, seq = 0;

  function ensureMenu() {
    if (menu) return menu;
    menu = document.createElement("div");
    menu.className = "hype-ac-menu";
    menu.style.display = "none";
    document.body.appendChild(menu);
    document.addEventListener("mousedown", function (e) {
      if (menu.style.display === "block" && e.target !== activeInput && !menu.contains(e.target)) {
        hide();
      }
    });
    window.addEventListener("resize", hide);
    return menu;
  }

  function hide() { if (menu) menu.style.display = "none"; }

  function position(input) {
    var r = input.getBoundingClientRect();
    menu.style.left = (window.scrollX + r.left) + "px";
    menu.style.top = (window.scrollY + r.bottom + 2) + "px";
    menu.style.width = r.width + "px";
  }

  function pick(input, item) {
    input.value = item.label;
    hide();
    if (window.Shiny && Shiny.setInputValue) {
      Shiny.setInputValue("address_pick",
        { lat: item.lat, lon: item.lon, label: item.label, nonce: Date.now() },
        { priority: "event" });
    }
  }

  function render(items, input) {
    ensureMenu();
    if (!items.length) { hide(); return; }
    position(input);
    menu.innerHTML = "";
    items.forEach(function (it) {
      var row = document.createElement("div");
      row.className = "hype-ac-item";
      var name = document.createElement("span");
      name.className = "hype-ac-name";
      name.textContent = it.label;
      var meta = document.createElement("span");
      meta.className = "hype-ac-meta";
      meta.textContent = it.meta || "";
      row.appendChild(name);
      row.appendChild(meta);
      // mousedown (not click) so selection beats the input's blur
      row.addEventListener("mousedown", function (ev) { ev.preventDefault(); pick(input, it); });
      menu.appendChild(row);
    });
    menu.style.display = "block";
  }

  function suggest(q, input) {
    var mine = ++seq;
    var url = "https://photon.komoot.io/api/?q=" + encodeURIComponent(q) +
      "&limit=6&lang=en&lat=" + CONUS.lat + "&lon=" + CONUS.lon;
    fetch(url)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || mine !== seq) return;            // ignore stale responses
        var items = (j.features || [])
          .filter(function (f) { return (f.properties || {}).countrycode === "US"; })
          .map(function (f) {
            var p = f.properties || {}, c = f.geometry.coordinates;
            var label = [p.name, p.state].filter(Boolean).join(", ") || p.name || q;
            var kind = (p.osm_value || p.osm_key || "").replace(/_/g, " ");
            var meta = [p.county, kind].filter(Boolean).join(" · ");
            return { label: label, meta: meta, lat: c[1], lon: c[0] };
          });
        render(items, input);
      })
      .catch(function () { hide(); });
  }

  document.addEventListener("input", function (e) {
    var t = e.target;
    if (!t || t.id !== "address") return;
    activeInput = t;
    var q = (t.value || "").trim();
    clearTimeout(timer);
    if (q.length < MIN_CHARS) { hide(); return; }
    timer = setTimeout(function () { suggest(q, t); }, DEBOUNCE_MS);
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && e.target && e.target.id === "address") hide();
  });
})();
