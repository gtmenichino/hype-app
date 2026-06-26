# Hyporheic web app (`hype-app`)

A StreamStats-style [Shiny for Python](https://shiny.posit.co/py/) application that runs
the Hyporheic tool's **MODFLOW 6 + MODPATH 7** engine (the vendored headless `hypetool`
core + FloPy) on **Posit Connect Cloud** (Linux). Modeled on the EASI app.

> **Status:** the full interactive app (`app.py`) is built and validated locally — the
> headless engine is parity-tested **bit-identical** to the ArcGIS/CLI engine, and
> K-zones, spatially-varying gradients, and the channel-WSE derivation are all exercised.
> **Still deploy `smoke_app.py` first** to confirm the Connect Cloud sandbox executes the
> bundled Linux binary, then deploy `app.py`.

## What it does

1. **Draw** the groundwater-domain polygon, then the **left** and **right** floodplain
   boundary lines on a USGS basemap (optional extra polygons become **K-zones**).
2. **Fetch** the USGS 3DEP terrain DEM for the area.
3. **Choose the water surface:** the full DEM, a **channel-only** mask (detects the
   hydro-flattened flat channel; tunable threshold), or an **uploaded** WSE raster.
4. **Configure** grid + hydraulic parameters (a live **green/amber/red grid guardrail**
   keeps runs in bounds) — including **4-corner** or **spatially-varying** gradient
   boundary conditions and optional K-zone conductivities.
5. **Run** MODFLOW 6 + MODPATH 7 on the bundled Linux binaries, watch the live log, then
   view **pathlines + particle points** on the map and **download** a results bundle.

## Layout

```
hype-app/
  smoke_app.py        # ← deploy this FIRST (Connect Cloud subprocess de-risk)
  app.py              # the full interactive app
  hype_app/           # geometry / dem / estimate / run / results / bundle
  hypetool/           # vendored headless engine (no separate install needed)
  bin/linux/          # mf6 (6.7.0), mp7 (7.2.001) — Linux x64, tracked via Git LFS
  requirements.txt    # manylinux wheels only (pip-only; no apt)
  www/styles.css
```

## Deploy to Posit Connect Cloud (VS Code + Posit Publisher)

Deployed straight from VS Code with the **Posit Publisher** extension; the runtime is
pinned to **Python 3.12** by the `.python-version` file at the repo root (the Publisher
reads it into the deployment's Python constraint, and Connect Cloud provisions a matching
3.12 interpreter).

1. **Install** the [Posit Publisher](https://marketplace.visualstudio.com/items?itemName=Posit.publisher)
   VS Code extension and open the `hype-app` folder.
2. **Smoke test first.** In the Publisher panel, **Add Deployment** → select
   **`smoke_app.py`** as the entrypoint (it detects *Shiny / python-shiny*, reads
   `.python-version` = 3.12 and `requirements.txt`), choose **Posit Connect Cloud** as the
   destination, sign in to add the credential, and **Deploy**. Open it, click **Run smoke
   test** → all imports `OK`, `mf6 --version` prints, and the solve reports
   `SUCCESS — heads = [1.0, 0.5, 0.0]` → green light.
3. **Then the app.** Add a second deployment with **`app.py`** as the entrypoint and deploy
   it. It installs `requirements.txt` and imports the vendored `hypetool`.

The Publisher uploads the **local** files (including the materialized `bin/linux/`
binaries), so Git LFS is **not** needed for this path — see *Source control* below.

> If `smoke_app.py` fails with `libgfortran.so.5: cannot open shared object file`, the
> sandbox lacks the gfortran runtime: drop `libgfortran.so.5` / `libquadmath.so.0` /
> `libgcc_s.so.1` into `bin/linux/` (already on `LD_LIBRARY_PATH` via
> `hype_app/run.py::_prepare_linux_bin`) and redeploy, or switch to static MF6 binaries.

### Connect Cloud content settings (for `app.py`)
- **Read/request timeout** → raise toward the 240-min max (a run is minutes).
- **Startup timeout** → raise (first cold import of the geo + FloPy stack is slow).
- **Memory** → 8–16 GB, **CPU** → 2–4 (MODFLOW solve + raster work).
- **Env:** `HYRIVER_CACHE_NAME=/tmp/hype_hyriver.sqlite` (3DEP cache → ephemeral /tmp).

## Source control (GitHub + Git LFS)

The repo lives at **[gtmenichino/hype-app](https://github.com/gtmenichino/hype-app)**
(public). The ~49 MB Linux `bin/linux/{mf6,mp7}` binaries are tracked with **Git LFS**
(`.gitattributes`), so clones stay lean:
```bash
git lfs install
git add -A && git commit -m "…"
git push
```
Confirm the GitHub file view shows the binaries as "Stored with Git LFS". (LFS matters
only for GitHub and the optional deploy-from-GitHub route; the VS Code Publisher above
bundles the local binaries directly.)

## Local run

```bash
python -m venv .venv && .venv\Scripts\activate      # (POSIX: . .venv/bin/activate)
pip install -r requirements.txt
# Windows dev: point at Windows MODFLOW binaries (bin/linux holds Linux ones):
set HYPE_MODFLOW_BIN=C:\path\to\hype-tool\src\hypetool\bin\modflow
shiny run app.py
```
`bin/linux/` holds **Linux** binaries, so an actual model run only executes on Linux /
Connect Cloud unless `HYPE_MODFLOW_BIN` points at platform-native binaries. The map,
drawing, DEM fetch, and UI work locally on any OS.

## Keeping the engine in sync / refreshing binaries

The `hypetool/` package is vendored from `hype-tool/src/hypetool`. Re-copy it after core
changes (exclude `bin/` and `esri/`). Refresh the Linux binaries with:
```bash
python -m flopy.utils.get_modflow bin/linux --subset mf6,mp7 --ostag linux
```
