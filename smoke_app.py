"""Connect Cloud subprocess smoke test for the hyporheic web app.

DEPLOY THIS FIRST. It verifies the three riskiest unknowns on Posit Connect Cloud
before we invest in the full UI:
  1) the heavy manylinux geo + flopy wheels import,
  2) the bundled Linux mf6/mp7 binaries are executable under the Connect sandbox, and
  3) a real (tiny) MODFLOW 6 solve runs end-to-end in writable /tmp.
If the page shows PASS, the full app is safe to build and deploy.
"""
from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import anyio
from shiny import App, reactive, render, ui

BIN = Path(__file__).parent / "bin" / "linux"
EXE = "mf6.exe" if sys.platform.startswith("win") else "mf6"

# Heavy stack the real app needs — verifying these import confirms the manylinux
# wheels resolve on Connect Cloud.
STACK = ["shiny", "shinywidgets", "ipyleaflet", "flopy", "numpy", "scipy", "pandas",
         "geopandas", "shapely", "pyproj", "rasterio", "rioxarray", "xarray",
         "netCDF4", "skimage", "pydantic", "py3dep"]


def _chmodx(p: Path) -> None:
    if p.exists():
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _imports_report() -> str:
    import importlib
    out = []
    for m in STACK:
        try:
            mod = importlib.import_module(m)
            out.append(f"OK    {m} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            out.append(f"MISS  {m}: {type(e).__name__}: {e}")
    return "\n".join(out)


def _mf6_version() -> str:
    mf6 = BIN / EXE
    _chmodx(mf6)
    try:
        r = subprocess.run([str(mf6), "--version"], capture_output=True, text=True, timeout=60)
        return f"exit={r.returncode}\n{(r.stdout or '').strip()}\n{(r.stderr or '').strip()}".strip()
    except Exception as e:  # noqa: BLE001
        return f"FAILED to execute {mf6}: {type(e).__name__}: {e}"


def _tiny_mf6_solve() -> str:
    """Build + run a trivial 1x1x3 MF6 model (CHD 1.0 -> 0.0) and read the heads."""
    import numpy as np
    import flopy
    mf6 = BIN / EXE
    _chmodx(mf6)
    ws = Path(tempfile.mkdtemp(prefix="mf6_smoke_"))
    try:
        sim = flopy.mf6.MFSimulation(sim_name="smoke", sim_ws=str(ws), exe_name=str(mf6))
        flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
        flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
        gwf = flopy.mf6.ModflowGwf(sim, modelname="smoke", save_flows=True)
        flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=3, delr=1.0, delc=1.0, top=1.0, botm=0.0)
        flopy.mf6.ModflowGwfic(gwf, strt=1.0)
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.0)
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.0], [(0, 0, 2), 0.0]])
        flopy.mf6.ModflowGwfoc(gwf, head_filerecord="smoke.hds", saverecord=[("HEAD", "ALL")])
        sim.write_simulation(silent=True)
        ok, buff = sim.run_simulation(silent=True)
        if not ok:
            tail = "".join(buff[-15:]) if buff else "(no buffer)"
            return f"run_simulation returned False:\n{tail}"
        head = flopy.utils.HeadFile(str(ws / "smoke.hds")).get_data().ravel()
        return (f"SUCCESS — heads = {np.round(head, 4).tolist()}  (expect ~[1.0, 0.5, 0.0])\n"
                f"workspace = {ws}")
    except Exception:  # noqa: BLE001
        return "EXCEPTION:\n" + traceback.format_exc()


app_ui = ui.page_fillable(
    ui.h2("Hyporheic web app — Connect Cloud smoke test"),
    ui.p("Verifies geo/flopy imports, Linux mf6 execution, and a real MODFLOW 6 solve "
         "under the Connect Cloud sandbox. Click Run, then read the panels below."),
    ui.input_action_button("run", "Run smoke test", class_="btn-primary"),
    ui.output_ui("status"),
    ui.tags.h4("Environment"),
    ui.output_text_verbatim("env"),
    ui.tags.h4("Imports (manylinux wheels)"),
    ui.output_text_verbatim("imports"),
    ui.tags.h4("mf6 --version (subprocess execution)"),
    ui.output_text_verbatim("mfver"),
    ui.tags.h4("Tiny MODFLOW 6 solve"),
    ui.output_text_verbatim("solve"),
    title="Hyporheic smoke test",
)


def server(input, output, session):
    res = reactive.value(None)

    @reactive.extended_task
    async def run_task() -> dict:
        def _work():
            return {
                "env": (f"platform : {platform.platform()}\n"
                        f"python   : {sys.version.split()[0]}\n"
                        f"cwd      : {os.getcwd()}\n"
                        f"bin dir  : {BIN}  (exists={BIN.exists()})\n"
                        f"bin files: {[p.name for p in BIN.glob('*')] if BIN.exists() else 'MISSING'}"),
                "imports": _imports_report(),
                "mfver": _mf6_version(),
                "solve": _tiny_mf6_solve(),
            }
        return await anyio.to_thread.run_sync(_work)

    @reactive.effect
    @reactive.event(input.run)
    def _go():
        res.set(None)
        run_task()

    @reactive.effect
    def _done():
        if run_task.status() in ("initial", "running"):
            return
        try:
            res.set(run_task.result())
        except Exception as e:  # noqa: BLE001
            res.set({"env": f"task error: {e}", "imports": "", "mfver": "", "solve": ""})

    @render.ui
    def status():
        s = run_task.status()
        if s == "running":
            return ui.div(ui.tags.b("Running… "), "building and solving a MODFLOW model")
        r = res()
        if r is not None:
            ok = "SUCCESS" in (r.get("solve") or "")
            return ui.div(ui.tags.b("PASS ✅" if ok else "CHECK RESULTS ⚠️"),
                          style="font-size:1.2rem;margin:.5rem 0;")
        return None

    @render.text
    def env():
        return (res() or {}).get("env", "(click Run)")

    @render.text
    def imports():
        return (res() or {}).get("imports", "")

    @render.text
    def mfver():
        return (res() or {}).get("mfver", "")

    @render.text
    def solve():
        return (res() or {}).get("solve", "")


app = App(app_ui, server)
