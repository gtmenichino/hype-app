"""Assemble + execute a run_hyporheic call (invoked inside a worker thread)."""
from __future__ import annotations

import os
import traceback
from pathlib import Path

from hypetool.core.run_headless import run_hyporheic

_APP_ROOT = Path(__file__).resolve().parent.parent


def modflow_bin_dir() -> str:
    """Where to find mf6/mp7. The env override HYPE_MODFLOW_BIN wins (handy for local
    Windows dev — point it at the hype-tool Windows bin); otherwise the bundled Linux
    binaries in bin/linux (Connect Cloud)."""
    env = os.environ.get("HYPE_MODFLOW_BIN")
    return env if env else str(_APP_ROOT / "bin" / "linux")


def _prepare_linux_bin(bin_dir: str) -> None:
    """On Linux (Connect Cloud), make the bundled mf6/mp7 executable — the +x bit is lost when the
    binaries are committed from Windows (git stores mode 100644), so FloPy's subprocess would hit
    'Permission denied' — and prepend the bin dir to LD_LIBRARY_PATH so any gfortran runtime .so's
    bundled alongside the binaries are found. No-op on Windows or for a missing dir."""
    import stat
    import sys
    if sys.platform.startswith("win"):
        return
    d = Path(bin_dir)
    if not d.is_dir():
        return
    for name in ("mf6", "mp7"):
        f = d / name
        if f.exists():
            try:
                f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except Exception:  # noqa: BLE001
                pass
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if str(d) not in cur.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(d), cur]) if cur else str(d)


def execute(*, domain_gdf, left_gdf, right_gdf, crs, dem_path, wse_path, wse_mode,
            wse_relief_thresh, kh_polygon_gdf, params, work_dir, log):
    """Thin wrapper so the worker thread has one obvious call. Returns the artifact dict."""
    return run_hyporheic(
        domain_gdf=domain_gdf,
        left_line_gdf=left_gdf,
        right_line_gdf=right_gdf,
        crs=crs,
        dem_path=dem_path,
        wse_path=wse_path,
        wse_mode=wse_mode,
        wse_relief_thresh=wse_relief_thresh,
        kh_polygon_gdf=kh_polygon_gdf,
        work_dir=str(work_dir),
        modflow_bin_dir=modflow_bin_dir(),
        log=log,
        make_figures=False,
        **params,
    )


def _modflow_diagnostics(work_dir) -> str:
    """Best-effort: gather the tail of MODFLOW's listing files so a failed run explains
    itself. On a hard crash MODFLOW writes nothing to the queue and the listing stops mid-setup,
    so we read it off disk and flag when it never reached 'Normal termination'."""
    try:
        wd = Path(work_dir)
        files = sorted(wd.glob("**/mfsim.lst")) + sorted(wd.glob("**/gwf_model.lst"))
        finished = False
        parts = []
        for f in files:
            txt = f.read_text(errors="ignore")
            finished = finished or ("Normal termination" in txt)
            parts.append(f"----- {f.name} (tail) -----\n" + "\n".join(txt.splitlines()[-40:]))
        note = ""
        if files and not finished:
            note = ("MODFLOW exited before completing — no solver output was written. This usually "
                    "means it ran out of memory or hit a setup error for a grid this large. Try a "
                    "coarser cell size, shallower depth, or thicker layers.\n\n")
        return (note + "\n\n".join(parts)).strip()
    except Exception:  # noqa: BLE001 — diagnostics must never mask the original error
        return ""


def child_run(payload: dict, q) -> None:
    """Run a job in a separate (spawned) process; stream logs + result over the queue.

    Top-level + picklable so it works under the 'spawn' start method. Rebuilds the
    GeoDataFrames from the payload's GeoJSON, runs the engine, and puts ('log', line)
    messages followed by ('result', dict) or ('error', traceback) onto `q`.
    """
    try:
        _prepare_linux_bin(modflow_bin_dir())   # ensure the Linux mf6/mp7 are executable + linkable
        from hype_app import geometry
        crs = payload["crs"]
        dom = geometry.single_feature_gdf(payload["domain"]).to_crs(crs)
        left = geometry.single_feature_gdf(payload["left"]).to_crs(crs)
        right = geometry.single_feature_gdf(payload["right"]).to_crs(crs)
        khgdf = None
        if payload.get("kzones"):
            khgdf = geometry.features_to_gdf(payload["kzones"])
            khgdf["KH"] = float(payload["kzone_kh"])
            khgdf["KV"] = float(payload["kzone_kv"])
            khgdf = khgdf.to_crs(crs)
        result = execute(
            domain_gdf=dom, left_gdf=left, right_gdf=right, crs=crs,
            dem_path=payload["dem"], wse_path=payload["wse_path"],
            wse_mode=payload["wse_mode"], wse_relief_thresh=payload["wse_relief_thresh"],
            kh_polygon_gdf=khgdf, params=payload["params"], work_dir=payload["work_dir"],
            log=lambda m: q.put(("log", str(m))),
        )
        q.put(("result", result))
    except Exception:
        diag = _modflow_diagnostics(payload.get("work_dir"))
        q.put(("error", (diag + "\n\n" if diag else "") + traceback.format_exc()))
