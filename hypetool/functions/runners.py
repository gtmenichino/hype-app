# src/hypetool/functions/runners.py
#A small, focused subprocess runner for MF6/MP7 with live stdout streaming (usable from CLI or the .pyt through a callback).
from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Optional


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _repo_root_from_here() -> Path:
    # this file -> functions/ -> hypetool/ -> src/ -> repo root
    return Path(__file__).resolve().parents[4]


def _platform_exe(name: str) -> str:
    """Return the platform-appropriate executable name: ``mf6.exe`` on Windows,
    ``mf6`` on Linux/macOS. Keeps the same logic FloPy uses so a bundled Linux
    binary directory resolves correctly off-Windows."""
    return f"{name}.exe" if sys.platform.startswith("win") else name


def resolve_exe(candidate: str | Path | None,
                *,
                exe_name: str,
                fallback_bin_subdir: str = "bin/modflow") -> str:
    """
    Resolve an executable path in a robust way:
      1) explicit candidate path if provided
      2) environment variables (MODFLOW6_BIN, MODPATH7_BIN)
      3) repo-local bin/modflow/<exe> (when running from source tree)
      4) PATH (shutil.which)
    """
    # 1) explicit path
    if candidate:
        p = Path(candidate)
        if p.exists():
            return str(p)

    # 2) environment hints
    env_var = "MODFLOW6_BIN" if exe_name.lower().startswith("mf6") else \
              "MODPATH7_BIN" if exe_name.lower().startswith("mp7") else None
    if env_var:
        env_dir = os.getenv(env_var)
        if env_dir:
            p = Path(env_dir) / exe_name
            if p.exists():
                return str(p)

    # 3) repo-local bin/modflow
    repo_root = _repo_root_from_here()
    local = repo_root / fallback_bin_subdir / exe_name
    if local.exists():
        return str(local)

    # 4) PATH
    found = shutil.which(exe_name)
    if found:
        return found

    raise FileNotFoundError(
        f"Could not locate executable '{exe_name}'. "
        f"Tried explicit path, ${env_var}, '{fallback_bin_subdir}', and PATH."
    )


def run_cmd_stream(cmd: list[str],
                   *,
                   cwd: str | Path | None = None,
                   line_callback: Optional[Callable[[str], None]] = None) -> int:
    """
    Run a command and stream combined stdout/stderr line-by-line to callback.
    Returns the process return code.
    """
    with subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line_callback:
                line_callback(line.rstrip("\n"))
        return proc.wait()


def run_mf6(workspace: str | Path,
            *,
            exe_path: str | Path | None = None,
            namefile: str = "mfsim.nam",
            line_callback: Optional[Callable[[str], None]] = None) -> None:
    mf6 = resolve_exe(exe_path, exe_name=_platform_exe("mf6"))
    rc = run_cmd_stream([mf6, "-i", namefile], cwd=workspace, line_callback=line_callback)
    if rc != 0:
        raise RuntimeError(f"MODFLOW 6 exited with code {rc}")


def run_mp7(workspace: str | Path,
            *,
            exe_path: str | Path | None = None,
            namefile: str | None = None,
            line_callback: Optional[Callable[[str], None]] = None) -> None:
    mp7 = resolve_exe(exe_path, exe_name=_platform_exe("mp7"))
    cmd = [mp7] + (["-i", namefile] if namefile else [])
    rc = run_cmd_stream(cmd, cwd=workspace, line_callback=line_callback)
    if rc != 0:
        raise RuntimeError(f"MODPATH 7 exited with code {rc}")

