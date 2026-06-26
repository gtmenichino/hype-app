from __future__ import annotations
"""Filesystem & environment helpers shared across the project.

Public helpers
--------------
* ``find_project_root(start, marker="inputs.yaml")`` â€“ walk upwards until a
  directory containing *marker* is found.
* ``default_modflow_bin()`` â€“ return the packaged MF bin folder
  (hypetool/bin/modflow).
* ``detect_modflow_exes(bin_dir)`` â€“ locate mf6(.exe)/mp7(.exe) in a folder.
* ``add_modflow_executables(folder)`` â€“ append folder to PATH so exes are runnable.
* ``ensure_modflow_on_path(preferred_dir=None)`` â€“ add preferred or packaged bin
  to PATH and return discovered paths.
* ``download_modflow(folder="...")`` â€“ fetch executables once (developer use).

Centralising these utilities avoids duplicating common patterns across scripts.
"""
from pathlib import Path
from typing import Iterable, Any, Optional, Dict
import os
import sys
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent  # .../hypetool/functions
PKG_ROOT = BASE_DIR.parent                          # .../hypetool

def resource(*parts: str) -> Path:
    """Return an absolute path inside the package folder."""
    return PKG_ROOT.joinpath(*parts)

# Optional import â€“ only required for the download helper
try:
    from flopy.utils import get_modflow as _get_modflow  # type: ignore
except ModuleNotFoundError:  # pragma: no cover â€“ Flopy may not be installed
    _get_modflow = None  # will raise inside download_modflow if used

__all__ = [
    "find_project_root",
    "default_modflow_bin",
    "detect_modflow_exes",
    "add_modflow_executables",
    "ensure_modflow_on_path",
    "download_modflow",
    "truncate_middle",
    "print_config_table",
]

# ----------------------------------------------------------------------------- #
# Root discovery (mostly useful for dev)
# ----------------------------------------------------------------------------- #

def find_project_root(start: Path, marker: str = "inputs.yaml") -> Path:
    """Walk up from *start* until a directory containing *marker* is found."""
    _cur = start.resolve()
    while _cur != _cur.parent:  # stop at filesystem root
        if (_cur / marker).exists():
            return _cur
        _cur = _cur.parent
    raise RuntimeError(
        f"Project root not found â€“ '{marker}' not encountered above {start}"
    )

# ----------------------------------------------------------------------------- #
# Executable resolution
# ----------------------------------------------------------------------------- #

def default_modflow_bin() -> Path:
    """Return the packaged bin folder: hypetool/bin/modflow."""
    return resource("bin", "modflow")

def _iter_executables(folder: Path) -> Iterable[Path]:
    """Yield files in *folder* that look like executables (platform agnostic)."""
    exts = {".exe", ""} if sys.platform.startswith("win") else {""}
    if not folder.exists():
        return
    for p in folder.iterdir():
        if p.is_file() and (p.suffix.lower() in exts or p.name.lower() in {"mf6", "mp7"}):
            yield p

def detect_modflow_exes(bin_dir: Path) -> Dict[str, Optional[Path]]:
    """Find mf6(.exe) and mp7(.exe) inside *bin_dir*."""
    mf6 = None
    mp7 = None
    if bin_dir and bin_dir.exists():
        # Prefer exact names
        cand_mf6 = bin_dir / ("mf6.exe" if sys.platform.startswith("win") else "mf6")
        cand_mp7 = bin_dir / ("mp7.exe" if sys.platform.startswith("win") else "mp7")
        if cand_mf6.exists():
            mf6 = cand_mf6
        if cand_mp7.exists():
            mp7 = cand_mp7
        # Fallback: scan folder (handles unpacked distro names)
        if mf6 is None or mp7 is None:
            for p in _iter_executables(bin_dir):
                name = p.name.lower()
                if mf6 is None and name.startswith("mf6"):
                    mf6 = p
                if mp7 is None and name.startswith("mp7"):
                    mp7 = p
                if mf6 is not None and mp7 is not None:
                    break
    return {"mf6": mf6, "mp7": mp7}

def add_modflow_executables(folder: str | Path) -> None:
    """Append *folder* to PATH so mf6/mp7 are runnable by name."""
    folder = Path(folder).resolve()
    if not folder.exists():
        raise FileNotFoundError(f"MF bin folder not found: {folder}")
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(folder) not in parts:
        os.environ["PATH"] = os.pathsep.join([os.environ.get("PATH", ""), str(folder)]) if parts[0] else str(folder)
        print(f"Added to PATH: {folder}")

def ensure_modflow_on_path(preferred_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Try preferred_dir, then the packaged bin, to ensure mf6/mp7 are runnable.
    Returns dict: {'ok': bool, 'dir': Path|None, 'mf6': Path|None, 'mp7': Path|None}
    """
    candidates = [preferred_dir] if preferred_dir else []
    candidates.append(default_modflow_bin())
    for d in candidates:
        if d and d.exists():
            try:
                add_modflow_executables(d)
            except Exception:
                continue
            exes = detect_modflow_exes(d)
            ok = bool(exes["mf6"] or exes["mp7"])
            return {"ok": ok, "dir": d, **exes}
    return {"ok": False, "dir": None, "mf6": None, "mp7": None}

# ----------------------------------------------------------------------------- #
# Convenience â€“ download MODFLOW binaries via Flopy (dev convenience)
# ----------------------------------------------------------------------------- #

def download_modflow(folder: str | Path = None) -> None:
    """Download MODFLOW executables â€“ **developer convenience only**.

    * If *folder* is omitted, downloads into the **packaged** bin (dev tree).
    * If at least one executable already exists there, returns immediately.
    """
    if _get_modflow is None:
        raise ImportError("Flopy is not installed â€“ `pip install flopy` to enable downloads.")
    target = Path(folder).resolve() if folder else default_modflow_bin()
    target.mkdir(parents=True, exist_ok=True)
    # Early exit if already present
    if list(_iter_executables(target)):
        print(f"âœ” MODFLOW executables already present in '{target}'. Download skipped.")
        return
    _get_modflow(bindir=str(target))
    print("MODFLOW executables downloaded successfully.")
    print("Contents of the MODFLOW executable directory:")
    for item in target.iterdir():
        print("  â€¢", item.name)

def default_modflow_bin() -> Path:
    """
    Return the packaged MODFLOW bin folder:
    .../src/hypetool/bin/modflow
    """
    # path_utils.py lives at .../src/hypetool/functions/path_utils.py
    pkg_root = Path(__file__).resolve().parents[1]  # .../src/hypetool
    path1 = pkg_root / "bin" / "modflow"
    return path1


def detect_modflow_exes(folder: str | Path) -> dict:
    """
    Return {'mf6': <path or None>, 'mp7': <path or None>} for files in *folder*.
    """
    folder = Path(folder)
    found = {"mf6": None, "mp7": None}
    if not folder.exists():
        return found

    exts = ([".exe", ""] if sys.platform.startswith("win") else [""])
    for p in folder.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        # MF6
        if any(name == f"mf6{e}" for e in exts):
            found["mf6"] = str(p.resolve())
        # MP7 (some packages use mp7.exe; others modpath.7.exe)
        if any(name == f"mp7{e}" for e in exts) or any(name == f"modpath.7{e}" for e in exts):
            found["mp7"] = str(p.resolve())
    return found

# ----------------------------------------------------------------------------- #
# Pretty config table
# ----------------------------------------------------------------------------- #

from tabulate import tabulate
from typing import Any

def truncate_middle(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    n = (max_len - 3) // 2
    return s[:n] + "..." + s[-n:]

def print_config_table(cfg: Any, max_value_width: int | None = None) -> None:
    data = list(cfg.__dict__.items())
    key_width = max(len(str(k)) for k, _ in data) if data else 4
    term_width = shutil.get_terminal_size().columns
    val_width = max_value_width if max_value_width is not None else max(term_width - key_width - 7, 10)
    truncated = [(k, truncate_middle(str(v), val_width)) for k, v in data]
    print("\nConfiguration settings:\n")
    print(tabulate(truncated, headers=["Key", "Value"], tablefmt="fancy_grid", stralign="left", showindex=False))

