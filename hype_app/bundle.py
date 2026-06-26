"""Zip a run's output directory for download (the app's results live in ephemeral /tmp,
so the user downloads before leaving)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path


def zip_dir(out_dir) -> bytes:
    """Zip everything under `out_dir/summary` (+ the resolved params if present)."""
    root = Path(out_dir)
    summary = root / "summary"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        base = summary if summary.exists() else root
        for p in base.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(base.parent))
    buf.seek(0)
    return buf.read()
