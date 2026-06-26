"""Utility package for reusable, side‑effect‑free helpers.

* Lazily exposes ``functions.path_utils`` and ``functions.raster_utils`` so the
  package works even if one of the helper modules is temporarily missing or the
  caller only needs one of them.

Usage examples
--------------
```python
from functions import path_utils as pu           # loads path_utils only
from functions import raster_utils as ru         # loads raster_utils on demand
```
"""
from importlib import import_module
import sys

__all__ = ["path_utils", "raster_utils","report_utils","model_utils"]  # modules to be lazily loaded

# ---------------------------------------------------------------------------
# Lazy loader – import sub‑modules only when they are first accessed
# ---------------------------------------------------------------------------

def __getattr__(name):  # PEP‑562
    if name in __all__:
        try:
            module = import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            raise AttributeError(
                f"Module '{name}' not available inside the 'functions' package"
            ) from exc
        sys.modules[f"{__name__}.{name}"] = module  # cache for future imports
        globals()[name] = module
        return module
    raise AttributeError(name)
