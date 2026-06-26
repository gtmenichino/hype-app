# src/hypetool/core/logging_bridge.py
# Tiny bridge so the same code can log to CLI or the ArcGIS Geoprocessing pane
from __future__ import annotations

from typing import Callable, Optional


def make_logger(messages: Optional[object] = None) -> Callable[[str], None]:
    """
    Return a callable that writes to:
      - ArcGIS GP messages (messages.AddMessage) if provided
      - else prints to stdout
    """
    if messages and hasattr(messages, "AddMessage"):
        def _log(msg: str) -> None:
            try:
                messages.AddMessage(str(msg))
            except Exception:
                print(str(msg))
        return _log
    else:
        def _log(msg: str) -> None:
            print(str(msg))
        return _log

