"""Connectors to upstream services. Importing this package registers all of them.

Each module defines one or more `Connector` subclasses decorated with
``@register`` (see ``core/connector.py``). A module that fails to import is recorded in
``IMPORT_FAILURES`` (and fails the test suite) rather than taking the server down.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

log = logging.getLogger("immuno_mcp.connectors")
IMPORT_FAILURES: dict[str, str] = {}

for _mod in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
    if _mod.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_mod.name}")
    except Exception as _exc:
        IMPORT_FAILURES[_mod.name] = f"{type(_exc).__name__}: {_exc}"
        log.exception("connector module %s failed to load", _mod.name)
