"""MCP tools, grouped by workflow.

Every module in this package may define ``register(reg: ToolRegistrar) -> None``. Dropping a
new module in here is enough: `register_all` discovers it. A module that fails to import is
recorded in ``IMPORT_FAILURES`` (and fails the test suite) instead of hiding every other tool.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from .registry import ToolRegistrar

log = logging.getLogger("immuno_mcp.tools")
IMPORT_FAILURES: dict[str, str] = {}


def register_all(reg: ToolRegistrar) -> None:
    for mod in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if mod.name.startswith("_") or mod.name == "registry":
            continue
        try:
            module = importlib.import_module(f"{__name__}.{mod.name}")
            register = getattr(module, "register", None)
            if register is not None:
                register(reg)
        except Exception as exc:
            IMPORT_FAILURES[mod.name] = f"{type(exc).__name__}: {exc}"
            log.exception("tool module %s failed to load", mod.name)
