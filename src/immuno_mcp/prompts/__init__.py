"""MCP prompts (workflow starters). Each module may define ``register(server, app)``."""

from __future__ import annotations

import importlib
import logging
import pkgutil

from mcp.server.mcpserver import MCPServer

from ..app import AppContext

log = logging.getLogger("immuno_mcp.prompts")
IMPORT_FAILURES: dict[str, str] = {}


def register_prompts(server: MCPServer, app: AppContext) -> None:
    for mod in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if mod.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{mod.name}")
            register = getattr(module, "register", None)
            if register is not None:
                register(server, app)
        except Exception as exc:
            IMPORT_FAILURES[mod.name] = f"{type(exc).__name__}: {exc}"
            log.exception("prompt module %s failed to load", mod.name)
