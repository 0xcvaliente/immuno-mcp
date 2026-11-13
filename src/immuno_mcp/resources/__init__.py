"""Reference documents served as MCP resources.

Every ``*.md`` file in this package is exposed at ``immuno://guide/<file name without .md>``.
The first ``# heading`` becomes the title and the first paragraph the description.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.resources import files

from mcp.server.mcpserver import MCPServer

from ..app import AppContext


def _reader(text: str) -> Callable[[], str]:
    def read() -> str:
        return text

    return read


def _title_and_description(stem: str, text: str) -> tuple[str, str]:
    title, description = stem.replace("-", " ").title(), ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and title == stem.replace("-", " ").title():
            title = stripped[2:].strip()
        elif stripped and not stripped.startswith("#") and not description:
            description = stripped[:200]
            break
    return title, description


def register_resources(server: MCPServer, app: AppContext) -> None:
    for entry in sorted(files(__package__).iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".md"):
            continue
        stem = entry.name[:-3]
        text = entry.read_text(encoding="utf-8")
        title, description = _title_and_description(stem, text)
        server.resource(
            f"immuno://guide/{stem}",
            name=stem,
            title=title,
            description=description,
            mime_type="text/markdown",
        )(_reader(text))

    def connector_cards() -> str:
        cards = [
            {
                "name": c.info.name,
                "display_name": c.info.display_name,
                "kind": c.info.kind,
                "homepage": c.info.homepage,
                "data_license": c.info.license,
                "cite": c.info.citation,
                "needs_key": c.info.auth_env if c.info.auth_required else None,
                "tier": c.info.tier,
                "notes": c.info.notes,
            }
            for c in sorted(app.connectors.values(), key=lambda c: c.info.name)
        ]
        return json.dumps(cards, indent=1, ensure_ascii=False)

    server.resource(
        "immuno://connectors",
        name="connectors",
        title="Data sources and their licences",
        description="Every connected source with its data licence and how to cite it.",
        mime_type="application/json",
    )(connector_cards)
