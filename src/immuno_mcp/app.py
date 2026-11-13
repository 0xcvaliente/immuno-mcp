"""Application context: everything a tool needs, built once per server process."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from .core.cache import DiskCache
from .core.connector import CONNECTORS, Connector
from .core.errors import ErrorCode, ImmunoError
from .core.http import HttpClient
from .core.settings import Settings
from .workspace import Workspace


@dataclass
class AppContext:
    settings: Settings
    http: HttpClient
    cache: DiskCache
    workspace: Workspace
    connectors: dict[str, Connector]
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    extras: dict[str, Any] = field(default_factory=dict)  # modules attach their own services here
    # Every distinct upstream source/version a tool touched this session, for the run manifest.
    provenance_log: list[dict[str, Any]] = field(default_factory=list)
    _seen_provenance: set[tuple[Any, ...]] = field(default_factory=set, repr=False)

    def note_result(self, result: Any) -> None:
        """Record the provenance of a tool result. Called for every tool by the guard."""
        structured = getattr(result, "structured_content", None)
        if not isinstance(structured, dict):
            return
        for p in structured.get("provenance") or []:
            key = (p.get("source"), p.get("url"), p.get("version"), p.get("sha256"))
            if key not in self._seen_provenance and len(self.provenance_log) < 2000:
                self._seen_provenance.add(key)
                self.provenance_log.append(p)

    def connector(self, name: str) -> Connector:
        try:
            return self.connectors[name]
        except KeyError as exc:
            raise ImmunoError(
                ErrorCode.NOT_CONFIGURED,
                f"connector {name!r} is not available in this build",
                hint=f"Available: {', '.join(sorted(self.connectors))}",
            ) from exc

    async def aclose(self) -> None:
        await self.http.aclose()
        self.cache.close()
        self.workspace.close()


def build_app(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    in_memory: bool = False,
) -> AppContext:
    """Create the context. ``in_memory=True`` keeps cache and workspace off disk (tests)."""
    from . import connectors as _connectors  # noqa: F401  (importing registers every connector)

    cache = DiskCache(":memory:" if in_memory else settings.cache_path)
    workspace = Workspace(
        ":memory:" if in_memory else settings.workspace_path,
        artifacts_dir=None if in_memory else settings.artifacts_dir,
    )
    http = HttpClient(settings, cache=cache, transport=transport)
    instances = {name: cls(http, settings) for name, cls in CONNECTORS.items()}
    return AppContext(
        settings=settings, http=http, cache=cache, workspace=workspace, connectors=instances
    )
