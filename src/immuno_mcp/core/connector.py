"""Base class for data and compute sources.

A connector is a thin, typed client for one upstream service. It is declared as data
(`ConnectorInfo`), takes an injected `HttpClient`, and must implement `health()` so the same
code path serves diagnostics and the live contract tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel

from .envelope import Provenance
from .errors import ErrorCode, ImmunoError
from .http import HttpClient, HttpResponse
from .settings import Settings


@dataclass(frozen=True)
class ConnectorInfo:
    name: str  # stable id, lowercase, used in tool arguments and provenance
    display_name: str
    kind: str  # structure | sequence | immunology | literature | binding | developability | compute
    base_url: str
    homepage: str
    license: str  # licence or terms of the DATA (not of this code)
    citation: str  # what a paper should cite
    rps: float = 2.0  # polite request rate for this source
    limit_key: str | None = None  # share a rate limit between connectors on one service
    cache_ttl: float = 24 * 3600
    auth_env: str | None = None  # environment variable holding a key, if any
    auth_required: bool = False
    resolves: frozenset[str] = frozenset()  # identifier kinds `resolve()` can confirm
    tier: str = "P0"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


class ResolvedIdentifier(BaseModel):
    """An identifier that was fetched from its source, never one written from memory."""

    kind: str
    id: str
    canonical_id: str
    title: str | None = None
    url: str | None = None
    source: str
    summary: dict[str, Any] = {}


class Connector:
    info: ClassVar[ConnectorInfo]

    def __init__(self, http: HttpClient, settings: Settings) -> None:
        self.http = http
        self.settings = settings

    # -- configuration ----------------------------------------------------------------------

    def configured(self) -> bool:
        """True when everything this connector needs (a key, for some) is available."""
        env = self.info.auth_env
        if not env or not self.info.auth_required:
            return True
        return self.settings.key_present(env)

    def require_configured(self) -> None:
        if not self.configured():
            raise ImmunoError(
                ErrorCode.NEEDS_KEY,
                f"{self.info.display_name} needs {self.info.auth_env} to be set",
                hint=f"Export {self.info.auth_env} in the shell that starts the server.",
            )

    # -- provenance -------------------------------------------------------------------------

    def provenance(
        self,
        resp: HttpResponse | None = None,
        *,
        url: str | None = None,
        version: str | None = None,
        query: dict[str, Any] | None = None,
    ) -> Provenance:
        return Provenance(
            source=self.info.name,
            url=url or (resp.url if resp else None),
            version=version,
            query=query,
            cached=bool(resp and resp.cached),
            license=self.info.license,
            citation=self.info.citation,
            sha256=resp.sha256 if resp else None,
        )

    # -- required by every connector ----------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """A cheap real call that proves the endpoint still returns the data we expect.

        Return ``{"ok": True, "detail": "..."}`` or raise ImmunoError. It must validate the shape
        of the data, not just the status code.
        """
        raise NotImplementedError(f"{type(self).__name__}.health() is not implemented")

    async def resolve(self, kind: str, value: str) -> ResolvedIdentifier | None:
        """Confirm an identifier against the source. None means 'not mine' or 'not found'."""
        return None


# ---- registry ----------------------------------------------------------------------------------

CONNECTORS: dict[str, type[Connector]] = {}


def register(cls: type[Connector]) -> type[Connector]:
    """Class decorator. Importing the module is enough to register the connector."""
    name = cls.info.name
    if name in CONNECTORS and CONNECTORS[name] is not cls:
        raise RuntimeError(f"duplicate connector name {name!r}")
    CONNECTORS[name] = cls
    return cls
