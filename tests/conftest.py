"""Shared fixtures. Unit tests never touch the network: they use httpx.MockTransport."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from immuno_mcp.app import AppContext, build_app
from immuno_mcp.core.cache import DiskCache
from immuno_mcp.core.http import HttpClient
from immuno_mcp.core.settings import Settings
from immuno_mcp.server import BuiltServer, build_server

Handler = Callable[[httpx.Request], httpx.Response]

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home", profile="lab")


@pytest.fixture
def make_http(settings: Settings):
    """Build an HttpClient whose network is a handler function."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def _make(handler: Handler, *, cache: bool = False, **settings_kw: object) -> HttpClient:
        from dataclasses import replace

        s = replace(settings, **settings_kw) if settings_kw else settings
        client = HttpClient(
            s,
            cache=DiskCache(":memory:") if cache else None,
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
        )
        client.sleeps = sleeps  # type: ignore[attr-defined]
        return client

    return _make


@pytest.fixture
def make_app(settings: Settings):
    def _make(handler: Handler | None = None, **settings_kw: object) -> AppContext:
        from dataclasses import replace

        s = replace(settings, **settings_kw) if settings_kw else settings
        transport = httpx.MockTransport(handler) if handler else None
        return build_app(s, transport=transport, in_memory=True)

    return _make


@pytest.fixture
def make_server(settings: Settings):
    def _make(handler: Handler | None = None, **settings_kw: object) -> BuiltServer:
        from dataclasses import replace

        s = replace(settings, **settings_kw) if settings_kw else settings
        transport = httpx.MockTransport(handler) if handler else None
        return build_server(s, transport=transport, in_memory=True, close_app_on_exit=False)

    return _make
