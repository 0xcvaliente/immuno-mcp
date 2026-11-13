"""Live contract check: every connector's `health()` must pass against the real service.

Run with:  uv run pytest -m live
A failure here means an endpoint moved or changed shape. That is exactly what this catches
before a wrong answer reaches a user.
"""

from __future__ import annotations

import pytest

from immuno_mcp.app import build_app
from immuno_mcp.core.connector import CONNECTORS
from immuno_mcp.core.errors import ImmunoError
from immuno_mcp.core.settings import Settings

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    settings = Settings(home=tmp_path_factory.mktemp("home"), profile="lab")
    return build_app(settings, in_memory=True)


@pytest.mark.parametrize("name", sorted(CONNECTORS))
async def test_connector_health_is_green(app, name):
    connector = app.connectors[name]
    if not connector.configured():
        pytest.skip(f"{name} needs {connector.info.auth_env}")
    try:
        result = await connector.health()
    except ImmunoError as exc:
        pytest.fail(f"{name}: {exc.to_message()}")
    assert result["ok"] is True, result
