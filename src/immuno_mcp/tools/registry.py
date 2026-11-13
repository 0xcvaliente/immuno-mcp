"""Tool registration: profile gating, a catalogue of every tool, and one error guard."""

from __future__ import annotations

import functools
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mcp import types
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from ..app import AppContext
from ..core.errors import ImmunoError
from ..core.settings import PROFILE_RANK, profile_allows

log = logging.getLogger("immuno_mcp.tools")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    group: str
    profile: str  # lowest profile that exposes the tool
    tier: str
    available: bool  # exposed under the ACTIVE profile
    read_only: bool
    open_world: bool
    spends_money: bool
    summary: str


def guarded(
    fn: Callable[..., Awaitable[Any]], app: AppContext | None = None
) -> Callable[..., Awaitable[Any]]:
    """Turn every failure into a message the model can act on, and never leak raw upstream text.

    Also records each successful result's provenance in the app's session log.
    """

    @functools.wraps(fn)
    async def inner(*args: Any, **kwargs: Any) -> Any:
        try:
            result = await fn(*args, **kwargs)
            if app is not None:
                app.note_result(result)
            return result
        except ToolError:
            raise
        except ImmunoError as exc:
            raise ToolError(exc.to_message()) from exc
        except ValidationError as exc:
            errors = exc.errors()
            first: dict[str, Any] = dict(errors[0]) if errors else {}
            where = ".".join(str(p) for p in first.get("loc", ()))
            raise ToolError(
                f"[INVALID_INPUT] {where}: {first.get('msg', 'invalid value')}"
            ) from exc
        except Exception as exc:
            error_id = uuid.uuid4().hex[:8]
            log.exception("tool %s failed (id %s)", getattr(fn, "__name__", "?"), error_id)
            raise ToolError(
                f"[INTERNAL] Unexpected error (id {error_id}). Details are in the server log."
            ) from exc

    return inner


class ToolRegistrar:
    def __init__(self, server: MCPServer, app: AppContext) -> None:
        self.server = server
        self.app = app
        self.catalog: list[ToolSpec] = []

    def tool(
        self,
        *,
        group: str,
        name: str | None = None,
        profile: str = "readonly",
        read_only: bool = True,
        open_world: bool = False,
        idempotent: bool = True,
        destructive: bool = False,
        spends_money: bool = False,
        tier: str = "P0",
        title: str | None = None,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        if profile not in PROFILE_RANK:
            raise ValueError(f"unknown profile {profile!r}")
        if spends_money and profile != "compute":
            raise ValueError("a tool that spends money must require the 'compute' profile")

        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            tool_name = name or fn.__name__
            available = profile_allows(self.app.settings.profile, profile)
            doc = (fn.__doc__ or "").strip()
            self.catalog.append(
                ToolSpec(
                    name=tool_name,
                    group=group,
                    profile=profile,
                    tier=tier,
                    available=available,
                    read_only=read_only,
                    open_world=open_world,
                    spends_money=spends_money,
                    summary=doc.split("\n", 1)[0] if doc else "",
                )
            )
            if available:
                annotations = types.ToolAnnotations(
                    read_only_hint=read_only,
                    destructive_hint=None if read_only else destructive,
                    idempotent_hint=idempotent,
                    open_world_hint=open_world,
                )
                self.server.add_tool(
                    guarded(fn, self.app), name=tool_name, title=title, annotations=annotations
                )
            return fn

        return decorator
