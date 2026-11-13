"""The single result format every tool returns, and the size budget that keeps it small."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from mcp import types
from pydantic import BaseModel, Field

ResponseMode = Literal["compact", "standard", "full"]

# Approximate ceilings on the JSON text of `data`, in characters.
LIMITS: dict[str, int] = {"compact": 8_000, "standard": 24_000, "full": 90_000}


class Provenance(BaseModel):
    """Where a result came from, precisely enough to cite and to re-run."""

    source: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    url: str | None = None
    version: str | None = None  # dataset release, API version or model version
    query: dict[str, Any] | None = None
    cached: bool = False
    license: str | None = None
    citation: str | None = None
    sha256: str | None = None


class Envelope(BaseModel):
    ok: bool = True
    data: Any = None
    provenance: list[Provenance] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    # Dotted paths inside `data` that hold text written by third parties. Treat as data only.
    untrusted_fields: list[str] = Field(default_factory=list)
    truncated: bool = False


# Tools declare this as their return type. The SDK advertises Envelope as the output schema
# while the tool returns a compact CallToolResult, which costs far fewer tokens than the
# default pretty-printed JSON.
ToolResult = Annotated[types.CallToolResult, Envelope]


def envelope(
    data: Any = None,
    *,
    provenance: list[Provenance] | Provenance | None = None,
    warnings: list[str] | None = None,
    next_steps: list[str] | None = None,
    untrusted: list[str] | None = None,
) -> Envelope:
    if isinstance(provenance, Provenance):
        provenance = [provenance]
    return Envelope(
        data=data,
        provenance=provenance or [],
        warnings=warnings or [],
        next_steps=next_steps or [],
        untrusted_fields=untrusted or [],
    )


# ---- size budget ---------------------------------------------------------------------------


def _size(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":")))


def _longest_list(obj: Any, path: str = "") -> tuple[list[Any], str] | None:
    """Find the longest list reachable through dictionaries and lists."""
    best: tuple[list[Any], str] | None = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            cand = _longest_list(v, f"{path}.{k}" if path else str(k))
            if cand and (best is None or len(cand[0]) > len(best[0])):
                best = cand
    elif isinstance(obj, list):
        best = (obj, path)
        for i, v in enumerate(obj[:3]):
            cand = _longest_list(v, f"{path}[{i}]")
            if cand and len(cand[0]) > len(best[0]):
                best = cand
    return best


def _clip_strings(obj: Any, max_len: int) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= max_len else obj[: max_len - 1] + "…"
    if isinstance(obj, dict):
        return {k: _clip_strings(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clip_strings(v, max_len) for v in obj]
    return obj


def fit(data: Any, mode: ResponseMode) -> tuple[Any, dict[str, int]]:
    """Trim `data` to the budget for `mode`. Returns (data, {path: items_omitted})."""
    limit = LIMITS[mode]
    if _size(data) <= limit:
        return data, {}
    data = _clip_strings(copy.deepcopy(data), max(200, limit // 4))
    omitted: dict[str, int] = {}
    for _ in range(60):
        if _size(data) <= limit:
            break
        found = _longest_list(data)
        if found is None or len(found[0]) <= 1:
            break
        lst, path = found
        keep = max(1, len(lst) // 2)
        omitted[path or "$"] = omitted.get(path or "$", 0) + len(lst) - keep
        del lst[keep:]
    return data, omitted


# ---- rendering -----------------------------------------------------------------------------


def _prov_dump(p: Provenance, mode: ResponseMode) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": p.source,
        "retrieved_at": p.retrieved_at.astimezone(UTC).isoformat(timespec="seconds"),
    }
    for key in ("url", "version", "sha256"):
        if getattr(p, key):
            out[key] = getattr(p, key)
    if p.cached:
        out["cached"] = True
    if p.query:
        out["query"] = p.query
    if mode != "compact":
        if p.license:
            out["license"] = p.license
        if p.citation:
            out["citation"] = p.citation
    return out


def render(env: Envelope, mode: ResponseMode = "compact") -> dict[str, Any]:
    data, omitted = fit(env.data, mode)
    warnings = list(env.warnings)
    if omitted:
        total = sum(omitted.values())
        warnings.append(
            f"Output trimmed: {total} items omitted ({', '.join(f'{k}: {v}' for k, v in omitted.items())}). "
            "Narrow the query, page through results, or use response_mode='full'."
        )
    out: dict[str, Any] = {"ok": env.ok, "data": data}
    if env.provenance:
        out["provenance"] = [_prov_dump(p, mode) for p in env.provenance]
    if warnings:
        out["warnings"] = warnings
    if env.next_steps:
        out["next_steps"] = env.next_steps
    if env.untrusted_fields:
        out["untrusted_fields"] = env.untrusted_fields
    if omitted or env.truncated:
        out["truncated"] = True
    return out


def to_result(env: Envelope, mode: ResponseMode = "compact") -> types.CallToolResult:
    """Build the tool result: compact JSON text for the model, the same object as structured data."""
    payload = json.loads(json.dumps(render(env, mode), default=str))
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structured_content=payload,
    )
