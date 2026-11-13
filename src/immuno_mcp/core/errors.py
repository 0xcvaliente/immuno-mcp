"""Typed errors. Every failure a tool can report has a stable code the model can act on."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from .fence import sanitize_message


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"  # the caller's arguments are wrong; fix them
    NOT_FOUND = "NOT_FOUND"  # the record does not exist upstream
    UPSTREAM_ERROR = "UPSTREAM_ERROR"  # the source failed (5xx, network)
    UPSTREAM_MOVED = "UPSTREAM_MOVED"  # the endpoint now redirects to a different site
    NON_DATA_RESPONSE = "NON_DATA_RESPONSE"  # got a web page or text error where data was expected
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    NEEDS_KEY = "NEEDS_KEY"  # an API key or token is missing or rejected
    NEEDS_EXTRA = "NEEDS_EXTRA"  # an optional dependency is not installed
    NOT_CONFIGURED = "NOT_CONFIGURED"  # the feature exists but is switched off
    PROFILE_DENIED = "PROFILE_DENIED"  # the active profile does not allow this
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"  # a paid action needs explicit confirmation
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    TOO_LARGE = "TOO_LARGE"
    INTERNAL = "INTERNAL"


_RETRYABLE = frozenset({ErrorCode.UPSTREAM_ERROR, ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT})


class ImmunoError(Exception):
    """A failure the caller can understand and react to."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        hint: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = sanitize_message(message, 600)
        self.hint = sanitize_message(hint, 400) if hint else None
        self.retryable = (code in _RETRYABLE) if retryable is None else retryable
        self.details = details or {}
        super().__init__(f"[{code}] {self.message}")

    def to_message(self) -> str:
        """The one-line text a model sees."""
        parts = [f"[{self.code}] {self.message}"]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if self.retryable:
            parts.append("Retryable.")
        return " ".join(parts)
