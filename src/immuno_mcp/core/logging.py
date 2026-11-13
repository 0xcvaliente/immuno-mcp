"""Logging that goes to stderr (stdout carries the MCP protocol) and never prints secrets."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable

_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bjf_live_[A-Za-z0-9_-]+"),
    re.compile(r"\bhf_[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|authorization)=)[^&\s]+"),
]


class SecretRedactionFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s and len(s) >= 6]

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[redacted]")
        for pat in _PATTERNS:
            text = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "[redacted]", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redact(record.getMessage())
        record.args = ()
        return True


def configure_logging(secrets: Iterable[str] = (), level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(SecretRedactionFilter(secrets))
    root = logging.getLogger("immuno_mcp")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
