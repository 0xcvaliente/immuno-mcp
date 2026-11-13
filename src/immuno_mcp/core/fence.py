"""Text hygiene for anything that comes from outside the process.

Abstracts, PDB titles, dataset notes and API error bodies are written by other people.
They are data. Before they reach the model we strip the code points that let text hide
from a human reader (control, zero-width, bidirectional and Unicode "tag" characters).
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any

_FORBIDDEN: frozenset[int] = frozenset(
    {
        *range(0x00, 0x09),
        0x0B,
        0x0C,
        *range(0x0E, 0x20),
        *range(0x7F, 0xA0),
        0x200B,
        0x200C,
        0x200D,
        0x2060,
        0xFEFF,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
        *range(0xE0000, 0xE0080),  # invisible "tag" characters
    }
)


def clean_text(raw: str) -> str:
    """NFC-normalise and drop hidden code points. Keeps tab, newline and carriage return."""
    normalized = unicodedata.normalize("NFC", raw)
    return "".join(ch for ch in normalized if ord(ch) not in _FORBIDDEN)


def sanitize_message(text: object, max_chars: int = 280) -> str:
    """Make any string safe to show as an error or warning: cleaned, one line, length-capped."""
    clean = " ".join(clean_text(str(text)).split())
    if len(clean) > max_chars:
        return clean[: max_chars - 1] + "…"
    return clean


def clean_tree(obj: Any) -> Any:
    """Recursively apply `clean_text` to every string in a JSON-like structure."""
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, dict):
        return {(clean_text(k) if isinstance(k, str) else k): clean_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clean_tree(v) for v in obj)
    return obj


def sha256_hex(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()
