"""A small on-disk response cache (SQLite, standard library only)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class DiskCache:
    """Key to bytes store with a per-entry time to live. Pass ``":memory:"`` for tests."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "key TEXT PRIMARY KEY, value BLOB NOT NULL, meta TEXT NOT NULL, "
            "created REAL NOT NULL, ttl REAL NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str, *, now: float | None = None) -> tuple[bytes, dict[str, Any]] | None:
        t = time.time() if now is None else now
        with self._lock:
            row = self._db.execute(
                "SELECT value, meta, created, ttl FROM entries WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, meta, created, ttl = row
            if t - created > ttl:
                self._db.execute("DELETE FROM entries WHERE key = ?", (key,))
                self._db.commit()
                return None
            return bytes(value), json.loads(meta)

    def put(
        self,
        key: str,
        value: bytes,
        meta: dict[str, Any],
        ttl: float,
        *,
        now: float | None = None,
    ) -> None:
        t = time.time() if now is None else now
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO entries (key, value, meta, created, ttl) VALUES (?,?,?,?,?)",
                (key, value, json.dumps(meta), t, ttl),
            )
            self._db.commit()

    def purge_expired(self, *, now: float | None = None) -> int:
        t = time.time() if now is None else now
        with self._lock:
            cur = self._db.execute("DELETE FROM entries WHERE ? - created > ttl", (t,))
            self._db.commit()
            return cur.rowcount

    def stats(self) -> dict[str, int]:
        with self._lock:
            n, size = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(value)), 0) FROM entries"
            ).fetchone()
        return {"entries": int(n), "bytes": int(size)}

    def close(self) -> None:
        with self._lock:
            self._db.close()
