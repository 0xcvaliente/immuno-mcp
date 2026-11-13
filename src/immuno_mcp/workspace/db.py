"""SQLite workspace. One file per user; safe to copy into a thesis archive.

The score ledger is the heart of it: every number is stored with the tool, version,
parameters and job that produced it, so a results table can always say where each value came
from.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    name TEXT,
    kind TEXT NOT NULL,            -- vhh | fab | scfv | igg | antigen | peptide | other
    chains TEXT NOT NULL,          -- JSON object {"chain_id": "SEQUENCE"}
    target TEXT,
    parent_id TEXT,
    note TEXT,
    project TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,           -- sha256 of the content
    path TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    mime TEXT,
    kind TEXT,                     -- pdb | mmcif | fasta | json | csv | other
    source TEXT,
    job_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,           -- local id
    provider TEXT NOT NULL,
    remote_id TEXT,
    kind TEXT NOT NULL,            -- prediction | design | embedding
    model TEXT,
    status TEXT NOT NULL,
    request TEXT,                  -- JSON, never contains keys
    estimate_usd REAL,
    actual_usd REAL,
    error TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    direction TEXT,                -- higher_better | lower_better | NULL
    meaning TEXT,                  -- what the number is and is not
    source TEXT NOT NULL,          -- connector, local tool or job provider
    tool TEXT,
    tool_version TEXT,
    params TEXT,                   -- JSON
    job_id TEXT,
    artifact_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scores_candidate ON scores(candidate_id, metric);
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    job_id TEXT,
    provider TEXT NOT NULL,
    usd REAL NOT NULL,
    kind TEXT NOT NULL,            -- estimate | actual
    created_at TEXT NOT NULL
);
"""

_JSON_COLUMNS = {"chains", "request", "params"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def candidate_id(kind: str, chains: dict[str, str]) -> str:
    """Deterministic id, so adding the same molecule twice finds the same row."""
    norm = json.dumps({k: v.strip().upper() for k, v in sorted(chains.items())}, sort_keys=True)
    return "c_" + hashlib.sha256(f"{kind}|{norm}".encode()).hexdigest()[:10]


class Workspace:
    def __init__(self, path: Path | str, artifacts_dir: Path | str | None = None) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ---- plumbing ---------------------------------------------------------------------------

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        for col in _JSON_COLUMNS & out.keys():
            if out[col] is not None:
                out[col] = json.loads(out[col])
        return out

    def _rows(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [self._row(r) for r in self._db.execute(sql, args).fetchall()]  # type: ignore[misc]

    def _exec(self, sql: str, args: tuple[Any, ...] = ()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid or 0

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- candidates -------------------------------------------------------------------------

    def add_candidate(
        self,
        *,
        kind: str,
        chains: dict[str, str],
        name: str | None = None,
        target: str | None = None,
        parent_id: str | None = None,
        note: str | None = None,
        project: str = "default",
    ) -> dict[str, Any]:
        cid = candidate_id(kind, chains)
        existing = self.get_candidate(cid)
        if existing:
            return existing
        self._exec(
            "INSERT INTO candidates (id,name,kind,chains,target,parent_id,note,project,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (cid, name, kind, json.dumps(chains), target, parent_id, note, project, now_iso()),
        )
        return self.get_candidate(cid)  # type: ignore[return-value]

    def get_candidate(self, candidate: str) -> dict[str, Any] | None:
        rows = self._rows(
            "SELECT * FROM candidates WHERE id = ? OR name = ?", (candidate, candidate)
        )
        return rows[0] if rows else None

    def list_candidates(
        self,
        *,
        project: str | None = None,
        kind: str | None = None,
        target: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where, args = [], []
        for col, val in (("project", project), ("kind", kind), ("target", target)):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        return self._rows(
            f"SELECT * FROM candidates {clause} ORDER BY created_at, id LIMIT ? OFFSET ?",
            (*args, limit, offset),
        )

    # ---- scores -----------------------------------------------------------------------------

    def add_score(
        self,
        *,
        candidate_id: str,
        metric: str,
        value: float,
        source: str,
        unit: str | None = None,
        direction: str | None = None,
        meaning: str | None = None,
        tool: str | None = None,
        tool_version: str | None = None,
        params: dict[str, Any] | None = None,
        job_id: str | None = None,
        artifact_id: str | None = None,
    ) -> int:
        if not self.get_candidate(candidate_id):
            raise KeyError(f"unknown candidate {candidate_id}")
        return self._exec(
            "INSERT INTO scores (candidate_id,metric,value,unit,direction,meaning,source,tool,"
            "tool_version,params,job_id,artifact_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                candidate_id,
                metric,
                float(value),
                unit,
                direction,
                meaning,
                source,
                tool,
                tool_version,
                json.dumps(params) if params else None,
                job_id,
                artifact_id,
                now_iso(),
            ),
        )

    def scores_for(
        self, candidate_ids: list[str] | None = None, metric: str | None = None
    ) -> list[dict[str, Any]]:
        where, args = [], []
        if candidate_ids:
            where.append(f"candidate_id IN ({','.join('?' * len(candidate_ids))})")
            args += candidate_ids
        if metric:
            where.append("metric = ?")
            args.append(metric)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        return self._rows(
            f"SELECT * FROM scores {clause} ORDER BY candidate_id, metric, id", tuple(args)
        )

    # ---- artifacts --------------------------------------------------------------------------

    def store_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        suffix: str,
        mime: str | None = None,
        source: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Save content under its own sha256 and register it. Same content is stored once."""
        if self.artifacts_dir is None:
            raise RuntimeError("this workspace has no artifacts directory")
        digest = hashlib.sha256(data).hexdigest()
        path = self.artifacts_dir / digest[:2] / f"{digest}{suffix}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        return self.register_artifact(
            path, digest, len(data), kind=kind, mime=mime, source=source, job_id=job_id
        )

    def register_artifact(
        self,
        path: Path,
        sha256: str,
        size: int,
        *,
        kind: str,
        mime: str | None = None,
        source: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get_artifact(sha256)
        if existing:
            return existing
        self._exec(
            "INSERT INTO artifacts (id,path,bytes,mime,kind,source,job_id,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (sha256, str(path), size, mime, kind, source, job_id, now_iso()),
        )
        return self.get_artifact(sha256)  # type: ignore[return-value]

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        return rows[0] if rows else None

    # ---- jobs and spend ---------------------------------------------------------------------

    def add_job(
        self,
        *,
        local_id: str,
        provider: str,
        kind: str,
        model: str | None,
        request: dict[str, Any],
        estimate_usd: float | None,
        session_id: str,
        status: str = "created",
        remote_id: str | None = None,
    ) -> dict[str, Any]:
        ts = now_iso()
        self._exec(
            "INSERT INTO jobs (id,provider,remote_id,kind,model,status,request,estimate_usd,"
            "session_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                local_id,
                provider,
                remote_id,
                kind,
                model,
                status,
                json.dumps(request),
                estimate_usd,
                session_id,
                ts,
                ts,
            ),
        )
        return self.get_job(local_id)  # type: ignore[return-value]

    def update_job(self, local_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {"remote_id", "status", "actual_usd", "error"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update {sorted(bad)}")
        if fields:
            sets = ", ".join(f"{k} = ?" for k in fields)
            self._exec(
                f"UPDATE jobs SET {sets}, updated_at = ? WHERE id = ?",
                (*fields.values(), now_iso(), local_id),
            )
        job = self.get_job(local_id)
        if job is None:
            raise KeyError(f"unknown job {local_id}")
        return job

    def get_job(self, local_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM jobs WHERE id = ? OR remote_id = ?", (local_id, local_id))
        return rows[0] if rows else None

    def list_jobs(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if status:
            return self._rows(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        return self._rows("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))

    def record_spend(
        self, *, session_id: str, provider: str, usd: float, kind: str, job_id: str | None = None
    ) -> None:
        self._exec(
            "INSERT INTO spend (session_id,job_id,provider,usd,kind,created_at) VALUES (?,?,?,?,?,?)",
            (session_id, job_id, provider, usd, kind, now_iso()),
        )

    def spend_total(self, session_id: str | None = None) -> float:
        """Committed spend: the latest figure per job, preferring actual over estimate."""
        sql = "SELECT job_id, kind, usd FROM spend" + (
            " WHERE session_id = ?" if session_id else ""
        )
        rows = self._rows(sql + " ORDER BY id", (session_id,) if session_id else ())
        per_job: dict[str, dict[str, float]] = {}
        for r in rows:
            per_job.setdefault(r["job_id"] or f"_{id(r)}", {})[r["kind"]] = r["usd"]
        return sum(v.get("actual", v.get("estimate", 0.0)) for v in per_job.values())
