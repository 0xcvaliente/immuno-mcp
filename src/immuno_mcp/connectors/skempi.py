"""SKEMPI 2.0: measured binding affinity changes upon mutation, as a locally cached CSV.

Verified on 2026-11-13: the CSV at ``SKEMPI_URL`` is 1,602,208 bytes, ``;``-separated, 29 columns,
7,085 rows, header line starting ``#Pdb``. The licence page (/info/terms) is CC BY 4.0.

Facts about the columns that this module relies on (read from the real file and the SKEMPI FAQ):

* ``#Pdb`` is ``<PDB id>_<chains of protein 1>_<chains of protein 2>``, for example ``1VFB_AB_C``.
* Mutations look like ``HA30A``: wild-type residue, chain, residue number (+ insertion code),
  mutant residue; several are comma separated. ``Mutation(s)_PDB`` uses PDB numbering.
* ``Affinity_*`` columns are Kd in M as reported (``n.b``/``n.b.`` = no binding, ``unf``,
  ``>1E-04``, ``<5E-11``, ``~2.2E-14``). ``Affinity_*_parsed`` holds the number only, so for a
  bound such as ``>1E-04`` the parsed value is the bound itself.
* ``Temperature`` is in K; ``298(assumed)`` means SKEMPI assumed 298 K; a few rows are empty.
* ``Hold_out_type`` labels complexes ``AB/AG`` (antibody-antigen), ``Pr/PI`` or ``TCR/pMHC``.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from immuno_mcp.core.connector import Connector, ConnectorInfo, register
from immuno_mcp.core.envelope import Provenance
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.fence import clean_text, sanitize_message
from immuno_mcp.core.http import HttpClient, HttpResponse
from immuno_mcp.core.settings import Settings

SKEMPI_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
SKEMPI_VERSION = "2.0 (CSV released 2018-06-06)"
R_KCAL = 0.0019872  # gas constant, kcal/mol/K
DEFAULT_T = 298.0
DDG_FORMULA = (
    "ddG = R*T*ln(Kd_mut/Kd_wt) in kcal/mol, R = 0.0019872 kcal/mol/K, T from the Temperature "
    "column (298 K when missing or unparsable). Positive ddG = weaker binding after mutation."
)
REQUIRED_COLUMNS = [
    "#Pdb",
    "Mutation(s)_PDB",
    "Mutation(s)_cleaned",
    "iMutation_Location(s)",
    "Hold_out_type",
    "Affinity_mut (M)",
    "Affinity_mut_parsed",
    "Affinity_wt (M)",
    "Affinity_wt_parsed",
    "Reference",
    "Protein 1",
    "Protein 2",
    "Temperature",
    "Method",
]
ANTIBODY_NAME = re.compile(r"antibod|\bfab|\bfv\b|scfv|nanobod|\bvhh|immunoglobulin(?!-like)", re.I)
ANTIBODY_RULE = (
    "antibody_only is a heuristic: a row is kept when SKEMPI labels the complex AB/AG "
    "(antibody-antigen, its Hold_out_type) or a protein name contains antibody, Fab, Fv, scFv, "
    "nanobody, VHH or immunoglobulin (case-insensitive; 'immunoglobulin-like' is not counted). "
    "It can miss antibodies named otherwise and keep rows that are not antibody-antigen binding."
)
_COMPLEX = re.compile(r"^([0-9][A-Za-z0-9]{3})_([A-Za-z0-9]+)_([A-Za-z0-9]+)$")
_NUMBER = re.compile(r"^\d+(\.\d+)?([eE][-+]?\d+)?$")


# ---- pure parsing and arithmetic ----------------------------------------------------------------


def check_header(text: str) -> list[str]:
    """The header columns, or NON_DATA_RESPONSE when this is not the SKEMPI table."""
    first = text.split("\n", 1)[0].rstrip("\r")
    header = first.split(";")
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if not first.startswith("#Pdb") or missing:
        raise ImmunoError(
            ErrorCode.NON_DATA_RESPONSE,
            f"skempi: not the SKEMPI CSV (missing {missing or ['#Pdb first']}): "
            f"{sanitize_message(first, 120)}",
        )
    return header


def parse_skempi(text: str) -> tuple[list[dict[str, str]], list[int]]:
    """Rows as {column: value}, plus the line numbers of rows that were malformed and skipped."""
    header = check_header(text)
    reader = csv.reader(io.StringIO(text), delimiter=";")
    next(reader)
    rows: list[dict[str, str]] = []
    skipped: list[int] = []
    for rec in reader:
        if not any(c.strip() for c in rec):
            continue
        if len(rec) != len(header) or not _COMPLEX.match(rec[0].strip()):
            skipped.append(reader.line_num)
            continue
        rows.append({h: clean_text(v.strip()) for h, v in zip(header, rec, strict=True)})
    return rows, skipped


def affinity(raw: str, parsed: str) -> tuple[float | None, str | None]:
    """(Kd in M or None, qualifier '<' '>' '~' or None)."""
    head = raw.strip()[:1]
    qualifier = head if head in ("<", ">", "~") else None
    try:
        value = float(parsed) if parsed else None
    except ValueError:
        value = None
    if value is not None and not (value > 0 and math.isfinite(value)):
        value = None
    return value, qualifier


def temperature(raw: str) -> tuple[float, str | None]:
    """(T in K, note). Missing or implausible values fall back to 298 K and say so."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)", raw or "")
    if m and 250.0 <= float(m.group(1)) <= 400.0:
        note = "298 K assumed by SKEMPI" if "assumed" in raw.lower() else None
        return float(m.group(1)), note
    return DEFAULT_T, "temperature missing or unparsable; 298 K used"


def ddg(
    kd_mut: float | None, q_mut: str | None, kd_wt: float | None, q_wt: str | None, t: float
) -> tuple[float | None, str | None, str | None]:
    """(ddG kcal/mol, qualifier, note). A bound on either Kd turns ddG into a bound."""
    if kd_mut is None or kd_wt is None:
        return None, None, "no numeric Kd for the mutant or the wild type"
    directions = set()
    if q_mut in (">", "<"):
        directions.add(q_mut)
    if q_wt == "<":
        directions.add(">")
    elif q_wt == ">":
        directions.add("<")
    if len(directions) > 1:
        return None, None, "the two Kd bounds point in opposite directions; ddG is undetermined"
    value = round(R_KCAL * t * math.log(kd_mut / kd_wt), 2)
    if directions:
        q = directions.pop()
        return (
            value,
            q,
            f"Kd reported as a bound, so ddG is {'at least' if q == '>' else 'at most'} this",
        )
    if "~" in (q_mut, q_wt):
        return value, "~", "Kd reported as approximate"
    return value, None, None


def antibody_evidence(row: dict[str, str]) -> str | None:
    label = "AB/AG" in row.get("Hold_out_type", "")
    name = bool(
        ANTIBODY_NAME.search(row.get("Protein 1", ""))
        or ANTIBODY_NAME.search(row.get("Protein 2", ""))
    )
    if label and name:
        return "skempi_label+name"
    return "skempi_label" if label else ("name" if name else None)


def _number(raw: str) -> float | None:
    raw = raw.strip()
    return float(raw) if _NUMBER.match(raw) else None


def derive(row: dict[str, str]) -> dict[str, Any]:
    """One SKEMPI row as a compact record with Kd values, ddG and the temperature used."""
    m = _COMPLEX.match(row["#Pdb"])
    kd_mut, q_mut = affinity(row["Affinity_mut (M)"], row["Affinity_mut_parsed"])
    kd_wt, q_wt = affinity(row["Affinity_wt (M)"], row["Affinity_wt_parsed"])
    t, t_note = temperature(row["Temperature"])
    value, qualifier, note = ddg(kd_mut, q_mut, kd_wt, q_wt, t)
    if value is None and qualifier is None:
        unusable = [
            raw
            for raw, kd in ((row["Affinity_mut (M)"], kd_mut), (row["Affinity_wt (M)"], kd_wt))
            if kd is None
        ]
        if any(u.lower().startswith("n.b") for u in unusable):
            note = "no binding detected (n.b.); ddG not computed"
        elif unusable:
            note = f"Kd reported as {', '.join(repr(u) for u in unusable)}; ddG not computed"
    ref = row["Reference"]
    out: dict[str, Any] = {
        "complex": row["#Pdb"],
        "pdb": m.group(1).upper() if m else None,
        "mutations": row["Mutation(s)_PDB"],
        "location": row["iMutation_Location(s)"] or None,
        "protein_1": row["Protein 1"],
        "protein_2": row["Protein 2"],
        "kd_wt_M": kd_wt,
        "kd_mut_M": kd_mut,
        "ddg_kcal_mol": value,
        "temperature_K": t,
        "method": row["Method"] or None,
        "reference": f"PMID:{ref}" if ref.isdigit() else ref,
    }
    for key, raw, num in (
        ("kd_wt_reported", row["Affinity_wt (M)"], kd_wt),
        ("kd_mut_reported", row["Affinity_mut (M)"], kd_mut),
    ):
        if raw and (num is None or _number(raw) is None):
            out[key] = raw  # "n.b", ">1E-04" and the like, verbatim
    if qualifier:
        out["ddg_qualifier"] = qualifier
    if note:
        out["ddg_note"] = note
    if t_note:
        out["temperature_note"] = t_note
    return out


def details(row: dict[str, str]) -> dict[str, Any]:
    """The less used columns, for response_mode 'standard' and 'full'."""
    keep = {
        "mutations_cleaned": "Mutation(s)_cleaned",
        "hold_out_type": "Hold_out_type",
        "kon_mut": "kon_mut_parsed",
        "kon_wt": "kon_wt_parsed",
        "koff_mut": "koff_mut_parsed",
        "koff_wt": "koff_wt_parsed",
        "dH_mut_kcal_mol": "dH_mut (kcal mol^(-1))",
        "dH_wt_kcal_mol": "dH_wt (kcal mol^(-1))",
        "dS_mut_cal_mol_K": "dS_mut (cal mol^(-1) K^(-1))",
        "dS_wt_cal_mol_K": "dS_wt (cal mol^(-1) K^(-1))",
        "notes": "Notes",
        "skempi_version": "SKEMPI version",
    }
    out: dict[str, Any] = {}
    for key, column in keep.items():
        raw = row.get(column, "")
        if raw:
            num = _number(raw)
            out[key] = num if num is not None and key not in ("skempi_version",) else raw
    return out


def _mutation_tokens(cell: str) -> set[str]:
    tokens = {t.strip().upper() for t in cell.split(",") if t.strip()}
    # also index each mutation without its chain letter: HA30A -> H30A
    return tokens | {t[0] + t[2:] for t in tokens if len(t) > 3}


def matches(
    row: dict[str, str],
    *,
    pdb_id: str | None,
    protein: str | None,
    mutations: list[str] | None,
    antibody_only: bool,
) -> bool:
    if pdb_id:
        complex_id = row["#Pdb"].upper()
        if ("_" in pdb_id and complex_id != pdb_id) or (
            "_" not in pdb_id and complex_id[:4] != pdb_id
        ):
            return False
    names = (row["Protein 1"].lower(), row["Protein 2"].lower())
    if protein and protein not in names[0] and protein not in names[1]:
        return False
    if mutations:
        have = _mutation_tokens(row["Mutation(s)_PDB"]) | _mutation_tokens(
            row["Mutation(s)_cleaned"]
        )
        if not all(m in have for m in mutations):
            return False
    return not (antibody_only and antibody_evidence(row) is None)


# ---- connector -----------------------------------------------------------------------------------


@dataclass
class SkempiTable:
    rows: list[dict[str, str]]
    skipped_lines: list[int]
    meta: dict[str, Any]  # the sidecar: source_url, sha256, size, retrieved_at
    reused: bool  # True when the local copy was used without downloading
    notes: list[str] = field(default_factory=list)


@register
class SkempiConnector(Connector):
    info: ClassVar[ConnectorInfo] = ConnectorInfo(
        name="skempi",
        display_name="SKEMPI 2.0",
        kind="binding",
        base_url="https://life.bsc.es/pid/skempi2",
        homepage="https://life.bsc.es/pid/skempi2",
        license="CC BY 4.0 (SKEMPI 2.0 terms of download and use)",
        citation=(
            "Jankauskaitė J, Jiménez-García B, Dapkūnas J, Fernández-Recio J, Moal IH (2019) "
            "SKEMPI 2.0: an updated benchmark of changes in protein-protein binding energy, "
            "kinetics and thermodynamics upon mutation. Bioinformatics 35(3):462-469. "
            "doi:10.1093/bioinformatics/bty635"
        ),
        rps=1.0,
        cache_ttl=0,
        notes="Downloaded once (about 1.6 MB) to <home>/datasets/skempi and reused.",
        extra={"download_url": SKEMPI_URL, "version": SKEMPI_VERSION},
    )

    def __init__(self, http: HttpClient, settings: Settings) -> None:
        super().__init__(http, settings)
        self._lock = asyncio.Lock()
        self._table: SkempiTable | None = None

    @property
    def data_dir(self) -> Path:
        return self.settings.home / "datasets" / "skempi"

    @property
    def csv_path(self) -> Path:
        return self.data_dir / "skempi_v2.csv"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "skempi_v2.csv.meta.json"

    def local_copy(self) -> dict[str, Any] | None:
        """The sidecar of the local copy, or None when there is no usable copy. Never downloads."""
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            size = self.csv_path.stat().st_size
        except (OSError, ValueError):
            return None
        if not isinstance(meta, dict) or meta.get("size") != size or not meta.get("sha256"):
            return None
        return meta

    def provenance_for(self, table: SkempiTable) -> Provenance:
        prov = self.provenance(url=SKEMPI_URL, version=SKEMPI_VERSION)
        update: dict[str, Any] = {"sha256": table.meta["sha256"], "cached": table.reused}
        with contextlib.suppress(KeyError, TypeError, ValueError):
            update["retrieved_at"] = datetime.fromisoformat(table.meta["retrieved_at"])
        return prov.model_copy(update=update)

    async def table(self, *, refresh: bool = False) -> SkempiTable:
        """The parsed table: the local copy when it is intact, otherwise a fresh download."""
        async with self._lock:
            notes: list[str] = []
            meta = None if refresh else self.local_copy()
            if meta and self._table and self._table.meta.get("sha256") == meta["sha256"]:
                self._table.reused, self._table.notes = True, []
                return self._table
            if meta:
                content = self.csv_path.read_bytes()
                if hashlib.sha256(content).hexdigest() == meta["sha256"]:
                    rows, skipped = parse_skempi(content.decode("utf-8", errors="replace"))
                    self._table = SkempiTable(rows, skipped, meta, reused=True)
                    return self._table
                notes.append("The local copy did not match its recorded sha256; downloaded again.")
            self._table = await self._download()
            self._table.notes = notes
            return self._table

    async def _download(self) -> SkempiTable:
        part = self.data_dir / "skempi_v2.csv.download"
        try:
            _, sha, size = await self.http.download(
                SKEMPI_URL, part, source=self.info.name, expect_prefix=b"#Pdb", max_bytes=50 << 20
            )
            text = part.read_bytes().decode("utf-8", errors="replace")
            rows, skipped = parse_skempi(text)
            if not rows:
                raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "skempi: the CSV has no data rows")
            part.replace(self.csv_path)
        finally:
            part.unlink(missing_ok=True)
        meta = {
            "source_url": SKEMPI_URL,
            "version": SKEMPI_VERSION,
            "sha256": sha,
            "size": size,
            "rows": len(rows),
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "license": self.info.license,
        }
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tmp.replace(self.meta_path)
        return SkempiTable(rows, skipped, meta, reused=False)

    async def query(
        self,
        *,
        pdb_id: str | None = None,
        protein: str | None = None,
        mutations: list[str] | None = None,
        antibody_only: bool = False,
        refresh: bool = False,
    ) -> tuple[list[dict[str, str]], SkempiTable]:
        """All matching raw rows (file order) and the table they came from."""
        table = await self.table(refresh=refresh)
        pdb = pdb_id.strip().upper() if pdb_id else None
        prot = protein.strip().lower() if protein else None
        muts = [m.strip().upper() for m in mutations] if mutations else None
        found = [
            r
            for r in table.rows
            if matches(r, pdb_id=pdb, protein=prot, mutations=muts, antibody_only=antibody_only)
        ]
        return found, table

    async def health(self) -> dict[str, Any]:
        def is_skempi(resp: HttpResponse) -> None:
            check_header(resp.text)

        resp = await self.http.get_text(
            self.info.name, SKEMPI_URL, rps=self.info.rps, validate=is_skempi
        )
        rows, skipped = parse_skempi(resp.text)
        if not rows:
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "skempi: the CSV has no data rows")
        detail = f"{len(rows)} rows ({len(skipped)} malformed), sha256 {resp.sha256[:12]}"
        local = self.local_copy()
        if local:
            same = local["sha256"] == resp.sha256
            detail += "; local copy is current" if same else "; local copy differs from upstream"
        return {"ok": True, "detail": detail}
