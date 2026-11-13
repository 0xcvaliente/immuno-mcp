"""RCSB Protein Data Bank: entry metadata, polymer entities and coordinate files.

Endpoints (checked 2026-11-13):
* REST   https://data.rcsb.org/rest/v1/core/entry/{ID}        (404 JSON for an unknown id)
* GraphQL https://data.rcsb.org/graphql                       (entry plus all entities in one call;
  an unknown id answers 200 with ``{"data": {"entry": null}}``)
* Files  https://files.rcsb.org/download/{ID}.cif | .pdb      (404 HTML page for an unknown id; the
  legacy .pdb form does not exist for large entries, which also answer 404)
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from immuno_mcp.core.connector import Connector, ConnectorInfo, ResolvedIdentifier, register
from immuno_mcp.core.envelope import Provenance
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.fence import clean_tree, sanitize_message
from immuno_mcp.core.http import HttpResponse, RetryPolicy, looks_like_html

DATA_URL = "https://data.rcsb.org"
GRAPHQL_URL = f"{DATA_URL}/graphql"
FILES_URL = "https://files.rcsb.org/download"
FILE_TTL = 7 * 24 * 3600
MAX_FILE_BYTES = 150 * 1024 * 1024

_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_EXT_PDB_ID = re.compile(r"^pdb_[0-9a-z]{8}$", re.IGNORECASE)
# Record names that can open a legacy PDB-format file.
_PDB_RECORDS = frozenset(
    {"HEADER", "OBSLTE", "TITLE", "SPLIT", "CAVEAT", "COMPND", "SOURCE", "KEYWDS", "EXPDTA"}
    | {"AUTHOR", "REVDAT", "REMARK", "CRYST1", "MODEL", "ATOM", "HETATM", "SEQRES", "DBREF"}
)

ENTRY_QUERY = """query($id: String!) {
  entry(entry_id: $id) {
    rcsb_id
    struct { title }
    exptl { method }
    rcsb_entry_info { resolution_combined polymer_entity_count deposited_polymer_entity_instance_count }
    rcsb_accession_info { deposit_date initial_release_date revision_date }
    rcsb_primary_citation { title journal_abbrev year pdbx_database_id_DOI pdbx_database_id_PubMed rcsb_authors }
    polymer_entities {
      rcsb_polymer_entity_container_identifiers { entity_id auth_asym_ids uniprot_ids }
      rcsb_polymer_entity { pdbx_description }
      entity_poly { type rcsb_sample_sequence_length pdbx_seq_one_letter_code_can }
      rcsb_entity_source_organism { scientific_name ncbi_taxonomy_id }
    }
  }
}"""


def normalize_pdb_id(value: str) -> str:
    """Return the id RCSB serves: '7kn5' and 'pdb_00007kn5' both become '7KN5'.

    Extended ids that do not start with 'pdb_0000' (issued from 2027) are kept, lower case.
    """
    v = (value or "").strip()
    if _PDB_ID.match(v):
        return v.upper()
    if _EXT_PDB_ID.match(v):
        v = v.lower()
        return v[8:].upper() if v.startswith("pdb_0000") else v
    raise ImmunoError(
        ErrorCode.INVALID_INPUT,
        f"{value!r} is not a PDB id",
        hint="Use a 4-character id such as 7KN5 or the extended form pdb_00007kn5.",
    )


def _graphql_ok(resp: HttpResponse) -> None:
    """Reject a GraphQL error answer (HTTP 200 with "errors") before it is cached."""
    try:
        body = resp.json()
    except ValueError:
        return  # the JSON check in the HTTP layer reports this
    errors = body.get("errors") if isinstance(body, dict) else None
    if errors:
        first = errors[0].get("message") if isinstance(errors[0], dict) else errors[0]
        raise ImmunoError(
            ErrorCode.NON_DATA_RESPONSE,
            f"rcsb: GraphQL error: {sanitize_message(first, 240)}",
            hint="The RCSB data schema may have changed. Report it so the connector is updated.",
        )


def check_structure_bytes(content: bytes, fmt: str, pdb_id: str) -> None:
    """Raise NON_DATA_RESPONSE unless ``content`` starts like an mmCIF or PDB file."""
    if not content.strip():
        raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, f"rcsb: {pdb_id}.{fmt} is empty")
    if looks_like_html(content):
        raise ImmunoError(
            ErrorCode.NON_DATA_RESPONSE,
            f"rcsb: expected a {fmt} structure file for {pdb_id} but received a web page",
        )
    head = content[:4096]
    if fmt == "cif":
        lines = [ln.strip() for ln in head.splitlines()]
        first = next((ln for ln in lines if ln and not ln.startswith(b"#")), b"")
        ok = first.startswith(b"data_")
    else:
        first = head.lstrip().split(b"\n", 1)[0]
        ok = first[:6].strip().decode("ascii", "replace").upper() in _PDB_RECORDS
    if not ok:
        raise ImmunoError(
            ErrorCode.NON_DATA_RESPONSE,
            f"rcsb: {pdb_id}.{fmt} does not look like a {fmt} file: "
            f"{sanitize_message(head[:80].decode('utf-8', 'replace'), 80)}",
        )


@register
class RcsbConnector(Connector):
    info: ClassVar[ConnectorInfo] = ConnectorInfo(
        name="rcsb",
        display_name="RCSB Protein Data Bank",
        kind="structure",
        base_url=DATA_URL,
        homepage="https://www.rcsb.org",
        license=(
            "CC0 1.0 (wwPDB policy for PDB archive files; RCSB PDB states its API data are "
            "under the same licence)"
        ),
        citation=(
            "Berman H.M. et al. (2000) The Protein Data Bank. Nucleic Acids Res. 28:235-242. "
            "doi:10.1093/nar/28.1.235"
        ),
        rps=2.0,
        cache_ttl=24 * 3600,
        resolves=frozenset({"pdb"}),
        notes="Entry metadata via REST and GraphQL; coordinates from files.rcsb.org.",
    )

    # ---- entry metadata --------------------------------------------------------------------

    async def entry(self, pdb_id: str) -> tuple[dict[str, Any], HttpResponse]:
        """The REST core entry document (large; project what you need)."""
        pid = normalize_pdb_id(pdb_id)
        try:
            data, resp = await self.http.get_json(
                self.info.name,
                f"{DATA_URL}/rest/v1/core/entry/{pid}",
                rps=self.info.rps,
                ttl=self.info.cache_ttl,
            )
        except ImmunoError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                raise _no_entry(pid) from exc
            raise
        if not isinstance(data, dict) or "rcsb_id" not in data:
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "rcsb: entry document has no rcsb_id")
        return clean_tree(data), resp

    async def entry_details(self, pdb_id: str) -> tuple[dict[str, Any], HttpResponse]:
        """Entry, primary citation and every polymer entity in one GraphQL call (projected)."""
        pid = normalize_pdb_id(pdb_id)
        data, resp = await self.http.post_json(
            self.info.name,
            GRAPHQL_URL,
            json_body={"query": ENTRY_QUERY, "variables": {"id": pid}},
            rps=self.info.rps,
            ttl=self.info.cache_ttl,
            retry=RetryPolicy(retry_post=True),  # a read-only query; safe to repeat
            validate=_graphql_ok,
        )
        if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "rcsb: GraphQL answer has no 'data'")
        entry = data["data"].get("entry")
        if entry is None:
            raise _no_entry(pid)
        return project_entry(clean_tree(entry)), resp

    # ---- coordinate files ------------------------------------------------------------------

    async def fetch_structure_file(
        self, pdb_id: str, fmt: Literal["cif", "pdb"] = "cif"
    ) -> tuple[bytes, Provenance]:
        """Download coordinates. NOT_FOUND for an unknown id, NON_DATA_RESPONSE for a non-file."""
        if fmt not in ("cif", "pdb"):
            raise ImmunoError(
                ErrorCode.INVALID_INPUT, f"unknown format {fmt!r}", hint="Use 'cif' or 'pdb'."
            )
        pid = normalize_pdb_id(pdb_id)
        url = f"{FILES_URL}/{pid}.{fmt}"
        try:
            resp = await self.http.request(
                "GET",
                url,
                source=self.info.name,
                expect="bytes",
                rps=self.info.rps,
                ttl=FILE_TTL,
                timeout=120.0,
                max_bytes=MAX_FILE_BYTES,
                validate=lambda r: check_structure_bytes(r.content, fmt, pid),
            )
        except ImmunoError as exc:
            if exc.code != ErrorCode.NOT_FOUND:
                raise
            if fmt == "pdb" and await self._entry_exists(pid):
                raise ImmunoError(
                    ErrorCode.NOT_FOUND,
                    f"rcsb: {pid} has no legacy PDB-format file (large entries are "
                    "distributed only as mmCIF)",
                    hint="Use fmt='cif'.",
                ) from exc
            raise _no_entry(pid) from exc
        return resp.content, self.provenance(resp, query={"pdb_id": pid, "format": fmt})

    async def _entry_exists(self, pid: str) -> bool:
        try:
            await self.entry(pid)
        except ImmunoError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                return False
            raise
        return True

    # ---- contract --------------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        data, _ = await self.http.get_json(
            self.info.name, f"{DATA_URL}/rest/v1/core/entry/7KN5", rps=self.info.rps
        )
        title = (data.get("struct") or {}).get("title") if isinstance(data, dict) else None
        if not isinstance(data, dict) or data.get("rcsb_id") != "7KN5" or not title:
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "rcsb: entry 7KN5 lacks rcsb_id or struct.title"
            )
        return {"ok": True, "detail": f"entry 7KN5: {sanitize_message(title, 60)}"}

    async def resolve(self, kind: str, value: str) -> ResolvedIdentifier | None:
        if kind != "pdb":
            return None
        try:
            pid = normalize_pdb_id(value)
            data, _ = await self.entry(pid)
        except ImmunoError as exc:
            if exc.code in (ErrorCode.NOT_FOUND, ErrorCode.INVALID_INPUT):
                return None
            raise
        info = data.get("rcsb_entry_info") or {}
        acc = data.get("rcsb_accession_info") or {}
        cit = data.get("rcsb_primary_citation") or {}
        summary = {
            "method": [e.get("method") for e in data.get("exptl") or [] if e.get("method")],
            "resolution": _first(info.get("resolution_combined")),
            "deposit_date": _date(acc.get("deposit_date")),
            "release_date": _date(acc.get("initial_release_date")),
            "polymer_entities": info.get("polymer_entity_count"),
            "doi": cit.get("pdbx_database_id_DOI"),
        }
        return ResolvedIdentifier(
            kind="pdb",
            id=value,
            canonical_id=data["rcsb_id"],
            title=(data.get("struct") or {}).get("title"),
            url=f"https://www.rcsb.org/structure/{data['rcsb_id']}",
            source=self.info.name,
            summary={k: v for k, v in summary.items() if v not in (None, [])},
        )


# ---- projection ----------------------------------------------------------------------------


def project_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Reduce the GraphQL entry to the fields a tool reports. Sequences stay under 'sequence'."""
    info = entry.get("rcsb_entry_info") or {}
    acc = entry.get("rcsb_accession_info") or {}
    cit = entry.get("rcsb_primary_citation") or None
    entities = []
    for ent in entry.get("polymer_entities") or []:
        ids = ent.get("rcsb_polymer_entity_container_identifiers") or {}
        poly = ent.get("entity_poly") or {}
        orgs = ent.get("rcsb_entity_source_organism") or []
        entities.append(
            {
                "entity_id": ids.get("entity_id"),
                "chains": ids.get("auth_asym_ids") or [],
                "molecule": (ent.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
                "type": poly.get("type"),
                "length": poly.get("rcsb_sample_sequence_length"),
                "organism": _first([o.get("scientific_name") for o in orgs]),
                "taxid": _first([o.get("ncbi_taxonomy_id") for o in orgs]),
                "uniprot": ids.get("uniprot_ids") or [],
                "sequence": poly.get("pdbx_seq_one_letter_code_can"),
            }
        )
    entities.sort(key=lambda e: int(e["entity_id"]) if str(e["entity_id"]).isdigit() else 0)
    citation = None
    if cit:
        authors = cit.get("rcsb_authors") or []
        citation = {
            "title": cit.get("title"),
            "journal": cit.get("journal_abbrev"),
            "year": cit.get("year"),
            "doi": cit.get("pdbx_database_id_DOI"),
            "pmid": cit.get("pdbx_database_id_PubMed"),
            "authors": (", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
            if authors
            else None,
        }
    return {
        "pdb_id": entry.get("rcsb_id"),
        "title": (entry.get("struct") or {}).get("title"),
        "method": [e.get("method") for e in entry.get("exptl") or [] if e.get("method")],
        "resolution": _first(info.get("resolution_combined")),
        "deposit_date": _date(acc.get("deposit_date")),
        "release_date": _date(acc.get("initial_release_date")),
        "revision_date": _date(acc.get("revision_date")),
        "polymer_entity_count": info.get("polymer_entity_count"),
        "chain_count": info.get("deposited_polymer_entity_instance_count"),
        "citation": citation,
        "entities": entities,
    }


def _no_entry(pid: str) -> ImmunoError:
    return ImmunoError(
        ErrorCode.NOT_FOUND,
        f"rcsb: no PDB entry {pid}",
        hint="Check the id with resolve_identifier. Do not use an id you could not resolve.",
    )


def _first(values: Any) -> Any:
    if isinstance(values, list):
        return next((v for v in values if v is not None), None)
    return values


def _date(value: Any) -> str | None:
    return value[:10] if isinstance(value, str) and value else None
