"""PDBe REST API: entry summaries, molecules and SIFTS best structures for a UniProt accession.

Endpoints (checked 2026-11-13) under https://www.ebi.ac.uk/pdbe/api :
/mappings/best_structures/{uniprot}, /pdb/entry/summary/{id}, /pdb/entry/molecules/{id}.
Every answer is keyed by the requested id. An unknown id answers HTTP 404 with
``{"message": "Requested endpoint does not contain any data"}``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from immuno_mcp.connectors.rcsb import normalize_pdb_id
from immuno_mcp.connectors.uniprot import ACCESSION_RE
from immuno_mcp.core.connector import Connector, ConnectorInfo, register
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.fence import clean_tree
from immuno_mcp.core.http import HttpResponse

BASE_URL = "https://www.ebi.ac.uk/pdbe/api"
_BEST_FIELDS = (
    "chain_id",
    "experimental_method",
    "resolution",
    "unp_start",
    "unp_end",
    "coverage",
    "tax_id",
)


@register
class PdbeConnector(Connector):
    info: ClassVar[ConnectorInfo] = ConnectorInfo(
        name="pdbe",
        display_name="PDBe (Protein Data Bank in Europe)",
        kind="structure",
        base_url=BASE_URL,
        homepage="https://www.ebi.ac.uk/pdbe",
        license=(
            "EMBL-EBI Terms of Use (no restriction beyond the data owner's); PDB archive data "
            "are CC0 1.0 under wwPDB policy. Attribution expected."
        ),
        citation=(
            "Armstrong D.R. et al. (2020) PDBe: improved findability of macromolecular structure "
            "data in the PDB. Nucleic Acids Res. 48:D335-D343. doi:10.1093/nar/gkz990"
        ),
        rps=2.0,
        cache_ttl=24 * 3600,
        notes="SIFTS UniProt-to-PDB mappings; RCSB resolves PDB ids, not this connector.",
    )

    async def _keyed(self, path: str, key: str, missing: str) -> tuple[list[Any], HttpResponse]:
        try:
            data, resp = await self.http.get_json(
                self.info.name, f"{BASE_URL}/{path}", rps=self.info.rps, ttl=self.info.cache_ttl
            )
        except ImmunoError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                raise ImmunoError(ErrorCode.NOT_FOUND, f"pdbe: {missing}") from exc
            raise
        if not isinstance(data, dict):
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "pdbe: answer is not a JSON object")
        value = next((v for k, v in data.items() if k.lower() == key.lower()), None)
        if not isinstance(value, list):
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, f"pdbe: answer is not keyed by {key}")
        return clean_tree(value), resp

    async def best_structures(self, accession: str) -> tuple[list[dict[str, Any]], HttpResponse]:
        """PDB chains covering a UniProt accession, best first (SIFTS ranking)."""
        acc = accession.strip().upper()
        if not ACCESSION_RE.match(acc):
            raise ImmunoError(ErrorCode.INVALID_INPUT, f"{accession!r} is not a UniProt accession")
        rows, resp = await self._keyed(
            f"mappings/best_structures/{acc}", acc, f"no PDB structures mapped to {acc}"
        )
        return [
            {"pdb_id": str(r.get("pdb_id", "")).upper(), **{k: r.get(k) for k in _BEST_FIELDS}}
            for r in rows
            if isinstance(r, dict)
        ], resp

    async def entry_summary(self, pdb_id: str) -> tuple[dict[str, Any], HttpResponse]:
        pid = normalize_pdb_id(pdb_id).lower()
        rows, resp = await self._keyed(f"pdb/entry/summary/{pid}", pid, f"no PDB entry {pid}")
        if not rows or not isinstance(rows[0], dict):
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "pdbe: empty entry summary")
        return rows[0], resp

    async def molecules(self, pdb_id: str) -> tuple[list[dict[str, Any]], HttpResponse]:
        pid = normalize_pdb_id(pdb_id).lower()
        rows, resp = await self._keyed(f"pdb/entry/molecules/{pid}", pid, f"no PDB entry {pid}")
        keep = ("entity_id", "molecule_type", "molecule_name", "in_chains", "length")
        return [{k: r.get(k) for k in keep} for r in rows if isinstance(r, dict)], resp

    async def health(self) -> dict[str, Any]:
        data, _ = await self.http.get_json(
            self.info.name, f"{BASE_URL}/pdb/entry/summary/7kn5", rps=self.info.rps
        )
        rows = data.get("7kn5") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows or not rows[0].get("title"):
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "pdbe: summary for 7kn5 has no title")
        return {"ok": True, "detail": f"7kn5 released {rows[0].get('release_date')}"}
