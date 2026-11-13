"""UniProtKB REST: protein entries, searches and accession resolution.

Endpoints (checked 2026-11-13): https://rest.uniprot.org/uniprotkb/{acc}?format=json&fields=...
and /uniprotkb/search?query=...&fields=...&format=json. Quirks handled here:

* A deleted or merged accession answers HTTP 200 with ``entryType: "Inactive"``. That is NOT_FOUND.
* A malformed accession answers HTTP 400 with a JSON ``messages`` list (INVALID_INPUT).
* The release is in the ``X-UniProt-Release`` header and the hit count in ``X-Total-Results``.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from immuno_mcp.core.connector import Connector, ConnectorInfo, ResolvedIdentifier, register
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.fence import clean_tree
from immuno_mcp.core.http import HttpResponse

BASE_URL = "https://rest.uniprot.org"
SUMMARY_FIELDS = "accession,id,protein_name,gene_names,organism_name,organism_id,length,reviewed"
ACCESSION_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$"
)


def is_accession(value: str) -> bool:
    return bool(ACCESSION_RE.match((value or "").strip().upper()))


def release(resp: HttpResponse | None) -> str | None:
    """UniProt release of a live response (cached responses carry no headers)."""
    return resp.headers.get("x-uniprot-release") if resp else None


def summarize(entry: dict[str, Any]) -> dict[str, Any]:
    """Project an entry to name, aliases, genes, organism and length."""
    desc = entry.get("proteinDescription") or {}
    rec = desc.get("recommendedName") or {}
    name = (rec.get("fullName") or {}).get("value")
    if not name:  # unreviewed entries carry submission names instead
        subs = desc.get("submissionNames") or []
        name = ((subs[0] if subs else {}).get("fullName") or {}).get("value")
    short = [s["value"] for s in rec.get("shortNames") or [] if s.get("value")]
    alt = [
        a["fullName"]["value"]
        for a in desc.get("alternativeNames") or []
        if (a.get("fullName") or {}).get("value")
    ]
    cd = [c["value"] for c in desc.get("cdAntigenNames") or [] if c.get("value")]
    genes: list[str] = []
    for g in entry.get("genes") or []:
        if (g.get("geneName") or {}).get("value"):
            genes.append(g["geneName"]["value"])
        genes.extend(s["value"] for s in g.get("synonyms") or [] if s.get("value"))
    org = entry.get("organism") or {}
    out = {
        "accession": entry.get("primaryAccession"),
        "entry_name": entry.get("uniProtkbId"),
        "name": name,
        "short_names": short + cd,
        "alt_names": alt[:5],
        "genes": genes,
        "organism": org.get("scientificName"),
        "taxid": org.get("taxonId"),
        "length": (entry.get("sequence") or {}).get("length"),
        "reviewed": str(entry.get("entryType", "")).startswith("UniProtKB reviewed"),
    }
    return {k: v for k, v in out.items() if v not in (None, [])}


@register
class UniprotConnector(Connector):
    info: ClassVar[ConnectorInfo] = ConnectorInfo(
        name="uniprot",
        display_name="UniProt Knowledgebase",
        kind="sequence",
        base_url=BASE_URL,
        homepage="https://www.uniprot.org",
        license="CC BY 4.0 (applied to all copyrightable parts of the UniProt databases)",
        citation=(
            "The UniProt Consortium (2025) UniProt: the Universal Protein Knowledgebase in 2025. "
            "Nucleic Acids Res. 53:D609-D617. doi:10.1093/nar/gkae1010"
        ),
        rps=2.0,
        cache_ttl=24 * 3600,
        resolves=frozenset({"uniprot"}),
    )

    async def entry(
        self, accession: str, *, fields: str | None = SUMMARY_FIELDS
    ) -> tuple[dict[str, Any], HttpResponse]:
        """One entry as JSON. NOT_FOUND for deleted or merged accessions."""
        acc = (accession or "").strip().upper()
        if not ACCESSION_RE.match(acc):
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"{accession!r} is not a UniProt accession",
                hint="Accessions look like Q15116 or A0A0B4J2F0 (isoforms add -2).",
            )
        params = {"format": "json", **({"fields": fields} if fields else {})}
        data, resp = await self.http.get_json(
            self.info.name,
            f"{BASE_URL}/uniprotkb/{acc}",
            params=params,
            rps=self.info.rps,
            ttl=self.info.cache_ttl,
        )
        if not isinstance(data, dict) or "primaryAccession" not in data:
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "uniprot: entry has no primaryAccession")
        if data.get("entryType") == "Inactive":
            reason = data.get("inactiveReason") or {}
            to = reason.get("mergeDemergeTo") or []
            raise ImmunoError(
                ErrorCode.NOT_FOUND,
                f"uniprot: {acc} is inactive ({reason.get('inactiveReasonType', 'unknown reason')})",
                hint=(f"Replaced by {', '.join(to)}." if to else "The entry was removed.")
                + " Search UniProt for the current accession.",
            )
        return clean_tree(data), resp

    async def search(
        self, query: str, *, fields: str = SUMMARY_FIELDS, size: int = 10
    ) -> tuple[list[dict[str, Any]], int | None, HttpResponse]:
        """Search UniProtKB. Returns (entries, total hits if reported, response)."""
        if not query.strip():
            raise ImmunoError(ErrorCode.INVALID_INPUT, "uniprot: empty query")
        data, resp = await self.http.get_json(
            self.info.name,
            f"{BASE_URL}/uniprotkb/search",
            params={"query": query, "fields": fields, "format": "json", "size": size},
            rps=self.info.rps,
            ttl=self.info.cache_ttl,
        )
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "uniprot: search answer has no results")
        raw_total = resp.headers.get("x-total-results")
        total = int(raw_total) if raw_total and raw_total.isdigit() else None
        return clean_tree(data["results"]), total, resp

    async def health(self) -> dict[str, Any]:
        data, resp = await self.http.get_json(
            self.info.name,
            f"{BASE_URL}/uniprotkb/Q15116",
            params={"format": "json", "fields": "accession,protein_name"},
            rps=self.info.rps,
        )
        name = summarize(data).get("name") if isinstance(data, dict) else None
        if not isinstance(data, dict) or data.get("primaryAccession") != "Q15116" or not name:
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "uniprot: Q15116 lacks primaryAccession or a name"
            )
        return {"ok": True, "detail": f"release {release(resp) or '?'}; Q15116 = {name}"}

    async def resolve(self, kind: str, value: str) -> ResolvedIdentifier | None:
        if kind != "uniprot" or not is_accession(value):
            return None
        try:
            data, _ = await self.entry(value)
        except ImmunoError as exc:
            if exc.code in (ErrorCode.NOT_FOUND, ErrorCode.INVALID_INPUT):
                return None
            raise
        s = summarize(data)
        acc = s["accession"]
        return ResolvedIdentifier(
            kind="uniprot",
            id=value,
            canonical_id=acc,
            title=s.get("name"),
            url=f"https://www.uniprot.org/uniprotkb/{acc}/entry",
            source=self.info.name,
            summary={
                k: s[k]
                for k in ("entry_name", "genes", "organism", "taxid", "length", "reviewed")
                if k in s
            },
        )
