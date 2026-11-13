"""PubTator3 (NCBI): PubMed search with entity annotation, and BioC passages with annotations.

Verified against the live API on 2026-11-13:

* ``GET /search/?text=...&page=N`` returns ``{"results": [...], "count", "page_size", ...}``; the
  page size is fixed at 10. ``sort=date desc`` orders by date (the web app's own value).
  ``filters={"year": ["2024"]}`` accepts ONE year only; two years give zero hits and a range is a 400.
* ``GET /publications/export/biocjson?pmids=a,b&full=true`` returns ``{"PubTator3": [doc, ...]}``.
  Any value of ``full`` (even ``false``) returns the PMC full text when PubTator3 has it, so the
  parameter is sent only when full text is wanted. PMIDs it does not know are silently left out;
  when none are known the answer is HTTP 400 ``{"detail": "Could not retrieve publications"}``.
* ``GET /entity/autocomplete/?query=...&concept=gene&limit=N`` returns a list of entities.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from immuno_mcp.core.connector import Connector, ConnectorInfo, register
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.fence import clean_tree, sanitize_message
from immuno_mcp.core.http import HttpResponse

BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
PAGE_SIZE = 10  # fixed by the service; page_size is ignored

# "@GENE_PDCD1 @GENE_5133 @@@<m>PD</m>-<m>1</m>@@@" in search highlights marks an entity mention.
_HL_ENTITY = re.compile(r"((?:@[A-Z]+_\S+\s+)+)@@@(.*?)@@@")
_TAGS = re.compile(r"<[^>]{0,40}>")
_DOI_IN_CITATION = re.compile(r"doi:\s*(10\.\d{4,9}/\S+?)\.?(?:\s|$)", re.I)


def year_of(*values: object) -> int | None:
    """First four-digit year found at the start of any of the values."""
    for v in values:
        if isinstance(v, str) and len(v) >= 4 and v[:4].isdigit():
            return int(v[:4])
    return None


def plain_highlight(text_hl: str | None) -> str:
    """The highlight snippet as plain text, without entity markup or <m> tags."""
    if not text_hl:
        return ""
    return " ".join(_TAGS.sub("", _HL_ENTITY.sub(lambda m: m.group(2), text_hl)).split())


def highlight_entities(text_hl: str | None) -> list[dict[str, Any]]:
    """Entity hints PubTator3 embeds in a search highlight. Only what is in that snippet."""
    if not text_hl:
        return []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for m in _HL_ENTITY.finditer(text_hl):
        tokens = m.group(1).split()
        etype, _, first = tokens[0][1:].partition("_")
        last = tokens[-1][1:].partition("_")[2]
        entity: dict[str, Any] = {"type": etype.lower()}
        if len(tokens) > 1:
            entity["name"] = first.replace("_", " ")
        entity["id"] = last
        entity["mention"] = _TAGS.sub("", m.group(2)).strip()
        out.setdefault((entity["type"], last), entity)
    return list(out.values())


def doi_from_citation(text: str | None) -> str | None:
    """PubTator3 abstract records carry the PubMed citation line, which includes the DOI."""
    if not text:
        return None
    m = _DOI_IN_CITATION.search(text)
    return m.group(1) if m else None


@register
class PubTatorConnector(Connector):
    info: ClassVar[ConnectorInfo] = ConnectorInfo(
        name="pubtator",
        display_name="PubTator3 (NCBI)",
        kind="literature",
        base_url=BASE_URL,
        homepage="https://www.ncbi.nlm.nih.gov/research/pubtator3/",
        license=(
            "Annotations made by NCBI are US government works in the public domain (NCBI website "
            "policy; acknowledge NLM). Abstracts and full texts keep their publishers' or authors' "
            "copyright, or the article's own licence."
        ),
        citation=(
            "Wei CH, Allot A, Lai PT, Leaman R, Tian S, Luo L, Jin Q, Wang Z, Chen Q, Lu Z (2024) "
            "PubTator 3.0: an AI-powered literature resource for unlocking biomedical knowledge. "
            "Nucleic Acids Res 52(W1):W540-W546. doi:10.1093/nar/gkae235"
        ),
        rps=2.5,
        limit_key="ncbi",  # NCBI allows 3 requests/s per client; shared with other NCBI sources
        cache_ttl=24 * 3600,
        notes="Search page size is fixed at 10. Its year filter takes one year, so ranges are applied here.",
    )

    async def _get(self, path: str, params: dict[str, Any]) -> tuple[Any, HttpResponse]:
        data, resp = await self.http.get_json(
            self.info.name,
            f"{self.info.base_url}{path}",
            params=params,
            rps=self.info.rps,
            limit_key=self.info.limit_key,
            ttl=self.info.cache_ttl,
        )
        return clean_tree(data), resp

    async def search(
        self,
        text: str,
        *,
        page: int = 1,
        sort: Literal["relevance", "date"] = "relevance",
    ) -> tuple[dict[str, Any], HttpResponse]:
        """One page (10 hits) of a PubTator3 search, by relevance (the default) or newest first."""
        params: dict[str, Any] = {"text": text, "page": page}
        if sort == "date":
            params["sort"] = "date desc"
        data, resp = await self._get("/search/", params)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            detail = data.get("detail") if isinstance(data, dict) else None
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE,
                "pubtator: search answer has no 'results' list"
                + (f": {sanitize_message(detail, 160)}" if detail else ""),
            )
        return data, resp

    async def export_biocjson(
        self, pmids: list[str], *, full: bool = True
    ) -> tuple[list[dict[str, Any]], HttpResponse | None]:
        """BioC JSON documents for the PMIDs PubTator3 knows. Unknown PMIDs are simply absent."""
        params: dict[str, Any] = {"pmids": ",".join(pmids)}
        if full:
            params["full"] = "true"
        try:
            data, resp = await self._get("/publications/export/biocjson", params)
        except ImmunoError as exc:
            if exc.code == ErrorCode.INVALID_INPUT and "could not retrieve" in exc.message.lower():
                return [], None  # none of the PMIDs is in PubTator3
            raise
        docs = data.get("PubTator3") if isinstance(data, dict) else None
        if not isinstance(docs, list) or not all(
            isinstance(d, dict) and isinstance(d.get("passages"), list) for d in docs
        ):
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "pubtator: export is not BioC JSON with passages"
            )
        return docs, resp

    async def autocomplete(
        self, query: str, *, concept: str | None = None, limit: int = 5
    ) -> tuple[list[dict[str, Any]], HttpResponse]:
        """Normalised entities matching a name, for example ``PD-1`` -> ``@GENE_PDCD1``."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if concept:
            params["concept"] = concept
        data, resp = await self._get("/entity/autocomplete/", params)
        if not isinstance(data, list) or not all(isinstance(e, dict) and "_id" in e for e in data):
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "pubtator: autocomplete is not a list of entities"
            )
        return data, resp

    async def health(self) -> dict[str, Any]:
        data, _ = await self.search("nanobody")
        first = data["results"][0] if data["results"] else None
        if not isinstance(data.get("count"), int) or not first or "pmid" not in first:
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "pubtator: search hits lack 'count' or 'pmid'"
            )
        return {"ok": True, "detail": f"search 'nanobody': {data['count']} hits"}
