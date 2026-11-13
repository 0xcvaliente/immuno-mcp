"""Europe PMC REST: identifier resolution (PMID, PMCID, DOI), abstracts, open-access full text.

Verified against the live API on 2026-11-13 (API version 6.9):

* ``GET /search?query=EXT_ID:<pmid> AND SRC:MED&format=json&resultType=lite|core&pageSize=N``
  returns ``{"hitCount", "resultList": {"result": [...]}}``. ``PMCID:PMC...`` and ``DOI:"10..."``
  work the same way. DOIs come back lower-cased. A miss is ``hitCount: 0`` with HTTP 200.
* ``GET /<PMCID>/fullTextXML`` returns JATS XML for articles in the open-access subset.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any, ClassVar

from immuno_mcp.core.connector import Connector, ConnectorInfo, ResolvedIdentifier, register
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.fence import clean_text, clean_tree, sanitize_message
from immuno_mcp.core.http import HttpResponse

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_FORMATS: dict[str, re.Pattern[str]] = {
    "pmid": re.compile(r"^\d{1,9}$"),
    "pmcid": re.compile(r"^PMC\d{1,9}$"),
    "doi": re.compile(r'^10\.\d{4,9}/[^\s"]+$'),
}
_TAG = re.compile(r"<[^>]*>")
_SECTION_WORDS = [
    ("intro", "introduction"),
    ("method", "methods"),
    ("material", "methods"),
    ("result", "results"),
    ("discuss", "discussion"),
    ("conclu", "conclusion"),
]


def normalise_id(kind: str, value: str) -> str | None:
    """Canonical spelling of a PMID, PMCID or DOI, or None when it is not well formed."""
    v = value.strip()
    if kind == "pmid":
        v = re.sub(r"^pmid:\s*", "", v, flags=re.I)
    elif kind == "pmcid":
        v = v.upper()
    elif kind == "doi":
        v = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", v, flags=re.I)
    pattern = _FORMATS.get(kind)
    return v if pattern and pattern.match(v) else None


def plain_text(markup: str | None) -> str:
    """Abstracts arrive with HTML tags (<h4>, <i>, <sup>). Keep only the words."""
    if not markup:
        return ""
    return " ".join(html.unescape(_TAG.sub(" ", markup)).split())


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _label(title: str, sec_type: str | None) -> str:
    probe = f"{sec_type or ''} {title}".lower()
    for word, label in _SECTION_WORDS:
        if word in probe:
            return label
    return "body"


def parse_jats(xml_text: str) -> dict[str, Any]:
    """Title, abstract paragraphs and body paragraphs (with section labels) from JATS XML."""
    if "<!ENTITY" in xml_text:
        raise ImmunoError(ErrorCode.NON_DATA_RESPONSE, "europepmc: full text declares entities")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ImmunoError(
            ErrorCode.NON_DATA_RESPONSE, "europepmc: full text is not well-formed XML"
        ) from exc
    meta = root.find("front/article-meta")
    ids = {
        el.get("pub-id-type"): (el.text or "").strip()
        for el in (meta.findall("article-id") if meta is not None else [])
    }
    passages: list[dict[str, str]] = []
    for p in root.findall("front/article-meta/abstract//p"):
        if text := _text(p):
            passages.append({"section": "abstract", "text": text})

    def walk(sec: ET.Element, label: str | None) -> None:
        heading = _text(sec.find("title"))
        here = label or _label(heading, sec.get("sec-type"))
        for child in sec:
            if child.tag == "p" and (text := _text(child)):
                passages.append({"section": here, "heading": heading, "text": text})
            elif child.tag == "sec":
                walk(child, here)
            elif child.tag in ("fig", "table-wrap"):
                caption = _text(child.find("caption"))
                if caption:
                    kind = "figure caption" if child.tag == "fig" else "table caption"
                    passages.append({"section": kind, "heading": heading, "text": caption})

    body = root.find("body")
    if body is not None:
        for child in body:
            if child.tag == "sec":
                walk(child, None)
            elif child.tag == "p" and (text := _text(child)):
                passages.append({"section": "body", "text": text})
    return clean_tree(
        {
            "title": _text(root.find("front/article-meta/title-group/article-title")),
            "journal": _text(
                root.find("front/journal-meta/journal-title-group/abbrev-journal-title")
            )
            or _text(root.find("front/journal-meta/journal-title-group/journal-title")),
            "pmid": ids.get("pmid"),
            "pmcid": ids.get("pmcid"),
            "doi": ids.get("doi"),
            "passages": passages,
        }
    )


def record_summary(rec: dict[str, Any]) -> dict[str, Any]:
    """The citable fields of a Europe PMC record (lite or core)."""
    journal_info = rec.get("journalInfo") or {}
    journal = (journal_info.get("journal") or {}).get("isoabbreviation") or rec.get("journalTitle")
    if not journal:
        journal = ((rec.get("bookOrReportDetails") or {}).get("publisher")) or None
    year = rec.get("pubYear")
    out = {
        "pmid": rec.get("pmid"),
        "pmcid": rec.get("pmcid"),
        "doi": rec.get("doi"),
        "title": clean_text(rec.get("title") or ""),
        "journal": journal,
        "year": int(year) if isinstance(year, str) and year.isdigit() else None,
        "source": rec.get("source"),
        "open_access": rec.get("isOpenAccess") == "Y",
        "in_europepmc": rec.get("inEPMC") == "Y",
    }
    if rec.get("license"):
        out["license"] = rec["license"]
    return {k: v for k, v in out.items() if v not in (None, "")}


@register
class EuropePmcConnector(Connector):
    info: ClassVar[ConnectorInfo] = ConnectorInfo(
        name="europepmc",
        display_name="Europe PMC",
        kind="literature",
        base_url=BASE_URL,
        homepage="https://europepmc.org",
        license=(
            "Article content is copyrighted by publishers or authors. Open-access subset articles "
            "carry their own licence (usually Creative Commons; terms differ per article, see the "
            "record's licence). Automated retrieval only through the REST, OAI or bulk services "
            "(Europe PMC copyright page)."
        ),
        citation=(
            "Rosonovski S, Levchenko M, Bhatnagar R, et al. (2024) Europe PMC in 2023. "
            "Nucleic Acids Res 52(D1):D1668-D1676. doi:10.1093/nar/gkad1085"
        ),
        rps=2.0,
        cache_ttl=24 * 3600,
        resolves=frozenset({"pmid", "pmcid", "doi"}),
    )

    async def search(
        self, query: str, *, result_type: str = "lite", page_size: int = 10
    ) -> tuple[dict[str, Any], HttpResponse]:
        data, resp = await self.http.get_json(
            self.info.name,
            f"{self.info.base_url}/search",
            params={
                "query": query,
                "format": "json",
                "resultType": result_type,
                "pageSize": page_size,
            },
            rps=self.info.rps,
            ttl=self.info.cache_ttl,
        )
        results = (data.get("resultList") or {}).get("result") if isinstance(data, dict) else None
        if not isinstance(results, list) or not isinstance(data.get("hitCount"), int):
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "europepmc: search answer has no resultList"
            )
        return clean_tree(data), resp

    async def lookup(
        self, kind: str, value: str, *, result_type: str = "lite"
    ) -> tuple[dict[str, Any] | None, HttpResponse]:
        """The one record whose PMID, PMCID or DOI equals ``value`` (already normalised)."""
        query = {
            "pmid": f"EXT_ID:{value} AND SRC:MED",
            "pmcid": f"PMCID:{value}",
            "doi": f'DOI:"{value}"',
        }[kind]
        data, resp = await self.search(query, result_type=result_type, page_size=5)
        field = {"pmid": "pmid", "pmcid": "pmcid", "doi": "doi"}[kind]
        matches = [
            r
            for r in data["resultList"]["result"]
            if str(r.get(field) or "").lower() == value.lower()
        ]
        matches.sort(key=lambda r: r.get("source") != "MED")  # the PubMed record first
        return (matches[0] if matches else None), resp

    async def resolve(self, kind: str, value: str) -> ResolvedIdentifier | None:
        if kind not in self.info.resolves:
            return None
        norm = normalise_id(kind, value)
        if norm is None:
            return None
        rec, _ = await self.lookup(kind, norm)
        if rec is None:
            return None
        summary = record_summary(rec)
        canonical = {"pmid": summary.get("pmid"), "pmcid": summary.get("pmcid")}.get(kind) or (
            summary.get("doi") if kind == "doi" else norm
        )
        return ResolvedIdentifier(
            kind=kind,
            id=value.strip(),
            canonical_id=str(canonical),
            title=summary.get("title"),
            url=f"https://europepmc.org/article/{rec.get('source', 'MED')}/{rec.get('id', norm)}",
            source=self.info.name,
            summary={k: v for k, v in summary.items() if k != "title"},
        )

    async def full_text_xml(self, pmcid: str) -> tuple[dict[str, Any], HttpResponse]:
        """Parsed JATS full text of an open-access article. NOT_FOUND when it is not available."""
        norm = normalise_id("pmcid", pmcid)
        if norm is None:
            raise ImmunoError(ErrorCode.INVALID_INPUT, f"not a PMCID: {pmcid!r}")

        def must_be_xml(resp: HttpResponse) -> None:
            head = resp.content[:200].lstrip().lower()
            if not (head.startswith(b"<?xml") or head.startswith(b"<article")):
                raise ImmunoError(
                    ErrorCode.NON_DATA_RESPONSE,
                    f"europepmc: expected JATS XML but got: {sanitize_message(resp.text, 120)}",
                )

        resp = await self.http.get_text(
            self.info.name,
            f"{self.info.base_url}/{norm}/fullTextXML",
            rps=self.info.rps,
            ttl=self.info.cache_ttl,
            validate=must_be_xml,
        )
        return parse_jats(resp.text), resp

    async def health(self) -> dict[str, Any]:
        rec, _ = await self.lookup("pmid", "36475449")
        if rec is None or not rec.get("title") or rec.get("pmcid") != "PMC9867944":
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "europepmc: PMID 36475449 did not resolve as expected"
            )
        return {"ok": True, "detail": "PMID 36475449 resolves to PMC9867944"}
