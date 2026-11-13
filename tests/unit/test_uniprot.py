"""UniProt connector, tested against responses captured from the live API on 2026-11-13."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from immuno_mcp.connectors.uniprot import UniprotConnector, is_accession, summarize
from immuno_mcp.core.errors import ErrorCode, ImmunoError

FIX = Path(__file__).parent.parent / "fixtures" / "uniprot"


def answer(name: str, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        text=(FIX / name).read_text(),
        headers={"content-type": "application/json", **(headers or {})},
    )


def test_accession_pattern():
    assert is_accession("Q15116") and is_accession("q15116") and is_accession("A0A0B4J2F0")
    assert is_accession("Q15116-2")
    assert not is_accession("PD-1") and not is_accession("PDCD1") and not is_accession("7KN5")


async def test_entry_is_summarised(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/uniprotkb/Q15116"
        assert (
            request.url.params["format"] == "json"
            and "protein_name" in request.url.params["fields"]
        )
        return answer("entry_q15116.json", headers={"X-UniProt-Release": "2026_04"})

    conn = UniprotConnector(make_http(handler), settings)
    entry, resp = await conn.entry("q15116")
    s = summarize(entry)
    assert s["accession"] == "Q15116" and s["name"] == "Programmed cell death protein 1"
    assert "Protein PD-1" in s["short_names"] and "CD279" in s["short_names"]
    assert s["genes"][0] == "PDCD1" and s["organism"] == "Homo sapiens" and s["taxid"] == 9606
    assert s["reviewed"] is True and s["length"] == 288
    assert resp.headers["x-uniprot-release"] == "2026_04"


async def test_inactive_entry_answered_with_200_is_not_found(make_http, settings):
    conn = UniprotConnector(make_http(lambda r: answer("inactive_a9z9z9.json")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.entry("A9Z9Z9")
    assert err.value.code == ErrorCode.NOT_FOUND and "inactive (DELETED)" in err.value.message


async def test_malformed_accession(make_http, settings):
    remote = UniprotConnector(
        make_http(lambda r: answer("invalid_accession_400.json", 400)), settings
    )
    with pytest.raises(ImmunoError) as err:
        await remote.http.get_json("uniprot", "https://rest.uniprot.org/uniprotkb/NOTANACC.json")
    assert err.value.code == ErrorCode.INVALID_INPUT and "invalid format" in err.value.message

    local = UniprotConnector(make_http(lambda r: pytest.fail("no request expected")), settings)
    with pytest.raises(ImmunoError) as err:
        await local.entry("NOTANACC")
    assert err.value.code == ErrorCode.INVALID_INPUT


async def test_search_reads_total_from_header(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == "gene:PDCD1 AND organism_id:9606 AND reviewed:true"
        return answer("search_pdcd1_human.json", headers={"X-Total-Results": "1"})

    conn = UniprotConnector(make_http(handler), settings)
    rows, total, _ = await conn.search("gene:PDCD1 AND organism_id:9606 AND reviewed:true", size=5)
    assert total == 1 and [summarize(r)["accession"] for r in rows] == ["Q15116"]


async def test_resolve(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("Q15116"):
            return answer("entry_q15116.json")
        return answer("inactive_a9z9z9.json")

    conn = UniprotConnector(make_http(handler), settings)
    found = await conn.resolve("uniprot", "Q15116")
    assert found is not None and found.canonical_id == "Q15116"
    assert found.title == "Programmed cell death protein 1"
    assert found.url == "https://www.uniprot.org/uniprotkb/Q15116/entry"
    assert found.summary["organism"] == "Homo sapiens" and "PDCD1" in found.summary["genes"]
    assert await conn.resolve("uniprot", "A9Z9Z9") is None
    assert await conn.resolve("pdb", "7KN5") is None
    assert await conn.resolve("uniprot", "PD-1") is None


async def test_health(make_http, settings):
    good = UniprotConnector(
        make_http(lambda r: answer("entry_q15116.json", headers={"X-UniProt-Release": "2026_04"})),
        settings,
    )
    result = await good.health()
    assert result["ok"] is True and "2026_04" in result["detail"]
    wrong = UniprotConnector(make_http(lambda r: answer("search_pdcd1_human.json")), settings)
    with pytest.raises(ImmunoError) as err:
        await wrong.health()
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
