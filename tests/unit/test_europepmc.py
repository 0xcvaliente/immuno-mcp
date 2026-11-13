from __future__ import annotations

import httpx
import pytest
from mcp import Client

from immuno_mcp.connectors.europepmc import (
    EuropePmcConnector,
    normalise_id,
    parse_jats,
    plain_text,
)
from immuno_mcp.core.errors import ErrorCode, ImmunoError


def _text(result) -> str:
    """The text of a tool result (the first content block)."""
    return result.content[0].text


def _fixture(fixtures_dir, name: str) -> str:
    return (fixtures_dir / "europepmc" / name).read_text(encoding="utf-8")


def _json(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": "application/json"})


def test_identifier_normalisation():
    assert normalise_id("pmid", " PMID: 36475449 ") == "36475449"
    assert normalise_id("pmcid", "pmc9867944") == "PMC9867944"
    assert normalise_id("doi", "https://doi.org/10.1042/BSR20221546") == "10.1042/BSR20221546"
    assert normalise_id("doi", 'doi:10.1/x"y') is None
    assert normalise_id("pmid", "36475449a") is None


async def test_resolve_pmid_sends_the_documented_query(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "search_pmid_36475449_core.json")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json(body)

    conn = EuropePmcConnector(make_http(handler), settings)
    found = await conn.resolve("pmid", "36475449")
    assert seen[0].url.path == "/europepmc/webservices/rest/search"
    assert seen[0].url.params["query"] == "EXT_ID:36475449 AND SRC:MED"
    assert seen[0].url.params["format"] == "json"
    assert found is not None and found.canonical_id == "36475449"
    assert (
        found.title
        == "Screening and identification of an anti-PD-1 nanobody with antitumor activity."
    )
    assert found.summary["pmcid"] == "PMC9867944" and found.summary["doi"] == "10.1042/bsr20221546"
    assert found.summary["journal"] == "Biosci Rep" and found.summary["year"] == 2022
    assert found.url == "https://europepmc.org/article/MED/36475449"


async def test_resolve_doi_is_case_insensitive(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "search_doi_lite.json")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json(body)

    conn = EuropePmcConnector(make_http(handler), settings)
    found = await conn.resolve("doi", "10.1042/BSR20221546")
    assert seen[0].url.params["query"] == 'DOI:"10.1042/BSR20221546"'
    assert found is not None and found.canonical_id == "10.1042/bsr20221546"
    assert found.summary["pmid"] == "36475449"


async def test_resolve_pmcid(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "search_pmcid_lite.json")
    conn = EuropePmcConnector(make_http(lambda r: _json(body)), settings)
    found = await conn.resolve("pmcid", "pmc9867944")
    assert found is not None and found.canonical_id == "PMC9867944"
    assert found.summary["open_access"] is True


async def test_resolve_miss_and_malformed_ids(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "search_notfound.json")
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _json(body)

    conn = EuropePmcConnector(make_http(handler), settings)
    assert await conn.resolve("pmid", "99999999") is None
    assert await conn.resolve("pmid", "not-a-pmid") is None
    assert await conn.resolve("pdb", "1VFB") is None
    assert len(calls) == 1  # malformed or foreign ids never reach the network


async def test_answer_without_result_list_is_an_error(make_http, settings):
    conn = EuropePmcConnector(make_http(lambda r: _json('{"version":"6.9"}')), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.resolve("pmid", "36475449")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_web_page_with_200_is_an_error(make_http, settings):
    page = "<!DOCTYPE html><html><head><title>Just a moment...</title></head></html>"
    conn = EuropePmcConnector(
        make_http(lambda r: httpx.Response(200, text=page, headers={"content-type": "text/html"})),
        settings,
    )
    with pytest.raises(ImmunoError) as err:
        await conn.resolve("pmid", "36475449")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
    with pytest.raises(ImmunoError) as err:
        await conn.full_text_xml("PMC9867944")  # text endpoint: our own XML check catches it
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_full_text_is_parsed_into_labelled_passages(make_http, settings, fixtures_dir):
    xml = _fixture(fixtures_dir, "fulltext_PMC9867944.xml")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=xml, headers={"content-type": "application/xml"})

    conn = EuropePmcConnector(make_http(handler), settings)
    full, _ = await conn.full_text_xml("PMC9867944")
    assert seen[0].url.path == "/europepmc/webservices/rest/PMC9867944/fullTextXML"
    assert full["pmid"] == "36475449" and full["doi"] == "10.1042/BSR20221546"
    assert full["journal"] == "Biosci Rep"
    sections = [p["section"] for p in full["passages"]]
    assert sections[0] == "abstract" and "introduction" in sections and "methods" in sections
    methods = next(p for p in full["passages"] if p["section"] == "methods")
    assert methods["heading"] == "Generation of the VHH phage display library"


async def test_full_text_404_is_not_found(make_http, settings):
    conn = EuropePmcConnector(make_http(lambda r: httpx.Response(404, text="")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.full_text_xml("PMC1")
    assert err.value.code == ErrorCode.NOT_FOUND


def test_xml_with_entity_declarations_is_refused():
    evil = '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "y">]><article>&x;</article>'
    with pytest.raises(ImmunoError) as err:
        parse_jats(evil)
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
    with pytest.raises(ImmunoError):
        parse_jats("<article><body>")


def test_abstract_markup_is_reduced_to_text():
    assert plain_text("<h4>Background</h4>PD-1 &amp; PD-L1 <i>in vivo</i>") == (
        "Background PD-1 & PD-L1 in vivo"
    )


async def test_health_validates_the_resolved_record(make_http, settings, fixtures_dir):
    ok = EuropePmcConnector(
        make_http(lambda r: _json(_fixture(fixtures_dir, "search_pmid_36475449_core.json"))),
        settings,
    )
    assert (await ok.health())["ok"] is True
    miss = EuropePmcConnector(
        make_http(lambda r: _json(_fixture(fixtures_dir, "search_notfound.json"))), settings
    )
    with pytest.raises(ImmunoError):
        await miss.health()


async def test_resolve_identifier_tool_uses_europepmc(make_server, fixtures_dir):
    core = _fixture(fixtures_dir, "search_pmid_36475449_core.json")
    doi = _fixture(fixtures_dir, "search_doi_lite.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.ebi.ac.uk"
        return _json(doi if "DOI:" in request.url.params["query"] else core)

    built = make_server(handler)
    async with Client(built.server) as c:
        by_pmid = await c.call_tool("resolve_identifier", {"value": "36475449"})
        by_doi = await c.call_tool("resolve_identifier", {"value": "10.1042/BSR20221546"})
    assert not by_pmid.is_error, _text(by_pmid)
    assert by_pmid.structured_content["data"]["canonical_id"] == "36475449"
    assert by_pmid.structured_content["provenance"][0]["source"] == "europepmc"
    assert by_doi.structured_content["data"]["summary"]["pmid"] == "36475449"
