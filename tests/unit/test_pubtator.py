from __future__ import annotations

import json

import httpx
import pytest

from immuno_mcp.connectors.pubtator import (
    PubTatorConnector,
    doi_from_citation,
    highlight_entities,
    plain_highlight,
    year_of,
)
from immuno_mcp.core.errors import ErrorCode, ImmunoError

HTML = "<!DOCTYPE html><html><body>NCBI maintenance</body></html>"


def _fixture(fixtures_dir, name: str) -> str:
    return (fixtures_dir / "pubtator" / name).read_text(encoding="utf-8")


def _json(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": "application/json"})


async def test_search_sends_text_and_page_and_returns_real_hits(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "search_pd1_nanobody.json")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json(body)

    conn = PubTatorConnector(make_http(handler), settings)
    data, resp = await conn.search("PD-1 nanobody", page=2)
    assert seen[0].url.path == "/research/pubtator3-api/search/"
    assert seen[0].url.params["text"] == "PD-1 nanobody" and seen[0].url.params["page"] == "2"
    assert "sort" not in seen[0].url.params
    assert data["count"] == 2673 and data["results"][0]["pmid"] == 36475449
    assert resp.status == 200


async def test_date_sort_uses_the_value_the_web_app_sends(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "search_pd1_nanobody_date.json")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json(body)

    conn = PubTatorConnector(make_http(handler), settings)
    await conn.search("PD-1 nanobody", sort="date")
    assert seen[0].url.params["sort"] == "date desc"


async def test_requests_share_the_ncbi_rate_limit(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "search_pd1_nanobody.json")
    http = make_http(lambda r: _json(body))
    await PubTatorConnector(http, settings).search("x y")
    assert "ncbi" in http._limiter._next  # the shared NCBI budget, not the host name


async def test_highlight_markup_becomes_entity_hints(fixtures_dir):
    hits = json.loads(_fixture(fixtures_dir, "search_pd1_nanobody.json"))["results"]
    hit = next(h for h in hits if h["pmid"] == 39897125)
    assert highlight_entities(hit["text_hl"]) == [
        {"type": "gene", "name": "PDCD1", "id": "5133", "mention": "PD-1"}
    ]
    assert plain_highlight(hit["text_hl"]) == (
        "Innovative CDR grafting and computational methods for PD-1 specific nanobody design."
    )
    assert highlight_entities(None) == [] and plain_highlight(None) == ""


def test_highlight_species_and_disease_forms():
    # forms seen in live highlights on 2026-11-13
    text = "@SPECIES_9606 @@@human@@@ and @DISEASE_Neoplasms @DISEASE_MESH:D009369 @@@cancer@@@"
    assert highlight_entities(text) == [
        {"type": "species", "id": "9606", "mention": "human"},
        {"type": "disease", "name": "Neoplasms", "id": "MESH:D009369", "mention": "cancer"},
    ]


def test_year_and_doi_helpers():
    assert year_of("2024", "2025-01-17T00:00:00Z") == 2024
    assert year_of(None, "", "2022-12-07T00:00:00Z") == 2022
    assert year_of("Dec", None) is None
    citation = "Cancer Lett. 2021 Dec 1;522:184-197. doi: 10.1016/j.canlet.2021.09.028. Epub 2021 "
    assert doi_from_citation(citation) == "10.1016/j.canlet.2021.09.028"
    assert doi_from_citation("Biosci Rep") is None


async def test_export_asks_for_full_text_and_returns_documents(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "export_biocjson_two.json")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json(body)

    conn = PubTatorConnector(make_http(handler), settings)
    docs, _ = await conn.export_biocjson(["36475449", "34562519"])
    assert seen[0].url.params["pmids"] == "36475449,34562519"
    assert seen[0].url.params["full"] == "true"
    assert [d["pmid"] for d in docs] == [36475449, 34562519]

    seen.clear()
    await conn.export_biocjson(["34562519"], full=False)
    assert "full" not in seen[0].url.params  # any value of `full` returns full text upstream


async def test_export_of_unknown_pmids_is_empty_not_an_error(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "export_missing_400.json")
    conn = PubTatorConnector(make_http(lambda r: _json(body, 400)), settings)
    docs, resp = await conn.export_biocjson(["99999999"])
    assert docs == [] and resp is None


async def test_export_that_is_not_bioc_is_rejected(make_http, settings):
    conn = PubTatorConnector(make_http(lambda r: _json('{"PubTator3": [{"id": "1"}]}')), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.export_biocjson(["1"])
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_web_page_with_200_is_an_error(make_http, settings):
    conn = PubTatorConnector(
        make_http(lambda r: httpx.Response(200, text=HTML, headers={"content-type": "text/html"})),
        settings,
    )
    with pytest.raises(ImmunoError) as err:
        await conn.search("PD-1")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_detail_message_without_results_is_an_error(make_http, settings):
    # what PubTator3 answered (with HTTP 400) to a year-range filter on 2026-11-13, here as a 200
    body = '{"detail":"We are currently updating the Database. Please try again later"}'
    conn = PubTatorConnector(make_http(lambda r: _json(body)), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.search("PD-1")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE and "updating the Database" in str(
        err.value
    )


async def test_404_is_not_found(make_http, settings):
    conn = PubTatorConnector(make_http(lambda r: httpx.Response(404, text="Not Found")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.search("PD-1")
    assert err.value.code == ErrorCode.NOT_FOUND


async def test_third_party_text_is_cleaned(make_http, settings, fixtures_dir):
    data = json.loads(_fixture(fixtures_dir, "search_pd1_nanobody.json"))
    data["results"][0]["title"] = "Screening" + chr(0x200B) + chr(0x202E) + " of a nanobody"
    conn = PubTatorConnector(make_http(lambda r: _json(json.dumps(data))), settings)
    out, _ = await conn.search("PD-1 nanobody")
    assert out["results"][0]["title"] == "Screening of a nanobody"


async def test_autocomplete_validates_shape(make_http, settings, fixtures_dir):
    body = _fixture(fixtures_dir, "autocomplete_pd1_gene.json")
    conn = PubTatorConnector(make_http(lambda r: _json(body)), settings)
    entities, _ = await conn.autocomplete("PD-1", concept="gene")
    assert entities[0]["_id"] == "@GENE_PDCD1" and entities[0]["db_id"] == "5133"
    bad = PubTatorConnector(make_http(lambda r: _json('{"results": []}')), settings)
    with pytest.raises(ImmunoError):
        await bad.autocomplete("PD-1")


async def test_health_checks_the_shape_of_search_hits(make_http, settings, fixtures_dir):
    ok = PubTatorConnector(
        make_http(lambda r: _json(_fixture(fixtures_dir, "search_pd1_nanobody.json"))), settings
    )
    assert (await ok.health())["ok"] is True
    empty = PubTatorConnector(
        make_http(lambda r: _json(_fixture(fixtures_dir, "search_empty.json"))), settings
    )
    with pytest.raises(ImmunoError) as err:
        await empty.health()
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
