"""Live checks for PubTator3, Europe PMC and SKEMPI 2.0, and the tools built on them.

Run with:  .venv/bin/python -m pytest -m live tests/live/test_literature_live.py -q
About 3 requests to each literature API and 2 downloads of the SKEMPI CSV (1.6 MB).
"""

from __future__ import annotations

import asyncio
import re

import pytest
from mcp import Client

from immuno_mcp.app import build_app
from immuno_mcp.core.settings import Settings
from immuno_mcp.server import build_server

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
async def polite():
    await asyncio.sleep(1.1)  # at most one request per second per host across tests


@pytest.fixture
def settings(tmp_path):
    return Settings(home=tmp_path / "home", profile="lab")


@pytest.mark.parametrize("name", ["pubtator", "europepmc"])
async def test_literature_connector_health(settings, name):
    app = build_app(settings, in_memory=True)
    try:
        result = await app.connectors[name].health()
    finally:
        await app.aclose()
    assert result["ok"] is True, result


async def test_search_literature_pd1_nanobody(settings):
    built = build_server(settings, in_memory=True)
    async with Client(built.server) as c:
        result = await c.call_tool("search_literature", {"query": "PD-1 nanobody", "limit": 10})
    assert not result.is_error, result.content
    rows = result.structured_content["data"]["rows"]
    assert len(rows) == 10 and all(re.fullmatch(r"\d+", r["pmid"]) for r in rows)
    assert all(r.get("title") and r.get("year") for r in rows)
    assert any("nanobod" in r["title"].lower() for r in rows)


async def test_get_passages_on_a_real_pmid(settings):
    built = build_server(settings, in_memory=True)
    async with Client(built.server) as c:
        result = await c.call_tool(
            "get_passages", {"pmids": ["36475449"], "query": "affinity", "max_passages_per_pmid": 3}
        )
    assert not result.is_error, result.content
    art = result.structured_content["data"]["articles"][0]
    assert art["source"] == "pubtator" and art["coverage"] == "full_text"
    assert art["doi"].lower() == "10.1042/bsr20221546"
    assert all(p["stable_citation_key"].startswith("PMID:36475449#p") for p in art["passages"])
    # ranked by word overlap; a clipped passage shows the excerpt around the match
    assert all(p["score"] == 1.0 and "affinit" in p["text"].lower() for p in art["passages"])


async def test_resolve_real_pmid_and_doi(settings):
    built = build_server(settings, in_memory=True)
    async with Client(built.server) as c:
        by_pmid = await c.call_tool("resolve_identifier", {"value": "36475449", "kind": "pmid"})
        await asyncio.sleep(1.1)
        by_doi = await c.call_tool("resolve_identifier", {"value": "10.1042/BSR20221546"})
    assert not by_pmid.is_error and not by_doi.is_error
    pmid = by_pmid.structured_content["data"]
    assert pmid["canonical_id"] == "36475449" and pmid["summary"]["pmcid"] == "PMC9867944"
    assert "anti-PD-1 nanobody" in pmid["title"]
    assert by_doi.structured_content["data"]["summary"]["pmid"] == "36475449"


async def test_skempi_health_and_an_antibody_antigen_query(settings):
    app = build_app(settings, in_memory=True)
    try:
        health = await app.connectors["skempi"].health()
    finally:
        await app.aclose()
    assert health["ok"] is True and int(health["detail"].split()[0]) >= 7000, health

    await asyncio.sleep(1.1)
    built = build_server(settings, in_memory=True)
    async with Client(built.server) as c:
        result = await c.call_tool(
            "query_binding_data", {"pdb_id": "1VFB", "antibody_only": True, "limit": 50}
        )
        again = await c.call_tool("query_binding_data", {"pdb_id": "1VFB", "mutation": "YA32A"})
    assert not result.is_error, result.content
    rows = result.structured_content["data"]["rows"]
    assert rows and all(r["pdb"] == "1VFB" for r in rows)
    assert all(r["antibody_evidence"] for r in rows)
    ya32a = again.structured_content["data"]["rows"][0]
    assert ya32a["ddg_kcal_mol"] == 1.34 and ya32a["temperature_K"] == 298
    assert again.structured_content["provenance"][0]["cached"] is True  # second call: no download
