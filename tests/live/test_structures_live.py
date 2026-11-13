"""Live contract checks for the structure sources: SAbDab2, RCSB PDB, PDBe and UniProt.

Run with:  .venv/bin/python -m pytest -m live tests/live/test_structures_live.py -q
About 16 requests in total, at most 4 per source, spaced by at least one second per test.
"""

from __future__ import annotations

import asyncio

import pytest
from mcp import Client

from immuno_mcp.app import build_app
from immuno_mcp.connectors.pdbe import PdbeConnector
from immuno_mcp.connectors.rcsb import RcsbConnector
from immuno_mcp.connectors.sabdab import ANTIBODY_TYPES, API_VERSION, BASE_URL, NAME_FILTER_MAX
from immuno_mcp.core.errors import ImmunoError
from immuno_mcp.core.settings import Settings

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
async def _polite():
    yield
    await asyncio.sleep(1.1)


@pytest.fixture
async def app(tmp_path):
    ctx = build_app(Settings(home=tmp_path / "home", profile="lab"), in_memory=True)
    yield ctx
    await ctx.aclose()


@pytest.mark.parametrize("name", ["sabdab", "rcsb", "pdbe", "uniprot"])
async def test_health(app, name):
    try:
        result = await app.connectors[name].health()
    except ImmunoError as exc:
        pytest.fail(f"{name}: {exc.to_message()}")
    assert result["ok"] is True, result


async def test_sabdab_api_is_the_version_the_connector_was_built_for(app):
    spec, _ = await app.http.get_json("sabdab", f"{BASE_URL}/openapi.json")
    assert spec["info"]["version"] == API_VERSION, "SAbDab2 API changed; re-check the connector"
    assert "CC BY 4.0" in spec["info"]["license"]["name"]
    assert tuple(spec["components"]["schemas"]["AntibodyType"]["enum"]) == ANTIBODY_TYPES
    params = {
        p["name"]: p
        for p in spec["paths"]["/frontend/antibody-instance-search"]["get"]["parameters"]
    }
    assert params["binds_to_antigen"]["schema"]["maxLength"] == NAME_FILTER_MAX


async def test_search_pd1_by_uniprot_accession(make_server):
    built = make_server()
    async with Client(built.server) as c:
        result = await c.call_tool("search_antibody_structures", {"antigen": "Q15116", "limit": 5})
    assert not result.is_error, result.content
    data = result.structured_content["data"]
    assert data["antigen_resolved"]["name"] == "Programmed cell death protein 1"
    assert data["total"] > 0 and 0 < len(data["rows"]) <= 5
    names = {a["name"].lower() for r in data["rows"] for a in r["antigens"] if a.get("name")}
    assert any("programmed cell death protein 1" in n for n in names), names
    assert all(r["pdb_id"] and r["antibody_type"] in ANTIBODY_TYPES for r in data["rows"])


async def test_get_structure_7kn5(make_server):
    built = make_server()
    async with Client(built.server) as c:
        result = await c.call_tool("get_structure", {"pdb_id": "7KN5"})
    assert not result.is_error, result.content
    data = result.structured_content["data"]
    assert "nanobodies VHH E and U" in data["title"] and data["resolution"] == 1.87
    assert data["release_date"] == "2021-01-20"
    assert [e["entity_id"] for e in data["entities"]] == ["1", "2", "3"]
    assert data["entities"][0]["uniprot"] == ["P0DTC2"]
    ann = data["sabdab"]
    assert ann["in_sabdab"] is True
    assert [a["antibody_type"] for a in ann["antibodies"]] == ["SD-H"] * 4
    assert {a["antigens"][0]["name"] for a in ann["antibodies"]} == {"Spike protein S1"}
    assert not result.structured_content.get("warnings")


async def test_fetch_structure_file_7kn5_cif(app):
    rcsb = app.connectors["rcsb"]
    assert isinstance(rcsb, RcsbConnector)
    content, prov = await rcsb.fetch_structure_file("7KN5")
    assert content.startswith(b"data_7KN5") and len(content) > 1_000_000
    assert prov.source == "rcsb" and prov.sha256 and prov.url and prov.url.endswith("7KN5.cif")


async def test_fetch_structure_file_7kn5_pdb(app):
    rcsb = app.connectors["rcsb"]
    assert isinstance(rcsb, RcsbConnector)
    content, _ = await rcsb.fetch_structure_file("7KN5", fmt="pdb")
    assert content.startswith(b"HEADER")


async def test_resolve_q15116_and_7kn5(make_server):
    built = make_server()
    async with Client(built.server) as c:
        uni = await c.call_tool("resolve_identifier", {"value": "Q15116"})
        await asyncio.sleep(1.1)
        pdb = await c.call_tool("resolve_identifier", {"value": "7KN5"})
    assert not uni.is_error and not pdb.is_error
    u, p = uni.structured_content["data"], pdb.structured_content["data"]
    assert (u["source"], u["canonical_id"]) == ("uniprot", "Q15116")
    assert u["title"] == "Programmed cell death protein 1"
    assert (p["source"], p["canonical_id"]) == ("rcsb", "7KN5")
    assert p["title"].startswith("Crystal structure of SARS-CoV-2 receptor binding domain")


async def test_pdbe_best_structures_for_pd1(app):
    pdbe = app.connectors["pdbe"]
    assert isinstance(pdbe, PdbeConnector)
    rows, _ = await pdbe.best_structures("Q15116")
    assert rows and all(len(r["pdb_id"]) == 4 for r in rows)
    assert all(r["tax_id"] == 9606 for r in rows[:5])
