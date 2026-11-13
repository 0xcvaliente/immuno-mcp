"""search_antibody_structures and get_structure through the MCP client, on recorded responses."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
from mcp import Client
from mcp.types import CallToolResult, TextContent

FIX = Path(__file__).parent.parent / "fixtures"
Handler = Callable[[httpx.Request], httpx.Response]


def answer(rel: str, status: int = 200, ctype: str = "application/json", text: str | None = None):
    content = text if text is not None else (FIX / rel).read_text()
    return httpx.Response(status, text=content, headers={"content-type": ctype})


def routes(
    *,
    sabdab: str | httpx.Response | None = None,
    uniprot: str | None = None,
    rcsb: str | httpx.Response | None = None,
    seen: list[httpx.Request] | None = None,
) -> Handler:
    """Serve one recorded body per upstream host; fail on any other request."""

    def pick(value: str | httpx.Response) -> httpx.Response:
        return value if isinstance(value, httpx.Response) else answer(value)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        host = request.url.host
        if host == "sabdab.opig.stats.ox.ac.uk" and sabdab is not None:
            return pick(sabdab)
        if host == "rest.uniprot.org" and uniprot is not None:
            return answer(uniprot)
        if host == "data.rcsb.org" and rcsb is not None:
            return pick(rcsb)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    return handler


def text_of(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def call(make_server, handler: Handler, tool: str, args: dict):
    built = make_server(handler)
    async with Client(built.server) as c:
        return await c.call_tool(tool, args)


# ---- search_antibody_structures ------------------------------------------------------------


async def test_search_by_uniprot_accession_resolves_the_name_first(make_server):
    seen: list[httpx.Request] = []
    handler = routes(
        uniprot="uniprot/entry_q15116.json", sabdab="sabdab/search_pd1_limit3.json", seen=seen
    )
    result = await call(
        make_server, handler, "search_antibody_structures", {"antigen": "Q15116", "limit": 3}
    )
    assert not result.is_error, text_of(result)
    out = result.structured_content
    sab_request = next(r for r in seen if r.url.host == "sabdab.opig.stats.ox.ac.uk")
    assert sab_request.url.params["binds_to_antigen"] == "Programmed cell death pro"
    assert sab_request.url.params["limit"] == "3"

    data = out["data"]
    assert data["total"] == 51 and data["returned"] == 3 and data["offset"] == 0
    assert data["antigen_resolved"] == {
        "accession": "Q15116",
        "name": "Programmed cell death protein 1",
        "organism": "Homo sapiens",
    }
    row = data["rows"][0]
    assert {k: row[k] for k in ("pdb_id", "antibody_type", "heavy_chain", "light_chain")} == {
        "pdb_id": "5B8C",
        "antibody_type": "FV",
        "heavy_chain": "B",
        "light_chain": "A",
    }
    assert row["antigens"][0]["name"] == "Programmed cell death protein 1"
    assert row["antigens"][0]["chain"] == "C"
    assert row["resolution"] == 2.146 and row["method"] == "X-ray diffraction"
    assert "cdr3" not in row  # compact mode

    assert [p["source"] for p in out["provenance"]] == ["uniprot", "sabdab"]
    assert out["provenance"][1]["version"] == "2.1.7"
    assert any("25 characters" in w for w in out["warnings"])
    assert any("not by accession" in w and "other species" in w for w in out["warnings"])
    assert row["antigens"][0]["organism"] == "homo sapiens"
    assert any("offset=3" in s for s in out["next_steps"])
    assert "rows[].antigens[].name" in out["untrusted_fields"]


async def test_aliases_are_translated_to_sabdab_filters(make_server):
    seen: list[httpx.Request] = []
    handler = routes(sabdab="sabdab/search_empty_pd-1.json", seen=seen)
    built = make_server(handler)
    async with Client(built.server) as c:
        r1 = await c.call_tool(
            "search_antibody_structures", {"antigen": "Spike", "antibody_type": "nanobody"}
        )
        r2 = await c.call_tool(
            "search_antibody_structures",
            {"antibody_type": "scfv", "pdb_entry": "7kn5", "cdr_sequence": "aaqsgsyy"},
        )
    assert not r1.is_error and not r2.is_error
    p1, p2 = seen[0].url.params, seen[1].url.params
    assert p1.get_list("antibody_type") == ["SD-H"] and p1["binds_to_antigen"] == "Spike"
    assert p2["construct_type"] == "SCFV" and "antibody_type" not in p2
    assert p2["pdb_entry"] == "pdb_00007kn5" and p2["cdr_sequence"] == "AAQSGSYY"
    assert r2.structured_content["data"]["filters"]["construct_type"] == "SCFV"


async def test_unknown_antibody_type_is_rejected_with_the_valid_values(make_server):
    handler = routes()  # no request may be sent
    result = await call(
        make_server, handler, "search_antibody_structures", {"antibody_type": "IgG"}
    )
    msg = text_of(result)
    assert result.is_error and "[INVALID_INPUT]" in msg
    assert "SD-H" in msg and "vhh" in msg


async def test_bad_cdr_sequence_and_pdb_entry_are_rejected(make_server):
    built = make_server(routes())
    async with Client(built.server) as c:
        r1 = await c.call_tool("search_antibody_structures", {"cdr_sequence": "CAR-123"})
        r2 = await c.call_tool("search_antibody_structures", {"pdb_entry": "7KN"})
    assert r1.is_error and "[INVALID_INPUT]" in text_of(r1)
    assert r2.is_error and "[INVALID_INPUT]" in text_of(r2)


async def test_sabdab_422_reaches_the_model_with_the_allowed_values(make_server):
    handler = routes(sabdab=answer("sabdab/search_422_antibody_type_vhh.json", 422))
    result = await call(make_server, handler, "search_antibody_structures", {"antigen": "Spike"})
    msg = text_of(result)
    assert result.is_error and "[INVALID_INPUT]" in msg
    assert "Input should be 'FV', 'FAB', 'FAB+FC', 'FV*', 'SD-H', 'SD-L' or 'VNAR'" in msg


async def test_web_page_served_as_200_is_an_error(make_server):
    page = answer("sabdab/old_site_app_page.html", ctype="text/html; charset=utf-8")
    result = await call(
        make_server, routes(sabdab=page), "search_antibody_structures", {"antigen": "Spike"}
    )
    assert result.is_error and "[NON_DATA_RESPONSE]" in text_of(result)


async def test_empty_result_explains_how_names_are_matched(make_server):
    handler = routes(uniprot="uniprot/entry_q15116.json", sabdab="sabdab/search_empty_pd-1.json")
    result = await call(make_server, handler, "search_antibody_structures", {"antigen": "Q15116"})
    assert not result.is_error
    out = result.structured_content
    assert out["data"]["total"] == 0 and out["data"]["rows"] == []
    note = next(w for w in out["warnings"] if w.startswith("No SAbDab2 antibody"))
    assert "gene symbols" in note and "'Protein PD-1'" in note and "'CD279'" in note
    assert not out.get("next_steps")


async def test_standard_mode_adds_imgt_cdr3(make_server):
    handler = routes(sabdab="sabdab/search_pd1_limit3.json")
    result = await call(
        make_server,
        handler,
        "search_antibody_structures",
        {"antigen": "programmed cell death", "response_mode": "standard"},
    )
    row = result.structured_content["data"]["rows"][0]
    assert row["cdr3"] == {"H": "ARRDYRFDMGFDY", "L": "QHSRDLPLT"} and row["cdr_scheme"] == "imgt"
    assert row["antibody_id"] == "sabdab2_H013ML00YK"
    assert "license" in result.structured_content["provenance"][0]


# ---- get_structure -------------------------------------------------------------------------


async def test_get_structure_combines_rcsb_and_sabdab(make_server):
    seen: list[httpx.Request] = []
    handler = routes(
        rcsb="rcsb/graphql_entry_7kn5.json", sabdab="sabdab/frontend_pdb_7kn5.json", seen=seen
    )
    result = await call(make_server, handler, "get_structure", {"pdb_id": "7kn5"})
    assert not result.is_error, text_of(result)
    out = result.structured_content
    data = out["data"]
    assert data["pdb_id"] == "7KN5" and "nanobodies VHH E and U" in data["title"]
    assert data["resolution"] == 1.87 and data["release_date"] == "2021-01-20"
    assert data["citation"]["pmid"] == 33436526
    assert [e["entity_id"] for e in data["entities"]] == ["1", "2", "3"]
    assert data["entities"][0]["uniprot"] == ["P0DTC2"]
    assert all("sequence" not in e for e in data["entities"])

    ann = data["sabdab"]
    assert ann["in_sabdab"] is True and ann["cdr_scheme"] == "imgt"
    assert [a["heavy"]["chain"] for a in ann["antibodies"]] == ["C", "D", "E", "F"]
    assert {a["antigens"][0]["name"] for a in ann["antibodies"]} == {"Spike protein S1"}

    assert [p["source"] for p in out["provenance"]] == ["rcsb", "sabdab"]
    assert out["provenance"][0]["query"] == {"api": "graphql", "entry_id": "7KN5"}
    assert any("chains C and antigen chains A" in s for s in out["next_steps"])
    assert {"title", "entities[].molecule", "sabdab.antibodies[].antigens[].name"} <= set(
        out["untrusted_fields"]
    )
    assert seen[1].url.path == "/api/frontend/pdb/pdb_00007kn5"
    assert not out.get("warnings")


async def test_get_structure_with_sequences(make_server):
    handler = routes(rcsb="rcsb/graphql_entry_7kn5.json", sabdab="sabdab/frontend_pdb_7kn5.json")
    result = await call(
        make_server, handler, "get_structure", {"pdb_id": "7KN5", "include_sequences": True}
    )
    data = result.structured_content["data"]
    assert data["entities"][1]["sequence"].startswith("QVQLVETGG")
    assert data["sabdab"]["antibodies"][0]["heavy"]["sequence"].startswith("QVQLVETGG")


async def test_entry_not_in_sabdab_is_reported_not_failed(make_server):
    handler = routes(
        rcsb="rcsb/graphql_entry_7kn5.json",
        sabdab=answer("sabdab/frontend_pdb_1crn_404.json", 404),
    )
    result = await call(make_server, handler, "get_structure", {"pdb_id": "7KN5"})
    assert not result.is_error
    assert result.structured_content["data"]["sabdab"] == {"in_sabdab": False}
    assert [p["source"] for p in result.structured_content["provenance"]] == ["rcsb"]


async def test_sabdab_outage_becomes_a_warning(make_server):
    page = answer("sabdab/old_site_app_page.html", ctype="text/html; charset=utf-8")
    handler = routes(rcsb="rcsb/graphql_entry_7kn5.json", sabdab=page)
    result = await call(make_server, handler, "get_structure", {"pdb_id": "7KN5"})
    assert not result.is_error
    out = result.structured_content
    assert out["data"]["sabdab"] is None
    assert any("SAbDab2 annotation unavailable: [NON_DATA_RESPONSE]" in w for w in out["warnings"])


async def test_unknown_or_malformed_pdb_id(make_server):
    built = make_server(routes(rcsb="rcsb/graphql_entry_missing_0zzz.json"))
    async with Client(built.server) as c:
        missing = await c.call_tool("get_structure", {"pdb_id": "0ZZZ"})
        bad = await c.call_tool("get_structure", {"pdb_id": "not a pdb id"})
    assert missing.is_error and "[NOT_FOUND] rcsb: no PDB entry 0ZZZ" in text_of(missing)
    assert bad.is_error and "[INVALID_INPUT]" in text_of(bad)


async def test_hidden_characters_in_third_party_text_are_removed(make_server):
    body = json.loads((FIX / "rcsb" / "graphql_entry_7kn5.json").read_text())
    body["data"]["entry"]["struct"]["title"] += chr(0x200B) + chr(0xE0041)
    rcsb = answer("", text=json.dumps(body))
    handler = routes(rcsb=rcsb, sabdab=answer("sabdab/frontend_pdb_1crn_404.json", 404))
    result = await call(make_server, handler, "get_structure", {"pdb_id": "7KN5"})
    title = result.structured_content["data"]["title"]
    assert title.endswith("VHH E and U") and chr(0x200B) not in text_of(result)
