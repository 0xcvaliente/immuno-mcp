from __future__ import annotations

import hashlib

import httpx
import pytest
from mcp import Client

from immuno_mcp.connectors.skempi import SKEMPI_URL


def _text(result) -> str:
    """The text of a tool result (the first content block)."""
    return result.content[0].text


@pytest.fixture
def excerpt(fixtures_dir) -> str:
    return (fixtures_dir / "skempi" / "skempi_v2_excerpt.csv").read_text(encoding="utf-8")


def _csv_handler(text: str, calls: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert str(request.url) == SKEMPI_URL
        return httpx.Response(200, text=text, headers={"content-type": "text/csv; charset=utf-8"})

    return handler


# ---- list_datasets -------------------------------------------------------------------------------


async def test_dataset_cards_are_honest_about_status_and_licence(make_server):
    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"list_datasets must not download: {request.url}")

    built = make_server(no_network)
    async with Client(built.server) as c:
        result = await c.call_tool("list_datasets", {})
    assert not result.is_error, _text(result)
    cards = {d["name"]: d for d in result.structured_content["data"]["datasets"]}
    skempi = cards["SKEMPI 2.0"]
    assert skempi["status"] == "available" and skempi["license"].startswith("CC BY 4.0")
    assert "doi:10.1093/bioinformatics/bty635" in skempi["cite"]
    assert skempi["local_copy"] == {"present": False}
    assert cards["Observed Antibody Space (OAS)"]["status"] == "planned (not built)"
    assert "CC BY 4.0" in cards["Observed Antibody Space (OAS)"]["license"]
    assert "no licence file" in cards["AB-Bind"]["license"]
    assert "never redistribute" in cards["AB-Bind"]["license"]
    assert "HF_TOKEN" in cards["Ginkgo Datapoints GDPa1"]["access"]
    assert "ODbL" in cards["Adaptyv Bio protein design competition data"]["license"]
    sabdab = [d for d in cards.values() if d["name"].startswith("SAbDab")]
    assert len(sabdab) == 1 and sabdab[0]["status"] in ("available", "planned (not built)")
    statuses = {d["status"] for d in cards.values()}
    assert statuses <= {"available", "planned (not built)"}


async def test_cards_show_the_local_skempi_copy_after_a_query(make_server, excerpt):
    calls: list[httpx.Request] = []
    built = make_server(_csv_handler(excerpt, calls))
    async with Client(built.server) as c:
        await c.call_tool("query_binding_data", {"pdb_id": "1VFB"})
        result = await c.call_tool("list_datasets", {})
    skempi = next(
        d for d in result.structured_content["data"]["datasets"] if d["name"] == "SKEMPI 2.0"
    )
    assert skempi["local_copy"]["present"] is True and skempi["local_copy"]["rows"] == 16
    assert skempi["local_copy"]["sha256"] == hashlib.sha256(excerpt.encode()).hexdigest()


async def test_readonly_profile_hides_the_downloading_tool(make_server):
    built = make_server(None, profile="readonly")
    async with Client(built.server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        result = await c.call_tool("list_datasets", {})
    assert "list_datasets" in names and "query_binding_data" not in names
    skempi = result.structured_content["data"]["datasets"][0]
    assert "lab profile" in skempi["note"]


# ---- query_binding_data --------------------------------------------------------------------------


async def test_first_query_downloads_then_reuses_the_local_copy(make_server, excerpt):
    calls: list[httpx.Request] = []
    built = make_server(_csv_handler(excerpt, calls))
    async with Client(built.server) as c:
        first = await c.call_tool("query_binding_data", {"pdb_id": "1VFB"})
        second = await c.call_tool("query_binding_data", {"pdb_id": "1VFB", "limit": 2})
    assert not first.is_error, _text(first)
    sc = first.structured_content
    assert len(calls) == 1
    rows = sc["data"]["rows"]
    assert sc["data"]["total_matched"] == 4 and len(rows) == 4
    ya32a = next(r for r in rows if r["mutations"] == "YA32A")
    assert ya32a["ddg_kcal_mol"] == 1.34 and ya32a["temperature_K"] == 298
    assert ya32a["kd_wt_M"] == 2e-08 and ya32a["kd_mut_M"] == 1.92e-07
    assert ya32a["reference"] == "PMID:8703938"
    assert "R*T*ln(Kd_mut/Kd_wt)" in sc["data"]["ddg_formula"]
    assert "0.0019872" in sc["data"]["ddg_formula"]
    prov = sc["provenance"][0]
    assert prov["source"] == "skempi" and prov["url"] == SKEMPI_URL
    assert prov["sha256"] == hashlib.sha256(excerpt.encode()).hexdigest()
    assert "cached" not in prov
    assert any(w.startswith("Downloaded SKEMPI 2.0") for w in sc["warnings"])
    assert "rows[].protein_1" in sc["untrusted_fields"]
    assert "get_passages(pmids=['8703938', '9609690'])" in sc["next_steps"][0]

    again = second.structured_content
    assert again["data"]["returned"] == 2 and again["data"]["total_matched"] == 4
    assert again["provenance"][0]["cached"] is True
    assert not any(w.startswith("Downloaded") for w in again.get("warnings", []))


async def test_antibody_only_is_labelled_a_heuristic(make_server, excerpt):
    built = make_server(_csv_handler(excerpt, []))
    async with Client(built.server) as c:
        result = await c.call_tool("query_binding_data", {"antibody_only": True})
    sc = result.structured_content
    rows = sc["data"]["rows"]
    assert {r["protein_1"] for r in rows} == {
        "IgG1-kappa D1.3 Fv",
        "HyHEL-10",
        "Lebrikizumab fab",
    }
    hyhel = next(r for r in rows if r["protein_1"] == "HyHEL-10")
    assert hyhel["antibody_evidence"] == "skempi_label"
    assert any(w.startswith("antibody_only is a heuristic") for w in sc["warnings"])
    lebri = next(r for r in rows if r["protein_1"] == "Lebrikizumab fab")
    assert lebri["ddg_kcal_mol"] is None and "opposite directions" in lebri["ddg_note"]


async def test_mutation_filter_and_bounds_warning(make_server, excerpt):
    built = make_server(_csv_handler(excerpt, []))
    async with Client(built.server) as c:
        any_chain = await c.call_tool("query_binding_data", {"mutation": "Y32A"})
        bound = await c.call_tool("query_binding_data", {"pdb_id": "1SBB"})
        standard = await c.call_tool(
            "query_binding_data", {"pdb_id": "1VFB_AB_C", "response_mode": "standard"}
        )
    assert [r["mutations"] for r in any_chain.structured_content["data"]["rows"]] == [
        "YA32A",
        "YB32A",
        "YA32A,QC121A",
    ]
    brow = bound.structured_content["data"]["rows"][0]
    assert brow["ddg_qualifier"] == ">" and brow["kd_mut_reported"] == ">1.5E-04"
    assert any("bound" in w for w in bound.structured_content["warnings"])
    srow = standard.structured_content["data"]["rows"][0]
    assert srow["hold_out_type"] == "AB/AG" and srow["mutations_cleaned"] == "HA30A"


async def test_no_match_is_ok_and_says_so(make_server, excerpt):
    built = make_server(_csv_handler(excerpt, []))
    async with Client(built.server) as c:
        result = await c.call_tool("query_binding_data", {"pdb_id": "9XYZ"})
    assert not result.is_error and result.structured_content["data"]["rows"] == []
    assert any("No SKEMPI 2.0 rows matched" in w for w in result.structured_content["warnings"])


async def test_malformed_csv_rows_are_reported(make_server, excerpt):
    lines = excerpt.splitlines()
    broken = "\n".join([lines[0], lines[1].rsplit(";", 3)[0], *lines[2:]]) + "\n"
    built = make_server(_csv_handler(broken, []))
    async with Client(built.server) as c:
        result = await c.call_tool("query_binding_data", {"protein": "eglin"})
    sc = result.structured_content
    assert sc["data"]["total_matched"] == 2  # one of the three eglin c rows was malformed
    assert any("1 malformed CSV rows skipped (lines 2)" in w for w in sc["warnings"])


async def test_bad_arguments_are_invalid_input(make_server):
    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    built = make_server(no_network)
    async with Client(built.server) as c:
        bad_pdb = await c.call_tool("query_binding_data", {"pdb_id": "lysozyme"})
        bad_mut = await c.call_tool("query_binding_data", {"mutation": "Tyr32Ala"})
    assert bad_pdb.is_error and "INVALID_INPUT" in _text(bad_pdb)
    assert bad_mut.is_error and "YA32A" in _text(bad_mut)


async def test_download_failures_are_typed(make_server):
    page = "<!DOCTYPE html><html><body>maintenance</body></html>"
    built = make_server(
        lambda r: httpx.Response(200, text=page, headers={"content-type": "text/html"})
    )
    async with Client(built.server) as c:
        html = await c.call_tool("query_binding_data", {"pdb_id": "1VFB"})
    assert html.is_error and "NON_DATA_RESPONSE" in _text(html)
    built = make_server(lambda r: httpx.Response(404, text=""))
    async with Client(built.server) as c:
        gone = await c.call_tool("query_binding_data", {"pdb_id": "1VFB"})
    assert gone.is_error and "NOT_FOUND" in _text(gone)
