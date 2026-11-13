"""RCSB connector, tested against responses captured from the live services on 2026-11-13."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from immuno_mcp.connectors.rcsb import RcsbConnector, check_structure_bytes, normalize_pdb_id
from immuno_mcp.core.errors import ErrorCode, ImmunoError

FIX = Path(__file__).parent.parent / "fixtures" / "rcsb"
JSON = {"content-type": "application/json"}


def raw(name: str) -> bytes:
    return (FIX / name).read_bytes()


def answer(name: str, status: int = 200, ctype: str = "application/json") -> httpx.Response:
    return httpx.Response(status, content=raw(name), headers={"content-type": ctype})


def test_pdb_id_forms():
    assert normalize_pdb_id(" 7kn5 ") == "7KN5"
    assert normalize_pdb_id("pdb_00007kn5") == "7KN5"
    assert normalize_pdb_id("PDB_1000AXYZ") == "pdb_1000axyz"  # 2027+ ids are kept as given
    for bad in ("7KN", "7KN5A", "hello", ""):
        with pytest.raises(ImmunoError) as err:
            normalize_pdb_id(bad)
        assert err.value.code == ErrorCode.INVALID_INPUT


async def test_entry_details_projects_entry_citation_and_entities(make_http, settings):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer("graphql_entry_7kn5.json")

    conn = RcsbConnector(make_http(handler), settings)
    entry, _ = await conn.entry_details("7kn5")
    sent = json.loads(seen[0].content)
    assert seen[0].method == "POST" and str(seen[0].url) == "https://data.rcsb.org/graphql"
    assert sent["variables"] == {"id": "7KN5"}
    assert entry["pdb_id"] == "7KN5" and entry["resolution"] == 1.87
    assert entry["method"] == ["X-RAY DIFFRACTION"]
    assert (entry["deposit_date"], entry["release_date"]) == ("2020-11-04", "2021-01-20")
    assert entry["citation"]["doi"] == "10.1126/science.abe6230"
    assert entry["citation"]["authors"] == "Koenig, P.A., Das, H., Liu, H. et al."
    assert [e["entity_id"] for e in entry["entities"]] == ["1", "2", "3"]  # sorted
    spike, vhh_e, _ = entry["entities"]
    assert spike["chains"] == ["A", "B"] and spike["uniprot"] == ["P0DTC2"]
    assert vhh_e["molecule"] == "VHH E" and vhh_e["organism"] == "Vicugna pacos"
    assert vhh_e["length"] == 129 and vhh_e["sequence"].startswith("QVQLVETGG")


async def test_unknown_entry_in_graphql_is_not_found(make_http, settings):
    conn = RcsbConnector(make_http(lambda r: answer("graphql_entry_missing_0zzz.json")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.entry_details("0ZZZ")
    assert err.value.code == ErrorCode.NOT_FOUND


async def test_graphql_error_with_status_200_is_reported_and_not_cached(make_http, settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return answer("graphql_error_undefined_field.json")

    conn = RcsbConnector(make_http(handler, cache=True), settings)
    for _ in range(2):
        with pytest.raises(ImmunoError) as err:
            await conn.entry_details("7KN5")
        assert err.value.code == ErrorCode.NON_DATA_RESPONSE
        assert "no_such_field" in err.value.message
    assert calls["n"] == 2


async def test_resolve_pdb(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/7KN5"):
            return answer("entry_7kn5.json")
        return answer("entry_not_found_0zzz.json", 404)

    conn = RcsbConnector(make_http(handler), settings)
    found = await conn.resolve("pdb", "7kn5")
    assert found is not None and found.canonical_id == "7KN5"
    assert (found.title or "").startswith("Crystal structure of SARS-CoV-2 receptor binding domain")
    assert found.summary["method"] == ["X-RAY DIFFRACTION"]
    assert found.summary["release_date"] == "2021-01-20" and found.summary["resolution"] == 1.87
    assert found.url == "https://www.rcsb.org/structure/7KN5"
    assert await conn.resolve("pdb", "0ZZZ") is None
    assert await conn.resolve("pdb", "not-an-id") is None
    assert await conn.resolve("uniprot", "Q15116") is None


async def test_fetch_structure_file_returns_bytes_and_provenance(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "files.rcsb.org"
        if request.url.path.endswith(".cif"):
            return answer("7kn5_head40.cif", ctype="chemical/x-cif")
        return answer("7kn5_head40.pdb", ctype="text/plain")

    conn = RcsbConnector(make_http(handler), settings)
    cif, prov = await conn.fetch_structure_file("7kn5")
    assert cif.startswith(b"data_7KN5")
    assert prov.source == "rcsb" and prov.url == "https://files.rcsb.org/download/7KN5.cif"
    assert prov.sha256 and len(prov.sha256) == 64 and (prov.license or "").startswith("CC0")
    assert prov.query == {"pdb_id": "7KN5", "format": "cif"}
    pdb, prov2 = await conn.fetch_structure_file("7KN5", fmt="pdb")
    assert pdb.startswith(b"HEADER") and (prov2.url or "").endswith("/7KN5.pdb")


async def test_unknown_id_download_is_not_found(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "files.rcsb.org":
            return answer("download_not_found.html", 404, "text/html")
        return answer("entry_not_found_0zzz.json", 404)

    conn = RcsbConnector(make_http(handler), settings)
    for fmt in ("cif", "pdb"):
        with pytest.raises(ImmunoError) as err:
            await conn.fetch_structure_file("0ZZZ", fmt=fmt)
        assert err.value.code == ErrorCode.NOT_FOUND and "no PDB entry 0ZZZ" in err.value.message


async def test_missing_legacy_pdb_file_for_an_existing_entry_says_use_cif(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "files.rcsb.org":
            return answer("download_not_found.html", 404, "text/html")
        return answer("entry_7kn5.json")

    conn = RcsbConnector(make_http(handler), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.fetch_structure_file("7KN5", fmt="pdb")
    assert err.value.code == ErrorCode.NOT_FOUND
    assert "no legacy PDB-format file" in err.value.message and "fmt='cif'" in (
        err.value.hint or ""
    )


async def test_a_web_page_or_wrong_format_is_not_a_structure_file(make_http, settings):
    page = RcsbConnector(
        make_http(lambda r: answer("download_not_found.html", 200, "text/html")), settings
    )
    with pytest.raises(ImmunoError) as err:
        await page.fetch_structure_file("7KN5")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE and "web page" in err.value.message

    wrong = RcsbConnector(
        make_http(lambda r: answer("7kn5_head40.pdb", ctype="text/plain")), settings
    )
    with pytest.raises(ImmunoError) as err:
        await wrong.fetch_structure_file("7KN5", fmt="cif")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_bad_format_argument(make_http, settings):
    conn = RcsbConnector(make_http(lambda r: pytest.fail("no request expected")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.fetch_structure_file("7KN5", fmt="mmtf")  # type: ignore[arg-type]
    assert err.value.code == ErrorCode.INVALID_INPUT


def test_structure_check_accepts_real_heads_only():
    check_structure_bytes(raw("7kn5_head40.cif"), "cif", "7KN5")
    check_structure_bytes(raw("7kn5_head40.pdb"), "pdb", "7KN5")
    with pytest.raises(ImmunoError):
        check_structure_bytes(raw("entry_7kn5.json"), "pdb", "7KN5")
    with pytest.raises(ImmunoError):
        check_structure_bytes(b"   ", "cif", "7KN5")


async def test_health_validates_shape(make_http, settings):
    good = RcsbConnector(make_http(lambda r: answer("entry_7kn5.json")), settings)
    assert (await good.health())["ok"] is True
    wrong = RcsbConnector(make_http(lambda r: answer("graphql_entry_missing_0zzz.json")), settings)
    with pytest.raises(ImmunoError) as err:
        await wrong.health()
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
