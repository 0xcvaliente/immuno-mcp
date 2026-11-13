"""PDBe connector, tested against responses captured from the live API on 2026-11-13."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from immuno_mcp.connectors.pdbe import PdbeConnector
from immuno_mcp.core.errors import ErrorCode, ImmunoError

FIX = Path(__file__).parent.parent / "fixtures" / "pdbe"


def answer(name: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, text=(FIX / name).read_text(), headers={"content-type": "application/json"}
    )


async def test_best_structures_for_pd1(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/pdbe/api/mappings/best_structures/Q15116"
        return answer("best_structures_q15116.json")

    rows, resp = await PdbeConnector(make_http(handler), settings).best_structures("q15116")
    assert len(rows) == 64 and resp.status == 200
    assert rows[0] == {
        "pdb_id": "9Q8L",
        "chain_id": "A",
        "experimental_method": "X-ray diffraction",
        "resolution": 1.85,
        "unp_start": 24,
        "unp_end": 170,
        "coverage": 0.553,
        "tax_id": 9606,
    }


async def test_entry_summary_and_molecules(make_http, settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if "summary" in request.url.path:
            return answer("entry_summary_7kn5.json")
        return answer("entry_molecules_7kn5.json")

    conn = PdbeConnector(make_http(handler), settings)
    summary, _ = await conn.entry_summary("7KN5")
    assert summary["release_date"] == "20210120" and "nanobodies" in summary["title"]
    molecules, _ = await conn.molecules("pdb_00007kn5")
    assert molecules[0]["molecule_name"] == ["Spike protein S1"]
    assert molecules[0]["in_chains"] == ["A", "B"]


async def test_unknown_entry_is_not_found(make_http, settings):
    conn = PdbeConnector(make_http(lambda r: answer("not_found_0zzz.json", 404)), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.entry_summary("0ZZZ")
    assert err.value.code == ErrorCode.NOT_FOUND and "0zzz" in err.value.message


async def test_answer_keyed_by_another_id_is_rejected(make_http, settings):
    conn = PdbeConnector(make_http(lambda r: answer("entry_summary_7kn5.json")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.entry_summary("1CRN")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_invalid_accession_is_rejected_before_any_request(make_http, settings):
    conn = PdbeConnector(make_http(lambda r: pytest.fail("no request expected")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.best_structures("PD-1")
    assert err.value.code == ErrorCode.INVALID_INPUT


async def test_health(make_http, settings):
    good = PdbeConnector(make_http(lambda r: answer("entry_summary_7kn5.json")), settings)
    assert (await good.health()) == {"ok": True, "detail": "7kn5 released 20210120"}
    page = "<!doctype html><html><body>Service unavailable</body></html>"
    html = PdbeConnector(
        make_http(lambda r: httpx.Response(200, text=page, headers={"content-type": "text/html"})),
        settings,
    )
    with pytest.raises(ImmunoError) as err:
        await html.health()
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
