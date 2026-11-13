from __future__ import annotations

import hashlib
import json
import math

import httpx
import pytest

from immuno_mcp.connectors.skempi import (
    R_KCAL,
    SKEMPI_URL,
    SkempiConnector,
    antibody_evidence,
    check_header,
    derive,
    parse_skempi,
)
from immuno_mcp.core.errors import ErrorCode, ImmunoError


@pytest.fixture
def excerpt(fixtures_dir) -> str:
    return (fixtures_dir / "skempi" / "skempi_v2_excerpt.csv").read_text(encoding="utf-8")


def _rows(excerpt: str) -> list[dict[str, str]]:
    rows, skipped = parse_skempi(excerpt)
    assert skipped == []
    return rows


def _row(rows, complex_id: str, mutation: str) -> dict[str, str]:
    return next(r for r in rows if r["#Pdb"] == complex_id and r["Mutation(s)_PDB"] == mutation)


def _csv(text: str) -> httpx.Response:
    return httpx.Response(200, text=text, headers={"content-type": "text/csv; charset=utf-8"})


def test_real_excerpt_parses_with_its_header(excerpt):
    rows = _rows(excerpt)
    assert len(rows) == 16
    assert check_header(excerpt)[:3] == ["#Pdb", "Mutation(s)_PDB", "Mutation(s)_cleaned"]
    assert rows[0]["#Pdb"] == "1CSE_E_I" and rows[0]["Protein 2"] == "Eglin c"


def test_malformed_rows_are_skipped_and_reported(excerpt):
    lines = excerpt.splitlines()
    broken = lines[1].rsplit(";", 5)[0]  # a real row with its last five fields cut off
    text = "\n".join([lines[0], broken, *lines[2:], "not a complex;x"]) + "\n"
    rows, skipped = parse_skempi(text)
    assert len(rows) == 15 and skipped == [2, 18]


def test_a_web_page_is_not_the_table():
    with pytest.raises(ImmunoError) as err:
        parse_skempi("<!DOCTYPE html><html><body>Login</body></html>")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


def test_ddg_uses_the_row_temperature(excerpt):
    rows = _rows(excerpt)
    d13 = derive(_row(rows, "1VFB_AB_C", "YA32A"))
    expected = R_KCAL * 298 * math.log(1.92e-07 / 2e-08)
    assert d13["ddg_kcal_mol"] == round(expected, 2) == 1.34
    assert d13["kd_wt_M"] == 2e-08 and d13["kd_mut_M"] == 1.92e-07
    assert d13["temperature_K"] == 298 and "temperature_note" not in d13
    assert d13["pdb"] == "1VFB" and d13["reference"] == "PMID:8703938"
    cse = derive(_row(rows, "1CSE_E_I", "LI45G"))
    assert cse["temperature_K"] == 294
    assert cse["ddg_kcal_mol"] == round(R_KCAL * 294 * math.log(5.26e-11 / 1.12e-12), 2)


def test_temperature_notes(excerpt):
    rows = _rows(excerpt)
    assumed = derive(_row(rows, "1SBN_E_I", "RI45K"))
    assert (
        assumed["temperature_K"] == 298 and assumed["temperature_note"] == "298 K assumed by SKEMPI"
    )
    missing = derive(_row(rows, "4G0N_A_B", "DA38A"))
    assert missing["temperature_K"] == 298 and "298 K used" in missing["temperature_note"]


def test_no_binding_bounds_and_approximations(excerpt):
    rows = _rows(excerpt)
    nb = derive(_row(rows, "1IAR_A_B", "EA9K"))
    assert nb["ddg_kcal_mol"] is None and nb["kd_mut_reported"] == "n.b"
    assert "no binding" in nb["ddg_note"]
    bound = derive(_row(rows, "1SBB_A_B", "LB20T,VB26Y,YB91V"))
    assert bound["ddg_qualifier"] == ">" and bound["kd_mut_reported"] == ">1.5E-04"
    assert "at least" in bound["ddg_note"]
    approx = derive(_row(rows, "2FTL_E_I", "KI15R"))
    assert approx["ddg_qualifier"] == "~"
    both = derive(_row(rows, "4I77_HL_Z", "YH97F"))  # mutant <1E-11 and wild type <1E-11
    assert both["ddg_kcal_mol"] is None and "opposite directions" in both["ddg_note"]


def test_free_text_reference_is_kept(excerpt):
    rows = _rows(excerpt)
    ref = derive(_row(rows, "1BRS_A_D", "DD35A"))["reference"]
    assert ref.startswith("Water-mediated interaction at a protein-protein interface")


def test_antibody_heuristic(excerpt):
    rows = _rows(excerpt)
    assert antibody_evidence(_row(rows, "1VFB_AB_C", "HA30A")) == "skempi_label+name"
    assert antibody_evidence(_row(rows, "3HFM_HL_Y", "YY20A")) == "skempi_label"  # HyHEL-10
    assert antibody_evidence(_row(rows, "4N8V_G_ABC", "PG16A")) is None  # immunoglobulin-like KIR
    assert antibody_evidence(_row(rows, "1CSE_E_I", "LI45G")) is None


async def test_first_use_downloads_once_and_writes_a_sidecar(make_http, settings, excerpt):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert str(request.url) == SKEMPI_URL
        return _csv(excerpt)

    conn = SkempiConnector(make_http(handler), settings)
    table = await conn.table()
    assert len(calls) == 1 and not table.reused and len(table.rows) == 16
    meta = json.loads(conn.meta_path.read_text())
    assert meta["sha256"] == hashlib.sha256(excerpt.encode()).hexdigest()
    assert meta["size"] == len(excerpt.encode()) and meta["source_url"] == SKEMPI_URL
    assert meta["retrieved_at"] and conn.csv_path.read_text() == excerpt
    assert not list(conn.data_dir.glob("*.download*"))

    again = await conn.table()
    assert len(calls) == 1 and again.reused

    # a new process (new connector) re-uses the file on disk without the network
    fresh = SkempiConnector(make_http(handler), settings)
    reused = await fresh.table()
    assert len(calls) == 1 and reused.reused and len(reused.rows) == 16
    assert fresh.provenance_for(reused).cached is True

    refreshed = await fresh.table(refresh=True)
    assert len(calls) == 2 and not refreshed.reused


async def test_a_tampered_local_copy_is_downloaded_again(make_http, settings, excerpt):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _csv(excerpt)

    conn = SkempiConnector(make_http(handler), settings)
    await conn.table()
    data = conn.csv_path.read_bytes()
    conn.csv_path.write_bytes(data.replace(b"Eglin c", b"Eglin X"))  # same size, other content
    fresh = SkempiConnector(make_http(handler), settings)
    table = await fresh.table()
    assert len(calls) == 2 and not table.reused
    assert "did not match its recorded sha256" in table.notes[0]


async def test_web_page_instead_of_csv_leaves_no_file(make_http, settings):
    page = "<!DOCTYPE html><html><body>Service unavailable</body></html>"
    conn = SkempiConnector(
        make_http(lambda r: httpx.Response(200, text=page, headers={"content-type": "text/html"})),
        settings,
    )
    with pytest.raises(ImmunoError) as err:
        await conn.table()
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
    assert not conn.csv_path.exists() and conn.local_copy() is None


async def test_csv_with_the_wrong_header_is_rejected(make_http, settings, excerpt):
    other = excerpt.replace("Affinity_wt_parsed", "Affinity_wild", 1)
    conn = SkempiConnector(make_http(lambda r: _csv(other)), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.table()
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE and not conn.csv_path.exists()


async def test_download_404_is_not_found(make_http, settings):
    conn = SkempiConnector(make_http(lambda r: httpx.Response(404, text="")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.table()
    assert err.value.code == ErrorCode.NOT_FOUND


async def test_query_filters(make_http, settings, excerpt):
    conn = SkempiConnector(make_http(lambda r: _csv(excerpt)), settings)
    by_pdb, _ = await conn.query(pdb_id="1vfb")
    assert [r["Mutation(s)_PDB"] for r in by_pdb] == ["HA30A", "YA32A", "YB32A", "YA32A,QC121A"]
    by_complex, _ = await conn.query(pdb_id="1VFB_AB_C", mutations=["YA32A", "QC121A"])
    assert [r["Mutation(s)_PDB"] for r in by_complex] == ["YA32A,QC121A"]
    any_chain, _ = await conn.query(mutations=["y32a"])
    assert [r["Mutation(s)_PDB"] for r in any_chain] == ["YA32A", "YB32A", "YA32A,QC121A"]
    lysozyme, _ = await conn.query(protein="lysozyme")
    assert len(lysozyme) == 5
    antibodies, _ = await conn.query(antibody_only=True)
    names = {r["Protein 1"] for r in antibodies}
    assert "HyHEL-10" in names and "Lebrikizumab fab" in names
    assert not any("Killer cell" in r["Protein 1"] for r in antibodies)


async def test_health_validates_the_table_and_compares_the_local_copy(make_http, settings, excerpt):
    conn = SkempiConnector(make_http(lambda r: _csv(excerpt)), settings)
    first = await conn.health()
    assert first["ok"] is True and first["detail"].startswith("16 rows")
    await conn.table()
    assert "local copy is current" in (await conn.health())["detail"]
    page = SkempiConnector(make_http(lambda r: _csv("<html>maintenance</html>")), settings)
    with pytest.raises(ImmunoError) as err:
        await page.health()
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE
