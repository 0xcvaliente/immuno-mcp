"""SAbDab2 connector, tested against responses captured from the live API on 2026-11-13."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from immuno_mcp.connectors import sabdab as sab
from immuno_mcp.connectors.sabdab import SabdabConnector
from immuno_mcp.core.errors import ErrorCode, ImmunoError

FIX = Path(__file__).parent.parent / "fixtures" / "sabdab"


def body(name: str) -> str:
    return (FIX / name).read_text()


def answer(name: str, status: int = 200, ctype: str = "application/json") -> httpx.Response:
    return httpx.Response(status, text=body(name), headers={"content-type": ctype})


def test_constants_match_the_published_openapi_spec():
    spec = json.loads(body("openapi_2.1.7_trimmed.json"))
    schemas = spec["components"]["schemas"]
    assert spec["info"]["version"] == sab.API_VERSION
    assert "CC BY 4.0" in spec["info"]["license"]["name"]
    assert tuple(schemas["AntibodyType"]["enum"]) == sab.ANTIBODY_TYPES
    assert set(sab.CONSTRUCT_TYPES) == set(schemas["ConstructType"]["enum"]) - {"OTHER"}
    params = {
        p["name"]: p
        for p in spec["paths"]["/frontend/antibody-instance-search"]["get"]["parameters"]
    }
    assert params["binds_to_antigen"]["schema"]["maxLength"] == sab.NAME_FILTER_MAX
    assert {"antibody_type", "construct_type", "pdb_entry", "cdr_sequence", "resolution"} <= set(
        params
    )


def test_pdb_ids_are_translated_to_the_extended_form():
    assert sab.sabdab_pdb_id("7kn5") == "pdb_00007kn5"
    assert sab.sabdab_pdb_id("PDB_00007KN5") == "pdb_00007kn5"
    assert sab.plain_pdb_id("pdb_00007kn5") == "7KN5"
    with pytest.raises(ImmunoError) as err:
        sab.sabdab_pdb_id("7KN")
    assert err.value.code == ErrorCode.INVALID_INPUT


@pytest.mark.parametrize(
    ("value", "types", "construct"),
    [
        ("vhh", ["SD-H"], None),
        ("Nanobody", ["SD-H"], None),
        ("single-domain", ["SD-H"], None),
        ("fab, FV", ["FAB", "FV"], None),
        ("Fab+Fc", ["FAB+FC"], None),
        ("scFv", [], "SCFV"),
        ("fv,scfv", ["FV"], "SCFV"),
        (None, [], None),
    ],
)
def test_antibody_type_aliases(value, types, construct):
    assert sab.parse_antibody_type(value) == (types, construct)


def test_unknown_antibody_type_lists_the_valid_values():
    with pytest.raises(ImmunoError) as err:
        sab.parse_antibody_type("IgG")
    assert err.value.code == ErrorCode.INVALID_INPUT
    hint = err.value.hint or ""
    assert "SD-H" in hint and "vhh" in hint and "scfv" in hint


async def test_search_sends_the_documented_parameters(make_http, settings):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer("search_pd1_limit3.json")

    conn = SabdabConnector(make_http(handler), settings)
    page, resp = await conn.search(
        binds_to_antigen="Programmed cell death protein 1",
        antibody_types=["FAB", "FV"],
        pdb_entry="5B8C",
        max_resolution=3.0,
        limit=3,
    )
    params = seen[0].url.params
    assert seen[0].url.path == "/api/frontend/antibody-instance-search"
    assert params["binds_to_antigen"] == "Programmed cell death pro"  # 25-character limit
    assert params.get_list("antibody_type") == ["FAB", "FV"]
    assert params["pdb_entry"] == "pdb_00005b8c"
    assert params["resolution"] == "3.0" and params["order_by"] == "id"
    assert page["total"] == 51 and len(page["results"]) == 3
    assert resp.status == 200


async def test_422_keeps_the_allowed_values_from_sabdab(make_http, settings):
    conn = SabdabConnector(
        make_http(lambda r: answer("search_422_antibody_type_vhh.json", 422)), settings
    )
    with pytest.raises(ImmunoError) as err:
        await conn.search(antibody_types=["VHH"], limit=1)
    assert err.value.code == ErrorCode.INVALID_INPUT
    assert "'SD-H'" in err.value.message
    assert (err.value.hint or "").startswith("SAbDab2 says antibody_type: Input should be 'FV'")


async def test_entry_missing_from_sabdab_is_not_found(make_http, settings):
    conn = SabdabConnector(make_http(lambda r: answer("frontend_pdb_1crn_404.json", 404)), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.pdb_detail("1CRN")
    assert err.value.code == ErrorCode.NOT_FOUND and "not in SAbDab2" in err.value.message


async def test_the_retired_site_page_served_as_200_is_not_data(make_http, settings):
    conn = SabdabConnector(
        make_http(lambda r: answer("old_site_app_page.html", ctype="text/html; charset=utf-8")),
        settings,
    )
    with pytest.raises(ImmunoError) as err:
        await conn.search(binds_to_antigen="Spike")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_a_search_page_without_antigen_instances_is_rejected(make_http, settings):
    # /stats/summary is valid JSON but not a search page: the shape check must catch it.
    conn = SabdabConnector(make_http(lambda r: answer("stats_summary.json")), settings)
    with pytest.raises(ImmunoError) as err:
        await conn.search(binds_to_antigen="Spike")
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


async def test_health_checks_the_shape_of_the_stats(make_http, settings):
    good = SabdabConnector(make_http(lambda r: answer("stats_summary.json")), settings)
    result = await good.health()
    assert result["ok"] is True and "22828 antibody instances" in result["detail"]

    wrong = SabdabConnector(make_http(lambda r: answer("search_empty_pd-1.json")), settings)
    with pytest.raises(ImmunoError) as err:
        await wrong.health()
    assert err.value.code == ErrorCode.NON_DATA_RESPONSE


def test_annotation_of_7kn5_lists_four_nanobodies_bound_to_spike():
    detail = json.loads(body("frontend_pdb_7kn5.json"))
    ann = sab.project_annotation(detail)
    assert ann["in_sabdab"] is True and ann["cdr_scheme"] == "imgt"
    abs_ = ann["antibodies"]
    assert [a["antibody_type"] for a in abs_] == ["SD-H"] * 4
    assert [a["heavy"]["chain"] for a in abs_] == ["C", "D", "E", "F"]
    assert all("light" not in a for a in abs_)
    # Chain C also touches two 1,2-ethanediol molecules; only the polymer antigen is reported.
    assert abs_[0]["antigens"] == [
        {
            "name": "Spike protein S1",
            "chain": "A",
            "type": "PROTEIN",
            "organism": "severe acute respiratory syndrome coronavirus 2",
        }
    ]
    assert abs_[0]["heavy"]["cdr3"] == "ALTVGTYYSGNYHYTCSDDMDY"
    assert "sequence" not in abs_[0]["heavy"]
    with_seq = sab.project_annotation(detail, sequences=True)
    assert with_seq["antibodies"][0]["heavy"]["sequence"].startswith("QVQLVETGG")


def test_cdr_strings_agree_with_imgt_ranges_of_the_numbering():
    detail = json.loads(body("frontend_pdb_7kn5.json"))
    page = json.loads(body("search_pd1_limit3.json"))
    segments = [i["heavy_like_segment"] for i in detail["antibody_instances"]] + [
        r[k] for r in page["results"] for k in ("heavy_like_segment", "light_like_segment")
    ]
    assert all(sab.cdr_mismatches(s) == [] for s in segments)
    altered = dict(segments[0], cdr3="AAAA")
    assert sab.cdr_mismatches(altered) == ["C:cdr3"]


def test_small_molecule_antigen_is_kept_when_no_polymer_is_bound_but_ions_are_not():
    page = json.loads(body("antigen_instances_first2.json"))
    index = sab.antigen_index(page["results"])
    bindings = [b for a in page["results"] for b in a["antibody_bindings"]]
    found = sab.bound_antigens(bindings, index)
    assert [a["type"] for a in found] == ["HAPTEN"]
    assert found[0]["name"].startswith("N-(2,6-dimethylphenyl)")


def test_search_rows_carry_chains_antigen_and_dates():
    page = json.loads(body("search_pd1_limit3.json"))
    index = sab.antigen_index(page["antigen_instances"])
    rows = [sab.project_row(r, index, detail=False) for r in page["results"]]
    assert rows[0]["pdb_id"] == "5B8C" and rows[0]["antibody_type"] == "FV"
    assert (rows[0]["heavy_chain"], rows[0]["light_chain"]) == ("B", "A")
    assert rows[0]["antigens"][0]["name"] == "Programmed cell death protein 1"
    assert rows[0]["deposition_date"] == "2016-06-14" and "release_date" not in rows[0]
    assert "cdr3" not in rows[0]
    detailed = sab.project_row(page["results"][0], index, detail=True)
    assert detailed["cdr3"] == {"H": "ARRDYRFDMGFDY", "L": "QHSRDLPLT"}
    assert detailed["cdr_scheme"] == "imgt"


async def test_third_party_text_is_cleaned(make_http, settings):
    dirty = body("search_pd1_limit3.json").replace(
        "Programmed cell death protein 1", "Programmed cell death protein 1" + chr(0x202E)
    )
    conn = SabdabConnector(
        make_http(
            lambda r: httpx.Response(200, text=dirty, headers={"content-type": "application/json"})
        ),
        settings,
    )
    page, _ = await conn.search(binds_to_antigen="programmed")
    names = [str((a["polymer_instance"] or {}).get("name")) for a in page["antigen_instances"]]
    assert "Programmed cell death protein 1" in names
    assert all(chr(0x202E) not in n for n in names)
