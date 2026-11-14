"""analysis/numbering.py and the number_sequence and annotate_cdrs tools, with real ANARCII.

The sequence is the dromedary VHH of PDB 1ZVY chain A (residues 1-125 as observed). Its IMGT
CDRs come from SAbDab2 (retrieved 2026-11-14 through get_structure) and agree with the
immunology-skills check of the same entry: CDR3 GATRKYVPVRFALDQSSYDY, residues 97-116.
"""

from __future__ import annotations

import pytest
from mcp import Client

from immuno_mcp.analysis import numbering as nb
from immuno_mcp.core.errors import ErrorCode, ImmunoError

pytest.importorskip("anarcii")

VHH_1ZVY = "DVQLVESGGGSVQAGGSLRLSCAASGSTDSIEYMTWFRQAPGKAREGVAALYTHTGNTYYTDSVKGRFTISQDKAKNMAYLRMDSVKSEDTAIYTCGATRKYVPVRFALDQSSYDYWGQGTQVTV"
CDR1, CDR2, CDR3 = "GSTDSIEY", "LYTHTGNT", "GATRKYVPVRFALDQSSYDY"


async def test_imgt_regions_match_the_sabdab_cdrs():
    chain = await nb.number_chain(VHH_1ZVY, "imgt")
    assert chain.chain_type == "H" and chain.scheme == "imgt"
    assert chain.regions["CDR1"][2] == CDR1
    assert chain.regions["CDR2"][2] == CDR2
    assert chain.regions["CDR3"][2] == CDR3
    assert [chain.regions[k][:2] for k in ("CDR1", "CDR2", "CDR3")] == [
        ("27", "38"),
        ("56", "65"),
        ("105", "117"),
    ]
    assert "".join(r[2] for r in chain.regions.values()) == "".join(a for _, a in chain.positions)
    assert not chain.low_confidence


@pytest.mark.parametrize("scheme", ["kabat", "chothia", "martin", "aho"])
async def test_other_schemes_keep_residues_and_imgt_regions(scheme):
    imgt = await nb.number_chain(VHH_1ZVY, "imgt")
    other = await nb.number_chain(VHH_1ZVY, scheme)
    assert [a for _, a in other.positions] == [a for _, a in imgt.positions]
    assert {k: v[2] for k, v in other.regions.items()} == {k: v[2] for k, v in imgt.regions.items()}
    assert other.regions_defined_by == "imgt"
    assert [label for label, _ in other.positions] != [label for label, _ in imgt.positions]


async def test_a_tag_after_the_domain_is_left_unlabelled():
    chain = await nb.number_chain(VHH_1ZVY + "RGRHHHHHH", "imgt")
    assert len(chain.positions) < len(VHH_1ZVY) + 9
    assert chain.query_start == 0


async def test_a_light_chain_is_reported_as_one():
    kappa = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
    chain = await nb.number_chain(kappa, "imgt")
    assert chain.chain_type in {"K", "L"}
    assert chain.regions["CDR3"][2] == "QQHYTTPPT"


async def test_two_calls_in_a_row_do_not_leak_a_scheme():
    a = await nb.number_chain(VHH_1ZVY, "kabat")
    b = await nb.number_chain(VHH_1ZVY, "imgt")
    c = await nb.number_chain(VHH_1ZVY, "kabat")
    assert a.positions == c.positions
    assert b.regions["CDR3"][2] == CDR3


@pytest.mark.parametrize(
    ("sequence", "scheme", "fragment"),
    [
        ("MKTAYIAKQR", "imgt", "residues"),
        (VHH_1ZVY.replace("D", "B", 1), "imgt", "not standard amino acids"),
        (VHH_1ZVY, "wolfram", "unknown numbering scheme"),
    ],
)
async def test_bad_input_is_a_typed_error(sequence, scheme, fragment):
    with pytest.raises(ImmunoError) as exc:
        await nb.number_chain(sequence, scheme)
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert fragment in str(exc.value)


async def test_a_non_antibody_is_refused_not_guessed():
    lysozyme = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    with pytest.raises(ImmunoError) as exc:
        await nb.number_chain(lysozyme, "imgt")
    assert exc.value.code == ErrorCode.INVALID_INPUT
    assert "could not number" in str(exc.value)


# ---- the tools --------------------------------------------------------------------------------


async def test_tools_are_registered_and_answer(make_server):
    server = make_server()
    async with Client(server.server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"number_sequence", "annotate_cdrs"} <= names
        r = await c.call_tool("number_sequence", {"sequence": VHH_1ZVY, "scheme": "kabat"})
        assert not r.is_error, r.content[0].text
        data = r.structured_content["data"]
        assert data["scheme"] == "kabat" and data["regions_defined_by"] == "imgt"
        assert data["domain_start"] == 1 and data["domain_end"] == len(VHH_1ZVY)
        assert "positions" not in data
        assert any("IMGT definition" in w for w in r.structured_content["warnings"])
        full = await c.call_tool(
            "number_sequence", {"sequence": VHH_1ZVY, "response_mode": "standard"}
        )
        assert len(full.structured_content["data"]["positions"]) == len(VHH_1ZVY)
        cdrs = await c.call_tool("annotate_cdrs", {"sequence": VHH_1ZVY})
        rows = {r["region"]: r["sequence"] for r in cdrs.structured_content["data"]["cdrs"]}
        assert rows == {"CDR1": CDR1, "CDR2": CDR2, "CDR3": CDR3}


async def test_a_bad_scheme_is_an_error_result_not_a_crash(make_server):
    async with Client(make_server().server) as c:
        r = await c.call_tool("number_sequence", {"sequence": VHH_1ZVY, "scheme": "wolfram"})
        assert r.is_error and "unknown numbering scheme" in r.content[0].text
