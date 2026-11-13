from __future__ import annotations

import json
from pathlib import Path

import pytest

from immuno_mcp.core.envelope import Envelope, Provenance, envelope, fit, to_result
from immuno_mcp.workspace import Workspace
from immuno_mcp.workspace.db import candidate_id


@pytest.fixture
def ws(tmp_path) -> Workspace:
    return Workspace(":memory:", artifacts_dir=tmp_path / "artifacts")


VHH = "QVQLVESGGGLVQPGGSLRLSCAASGFPVNRYSMRWYRQAPGKEREWVAGMSSAGDRSSYEDSVKGRFTISRDDARNTVYLQMNSLKPEDTAVYYCNVNVGFEYWGQGTQVTVSS"


def test_adding_the_same_molecule_twice_returns_the_same_row(ws):
    a = ws.add_candidate(kind="vhh", chains={"A": VHH}, name="nb1")
    b = ws.add_candidate(kind="vhh", chains={"A": VHH.lower()}, name="again")
    assert a["id"] == b["id"] == candidate_id("vhh", {"A": VHH})
    assert len(ws.list_candidates()) == 1 and ws.get_candidate("nb1")["id"] == a["id"]


def test_score_ledger_keeps_where_each_number_came_from(ws):
    c = ws.add_candidate(kind="vhh", chains={"A": VHH})
    ws.add_score(
        candidate_id=c["id"],
        metric="ipTM",
        value=0.81,
        source="japanfold",
        tool="boltz2",
        tool_version="2.0",
        params={"seed": 1},
        job_id="j1",
        direction="higher_better",
        meaning="confidence in chain placement, not affinity",
    )
    (row,) = ws.scores_for([c["id"]], "ipTM")
    assert row["value"] == 0.81 and row["tool"] == "boltz2" and row["params"] == {"seed": 1}
    assert "not affinity" in row["meaning"]


def test_scores_cannot_point_at_unknown_candidates(ws):
    with pytest.raises(KeyError):
        ws.add_score(candidate_id="c_missing", metric="x", value=1.0, source="t")


def test_artifacts_are_stored_once_by_content(ws):
    a = ws.store_bytes(b"data_x\n", kind="mmcif", suffix=".cif", source="rcsb")
    b = ws.store_bytes(b"data_x\n", kind="mmcif", suffix=".cif", source="rcsb")
    assert a["id"] == b["id"] and a["path"].endswith(".cif")
    assert Path(a["path"]).read_bytes() == b"data_x\n"


def test_artifacts_can_be_listed_by_job_and_kind(ws):
    ws.store_bytes(b"data_a\n", kind="mmcif", suffix=".cif", job_id="j1")
    ws.store_bytes(b"{}", kind="json", suffix=".json", job_id="j1")
    ws.store_bytes(b"data_b\n", kind="mmcif", suffix=".cif", job_id="j2")
    assert len(ws.list_artifacts()) == 3
    assert {a["kind"] for a in ws.list_artifacts(job_id="j1")} == {"mmcif", "json"}
    assert [a["job_id"] for a in ws.list_artifacts(kind="mmcif", job_id="j2")] == ["j2"]
    assert ws.list_artifacts(job_id="nope") == []


def test_spend_prefers_actual_over_estimate_per_job(ws):
    ws.record_spend(session_id="s1", provider="japanfold", usd=1.0, kind="estimate", job_id="j1")
    ws.record_spend(session_id="s1", provider="japanfold", usd=0.4, kind="actual", job_id="j1")
    ws.record_spend(session_id="s1", provider="japanfold", usd=2.0, kind="estimate", job_id="j2")
    ws.record_spend(session_id="s2", provider="japanfold", usd=5.0, kind="estimate", job_id="j3")
    assert ws.spend_total("s1") == pytest.approx(2.4)
    assert ws.spend_total() == pytest.approx(7.4)


def test_job_lifecycle(ws):
    ws.add_job(
        local_id="j1",
        provider="japanfold",
        kind="prediction",
        model="boltz2",
        request={"sequence": "AAA"},
        estimate_usd=0.5,
        session_id="s1",
    )
    job = ws.update_job("j1", status="succeeded", remote_id="r-9", actual_usd=0.4)
    assert job["status"] == "succeeded" and ws.get_job("r-9")["id"] == "j1"
    with pytest.raises(ValueError):
        ws.update_job("j1", request="x")


def test_envelope_text_is_compact_and_structured_copy_matches():
    env = envelope(
        {"rows": [1, 2]},
        provenance=Provenance(source="sabdab2", url="https://x", license="CC-BY"),
        warnings=["w"],
        untrusted=["rows"],
    )
    result = to_result(env)
    text = result.content[0].text
    assert "\n" not in text and ", " not in text
    assert json.loads(text) == result.structured_content
    assert result.structured_content["provenance"][0]["source"] == "sabdab2"
    assert "license" not in result.structured_content["provenance"][0]  # compact mode drops it
    assert "license" in to_result(env, "standard").structured_content["provenance"][0]


def test_budget_trims_the_longest_list_and_says_so():
    big = {"rows": [{"id": i, "note": "x" * 50} for i in range(2000)], "total": 2000}
    trimmed, omitted = fit(big, "compact")
    assert 0 < len(trimmed["rows"]) < 2000 and omitted["rows"] == 2000 - len(trimmed["rows"])
    assert trimmed["total"] == 2000
    result = to_result(Envelope(data=big))
    assert result.structured_content["truncated"] is True
    assert any("trimmed" in w for w in result.structured_content["warnings"])


def test_small_payloads_are_untouched():
    data = {"rows": [1, 2, 3]}
    assert fit(data, "compact") == (data, {})
