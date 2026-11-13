"""analysis/scores.py: confidence formats and ipSAE.

ipSAE is checked two ways: (1) small synthetic PAE matrices whose value is worked out by hand
from Eq. 14-16 of Dunbrack (2025); (2) the AlphaFold 3 example shipped with the authors'
reference code, against their own pre-generated output files (PAE cutoff 10 Å).
"""

from __future__ import annotations

import gzip
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from immuno_mcp.analysis import scores as sc
from immuno_mcp.core.errors import ErrorCode, ImmunoError

FX = Path(__file__).parent.parent / "fixtures" / "structure"
AURKA = FX / "ipsae_aurka_tpx2"


def ptm(x: float, d0: float) -> float:
    return 1.0 / (1.0 + (x / d0) ** 2)


# ---- synthetic, worked by hand ----------------------------------------------------------------


def two_chain_matrix() -> tuple[np.ndarray, list[str]]:
    # tokens 0-1 are chain A, 2-4 chain B; only the inter-chain blocks matter
    pae = np.full((5, 5), 0.5)
    pae[0, 2:5] = [1.0, 2.0, 20.0]  # A0 -> B: two below 10 Å
    pae[1, 2:5] = [3.0, 30.0, 30.0]  # A1 -> B: one below
    pae[2, 0:2] = [0.5, 25.0]  # B2 -> A
    pae[3, 0:2] = [12.0, 12.0]  # B3 -> A: none below
    pae[4, 0:2] = [4.0, 2.0]  # B4 -> A
    return pae, ["A", "A", "B", "B", "B"]


def test_hand_computed_small_matrix():
    pae, ids = two_chain_matrix()
    r = sc.compute_ipsae(pae.tolist(), ids, ["A"], ["B"], pae_cutoff=10.0)
    # Fewer than 27 residues below the cutoff, so d0 = 1 everywhere (Eq. 15).
    a_rows = [(ptm(1, 1) + ptm(2, 1)) / 2, ptm(3, 1)]  # 0.35, 0.1
    b_rows = [ptm(0.5, 1), 0.0, (ptm(4, 1) + ptm(2, 1)) / 2]  # 0.8, 0, 0.1294
    assert r["ipsae_a_to_b"] == pytest.approx(max(a_rows), abs=1e-6) == pytest.approx(0.35)
    assert r["ipsae_b_to_a"] == pytest.approx(max(b_rows), abs=1e-6) == pytest.approx(0.8)
    assert r["ipsae"] == pytest.approx(0.8) and r["max_direction"] == "b_to_a"
    ab, ba = r["directional"]["a_to_b"], r["directional"]["b_to_a"]
    assert (ab["n0res"], ab["d0res"], ab["aligned_token"]) == (2, 1.0, 0)
    assert (ba["n0res"], ba["d0res"], ba["aligned_token"]) == (1, 1.0, 2)
    assert ab["aligned_residues_with_good_pae"] == 2 and ba["aligned_residues_with_good_pae"] == 2
    assert r["pae_cutoff"] == 10.0 and "not binding affinity" in r["note"]
    assert "Dunbrack" in r["reference"] and "10.1101/2025.02.10.637595" in r["reference"]
    assert "d0res" in r["variant"] and "asymmetric" in r["variant"]


def test_pae_equal_to_the_cutoff_is_excluded():
    pae, ids = two_chain_matrix()
    pae[0, 4] = 10.0  # exactly the cutoff: PAE_ij < cutoff is strict
    r = sc.compute_ipsae(pae, ids, ["A"], ["B"], pae_cutoff=10.0)
    assert r["directional"]["a_to_b"]["n0res"] == 2


def test_d0_follows_equation_15():
    d0 = lambda n: float(sc.d0_from_count(np.array([n]))[0])  # noqa: E731
    assert d0(0) == d0(26) == 1.0
    assert d0(27) == pytest.approx(1.24 * 12 ** (1 / 3) - 1.8)  # 1.0389, the first L >= 27
    assert d0(40) == pytest.approx(1.24 * 25 ** (1 / 3) - 1.8)  # 1.8258
    assert d0(500) == pytest.approx(1.24 * 485 ** (1 / 3) - 1.8)  # about 8, as in the paper
    assert float(sc.d0_from_count(np.array([5]), nucleic=True)[0]) == 2.0


def test_d0_grows_with_the_number_of_good_residues():
    # one A residue aligned against 40 B residues, all at 2 Å: n0 = 40, d0 = 1.8258
    n = 41
    pae = np.full((n, n), 25.0)
    pae[0, 1:] = 2.0
    ids = ["A"] + ["B"] * 40
    r = sc.compute_ipsae(pae, ids, ["A"], ["B"], pae_cutoff=10)
    d0 = 1.24 * 25 ** (1 / 3) - 1.8
    assert r["ipsae_a_to_b"] == pytest.approx(ptm(2.0, d0), abs=1e-6)
    assert r["directional"]["a_to_b"]["n0res"] == 40
    assert r["ipsae_b_to_a"] == 0.0 and r["directional"]["b_to_a"]["n0res"] == 0


def test_groups_are_pooled_as_one_chain():
    pae, ids = two_chain_matrix()
    big = np.full((6, 6), 0.5)
    big[:5, :5] = pae
    big[5, :] = 30.0
    big[:, 5] = 30.0
    big[5, 5] = 0.5
    ids6 = ["H", "L", "B", "B", "B", "X"]
    pooled = sc.compute_ipsae(big, ids6, ["H", "L"], ["B"])
    single = sc.compute_ipsae(pae, ids, ["A"], ["B"])
    assert pooled["ipsae_a_to_b"] == single["ipsae_a_to_b"]
    assert pooled["ipsae_b_to_a"] == single["ipsae_b_to_a"]
    assert "pooled" in pooled["group_rule"]


def test_nucleic_pairs_use_a_d0_floor_of_two():
    pae, ids = two_chain_matrix()
    r = sc.compute_ipsae(pae, ids, ["A"], ["B"], nucleic_acid_chains=["B"])
    assert r["nucleic_acid_pair"] is True
    assert r["ipsae_b_to_a"] == pytest.approx(ptm(0.5, 2.0))


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"group_a": ["Z"]}, "not in the PAE"),
        ({"group_a": []}, "group_a is empty"),
        ({"group_b": ["A"]}, "both groups"),
        ({"pae_cutoff": 0}, "pae_cutoff"),
        ({"pae_cutoff": -3}, "pae_cutoff"),
        ({"pae_cutoff": float("nan")}, "pae_cutoff"),
        ({"pae_cutoff": 1000}, "pae_cutoff"),
        ({"token_chain_ids": ["A", "B"]}, "token chain ids"),
    ],
)
def test_compute_ipsae_rejects_bad_arguments(kwargs, needle):
    pae, ids = two_chain_matrix()
    args = {"pae": pae, "token_chain_ids": ids, "group_a": ["A"], "group_b": ["B"]} | kwargs
    cutoff = args.pop("pae_cutoff", 10.0)
    with pytest.raises(ImmunoError) as err:
        sc.compute_ipsae(**args, pae_cutoff=cutoff)
    assert err.value.code == ErrorCode.INVALID_INPUT and needle in err.value.message


@pytest.mark.parametrize(
    "bad", [[[1.0, 2.0]], [[1.0, -1.0], [0.0, 1.0]], [[1.0, float("inf")], [0.0, 1.0]], "x"]
)
def test_bad_matrices(bad):
    with pytest.raises(ImmunoError) as err:
        sc.compute_ipsae(bad, ["A", "B"], ["A"], ["B"])
    assert err.value.code == ErrorCode.INVALID_INPUT


# ---- the authors' AlphaFold 3 example and their own output ------------------------------------


@pytest.fixture(scope="module")
def aurka() -> dict:
    if not AURKA.exists():
        pytest.skip("AlphaFold Server example not present; run scripts/fetch_reference_fixtures.py")
    cif = gzip.decompress((AURKA / "fold_aurka_0_tpx2_0_model_0.cif.gz").read_bytes()).decode()
    js = gzip.decompress(
        (AURKA / "fold_aurka_0_tpx2_0_full_data_0.no_contact_probs.json.gz").read_bytes()
    ).decode()
    summary = sc.parse_confidence(
        structure=cif, structure_format="cif", summary_json=None, pae_json=js
    )
    assert summary.pae_matrix is not None and summary.token_chain_ids is not None
    pae, ids = sc.residue_level(
        summary.pae_matrix, summary.token_chain_ids, summary.residue_token_mask
    )
    return {"cif": cif, "json": js, "summary": summary, "pae": pae, "ids": ids}


def reference_rows() -> dict[tuple[str, str, str], list[str]]:
    rows = {}
    for line in (AURKA / "fold_aurka_0_tpx2_0_model_0_10_10.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) > 20 and parts[0] in ("A", "B"):
            rows[(parts[0], parts[1], parts[4])] = parts
    return rows


def test_token_layout_matches_alphafold3_tokens(aurka):
    s = aurka["summary"]
    assert len(s.token_chain_ids) == len(s.pae_matrix) == 372
    data = json.loads(aurka["json"])
    assert s.token_chain_ids == data["token_chain_ids"]  # structure layout == AF3's own labels
    assert aurka["ids"].count("A") == 276 and aurka["ids"].count("B") == 43  # TPO collapsed


def test_reproduces_the_reference_output_at_cutoff_10(aurka):
    ref = reference_rows()
    r = sc.compute_ipsae(aurka["pae"], aurka["ids"], ["A"], ["B"], pae_cutoff=10)
    ab, ba = r["directional"]["a_to_b"], r["directional"]["b_to_a"]
    # columns: 5 ipSAE, 13 n0res, 16 d0res, 19 nres1, 20 nres2
    for row, mine, val in (
        (ref[("A", "B", "asym")], ab, r["ipsae_a_to_b"]),
        (ref[("B", "A", "asym")], ba, r["ipsae_b_to_a"]),
    ):
        assert val == pytest.approx(float(row[5]), abs=1e-6)
        assert mine["n0res"] == int(row[13])
        assert mine["d0res"] == pytest.approx(float(row[16]), abs=0.006)
        assert mine["aligned_residues_with_good_pae"] == int(row[19])
        assert mine["scored_residues_with_good_pae"] == int(row[20])
    assert r["ipsae"] == pytest.approx(float(ref[("A", "B", "max")][5]), abs=1e-6)


def test_reproduces_every_per_residue_value_of_the_reference(aurka):
    """The shipped _byres.txt was committed on 2025-04-06, before ipsae.py v4 (2026-01-03),
    whose calc_d0_array changed the floor: "now returns 1.00 instead of 1.04 for minimum value".
    v4 and the paper's Eq. 15 give d0 = 1 for L < 27; the older file used L = 27 there
    (d0 = 1.0389). So rows with n0 >= 27 must match exactly, and rows with n0 < 27 must equal the
    same mean computed with d0 = 1.0389. Nothing else may differ."""
    text = gzip.decompress(
        (AURKA / "fold_aurka_0_tpx2_0_model_0_10_10_byres.txt.gz").read_bytes()
    ).decode()
    ref = [ln.split() for ln in text.splitlines()[1:] if ln.strip()]
    ids = np.array(aurka["ids"])
    subs, got = {}, {}
    for c1, c2 in (("A", "B"), ("B", "A")):
        subs[c1] = aurka["pae"][np.ix_(ids == c1, ids == c2)]
        got[c1] = sc.ipsae_per_residue(subs[c1], 10.0)
    old_floor = 1.24 * 12 ** (1 / 3) - 1.8
    seen = {"A": 0, "B": 0}
    small = 0
    for row in ref:
        chain = row[1]
        k = seen[chain]
        seen[chain] += 1
        values, n0, d0 = got[chain]
        assert n0[k] == int(row[8]), row
        if n0[k] >= 27:
            assert values[k] == pytest.approx(float(row[15]), abs=6e-5), row
            assert d0[k] == pytest.approx(float(row[11]), abs=6e-4)
        elif n0[k] > 0:
            small += 1
            assert d0[k] == 1.0 and float(row[11]) == pytest.approx(old_floor, abs=6e-4)
            good = subs[chain][k][subs[chain][k] < 10.0]
            old_value = float(np.mean(1.0 / (1.0 + (good / old_floor) ** 2)))
            assert old_value == pytest.approx(float(row[15]), abs=6e-5), row
        else:
            assert values[k] == 0.0 == float(row[15])
    assert seen == {"A": 276, "B": 43} and small > 0


def test_json_alone_refuses_chains_with_atom_tokens(aurka):
    s = sc.parse_confidence(
        structure=None, structure_format=None, summary_json=None, pae_json=aurka["json"]
    )
    assert s.extras["multi_token_chains"] == ["A", "C"]
    with pytest.raises(ImmunoError) as err:
        sc.prepare_ipsae_input(
            s.pae_matrix,
            groups=["A", "B"],
            file_chain_ids=s.token_chain_ids,
            file_res_ids=s.token_res_ids,
        )
    assert "structure" in (err.value.hint or "")


def test_prepare_from_structure_layout_equals_direct(aurka):
    layout = sc.token_layout(aurka["cif"], "cif")
    prep = sc.prepare_ipsae_input(
        aurka["summary"].pae_matrix,
        groups=["A", "B"],
        layout=layout,
        file_chain_ids=aurka["summary"].token_chain_ids,
    )
    assert prep.token_source == "structure" and prep.pae.shape == (319, 319)
    with pytest.raises(ImmunoError):
        sc.prepare_ipsae_input(np.zeros((10, 10)), groups=["A"], layout=layout)
    with pytest.raises(ImmunoError):
        sc.prepare_ipsae_input(np.zeros((10, 10)), groups=["A"])


def test_npz_round_trip_gives_the_same_matrix(aurka):
    buf = io.BytesIO()
    np.savez_compressed(buf, pae=np.asarray(aurka["summary"].pae_matrix, dtype=np.float32))
    arr = sc.load_pae_npz(buf.getvalue())
    assert arr.shape == (372, 372)
    assert np.allclose(arr, np.asarray(aurka["summary"].pae_matrix), atol=1e-5)
    empty = io.BytesIO()
    np.savez_compressed(empty, pde=np.zeros((3, 3)))
    with pytest.raises(ImmunoError, match="no 'pae'"):
        sc.load_pae_npz(empty.getvalue())
    with pytest.raises(ImmunoError, match="npz"):
        sc.load_pae_npz(b"PK\x03\x04 not really a zip")


# ---- confidence formats ----------------------------------------------------------------------


def test_alphafold3_full_data_and_structure(aurka):
    s = aurka["summary"]
    atom = json.loads(aurka["json"])["atom_plddts"]
    assert s.plddt_mean == pytest.approx(sum(atom) / len(atom), abs=1e-3)
    assert s.extras["plddt_basis"].startswith("mean over atoms")
    assert s.source_format == "alphafold3_confidences+structure_cif"
    assert s.pae_mean == pytest.approx(float(np.mean(s.pae_matrix)), abs=1e-3)
    assert {r["metric"] for r in s.score_rows()} == {"plddt_mean", "pae_mean"}


def test_plddt_from_the_b_factor_column(aurka):
    s = sc.parse_confidence(
        structure=aurka["cif"], structure_format="cif", summary_json=None, pae_json=None
    )
    assert s.plddt_mean is not None and 50 < s.plddt_mean < 100
    assert "B-factor" in s.extras["plddt_basis"]
    assert s.ptm is None and s.pae_matrix is None


def test_experimental_b_factors_are_not_plddt():
    text = gzip.decompress((FX / "1VFB.cif.gz").read_bytes()).decode()
    with pytest.raises(ImmunoError) as err:
        sc.parse_confidence(
            structure=text, structure_format="cif", summary_json=None, pae_json=None
        )
    assert "X-RAY DIFFRACTION" in err.value.message


def test_colabfold_scores_json():
    text = (FX / "colabfold_raf1_scores_first40.json").read_text()
    obj = json.loads(text)
    s = sc.parse_confidence(structure=None, structure_format=None, summary_json=text, pae_json=None)
    assert s.source_format == "colabfold_scores"
    assert s.ptm == obj["ptm"] and s.iptm == obj["iptm"]
    assert s.plddt_mean == pytest.approx(sum(obj["plddt"]) / 40, abs=1e-3)
    assert s.chain_pair_iptm is not None and s.pae_matrix is not None
    assert s.chain_pair_iptm["A"]["B"] == s.chain_pair_iptm["B"]["A"] == 0.484
    assert s.chain_ptm == obj["per_chain_ptm"] and s.extras["actifptm"] == obj["actifptm"]
    assert len(s.pae_matrix) == 40 and s.token_chain_ids is None
    assert any("no chain labels" in w for w in s.warnings)


def test_boltz2_confidence_docs_example():
    text = (FX / "boltz2_confidence_docs_example.json").read_text()
    s = sc.parse_confidence(structure=None, structure_format=None, summary_json=text, pae_json=None)
    assert s.source_format == "boltz2_confidence"
    assert s.plddt_mean == pytest.approx(84.02) and s.ptm == 0.8425 and s.iptm == 0.8225
    assert s.extras["confidence_score"] == 0.8367 and s.extras["complex_iplddt"] == 0.8241
    assert s.chain_pair_iptm is not None
    assert s.chain_pair_iptm["0"]["1"] == 0.809 and s.chain_ptm == {"0": 0.8533, "1": 0.833}


# The objects below are BUILT from documented key names (no real file was captured).


def test_alphafold3_summary_keys_from_its_source_code():
    obj = {
        "ptm": 0.81,
        "iptm": 0.74,
        "ranking_score": 0.79,
        "fraction_disordered": 0.02,
        "has_clash": 0.0,
        "chain_pair_pae_min": [[0.76, 2.1], [2.4, 0.77]],
        "chain_pair_iptm": [[0.83, 0.74], [0.74, 0.88]],
        "chain_ptm": [0.83, 0.88],
        "chain_iptm": [0.74, 0.74],
        "chain_ids": ["H", "A"],
    }
    s = sc.parse_confidence(
        structure=None, structure_format=None, summary_json=json.dumps(obj), pae_json=None
    )
    assert s.source_format == "alphafold3_summary_confidences"
    assert s.chain_pair_iptm is not None
    assert s.chain_pair_iptm["H"]["A"] == 0.74 and s.chain_ptm == {"H": 0.83, "A": 0.88}
    assert s.extras["has_clash"] is False and s.plddt_mean is None


def test_japanfold_rows_and_manifest():
    boltz_row = json.loads((FX / "boltz2_confidence_docs_example.json").read_text())
    one = {"object": "results", "kind": "predict", "rows": [boltz_row]}
    s = sc.parse_confidence(
        structure=None, structure_format=None, summary_json=json.dumps(one), pae_json=None
    )
    assert s.source_format == "japanfold_results:boltz2_confidence" and s.iptm == 0.8225
    two = {"rows": [boltz_row | {"name": "t1"}, boltz_row | {"name": "t2"}]}
    with pytest.raises(ImmunoError) as err:
        sc.parse_confidence(
            structure=None, structure_format=None, summary_json=json.dumps(two), pae_json=None
        )
    assert "2 rows" in err.value.message and "t1" in (err.value.hint or "")
    af2ig = {
        "plddt": 91.2,
        "ptm": 0.88,
        "iptm": 0.86,
        "pae": 4.1,
        "ipae": 6.0,
        "interface_pae": 6.0,
    }
    s = sc.parse_confidence(
        structure=None, structure_format=None, summary_json=json.dumps(af2ig), pae_json=None
    )
    assert s.source_format == "japanfold_af2ig_row"
    assert (s.plddt_mean, s.pae_mean, s.extras["interface_pae"]) == (91.2, 4.1, 6.0)


@pytest.mark.parametrize(
    ("obj", "needle"),
    [
        ({"arrays": ["pae", "pde"], "absent": {}, "file": "t1_pae.npz"}, "sidecar"),
        ({"affinity_pred_value": -1.2, "affinity_probability_binary": 0.7}, "affinity"),
        ({"hello": 1}, "not a recognised confidence format"),
        ([1, 2, 3], "not a recognised confidence format"),
        ({"ptm": 2.5, "confidence_score": 0.5}, "outside"),
    ],
)
def test_unrecognised_or_wrong_json_is_invalid_input(obj, needle):
    with pytest.raises(ImmunoError) as err:
        sc.parse_confidence(
            structure=None, structure_format=None, summary_json=json.dumps(obj), pae_json=None
        )
    assert err.value.code == ErrorCode.INVALID_INPUT and needle in err.value.message
    if "recognised" in needle:
        assert "AlphaFold 3" in (err.value.hint or "")


def test_alphafold_db_pae_json():
    obj = [
        {
            "predicted_aligned_error": [[0.25, 3.0], [2.5, 0.25]],
            "max_predicted_aligned_error": 31.75,
        }
    ]
    s = sc.parse_confidence(
        structure=None, structure_format=None, summary_json=None, pae_json=json.dumps(obj)
    )
    assert s.source_format == "alphafold_db_pae" and s.pae_mean == pytest.approx(1.5)


def test_nothing_or_broken_input():
    with pytest.raises(ImmunoError, match="nothing to parse"):
        sc.parse_confidence(structure=None, structure_format=None, summary_json=None, pae_json=None)
    with pytest.raises(ImmunoError, match="not valid JSON"):
        sc.parse_confidence(
            structure=None, structure_format=None, summary_json="{oops", pae_json=None
        )
    with pytest.raises(ImmunoError, match="structure_format"):
        sc.parse_confidence(
            structure="data_x", structure_format=None, summary_json=None, pae_json=None
        )
    assert math.isfinite(sc.d0_from_count(np.array([10**6]))[0])
