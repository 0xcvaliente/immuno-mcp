"""analysis/structure.py on real PDB entries (tests/fixtures/structure, see README there)."""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pytest

from immuno_mcp.analysis import structure as st
from immuno_mcp.core.errors import ErrorCode, ImmunoError

FX = Path(__file__).parent.parent / "fixtures" / "structure"


def text_of(name: str) -> str:
    return gzip.decompress((FX / name).read_bytes()).decode()


@pytest.fixture(scope="module")
def vfb_cif() -> str:
    return text_of("1VFB.cif.gz")


@pytest.fixture(scope="module")
def vfb_pdb() -> str:
    return text_of("1VFB.pdb.gz")


# ---- independent brute force, parsed by hand from the PDB columns -----------------------------


def pdb_heavy_atoms(text: str) -> list[tuple[str, int, str, str, np.ndarray]]:
    """ATOM records of model 1 by fixed columns, without Biotite."""
    atoms = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        assert line[16] == " ", "1VFB has no altlocs; this parser does not handle them"
        if line[76:78].strip() in ("H", "D"):
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        atoms.append((line[21], int(line[22:26]), line[26].strip(), line[17:20].strip(), xyz))
    return atoms


def brute_force_pairs(text: str, a: set[str], b: set[str]) -> dict[tuple[str, str], float]:
    atoms = pdb_heavy_atoms(text)
    side_a = [x for x in atoms if x[0] in a]
    side_b = [x for x in atoms if x[0] in b]
    xa = np.array([x[4] for x in side_a])
    xb = np.array([x[4] for x in side_b])
    dist = np.sqrt(((xa[:, None, :] - xb[None, :, :]) ** 2).sum(axis=2))
    out: dict[tuple[str, str], float] = {}
    for i, j in zip(*np.nonzero(dist <= 4.5 + 0.01), strict=True):
        ka = f"{side_a[i][0]}:{side_a[i][3]}{side_a[i][1]}{side_a[i][2]}"
        kb = f"{side_b[j][0]}:{side_b[j][3]}{side_b[j][1]}{side_b[j][2]}"
        out[(ka, kb)] = min(out.get((ka, kb), math.inf), float(dist[i, j]))
    return out


def test_contacts_match_an_independent_brute_force_distance_check(vfb_pdb):
    result = st.interface_contacts(vfb_pdb, "pdb", ["A", "B"], ["C"], cutoff=4.5)
    reported = {(p["a"], p["b"]): p["min_distance"] for p in result["residue_pairs"]}
    brute = brute_force_pairs(vfb_pdb, {"A", "B"}, {"C"})
    within = {k: d for k, d in brute.items() if d <= 4.5}
    # Pairs within 1e-3 Å of the cutoff may flip with float32 coordinates; there are none here.
    borderline = {k for k, d in brute.items() if abs(d - 4.5) < 1e-3}
    assert set(reported) - borderline == set(within) - borderline
    assert len(reported) == result["n_residue_pairs"] > 20
    for key, d in reported.items():
        assert d == pytest.approx(within[key], abs=0.006)
    atom_pairs = sum(p["atom_pairs"] for p in result["residue_pairs"])
    assert atom_pairs == result["n_atom_pairs"]


def test_cif_and_pdb_give_the_same_interface(vfb_cif, vfb_pdb):
    a = st.interface_contacts(vfb_cif, "cif", ["A", "B"], ["C"])
    b = st.interface_contacts(vfb_pdb, "pdb", ["A", "B"], ["C"])
    assert a["residue_pairs"] == b["residue_pairs"]
    assert a["buried_sasa"]["total"] == pytest.approx(b["buried_sasa"]["total"], abs=0.5)


def test_buried_surface_area_is_consistent_and_symmetric(vfb_cif):
    r = st.interface_contacts(vfb_cif, "cif", ["A", "B"], ["C"])
    bsa = r["buried_sasa"]
    assert bsa["unit"] == "Å²" and "Shrake-Rupley" in bsa["method"]
    assert bsa["total"] == pytest.approx(bsa["side_a"] + bsa["side_b"], abs=0.2)
    assert bsa["total"] == pytest.approx(
        bsa["sasa_a_alone"] + bsa["sasa_b_alone"] - bsa["sasa_complex"], abs=0.3
    )
    assert 800 < bsa["total"] < 3000  # an antibody-protein interface, not a crystal contact
    swapped = st.buried_surface_area(vfb_cif, "cif", ["C"], ["A", "B"])
    assert swapped["total"] == pytest.approx(bsa["total"], abs=0.2)
    assert swapped["side_a"] == pytest.approx(bsa["side_b"], abs=0.2)


def test_epitope_and_paratope_rows_carry_author_numbering(vfb_cif):
    r = st.epitope_paratope(vfb_cif, "cif", ["A", "B"], ["C"])
    seqs = {c.chain_id: c.sequence for c in st.read_chains(vfb_cif, "cif")}
    assert r["paratope"] and r["epitope"]
    assert {row["chain"] for row in r["paratope"]} == {"A", "B"}
    assert {row["chain"] for row in r["epitope"]} == {"C"}
    for row in r["paratope"] + r["epitope"]:
        assert set(row) >= {"chain", "res_name", "res_num", "ins_code", "min_distance"}
        assert row["min_distance"] <= 4.5 and row["partners"] >= 1
        one = st._one_letter(row["res_name"])
        assert seqs[row["chain"]][row["seq_index"]] == one
    # Residues listed are exactly the residues appearing in the residue pairs.
    in_pairs = {p["b"] for p in r["residue_pairs"]}
    assert {f"C:{x['res_name']}{x['res_num']}{x['ins_code']}" for x in r["epitope"]} == in_pairs


def test_read_chains_on_an_antibody_complex(vfb_cif, vfb_pdb):
    chains = st.read_chains(vfb_cif, "cif")
    assert [(c.chain_id, c.molecule) for c in chains] == [
        ("A", "protein"),
        ("B", "protein"),
        ("C", "protein"),
    ]
    assert [c.length for c in chains] == [len(c.sequence) for c in chains]
    assert chains[2].length == 129  # hen lysozyme, all residues observed
    assert all(c.gaps == 0 for c in chains)
    assert [c.sequence for c in st.read_chains(vfb_pdb, "pdb")] == [c.sequence for c in chains]


def test_vhh_complex_with_two_copies():
    text = text_of("1MEL.cif.gz")
    ids = [c.chain_id for c in st.read_chains(text, "cif")]
    assert ids == ["A", "B", "L", "M"]
    bound = st.epitope_paratope(text, "cif", ["A"], ["L"])
    assert bound["n_residue_pairs"] > 10 and bound["buried_sasa"]["total"] > 500
    apart = st.interface_contacts(text, "cif", ["A"], ["M"])
    assert apart["n_residue_pairs"] == 0 and apart["residue_pairs"] == []
    assert any("do not touch" in w for w in apart["warnings"])


def cif_atom_rows(text: str) -> list[dict[str, str]]:
    """Rows of the atom_site loop, split on whitespace (fine for these files)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("_atom_site."))
    names = []
    i = start
    while lines[i].startswith("_atom_site."):
        names.append(lines[i].split(".", 1)[1].strip())
        i += 1
    rows = []
    while i < len(lines) and lines[i].startswith(("ATOM", "HETATM")):
        rows.append(dict(zip(names, lines[i].split(), strict=True)))
        i += 1
    return rows


def test_altlocs_keep_the_highest_occupancy_residue_once():
    text = text_of("1EJG.cif.gz")
    rows = cif_atom_rows(text)
    assert sum(r["label_alt_id"] != "." for r in rows) > 100
    # Independent expectation: per residue number, the residue name of the altloc with the
    # highest total occupancy (1EJG has PRO/SER and LEU/ILE microheterogeneity).
    expected: dict[int, str] = {}
    for num in sorted({int(r["auth_seq_id"]) for r in rows}):
        occ: dict[tuple[str, str], float] = {}
        for r in rows:
            if int(r["auth_seq_id"]) == num:
                key = (r["label_alt_id"], r["label_comp_id"])
                occ[key] = occ.get(key, 0.0) + float(r["occupancy"])
        named = {k: v for k, v in occ.items() if k[0] != "."} or occ
        expected[num] = max(named, key=lambda k: named[k])[1]
    chain = st.read_chains(text, "cif")[0]
    assert chain.length == len(expected) == 46
    assert chain.sequence == "".join(st._one_letter(expected[n]) for n in sorted(expected))
    ls = st.load_structure(text, "cif")
    names = [(str(r), str(a)) for r, a in zip(ls.atoms.res_id, ls.atoms.atom_name, strict=True)]
    assert len(names) == len(set(names)), "an atom appears twice after altloc filtering"


def test_nucleic_acids_ions_and_water():
    text = text_of("1AAY.cif.gz")
    chains = {c.chain_id: c for c in st.read_chains(text, "cif")}
    assert chains["B"].molecule == chains["C"].molecule == "nucleic"
    assert chains["A"].molecule == "protein" and chains["A"].ligands == ["ZN"]
    assert chains["B"].sequence and set(chains["B"].sequence) <= set("ACGT")
    r = st.interface_contacts(text, "cif", ["A"], ["B", "C"])
    assert r["n_residue_pairs"] > 20
    names = {p["a"].split(":")[1][:3] for p in r["residue_pairs"]}
    assert "ZN" not in names and "HOH" not in names
    assert any("ProtOr" in w for w in r["warnings"])  # phosphorus has no ProtOr radius


def test_multiple_models_use_the_first_and_say_so():
    text = text_of("1L2Y.cif.gz")
    ls = st.load_structure(text, "cif")
    assert ls.model_count == 38 and ls.describe()["model"] == 1
    assert any("38 models" in w for w in ls.warnings)
    chain = st.read_chains(text, "cif")[0]
    assert chain.length == 20 and chain.molecule == "protein"
    assert "H" in set(ls.atoms.element.tolist())  # NMR hydrogens are present ...
    heavy = st._group_atoms(ls, ["A"])
    assert "H" not in set(heavy.element.tolist())  # ... but never used for contacts


@pytest.mark.skipif(
    not (FX / "ipsae_aurka_tpx2").exists(),
    reason="AlphaFold Server example not present; run scripts/fetch_reference_fixtures.py",
)
def test_modified_residues_and_ligands_in_a_predicted_model():
    text = text_of("ipsae_aurka_tpx2/fold_aurka_0_tpx2_0_model_0.cif.gz")
    chains = {c.chain_id: c for c in st.read_chains(text, "cif")}
    assert chains["A"].nonstandard == ["TPO"] and chains["A"].length == 276
    assert chains["C"].molecule == "other" and chains["C"].ligands == ["ATP"]
    assert chains["D"].ligands == ["MG"]
    with pytest.raises(ImmunoError) as err:
        st.interface_contacts(text, "cif", ["A"], ["C"])  # ATP is not a polymer chain
    assert "C" in err.value.message


@pytest.mark.parametrize("cutoff", [0, -1, float("nan"), 20, "abc"])
def test_bad_cutoff_is_invalid_input(vfb_cif, cutoff):
    with pytest.raises(ImmunoError) as err:
        st.interface_contacts(vfb_cif, "cif", ["A"], ["C"], cutoff=cutoff)
    assert err.value.code == ErrorCode.INVALID_INPUT


def test_unknown_chain_lists_the_real_chains(vfb_cif):
    with pytest.raises(ImmunoError) as err:
        st.interface_contacts(vfb_cif, "cif", ["H", "L"], ["C"])
    assert err.value.code == ErrorCode.INVALID_INPUT
    assert "H, L" in err.value.message and "A, B, C" in (err.value.hint or "")


def test_lowercase_chain_gets_a_suggestion(vfb_cif):
    with pytest.raises(ImmunoError) as err:
        st.interface_contacts(vfb_cif, "cif", ["a"], ["C"])
    assert "Did you mean A" in (err.value.hint or "")


@pytest.mark.parametrize(
    ("a", "b", "needle"),
    [
        ([], ["C"], "group_a is empty"),
        (["A"], [" "], "group_b is empty"),
        (["A", "C"], ["C"], "both"),
    ],
)
def test_empty_or_overlapping_groups(vfb_cif, a, b, needle):
    with pytest.raises(ImmunoError) as err:
        st.interface_contacts(vfb_cif, "cif", a, b)
    assert needle in err.value.message


def test_wrong_format_and_garbage_are_named(vfb_pdb):
    with pytest.raises(ImmunoError) as err:
        st.read_chains(vfb_pdb, "cif")
    assert "fmt='pdb'" in (err.value.hint or "")
    for bad in ("", "hello world", "<!doctype html><html></html>"):
        with pytest.raises(ImmunoError) as err:
            st.read_chains(bad, "pdb")
        assert err.value.code == ErrorCode.INVALID_INPUT
    with pytest.raises(ImmunoError):
        st.read_chains(vfb_pdb, "xyz")
