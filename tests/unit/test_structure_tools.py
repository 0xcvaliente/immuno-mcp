"""MCP tools extract_chains, interface_contacts, map_epitope_paratope and compute_ipsae.

A fake ``rcsb`` connector serves the real 1VFB file; workspace artifacts are real files.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import sys
import types
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pytest
from mcp import Client

import immuno_mcp.connectors  # noqa: F401  (register the real rcsb before the fake replaces it)
from immuno_mcp.analysis import scores as sc
from immuno_mcp.analysis import structure as st
from immuno_mcp.core.connector import CONNECTORS, Connector, ConnectorInfo
from immuno_mcp.core.envelope import Provenance
from immuno_mcp.core.errors import ErrorCode, ImmunoError

FX = Path(__file__).parent.parent / "fixtures" / "structure"
VFB = gzip.decompress((FX / "1VFB.cif.gz").read_bytes())
AURKA_DIR = FX / "ipsae_aurka_tpx2"
_HAVE_AURKA = AURKA_DIR.exists()
needs_aurka = pytest.mark.skipif(
    not _HAVE_AURKA,
    reason="AlphaFold Server example not present; run scripts/fetch_reference_fixtures.py",
)
AURKA_CIF = (
    gzip.decompress((AURKA_DIR / "fold_aurka_0_tpx2_0_model_0.cif.gz").read_bytes())
    if _HAVE_AURKA
    else b""
)
AURKA_JSON = (
    gzip.decompress(
        (AURKA_DIR / "fold_aurka_0_tpx2_0_full_data_0.no_contact_probs.json.gz").read_bytes()
    )
    if _HAVE_AURKA
    else b""
)


class FakeRcsb(Connector):
    info: ClassVar[ConnectorInfo] = ConnectorInfo(
        name="rcsb",
        display_name="RCSB PDB (test double)",
        kind="structure",
        base_url="https://files.rcsb.org",
        homepage="https://www.rcsb.org",
        license="CC0 1.0",
        citation="test",
    )
    calls: ClassVar[list[tuple[str, str]]] = []

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "detail": "fake"}

    async def fetch_structure_file(
        self, pdb_id: str, fmt: Literal["cif", "pdb"] = "cif"
    ) -> tuple[bytes, Provenance]:
        FakeRcsb.calls.append((pdb_id, fmt))
        if pdb_id.upper() != "1VFB":
            raise ImmunoError(ErrorCode.NOT_FOUND, f"rcsb: {pdb_id} not found")
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{fmt}"
        return VFB, Provenance(source="rcsb", url=url, sha256=hashlib.sha256(VFB).hexdigest())


@pytest.fixture
def server(make_server, monkeypatch):
    monkeypatch.setitem(CONNECTORS, "rcsb", FakeRcsb)
    FakeRcsb.calls = []
    return make_server()


def add_artifact(app, tmp_path: Path, data: bytes, kind: str, suffix: str) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path = tmp_path / f"{digest}{suffix}"
    path.write_bytes(data)
    app.workspace.register_artifact(path, digest, len(data), kind=kind, source="test")
    return digest


async def call(server, name: str, args: dict[str, Any]):
    async with Client(server.server) as c:
        return await c.call_tool(name, args)


# ---- extract_chains ---------------------------------------------------------------------------


async def test_extract_chains_from_a_pdb_id(server):
    r = await call(server, "extract_chains", {"pdb_id": "1VFB"})
    assert not r.is_error
    data = r.structured_content["data"]
    assert [c["chain_id"] for c in data["chains"]] == ["A", "B", "C"]
    assert all("sequence" not in c for c in data["chains"])
    assert data["structure"]["chain_ids"].startswith("author")
    sources = [p["source"] for p in r.structured_content["provenance"]]
    assert sources == ["rcsb", "local_analysis"]
    assert FakeRcsb.calls == [("1VFB", "cif")]


@needs_aurka
async def test_extract_chains_with_sequences_from_an_artifact(server, tmp_path):
    aid = add_artifact(server.app, tmp_path, AURKA_CIF, "mmcif", ".cif")
    r = await call(server, "extract_chains", {"artifact_id": aid, "include_sequences": True})
    assert not r.is_error
    chains = {c["chain_id"]: c for c in r.structured_content["data"]["chains"]}
    assert chains["A"]["nonstandard_residues"] == ["TPO"] and len(chains["A"]["sequence"]) == 276
    assert chains["C"]["molecule"] == "other" and "sequence" not in chains["C"]
    assert r.structured_content["provenance"][0]["source"] == "workspace"
    assert FakeRcsb.calls == []


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({}, "INVALID_INPUT"),
        ({"pdb_id": "1VFB", "artifact_id": "ab" * 32}, "INVALID_INPUT"),
        ({"artifact_id": "0" * 64}, "NOT_FOUND"),
        ({"pdb_id": "9ZZZ"}, "NOT_FOUND"),
    ],
)
async def test_structure_source_errors(server, args, code):
    r = await call(server, "extract_chains", args)
    assert r.is_error and f"[{code}]" in r.content[0].text


@needs_aurka
async def test_changed_or_non_structure_artifacts_are_refused(server, tmp_path):
    aid = add_artifact(server.app, tmp_path, VFB, "mmcif", ".cif")
    (tmp_path / f"{aid}.cif").write_bytes(VFB + b"\n# edited\n")
    r = await call(server, "extract_chains", {"artifact_id": aid})
    assert r.is_error and "no longer matches" in r.content[0].text
    jid = add_artifact(server.app, tmp_path, AURKA_JSON, "json", ".json")
    r = await call(server, "extract_chains", {"artifact_id": jid})
    assert r.is_error and "not an mmCIF or PDB" in r.content[0].text


async def test_gzipped_artifacts_are_read(server, tmp_path):
    aid = add_artifact(server.app, tmp_path, gzip.compress(VFB), "mmcif", ".cif.gz")
    r = await call(server, "extract_chains", {"artifact_id": aid})
    assert not r.is_error and len(r.structured_content["data"]["chains"]) == 3


# ---- interface_contacts ------------------------------------------------------------------------


async def test_interface_contacts_returns_scores_rows(server):
    r = await call(
        server, "interface_contacts", {"pdb_id": "1VFB", "group_a": ["A", "B"], "group_b": ["C"]}
    )
    assert not r.is_error
    data = r.structured_content["data"]
    direct = st.interface_contacts(VFB.decode(), "cif", ["A", "B"], ["C"])
    assert data["n_residue_pairs"] == direct["n_residue_pairs"]
    rows = {s["metric"]: s for s in data["scores"]}
    assert set(rows) == {"interface_contacts", "interface_buried_sasa"}
    assert rows["interface_contacts"]["value"] == direct["n_residue_pairs"]
    assert rows["interface_buried_sasa"]["value"] == direct["buried_sasa"]["total"]
    assert rows["interface_contacts"]["params"]["cutoff_angstrom"] == 4.5
    assert all(set(s) == {"metric", "value", "params"} for s in data["scores"])
    assert server.app.workspace.scores_for() == []  # this tool never writes the ledger


async def test_interface_contacts_without_residue_lists(server):
    r = await call(
        server,
        "interface_contacts",
        {
            "pdb_id": "1VFB",
            "group_a": ["B"],
            "group_b": ["C"],
            "include_residues": False,
            "cutoff": 4.0,
        },
    )
    data = r.structured_content["data"]
    assert "residue_pairs" not in data and "residues_a" not in data
    assert data["cutoff"] == 4.0 and data["n_residue_pairs"] > 0


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ({"group_a": ["H"], "group_b": ["C"]}, "not found among polymer chains"),
        # schema violations are rejected by the SDK before the tool runs; message names the field
        ({"group_a": [], "group_b": ["C"]}, "group_a"),
        ({"group_a": ["A"], "group_b": ["C"], "cutoff": 0}, "cutoff"),
        ({"group_a": ["A"], "group_b": ["C"], "cutoff": 50}, "cutoff"),
        ({"group_a": ["A"], "group_b": ["A"]}, "in both"),
    ],
)
async def test_interface_contacts_argument_errors(server, args, needle):
    r = await call(server, "interface_contacts", {"pdb_id": "1VFB", **args})
    assert r.is_error and needle in r.content[0].text


# ---- map_epitope_paratope -----------------------------------------------------------------------

REGIONS = {
    "FR1": ("1", "25"),
    "CDR1": ("26", "35"),
    "FR2": ("36", "49"),
    "CDR2": ("50", "65"),
    "FR3": ("66", "94"),
    "CDR3": ("95", "102"),
}


def fake_numbering(fail: bool = False) -> types.ModuleType:
    """Stands in for agent C1's module: positions are 1-based indices, fixed regions."""
    mod = types.ModuleType("immuno_mcp.analysis.numbering")

    async def number_chain(sequence: str, scheme: str = "imgt"):
        if fail:
            raise ImmunoError(ErrorCode.NEEDS_EXTRA, "anarcii is not installed")
        positions = [(str(i + 1), aa) for i, aa in enumerate(sequence)]
        regions = {k: (a, b, sequence[int(a) - 1 : int(b)]) for k, (a, b) in REGIONS.items()}
        return types.SimpleNamespace(
            chain_type="H", scheme=scheme, positions=positions, regions=regions
        )

    mod.number_chain = number_chain  # type: ignore[attr-defined]
    return mod


def expected_region(seq_index: int) -> str | None:
    pos = seq_index + 1
    for name, (a, b) in REGIONS.items():
        if int(a) <= pos <= int(b):
            return name
    return None


async def test_epitope_paratope_with_region_labels(server, monkeypatch):
    monkeypatch.setitem(sys.modules, "immuno_mcp.analysis.numbering", fake_numbering())
    r = await call(
        server,
        "map_epitope_paratope",
        {
            "pdb_id": "1VFB",
            "antibody_chains": ["A", "B"],
            "antigen_chains": ["C"],
            "scheme": "IMGT",
        },
    )
    assert not r.is_error, r.content[0].text
    assert "truncated" not in r.structured_content  # compact output keeps the lists whole
    data = r.structured_content["data"]
    assert "residue_pairs" not in data and data["n_residue_pairs"] > 0
    assert data["regions"]["scheme"] == "imgt" and set(data["regions"]["chains"]) == {"A", "B"}
    for row in data["paratope"]:
        assert row["position"] == str(row["seq_index"] + 1)
        assert row["region"] == expected_region(row["seq_index"])
    assert {row["chain"] for row in data["epitope"]} == {"C"}
    assert {s["metric"] for s in data["scores"]} == {"interface_contacts", "interface_buried_sasa"}


async def test_numbering_unavailable_gives_null_regions_not_an_error(server, monkeypatch):
    monkeypatch.setitem(sys.modules, "immuno_mcp.analysis.numbering", None)
    r = await call(
        server,
        "map_epitope_paratope",
        {"pdb_id": "1VFB", "antibody_chains": ["B"], "antigen_chains": ["C"], "scheme": "kabat"},
    )
    assert not r.is_error
    assert r.structured_content["data"]["regions"] is None
    assert all(row["region"] is None for row in r.structured_content["data"]["paratope"])
    assert any("Numbering is not available" in w for w in r.structured_content["warnings"])


async def test_numbering_backend_failure_is_a_warning(server, monkeypatch):
    monkeypatch.setitem(sys.modules, "immuno_mcp.analysis.numbering", fake_numbering(fail=True))
    r = await call(
        server,
        "map_epitope_paratope",
        {"pdb_id": "1VFB", "antibody_chains": ["B"], "antigen_chains": ["C"], "scheme": "imgt"},
    )
    assert not r.is_error and r.structured_content["data"]["regions"] is None
    assert any("anarcii is not installed" in w for w in r.structured_content["warnings"])


async def test_epitope_paratope_without_scheme_and_bad_scheme(server):
    r = await call(
        server,
        "map_epitope_paratope",
        {"pdb_id": "1VFB", "antibody_chains": ["A", "B"], "antigen_chains": ["C"]},
    )
    assert not r.is_error and r.structured_content["data"]["regions"] is None
    assert "region" not in r.structured_content["data"]["paratope"][0]
    bad = await call(
        server,
        "map_epitope_paratope",
        {"pdb_id": "1VFB", "antibody_chains": ["B"], "antigen_chains": ["C"], "scheme": "abm"},
    )
    assert bad.is_error and "unknown numbering scheme" in bad.content[0].text


# ---- compute_ipsae -------------------------------------------------------------------------------


def npz_bytes(matrix: Any) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, pae=np.asarray(matrix, dtype=np.float32))
    return buf.getvalue()


@needs_aurka
async def test_compute_ipsae_reproduces_the_reference_example(server, tmp_path):
    pae_id = add_artifact(server.app, tmp_path, AURKA_JSON, "json", ".json")
    cif_id = add_artifact(server.app, tmp_path, AURKA_CIF, "mmcif", ".cif")
    r = await call(
        server,
        "compute_ipsae",
        {
            "pae_artifact_id": pae_id,
            "structure_artifact_id": cif_id,
            "group_a": ["A"],
            "group_b": ["B"],
        },
    )
    assert not r.is_error, r.content[0].text
    data = r.structured_content["data"]
    assert data["ipsae"] == pytest.approx(0.866498, abs=1e-6)  # authors' output, cutoff 10 Å
    assert data["ipsae_a_to_b"] == pytest.approx(0.448952, abs=1e-6)
    assert data["pae_cutoff"] == 10.0 and "d0res" in data["variant"]
    assert data["token_source"] == "structure" and data["residues_in_matrix"] == 319
    assert data["ledger"] == {"written": False}
    assert any("not binding affinity" in w for w in r.structured_content["warnings"])
    assert [p["source"] for p in r.structured_content["provenance"]] == [
        "workspace",
        "workspace",
        "local_analysis",
    ]


@needs_aurka
async def test_compute_ipsae_from_npz_needs_the_structure(server, tmp_path):
    matrix = sc.parse_confidence(
        structure=None, structure_format=None, summary_json=None, pae_json=AURKA_JSON.decode()
    ).pae_matrix
    npz_id = add_artifact(server.app, tmp_path, npz_bytes(matrix), "other", ".npz")
    cif_id = add_artifact(server.app, tmp_path, AURKA_CIF, "mmcif", ".cif")
    alone = await call(
        server, "compute_ipsae", {"pae_artifact_id": npz_id, "group_a": ["A"], "group_b": ["B"]}
    )
    assert alone.is_error and "no chain labels" in alone.content[0].text
    ok = await call(
        server,
        "compute_ipsae",
        {
            "pae_artifact_id": npz_id,
            "structure_artifact_id": cif_id,
            "group_a": ["A"],
            "group_b": ["B"],
        },
    )
    assert not ok.is_error
    assert ok.structured_content["data"]["ipsae"] == pytest.approx(0.866498, abs=2e-5)
    assert ok.structured_content["data"]["pae_format"] == "npz"


@needs_aurka
async def test_compute_ipsae_json_alone_refuses_modified_residue_chains(server, tmp_path):
    pae_id = add_artifact(server.app, tmp_path, AURKA_JSON, "json", ".json")
    r = await call(
        server, "compute_ipsae", {"pae_artifact_id": pae_id, "group_a": ["A"], "group_b": ["B"]}
    )
    assert r.is_error and "structure_artifact_id" in r.content[0].text


@needs_aurka
async def test_compute_ipsae_writes_the_ledger_only_with_a_candidate(server, tmp_path):
    ws = server.app.workspace
    cand = ws.add_candidate(kind="other", chains={"A": "MKT", "B": "GSA"}, name="aurka_tpx2")
    pae_id = add_artifact(server.app, tmp_path, AURKA_JSON, "json", ".json")
    cif_id = add_artifact(server.app, tmp_path, AURKA_CIF, "mmcif", ".cif")
    args = {
        "pae_artifact_id": pae_id,
        "structure_artifact_id": cif_id,
        "group_a": ["A"],
        "group_b": ["B"],
        "pae_cutoff": 15,
    }
    missing = await call(server, "compute_ipsae", {**args, "candidate": "nobody"})
    assert missing.is_error and "[NOT_FOUND]" in missing.content[0].text
    assert ws.scores_for() == []
    r = await call(server, "compute_ipsae", {**args, "candidate": "aurka_tpx2"})
    assert not r.is_error
    ledger = r.structured_content["data"]["ledger"]
    assert ledger["written"] and ledger["candidate_id"] == cand["id"]
    (row,) = ws.scores_for([cand["id"]])
    assert row["metric"] == "ipsae" and row["value"] == r.structured_content["data"]["ipsae"]
    assert row["unit"] == "0-1" and row["direction"] == "higher_better"
    assert "not binding affinity" in row["meaning"].lower()
    assert row["tool"] == "compute_ipsae" and row["artifact_id"] == pae_id
    assert row["params"]["pae_cutoff"] == 15.0 and "Eq. 14-16" in row["params"]["variant"]
    assert row["params"]["structure_artifact_id"] == cif_id


@needs_aurka
async def test_compute_ipsae_argument_and_format_errors(server, tmp_path):
    pae_id = add_artifact(server.app, tmp_path, AURKA_JSON, "json", ".json")
    cif_id = add_artifact(server.app, tmp_path, AURKA_CIF, "mmcif", ".cif")
    base = {"pae_artifact_id": pae_id, "structure_artifact_id": cif_id}
    cases: list[tuple[dict[str, Any], str]] = [
        ({"group_a": ["A"], "group_b": ["Q"]}, "not in the PAE"),
        ({"group_a": [], "group_b": ["B"]}, "group_a"),
        ({"group_a": ["A"], "group_b": ["B"], "pae_cutoff": 0}, "pae_cutoff"),
    ]
    for args, needle in cases:
        r = await call(server, "compute_ipsae", {**base, **args})
        assert r.is_error and needle in r.content[0].text
    junk_id = add_artifact(server.app, tmp_path, b'{"hello": "world"}', "json", ".json")
    r = await call(
        server, "compute_ipsae", {"pae_artifact_id": junk_id, "group_a": ["A"], "group_b": ["B"]}
    )
    assert r.is_error and "not a recognised confidence format" in r.content[0].text
    vfb_id = add_artifact(server.app, tmp_path, VFB, "mmcif", ".cif")
    r = await call(
        server,
        "compute_ipsae",
        {
            "pae_artifact_id": pae_id,
            "structure_artifact_id": vfb_id,
            "group_a": ["A"],
            "group_b": ["B"],
        },
    )
    assert r.is_error and "tokens" in r.content[0].text


async def test_profiles(make_server, monkeypatch):
    monkeypatch.setitem(CONNECTORS, "rcsb", FakeRcsb)
    built = make_server(profile="readonly")
    async with Client(built.server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert {"extract_chains", "interface_contacts", "map_epitope_paratope"} <= names
    assert "compute_ipsae" not in names
    spec = next(s for s in built.catalog if s.name == "compute_ipsae")
    assert spec.profile == "lab" and not spec.read_only
