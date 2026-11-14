"""Local structure analysis tools: chains, interface contacts, epitope and paratope.

The structure comes from a PDB id (downloaded through the ``rcsb`` connector, once per call) or
from a workspace artifact id. Filesystem paths are never accepted. The analysis itself is pure
(``analysis/structure.py``) and runs off the event loop.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import biotite
from pydantic import Field

from .. import __version__
from ..analysis import structure as st
from ..app import AppContext
from ..core.envelope import Provenance, ResponseMode, ToolResult, envelope, to_result
from ..core.errors import ErrorCode, ImmunoError
from ..core.fence import clean_tree
from .registry import ToolRegistrar

MAX_ARTIFACT_BYTES = 200 * 1024 * 1024
SCHEMES = ("imgt", "kabat", "chothia", "martin", "aho")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

PdbId = Annotated[
    str | None,
    Field(description="PDB id, e.g. 1VFB. Give this OR artifact_id, not both."),
]
ArtifactId = Annotated[
    str | None,
    Field(description="Workspace artifact id (sha256) of a .cif/.pdb file, e.g. a predicted model"),
]
Cutoff = Annotated[
    float,
    Field(gt=0, le=st.MAX_CUTOFF, description="Heavy-atom distance cutoff in Å (default 4.5)"),
]


def local_provenance(query: dict[str, Any]) -> Provenance:
    return Provenance(
        source="local_analysis",
        version=f"immuno-mcp {__version__}; biotite {biotite.__version__}",
        query=query,
    )


# ---- reading inputs ------------------------------------------------------------------------


@dataclass
class ArtifactBytes:
    data: bytes
    row: dict[str, Any]
    provenance: Provenance


def read_artifact(app: AppContext, artifact_id: str) -> ArtifactBytes:
    """Bytes of a workspace artifact, checked against its sha256 id. Gzip is unpacked."""
    aid = (artifact_id or "").strip()
    row = app.workspace.get_artifact(aid) if aid else None
    if row is None:
        raise ImmunoError(
            ErrorCode.NOT_FOUND,
            f"no artifact {aid!r} in the workspace",
            hint="Artifact ids are the sha256 values listed by get_job or list_jobs.",
        )
    path = Path(row["path"])
    if not path.is_file():
        raise ImmunoError(ErrorCode.NOT_FOUND, f"the file of artifact {aid[:12]}… is missing")
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ImmunoError(ErrorCode.TOO_LARGE, f"artifact {aid[:12]}… is larger than 200 MB")
    data = path.read_bytes()
    if _HEX64.match(aid) and hashlib.sha256(data).hexdigest() != aid:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"artifact {aid[:12]}… no longer matches its sha256 id; the file was changed",
        )
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except OSError as exc:
            raise ImmunoError(ErrorCode.INVALID_INPUT, "artifact is a broken gzip file") from exc
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ImmunoError(ErrorCode.TOO_LARGE, "artifact is larger than 200 MB unpacked")
    prov = Provenance(
        source="workspace",
        sha256=row["id"],
        query={
            k: v
            for k, v in {
                "artifact_id": row["id"],
                "kind": row.get("kind"),
                "from": row.get("source"),
                "job_id": row.get("job_id"),
            }.items()
            if v
        },
    )
    return ArtifactBytes(data=data, row=row, provenance=prov)


@dataclass
class StructureSource:
    text: str
    fmt: str
    provenance: Provenance
    label: str  # "pdb:1VFB" or "artifact:<sha12>"


def _structure_format(row: dict[str, Any], text: str) -> str:
    """The format is read from the content; the artifact's declared kind is only reported."""
    sniffed = st.sniff_format(text)
    if sniffed is None:
        kind = str(row.get("kind") or "unknown")
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"artifact (kind {kind}) is not an mmCIF or PDB coordinate file",
            hint="Pass the artifact of a .cif or .pdb structure.",
        )
    return sniffed


async def load_structure_source(
    app: AppContext, pdb_id: str | None, artifact_id: str | None
) -> StructureSource:
    if bool(pdb_id) == bool(artifact_id):
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            "give exactly one of pdb_id or artifact_id",
            hint="pdb_id for a PDB entry (e.g. 1VFB); artifact_id for a file in the workspace.",
        )
    if pdb_id:
        conn = app.connector("rcsb")
        data, prov = await conn.fetch_structure_file(pdb_id.strip(), "cif")  # type: ignore[attr-defined]
        return StructureSource(
            text=data.decode("utf-8", errors="replace"),
            fmt="cif",
            provenance=prov,
            label=f"pdb:{pdb_id.strip().upper()}",
        )
    art = read_artifact(app, artifact_id or "")
    text = art.data.decode("utf-8", errors="replace")
    return StructureSource(
        text=text,
        fmt=_structure_format(art.row, text),
        provenance=art.provenance,
        label=f"artifact:{art.row['id'][:12]}",
    )


# ---- numbering (optional, agent C1's module) -----------------------------------------------


async def _number_chains(
    sequences: dict[str, str], scheme: str
) -> tuple[dict[str, dict[int, tuple[str, str | None]]], dict[str, Any], list[str]]:
    """Map sequence index -> (position label, region name) for each antibody chain.

    Never raises: anything that goes wrong becomes a warning and that chain gets no labels.
    """
    warnings: list[str] = []
    try:
        numbering = importlib.import_module("immuno_mcp.analysis.numbering")
        number_chain = numbering.number_chain
    except Exception as exc:
        return (
            {},
            {},
            [
                f"Numbering is not available ({type(exc).__name__}); regions are null. "
                "Call number_sequence on the chain sequences instead."
            ],
        )

    async def one(chain: str, seq: str) -> tuple[str, Any]:
        try:
            return chain, await number_chain(seq, scheme)
        except Exception as exc:
            return chain, exc

    results = await asyncio.gather(*(one(c, s) for c, s in sequences.items() if s))
    maps: dict[str, dict[int, tuple[str, str | None]]] = {}
    info: dict[str, Any] = {}
    for chain, nc in results:
        if isinstance(nc, BaseException):
            msg = getattr(nc, "message", None) or type(nc).__name__
            warnings.append(f"Chain {chain} could not be numbered: {msg}. Its regions are null.")
            continue
        try:
            positions = [(str(lab), str(res)) for lab, res in nc.positions if res and res != "-"]
            domain = "".join(res for _, res in positions)
            offset = sequences[chain].find(domain) if domain else -1
            if offset < 0:
                warnings.append(
                    f"Chain {chain}: the numbered domain does not match the structure sequence; "
                    "regions are null for this chain."
                )
                continue
            order = {lab: k for k, (lab, _) in enumerate(positions)}
            region_of: dict[int, str] = {}
            for name, bounds in dict(nc.regions).items():
                start, end = str(bounds[0]), str(bounds[1])
                if start in order and end in order:
                    for k in range(order[start], order[end] + 1):
                        region_of[k] = str(name)
                else:
                    warnings.append(f"Chain {chain}: region {name} has labels outside the domain.")
            if scheme != "imgt":
                warnings.append(
                    f"Chain {chain}: positions are {scheme} labels, but regions are the IMGT "
                    "definition."
                )
            maps[chain] = {
                offset + k: (lab, region_of.get(k)) for k, (lab, _) in enumerate(positions)
            }
            info[chain] = {
                "chain_type": str(getattr(nc, "chain_type", "")) or None,
                "scheme": str(getattr(nc, "scheme", scheme)),
                "numbered_residues": len(positions),
            }
        except Exception as exc:
            warnings.append(
                f"Chain {chain}: numbering output could not be read ({type(exc).__name__}); "
                "regions are null."
            )
    return maps, info, warnings


# ---- tools ---------------------------------------------------------------------------------


def _score_rows(out: dict[str, Any], a: list[str], b: list[str], source: str) -> list[dict]:
    params = {
        "group_a": a,
        "group_b": b,
        "cutoff_angstrom": out["cutoff"],
        "contact_rule": out["contact_rule"],
        "atoms_used": st.ATOM_SCOPE,
        "model": 1,
        "altloc": st.ALTLOC_RULE,
        "structure": source,
        "tool_version": __version__,
    }
    rows = [{"metric": "interface_contacts", "value": out["n_residue_pairs"], "params": params}]
    if out.get("buried_sasa"):
        rows.append(
            {
                "metric": "interface_buried_sasa",
                "value": out["buried_sasa"]["total"],
                "params": {
                    "group_a": a,
                    "group_b": b,
                    "method": st.SASA_METHOD,
                    "atoms_used": st.ATOM_SCOPE,
                    "structure": source,
                    "tool_version": __version__,
                },
            }
        )
    return rows


def register(reg: ToolRegistrar) -> None:
    app = reg.app

    @reg.tool(group="structure_local", open_world=True)
    async def extract_chains(
        pdb_id: PdbId = None,
        artifact_id: ArtifactId = None,
        include_sequences: Annotated[
            bool, Field(description="Include each chain's observed sequence (one-letter)")
        ] = False,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """List the chains of a structure: molecule type, length, gaps, modified residues, ligands.

        Reads model 1 of a PDB entry (pdb_id) or of a workspace file (artifact_id, e.g. a
        predicted model). Chain ids are AUTHOR ids (as in papers and SAbDab). Sequences are the
        residues observed in the coordinates, not the deposited SEQRES; `gaps` counts breaks
        where residues are missing. Use include_sequences=true to number a chain next
        (number_sequence); use interface_contacts or map_epitope_paratope for interfaces.
        For entry metadata (title, resolution, antigen) use get_structure instead.
        """
        src = await load_structure_source(app, pdb_id, artifact_id)
        ls = await asyncio.to_thread(st.load_structure, src.text, src.fmt)
        chains = await asyncio.to_thread(st.chains_from_loaded, ls)
        rows = []
        for c in chains:
            row: dict[str, Any] = {
                "chain_id": c.chain_id,
                "molecule": c.molecule,
                "length": c.length,
            }
            if c.length:
                row.update(first_residue=c.first_residue, last_residue=c.last_residue, gaps=c.gaps)
            if c.nonstandard:
                row["nonstandard_residues"] = c.nonstandard
            if c.ligands:
                row["ligands"] = c.ligands
            if include_sequences and c.sequence:
                row["sequence"] = c.sequence
            rows.append(row)
        data = clean_tree({"source": src.label, "structure": ls.describe(), "chains": rows})
        steps = ["interface_contacts(group_a, group_b) or map_epitope_paratope for interfaces"]
        if include_sequences:
            steps.insert(
                0, "number_sequence(sequence) to number an antibody chain; state the scheme"
            )
        else:
            steps.insert(0, "extract_chains(..., include_sequences=true) for the sequences")
        return to_result(
            envelope(
                data,
                provenance=[src.provenance, local_provenance({"tool": "extract_chains"})],
                warnings=ls.warnings,
                next_steps=steps,
            ),
            response_mode,
        )

    @reg.tool(group="structure_local", open_world=True)
    async def interface_contacts(
        group_a: Annotated[
            list[str], Field(min_length=1, description="Chain ids of one side, e.g. ['H','L']")
        ],
        group_b: Annotated[
            list[str], Field(min_length=1, description="Chain ids of the other side, e.g. ['A']")
        ],
        pdb_id: PdbId = None,
        artifact_id: ArtifactId = None,
        cutoff: Cutoff = 4.5,
        include_residues: Annotated[
            bool, Field(description="List residue pairs and interface residues (default true)")
        ] = True,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """Residue contacts and buried surface area between two groups of chains in one structure.

        A residue pair counts when any heavy-atom pair is within `cutoff` Å (default 4.5).
        Uses model 1, the highest-occupancy altloc, and polymer residues only (water, ions,
        ligands and glycans left out). Buried surface area is SASA(A)+SASA(B)-SASA(A+B) in Å²
        (Shrake-Rupley, probe 1.4 Å). Residues use author chain ids and numbering with
        insertion codes. Returns `scores` rows (interface_contacts, interface_buried_sasa) in the
        score-ledger format; this tool writes nothing. These are geometric facts of the
        given model, not binding strength. For antibody-antigen interfaces with CDR labels,
        use map_epitope_paratope.
        """
        src = await load_structure_source(app, pdb_id, artifact_id)
        out = await asyncio.to_thread(
            st.interface_contacts, src.text, src.fmt, group_a, group_b, cutoff
        )
        warnings = out.pop("warnings")
        data: dict[str, Any] = {"source": src.label, **out}
        if not include_residues:
            for key in ("residue_pairs", "residues_a", "residues_b"):
                data.pop(key, None)
        data["scores"] = _score_rows(out, out["group_a"], out["group_b"], src.label)
        return to_result(
            envelope(
                clean_tree(data),
                provenance=[
                    src.provenance,
                    local_provenance({"tool": "interface_contacts", "cutoff": out["cutoff"]}),
                ],
                warnings=warnings,
                next_steps=[
                    "map_epitope_paratope(antibody_chains, antigen_chains, scheme='imgt') "
                    "for CDR-labelled paratope and epitope",
                ],
            ),
            response_mode,
        )

    @reg.tool(group="structure_local", open_world=True)
    async def map_epitope_paratope(
        antibody_chains: Annotated[
            list[str],
            Field(min_length=1, description="Antibody chain ids: VH and VL, or the VHH chain"),
        ],
        antigen_chains: Annotated[list[str], Field(min_length=1, description="Antigen chain ids")],
        pdb_id: PdbId = None,
        artifact_id: ArtifactId = None,
        scheme: Annotated[
            str | None,
            Field(
                description="Numbering scheme to label paratope residues with CDR/framework "
                "regions: imgt, kabat, chothia, martin or aho. Empty: no labels."
            ),
        ] = None,
        cutoff: Cutoff = 4.5,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """Epitope and paratope residues of an antibody or nanobody bound to its antigen.

        Paratope = antibody residues with a heavy atom within `cutoff` Å of the antigen;
        epitope = antigen residues within `cutoff` Å of the antibody. Each residue has its
        author chain id, residue name, number and insertion code, closest distance, number of
        partner residues and buried SASA (Å²). With `scheme`, paratope residues also get their
        position and region (e.g. CDR3) under that scheme; if numbering is unavailable the
        regions are null and a warning says why. Model 1, polymer residues only. Residue pairs
        are listed with response_mode='standard' or 'full'. This is the contact footprint in
        this structure, not an affinity or an experimental epitope map.
        """
        if scheme is not None:
            scheme = scheme.strip().lower() or None
            if scheme and scheme not in SCHEMES:
                raise ImmunoError(
                    ErrorCode.INVALID_INPUT,
                    f"unknown numbering scheme {scheme!r}",
                    hint=f"Use one of: {', '.join(SCHEMES)}.",
                )
        src = await load_structure_source(app, pdb_id, artifact_id)

        def run() -> tuple[dict[str, Any], dict[str, str]]:
            ls = st.load_structure(src.text, src.fmt)
            out = st.epitope_paratope_loaded(ls, antibody_chains, antigen_chains, cutoff)
            seqs = {c.chain_id: c.sequence for c in st.chains_from_loaded(ls)}
            return out, {c: seqs.get(c, "") for c in out["antibody_chains"]}

        out, ab_seqs = await asyncio.to_thread(run)
        warnings: list[str] = out.pop("warnings")
        numbering: dict[str, Any] | None = None
        if scheme:
            maps, info, nwarn = await _number_chains(ab_seqs, scheme)
            warnings += nwarn
            for row in out["paratope"]:
                hit = maps.get(row["chain"], {}).get(row["seq_index"])
                row["position"] = hit[0] if hit else None
                row["region"] = hit[1] if hit else None
            numbering = {"scheme": scheme, "chains": info} if info else None
            if numbering is None:
                for row in out["paratope"]:
                    row["region"] = None
        data = {
            "source": src.label,
            **out,
            "regions": numbering,
            "scores": _score_rows(out, out["antibody_chains"], out["antigen_chains"], src.label),
        }
        if response_mode == "compact":
            # Keep the residue lists whole within the compact budget; pairs on request.
            data.pop("residue_pairs", None)
        return to_result(
            envelope(
                clean_tree(data),
                provenance=[
                    src.provenance,
                    local_provenance(
                        {"tool": "map_epitope_paratope", "cutoff": out["cutoff"], "scheme": scheme}
                    ),
                ],
                warnings=warnings,
                next_steps=[
                    "annotate_cdrs(sequence, scheme) for full CDR definitions of each chain",
                ],
            ),
            response_mode,
        )
