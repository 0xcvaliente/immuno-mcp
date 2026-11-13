"""Interface scores from prediction confidence: ipSAE from a PAE matrix.

The arithmetic lives in ``analysis/scores.py`` (ported from the authors' MIT-licensed code and
checked against their published outputs). This module reads workspace artifacts, works out the
token axis, and writes to the score ledger only when a candidate is named.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from pydantic import Field

from .. import __version__
from ..analysis import scores as sc
from ..analysis import structure as st
from ..core.envelope import ResponseMode, ToolResult, envelope, to_result
from ..core.errors import ErrorCode, ImmunoError
from ..core.fence import clean_tree
from ..core.metrics import metric_defaults
from .registry import ToolRegistrar
from .structure_local import local_provenance, read_artifact


def _read_pae(data: bytes) -> tuple[Any, list[str] | None, list[int] | None, str]:
    """(matrix, chain ids from the file, residue ids from the file, format name)."""
    if data[:4] == b"PK\x03\x04":
        return sc.load_pae_npz(data), None, None, "npz"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            "the PAE artifact is neither a numpy .npz nor UTF-8 JSON",
            hint=f"Expected a .npz with a 'pae' array or {sc.EXPECTED_JSON}.",
        ) from exc
    summary = sc.parse_confidence(
        structure=None, structure_format=None, summary_json=None, pae_json=text
    )
    if summary.pae_matrix is None:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"the artifact ({summary.source_format}) has no PAE matrix",
            hint="Use the file that holds the full matrix: AlphaFold 3 *_confidences.json or "
            "*_full_data_*.json, ColabFold *_scores_*.json, or <name>_pae.npz.",
        )
    return summary.pae_matrix, summary.token_chain_ids, summary.token_res_ids, summary.source_format


def register(reg: ToolRegistrar) -> None:
    app = reg.app

    @reg.tool(group="scores", profile="lab", read_only=False, idempotent=False)
    async def compute_ipsae(
        pae_artifact_id: Annotated[
            str,
            Field(
                description="Workspace artifact holding the PAE matrix: AlphaFold 3 confidences/"
                "full_data JSON, ColabFold scores JSON, AlphaFold DB PAE JSON, or a Boltz/"
                "JapanFold .npz with a 'pae' array"
            ),
        ],
        group_a: Annotated[
            list[str], Field(min_length=1, description="Chains of one side, e.g. ['H','L']")
        ],
        group_b: Annotated[
            list[str], Field(min_length=1, description="Chains of the other side, e.g. ['A']")
        ],
        structure_artifact_id: Annotated[
            str | None,
            Field(
                description="Predicted model (.cif/.pdb) of the SAME prediction. Needed when the "
                "PAE file has no chain labels (ColabFold, Boltz, JapanFold) or has modified "
                "residues; gives the exact token layout used by the reference code"
            ),
        ] = None,
        token_chain_ids: Annotated[
            list[str] | None,
            Field(description="One chain id per PAE row, if you know them and have no structure"),
        ] = None,
        pae_cutoff: Annotated[
            float, Field(gt=0, le=100, description="PAE cutoff in Å; the paper uses 10 or 15")
        ] = 10.0,
        candidate: Annotated[
            str | None,
            Field(
                description="Candidate id or name; when given, the ipSAE is written to the ledger"
            ),
        ] = None,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """ipSAE interface score (Dunbrack 2025) between two chain groups of a predicted complex.

        Computes the chain-asymmetric ipSAE (A->B aligns on A and scores B, and B->A) with d0
        from the number of residues whose PAE is below `pae_cutoff`, and ipsae = the larger of
        the two, as in the paper and its reference code (outputs reproduced exactly). Groups
        with several chains (heavy+light) are pooled as one chain. Reads the PAE from a
        workspace artifact; pass `structure_artifact_id` for files without chain labels.
        With `candidate`, writes one 'ipsae' score (variant, cutoff and both directions in its
        parameters) to the ledger; without it nothing is written. ipSAE is confidence in the
        predicted interface, NOT binding affinity: rank on it together with developability.
        """
        pae_art = read_artifact(app, pae_artifact_id)
        structure_src = None
        if structure_artifact_id:
            structure_src = read_artifact(app, structure_artifact_id)

        def run() -> dict[str, Any]:
            matrix, file_ids, file_res, pae_format = _read_pae(pae_art.data)
            layout = None
            if structure_src is not None:
                text = structure_src.data.decode("utf-8", errors="replace")
                fmt = st.sniff_format(text)
                if fmt is None:
                    raise ImmunoError(
                        ErrorCode.INVALID_INPUT,
                        "structure_artifact_id is not an mmCIF or PDB file",
                    )
                layout = sc.token_layout(text, fmt)  # type: ignore[arg-type]
            prepared = sc.prepare_ipsae_input(
                matrix,
                groups=[*group_a, *group_b],
                file_chain_ids=file_ids,
                file_res_ids=file_res,
                layout=layout,
                explicit_chain_ids=token_chain_ids,
            )
            result = sc.compute_ipsae(
                prepared.pae,
                prepared.chain_ids,
                group_a,
                group_b,
                pae_cutoff=pae_cutoff,
                nucleic_acid_chains=prepared.nucleic_chains,
            )
            result["pae_format"] = pae_format
            result["token_source"] = prepared.token_source
            result["pae_tokens"] = prepared.pae_tokens
            result["residues_in_matrix"] = len(prepared.chain_ids)
            result["_warnings"] = prepared.warnings
            return result

        result = await asyncio.to_thread(run)
        warnings: list[str] = result.pop("_warnings")
        warnings.append(sc.NOT_AFFINITY)

        ledger: dict[str, Any] = {"written": False}
        if candidate:
            cand = app.workspace.get_candidate(candidate)
            if cand is None:
                raise ImmunoError(
                    ErrorCode.NOT_FOUND,
                    f"no candidate {candidate!r} in the workspace",
                    hint="Use list_candidates, or add_candidate first. Nothing was written.",
                )
            mdef = metric_defaults("ipsae")
            params = {
                "variant": "ipSAE d0res, chain-asymmetric, max of A->B and B->A (Eq. 14-16)",
                "pae_cutoff": result["pae_cutoff"],
                "group_a": result["group_a"],
                "group_b": result["group_b"],
                "ipsae_a_to_b": result["ipsae_a_to_b"],
                "ipsae_b_to_a": result["ipsae_b_to_a"],
                "group_rule": "groups pooled as one chain each",
                "token_source": result["token_source"],
                "structure_artifact_id": structure_src.row["id"] if structure_src else None,
                "reference": sc.IPSAE_REFERENCE,
            }
            score_id = app.workspace.add_score(
                candidate_id=cand["id"],
                metric="ipsae",
                value=result["ipsae"],
                source="local",
                unit=mdef.unit if mdef else None,
                direction=mdef.direction if mdef else None,
                meaning=mdef.meaning if mdef else None,
                tool="compute_ipsae",
                tool_version=__version__,
                params=params,
                job_id=pae_art.row.get("job_id"),
                artifact_id=pae_art.row["id"],
            )
            ledger = {"written": True, "score_id": score_id, "candidate_id": cand["id"]}
        result["ledger"] = ledger

        provenance = [pae_art.provenance]
        if structure_src is not None:
            provenance.append(structure_src.provenance)
        provenance.append(
            local_provenance({"tool": "compute_ipsae", "pae_cutoff": result["pae_cutoff"]})
        )
        steps = ["rank_candidates: combine with developability; ipSAE is not affinity"]
        if not ledger["written"]:
            steps.insert(0, "compute_ipsae(..., candidate=<id>) to record the value in the ledger")
        return to_result(
            envelope(
                clean_tree(result), provenance=provenance, warnings=warnings, next_steps=steps
            ),
            response_mode,
        )
