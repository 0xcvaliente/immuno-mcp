"""Numbering tools: number_sequence and annotate_cdrs (ANARCII, local, no network)."""

from __future__ import annotations

from importlib import metadata
from typing import Annotated, Any

from pydantic import Field

from .. import __version__
from ..analysis import numbering as nb
from ..core.envelope import Provenance, ResponseMode, ToolResult, envelope, to_result
from .registry import ToolRegistrar

Sequence = Annotated[
    str,
    Field(description="One variable domain (VH, VHH or VL) as one-letter amino acids"),
]
Scheme = Annotated[
    str,
    Field(description="Residue labels: imgt, kabat, chothia, martin or aho. Regions are IMGT."),
]


def _anarcii_version() -> str:
    try:
        return metadata.version("anarcii")
    except metadata.PackageNotFoundError:
        return "unknown"


def _provenance(tool: str, scheme: str) -> Provenance:
    return Provenance(
        source="local_analysis",
        version=f"immuno-mcp {__version__}; anarcii {_anarcii_version()}",
        query={"tool": tool, "scheme": scheme},
    )


def _warnings(chain: nb.NumberedChain, input_length: int) -> list[str]:
    out: list[str] = []
    if chain.scheme != "imgt":
        out.append(
            f"Residue labels are {chain.scheme}, but the regions (FR1-FR4, CDR1-CDR3) are the "
            f"IMGT definition: a {chain.scheme} CDR boundary can differ by several residues."
        )
    if chain.low_confidence:
        out.append(
            f"ANARCII score {chain.score} is below {nb.LOW_SCORE}: on heavy-chain domains this "
            "floor separated conventional VH and VHH from shark VNAR, so check that this is a "
            "VH, VHH or VL domain. The floor was not measured for light chains."
        )
    n = len(chain.positions)
    if n < input_length:
        out.append(
            f"{input_length - n} of {input_length} input residues lie outside the numbered "
            "domain (tag, linker or constant region) and have no label."
        )
    return out


def _region_rows(chain: nb.NumberedChain) -> list[dict[str, Any]]:
    return [
        {"region": name, "start": a, "end": b, "sequence": seq, "length": len(seq)}
        for name, (a, b, seq) in chain.regions.items()
    ]


def register(reg: ToolRegistrar) -> None:
    @reg.tool(group="numbering")
    async def number_sequence(
        sequence: Sequence,
        scheme: Scheme = "imgt",
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """Number an antibody variable domain: chain type, regions and, in standard or full mode, every label.

        Runs ANARCII locally. Always say which scheme the labels are in. The regions are the
        IMGT definition whatever the scheme. `domain_start` and `domain_end` are 1-based
        positions in your input. The ANARCII score is a numbering score, not a quality score.
        """
        seq = nb.clean_sequence(sequence)
        chain = await nb.number_chain(seq, scheme)
        data: dict[str, Any] = {
            "chain_type": chain.chain_type,
            "scheme": chain.scheme,
            "regions_defined_by": chain.regions_defined_by,
            "anarcii_score": chain.score,
            "input_length": len(seq),
            "domain_start": chain.query_start + 1,
            "domain_end": chain.query_end + 1,
            "regions": _region_rows(chain),
        }
        if response_mode != "compact":
            data["positions"] = [[label, aa] for label, aa in chain.positions]
        return to_result(
            envelope(
                data,
                provenance=[_provenance("number_sequence", chain.scheme)],
                warnings=_warnings(chain, len(seq)),
                next_steps=[
                    "annotate_cdrs(sequence) for the CDR sequences alone",
                    "response_mode='standard' adds the label of every residue",
                ],
            ),
            response_mode,
        )

    @reg.tool(group="numbering")
    async def annotate_cdrs(
        sequence: Sequence,
        scheme: Scheme = "imgt",
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """CDR1, CDR2 and CDR3 of a variable domain: sequence, first and last label, length.

        The CDRs are the IMGT definition (27-38, 56-65, 105-117), labelled in the scheme asked
        for. Framework regions come with response_mode='standard'. Runs ANARCII locally.
        """
        seq = nb.clean_sequence(sequence)
        chain = await nb.number_chain(seq, scheme)
        rows = _region_rows(chain)
        data: dict[str, Any] = {
            "chain_type": chain.chain_type,
            "scheme": chain.scheme,
            "regions_defined_by": chain.regions_defined_by,
            "cdrs": [r for r in rows if r["region"].startswith("CDR")],
        }
        if response_mode != "compact":
            data["frameworks"] = [r for r in rows if r["region"].startswith("FR")]
        return to_result(
            envelope(
                data,
                provenance=[_provenance("annotate_cdrs", chain.scheme)],
                warnings=_warnings(chain, len(seq)),
                next_steps=[
                    "search_antibody_structures(cdr_sequence=...) to find structures with the same CDR"
                ],
            ),
            response_mode,
        )
