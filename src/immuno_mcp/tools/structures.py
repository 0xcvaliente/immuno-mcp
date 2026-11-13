"""Structure tools: find antibody and nanobody structures (SAbDab2) and describe one PDB entry."""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import Field

from immuno_mcp.app import AppContext
from immuno_mcp.connectors.rcsb import RcsbConnector, normalize_pdb_id
from immuno_mcp.connectors.sabdab import (
    API_VERSION,
    NAME_FILTER_MAX,
    SabdabConnector,
    antigen_index,
    cdr_mismatches,
    parse_antibody_type,
    project_annotation,
    project_row,
)
from immuno_mcp.connectors.uniprot import UniprotConnector, is_accession, release, summarize
from immuno_mcp.core.connector import Connector
from immuno_mcp.core.envelope import Provenance, ResponseMode, ToolResult, envelope, to_result
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.tools.registry import ToolRegistrar

_CDR = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{1,40}$")


def _connector[C: Connector](app: AppContext, name: str, cls: type[C]) -> C:
    conn = app.connector(name)
    if not isinstance(conn, cls):
        raise ImmunoError(ErrorCode.INTERNAL, f"connector {name!r} has an unexpected type")
    return conn


def _mismatch_warning(segments: list[dict[str, Any] | None]) -> list[str]:
    bad = [m for seg in segments for m in cdr_mismatches(seg)]
    if not bad:
        return []
    return [
        f"CDR strings from SAbDab2 differ from the IMGT ranges of its own numbering for "
        f"{', '.join(bad[:6])}; re-number those chains with number_sequence before using them."
    ]


def register(reg: ToolRegistrar) -> None:
    app = reg.app

    @reg.tool(group="structures", open_world=True)
    async def search_antibody_structures(
        antigen: Annotated[
            str | None,
            Field(
                description=(
                    "Antigen name as written in PDB entries (case-insensitive substring, e.g. "
                    "'Spike', 'Programmed cell death protein 1') or a UniProt accession "
                    "(e.g. Q15116), which is first resolved to its UniProt protein name"
                )
            ),
        ] = None,
        antibody_type: Annotated[
            str | None,
            Field(
                description=(
                    "FV, FAB, FAB+FC, FV*, SD-H, SD-L or VNAR; vhh, nanobody, single-domain = "
                    "SD-H; scfv selects the scFv construct. Comma-separate several types."
                )
            ),
        ] = None,
        pdb_entry: Annotated[
            str | None, Field(description="Only antibodies in this PDB entry, e.g. 7KN5")
        ] = None,
        cdr_sequence: Annotated[
            str | None,
            Field(description="Exact amino-acid sequence of any CDR (IMGT definition)"),
        ] = None,
        max_resolution: Annotated[
            float | None,
            Field(gt=0, le=100, description="Keep structures at this resolution (Å) or better"),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100, description="Rows per page")] = 10,
        offset: Annotated[int, Field(ge=0, description="Rows to skip, for paging")] = 0,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """Find experimental antibody and nanobody structures in SAbDab2. One row per antibody.

        Each row is one antibody instance (a VH/VL pair or a single domain) in a PDB entry, so an
        entry with two copies of a Fab gives two rows. Rows carry pdb_id, antibody_type (SD-H is
        a nanobody/VHH), heavy_chain and light_chain (SAbDab2 author chain ids), the bound
        antigens (name, chain, type, organism), resolution, method and deposition date as SAbDab2
        records them (SAbDab2's release date repeats the deposition date, so get the release
        date from get_structure). response_mode 'standard' or 'full' adds the CDR3s (IMGT) and
        the antibody id.
        The result reports the total number of matches; page with offset.

        The antigen filter matches names as written in PDB entries, not gene symbols, and does
        not check species, so a human protein's name can also return mouse orthologue
        complexes. Use get_structure(pdb_id) next for RCSB metadata, UniProt accessions and the
        full SAbDab2 annotation. Not for predicted models: SAbDab2 holds experimental
        structures only.
        """
        sab = _connector(app, "sabdab", SabdabConnector)
        types, construct = parse_antibody_type(antibody_type)
        cdr = None
        if cdr_sequence is not None and cdr_sequence.strip():
            cdr = "".join(cdr_sequence.split()).upper()
            if not _CDR.match(cdr):
                raise ImmunoError(
                    ErrorCode.INVALID_INPUT,
                    f"cdr_sequence {cdr_sequence!r} is not 1-40 standard amino-acid letters",
                )
        pdb = normalize_pdb_id(pdb_entry) if pdb_entry and pdb_entry.strip() else None

        provenance: list[Provenance] = []
        warnings: list[str] = []
        name_query: str | None = None
        resolved: dict[str, Any] | None = None
        aliases: list[str] = []
        if antigen and antigen.strip():
            name = " ".join(antigen.split())
            if is_accession(name):
                uni = _connector(app, "uniprot", UniprotConnector)
                entry, uresp = await uni.entry(name)
                s = summarize(entry)
                if not s.get("name"):
                    raise ImmunoError(
                        ErrorCode.NOT_FOUND, f"uniprot: {name} has no protein name to search by"
                    )
                resolved = {k: s[k] for k in ("accession", "name", "organism") if k in s}
                aliases = [*s.get("short_names", []), *s.get("alt_names", [])]
                provenance.append(
                    uni.provenance(uresp, version=release(uresp), query={"accession": name})
                )
                warnings.append(
                    f"{name} was matched by its UniProt name, not by accession or sequence: PDB "
                    "entries that name the antigen differently are missed, and other species "
                    "with the same protein name are included (see rows[].antigens[].organism)."
                )
                name = s["name"]
            name_query = name[:NAME_FILTER_MAX].rstrip()
            if len(name) > NAME_FILTER_MAX:
                warnings.append(
                    f"SAbDab2 matches at most {NAME_FILTER_MAX} characters of an antigen name, so "
                    f"the filter was {name_query!r}; it also matches other antigens whose names "
                    "contain that text."
                )

        page, sresp = await sab.search(
            binds_to_antigen=name_query,
            antibody_types=types or None,
            construct_type=construct,
            pdb_entry=pdb,
            cdr_sequence=cdr,
            max_resolution=max_resolution,
            limit=limit,
            offset=offset,
        )
        provenance.append(sab.provenance(sresp, version=API_VERSION))
        detail = response_mode != "compact"
        index = antigen_index(page["antigen_instances"])
        rows = [project_row(r, index, detail=detail) for r in page["results"]]
        total = page["total"]
        if detail:
            warnings += _mismatch_warning(
                [
                    r.get(k)
                    for r in page["results"]
                    for k in ("heavy_like_segment", "light_like_segment")
                ]
            )

        filters = {
            "antigen_name_contains": name_query,
            "antibody_type": types or None,
            "construct_type": construct,
            "pdb_entry": pdb,
            "cdr_sequence": cdr,
            "max_resolution": max_resolution,
        }
        data: dict[str, Any] = {
            "total": total,
            "offset": offset,
            "returned": len(rows),
            "filters": {k: v for k, v in filters.items() if v is not None},
            "rows": rows,
        }
        if resolved:
            data["antigen_resolved"] = resolved

        next_steps: list[str] = []
        if offset + len(rows) < total:
            next_steps.append(
                f"{total} matches in total: call again with the same filters and "
                f"offset={offset + len(rows)} for the next page."
            )
        if rows:
            next_steps.append(
                "get_structure(pdb_id) for RCSB metadata, entity UniProt accessions and the "
                "full SAbDab2 annotation of one entry."
            )
        elif name_query:
            hint = (
                f" UniProt also names it: {', '.join(repr(a) for a in aliases[:5])}."
                if aliases
                else ""
            )
            warnings.append(
                f"No SAbDab2 antibody binds an antigen whose name contains {name_query!r}. Names "
                "come from PDB entries; gene symbols and short names often do not match." + hint
            )
        return to_result(
            envelope(
                data,
                provenance=provenance,
                warnings=warnings,
                next_steps=next_steps,
                untrusted=["rows[].antigens[].name", "antigen_resolved.name"],
            ),
            response_mode,
        )

    @reg.tool(group="structures", open_world=True)
    async def get_structure(
        pdb_id: Annotated[
            str, Field(description="PDB id, e.g. 7KN5 (the extended form pdb_00007kn5 also works)")
        ],
        include_sequences: Annotated[
            bool,
            Field(description="Add the sequence of every polymer entity and variable domain"),
        ] = False,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """Describe one PDB entry: RCSB metadata, every polymer entity and the SAbDab2 annotation.

        Returns title, method, resolution, deposit and release dates (RCSB), the primary
        citation, and one item per polymer entity with its chain ids, molecule name, organism,
        length and UniProt accessions. When the entry is in SAbDab2, 'sabdab' lists each
        antibody with its type, heavy/light chains, IMGT CDRs, germline calls (when SAbDab2 has
        them) and bound antigen chains; 'sabdab.in_sabdab' is false when SAbDab2 does not hold
        the entry. Use it after search_antibody_structures or resolve_identifier. It does not
        return coordinates; use extract_chains or interface_contacts for those.
        """
        pid = normalize_pdb_id(pdb_id)
        rcsb = _connector(app, "rcsb", RcsbConnector)
        sab = _connector(app, "sabdab", SabdabConnector)
        entry, rresp = await rcsb.entry_details(pid)
        if not include_sequences:
            for ent in entry["entities"]:
                ent.pop("sequence", None)
        provenance = [rcsb.provenance(rresp, query={"api": "graphql", "entry_id": pid})]
        warnings: list[str] = []
        annotation: dict[str, Any] | None
        try:
            detail, sresp = await sab.pdb_detail(pid)
        except ImmunoError as exc:
            if exc.code != ErrorCode.NOT_FOUND:
                annotation = None
                warnings.append(f"SAbDab2 annotation unavailable: {exc.to_message()}")
            else:
                annotation = {"in_sabdab": False}
        else:
            annotation = project_annotation(detail, sequences=include_sequences)
            provenance.append(sab.provenance(sresp, version=API_VERSION))
            warnings += _mismatch_warning(
                [
                    inst.get(k)
                    for inst in detail["antibody_instances"]
                    for k in ("heavy_like_segment", "light_like_segment")
                ]
            )
            if not annotation["antibodies"]:
                warnings.append(
                    f"SAbDab2 lists {pid} (status {annotation.get('status')}) but records no "
                    "antibody in it."
                )
        entry["sabdab"] = annotation

        next_steps = ["extract_chains to read chain sequences from the coordinates."]
        antibodies = (annotation or {}).get("antibodies") or []
        if antibodies:
            ab = antibodies[0]
            ab_chains = [d["chain"] for d in (ab.get("heavy"), ab.get("light")) if d]
            ag_chains = [a["chain"] for a in ab.get("antigens", []) if a.get("chain")]
            if ab_chains and ag_chains:
                next_steps.append(
                    f"interface_contacts between antibody chains {'+'.join(ab_chains)} and "
                    f"antigen chains {'+'.join(ag_chains)}."
                )
            next_steps.append("number_sequence on a variable domain; always state the scheme.")
        return to_result(
            envelope(
                entry,
                provenance=provenance,
                warnings=warnings,
                next_steps=next_steps,
                untrusted=[
                    "title",
                    "citation.title",
                    "citation.authors",
                    "entities[].molecule",
                    "sabdab.antibodies[].antigens[].name",
                ],
            ),
            response_mode,
        )
