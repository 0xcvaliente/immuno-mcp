"""Dataset tools: honest dataset cards, and SKEMPI 2.0 binding-change queries."""

from __future__ import annotations

import re
from typing import Annotated, Any, cast

from pydantic import Field

from ..app import AppContext
from ..connectors.skempi import (
    ANTIBODY_RULE,
    DDG_FORMULA,
    SkempiConnector,
    antibody_evidence,
    derive,
    details,
)
from ..core.envelope import Provenance, ResponseMode, ToolResult, envelope, to_result
from ..core.errors import ErrorCode, ImmunoError
from .registry import ToolRegistrar

CARDS_VERIFIED = "2026-11-13"
_PDB = re.compile(r"^[0-9][A-Za-z0-9]{3}(_[A-Za-z0-9]+_[A-Za-z0-9]+)?$")
_MUTATION = re.compile(r"^[A-Z][A-Z0-9]?-?\d+[A-Z]?[A-Z]$", re.I)

# Facts below were read from each source's own pages on CARDS_VERIFIED. Nothing here is built yet.
PLANNED: list[dict[str, Any]] = [
    {
        "name": "Observed Antibody Space (OAS)",
        "kind": "antibody repertoire sequences (unpaired and paired), cleaned, annotated, numbered",
        "license": "CC BY 4.0 (stated on the OAS home page)",
        "cite": "Olsen TH, Boyles F, Deane CM (2022) Observed Antibody Space: a diverse database of "
        "cleaned, annotated, and translated unpaired and paired antibody sequences. Protein Sci "
        "31(1):141-146. doi:10.1002/pro.4205; and Kovaltsuk A et al. (2018) J Immunol "
        "201(8):2502-2509. doi:10.4049/jimmunol.1800708",
        "access": "bulk file download only: the whole dataset, or subsets chosen with the web "
        "search forms; no query API was found",
        "homepage": "https://opig.stats.ox.ac.uk/webapps/oas/",
        "status": "planned (not built)",
    },
    {
        "name": "AB-Bind",
        "kind": "antibody binding changes on mutation (ddG)",
        "license": "none stated: the GitHub repository has no licence file. Fetch on demand for "
        "your own research; never redistribute it",
        "cite": "Sirin S, Apgar JR, Bennett EM, Keating AE (2016) AB-Bind: Antibody binding "
        "mutational database for computational affinity predictions. Protein Sci 25(2):393-409. "
        "doi:10.1002/pro.2829",
        "access": "CSV (AB-Bind_experimental_data.csv) in the GitHub repository "
        "sarahsirin/AB-Bind-Database",
        "homepage": "https://github.com/sarahsirin/AB-Bind-Database",
        "status": "planned (not built)",
    },
    {
        "name": "Ginkgo Datapoints GDPa1",
        "kind": "antibody developability assays (246 antibodies, 10 assays)",
        "license": "'cc-by-with-restrictions' (dataset LICENSE.md): CC BY 4.0 for commercial use "
        "not involving the sale, transfer or licensing of the dataset itself",
        "cite": "Arsiwala A, Bhatt R, van Niekerk L, et al. (2025) A high-throughput platform for "
        "biophysical antibody developability assessment to enable AI/ML model training. bioRxiv. "
        "doi:10.1101/2025.05.01.651684; dataset ginkgo-datapoints/GDPa1 v1.4 (2026-11-03)",
        "access": "Hugging Face dataset ginkgo-datapoints/GDPa1, gated: log in, accept the terms "
        "(name, company, work email, intended use) and set HF_TOKEN",
        "homepage": "https://huggingface.co/datasets/ginkgo-datapoints/GDPa1",
        "status": "planned (not built)",
    },
    {
        "name": "Adaptyv Bio protein design competition data",
        "kind": "measured binding of designed EGFR binders (competition rounds 1 and 2)",
        "license": "data ODbL, code Apache-2.0 (LICENSE.md in each repository)",
        "cite": "Cotet T, Krawczuk I, Stocco F, et al. (2025) Crowdsourced Protein Design: Lessons "
        "From the Adaptyv EGFR Binder Competition. bioRxiv. doi:10.1101/2025.04.17.648362",
        "access": "GitHub repositories adaptyvbio/egfr_competition_1 and "
        "adaptyvbio/egfr_competition_2 (results folders); later competitions were not checked",
        "homepage": "https://github.com/adaptyvbio/egfr_competition_1",
        "status": "planned (not built)",
    },
]


def _tool_available(app: AppContext, name: str) -> bool | None:
    catalog = app.extras.get("catalog") or []
    spec = next((s for s in catalog if s.name == name), None)
    return None if spec is None else bool(spec.available)


def _skempi_card(app: AppContext) -> dict[str, Any]:
    conn = cast(SkempiConnector, app.connector("skempi"))
    local = conn.local_copy()
    card: dict[str, Any] = {
        "name": "SKEMPI 2.0",
        "kind": "binding: measured affinity changes on mutation (Kd, ddG, kinetics) for "
        "protein-protein complexes with a structure, including antibody-antigen",
        "license": conn.info.license,
        "cite": conn.info.citation,
        "access": "CSV download (about 1.6 MB), cached in the server's home folder on the first "
        "query_binding_data call and reused",
        "homepage": conn.info.homepage,
        "status": "available",
        "tool": "query_binding_data",
        "local_copy": (
            {
                "present": True,
                "sha256": local.get("sha256"),
                "retrieved_at": local.get("retrieved_at"),
                "rows": local.get("rows"),
            }
            if local
            else {"present": False}
        ),
    }
    if _tool_available(app, "query_binding_data") is False:
        card["note"] = "query_binding_data needs the lab profile (it downloads and caches a file)."
    return card


def _sabdab_card(app: AppContext) -> dict[str, Any]:
    conn = app.connectors.get("sabdab") or app.connectors.get("sabdab2")
    if conn is None:
        return {
            "name": "SAbDab2 (Structural Antibody Database)",
            "kind": "antibody and nanobody structures",
            "license": "not stated here: the SAbDab connector is not in this build",
            "cite": "not stated here: the SAbDab connector is not in this build",
            "access": "connector 'sabdab' (structures package) is not loaded",
            "status": "planned (not built)",
        }
    tools = [t for t in ("search_antibody_structures", "get_structure") if _tool_available(app, t)]
    return {
        "name": conn.info.display_name,
        "kind": "antibody and nanobody structures",
        "license": conn.info.license,
        "cite": conn.info.citation,
        "access": f"web API through connector '{conn.info.name}'"
        + (f"; tools {', '.join(tools)}" if tools else ""),
        "homepage": conn.info.homepage,
        "status": "available",
    }


def register(reg: ToolRegistrar) -> None:
    app = reg.app

    @reg.tool(group="datasets")
    async def list_datasets(response_mode: ResponseMode = "compact") -> ToolResult:
        """Dataset cards: name, kind, licence, how to cite, access method and status (available or planned).

        Use it before relying on a dataset, to check its licence and whether this server can
        query it. Planned datasets are listed so you can tell the user they are not built yet;
        do not claim results from them. Nothing is downloaded by this tool.
        """
        cards = [_skempi_card(app), _sabdab_card(app), *PLANNED]
        return to_result(
            envelope(
                {"datasets": cards},
                provenance=Provenance(
                    source="immuno-mcp",
                    version=f"dataset cards verified against each source on {CARDS_VERIFIED}",
                ),
                warnings=[
                    "Licences and access terms can change; check the homepage before "
                    "redistributing any data."
                ],
                next_steps=[
                    "query_binding_data(pdb_id=..., antibody_only=true) for SKEMPI 2.0 rows."
                ],
            ),
            response_mode,
        )

    @reg.tool(group="datasets", profile="lab", read_only=False, open_world=True)
    async def query_binding_data(
        pdb_id: Annotated[
            str | None,
            Field(description="PDB id such as 1VFB, or a SKEMPI complex id such as 1VFB_AB_C"),
        ] = None,
        protein: Annotated[
            str | None,
            Field(min_length=2, max_length=100, description="Text in either protein name"),
        ] = None,
        mutation: Annotated[
            str | None,
            Field(
                max_length=200,
                description="Mutation as wild type, chain, number, mutant (YA32A), or without the "
                "chain (Y32A). Comma separated mutations must all be present in a row.",
            ),
        ] = None,
        antibody_only: Annotated[
            bool, Field(description="Keep antibody-antigen rows only (a heuristic, see output)")
        ] = False,
        limit: Annotated[int, Field(ge=1, le=200, description="Maximum rows")] = 20,
        refresh: Annotated[
            bool, Field(description="Download SKEMPI again even though a local copy exists")
        ] = False,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """Measured binding changes on mutation from SKEMPI 2.0: wild-type and mutant Kd, ddG, temperature, reference.

        Needs the lab profile because it may download and cache a dataset: the first call stores
        the SKEMPI 2.0 CSV (about 1.6 MB) in the server's home folder with its sha256, and later
        calls reuse that copy (refresh=true downloads it again). ddG = R*T*ln(Kd_mut/Kd_wt) in
        kcal/mol, positive meaning weaker binding; the temperature used is in every row. Bounds
        such as Kd > 1E-04 make ddG a bound (ddg_qualifier). Filter by pdb_id, protein name,
        mutation and antibody_only. These are experimental values from the cited papers, not
        predictions. Use get_passages on a row's reference PMID to read the source.
        """
        pdb = pdb_id.strip() if pdb_id else None
        if pdb and not _PDB.match(pdb):
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"not a PDB id: {pdb_id!r}",
                hint="Use a 4-character PDB id such as 1VFB, or a SKEMPI complex id like 1VFB_AB_C.",
            )
        mutations = [m.strip() for m in mutation.split(",") if m.strip()] if mutation else None
        bad = [m for m in mutations or [] if not _MUTATION.match(m)]
        if bad:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"not a mutation: {', '.join(repr(b) for b in bad)}",
                hint="Write wild type, chain, residue number, mutant, e.g. YA32A, or Y32A for "
                "any chain.",
            )
        conn = cast(SkempiConnector, app.connector("skempi"))
        found, table = await conn.query(
            pdb_id=pdb,
            protein=protein,
            mutations=mutations,
            antibody_only=antibody_only,
            refresh=refresh,
        )
        rows = []
        for raw in found[:limit]:
            row = derive(raw)
            if antibody_only:
                row["antibody_evidence"] = antibody_evidence(raw)
            if response_mode != "compact":
                row.update(details(raw))
            rows.append(row)

        warnings = list(table.notes)
        if not table.reused:
            warnings.append(
                f"Downloaded SKEMPI 2.0 ({table.meta['size']} bytes, sha256 "
                f"{table.meta['sha256'][:12]}) to <IMMUNOMCP_HOME>/datasets/skempi/; later calls reuse it."
            )
        if table.skipped_lines:
            lines = ", ".join(str(n) for n in table.skipped_lines[:10])
            warnings.append(
                f"{len(table.skipped_lines)} malformed CSV rows skipped (lines {lines})."
            )
        if antibody_only:
            warnings.append(ANTIBODY_RULE)
        if any(r.get("ddg_qualifier") in (">", "<") for r in rows):
            warnings.append("Some Kd values are bounds, so their ddG is a bound (ddg_qualifier).")
        if not found:
            warnings.append("No SKEMPI 2.0 rows matched these filters.")
        refs = sorted({r["reference"][5:] for r in rows if str(r["reference"]).startswith("PMID:")})
        filters = {
            k: v
            for k, v in (
                ("pdb_id", pdb),
                ("protein", protein),
                ("mutation", mutations),
                ("antibody_only", antibody_only or None),
            )
            if v
        }
        next_steps = []
        if refs:
            next_steps.append(f"get_passages(pmids={refs[:5]}) to read the source of the values")
        if pdb:
            next_steps.append(f"get_structure('{pdb[:4]}') for the complex")
        return to_result(
            envelope(
                {
                    "total_matched": len(found),
                    "returned": len(rows),
                    "filters": filters,
                    "ddg_formula": DDG_FORMULA,
                    "rows": rows,
                },
                provenance=conn.provenance_for(table),
                warnings=warnings,
                untrusted=[
                    "rows[].protein_1",
                    "rows[].protein_2",
                    "rows[].reference",
                    "rows[].notes",
                ],
                next_steps=next_steps,
            ),
            response_mode,
        )
