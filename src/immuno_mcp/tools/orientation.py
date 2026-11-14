"""Orientation tools: how to use the server, what is connected, and identifier resolution."""

from __future__ import annotations

import asyncio
import importlib.util
import re
import time
from typing import Annotated, Any

from pydantic import Field

from .. import __version__
from ..core.envelope import ResponseMode, ToolResult, envelope, to_result
from ..core.errors import ErrorCode, ImmunoError
from ..core.settings import PROFILE_HELP
from .registry import ToolRegistrar

SCOPE = (
    "Research use only. Not clinical decision support. Not for engineering pathogens or for "
    "designing escape from vaccines, antibodies or other countermeasures."
)

# Each step is (tool name, why). Steps whose tool is not available are flagged, not hidden.
WORKFLOWS: dict[str, dict[str, Any]] = {
    "structures": {
        "title": "Find and characterise antibody or nanobody structures",
        "steps": [
            ("resolve_identifier", "confirm the target's UniProt or PDB id before using it"),
            ("search_antibody_structures", "find antibody or VHH structures for the target"),
            ("get_structure", "chains, antigen, resolution, method, deposition date"),
            ("extract_chains", "pull the sequences out of the structure"),
            ("number_sequence", "number a chain; always state the scheme"),
            ("annotate_cdrs", "CDR boundaries in that scheme"),
        ],
    },
    "triage": {
        "title": "Triage candidate sequences",
        "steps": [
            ("add_candidate", "register each sequence in the workspace"),
            ("number_sequence", "confirm it is an antibody domain and which chain type"),
            ("annotate_cdrs", "CDRs under a named scheme"),
            ("scan_liabilities", "motifs that degrade, glycosylate or aggregate"),
            ("profile_developability", "charge, hydrophobicity and related sequence metrics"),
            ("rank_candidates", "rank on more than the binding score"),
            ("export_score_table", "scores with where each one came from"),
        ],
    },
    "design": {
        "title": "Design or predict with hosted models (spends money)",
        "steps": [
            ("get_structure", "prepare the target structure and choose the epitope chain"),
            ("estimate_cost", "see the cost before anything is submitted"),
            ("submit_design", "binder design on a hosted service; needs the compute profile"),
            ("get_job", "wait for the job; results are saved to the workspace"),
            ("ingest_scores", "record pLDDT, PAE and ipTM with their source"),
            ("compute_ipsae", "interface score from PAE; the variant used is recorded"),
            ("rank_candidates", "confidence is not affinity; rank with developability too"),
            ("export_score_table", "table plus a methods paragraph"),
        ],
    },
    "immunology": {
        "title": "Epitopes, T cell receptors and MHC binding",
        "steps": [
            ("resolve_identifier", "confirm the antigen"),
            ("search_epitopes", "known epitopes and assays (IEDB)"),
            ("search_tcr_specificity", "TCRs with known specificity (VDJdb, iReceptor)"),
            ("predict_mhc_binding", "MHC class I or II binding prediction for peptides"),
        ],
    },
    "literature": {
        "title": "Evidence from the literature",
        "steps": [
            ("search_literature", "PubMed search with entity annotation"),
            ("get_passages", "citable passages; cite the stable key given"),
        ],
    },
    "mutations": {
        "title": "Effect of mutations on binding",
        "steps": [
            ("query_binding_data", "measured binding changes on mutation (SKEMPI 2.0)"),
        ],
    },
    "external_results": {
        "title": "Analyse results produced outside immuno-mcp (a CLI download, a notebook, a "
        "lab export)",
        "steps": [
            (
                "import_artifact",
                "list the inbox (no arguments), then import the files by name",
            ),
            ("extract_chains", "chains and sequences of the imported model"),
            ("interface_contacts", "residue contacts between two chain groups"),
            ("map_epitope_paratope", "or: epitope and paratope residues with CDR labels"),
            ("compute_ipsae", "interface score from the imported PAE file and the model"),
        ],
    },
}

_ID_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("doi", re.compile(r"^10\.\d{4,9}/\S+$")),
    ("pmcid", re.compile(r"^PMC\d+$", re.I)),
    ("pdb", re.compile(r"^[0-9][A-Za-z0-9]{3}$")),
    (
        "uniprot",
        re.compile(
            r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$",
            re.I,
        ),
    ),
    ("pmid", re.compile(r"^\d{5,9}$")),
]


def infer_kinds(value: str) -> list[str]:
    v = value.strip()
    return [kind for kind, pat in _ID_PATTERNS if pat.match(v)]


def register(reg: ToolRegistrar) -> None:
    app = reg.app

    @reg.tool(group="orientation")
    async def workflow_help(
        topic: Annotated[
            str | None, Field(description="One of the topics listed, or empty for all")
        ] = None,
    ) -> ToolResult:
        """Standard workflows and which tool to call at each step. Call this first."""
        available = {s.name for s in reg.catalog if s.available}
        wanted = {topic: WORKFLOWS[topic]} if topic in WORKFLOWS else WORKFLOWS
        if topic and topic not in WORKFLOWS:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"unknown topic {topic!r}",
                hint=f"Choose one of: {', '.join(WORKFLOWS)}",
            )
        out = {
            name: {
                "title": wf["title"],
                "steps": [
                    {"tool": t, "why": why, **({} if t in available else {"available": False})}
                    for t, why in wf["steps"]
                ],
            }
            for name, wf in wanted.items()
        }
        return to_result(
            envelope(
                out,
                next_steps=[
                    "Call get_server_capabilities to see connectors and the active profile."
                ],
            )
        )

    @reg.tool(group="orientation")
    async def get_server_capabilities() -> ToolResult:
        """Version, active profile, every tool, every connector and which optional parts are installed."""
        from .. import connectors as connector_pkg
        from .. import tools as tool_pkg

        tools: dict[str, list[dict[str, Any]]] = {}
        for s in reg.catalog:
            tools.setdefault(s.group, []).append(
                {
                    "name": s.name,
                    "available": s.available,
                    **({"needs_profile": s.profile} if not s.available else {}),
                    **({"spends_money": True} if s.spends_money else {}),
                }
            )
        data = {
            "version": __version__,
            "profile": app.settings.profile,
            "profiles": PROFILE_HELP,
            "scope": SCOPE,
            "spend_cap_usd": app.settings.max_spend_usd,
            "tools": tools,
            "connectors": [
                {
                    "name": c.info.name,
                    "kind": c.info.kind,
                    "tier": c.info.tier,
                    "configured": c.configured(),
                    **({"needs_key": c.info.auth_env} if not c.configured() else {}),
                }
                for c in sorted(app.connectors.values(), key=lambda c: c.info.name)
            ],
            "optional": {
                "anarcii (numbering)": importlib.util.find_spec("anarcii") is not None,
                "duckdb (local datasets)": importlib.util.find_spec("duckdb") is not None,
            },
            "load_failures": {
                **{f"connector:{k}": v for k, v in connector_pkg.IMPORT_FAILURES.items()},
                **{f"tool:{k}": v for k, v in tool_pkg.IMPORT_FAILURES.items()},
            },
        }
        return to_result(envelope(data))

    @reg.tool(group="orientation", open_world=True)
    async def diagnostics(
        live: Annotated[
            bool, Field(description="Call every connector's health check (uses the network)")
        ] = False,
    ) -> ToolResult:
        """Check configuration, and optionally every connector's live health, workspace and spend."""
        report: dict[str, Any] = {
            "workspace": {
                "path": str(app.settings.workspace_path),
                "candidates": len(app.workspace.list_candidates(limit=10_000)),
                "spent_usd_this_session": app.workspace.spend_total(app.session_id),
                "spend_cap_usd": app.settings.max_spend_usd,
            },
            "cache": app.cache.stats(),
        }

        async def check(name: str) -> tuple[str, dict[str, Any]]:
            conn = app.connectors[name]
            if not conn.configured():
                return name, {"ok": False, "code": "NEEDS_KEY", "needs": conn.info.auth_env}
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(conn.health(), timeout=20)
                return name, {
                    "ok": bool(result.get("ok")),
                    "ms": int((time.monotonic() - started) * 1000),
                    "detail": result.get("detail"),
                }
            except ImmunoError as exc:
                return name, {"ok": False, "code": str(exc.code), "message": exc.message}
            except NotImplementedError:
                return name, {"ok": None, "detail": "no health check implemented"}
            except TimeoutError:
                return name, {"ok": False, "code": "TIMEOUT"}

        if live:
            results = await asyncio.gather(*(check(n) for n in sorted(app.connectors)))
            report["connectors"] = dict(results)
            failing = [n for n, r in results if r.get("ok") is False]
            next_steps = [f"Failing: {', '.join(failing)}."] if failing else []
        else:
            report["connectors"] = {
                n: {"configured": c.configured()} for n, c in sorted(app.connectors.items())
            }
            next_steps = ["Call diagnostics(live=true) to test every connector over the network."]
        return to_result(envelope(report, next_steps=next_steps))

    @reg.tool(group="find", open_world=True)
    async def resolve_identifier(
        value: Annotated[str, Field(description="PDB id, UniProt accession, PMID, PMCID or DOI")],
        kind: Annotated[
            str | None, Field(description="pdb, uniprot, pmid, pmcid or doi; guessed if empty")
        ] = None,
        response_mode: ResponseMode = "compact",
    ) -> ToolResult:
        """Confirm an identifier against its source and return its canonical form and title."""
        value = value.strip()
        kinds = [kind.lower()] if kind else infer_kinds(value)
        if not kinds:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"cannot tell what kind of identifier {value!r} is",
                hint="Pass kind= one of pdb, uniprot, pmid, pmcid, doi.",
            )
        tried: list[str] = []
        for k in kinds:
            for conn in app.connectors.values():
                if k not in conn.info.resolves or not conn.configured():
                    continue
                tried.append(conn.info.name)
                found = await conn.resolve(k, value)
                if found is not None:
                    return to_result(
                        envelope(
                            found.model_dump(exclude_none=True),
                            provenance=conn.provenance(url=found.url),
                            untrusted=["title", "summary"],
                        ),
                        response_mode,
                    )
        if not tried:
            raise ImmunoError(
                ErrorCode.NOT_CONFIGURED,
                f"no connector in this build can resolve {'/'.join(kinds)} identifiers",
            )
        raise ImmunoError(
            ErrorCode.NOT_FOUND,
            f"{value!r} was not found as {'/'.join(kinds)} (tried {', '.join(tried)})",
            hint="Check for typos. Do not write an identifier you could not resolve.",
        )
