"""SAbDab2, the Structural Antibody Database (Oxford Protein Informatics Group).

API base https://sabdab.opig.stats.ox.ac.uk/api (OpenAPI version 2.1.7, checked 2026-11-13).
What the live service does, and how this connector handles it:

* PDB entries use extended 12-character ids: 7KN5 is ``pdb_00007kn5``; ``/pdb/7kn5`` is a 404.
* ``/frontend/antibody-instance-search`` takes the filters of ``/antibody-instances`` and also
  returns the bound antigen instances (names, chains), so one call yields complete rows.
* ``binds_to_antigen`` is a case-insensitive substring match on antigen names, at most 25
  characters. ``/antigen-instances?antigen_name=`` found nothing even for names that exist
  ('Spike'), so it is not used.
* ``release_date`` equalled ``deposition_date`` in every record checked, while RCSB gives 7KN5 a
  later release date. Only the deposition date is reported.
* Chain ids are SAbDab2 author chain ids. They equal the PDB ``auth_asym_id`` unless one PDB chain
  carries two or more variable domains (scFv, diabodies), which SAbDab2 splits into separate chains.
* CDR strings follow IMGT definitions on SAbDab2's IMGT numbering (positions 27-38, 56-65 and
  105-117); ``cdr_mismatches`` re-checks that against the numbering returned.
* Invalid filter values answer HTTP 422 with the allowed values; that text is kept in the error.
* The retired site (opig.stats.ox.ac.uk/webapps/sabdab-sabpred/...) now serves the new app page.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from immuno_mcp.connectors.rcsb import normalize_pdb_id
from immuno_mcp.core.connector import Connector, ConnectorInfo, register
from immuno_mcp.core.errors import ErrorCode, ImmunoError
from immuno_mcp.core.fence import clean_tree
from immuno_mcp.core.http import HttpResponse

BASE_URL = "https://sabdab.opig.stats.ox.ac.uk/api"
API_VERSION = "2.1.7"
ANTIBODY_TYPES = ("FV", "FAB", "FAB+FC", "FV*", "SD-H", "SD-L", "VNAR")
CONSTRUCT_TYPES = ("SCFV", "DIABODY", "CODV", "DVD", "SCDB", "FULLAB")
NAME_FILTER_MAX = 25  # maxLength of binds_to_antigen in the OpenAPI spec
CDR_SCHEME = "imgt"
_IMGT_CDRS = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}
_NEVER_ANTIGEN = frozenset({"ION", "WATER", "CHEMICAL_ELEMENT"})

_TYPE_ALIASES: dict[str, str] = {
    **{t.lower(): t for t in ANTIBODY_TYPES},
    "vhh": "SD-H",
    "nanobody": "SD-H",
    "nanobodies": "SD-H",
    "single-domain": "SD-H",
    "single domain": "SD-H",
    "sdab": "SD-H",
}
_CONSTRUCT_ALIASES: dict[str, str] = {c.lower(): c for c in CONSTRUCT_TYPES}
TYPE_HELP = (
    f"SAbDab2 antibody types: {', '.join(ANTIBODY_TYPES)}. Aliases: vhh, nanobody, "
    "single-domain and sdab mean SD-H. Construct types (scFv and others) are also accepted: "
    f"{', '.join(c.lower() for c in CONSTRUCT_TYPES)}. Separate several values with commas."
)


def sabdab_pdb_id(pdb_id: str) -> str:
    """'7KN5' -> 'pdb_00007kn5' (the id SAbDab2 uses)."""
    pid = normalize_pdb_id(pdb_id)
    return f"pdb_0000{pid.lower()}" if len(pid) == 4 else pid.lower()


def plain_pdb_id(sabdab_id: str) -> str:
    """'pdb_00007kn5' -> '7KN5'."""
    s = (sabdab_id or "").lower()
    return s[8:].upper() if s.startswith("pdb_0000") else s


def parse_antibody_type(value: str | None) -> tuple[list[str], str | None]:
    """Map user words to (SAbDab2 antibody types, construct type). INVALID_INPUT lists the options."""
    if value is None or not value.strip():
        return [], None
    types: list[str] = []
    construct: str | None = None
    for raw in value.split(","):
        token = " ".join(raw.strip().lower().replace("_", "-").split())
        if not token:
            continue
        if token in _TYPE_ALIASES:
            if _TYPE_ALIASES[token] not in types:
                types.append(_TYPE_ALIASES[token])
        elif token in _CONSTRUCT_ALIASES:
            if construct and construct != _CONSTRUCT_ALIASES[token]:
                raise ImmunoError(
                    ErrorCode.INVALID_INPUT, "only one construct type can be searched at a time"
                )
            construct = _CONSTRUCT_ALIASES[token]
        else:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT, f"unknown antibody_type {raw.strip()!r}", hint=TYPE_HELP
            )
    return types, construct


def _hint_from_422(message: str) -> str | None:
    """Pull 'parameter: allowed values' out of a FastAPI validation error body."""
    m = re.search(r'"loc":\["query","([^"]+)".*?"msg":"([^"]+)"', message)
    return f"SAbDab2 says {m.group(1)}: {m.group(2)}" if m else None


# ---- projection ----------------------------------------------------------------------------


def antigen_index(instances: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for a in instances or []:
        inst = a.get("polymer_instance") or a.get("non_polymer_instance") or {}
        index[str(a.get("id"))] = {
            "name": inst.get("name"),
            "chain": inst.get("sabdab_auth_asym_id"),
            "type": inst.get("entity_type"),
            "organism": inst.get("organism_scientific"),
            "polymer": a.get("polymer_instance") is not None,
        }
    return index


def bound_antigens(
    bindings: list[dict[str, Any]] | None, index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Antigens an instance binds. Ions and water are dropped; small molecules are shown only
    when no polymer antigen is bound (they are then usually the hapten, otherwise additives)."""
    found: list[dict[str, Any]] = []
    for b in bindings or []:
        aid = str(b.get("antigen_instance_id"))
        a = index.get(aid)
        if a is None:
            found.append({"name": None, "chain": None, "type": None, "polymer": False, "id": aid})
        elif a["type"] not in _NEVER_ANTIGEN:
            found.append(a)
    chosen = [a for a in found if a["polymer"]] or found
    out = []
    for a in chosen:
        row = {"name": a["name"], "chain": a["chain"], "type": a["type"]}
        if a.get("organism"):
            row["organism"] = a["organism"]
        if "id" in a:
            row["instance_id"] = a["id"]
        out.append(row)
    return out


def segment(seg: dict[str, Any] | None, *, sequences: bool = False) -> dict[str, Any] | None:
    """One variable domain: chain, chain type, IMGT CDRs, germline calls."""
    if not seg:
        return None
    out = {
        "chain": seg.get("sabdab_auth_asym_id"),
        "chain_type": seg.get("type"),
        "cdr1": seg.get("cdr1"),
        "cdr2": seg.get("cdr2"),
        "cdr3": seg.get("cdr3"),
        "v_gene": seg.get("v_gene"),
        "j_gene": seg.get("j_gene"),
    }
    if sequences:
        out["sequence"] = seg.get("sequence")
    return {k: v for k, v in out.items() if v is not None}


def cdr_mismatches(seg: dict[str, Any] | None) -> list[str]:
    """CDRs whose string differs from the IMGT range of the numbering SAbDab2 returned."""
    if not seg:
        return []
    numbering = seg.get("numbering")
    if not isinstance(numbering, list) or not numbering:
        return []
    bad = []
    for name, (lo, hi) in _IMGT_CDRS.items():
        derived = "".join(
            str(item[1])
            for item in numbering
            if isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], list)
            and isinstance(item[0][0], int)
            and lo <= item[0][0] <= hi
            and item[1] != "-"
        )
        if seg.get(name) is not None and derived != seg[name]:
            bad.append(f"{seg.get('sabdab_auth_asym_id')}:{name}")
    return bad


def project_row(inst: dict[str, Any], index: dict[str, dict[str, Any]], *, detail: bool) -> dict:
    """One search row. ``detail`` adds the CDR3s (IMGT) and the canonical antibody id."""
    pe = inst.get("pdb_entry") or {}
    heavy = inst.get("heavy_like_segment") or {}
    light = inst.get("light_like_segment") or {}
    row: dict[str, Any] = {
        "pdb_id": plain_pdb_id(pe.get("id") or str(inst.get("id", "")).split("-")[0]),
        "antibody_type": inst.get("type"),
        "heavy_chain": heavy.get("sabdab_auth_asym_id"),
        "light_chain": light.get("sabdab_auth_asym_id"),
        "antigens": bound_antigens(inst.get("antigen_bindings"), index),
        "resolution": pe.get("resolution"),
        "method": pe.get("structure_method"),
        "deposition_date": pe.get("deposition_date"),
        "instance_id": inst.get("id"),
    }
    construct = (inst.get("antibody_construct") or {}).get("construct_type")
    if construct:
        row["construct"] = construct
    if detail:
        row["antibody_id"] = inst.get("antibody_id")
        row["cdr3"] = {k: v for k, v in (("H", heavy.get("cdr3")), ("L", light.get("cdr3"))) if v}
        row["cdr_scheme"] = CDR_SCHEME
    return row


def project_annotation(detail: dict[str, Any], *, sequences: bool = False) -> dict[str, Any]:
    """The SAbDab2 view of one PDB entry: every antibody instance with chains, CDRs, antigens."""
    index = antigen_index(detail.get("antigen_instances"))
    to_pdb_chain = {
        p.get("sabdab_auth_asym_id"): p.get("pdb_auth_asym_id")
        for p in detail.get("polymer_instances") or []
    }
    antibodies = []
    for inst in detail.get("antibody_instances") or []:
        domains = {}
        for key, field in (("heavy", "heavy_like_segment"), ("light", "light_like_segment")):
            seg = segment(inst.get(field), sequences=sequences)
            if seg is None:
                continue
            pdb_chain = to_pdb_chain.get(seg.get("chain"))
            if pdb_chain and pdb_chain != seg.get("chain"):
                seg["pdb_chain"] = pdb_chain
            domains[key] = seg
        ab: dict[str, Any] = {
            "instance_id": inst.get("id"),
            "antibody_id": inst.get("antibody_id"),
            "antibody_type": inst.get("type"),
            **domains,
            "antigens": bound_antigens(inst.get("antigen_bindings"), index),
        }
        construct = (inst.get("antibody_construct") or {}).get("construct_type")
        if construct:
            ab["construct"] = construct
        antibodies.append(ab)
    return {
        "in_sabdab": True,
        "status": detail.get("status"),
        "cdr_scheme": CDR_SCHEME,
        "antibodies": antibodies,
    }


# ---- connector -----------------------------------------------------------------------------


@register
class SabdabConnector(Connector):
    info: ClassVar[ConnectorInfo] = ConnectorInfo(
        name="sabdab",
        display_name="SAbDab2 (Structural Antibody Database)",
        kind="structure",
        base_url=BASE_URL,
        homepage="https://sabdab.opig.stats.ox.ac.uk",
        license="CC BY 4.0",
        citation=(
            "Capel H.L., Vavourakis O., Williams B.H., Taylor C.R., Deane C.M. (2026) SAbDab2: "
            "the structural antibody database in the age of machine learning. Nucleic Acids Res. "
            "gkag1091. doi:10.1093/nar/gkag1091"
        ),
        rps=1.0,  # an academic server; be gentle
        cache_ttl=12 * 3600,  # SAbDab2 updates weekly
        notes=f"API version {API_VERSION}; extended PDB ids (pdb_0000xxxx).",
        extra={"api_version": API_VERSION},
    )

    async def _get(
        self, path: str, params: dict[str, Any] | None = None, *, cache: bool = True
    ) -> tuple[Any, HttpResponse]:
        try:
            data, resp = await self.http.get_json(
                self.info.name,
                f"{BASE_URL}{path}",
                params=params,
                rps=self.info.rps,
                ttl=self.info.cache_ttl if cache else None,
            )
        except ImmunoError as exc:
            if exc.code == ErrorCode.INVALID_INPUT:  # 422: keep SAbDab2's list of allowed values
                raise ImmunoError(
                    ErrorCode.INVALID_INPUT, exc.message, hint=_hint_from_422(exc.message)
                ) from exc
            raise
        return clean_tree(data), resp

    async def stats(self, *, cache: bool = True) -> dict[str, Any]:
        data, _ = await self._get("/stats/summary", cache=cache)
        if not isinstance(data, dict) or not isinstance(data.get("antibody_instances"), int):
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "sabdab: stats summary has no antibody_instances"
            )
        return data

    async def search(
        self,
        *,
        binds_to_antigen: str | None = None,
        antibody_types: list[str] | None = None,
        construct_type: str | None = None,
        pdb_entry: str | None = None,
        cdr_sequence: str | None = None,
        max_resolution: float | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[dict[str, Any], HttpResponse]:
        """One page of antibody instances plus the antigen instances they bind."""
        params: dict[str, Any] = {"limit": limit, "offset": offset, "order_by": "id"}
        if binds_to_antigen:
            params["binds_to_antigen"] = binds_to_antigen[:NAME_FILTER_MAX]
        if antibody_types:
            params["antibody_type"] = list(antibody_types)
        if construct_type:
            params["construct_type"] = construct_type
        if pdb_entry:
            params["pdb_entry"] = sabdab_pdb_id(pdb_entry)
        if cdr_sequence:
            params["cdr_sequence"] = cdr_sequence
        if max_resolution is not None:
            params["resolution"] = max_resolution
        page, resp = await self._get("/frontend/antibody-instance-search", params)
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("results"), list)
            or not isinstance(page.get("total"), int)
            or not isinstance(page.get("antigen_instances"), list)
        ):
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE,
                "sabdab: search page lacks results, total or antigen_instances",
                hint="The SAbDab2 API may have changed. Report it so the connector is updated.",
            )
        return page, resp

    async def pdb_detail(self, pdb_id: str) -> tuple[dict[str, Any], HttpResponse]:
        """Everything SAbDab2 holds for one PDB entry. NOT_FOUND if the entry is not in SAbDab2."""
        sid = sabdab_pdb_id(pdb_id)
        try:
            data, resp = await self._get(f"/frontend/pdb/{sid}")
        except ImmunoError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                raise ImmunoError(
                    ErrorCode.NOT_FOUND, f"sabdab: {plain_pdb_id(sid)} is not in SAbDab2"
                ) from exc
            raise
        if not isinstance(data, dict) or not isinstance(data.get("antibody_instances"), list):
            raise ImmunoError(
                ErrorCode.NON_DATA_RESPONSE, "sabdab: PDB detail has no antibody_instances"
            )
        return data, resp

    async def health(self) -> dict[str, Any]:
        s = await self.stats(cache=False)
        return {
            "ok": True,
            "detail": (
                f"{s['antibody_instances']} antibody instances in {s.get('PDB_entries')} PDB "
                f"entries; last update {s.get('last_update')}"
            ),
        }
