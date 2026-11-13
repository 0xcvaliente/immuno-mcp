"""Confidence scores of predicted structures, and the ipSAE interface score.

Pure functions on file TEXT (and bytes for numpy ``.npz``); no network, no MCP imports.

Formats read by ``parse_confidence`` (each checked against its primary documentation, never
guessed; anything else raises INVALID_INPUT naming what was expected):

* AlphaFold 3 ``*_confidences.json`` / server ``*_full_data_*.json``: ``pae``, ``atom_plddts``,
  ``token_chain_ids``, ``token_res_ids`` (docs/output.md of google-deepmind/alphafold3).
* AlphaFold 3 ``*_summary_confidences.json``: ``ptm``, ``iptm``, ``chain_pair_iptm``,
  ``chain_ptm`` and optional ``chain_ids``.
* AlphaFold 2 / ColabFold ``*_scores_*.json``: ``plddt`` (list), ``pae`` (matrix), ``ptm``,
  optional ``iptm``; AlphaFold DB PAE JSON (``predicted_aligned_error``).
* Boltz-2 ``confidence_*.json``: ``confidence_score``, ``ptm``, ``iptm``, ``complex_plddt``,
  ``complex_iplddt``, ``chains_ptm``, ``pair_chains_iptm`` (pLDDT on a 0-1 scale; docs/prediction.md
  of jwohlwend/boltz).
* JapanFold (https://api.japanfold.aiand.com, read 2026-11-13 from /v1/openapi.json and
  /skill.md): result ``rows`` mirror Boltz-2 (``confidence_score``, ``complex_plddt``, ``iptm``,
  ``chains_ptm``, ``pair_chains_iptm`` keyed by chain position), AF2-IG rows carry ``plddt``,
  ``ptm``, ``iptm``, ``pae``, ``ipae``, ``interface_pae``; a results manifest wraps them in
  ``rows``. The PAE matrix is ``<name>_pae.npz`` (key ``pae``; read with ``load_pae_npz``); its
  sidecar ``<name>_pae.json`` only describes the arrays and is rejected with a pointer to the npz.
* Mean pLDDT from the B-factor column of a predicted structure (AlphaFold 2/3 and Boltz write
  pLDDT x 100 there). Experimental structures are refused.

ipSAE: see ``compute_ipsae``.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..core.errors import ErrorCode, ImmunoError
from .structure import LoadedStructure, load_structure

MAX_TOKENS = 6000
NOT_AFFINITY = (
    "ipSAE scores confidence in the predicted interface from predicted aligned error. "
    "It is not binding affinity and not a KD."
)
IPSAE_REFERENCE = (
    "Dunbrack RL Jr. (2025) Rēs ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and "
    "how to fix it. bioRxiv 2025.02.10.637595, version 2 (15 Dec 2025). "
    "doi:10.1101/2025.02.10.637595. Equations 14-16. Reference code: "
    "https://github.com/DunbrackLab/IPSAE ipsae.py version 4 (3 Jan 2026, commit 6174cf9), MIT licence."
)
IPSAE_VARIANT = (
    "ipSAE (d0res): for each aligned residue i of one group, mean over residues j of the other "
    "group with PAE_ij < cutoff of 1/(1+(PAE_ij/d0)^2), d0 from the number of such j; "
    "maximised over i. Chain-asymmetric (A->B aligns on A and scores B; rows of the PAE matrix "
    "are aligned residues). ipsae = max(A->B, B->A) as in the paper's Eq. 16."
)
D0_RULE = (
    "d0 = max(d0_min, 1.24*(max(L,26)-15)^(1/3) - 1.8), L = residues of the scored group with "
    "PAE < cutoff for aligned residue i (Eq. 15: d0 = 1 for L < 27); d0_min = 1.0, or 2.0 when "
    "either group contains nucleic acid (rule from the reference code, not the paper)."
)
GROUP_RULE = (
    "Each group is pooled as one chain: with one chain per group this is exactly the reference "
    "pairwise chain score; with several chains per group (e.g. heavy+light) it is the same "
    "equations applied to the union, an extension not in the paper."
)

# Residue names the reference implementation (and AlphaFold 3 tokenisation) treat as one token.
_ONE_TOKEN_RESIDUES = frozenset(
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "DA",
        "DC",
        "DT",
        "DG",
        "A",
        "C",
        "U",
        "G",
    ]
)


# ---- data types ------------------------------------------------------------------------------


@dataclass
class ConfidenceSummary:
    """Confidence numbers from one prediction. Fields after ``source_format`` have defaults.

    ``plddt_mean`` is on a 0-100 scale. ``pae_matrix`` rows are aligned tokens, columns scored
    tokens. ``token_chain_ids`` has one chain id per PAE row. ``residue_token_mask`` (when known)
    marks the one token that represents each polymer residue; ligand atoms and the extra atom
    tokens of modified residues are False.
    """

    plddt_mean: float | None
    ptm: float | None
    iptm: float | None
    pae_mean: float | None
    pae_matrix: list[list[float]] | None
    token_chain_ids: list[str] | None
    source_format: str
    chain_ptm: dict[str, float] | None = None
    chain_pair_iptm: dict[str, dict[str, float]] | None = None
    token_res_ids: list[int] | None = None
    residue_token_mask: list[bool] | None = None
    nucleic_chains: list[str] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def score_rows(self) -> list[dict[str, Any]]:
        """``{"metric", "value", "params"}`` rows named as in core/metrics.py."""
        rows = []
        for metric in ("plddt_mean", "ptm", "iptm", "pae_mean"):
            value = getattr(self, metric)
            if value is not None:
                params: dict[str, Any] = {"source_format": self.source_format}
                if metric == "plddt_mean" and "plddt_basis" in self.extras:
                    params["basis"] = self.extras["plddt_basis"]
                rows.append({"metric": metric, "value": value, "params": params})
        return rows


@dataclass
class TokenLayout:
    """Token axis of a PAE matrix rebuilt from the structure, as the reference code does."""

    chain_ids: list[str]
    representative: list[bool]
    nucleic_chains: list[str]

    @property
    def n_tokens(self) -> int:
        return len(self.chain_ids)


# ---- helpers ---------------------------------------------------------------------------------


def _bad(msg: str, hint: str | None = None) -> ImmunoError:
    return ImmunoError(ErrorCode.INVALID_INPUT, msg, hint=hint)


EXPECTED_JSON = (
    "AlphaFold 3 confidences/full_data JSON (pae, atom_plddts, token_chain_ids), AlphaFold 3 "
    "summary_confidences JSON (ptm, iptm, chain_pair_iptm), AlphaFold 2/ColabFold scores JSON "
    "(plddt list, pae, ptm), AlphaFold DB PAE JSON (predicted_aligned_error), Boltz-2 "
    "confidence JSON (confidence_score, complex_plddt, pair_chains_iptm), or a JapanFold "
    "results row/manifest"
)


def _load_json(text: str, what: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise _bad(f"{what} is empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad(f"{what} is not valid JSON ({exc.msg} at line {exc.lineno})") from exc


def _number(value: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad(f"'{name}' should be a number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v) or v < lo or v > hi:
        raise _bad(f"'{name}' = {value!r} is outside {lo}-{hi}")
    return v


def _opt_number(obj: dict[str, Any], key: str, lo: float, hi: float) -> float | None:
    value = obj.get(key)
    return None if value is None else _number(value, key, lo, hi)


def as_pae_array(pae: Any, *, name: str = "pae") -> np.ndarray:
    """A square, finite, non-negative float64 matrix, or INVALID_INPUT."""
    try:
        arr = np.asarray(pae, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _bad(f"'{name}' is not a numeric matrix") from exc
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise _bad(f"'{name}' must be a square N x N matrix, got shape {arr.shape}")
    if arr.shape[0] > MAX_TOKENS:
        raise ImmunoError(
            ErrorCode.TOO_LARGE, f"'{name}' has {arr.shape[0]} tokens (limit {MAX_TOKENS})"
        )
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise _bad(f"'{name}' has negative or non-finite values")
    return arr


def load_pae_npz(data: bytes) -> np.ndarray:
    """PAE matrix from a numpy ``.npz`` (JapanFold ``<name>_pae.npz``, Boltz ``pae_*.npz``)."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            keys = list(npz.files)
            if "pae" not in keys:
                raise _bad(
                    f"the .npz has arrays {keys} but no 'pae'",
                    hint="Pass the PAE file (<name>_pae.npz or pae_<name>_model_0.npz).",
                )
            return as_pae_array(npz["pae"])
    except ImmunoError:
        raise
    except (ValueError, OSError, zipfile.BadZipFile, EOFError) as exc:
        raise _bad(f"not a readable numpy .npz file ({type(exc).__name__})") from exc


def _plddt_list(values: Any, name: str) -> tuple[float, str | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise _bad(f"'{name}' must be a non-empty list of numbers")
    if arr.min() < 0 or arr.max() > 100:
        raise _bad(f"'{name}' has values outside 0-100")
    return float(arr.mean()), None


def _matrix_dict(value: Any, name: str, chain_ids: list[str] | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if isinstance(value, dict):
        for k1, row in value.items():
            if not isinstance(row, dict):
                raise _bad(f"'{name}' must map chain to chain to number")
            out[str(k1)] = {str(k2): _number(v, name, 0, 1) for k2, v in row.items()}
        return out
    if isinstance(value, list):
        keys = chain_ids if chain_ids and len(chain_ids) == len(value) else None
        for i, row in enumerate(value):
            if not isinstance(row, list):
                raise _bad(f"'{name}' must be a square list of lists")
            k1 = keys[i] if keys else str(i)
            out[k1] = {}
            for j, v in enumerate(row):
                if v is None:
                    continue
                out[k1][keys[j] if keys else str(j)] = _number(v, name, 0, 1)
        return out
    raise _bad(f"'{name}' must be an object or a list of lists")


def _vector_dict(value: Any, name: str, chain_ids: list[str] | None) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): _number(v, name, 0, 1) for k, v in value.items()}
    if isinstance(value, list):
        keys = chain_ids if chain_ids and len(chain_ids) == len(value) else None
        return {
            (keys[i] if keys else str(i)): _number(v, name, 0, 1)
            for i, v in enumerate(value)
            if v is not None
        }
    raise _bad(f"'{name}' must be an object or a list")


# ---- format recognition ----------------------------------------------------------------------


def classify_json(obj: Any) -> str:
    """Name of the confidence format, or 'unknown'. Decided by documented keys only."""
    if isinstance(obj, list):
        if len(obj) == 1 and isinstance(obj[0], dict) and "predicted_aligned_error" in obj[0]:
            return "alphafold_db_pae"
        return "unknown"
    if not isinstance(obj, dict):
        return "unknown"
    keys = set(obj)
    if isinstance(obj.get("rows"), list):
        return "japanfold_results"
    if "affinity_pred_value" in keys or "affinity_probability_binary" in keys:
        return "boltz_affinity"
    if "arrays" in keys and "pae" not in keys and ("absent" in keys or "file" in keys):
        return "japanfold_pae_sidecar"
    if "pae" in keys and "token_chain_ids" in keys:
        return "alphafold3_confidences"
    if "chain_pair_iptm" in keys and ("ptm" in keys or "iptm" in keys):
        return "alphafold3_summary_confidences"
    if "confidence_score" in keys and keys & {"complex_plddt", "pair_chains_iptm", "ptm"}:
        return "boltz2_confidence"
    if "predicted_aligned_error" in keys:
        return "alphafold_db_pae"
    if "pae" in keys and isinstance(obj.get("plddt"), list):
        return "colabfold_scores"
    if (
        keys & {"interface_pae", "ipae"}
        and "iptm" in keys
        and not isinstance(obj.get("plddt"), list)
    ):
        return "japanfold_af2ig_row"
    return "unknown"


@dataclass
class _Parsed:
    fmt: str
    plddt_mean: float | None = None
    plddt_basis: str | None = None
    ptm: float | None = None
    iptm: float | None = None
    pae: np.ndarray | None = None
    pae_scalar: float | None = None
    token_chain_ids: list[str] | None = None
    token_res_ids: list[int] | None = None
    chain_ptm: dict[str, float] | None = None
    chain_pair_iptm: dict[str, dict[str, float]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _parse_obj(obj: Any, what: str) -> _Parsed:
    kind = classify_json(obj)
    if kind == "japanfold_results":
        rows = obj["rows"]
        if len(rows) != 1 or not isinstance(rows[0], dict):
            names = [r.get("name") for r in rows if isinstance(r, dict) and r.get("name")]
            raise _bad(
                f"{what} is a JapanFold results manifest with {len(rows)} rows",
                hint="Pass one row (one target) as the JSON object."
                + (f" Targets: {', '.join(map(str, names[:20]))}." if names else ""),
            )
        inner = _parse_obj(rows[0], what)
        inner.fmt = f"japanfold_results:{inner.fmt}"
        return inner
    if kind == "japanfold_pae_sidecar":
        raise _bad(
            f"{what} is a JapanFold PAE sidecar (<name>_pae.json): it lists the arrays but holds "
            "no matrix",
            hint="Use the matching <name>_pae.npz artifact for PAE.",
        )
    if kind == "boltz_affinity":
        raise _bad(
            f"{what} is a Boltz affinity file, not a confidence file",
            hint="Pass confidence_<name>_model_0.json for pTM/ipTM/pLDDT.",
        )
    if kind == "unknown":
        found = sorted(obj)[:12] if isinstance(obj, dict) else type(obj).__name__
        raise _bad(
            f"{what} is not a recognised confidence format (found keys: {found})",
            hint=f"Expected one of: {EXPECTED_JSON}.",
        )

    p = _Parsed(fmt=kind)
    if kind == "alphafold3_confidences":
        p.pae = as_pae_array(obj["pae"])
        ids = obj["token_chain_ids"]
        if not isinstance(ids, list) or len(ids) != p.pae.shape[0]:
            raise _bad("'token_chain_ids' must list one chain id per PAE row")
        p.token_chain_ids = [str(c) for c in ids]
        res = obj.get("token_res_ids")
        if isinstance(res, list) and len(res) == len(ids):
            p.token_res_ids = [int(r) for r in res]
        if obj.get("atom_plddts") is not None:
            p.plddt_mean, _ = _plddt_list(obj["atom_plddts"], "atom_plddts")
            p.plddt_basis = "mean over atoms (atom_plddts)"
    elif kind == "alphafold3_summary_confidences":
        chain_ids = obj.get("chain_ids") if isinstance(obj.get("chain_ids"), list) else None
        chain_ids = [str(c) for c in chain_ids] if chain_ids else None
        p.ptm = _opt_number(obj, "ptm", 0, 1)
        p.iptm = _opt_number(obj, "iptm", 0, 1)
        p.chain_pair_iptm = _matrix_dict(obj["chain_pair_iptm"], "chain_pair_iptm", chain_ids)
        if obj.get("chain_ptm") is not None:
            p.chain_ptm = _vector_dict(obj["chain_ptm"], "chain_ptm", chain_ids)
        for key in ("ranking_score", "fraction_disordered"):
            if isinstance(obj.get(key), (int, float)) and not isinstance(obj.get(key), bool):
                p.extras[key] = float(obj[key])
        if isinstance(obj.get("has_clash"), (bool, int, float)):
            p.extras["has_clash"] = bool(obj["has_clash"])
        if not chain_ids:
            p.extras["chain_keys"] = "chain position (0-based); file has no chain_ids"
    elif kind == "boltz2_confidence":
        p.ptm = _opt_number(obj, "ptm", 0, 1)
        p.iptm = _opt_number(obj, "iptm", 0, 1)
        cp = _opt_number(obj, "complex_plddt", 0, 1)
        if cp is not None:
            p.plddt_mean = 100.0 * cp
            p.plddt_basis = "Boltz complex_plddt (0-1) x 100"
        for key in (
            "confidence_score",
            "complex_iplddt",
            "complex_pde",
            "complex_ipde",
            "ligand_iptm",
            "protein_iptm",
        ):
            value = obj.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                p.extras[key] = float(value)
        if obj.get("chains_ptm") is not None:
            p.chain_ptm = _vector_dict(obj["chains_ptm"], "chains_ptm", None)
        if obj.get("pair_chains_iptm") is not None:
            p.chain_pair_iptm = _matrix_dict(obj["pair_chains_iptm"], "pair_chains_iptm", None)
        p.extras["chain_keys"] = "as in the file (Boltz: chain position as a string)"
    elif kind == "colabfold_scores":
        p.pae = as_pae_array(obj["pae"])
        p.plddt_mean, _ = _plddt_list(obj["plddt"], "plddt")
        p.plddt_basis = "mean over residues (plddt list)"
        p.ptm = _opt_number(obj, "ptm", 0, 1)
        p.iptm = _opt_number(obj, "iptm", 0, 1)
        if isinstance(obj.get("per_chain_ptm"), dict):
            p.chain_ptm = _vector_dict(obj["per_chain_ptm"], "per_chain_ptm", None)
        pairwise = obj.get("pairwise_iptm")
        if isinstance(pairwise, dict):  # ColabFold: {"A-B": 0.48, ...}
            matrix: dict[str, dict[str, float]] = {}
            for key, value in pairwise.items():
                c1, sep, c2 = str(key).partition("-")
                if not sep:
                    raise _bad(f"'pairwise_iptm' key {key!r} is not 'X-Y'")
                v = _number(value, "pairwise_iptm", 0, 1)
                matrix.setdefault(c1, {})[c2] = v
                matrix.setdefault(c2, {})[c1] = v
            p.chain_pair_iptm = matrix
        for key in ("actifptm", "max_pae"):
            value = obj.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                p.extras[key] = float(value)
    elif kind == "alphafold_db_pae":
        body = obj[0] if isinstance(obj, list) else obj
        p.pae = as_pae_array(body["predicted_aligned_error"], name="predicted_aligned_error")
    elif kind == "japanfold_af2ig_row":
        p.ptm = _opt_number(obj, "ptm", 0, 1)
        p.iptm = _opt_number(obj, "iptm", 0, 1)
        plddt = obj.get("plddt")
        if plddt is not None:
            v = _number(plddt, "plddt", 0, 100)
            if v <= 1.0:
                v *= 100.0
                p.warnings.append(
                    "plddt <= 1 was read as a 0-1 scale and multiplied by 100; check the source."
                )
            p.plddt_mean = v
            p.plddt_basis = "JapanFold AF2-IG row plddt"
        pae = obj.get("pae")
        if isinstance(pae, list):
            p.pae = as_pae_array(pae)
        elif pae is not None:
            p.pae_scalar = _number(pae, "pae", 0, 100)
        for key in ("ipae", "interface_pae"):
            value = obj.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                p.extras[key] = float(value)
    return p


# ---- structure-derived values ----------------------------------------------------------------


def declared_experimental_method(text: str, fmt: str) -> str | None:
    """The experimental method a file declares (_exptl.method / EXPDTA), or None."""
    if fmt == "cif":
        for line in io.StringIO(text):
            if line.startswith("_exptl.method"):
                value = line[len("_exptl.method") :].strip().strip("'\"")
                return value or "declared in a loop"
        return None
    for line in io.StringIO(text):
        if line.startswith("EXPDTA"):
            return line[6:].strip() or "EXPDTA"
    return None


def plddt_from_bfactors(ls: LoadedStructure) -> tuple[float, list[str]]:
    """Mean over polymer residues of the per-residue mean B-factor, on a 0-100 scale."""
    import biotite.structure as struc

    atoms = ls.atoms[ls.polymer]
    if atoms.array_length() == 0:
        raise _bad("the structure has no polymer residues to read pLDDT from")
    b = np.asarray(atoms.b_factor, dtype=np.float64)
    if not np.isfinite(b).all() or b.min() < 0 or b.max() > 100:
        raise _bad(
            "B-factors are outside 0-100, so they are not pLDDT values",
            hint="Pass the confidence JSON of the prediction instead.",
        )
    starts: np.ndarray = struc.get_residue_starts(atoms)
    per_res = np.add.reduceat(b, starts) / np.diff(np.append(starts, len(b)))
    warnings: list[str] = []
    mean = float(per_res.mean())
    if b.max() <= 1.0:
        mean *= 100.0
        warnings.append(
            "All B-factors are <= 1: read as pLDDT on a 0-1 scale and multiplied by 100."
        )
    return mean, warnings


def token_layout_from_loaded(ls: LoadedStructure) -> TokenLayout:
    """Rebuild the token axis of an AlphaFold 3 / Boltz PAE matrix from the structure.

    Same rules as the reference ipsae.py (v4): a standard residue (20 amino acids, DA DC DT DG
    A C U G) is one token, represented by CA (C1' for nucleotides); every heavy atom of a
    modified residue and of a ligand or ion is its own token. The representative tokens (one per
    polymer residue) are the ones ipSAE uses. AlphaFold 2 matrices have one token per residue,
    which these rules also reproduce for standard sequences.
    """
    import biotite.structure as struc

    atoms = ls.atoms
    keep = ~np.asarray(struc.filter_solvent(atoms), dtype=bool) & ~np.isin(
        atoms.element, ["H", "D"]
    )
    atoms = atoms[keep]
    polymer = ls.polymer[keep]
    is_nuc = np.asarray(struc.filter_nucleotides(atoms), dtype=bool)
    chain_ids: list[str] = []
    rep: list[bool] = []
    nucleic: dict[str, None] = {}
    starts: np.ndarray = struc.get_residue_starts(atoms)
    stops = np.append(starts[1:], atoms.array_length())
    for s, e in zip(starts, stops, strict=True):
        chain = str(atoms.chain_id[s])
        name = str(atoms.res_name[s])
        if not polymer[s]:
            chain_ids.extend([chain] * (e - s))
            rep.extend([False] * (e - s))
            continue
        if is_nuc[s]:
            nucleic.setdefault(chain, None)
        names = [str(n) for n in atoms.atom_name[s:e]]
        rep_name = "C1'" if is_nuc[s] else "CA"
        if rep_name not in names:
            rep_name = "CA" if "CA" in names else ("C1'" if "C1'" in names else "")
        if not rep_name:
            raise _bad(
                f"residue {chain}:{name}{atoms.res_id[s]} has no CA or C1' atom, so its token "
                "cannot be placed"
            )
        if name in _ONE_TOKEN_RESIDUES:
            chain_ids.append(chain)
            rep.append(True)
        else:
            chain_ids.extend([chain] * (e - s))
            rep.extend(n == rep_name for n in names)
    return TokenLayout(chain_ids=chain_ids, representative=rep, nucleic_chains=list(nucleic))


def token_layout(text: str, fmt: Literal["cif", "pdb"]) -> TokenLayout:
    return token_layout_from_loaded(load_structure(text, fmt))


def residue_level(
    pae: Any, token_chain_ids: Sequence[str], mask: Sequence[bool] | None
) -> tuple[np.ndarray, list[str]]:
    """Keep only the representative token of each residue (rows and columns)."""
    arr = as_pae_array(pae)
    ids = [str(c) for c in token_chain_ids]
    if len(ids) != arr.shape[0]:
        raise _bad(f"{len(ids)} token chain ids for a {arr.shape[0]} x {arr.shape[0]} PAE matrix")
    if mask is None:
        return arr, ids
    m = np.asarray(mask, dtype=bool)
    if m.shape != (arr.shape[0],):
        raise _bad("the residue token mask does not match the PAE matrix")
    return arr[np.ix_(m, m)], [c for c, k in zip(ids, m, strict=True) if k]


def multi_token_chains(token_chain_ids: Sequence[str], token_res_ids: Sequence[int]) -> list[str]:
    """Chains in which one residue number spans several consecutive tokens (modified residues,
    ligands). Their per-atom tokens cannot be collapsed without the structure."""
    out: dict[str, None] = {}
    for k in range(1, len(token_chain_ids)):
        if (
            token_chain_ids[k] == token_chain_ids[k - 1]
            and token_res_ids[k] == token_res_ids[k - 1]
        ):
            out.setdefault(str(token_chain_ids[k]), None)
    return list(out)


@dataclass
class IpsaeInput:
    """A residue-level PAE matrix with one chain id per row, ready for ``compute_ipsae``."""

    pae: np.ndarray
    chain_ids: list[str]
    nucleic_chains: list[str]
    token_source: str
    pae_tokens: int
    warnings: list[str]


def prepare_ipsae_input(
    pae: Any,
    *,
    groups: Sequence[str],
    file_chain_ids: Sequence[str] | None = None,
    file_res_ids: Sequence[int] | None = None,
    layout: TokenLayout | None = None,
    explicit_chain_ids: Sequence[str] | None = None,
) -> IpsaeInput:
    """Decide the token axis of a PAE matrix and collapse it to one token per residue.

    Precedence: explicit chain ids, then the paired structure (exact reference behaviour), then
    the chain ids stored in the PAE file. Refuses to guess when a selected chain has residues
    spread over several tokens and no structure says which token represents the residue.
    """
    arr = as_pae_array(pae)
    n = arr.shape[0]
    warnings: list[str] = []
    wanted = set(groups)
    if layout is not None and layout.n_tokens != n:
        raise _bad(
            f"the structure gives {layout.n_tokens} tokens but the PAE matrix has {n}",
            hint="Pass the structure and PAE file of the same prediction (same model/sample).",
        )
    if explicit_chain_ids is not None:
        ids = [str(c) for c in explicit_chain_ids]
        if len(ids) != n:
            raise _bad(
                f"token_chain_ids has {len(ids)} entries but the PAE matrix has {n} rows",
                hint="Give one chain id per PAE row, in matrix order.",
            )
        if layout is not None and layout.chain_ids == ids:
            res, rids = residue_level(arr, ids, layout.representative)
            return IpsaeInput(res, rids, layout.nucleic_chains, "token_chain_ids+structure", n, [])
        if file_res_ids is not None and len(file_res_ids) == n:
            multi = set(multi_token_chains(ids, file_res_ids)) & wanted
            if multi:
                raise _bad(
                    f"chain(s) {', '.join(sorted(multi))} have residues spread over several "
                    "tokens (modified residues or ligands)",
                    hint="Pass structure_artifact_id so the representative token is chosen.",
                )
        warnings.append("token_chain_ids given: each PAE row was taken as one residue.")
        return IpsaeInput(arr, ids, [], "token_chain_ids", n, warnings)
    if layout is not None:
        if file_chain_ids is not None and list(map(str, file_chain_ids)) != layout.chain_ids:
            raise _bad(
                "the structure's chains do not match the PAE file's token_chain_ids",
                hint="Pass the structure and PAE file of the same prediction.",
            )
        res, rids = residue_level(arr, layout.chain_ids, layout.representative)
        return IpsaeInput(res, rids, layout.nucleic_chains, "structure", n, warnings)
    if file_chain_ids is not None:
        ids = [str(c) for c in file_chain_ids]
        if len(ids) != n:
            raise _bad("the PAE file's token_chain_ids do not match its matrix")
        if file_res_ids is not None:
            multi = set(multi_token_chains(ids, file_res_ids)) & wanted
            if multi:
                raise _bad(
                    f"chain(s) {', '.join(sorted(multi))} contain modified residues or ligands "
                    "whose atoms are separate PAE tokens; without the structure the residue's "
                    "representative token (CA) cannot be chosen",
                    hint="Pass structure_artifact_id (the model file of the same prediction).",
                )
        warnings.append(
            "Chain ids from the PAE file; nucleic-acid chains cannot be detected without the "
            "structure, so the protein d0 floor (1.0) was used."
        )
        return IpsaeInput(arr, ids, [], "pae_file", n, warnings)
    raise _bad(
        "the PAE matrix has no chain labels (AlphaFold 2/ColabFold JSON, Boltz or JapanFold npz)",
        hint="Pass structure_artifact_id (the predicted model of the same run) or token_chain_ids.",
    )


# ---- parse_confidence ------------------------------------------------------------------------


def parse_confidence(
    *,
    structure: str | None,
    structure_format: Literal["cif", "pdb"] | None,
    summary_json: str | None,
    pae_json: str | None,
) -> ConfidenceSummary:
    """Read confidence values from any supported mix of JSON files and a predicted structure.

    JSON values win over the structure. The structure supplies mean pLDDT (B-factor column)
    when no JSON gives one, and the token layout of the PAE matrix (chain per token, and which
    token represents each residue) when it matches the matrix size.
    """
    if not any((structure, summary_json, pae_json)):
        raise _bad(
            "nothing to parse",
            hint="Pass a summary JSON, a PAE JSON and/or a predicted structure.",
        )
    parsed: list[_Parsed] = []
    if summary_json is not None:
        parsed.append(_parse_obj(_load_json(summary_json, "summary_json"), "summary_json"))
    if pae_json is not None:
        parsed.append(_parse_obj(_load_json(pae_json, "pae_json"), "pae_json"))

    def first(attr: str) -> Any:
        for p in parsed:
            value = getattr(p, attr)
            if value is not None:
                return value
        return None

    pae: np.ndarray | None = None
    for p in reversed(parsed):  # the PAE file wins for the matrix
        if p.pae is not None:
            pae = p.pae
            break
    warnings = [w for p in parsed for w in p.warnings]
    extras: dict[str, Any] = {}
    for p in parsed:
        for k, v in p.extras.items():
            extras.setdefault(k, v)
    plddt_mean = first("plddt_mean")
    if plddt_mean is not None:
        extras["plddt_basis"] = first("plddt_basis")
    token_chain_ids: list[str] | None = first("token_chain_ids")
    token_res_ids: list[int] | None = first("token_res_ids")
    mask: list[bool] | None = None
    nucleic: list[str] | None = None
    formats = [p.fmt for p in parsed]

    if structure is not None:
        if structure_format is None:
            raise _bad("structure_format is required with structure", hint="Use 'cif' or 'pdb'.")
        ls = load_structure(structure, structure_format)
        formats.append(f"structure_{ls.fmt}")
        if plddt_mean is None:
            method = declared_experimental_method(structure, ls.fmt)
            if method:
                raise _bad(
                    f"the structure declares an experimental method ({method}); its B-factors "
                    "are atomic displacement, not pLDDT",
                    hint="pLDDT is only read from predicted models (AlphaFold, Boltz, ...).",
                )
            plddt_mean, bwarn = plddt_from_bfactors(ls)
            warnings += bwarn
            extras["plddt_basis"] = "B-factor column: mean over polymer residues of residue means"
        if pae is not None:
            layout = token_layout_from_loaded(ls)
            if layout.n_tokens != pae.shape[0]:
                warnings.append(
                    f"The structure gives {layout.n_tokens} tokens but the PAE matrix has "
                    f"{pae.shape[0]}; the token layout was not taken from the structure."
                )
            elif token_chain_ids is not None and token_chain_ids != layout.chain_ids:
                raise _bad(
                    "the structure's chains do not match the PAE file's token_chain_ids",
                    hint="Pass the structure and the PAE file of the same prediction.",
                )
            else:
                token_chain_ids = layout.chain_ids
                mask = layout.representative
                nucleic = layout.nucleic_chains
    if pae is not None and token_chain_ids is None:
        warnings.append(
            "The PAE matrix has no chain labels; pass the predicted structure (or token chain "
            "ids) to compute interface scores."
        )
    if pae is not None and mask is None and token_res_ids is not None and token_chain_ids:
        multi = multi_token_chains(token_chain_ids, token_res_ids)
        if multi:
            extras["multi_token_chains"] = multi

    pae_mean = float(pae.mean()) if pae is not None else first("pae_scalar")
    return ConfidenceSummary(
        plddt_mean=None if plddt_mean is None else round(float(plddt_mean), 3),
        ptm=first("ptm"),
        iptm=first("iptm"),
        pae_mean=None if pae_mean is None else round(float(pae_mean), 3),
        pae_matrix=pae.tolist() if pae is not None else None,
        token_chain_ids=token_chain_ids,
        source_format="+".join(formats),
        chain_ptm=first("chain_ptm"),
        chain_pair_iptm=first("chain_pair_iptm"),
        token_res_ids=token_res_ids,
        residue_token_mask=mask,
        nucleic_chains=nucleic,
        extras=extras,
        warnings=warnings,
    )


# ---- ipSAE -----------------------------------------------------------------------------------
#
# Ported from ipsae.py (version 4, 3 Jan 2026) by Roland Dunbrack, Fox Chase Cancer Center,
# https://github.com/DunbrackLab/IPSAE, MIT licence:
#   "MIT license: script can be modified and redistributed for non-commercial and commercial use,
#    as long as this information is reproduced."
#   Copyright (c) 2025 Lab of Dr. Roland Dunbrack. Permission is hereby granted, free of charge,
#   to any person obtaining a copy of this software ... THE SOFTWARE IS PROVIDED "AS IS",
#   WITHOUT WARRANTY OF ANY KIND.
# Only the ipSAE (d0res) part is ported: ptm_func, calc_d0_array and the per-residue
# ipsae_d0res_byres / ipsae_d0res_asym / ipsae_d0res_max logic. Checked against the paper's
# Eq. 14-16 (bioRxiv 2025.02.10.637595 v2) and against the reference output for the AF3
# AURKA/TPX2 example shipped in that repository (tests/fixtures/structure/ipsae_aurka_tpx2).


def d0_from_count(n: np.ndarray, *, nucleic: bool = False) -> np.ndarray:
    """Reference calc_d0_array: d0 = max(d0_min, 1.24*(max(L,26)-15)^(1/3) - 1.8)."""
    length = np.maximum(26.0, np.asarray(n, dtype=np.float64))
    return np.maximum(2.0 if nucleic else 1.0, 1.24 * np.cbrt(length - 15.0) - 1.8)


def ipsae_per_residue(
    sub: np.ndarray, cutoff: float, *, nucleic: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per aligned residue (rows of ``sub``, columns = scored residues): ipSAE, n0res, d0res.

    Reference ipsae_d0res_byres: a row with no PAE below the cutoff scores 0.
    """
    sub = np.asarray(sub, dtype=np.float64)
    valid = sub < cutoff
    n0 = valid.sum(axis=1)
    d0 = d0_from_count(n0, nucleic=nucleic)
    ptm = 1.0 / (1.0 + (sub / d0[:, None]) ** 2.0)
    sums = np.where(valid, ptm, 0.0).sum(axis=1)
    byres = np.divide(sums, n0, out=np.zeros_like(sums), where=n0 > 0)
    return byres, n0, d0


def _directional(sub: np.ndarray, cutoff: float, nucleic: bool) -> dict[str, Any]:
    byres, n0, d0 = ipsae_per_residue(sub, cutoff, nucleic=nucleic)
    valid = sub < cutoff
    k = int(np.argmax(byres))
    return {
        "value": float(byres[k]),
        "best_row": k,
        "n0res": int(n0[k]),
        "d0res": float(d0[k]),
        "aligned_residues_with_good_pae": int((n0 > 0).sum()),
        "scored_residues_with_good_pae": int(valid.any(axis=0).sum()),
    }


def compute_ipsae(
    pae: list[list[float]] | np.ndarray,
    token_chain_ids: list[str],
    group_a: list[str],
    group_b: list[str],
    *,
    pae_cutoff: float = 10.0,
    nucleic_acid_chains: Collection[str] = (),
) -> dict[str, Any]:
    """ipSAE between two chain groups from a residue-level PAE matrix.

    VERIFIED against the primary source: Dunbrack (2025) bioRxiv 2025.02.10.637595 v2,
    Eq. 14-16, and the MIT-licensed reference ipsae.py v4 (its AF3 example output is reproduced
    in the unit tests). ``pae`` must have ONE token per residue (row = aligned residue,
    column = scored residue); collapse AlphaFold 3 / Boltz atom tokens first with
    ``residue_level``. Returns ``ipsae`` (= max of the two directions, Eq. 16), the directional
    values, the variant, the cutoff and the reference. Not binding affinity.
    """
    arr = as_pae_array(pae)
    ids = [str(c) for c in token_chain_ids]
    if len(ids) != arr.shape[0]:
        raise _bad(
            f"{len(ids)} token chain ids for a {arr.shape[0]} x {arr.shape[0]} PAE matrix",
            hint="token_chain_ids needs one chain id per PAE row.",
        )
    try:
        cutoff = float(pae_cutoff)
    except (TypeError, ValueError) as exc:
        raise _bad(f"pae_cutoff must be a number, got {pae_cutoff!r}") from exc
    if not math.isfinite(cutoff) or cutoff <= 0 or cutoff > 100:
        raise _bad(
            f"pae_cutoff must be > 0 and <= 100 Å, got {pae_cutoff!r}",
            hint="The paper recommends 10 or 15 Å.",
        )
    available = list(dict.fromkeys(ids))
    groups = []
    for label, group in (("group_a", group_a), ("group_b", group_b)):
        if isinstance(group, str):
            group = [group]
        g = list(dict.fromkeys(str(c).strip() for c in (group or []) if str(c).strip()))
        if not g:
            raise _bad(f"{label} is empty", hint=f"Chains in the matrix: {', '.join(available)}.")
        missing = [c for c in g if c not in available]
        if missing:
            raise _bad(
                f"{label}: chain(s) {', '.join(missing)} are not in the PAE token chain ids",
                hint=f"Chains in the matrix: {', '.join(available)}.",
            )
        groups.append(g)
    overlap = sorted(set(groups[0]) & set(groups[1]))
    if overlap:
        raise _bad(f"chain(s) {', '.join(overlap)} are in both groups")

    id_arr = np.asarray(ids)
    in_a = np.isin(id_arr, groups[0])
    in_b = np.isin(id_arr, groups[1])
    nucleic = bool(set(nucleic_acid_chains) & (set(groups[0]) | set(groups[1])))
    idx_a, idx_b = np.flatnonzero(in_a), np.flatnonzero(in_b)
    ab = _directional(arr[np.ix_(in_a, in_b)], cutoff, nucleic)
    ba = _directional(arr[np.ix_(in_b, in_a)], cutoff, nucleic)
    ab["aligned_token"] = int(idx_a[ab.pop("best_row")])
    ba["aligned_token"] = int(idx_b[ba.pop("best_row")])
    for d in (ab, ba):
        d["aligned_chain"] = ids[d["aligned_token"]]
        d["value"] = round(d["value"], 6)
        d["d0res"] = round(d["d0res"], 4)
    best = max(ab["value"], ba["value"])
    return {
        "ipsae": best,
        "ipsae_a_to_b": ab["value"],
        "ipsae_b_to_a": ba["value"],
        "max_direction": "a_to_b" if ab["value"] >= ba["value"] else "b_to_a",
        "directional": {"a_to_b": ab, "b_to_a": ba},
        "group_a": groups[0],
        "group_b": groups[1],
        "n_residues_a": int(in_a.sum()),
        "n_residues_b": int(in_b.sum()),
        "pae_cutoff": cutoff,
        "nucleic_acid_pair": nucleic,
        "variant": IPSAE_VARIANT,
        "d0_rule": D0_RULE,
        "group_rule": GROUP_RULE,
        "reference": IPSAE_REFERENCE,
        "note": NOT_AFFINITY,
    }
