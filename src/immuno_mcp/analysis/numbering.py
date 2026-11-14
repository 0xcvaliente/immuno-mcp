"""analysis/numbering.py: antibody variable-domain numbering with ANARCII.

`number_chain(sequence, scheme)` numbers one VH, VHH or VL domain and returns the residue labels
in the requested scheme (imgt, kabat, chothia, martin or aho) plus the framework and CDR regions.

Regions are always the IMGT definition (FR1 1-26, CDR1 27-38, FR2 39-55, CDR2 56-65, FR3 66-104,
CDR3 105-117, FR4 118-128). For another scheme the domain is numbered twice, the residues are
paired by order, and each IMGT region is reported with the labels of the requested scheme. A Kabat
or Chothia CDR boundary is therefore NOT what this returns: the result says `regions_defined_by`.

ANARCII is loaded once and used under a lock, because converting to a scheme changes the model's
state. The call runs in a worker thread, so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ..core.errors import ErrorCode, ImmunoError

SCHEMES = ("imgt", "kabat", "chothia", "martin", "aho")
IMGT_REGIONS: dict[str, tuple[int, int]] = {
    "FR1": (1, 26),
    "CDR1": (27, 38),
    "FR2": (39, 55),
    "CDR2": (56, 65),
    "FR3": (66, 104),
    "CDR3": (105, 117),
    "FR4": (118, 128),
}
# Measured on heavy-chain domains from PDB (the immunology-skills repository, vhh.md): every
# heavy-chain domain scored 26.2 or more and two shark VNARs scored 17.9 and 19.5. An empirical
# floor, not a theorem, and not measured for light chains.
LOW_SCORE = 25.0
MIN_LENGTH = 50
MAX_LENGTH = 1000
_AMINO = frozenset("ACDEFGHIKLMNPQRSTVWY")

_LOCK = threading.Lock()


@dataclass(frozen=True)
class NumberedChain:
    chain_type: str  # H, K or L
    scheme: str
    score: float
    positions: list[tuple[str, str]]  # (label in `scheme`, residue), gaps removed
    regions: dict[str, tuple[str, str, str]]  # name -> (first label, last label, sequence)
    query_start: int  # 0-based index of the first numbered residue in the input
    query_end: int  # 0-based index of the last numbered residue in the input
    regions_defined_by: str = "imgt"
    low_confidence: bool = field(default=False)


def clean_sequence(sequence: str) -> str:
    """Upper-case, drop whitespace and gap characters, and refuse anything that is not protein."""
    seq = "".join(sequence.split()).upper().replace("-", "").replace("*", "")
    bad = sorted(set(seq) - _AMINO)
    if bad:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"the sequence has characters that are not standard amino acids: {''.join(bad)}",
            hint="Give one chain as one-letter codes (20 standard amino acids).",
        )
    if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"the sequence has {len(seq)} residues; a variable domain takes {MIN_LENGTH}-{MAX_LENGTH}",
            hint="Give one chain, not a table of several.",
        )
    return seq


@lru_cache(maxsize=1)
def _model() -> Any:
    try:
        from anarcii import Anarcii
    except ImportError as exc:
        raise ImmunoError(
            ErrorCode.NEEDS_EXTRA,
            "numbering needs ANARCII, which is not installed",
            hint="Install the numbering extra: pip install anarcii",
        ) from exc
    return Anarcii(seq_type="antibody", cpu=True, verbose=False)


def _label(entry: tuple[Any, Any]) -> str:
    (num, icode), _ = entry
    return f"{num}{str(icode).strip()}"


def _residues(numbering: list[Any]) -> list[tuple[Any, str]]:
    return [(entry[0], entry[1]) for entry in numbering if entry[1] != "-"]


def _number_sync(sequence: str, scheme: str) -> NumberedChain:
    with _LOCK:
        model = _model()
        result = next(iter(model.number([sequence]).values()))
        error = result.get("error")
        numbering = result.get("numbering")
        if error or not numbering:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"ANARCII could not number this as an antibody variable domain: {error or 'no numbering'}",
                hint="Check that it is one VH, VHH or VL domain and not a full-length protein.",
            )
        imgt = _residues(list(numbering))
        chain_type = str(result.get("chain_type") or "")
        score = float(result.get("score") or 0.0)
        start = int(result.get("query_start", 0))
        end = int(result.get("query_end", start + len(imgt) - 1))
        if scheme == "imgt":
            target = imgt
        else:
            converted = next(iter(model.to_scheme(scheme).values()))
            target = _residues(list(converted["numbering"]))
            if [aa for _, aa in target] != [aa for _, aa in imgt]:
                raise ImmunoError(
                    ErrorCode.INTERNAL,
                    f"the {scheme} numbering does not hold the same residues as the IMGT numbering",
                )
    positions = [(f"{n}{str(ic).strip()}", aa) for ((n, ic), aa) in target]
    spans: dict[str, list[int]] = {}
    for k, ((num, _), _aa) in enumerate(imgt):
        for name, (lo, hi) in IMGT_REGIONS.items():
            if lo <= int(num) <= hi:
                spans.setdefault(name, []).append(k)
                break
    regions = {
        name: (positions[ks[0]][0], positions[ks[-1]][0], "".join(positions[k][1] for k in ks))
        for name, ks in spans.items()
    }
    return NumberedChain(
        chain_type=chain_type,
        scheme=scheme,
        score=round(score, 1),
        positions=positions,
        regions=regions,
        query_start=start,
        query_end=end,
        low_confidence=score < LOW_SCORE,
    )


async def number_chain(sequence: str, scheme: str = "imgt") -> NumberedChain:
    """Number one variable domain. Raises ImmunoError when it cannot be numbered."""
    scheme = scheme.strip().lower()
    if scheme not in SCHEMES:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"unknown numbering scheme {scheme!r}",
            hint=f"Use one of: {', '.join(SCHEMES)}.",
        )
    seq = clean_sequence(sequence)
    return await asyncio.to_thread(_number_sync, seq, scheme)
