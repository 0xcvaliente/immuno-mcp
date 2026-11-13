"""Structure analysis on file TEXT: chains, interface contacts, buried surface area, epitope and
paratope.

Pure functions: no network, no file paths, no MCP imports. Every function takes the text of an
mmCIF or PDB file and its format, and raises ``ImmunoError(INVALID_INPUT)`` for input it cannot
read.

Conventions, repeated in every result so a number is never read out of context:

* Chains are AUTHOR chain ids (``auth_asym_id`` in mmCIF, column 22 in PDB format): the ids used
  in papers, SAbDab and PDB-format files. Residues carry the author sequence number and the
  insertion code.
* Only the first model is analysed. Alternate locations: the highest-occupancy altloc of each
  residue (Biotite ``altloc="occupancy"``).
* Contacts and surfaces use the heavy atoms of polymer residues: amino acids (including modified
  ones such as MSE, PCA or TPO) and nucleotides. Water, ions and other ligands, including
  glycans, are left out.
* A residue pair is in contact when any heavy-atom pair is within the cutoff (distance <= cutoff).
* Buried surface area is SASA(A) + SASA(B) - SASA(A and B together), computed with Biotite's
  Shrake-Rupley ``sasa`` (probe 1.4 Å, ProtOr radii, 1000 points per atom) on the selected chains
  only, so other chains in the file do not affect it.
"""

from __future__ import annotations

import io
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import biotite
import biotite.structure as struc
import numpy as np
from biotite.structure.info import one_letter_code, vdw_radius_protor
from biotite.structure.io import pdb as bpdb
from biotite.structure.io import pdbx

from ..core.errors import ErrorCode, ImmunoError

MAX_CUTOFF = 15.0
PROBE_RADIUS = 1.4
SASA_POINTS = 1000
DEFAULT_RADIUS = 1.8  # Biotite's own fallback for atoms ProtOr does not cover

STANDARD_AA = frozenset(
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
    ]
)
STANDARD_NUC = frozenset({"A", "C", "G", "U", "DA", "DC", "DG", "DT", "DU", "I", "DI"})

ALTLOC_RULE = "highest-occupancy altloc per residue"
ATOM_SCOPE = (
    "heavy atoms of polymer residues (amino acids incl. modified ones, nucleotides); "
    "water, ions and ligands (including glycans) excluded"
)
SASA_METHOD = (
    f"Shrake-Rupley SASA (Biotite {biotite.__version__} structure.sasa), probe {PROBE_RADIUS} Å, "
    f"ProtOr radii, {SASA_POINTS} points per atom; buried = SASA(A) + SASA(B) - SASA(A+B)"
)


# ---- public data types -----------------------------------------------------------------------


@dataclass
class ChainInfo:
    """One author chain. ``sequence`` holds the residues OBSERVED in the coordinates, in order.

    Extra fields beyond the contract have defaults: ``gaps`` counts chain breaks found from
    backbone geometry (missing residues), ``nonstandard`` lists modified residue names written
    as their parent one-letter code (X when there is none), ``ligands`` lists non-polymer
    residue names (not water) that carry this chain id.
    """

    chain_id: str
    sequence: str
    length: int
    molecule: str  # "protein" | "nucleic" | "other"
    gaps: int = 0
    first_residue: str | None = None
    last_residue: str | None = None
    nonstandard: list[str] = field(default_factory=list)
    ligands: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    res_num: int
    ins_code: str
    res_name: str

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.res_name}{self.res_num}{self.ins_code}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain": self.chain,
            "res_name": self.res_name,
            "res_num": self.res_num,
            "ins_code": self.ins_code,
        }


@dataclass
class LoadedStructure:
    """First model of a structure, altlocs resolved, with a polymer mask per atom."""

    atoms: struc.AtomArray
    fmt: str
    model_count: int
    polymer: np.ndarray  # bool per atom
    has_label_seq: bool
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "format": self.fmt,
            "model": 1,
            "models_in_file": self.model_count,
            "altloc": ALTLOC_RULE,
            "chain_ids": "author (auth_asym_id)",
            "atoms_used": ATOM_SCOPE,
        }


# ---- loading ---------------------------------------------------------------------------------


def normalize_format(fmt: str) -> str:
    f = (fmt or "").strip().lower().lstrip(".")
    if f in ("cif", "mmcif"):
        return "cif"
    if f in ("pdb", "ent"):
        return "pdb"
    raise ImmunoError(
        ErrorCode.INVALID_INPUT,
        f"unknown structure format {fmt!r}",
        hint="Use 'cif' (mmCIF) or 'pdb'.",
    )


_PDB_RECORDS = ("ATOM  ", "HETATM", "HEADER", "MODEL ", "CRYST1", "REMARK", "EXPDTA", "TITLE ")


def sniff_format(text: str) -> str | None:
    """'cif' or 'pdb' from the content, or None when it is neither.

    mmCIF: the first line that is not blank or a ``#`` comment starts with ``data_``.
    PDB: a fixed-column record (ATOM, HETATM, HEADER, ...) appears near the top.
    """
    for n, line in enumerate(io.StringIO(text)):
        if n > 5000:
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("data_"):
            return "cif"
        if line.startswith(_PDB_RECORDS):
            return "pdb"
        if n > 200:
            break
    return None


def load_structure(text: str, fmt: str, *, extra_fields: Iterable[str] = ()) -> LoadedStructure:
    """Parse the first model. Raises INVALID_INPUT with a useful message on bad input."""
    fmt = normalize_format(fmt)
    if not isinstance(text, str) or not text.strip():
        raise ImmunoError(ErrorCode.INVALID_INPUT, "the structure text is empty")
    found = sniff_format(text)
    if found is None:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"the text is not an mmCIF or PDB coordinate file (format {fmt!r} was given)",
            hint="Pass the full text of a .cif (starts with 'data_') or .pdb file with ATOM records.",
        )
    if found != fmt:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"the text looks like {'mmCIF' if found == 'cif' else 'PDB format'}, not {fmt!r}",
            hint=f"Pass fmt='{found}'.",
        )
    wanted = {"b_factor", "occupancy", *extra_fields}
    try:
        with warnings.catch_warnings():
            # Biotite warns when optional auth_* columns are missing (AlphaFold 3 files) and
            # falls back to label_* columns, which is what we want.
            warnings.simplefilter("ignore")
            if fmt == "cif":
                cif = pdbx.CIFFile.read(io.StringIO(text))
                block = cif.block
                if "atom_site" not in block:
                    raise ImmunoError(
                        ErrorCode.INVALID_INPUT, "the mmCIF has no atom_site category"
                    )
                has_label_seq = "label_seq_id" in block["atom_site"]
                if has_label_seq:
                    wanted.add("label_seq_id")
                model_count = int(pdbx.get_model_count(cif))
                atoms = pdbx.get_structure(
                    cif,
                    model=1,
                    use_author_fields=True,
                    altloc="occupancy",
                    extra_fields=sorted(wanted),
                )
            else:
                has_label_seq = False
                pfile = bpdb.PDBFile.read(io.StringIO(text))
                model_count = int(pfile.get_model_count())
                atoms = pfile.get_structure(
                    model=1, altloc="occupancy", extra_fields=sorted(wanted)
                )
    except ImmunoError:
        raise
    except Exception as exc:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"could not read the structure as {fmt}: {type(exc).__name__}: {exc}",
        ) from exc
    if atoms.array_length() == 0:
        raise ImmunoError(ErrorCode.INVALID_INPUT, "the structure has no atoms in model 1")
    if not np.isfinite(atoms.coord).all():
        raise ImmunoError(ErrorCode.INVALID_INPUT, "the structure has non-numeric coordinates")

    water = struc.filter_solvent(atoms)
    if has_label_seq:
        label_seq = np.asarray(atoms.get_annotation("label_seq_id"))
        polymer = (label_seq != ".") & (label_seq != "?") & ~water
    else:
        polymer = (struc.filter_amino_acids(atoms) | struc.filter_nucleotides(atoms)) & ~water
    notes: list[str] = []
    if model_count > 1:
        notes.append(f"The file has {model_count} models; only model 1 was analysed.")
    return LoadedStructure(
        atoms=atoms,
        fmt=fmt,
        model_count=model_count,
        polymer=np.asarray(polymer, dtype=bool),
        has_label_seq=has_label_seq,
        warnings=notes,
    )


# ---- residues and chains ---------------------------------------------------------------------


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        seen.setdefault(str(v), None)
    return list(seen)


def _residue_keys(atoms: struc.AtomArray, starts: np.ndarray) -> list[ResidueKey]:
    return [
        ResidueKey(
            chain=str(atoms.chain_id[s]),
            res_num=int(atoms.res_id[s]),
            ins_code=str(atoms.ins_code[s]),
            res_name=str(atoms.res_name[s]),
        )
        for s in starts
    ]


def _atom_to_residue(n_atoms: int, starts: np.ndarray) -> np.ndarray:
    return np.searchsorted(starts, np.arange(n_atoms), side="right") - 1


@lru_cache(maxsize=1024)
def _one_letter(res_name: str) -> str:
    code = one_letter_code(res_name)
    return code if code and len(code) == 1 else "X"


def _molecule_of(atoms: struc.AtomArray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(struc.filter_amino_acids(atoms), dtype=bool),
        np.asarray(struc.filter_nucleotides(atoms), dtype=bool),
    )


def _atom_xyz(res: struc.AtomArray, name: str) -> np.ndarray | None:
    idx = np.flatnonzero(res.atom_name == name)
    return res.coord[idx[0]].astype(float) if idx.size else None


def _count_gaps(residues: list[struc.AtomArray], kinds: list[str]) -> int:
    gaps = 0
    for prev, nxt, k1, k2 in zip(residues, residues[1:], kinds, kinds[1:], strict=False):
        if k1 == "aa" and k2 == "aa":
            c, n = _atom_xyz(prev, "C"), _atom_xyz(nxt, "N")
            if c is not None and n is not None:
                gaps += float(np.linalg.norm(c - n)) > 2.0
                continue
            ca1, ca2 = _atom_xyz(prev, "CA"), _atom_xyz(nxt, "CA")
            if ca1 is not None and ca2 is not None:
                gaps += float(np.linalg.norm(ca1 - ca2)) > 4.5
        elif k1 == "nuc" and k2 == "nuc":
            o3, p = _atom_xyz(prev, "O3'"), _atom_xyz(nxt, "P")
            if o3 is not None and p is not None:
                gaps += float(np.linalg.norm(o3 - p)) > 2.2
    return int(gaps)


def chains_from_loaded(ls: LoadedStructure) -> list[ChainInfo]:
    atoms = ls.atoms
    keep = ~np.asarray(struc.filter_solvent(atoms), dtype=bool)
    atoms_nw = atoms[keep]
    polymer = ls.polymer[keep]
    if atoms_nw.array_length() == 0:
        return []
    is_aa, is_nuc = _molecule_of(atoms_nw)
    starts: np.ndarray = struc.get_residue_starts(atoms_nw)
    stops = np.append(starts[1:], atoms_nw.array_length())
    by_chain: dict[str, dict[str, Any]] = {}
    for s, e in zip(starts, stops, strict=True):
        cid = str(atoms_nw.chain_id[s])
        entry = by_chain.setdefault(
            cid, {"poly": [], "kinds": [], "keys": [], "ligands": {}, "nonstd": {}}
        )
        name = str(atoms_nw.res_name[s])
        if polymer[s]:
            kind = "aa" if is_aa[s] else ("nuc" if is_nuc[s] else "other")
            entry["poly"].append(atoms_nw[s:e])
            entry["kinds"].append(kind)
            entry["keys"].append(
                ResidueKey(cid, int(atoms_nw.res_id[s]), str(atoms_nw.ins_code[s]), name)
            )
            if name not in STANDARD_AA and name not in STANDARD_NUC:
                entry["nonstd"][name] = None
        else:
            entry["ligands"][name] = None

    out: list[ChainInfo] = []
    for cid, entry in by_chain.items():
        kinds: list[str] = entry["kinds"]
        keys: list[ResidueKey] = entry["keys"]
        n_aa, n_nuc = kinds.count("aa"), kinds.count("nuc")
        if keys:
            molecule = (
                "protein" if n_aa >= n_nuc and n_aa > 0 else ("nucleic" if n_nuc else "other")
            )
            seq = "".join(_one_letter(k.res_name) for k in keys)
            out.append(
                ChainInfo(
                    chain_id=cid,
                    sequence=seq,
                    length=len(seq),
                    molecule=molecule,
                    gaps=_count_gaps(entry["poly"], kinds),
                    first_residue=f"{keys[0].res_name}{keys[0].res_num}{keys[0].ins_code}",
                    last_residue=f"{keys[-1].res_name}{keys[-1].res_num}{keys[-1].ins_code}",
                    nonstandard=list(entry["nonstd"]),
                    ligands=sorted(entry["ligands"]),
                )
            )
        else:
            ligands = sorted(entry["ligands"])
            out.append(
                ChainInfo(
                    chain_id=cid,
                    sequence="",
                    length=0,
                    molecule="other",
                    ligands=ligands,
                )
            )
    return out


def read_chains(text: str, fmt: str) -> list[ChainInfo]:
    """Chains of model 1 in file order, with the observed sequence of each polymer chain."""
    return chains_from_loaded(load_structure(text, fmt))


# ---- selections and validation ---------------------------------------------------------------


def polymer_chain_ids(ls: LoadedStructure) -> list[str]:
    return _unique_in_order(ls.atoms.chain_id[ls.polymer])


def _validate_groups(
    ls: LoadedStructure, group_a: list[str], group_b: list[str], names: tuple[str, str]
) -> tuple[list[str], list[str]]:
    available = polymer_chain_ids(ls)
    clean: list[list[str]] = []
    for label, group in zip(names, (group_a, group_b), strict=True):
        if isinstance(group, str):
            group = [group]
        ids = [str(c).strip() for c in (group or []) if str(c).strip()]
        if not ids:
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"{label} is empty",
                hint=f"Name at least one chain. Polymer chains here: {', '.join(available)}.",
            )
        missing = [c for c in ids if c not in available]
        if missing:
            near = [a for a in available if a.lower() in {m.lower() for m in missing}]
            raise ImmunoError(
                ErrorCode.INVALID_INPUT,
                f"{label}: chain(s) {', '.join(missing)} not found among polymer chains",
                hint=f"Polymer chains (author ids, case-sensitive): {', '.join(available)}."
                + (f" Did you mean {', '.join(near)}?" if near else ""),
            )
        clean.append(_unique_in_order(ids))
    overlap = sorted(set(clean[0]) & set(clean[1]))
    if overlap:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"chain(s) {', '.join(overlap)} are in both {names[0]} and {names[1]}",
            hint="Each chain belongs to one side of the interface.",
        )
    return clean[0], clean[1]


def _validate_cutoff(cutoff: float) -> float:
    try:
        value = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT, f"cutoff must be a number, got {cutoff!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0 or value > MAX_CUTOFF:
        raise ImmunoError(
            ErrorCode.INVALID_INPUT,
            f"cutoff must be > 0 and <= {MAX_CUTOFF} Å, got {cutoff!r}",
            hint="4.5 Å (heavy atoms) is a common contact definition; 4.0-5.0 Å are typical.",
        )
    return value


def _group_atoms(ls: LoadedStructure, chains: list[str]) -> struc.AtomArray:
    atoms = ls.atoms
    mask = ls.polymer & np.isin(atoms.chain_id, chains) & ~np.isin(atoms.element, ["H", "D"])
    return atoms[mask]


@dataclass
class _Side:
    atoms: struc.AtomArray
    starts: np.ndarray
    keys: list[ResidueKey]
    atom_res: np.ndarray
    seq_index: list[int]


def _side(atoms: struc.AtomArray) -> _Side:
    starts: np.ndarray = struc.get_residue_starts(atoms)
    keys = _residue_keys(atoms, starts)
    counters: dict[str, int] = {}
    seq_index = []
    for k in keys:
        seq_index.append(counters.get(k.chain, 0))
        counters[k.chain] = counters.get(k.chain, 0) + 1
    return _Side(atoms, starts, keys, _atom_to_residue(atoms.array_length(), starts), seq_index)


# ---- contacts --------------------------------------------------------------------------------


def contact_atom_pairs(
    coord_a: np.ndarray, coord_b: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Atom index pairs (i in A, j in B) with distance <= cutoff, and their distances."""
    if len(coord_a) == 0 or len(coord_b) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0)
    cells: Any = struc.CellList(coord_b, cell_size=max(cutoff, 1.0))
    pairs = cells.get_atoms(coord_a, cutoff + 1e-3, result_format=struc.CellList.Result.PAIRS)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    ia, ib = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(coord_a[ia].astype(np.float64) - coord_b[ib].astype(np.float64), axis=1)
    keep = dist <= cutoff
    return ia[keep], ib[keep], dist[keep]


@lru_cache(maxsize=8192)
def _protor(res_name: str, atom_name: str) -> float | None:
    try:
        return vdw_radius_protor(res_name, atom_name)
    except (KeyError, ValueError):
        return None


def _radii(atoms: struc.AtomArray) -> tuple[np.ndarray, int]:
    radii = np.empty(atoms.array_length(), dtype=np.float32)
    defaulted = 0
    for i, (rn, an) in enumerate(zip(atoms.res_name, atoms.atom_name, strict=True)):
        r = _protor(str(rn), str(an))
        if r is None:
            defaulted += 1
            r = DEFAULT_RADIUS
        radii[i] = r
    return radii, defaulted


def _sasa(atoms: struc.AtomArray, radii: np.ndarray) -> np.ndarray:
    values = struc.sasa(
        atoms,
        probe_radius=PROBE_RADIUS,
        point_number=SASA_POINTS,
        vdw_radii=radii,
        ignore_ions=False,
    )
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)


def _analyse(
    ls: LoadedStructure,
    chains_a: list[str],
    chains_b: list[str],
    cutoff: float,
    *,
    with_sasa: bool = True,
) -> dict[str, Any]:
    a = _side(_group_atoms(ls, chains_a))
    b = _side(_group_atoms(ls, chains_b))
    warn = list(ls.warnings)
    ia, ib, dist = contact_atom_pairs(a.atoms.coord, b.atoms.coord, cutoff)
    ra, rb = a.atom_res[ia], b.atom_res[ib]

    n_b = max(len(b.keys), 1)
    pair_id = ra * n_b + rb
    uniq, inverse = np.unique(pair_id, return_inverse=True)
    pair_min = np.full(len(uniq), np.inf)
    np.minimum.at(pair_min, inverse, dist)
    pair_count = np.bincount(inverse, minlength=len(uniq))

    res_min_a = np.full(len(a.keys), np.inf)
    res_min_b = np.full(len(b.keys), np.inf)
    np.minimum.at(res_min_a, ra, dist)
    np.minimum.at(res_min_b, rb, dist)
    partners_a = np.zeros(len(a.keys), dtype=int)
    partners_b = np.zeros(len(b.keys), dtype=int)
    for pid in uniq:
        partners_a[pid // n_b] += 1
        partners_b[pid % n_b] += 1

    buried_res_a = np.zeros(len(a.keys))
    buried_res_b = np.zeros(len(b.keys))
    bsa: dict[str, Any] | None = None
    if with_sasa:
        radii_a, def_a = _radii(a.atoms)
        radii_b, def_b = _radii(b.atoms)
        alone_a = _sasa(a.atoms, radii_a)
        alone_b = _sasa(b.atoms, radii_b)
        both = _sasa(a.atoms + b.atoms, np.concatenate([radii_a, radii_b]))
        n_a_atoms = a.atoms.array_length()
        buried_a = alone_a - both[:n_a_atoms]
        buried_b = alone_b - both[n_a_atoms:]
        np.add.at(buried_res_a, a.atom_res, buried_a)
        np.add.at(buried_res_b, b.atom_res, buried_b)
        total = float(buried_a.sum() + buried_b.sum())
        bsa = {
            "total": round(total, 1),
            "side_a": round(float(buried_a.sum()), 1),
            "side_b": round(float(buried_b.sum()), 1),
            "sasa_a_alone": round(float(alone_a.sum()), 1),
            "sasa_b_alone": round(float(alone_b.sum()), 1),
            "sasa_complex": round(float(both.sum()), 1),
            "unit": "Å²",
            "method": SASA_METHOD,
        }
        if def_a + def_b:
            warn.append(
                f"{def_a + def_b} atoms have no ProtOr radius (for example phosphorus, selenium "
                f"or unusual atom names) and used {DEFAULT_RADIUS} Å in the SASA calculation."
            )

    def rows(side: _Side, res_min: np.ndarray, partners: np.ndarray, buried: np.ndarray) -> list:
        out = []
        for idx in np.flatnonzero(partners > 0):
            row = side.keys[idx].as_dict()
            row["min_distance"] = round(float(res_min[idx]), 2)
            row["partners"] = int(partners[idx])
            if with_sasa:
                row["buried_sasa"] = round(float(buried[idx]), 1)
            row["seq_index"] = side.seq_index[idx]
            out.append(row)
        return out

    pairs = [
        {
            "a": a.keys[int(pid // n_b)].label,
            "b": b.keys[int(pid % n_b)].label,
            "min_distance": round(float(pair_min[k]), 2),
            "atom_pairs": int(pair_count[k]),
        }
        for k, pid in enumerate(uniq)
    ]
    if not pairs:
        warn.append(
            f"No residue pairs within {cutoff} Å: these chain groups do not touch in model 1."
        )
    return {
        "cutoff": cutoff,
        "contact_rule": "residue pair with any heavy-atom pair at distance <= cutoff",
        "n_residue_pairs": len(pairs),
        "n_atom_pairs": len(dist),
        "residue_pairs": pairs,
        "residues_a": rows(a, res_min_a, partners_a, buried_res_a),
        "residues_b": rows(b, res_min_b, partners_b, buried_res_b),
        "buried_sasa": bsa,
        "structure": ls.describe(),
        "warnings": warn,
    }


def interface_contacts(
    text: str, fmt: str, group_a: list[str], group_b: list[str], cutoff: float = 4.5
) -> dict[str, Any]:
    """Residue contacts between two chain groups, interface residues and buried surface area.

    Returns ``residue_pairs`` (labels ``chain:RESnum[ins]``), ``residues_a``/``residues_b``
    (interface residues with author numbering, closest distance, number of partner residues
    and buried SASA), ``buried_sasa`` and a ``structure`` block stating the conventions used.
    """
    cutoff = _validate_cutoff(cutoff)
    ls = load_structure(text, fmt)
    chains_a, chains_b = _validate_groups(ls, group_a, group_b, ("group_a", "group_b"))
    out = _analyse(ls, chains_a, chains_b, cutoff)
    return {"group_a": chains_a, "group_b": chains_b, **out}


def buried_surface_area(
    text: str, fmt: str, group_a: list[str], group_b: list[str]
) -> dict[str, Any]:
    """Buried surface area between two chain groups (Å²), with the method stated."""
    ls = load_structure(text, fmt)
    chains_a, chains_b = _validate_groups(ls, group_a, group_b, ("group_a", "group_b"))
    out = _analyse(ls, chains_a, chains_b, 4.5)
    return {
        "group_a": chains_a,
        "group_b": chains_b,
        **out["buried_sasa"],
        "structure": out["structure"],
        "warnings": out["warnings"],
    }


def epitope_paratope(
    text: str,
    fmt: str,
    antibody_chains: list[str],
    antigen_chains: list[str],
    cutoff: float = 4.5,
) -> dict[str, Any]:
    """Paratope (antibody residues) and epitope (antigen residues) within ``cutoff`` of each other.

    Each residue row has chain, res_name, res_num (author), ins_code, the closest heavy-atom
    distance to the other side, the number of partner residues, its buried SASA and
    ``seq_index`` (0-based position in the chain's observed sequence from ``read_chains``).
    """
    return epitope_paratope_loaded(
        load_structure(text, fmt), antibody_chains, antigen_chains, cutoff
    )


def epitope_paratope_loaded(
    ls: LoadedStructure, antibody_chains: list[str], antigen_chains: list[str], cutoff: float = 4.5
) -> dict[str, Any]:
    """``epitope_paratope`` on an already parsed structure."""
    cutoff = _validate_cutoff(cutoff)
    ab, ag = _validate_groups(
        ls, antibody_chains, antigen_chains, ("antibody_chains", "antigen_chains")
    )
    out = _analyse(ls, ab, ag, cutoff)
    return {
        "antibody_chains": ab,
        "antigen_chains": ag,
        "cutoff": out["cutoff"],
        "contact_rule": out["contact_rule"],
        "n_residue_pairs": out["n_residue_pairs"],
        "paratope": out["residues_a"],
        "epitope": out["residues_b"],
        "residue_pairs": out["residue_pairs"],
        "buried_sasa": out["buried_sasa"],
        "structure": out["structure"],
        "warnings": out["warnings"],
    }
