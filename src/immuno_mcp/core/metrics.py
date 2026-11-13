"""Canonical definitions of the scores that go into the ledger.

Every ledger row carries a `meaning`, so a table of numbers can never be read as more than it is.
Use `metric_defaults(name)` when writing a score; add new metrics here, not ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass

HIGHER, LOWER = "higher_better", "lower_better"


@dataclass(frozen=True)
class MetricDef:
    name: str
    unit: str | None
    direction: str | None  # higher_better | lower_better | None when it has no good direction
    meaning: str


_DEFS = [
    MetricDef(
        "plddt_mean",
        "0-100",
        HIGHER,
        "Mean per-residue confidence of a predicted structure. Local confidence, not binding.",
    ),
    MetricDef(
        "ptm",
        "0-1",
        HIGHER,
        "Predicted TM-score: confidence in the overall fold or complex topology.",
    ),
    MetricDef(
        "iptm",
        "0-1",
        HIGHER,
        "Confidence in the relative placement of chains in a complex. Not binding affinity.",
    ),
    MetricDef(
        "ipsae",
        "0-1",
        HIGHER,
        "Interface score derived from predicted aligned error; the variant is recorded in the "
        "parameters because implementations differ. Not binding affinity.",
    ),
    MetricDef("pae_mean", "Å", LOWER, "Mean predicted aligned error over all residue pairs."),
    MetricDef(
        "pae_interface_mean",
        "Å",
        LOWER,
        "Mean predicted aligned error between the two chain groups of an interface.",
    ),
    MetricDef(
        "net_charge_ph74",
        "e",
        None,
        "Net charge at pH 7.4 from side-chain pKa values. A sequence heuristic.",
    ),
    MetricDef(
        "pi", "pH", None, "Estimated isoelectric point from side-chain pKa values. A heuristic."
    ),
    MetricDef(
        "gravy",
        None,
        None,
        "Mean Kyte-Doolittle hydropathy. A heuristic for surface hydrophobicity.",
    ),
    MetricDef("cdr3_length", "residues", None, "CDR3 length under the numbering scheme recorded."),
    MetricDef(
        "liability_count",
        "count",
        LOWER,
        "Sequence motifs that can degrade, glycosylate or aggregate. Predictions, not measurements.",
    ),
    MetricDef(
        "liability_cdr_count",
        "count",
        LOWER,
        "Liability motifs inside CDRs, where they matter most. Predictions, not measurements.",
    ),
    MetricDef(
        "interface_contacts",
        "count",
        None,
        "Residue pairs within the distance cutoff recorded in the parameters.",
    ),
    MetricDef(
        "interface_buried_sasa",
        "Å²",
        None,
        "Surface area buried on binding, from the structure supplied (predicted or experimental).",
    ),
]

METRICS: dict[str, MetricDef] = {m.name: m for m in _DEFS}


def metric_defaults(name: str) -> MetricDef | None:
    return METRICS.get(name)
