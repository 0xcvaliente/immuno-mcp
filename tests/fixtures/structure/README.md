# Structure and confidence fixtures (agent C2)

Captured 2026-11-13. Files are real downloads; `.gz` files are gzip of the unmodified download
unless a trim is stated. SHA-256 values are of the uncompressed download.

## Experimental structures (RCSB PDB, data CC0 1.0)

| File | URL | What it exercises | SHA-256 |
|---|---|---|---|
| `1VFB.cif.gz` | https://files.rcsb.org/download/1VFB.cif | D1.3 Fv (A light, B heavy) with hen lysozyme (C); contacts, BSA, epitope/paratope | d599fbb9391deb66c3b65d62944c6eae87a6b74cc0661d2ac9c2ced1df243a45 |
| `1VFB.pdb.gz` | https://files.rcsb.org/download/1VFB.pdb | same entry in PDB format; independent brute-force check | e95e6f68eb94395d3967723203c231006f10647c17e9cccd402cc936cffc788b |
| `1MEL.cif.gz` | https://files.rcsb.org/download/1MEL.cif | camel VHH cAb-Lys3 (A, B) with lysozyme (L, M), two copies | b82586a12aef8e1c5c53399904cf5c70885e92c6884dce9b9620d128bd9d1de0 |
| `1EJG.cif.gz` | https://files.rcsb.org/download/1EJG.cif | crambin at 0.54 Å: alternate locations, including sequence microheterogeneity | c59b29d7f079064d73667d3b510fdd9d39eb74e04cbd485022bcc9fc2383978d |
| `1AAY.cif.gz` | https://files.rcsb.org/download/1AAY.cif | Zif268 zinc fingers (A) on DNA (B, C): nucleic acids, Zn ions, water, altlocs | 41b8a72426810c08997e89928f8fb8dc897120dabff4011760bc0864aa8afe28 |
| `1L2Y.cif.gz` | https://files.rcsb.org/download/1L2Y.cif | Trp-cage NMR ensemble: 38 models, hydrogens | 8face44439c270c7bb9734a5eda12551460a9228969153f9caa306e1da75c671 |

## ipSAE reference example (`ipsae_aurka_tpx2/`) - NOT COMMITTED

**This directory is git-ignored on purpose.** It contains AlphaFold Server output, which is
for non-commercial use only and may only be redistributed together with those restrictions
(https://alphafoldserver.com/output-terms, read 2026-11-14). This repository is MIT licensed,
so it does not ship the files. Recreate them for your own non-commercial research with
`python scripts/fetch_reference_fixtures.py`; every download is checked against the SHA-256
below. The tests that need them skip themselves when the directory is missing.

From https://github.com/DunbrackLab/IPSAE (commit 6174cf9e71cb1bd660cc805856a18c4871a6dec3,
`Example/`), MIT licence for the code (copied as `LICENSE.IPSAE`). The model files are
AlphaFold Server output: the mmCIF header states that use is subject to
alphafoldserver.com/output-terms (kept unchanged in the file).

| File | Source path | Note |
|---|---|---|
| `fold_aurka_0_tpx2_0_model_0.cif.gz` | `Example/fold_aurka_0_tpx2_0_model_0.cif` | unmodified; SHA-256 65d7aa6d739bad7dda4fa4ed48b2e95c7af2014ed43faa4300342c35725bd6ad. Chain A has two TPO (phosphothreonine), C is ATP, D/E are Mg |
| `fold_aurka_0_tpx2_0_full_data_0.no_contact_probs.json.gz` | `Example/fold_aurka_0_tpx2_0_full_data_0.json` | TRIMMED: the `contact_probs` array was removed to halve the size; all other keys unchanged. Original SHA-256 419dba4195671c4f710f12441a1bce09211adbb4196ed788c8146c2a6f06ed49 |
| `fold_aurka_0_tpx2_0_model_0_10_10.txt` | same name | reference ipsae.py output, PAE cutoff 10 Å, distance cutoff 10 Å |
| `fold_aurka_0_tpx2_0_model_0_10_10_byres.txt.gz` | same name, gzipped | reference per-residue output |

## Confidence JSON

* `colabfold_raf1_scores_first40.json`: from
  `Example/RAF1_KSR1_MEK1_9f755_scores_alphafold2_multimer_v3_model_1_seed_000.json.gz` in the
  same IPSAE commit (ColabFold AlphaFold2-multimer v3 output). TRIMMED: `plddt` cut to the
  first 40 residues and `pae` to the first 40 x 40 block; every other key unchanged.
  Used only to test that the ColabFold format is recognised and read.
* `boltz2_confidence_docs_example.json`: the example confidence file printed in Boltz's
  `docs/prediction.md` (https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md), copied
  with its `#` comments removed so it is valid JSON. It is a documentation example, not a run.

No JapanFold result file is included: getting one requires a paid job. The JapanFold formats
are supported from their documentation (https://api.japanfold.aiand.com/v1/openapi.json and
https://japanfold.aiand.com/skill.md, read 2026-11-13) and are tested with objects built from
the documented key names only.
