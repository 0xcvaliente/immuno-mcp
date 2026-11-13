# PDBe fixtures

Captured 2026-11-13 from `https://www.ebi.ac.uk/pdbe/api` with a generic User-Agent. Verbatim.

| File | Request | Status |
|---|---|---|
| `best_structures_q15116.json` | `GET /mappings/best_structures/Q15116` (PD-1; 64 chains) | 200 |
| `entry_summary_7kn5.json` | `GET /pdb/entry/summary/7kn5` | 200 |
| `entry_molecules_7kn5.json` | `GET /pdb/entry/molecules/7kn5` | 200 |
| `not_found_0zzz.json` | `GET /pdb/entry/summary/0zzz` (same body for `/mappings/best_structures/P00000`) | 404 JSON |
