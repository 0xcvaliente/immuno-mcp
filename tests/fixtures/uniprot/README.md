# UniProt fixtures

Captured 2026-11-13 from `https://rest.uniprot.org` (UniProt release 2026_04, from the
`X-UniProt-Release` header) with a generic User-Agent. Verbatim.

| File | Request | Status |
|---|---|---|
| `entry_q15116.json` | `GET /uniprotkb/Q15116?format=json&fields=accession,id,protein_name,gene_names,organism_name,organism_id,length,reviewed` | 200 |
| `search_pdcd1_human.json` | `GET /uniprotkb/search?query=gene:PDCD1 AND organism_id:9606 AND reviewed:true&fields=(same)&format=json&size=5` | 200, `X-Total-Results: 1` |
| `inactive_a9z9z9.json` | `GET /uniprotkb/A9Z9Z9.json` (a deleted entry) | 200, `entryType: Inactive` |
| `invalid_accession_400.json` | `GET /uniprotkb/NOTANACC.json` | 400 JSON |

`Q00001` turned out to be a live entry (rhamnogalacturonase A), so it is not a "missing" example.
