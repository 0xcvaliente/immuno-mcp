# PubTator3 fixtures

Real responses from the PubTator3 API (https://www.ncbi.nlm.nih.gov/research/pubtator3-api),
captured on 2026-11-13. Trimmed only by removing items or fields (no values edited).

| File | Request | Trimming |
|---|---|---|
| `search_pd1_nanobody.json` | `GET /search/?text=PD-1 nanobody&page=1` | first 6 hits, `citations` and `facets` removed |
| `search_pd1_nanobody_date.json` | `GET /search/?text=PD-1 nanobody&page=1&sort=date desc` | first 4 hits, `citations` and `facets` removed |
| `search_empty.json` | `GET /search/?text=qzxqzxwvvq nanobodyqqq&page=1` | none |
| `export_biocjson_two.json` | `GET /publications/export/biocjson?pmids=36475449,34562519&full=true` | 36475449 (PMC full text) keeps passages 0-16, 43, 46, 47 and the first ABBR, REF and COMP_INT passages, at most 6 annotations each; relations cut to 1 per document |
| `export_missing_400.json` | `GET /publications/export/biocjson?pmids=99999999&full=true` (HTTP 400) | none |
| `autocomplete_pd1_gene.json` | `GET /entity/autocomplete/?query=PD-1&concept=gene&limit=5` | none |

Observed behaviour recorded in `connectors/pubtator.py`: page size is fixed at 10; any `full`
value returns PMC full text; unknown PMIDs are left out, and a request with none known is a 400.
