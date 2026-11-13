# RCSB PDB fixtures

Captured 2026-11-13 with a generic User-Agent. Bodies are verbatim unless noted.

| File | Request | Status |
|---|---|---|
| `graphql_entry_7kn5.json` | `POST https://data.rcsb.org/graphql` with `ENTRY_QUERY` from `connectors/rcsb.py`, variables `{"id": "7KN5"}` | 200 |
| `graphql_entry_missing_0zzz.json` | same query, `{"id": "0ZZZ"}` | 200, `entry: null` |
| `graphql_error_undefined_field.json` | `GET https://data.rcsb.org/graphql?query={entry(entry_id:"7KN5"){no_such_field}}` | 200 with `errors` |
| `entry_7kn5.json` | `GET https://data.rcsb.org/rest/v1/core/entry/7KN5` | 200 |
| `entry_not_found_0zzz.json` | `GET https://data.rcsb.org/rest/v1/core/entry/0ZZZ` | 404 JSON |
| `7kn5_head40.cif` | `GET https://files.rcsb.org/download/7KN5.cif`, first 40 lines of 1,540,881 bytes (sha256 966393f7...adcab) | 200 |
| `7kn5_head40.pdb` | `GET https://files.rcsb.org/download/7KN5.pdb`, first 40 lines of 1,228,851 bytes (sha256 2a0f6e31...3b9a) | 200 |
| `download_not_found.html` | `GET https://files.rcsb.org/download/0ZZZ.cif` | 404 HTML |

`GET https://files.rcsb.org/download/4V6X.pdb` (a ribosome, mmCIF only) returned the same 404
HTML page as an unknown id, so the connector checks the entry before reporting which case it is.
