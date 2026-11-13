# Europe PMC fixtures

Real responses from the Europe PMC REST API (https://www.ebi.ac.uk/europepmc/webservices/rest,
version 6.9), captured on 2026-11-13. Trimmed only by removing items or fields.

| File | Request | Trimming |
|---|---|---|
| `search_pmid_36475449_core.json` | `GET /search?query=EXT_ID:36475449 AND SRC:MED&format=json&resultType=core&pageSize=1` | record keeps id, source, pmid, pmcid, doi, title, authorString, journalInfo, pubYear, abstractText, isOpenAccess, inEPMC, inPMC, license, firstPublicationDate, pubModel |
| `search_doi_lite.json` | `GET /search?query=DOI:"10.1042/BSR20221546"&format=json&resultType=lite&pageSize=2` | none |
| `search_pmcid_lite.json` | `GET /search?query=PMCID:PMC9867944&format=json&resultType=lite&pageSize=2` | none |
| `search_notfound.json` | `GET /search?query=EXT_ID:99999999 AND SRC:MED&format=json&resultType=lite&pageSize=1` | none |
| `fulltext_PMC9867944.xml` | `GET /PMC9867944/fullTextXML` | front keeps journal-meta and article ids, title, dates, abstract; body keeps the Introduction and the first two Methods subsections; back matter removed |
