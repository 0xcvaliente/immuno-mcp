# SAbDab2 fixtures

Captured 2026-11-13 from the live SAbDab2 API, `https://sabdab.opig.stats.ox.ac.uk/api`
(OpenAPI version 2.1.7), with a generic User-Agent. Bodies are verbatim unless noted.

| File | Request | Status |
|---|---|---|
| `stats_summary.json` | `GET /stats/summary` | 200 |
| `search_pd1_limit3.json` | `GET /frontend/antibody-instance-search?limit=3&offset=0&order_by=id&binds_to_antigen=Programmed+cell+death+pro` | 200, total 51 |
| `search_empty_pd-1.json` | `GET /frontend/antibody-instance-search?limit=3&offset=0&order_by=id&binds_to_antigen=PD-1` | 200, total 0 |
| `search_422_antibody_type_vhh.json` | `GET /frontend/antibody-instance-search?limit=1&offset=0&order_by=id&antibody_type=VHH` | 422 |
| `frontend_pdb_7kn5.json` | `GET /frontend/pdb/pdb_00007kn5` | 200 |
| `frontend_pdb_1crn_404.json` | `GET /frontend/pdb/pdb_00001crn` (crambin, no antibody) | 404 |
| `antigen_instances_first2.json` | `GET /antigen-instances?limit=2` (a hapten and a chloride ion bound to one Fab) | 200 |
| `old_site_app_page.html` | `GET https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/7kn5/` (follows a redirect to the new app page) | 200, HTML |
| `openapi_2.1.7_trimmed.json` | `GET /openapi.json`, trimmed to `info` (contact removed), three paths and five schemas | 200 |

Observations recorded while capturing (they shape `connectors/sabdab.py`):

* `/pdb/7kn5` is a 404; SAbDab2 uses extended ids (`pdb_00007kn5`).
* `binds_to_antigen` is case-insensitive and limited to 25 characters. `PD-1` matches nothing.
* `/antigen-instances?antigen_name=Spike` returned 0 although `binds_to_antigen=Spike` returned 3247.
* `release_date` equals `deposition_date` in every record captured (8 entries). RCSB gives 7KN5
  deposit 2020-11-04 and release 2021-01-20.
* PDB 9ZZZ exists in SAbDab2 (a scFv complex), so it cannot serve as a "missing" example.
