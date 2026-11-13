# SKEMPI 2.0 fixture

`skempi_v2_excerpt.csv`: the header line and 16 data lines copied byte for byte from
https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv, downloaded on 2026-11-13
(1,602,208 bytes, 7,085 data rows, sha256
76f8e60d09eaa4f05fb265063886859a86895c0b2fb225e31fe82928a883972d). SKEMPI 2.0 is CC BY 4.0
(https://life.bsc.es/pid/skempi2/info/terms).

The lines were chosen to cover: protease-inhibitor rows (1CSE), the D1.3 Fv / lysozyme complex
(1VFB, single and double mutants on chains A and B), `n.b` (no binding), `>`, `<` and `~`
affinities, `298(assumed)` and empty temperatures, an AB/AG complex whose names do not say
"antibody" (HyHEL-10), a non-antibody "immunoglobulin-like" receptor, an IgG1 Fc complex and a
free-text reference.
