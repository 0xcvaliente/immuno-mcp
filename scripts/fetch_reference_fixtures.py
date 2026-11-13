"""Recreate the ipSAE reference fixture that is deliberately NOT committed to git.

The AURKA/TPX2 example from https://github.com/DunbrackLab/IPSAE contains AlphaFold Server
output. Those outputs are for non-commercial use, and anyone who redistributes them must pass the
restrictions on (https://alphafoldserver.com/output-terms). This repository is MIT licensed, so it
does not ship them. Run this script to download them for your own non-commercial research use.

    python scripts/fetch_reference_fixtures.py            # writes to tests/fixtures/structure/
    python scripts/fetch_reference_fixtures.py --dest DIR  # write somewhere else

Every download is checked against a SHA-256 recorded from the upstream commit below. Tests that
need the files skip themselves when the directory is missing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

COMMIT = "6174cf9e71cb1bd660cc805856a18c4871a6dec3"
RAW = f"https://raw.githubusercontent.com/DunbrackLab/IPSAE/{COMMIT}/"
DEFAULT_DEST = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "structure" / "ipsae_aurka_tpx2"
)

# upstream path -> (sha256 of the upstream file, how it is stored here, stored file name)
FILES = {
    "Example/fold_aurka_0_tpx2_0_model_0.cif": (
        "65d7aa6d739bad7dda4fa4ed48b2e95c7af2014ed43faa4300342c35725bd6ad",
        "gzip",
        "fold_aurka_0_tpx2_0_model_0.cif.gz",
    ),
    "Example/fold_aurka_0_tpx2_0_full_data_0.json": (
        "419dba4195671c4f710f12441a1bce09211adbb4196ed788c8146c2a6f06ed49",
        "drop_contact_probs_gzip",
        "fold_aurka_0_tpx2_0_full_data_0.no_contact_probs.json.gz",
    ),
    "Example/fold_aurka_0_tpx2_0_model_0_10_10.txt": (
        "0351ddf95b8e8188244c0df7214c7c55d208a77cd8dae11d3f4acf9f65e8e187",
        "plain",
        "fold_aurka_0_tpx2_0_model_0_10_10.txt",
    ),
    "Example/fold_aurka_0_tpx2_0_model_0_10_10_byres.txt": (
        "c8595f0a25abae3d18fb803b9a2218f961debd791dbfdd4288faa264e6c2bf2b",
        "gzip",
        "fold_aurka_0_tpx2_0_model_0_10_10_byres.txt.gz",
    ),
    "LICENSE": (
        "11b38cd2c82e0d197145ddc8d6312937f569868c8e58f11df9b40c8a37453ff3",
        "plain",
        "LICENSE.IPSAE",
    ),
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "immuno-mcp-fixture-fetch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def store(data: bytes, mode: str) -> bytes:
    if mode == "plain":
        return data
    if mode == "gzip":
        return gzip.compress(data, mtime=0)
    if mode == "drop_contact_probs_gzip":
        # The contact_probs array is large and unused by the tests; every other key is unchanged.
        parsed = json.loads(data)
        parsed.pop("contact_probs", None)
        return gzip.compress(json.dumps(parsed, separators=(",", ":")).encode(), mtime=0)
    raise ValueError(mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    print(
        "These files include AlphaFold Server output, which is for non-commercial use only and\n"
        "carries redistribution conditions: https://alphafoldserver.com/output-terms\n",
        file=sys.stderr,
    )
    args.dest.mkdir(parents=True, exist_ok=True)
    for upstream, (expected_sha, mode, name) in FILES.items():
        data = fetch(RAW + upstream)
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha:
            print(f"FAILED {upstream}: sha256 {digest} != expected {expected_sha}", file=sys.stderr)
            return 1
        (args.dest / name).write_bytes(store(data, mode))
        print(f"ok  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
