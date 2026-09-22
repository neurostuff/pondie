"""Fetch the vocabulary releases `data/vocab/` is expected to hold.

    python -m pondie.vocabularies.fetch mondo

Here because the expectation existed and the fetcher did not: `paths.py` documented
`mondo.json`, `load_mondo` read it, and nothing had ever written it. Pinned by URL and
checked for size rather than hashed -- MONDO ships monthly, and a hash would turn a data
update into a code change. The version is in the graph's own metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from pondie import paths

#: name -> (url, destination, smallest plausible size). The floor catches a redirect or
#: an error page written to the destination and read later as an empty vocabulary.
RELEASES: dict[str, tuple[str, Path, int]] = {
    "mondo": (
        "https://purl.obolibrary.org/obo/mondo.json",
        paths.VOCAB / "mondo.json",
        50_000_000,
    ),
}


def fetch(name: str, force: bool = False) -> Path:
    """Download one release to `data/vocab/`, atomically. Returns where it landed."""
    url, destination, floor = RELEASES[name]
    if destination.is_file() and not force:
        print(f"{name}: already at {destination} ({destination.stat().st_size:,} bytes)")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"{name}: fetching {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:  # noqa: S310
        while chunk := response.read(1 << 20):
            out.write(chunk)
    size = partial.stat().st_size
    if size < floor:
        partial.unlink()
        raise OSError(f"{name}: {size:,} bytes is too small to be the release; not saved")
    # Moved rather than written in place, so an interrupted fetch leaves nothing.
    partial.replace(destination)
    print(f"{name}: {size:,} bytes -> {destination}")
    return destination


def version(name: str) -> str:
    """What the fetched file says it is. For a run to record beside its mappings."""
    _url, destination, _floor = RELEASES[name]
    if not destination.is_file():
        return ""
    graph = json.loads(destination.read_text())["graphs"][0]
    return str((graph.get("meta") or {}).get("version") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=sorted(RELEASES), choices=sorted(RELEASES))
    parser.add_argument("--force", action="store_true", help="re-download an existing file")
    args = parser.parse_args()
    for name in args.names or sorted(RELEASES):
        path = fetch(name, args.force)
        if path.suffix == ".json":
            print(f"{name}: version {version(name) or '(none declared)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
