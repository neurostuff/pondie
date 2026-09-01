"""Where this repository keeps its data, in one place.

Every module that needed a data path used to count `..` from its own file, so a module that
moved directory silently started reading somewhere else -- `_abbreviations` pointed at a
`pondie/data/` that has never existed, and nothing failed until a lookup returned no matches,
which is indistinguishable from a vocabulary that simply does not have the word.

One layout, and the shape of it is the point:

    data/corpus/<study_id>/     the synced paper. An INPUT: fetched, never written by a run
      stage1/analyses.json          the coordinate-table parse
      stage1/table-map.json         manifest table_id -> record Table local_id
      processed/<flavour>/          the text every offset addresses
      source/<flavour>/             what the text was built from
    data/runs/<name>/           one extraction. Everything a run produces, together
      payloads/<study_id>/*.json    per-stage payloads, and noev/ before evidence
      records/<study_id>.extraction.json
      usage.jsonl                   what it cost
    data/vocab/                 fetched vocabularies. An input, shared by every run
    data/selection/             which papers to run at all: candidates, scores, pmids lists

A run is a directory rather than three trees keyed by study id, because the question asked of
these files is almost always "what did this run produce" and the old layout could only answer
"what does this paper have, from whichever run wrote last".

`benchmarks/` is deliberately not here: it is tracked in git as fixtures, and `data/` is not.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

#: The checkout. `parents[1]` from `pondie/paths.py`, and stated once so no other module
#: has to know how deep it sits.
REPO = Path(__file__).resolve().parents[1]

#: Bulk material: large, fetched, rebuildable, and gitignored. `PONDIE_DATA_DIR` moves the
#: whole tree, which is how a run reads a corpus that does not fit beside the checkout.
DATA = Path(os.environ.get("PONDIE_DATA_DIR") or REPO / "data").expanduser()

#: Synced papers, one directory per study. Read by every stage, written only by the sync.
CORPUS = DATA / "corpus"

#: Extraction runs, one directory each.
RUNS = DATA / "runs"

#: Fetched vocabularies: onvoc.json, cognitiveatlas-*.json, abbreviations.json, mondo.json.
VOCAB = DATA / "vocab"

#: Corpus selection: the candidate pool, the screen and adjudication scores, and the pmids
#: lists a run is pointed at. Separate from `CORPUS` because these describe *which* papers
#: rather than holding any, and they are written before a single paper is synced.
SELECTION = DATA / "selection"

#: Rebuildable disk caches -- embeddings, mostly. Never an input and never an output worth
#: keeping: deleting the whole directory costs time and nothing else.
CACHE = REPO / ".cache"


def run(name: str) -> Path:
    """The directory holding everything one extraction produced."""
    return RUNS / name


class Flavour(str, Enum):
    """Which render of a paper the text came from. The pipeline reads one per paper.

    Here rather than in `extraction.models`, because a render is a fact about the corpus
    layout and this module owns that. `models.Paper` re-exports it, so a caller that only
    wants to read a paper -- the normalizer, an audit, the query engine -- does not have to
    import the extraction package to name one.

    Declared best-first, by how much of a paper's tables survive the render, and the order
    is measured rather than conventional: over the 39,270-study corpus pubget ships a table
    manifest for 12,390 of its 13,313 papers and elsevier for all 10,595 of its own, while
    ace ships none. Taking ace when elsevier exists costs that paper its tables, and a
    locator searching a table-free flavour cannot find the sentence a group size came from.
    """

    local = "local"
    pubget = "pubget"
    elsevier = "elsevier"
    ace = "ace"

    @property
    def filename(self) -> str:
        """What the text is called under `processed/<flavour>/`.

        `local` is the only one that differs, and it differs because it is not a fetched
        render but a built one: `text.tables.txt` sits beside the `text.txt` it was built
        from so the two are never confused. Reading `text.txt` from the local directory
        finds nothing, which reads downstream as a paper with no text at all.
        """
        return "text.tables.txt" if self is Flavour.local else "text.txt"


# The accessors below are for code that READS the corpus. `extraction.corpus` builds it --
# it makes each directory and writes each file in turn -- so it necessarily names the parts,
# and asking it to reassemble them through here would be a fiction. Every other module goes
# through these: a reader that spells the layout itself is one that can silently read the
# wrong place, which is what `query.engine` did for as long as it looked under `texts/`.


def stage1(study: str, corpus: Path = CORPUS) -> Path:
    """The coordinate-table parse for one study. An input; no run writes it."""
    return corpus / study / "stage1" / "analyses.json"


def table_map(study: str, corpus: Path = CORPUS) -> Path:
    """Manifest `table_id` -> the `Table.local_id` an `Analysis.tables` reference holds."""
    return corpus / study / "stage1" / "table-map.json"


def text(study: str, flavour: Flavour, corpus: Path = CORPUS) -> Path:
    """The paper text for one flavour. THE document every `start_char` addresses.

    Takes the enum and not a string, so a typo is a `ValueError` at the call site rather
    than a real-looking path under a directory that will never exist -- which a caller
    reads as "this paper has no text", the exact ambiguity this module exists to prevent.
    """
    return corpus / study / "processed" / flavour.value / flavour.filename


def best_text(study: str, corpus: Path = CORPUS) -> Path:
    """The paper on the best flavour it actually has, by the ranking on `Flavour`."""
    for flavour in Flavour:
        candidate = text(study, flavour, corpus)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{study}: no text under {corpus / study}")


__all__ = [
    "REPO",
    "DATA",
    "CORPUS",
    "RUNS",
    "VOCAB",
    "SELECTION",
    "CACHE",
    "Flavour",
    "run",
    "stage1",
    "table_map",
    "text",
    "best_text",
]
