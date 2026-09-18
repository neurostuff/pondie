"""What every test file shares: the two schemas, and the example paper.

Twelve fixtures across twelve files loaded one of these two, under three names -- `sch`,
`classes` and `schema` -- and the names did not track the schemas. `sch` was the extraction
schema in five files and the storage schema in two, so the same identifier meant a different
document depending on which file you were reading. `schema` meant storage in one file and
extraction in another, and shadowed the `pondie.schema` module in both.

The extraction schema was reached by six spellings that all resolve to one path:
`schema.EXTRACTION`, `render.EXTRACTION_SCHEMA`, `builder.EXTRACTION_SCHEMA`,
`validate.EXTRACTION_SCHEMA`, `scoring.SCHEMA` and `pondie.schema.EXTRACTION`. Which one a
test used recorded nothing except which module its author had open.

Session-scoped, not module-scoped, because there is one document per path for the whole run.
That is not a speed claim -- `reader._load` is `lru_cache`d, so twelve module-scoped fixtures
already cost one parse each -- it is that a fixture rebuilt per module invites a test to
mutate it, and `Schema.attributes` returns copies precisely so that cannot happen.

The paper harness below is the other half, and it is most of why the suite had a 2,559-line
file. A record and the text its offsets address are one artefact; finding a pair that agrees
is a dozen lines of hash checking; and every test that reads a real record needs it. So the
tests that needed it all lived in the file that happened to define it. From here they can
live where they belong.

`requires_paper` and `requires_current_record` stay marks rather than becoming fixtures
because they decide at collection time. A test file takes them by name:
`from conftest import requires_paper`.
"""

from __future__ import annotations

import pytest

import json
from pathlib import Path

from pondie import paths, schema
from pondie.extraction.record import validate as validate_record
from pondie.formats import text_index
from pondie.schema import reader


@pytest.fixture(scope="session")
def extraction_schema():
    """What a model is asked for: the storage schema projected down to the askable slots."""
    return reader.load(schema.EXTRACTION)


@pytest.fixture(scope="session")
def storage_schema():
    """What a record is stored as, and what the validator holds a built record to."""
    return reader.load(schema.STORAGE)


#: This suite's own fixtures, and the bulk corpus `sync_texts` writes. `data/corpus` is
#: gitignored: which papers a checkout has is a property of that checkout, not of the repo.
REPO = paths.REPO
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEXTS = paths.CORPUS

#: The paper these tests run against: a record and the text it was extracted from.
#:
#: Both ship. `tests/fixtures/paper/` carries the text -- CC-BY, 27 KB -- and the record is
#: `benchmarks/gold/xevP8UDRAVh9.extraction.json`, which was already here. They are one
#: artefact and had been split across a checkout: the examples under `fixtures/examples/`
#: were built by `review-bootstrap-0.1.0` against a `text.tables.txt` that no longer exists
#: anywhere because a later pubget commit changed how tables are inlined. Seventeen tests
#: over spans, offsets and the section index could not run, and had not for a long time --
#: the skip said "sync the corpus", and syncing it did not help, because the text that comes
#: back is a different build.
#:
#: A synced corpus is still used if it happens to hold a matching pair, so a checkout with
#: real data exercises these on its own papers too. The hash is what decides: a text that is
#: merely *present* is not the text a record addresses, and gating on presence alone turned
#: five passing skips into five failures about a hash mismatch, which is not what any of
#: them tests.


def _pairs_with(record_path: Path, text_path: Path) -> bool:
    """Whether this text is THE text this record's offsets address."""
    if not (record_path.is_file() and text_path.is_file()):
        return False
    declared = (
        json.loads(record_path.read_text(encoding="utf-8"))
        .get("extraction_metadata", {})
        .get("source_text_hash")
    )
    digest = text_index.text_hash(text_index.normalize(text_path.read_text(encoding="utf-8")))
    return bool(declared) and declared == digest


def _example_paper() -> tuple[str, Path, Path]:
    """(paper, record, text) for the first pair that agrees, shipped or synced."""
    shipped = FIXTURES / "paper"
    for text in sorted(shipped.glob("*.text.tables.txt")):
        paper = text.name.removesuffix(".text.tables.txt")
        for record in (
            REPO / "benchmarks" / "gold" / f"{paper}.extraction.json",
            FIXTURES / "examples" / f"{paper}.extraction.json",
        ):
            if _pairs_with(record, text):
                return paper, record, text

    for record in sorted((FIXTURES / "examples").glob("*.extraction.json")):
        paper = record.name.removesuffix(".extraction.json")
        text = TEXTS / paper / "processed" / "local" / "text.tables.txt"
        if _pairs_with(record, text):
            return paper, record, text
    return "", Path(), Path()


PAPER, RECORD, TEXT = _example_paper()
PAYLOADS = FIXTURES / "payloads" / PAPER
IDENTIFIERS = FIXTURES / "paper" / f"{PAPER}.identifiers.json"
if not IDENTIFIERS.is_file():
    IDENTIFIERS = TEXTS / PAPER / "identifiers.json"

requires_paper = pytest.mark.skipif(
    not PAPER,
    reason=(
        "no record pairs with a text: neither the shipped fixture pair nor any synced "
        "paper has a text whose hash matches a record's `source_text_hash`"
    ),
)


def _schema_drift() -> list[str]:
    """Slots the example record carries that the schema no longer declares, at any depth.

    The extraction schema became a projection of the storage schema, which moved several
    things -- Study.terms became ModelTerm under ModelEstimation, arms and timepoints moved
    under Study.design, the per-method Analysis payloads collapsed into Analysis.details.
    Rather than migrate a record by hand and call the result extracted, the tests that read
    it skip until a fresh extraction replaces it, and light up on their own when one does.

    At any depth, which is the part this missed. It compared the record's TOP-LEVEL keys
    against `Study`, so a slot renamed three levels down was invisible: the shipped gold
    still carries `InferenceSettings.voxelwise_threshold_value` and two siblings, renamed to
    `height_threshold_*`, and the gate reported the record as current while the validator
    reported six errors.
    """

    if not RECORD.is_file():
        return ["no record"]
    validator = validate_record.Validator(reader.load(schema.EXTRACTION), None)
    validator.check_record(json.loads(RECORD.read_text(encoding="utf-8")))
    return [error for error in validator.errors if "is not declared on" in error]


_DRIFT = _schema_drift()

requires_current_record = pytest.mark.skipif(
    bool(_DRIFT),
    reason=(
        f"{PAPER}'s record predates the schema now in the tree -- "
        + "; ".join(sorted({d.split(": ", 1)[-1] for d in _DRIFT})[:2])
        + ". Re-extract the paper to re-enable"
        if PAPER
        else "no example paper to check against the schema"
    ),
)


@pytest.fixture(scope="module")
def normalized() -> str:
    return text_index.normalize(TEXT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def enums() -> dict:
    """The vocabularies, which `Validator` now takes from the schema by default.

    A validator built without them silently checks no vocabulary at all -- neither the
    closed ones it should reject on nor the open ones it should warn on -- so a test of
    either has to pass this.
    """

    return reader.load(schema.EXTRACTION).enums
