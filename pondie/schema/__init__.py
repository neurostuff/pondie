"""Where the schema is, and the code that reads it.

The schema itself -- the LinkML YAML and the prose the model is shown -- is authored in the
`study-schema` repository and carried here as a git submodule. It is data: pondie owns every
line of Python that reads it, so there is one implementation of "what a record is" and it is
in one place.

    pondie.schema.reader     READ the schema: classes, slots, ranges, what a slot means
    pondie.schema.generate   the extraction schema, projected from the storage schema
    pondie.schema.checks     what has to hold for the two schemas to agree
    pondie.schema.authoring  read the YAML as YAML, for the two above

`reader` is the one every consumer wants. It answers through LinkML's own `SchemaView`
rather than by walking the documents, so inheritance, `slot_usage` and `default_range` are
the language's answers and not ours. `authoring` is the narrow remainder: questions about
the marks on the source files, which only the generator and the checks ask, and which a
consumer of a record has no reason to.

`ROOT` is resolved rather than imported, because the schema is no longer a Python
distribution to anchor on. `PONDIE_SCHEMA_DIR` wins when set -- that is how a caller points at
a different checkout -- and the submodule beside this package is the default. Neither is
trusted: a directory without the storage schema in it is reported here, by name, rather than
as a missing-file error from whichever module happened to read first.
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    override = os.environ.get("PONDIE_SCHEMA_DIR")
    candidate = (
        Path(override).expanduser().resolve()
        if override
        else Path(__file__).resolve().parents[2] / "study_schema"
    )
    if not (candidate / "neuroimaging-study-storage.yaml").is_file():
        remedy = (
            "point PONDIE_SCHEMA_DIR at a study-schema checkout"
            if override
            else "run `git submodule update --init`, or set PONDIE_SCHEMA_DIR to a checkout"
        )
        source = "PONDIE_SCHEMA_DIR" if override else "the study_schema submodule"
        raise RuntimeError(f"no schema at {candidate} (from {source}); {remedy}")
    return candidate


#: The study-schema checkout: LinkML YAML, and the prose sent to the model as prompt input.
ROOT = _root()

EXTRACTION = ROOT / "neuroimaging-study-extraction.yaml"
STORAGE = ROOT / "neuroimaging-study-storage.yaml"

__all__ = ["ROOT", "EXTRACTION", "STORAGE"]
