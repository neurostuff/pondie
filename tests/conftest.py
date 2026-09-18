"""The two schemas, under names that say which one they are.

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
"""

from __future__ import annotations

import pytest

from pondie import schema
from pondie.schema import reader


@pytest.fixture(scope="session")
def extraction_schema():
    """What a model is asked for: the storage schema projected down to the askable slots."""
    return reader.load(schema.EXTRACTION)


@pytest.fixture(scope="session")
def storage_schema():
    """What a record is stored as, and what the validator holds a built record to."""
    return reader.load(schema.STORAGE)
