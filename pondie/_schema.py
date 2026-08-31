"""Make the schema submodule importable, once, from anywhere.

`study_schema` is a git submodule rather than a package on the path, and the three modules
it carries -- `schema_utils`, `text_index`, `table_parse` -- are imported by bare name.
Rather than have every caller and every test insert paths, the setup happens here and
importing this module is the whole interface.

Those three are in the schema repository because they *define* the schema rather than merely
read it: `ExtractionMetadata.source_text_hash` is the sha256 `text_index` computes, and a
coordinate table is whatever `table_parse` says its rows are. A second copy would be a second
definition. Everything that acts on a record -- extraction, normalization, the benchmark --
is here instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "study_schema"


def ensure() -> Path:
    """Put the schema on the path if it is not already there. Idempotent."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not (ROOT / "schema_utils.py").is_file():
        raise ModuleNotFoundError(
            f"{ROOT} is empty. It is a submodule: "
            "git submodule update --init --recursive")
    return ROOT


ensure()
