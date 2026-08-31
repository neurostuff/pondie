"""The extraction passes: prompt construction, assembly, repair, evidence, validation.

These are the working implementation, moved here from the schema repository because they are
extraction logic and a schema repository should hold a schema. ~7,600 lines, carried across
unedited so the move is a move and not a rewrite -- the benchmark in `benchmarks/` is what
would catch a rewrite going wrong, and it should be green before and after.

They import each other by bare name, which is why this package puts its own directory on the
path. That is a transitional shim, not a design: converting them to relative imports is a
mechanical change worth doing on its own, where a diff of it is readable.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE / "pipeline"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pondie import _schema  # noqa: E402,F401 -- absolute: a relative import re-enters
#: `pondie.extraction`, which is still initialising when this package is first imported.
