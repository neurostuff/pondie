"""Make the schema submodule importable, once, from anywhere.

`study_schema` is a git submodule rather than a package on the path, and its `review` modules
import each other by bare name. Rather than have every caller and every test insert paths, the
setup happens here and importing this module is the whole interface.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "study_schema"
REVIEW = ROOT / "review"


def ensure() -> Path:
    """Put the schema on the path if it is not already there. Idempotent."""
    for path in (ROOT, REVIEW):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not (REVIEW / "build_record.py").is_file():
        raise ModuleNotFoundError(
            f"{ROOT} is empty. It is a submodule: "
            "git submodule update --init --recursive")
    return ROOT


ensure()
