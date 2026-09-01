"""The stage-1 parse: every analysis read off a paper's coordinate tables.

Stage 1 is an input to extraction rather than a step of it -- `parse_tables` produces it with
one model call per table, and the pipeline only reads and annotates it. It is a document
rather than a list because one fact about the whole parse has to travel with the entries:
`sign_split_applied` distinguishes a parse the sign rule found nothing to do in from one
written before that rule existed, and only the first should be left alone.

Nothing here calls a model or writes a record, so a parse can be built in a test without a
paper on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ParsedAnalysis:
    """One entry from the coordinate-table parse, before any model has seen it.

    The sign split lives here rather than in a loose dict because it is the one place the
    pipeline deliberately hides work from the model: a table reporting both signs is two
    contrasts, the paper's prose describes one of them, and the other is rebuilt by
    arithmetic. `is_withheld` and `mirror_of` are what make that visible to a reader
    instead of implied by the presence of a key.
    """

    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.raw.get("name") or "")

    @property
    def table_id(self) -> str:
        return str(self.raw.get("table_id") or "")

    @property
    def points(self) -> list[dict[str, Any]]:
        return self.raw.get("points") or self.raw.get("coordinates") or []

    @property
    def is_withheld(self) -> bool:
        """Kept out of the extraction prompt because the paper does not describe it."""
        return bool(self.raw.get("withhold"))

    @property
    def mirror_of(self) -> str | None:
        return self.raw.get("mirror_of")

    @property
    def split_direction(self) -> str | None:
        return self.raw.get("split_direction")

    def __repr__(self) -> str:
        mark = " [withheld]" if self.is_withheld else ""
        return f"<ParsedAnalysis {self.name!r} {len(self.points)} point(s){mark}>"


@dataclass
class TableParse:
    """Every analysis parsed from one paper's coordinate tables.

    Loaded and saved as one document so the sign-split flag lives with the analyses it
    describes: a file partitioned before that rule existed is distinguishable from one
    the rule found nothing to do in, and only the second should be left alone.
    """

    path: Path
    document: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "TableParse":
        return cls(path, json.loads(path.read_text(encoding="utf-8")))

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @property
    def analyses(self) -> list[ParsedAnalysis]:
        return [ParsedAnalysis(entry) for entry in self.document.get("analyses") or []]

    @property
    def sign_split_applied(self) -> bool:
        return bool(self.document.get("sign_split_applied"))

    def described(self) -> list[ParsedAnalysis]:
        """The analyses the extraction pass is allowed to see."""
        return [a for a in self.analyses if not a.is_withheld]

    def withheld(self) -> list[ParsedAnalysis]:
        """The reversed halves, to be rebuilt from the record after extraction."""
        return [a for a in self.analyses if a.is_withheld]

    def replace_analyses(self, entries: list[dict[str, Any]]) -> None:
        self.document["analyses"] = entries
        self.document["sign_split_applied"] = True
