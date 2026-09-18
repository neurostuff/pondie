"""The stage-1 parse: every analysis read off a paper's coordinate tables.

Stage 1 is an input to extraction rather than a step of it -- `parse_tables` produces it with
one model call per table, and the pipeline only reads and annotates it. It is a document
rather than a list because one fact about the whole parse has to travel with the entries:
`sign_split_applied` distinguishes a parse the sign rule found nothing to do in from one
written before that rule existed, and only the first should be left alone.

Every way of reading a parse is here. `coordinates` and `source_tables` were private
helpers in `stages.py`, reached by whichever stage needed them first, so two of the three
readers of this document lived in the file that runs the pipeline. A caller asking what a
parse holds should find the answer in the module named for it.

Nothing here calls a model or writes a record, so a parse can be built in a test without a
paper on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pondie.formats import parse_keys


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
    def is_prose(self) -> bool:
        """Read from a sentence rather than a table, so there is no table to point at."""
        return self.table_id == parse_keys.PROSE_TABLE_ID

    @property
    def coordinates(self) -> list[tuple[float, float, float]]:
        """The xyz triples this entry reports, in either shape the parse writes them.

        Older parses hold `{"x": .., "y": .., "z": ..}` and newer ones a bare triple, and a
        caller wanting the numbers should not have to know which. Unparseable points are
        skipped rather than raising: the parse is an input this pipeline does not write.
        """
        out: list[tuple[float, float, float]] = []
        for point in self.points:
            coords = point.get("coordinates")
            if isinstance(coords, Mapping):
                coords = [coords.get("x"), coords.get("y"), coords.get("z")]
            if isinstance(coords, (list, tuple)) and len(coords) == 3:
                try:
                    x, y, z = (float(v) for v in coords)
                except (TypeError, ValueError):
                    continue
                out.append((x, y, z))
        return out

    @property
    def source_table(self) -> dict[str, Any]:
        """The table this entry was read off, in the shape the manifest reports one.

        `corpus.tables` stamps every parsed analysis with its table, so the parse carries
        the same fields `formats.table_parse.read_manifest` does -- which is what lets the
        `tables` stage take either source without knowing which it got.
        """
        return {
            "table_id": self.table_id,
            "table_number": self.raw.get("table_number"),
            "table_label": self.raw.get("table_label"),
            "caption": self.raw.get("table_caption"),
            "footer": self.raw.get("table_footer"),
        }

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

    @classmethod
    def read(cls, path: Path) -> "TableParse":
        """The parse at `path`, or an empty one where there is nothing readable.

        For the callers that consult the parse as a fallback and have somewhere else to
        go. `load` raises, which is right for the stages whose whole job is the parse.
        """
        try:
            return cls.load(path)
        except (OSError, json.JSONDecodeError):
            return cls(path, {})

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

    @property
    def coordinates(self) -> list[tuple[float, float, float]]:
        """Every coordinate the parse already holds, so the prose pass does not repeat one."""
        return [xyz for analysis in self.analyses for xyz in analysis.coordinates]

    def source_tables(self) -> list[dict[str, Any]]:
        """One entry per table the parse read, first mention winning.

        The prose pseudo-table is excluded: it is not a table, and the prompt renders those
        entries under a heading telling the model to omit `tables`.
        """
        out: dict[str, dict[str, Any]] = {}
        for analysis in self.analyses:
            if analysis.table_id and not analysis.is_prose:
                out.setdefault(analysis.table_id, analysis.source_table)
        return list(out.values())
