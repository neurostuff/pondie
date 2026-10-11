"""Reading ingestion's CoordinateParse as the document every parse reader here takes.

`<study>/parse/coordinate_parse.json` is study_schema's contract and the one producer of a
paper's analyses. Each analysis carries its own `key`, derived from where it was read, so a
re-run, a reordering or a dropped sibling leaves it alone. `stage1/analyses.json` keys its
entries by position (`t1#2`), which is why it is read only for a paper that has no parse.

The parse is converted to the shape `stage1/analyses.json` has, with the key added to each
entry, rather than every reader learning a second shape: the listing, the record repairs,
the query join and the benchmark all go on zipping `parse_keys.parse_keys(entries)` with
the entries, and that function returns the parse's key wherever an entry has one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from study_schema.keys import normalize_name
from study_schema.models.paper_parse import CoordinateParse, ParsedPaper

from pondie.formats import parse_keys

_log = logging.getLogger(__name__)

#: The file names, as `study_schema.layouts.PaperParse` declares them.
PARSE_DIR = "parse"
COORDINATE_PARSE = "coordinate_parse.json"
PARSED_PAPER = "parsed_paper.json"

#: What `Paper.parse_source` reports for each path.
FROM_PARSE = "coordinate_parse"
FROM_STAGE1 = "stage1"


def is_coordinate_parse(path: Path | None) -> bool:
    return bool(path) and path.name == COORDINATE_PARSE and path.parent.name == PARSE_DIR


def legacy_stage1(path: Path) -> Path:
    """The `stage1/analyses.json` beside a study's `parse/coordinate_parse.json`."""
    return path.parent.parent / "stage1" / "analyses.json"


def read_document(path: Path) -> dict[str, Any]:
    """The parse at `path` as a stage1-shaped document, whichever of the two files it is.

    Raises as `json.loads` and pydantic do: a coordinate parse that fails study_schema's
    model is a broken contract, not a paper with no analyses.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not is_coordinate_parse(path):
        return raw
    parse = CoordinateParse.model_validate(raw)
    paper_path = path.parent / PARSED_PAPER
    paper = (
        ParsedPaper.model_validate(json.loads(paper_path.read_text(encoding="utf-8")))
        if paper_path.is_file()
        else None
    )
    if paper is None:
        _log.warning("%s: no %s beside it; table labels and captions unknown", path, PARSED_PAPER)
    return to_document(parse, paper)


def to_document(parse: CoordinateParse, paper: ParsedPaper | None = None) -> dict[str, Any]:
    tables = {t.table_id: t for t in (paper.tables or [])} if paper else {}
    names = {a.key: a.name for a in parse.analyses}
    entries = []
    for analysis in parse.analyses:
        is_table = analysis.origin == "table"
        table_id = analysis.table_id if is_table else parse_keys.PROSE_TABLE_ID
        table = tables.get(table_id) if is_table else None
        entry: dict[str, Any] = {
            "key": analysis.key,
            "source": analysis.origin,
            "name": analysis.name,
            "description": analysis.description,
            "role": analysis.role,
            "role_source": analysis.role_source,
            "anchor_kind": analysis.anchor_kind,
            "from_prior_study": analysis.from_prior_study,
            "table_id": table_id,
            "table_number": table.number if table else None,
            "table_label": table.label if table else None,
            "table_caption": (table.caption if table else None) or "",
            "table_footer": (table.footer if table else None) or "",
            "coordinate_space": analysis.coordinate_space,
            "statistic": analysis.statistic,
            "points": [
                {
                    "coordinates": list(point.coordinates),
                    # A point states its space only where it differs from the analysis's,
                    # and the readers here look for it on the point.
                    "space": point.space or analysis.coordinate_space,
                    "values": [v.model_dump(exclude_none=True) for v in point.values or []],
                    "cluster_size": point.cluster_size,
                    "is_subpeak": point.is_subpeak,
                    "label": point.label,
                }
                for point in analysis.points or []
            ],
        }
        # The parse declares a sign split on both halves; pondie's readers know the inverse
        # half as the entry marked `withhold`, rebuilt from the described one it mirrors.
        split = analysis.split
        if split is not None and split.half == "inverse":
            entry["withhold"] = True
            entry["mirror_of"] = names.get(split.original_analysis or "")
        entries.append(entry)
    return {
        "analyses": entries,
        "parse_id": parse.parse_id,
        "source": FROM_PARSE,
        # The parse made its own split; `SignSplit` has nothing to do to it.
        "sign_split_applied": True,
    }


def _signature(entry: Mapping[str, Any]) -> tuple:
    return tuple(sorted(tuple(p.get("coordinates") or ()) for p in entry.get("points") or []))


def legacy_key_map(
    stage1_entries: list[dict[str, Any]], parse_entries: Iterable[Mapping[str, Any]]
) -> dict[str, str]:
    """Positional stage-1 key -> the parse's key for the same analysis, where it is certain.

    For a record written against stage 1 (`tbl0003#2`) read once the paper has a parse. The
    same analysis is the entry under the same table with the same normalized name; where
    the name recurs under one table, its points decide, and where they do not decide the
    key is left unmapped rather than guessed.
    """
    candidates: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for entry in parse_entries:
        slot = (str(entry.get("table_id") or ""), normalize_name(str(entry.get("name") or "")))
        candidates.setdefault(slot, []).append(entry)
    mapping: dict[str, str] = {}
    for old, entry in zip(parse_keys.positional_keys(stage1_entries), stage1_entries):
        slot = (str(entry.get("table_id") or ""), normalize_name(str(entry.get("name") or "")))
        hits = candidates.get(slot, [])
        if len(hits) > 1:
            hits = [h for h in hits if _signature(h) == _signature(entry)]
        if len(hits) == 1:
            mapping[old] = str(hits[0]["key"])
    return mapping
