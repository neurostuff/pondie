"""Stage 1: split each coordinate table into the analyses it reports.

The extraction passes never see the table rows -- the normalized paper text carries
captions but not cell values -- so this is the only place the reported effects are
enumerated. Everything downstream annotates the list this produces, which makes a
stage-1 regression a stage-2 outage rather than a degradation.

The parse itself is Autonima's `parse_single_table`: one LLM call per table. Nothing
about the splitting rules lives here, so the prompt version travels with autonima and is
recorded in the output.

This module handles transport. Autonima constrains output with legacy function calling,
which the gateway rejects for a reasoning model:

    Function tools with reasoning_effort are not supported for gpt-5.6-luna in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

`_StructuredCoordinateClient` therefore sends the same Pydantic schema as a strict
`response_format`, which the endpoint accepts with `reasoning_effort`. Autonima remains
unchanged because its other callers use models that support function calling. Importing
the schema and sanitizer also keeps this client aligned with `parse_single_table`.

The pond corpus already holds a parse of these same tables under
`processed/pubget/analyses.jsonl`. It is not used as input -- it is diffed against, so
a change in the upstream prompt is visible rather than assumed.

    python -m pondie.extraction.corpus.tables --pmids bench-baseline.pmids \
        --autonima .tmp_repos/autonima --key-file .env
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
from pondie import paths
from pondie.extraction.corpus.sync import read_pmids  # noqa: E402
from pondie.extraction.llm import load_env
from pondie.extraction.sign_split import split_opposite_signs  # noqa: E402
from pondie.formats.table_parse import READERS, read_manifest  # noqa: E402

DEFAULT_MODEL = "@psyc-aid338-ope-333f18/gpt-5.6-luna"

#: Matches the extraction passes. The baseline run measured 220-410 reasoning tokens per
#: call at this setting and found nothing in the error profile that looked like a
#: reasoning shortfall, so the tables get the same budget the prose does.
DEFAULT_EFFORT = "low"

#: The only parsed value kind that cannot carry a sign. Everything else -- `t-statistic`,
#: `z-statistic`, `correlation`, `beta`, and the `other` catch-all -- is a quantity whose
#: sign means something when the table prints one.
#:
#: `other` is included deliberately. It holds statistic-like values the parser could not
#: label (one study contributes 124 of them in the 0.61-3.75 range, which is a t or z that
#: lost its heading), so excluding it would discard real directions. No kind is judged
#: non-directional because this corpus happens to show no negatives for it: most tables
#: print |t|, so an all-positive column is evidence about the table's conventions and not
#: about the quantity.
def strict_schema(model_class) -> dict:
    """A Pydantic model's JSON schema, tightened until the API will accept it as strict.

    Structured outputs are stricter than function parameters were: every property has to
    be listed in `required` and every object has to forbid extra keys. Pydantic omits a
    field from `required` as soon as it has a default, which is most of this schema, so
    the fields are made *nullable and required* rather than optional -- the same shape
    `Optional[X] = None` already meant, said in the way the API demands.
    """

    def tighten(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                tighten(item)
            return
        if not isinstance(node, dict):
            return

        # A default is advisory and strict mode rejects it outright.
        node.pop("default", None)

        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            properties = node["properties"]
            for name, sub in properties.items():
                # Require the key but allow null, so callers must explicitly represent a
                # missing value.
                if "$ref" not in sub and "anyOf" not in sub and "type" in sub:
                    if sub["type"] != "null" and name not in node.get("required", []):
                        sub["type"] = [sub["type"], "null"]
            node["required"] = list(properties)

        for value in node.values():
            tighten(value)

    schema = model_class.model_json_schema()
    tighten(schema)
    return schema


def build_client(effort: str):
    """An autonima coordinate client that speaks structured outputs instead of functions.

    Subclasses autonima's own client rather than reimplementing it: the api-key and
    `OPENAI_API_GATEWAY` base-url handling is already there and is not worth a second
    copy. Only the one call is replaced.
    """

    from autonima.coordinates.openai_client import (  # noqa: PLC0415
        CoordinateParsingClient,
        _sanitize_parse_result,
    )
    from autonima.coordinates.schema import ParseAnalysesOutput  # noqa: PLC0415

    class _StructuredCoordinateClient(CoordinateParsingClient):
        def parse_analyses(self, prompt: str, model: str = DEFAULT_MODEL):
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that parses neuroimaging results "
                            "tables into structured JSON for downstream analysis."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "parse_analyses",
                        "strict": True,
                        "schema": strict_schema(ParseAnalysesOutput),
                    },
                },
            }
            if effort:
                kwargs["reasoning_effort"] = effort

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                # A refusal or a length stop arrives as an empty body; the caller counts
                # it as a failed table rather than an empty one, which is the difference
                # between "this table reports nothing" and "this table was not read".
                raise ValueError(
                    f"empty response from {model} "
                    f"(finish_reason={response.choices[0].finish_reason})"
                )
            return ParseAnalysesOutput(**_sanitize_parse_result(json.loads(content)))

    return _StructuredCoordinateClient()




def table_flavour(study_dir: Path) -> str | None:
    """The best flavour this study has a table manifest for, or None if it has none.

    `paths.Flavour` order, minus ace, which ships no manifest. Stage 1 read pubget and only
    pubget, so a study whose tables come from elsevier -- 51 of the 100 in
    `data/selection/old-extraction-defects.pmids` -- raised FileNotFoundError on a corpus
    the sync had fetched correctly.
    """

    for flavour in ("pubget", "elsevier"):
        if (study_dir / "processed" / flavour / "tables.jsonl").is_file():
            return flavour
    return None


def _csv_text(rows: list[list[str]]) -> str:
    """Parsed rows as CSV, so every flavour reaches the parser as the same kind of text."""

    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


def _rows_of(path: Path) -> list[list[str]]:
    """The grid behind one raw table, by what the file actually is.

    `.xml` is two different formats: elsevier writes CALS (`<tgroup>`), pubget writes the
    article's own JATS `<table>`. Trying CALS first and falling back on an empty read tells
    them apart without a second manifest field to trust.
    """

    if path.suffix.lower() in {".html", ".htm"}:
        return READERS["ace"](path)[0]
    rows = READERS["elsevier"](path)[0]
    return rows or READERS["ace"](path)[0]


def _fallback_path(tables_dir: Path, table_id: str, named: str) -> Path | None:
    """A raw table under another name, when the manifest names one that was never written.

    pubget's manifest says `table_001.csv` for four of the hundred studies here and wrote
    `t2.xml` instead -- the article's own table markup, no CSV render. The manifest is not
    wrong about which tables hold coordinates, only about where the bytes are, so the id is
    the thing to search on.
    """

    stem = Path(named).stem
    for candidate in (table_id, stem):
        for suffix in (".csv", ".xml", ".html"):
            path = tables_dir / f"{candidate}{suffix}"
            if path.is_file():
                return path
    return None


def coordinate_tables(study_dir: Path, flavour: str | None = None) -> list[dict]:
    """The tables the manifest found coordinates in, with the CSV text to parse.

    `contains_coordinates` is the flavour's own determination, the same filter the
    corpus used; parsing the demographics tables would spend calls to produce
    analyses with no points.

    pubget's raw table is already CSV and is passed through byte for byte, so its parse is
    what it always was. elsevier's is CALS XML, and `formats.table_parse` already reads it
    -- the grid goes to CSV here rather than to markdown so the prompt sees the shape it
    was written against.
    """

    flavour = flavour or table_flavour(study_dir)
    if flavour is None:
        return []
    manifest = read_manifest(study_dir, flavour)
    tables_dir = study_dir / "source" / flavour / "tables"
    out = []
    for table in manifest.values():
        if not table["contains_coordinates"]:
            continue
        name = table["data_file"]
        raw_path = tables_dir / name if name else None
        if raw_path is None or not raw_path.is_file():
            raw_path = _fallback_path(tables_dir, table["table_id"], name)
        if raw_path is None:
            print(
                f"    WARNING: no {flavour} table for {table['table_id']} "
                f"({name or 'no data path'})",
                file=sys.stderr,
            )
            continue
        if flavour == "pubget" and raw_path.suffix.lower() == ".csv":
            # Byte for byte what stage 1 has always sent, so the 49 pubget studies here
            # parse exactly as they would have before elsevier was supported.
            text = raw_path.read_text(encoding="utf-8")
        else:
            rows = _rows_of(raw_path)
            if not rows:
                print(
                    f"    WARNING: {flavour} table {table['table_id']} "
                    f"({raw_path.name}) read as empty",
                    file=sys.stderr,
                )
                continue
            text = _csv_text(rows)
        out.append(
            {
                "table_id": table["table_id"],
                "table_number": table["table_number"],
                "table_label": table["table_label"],
                "caption": table["caption"],
                "footer": table["footer"],
                "csv_path": raw_path,
                "csv_text": text,
            }
        )
    return out


def pond_analyses(study_dir: Path) -> list[dict]:
    """The corpus's own parse, for comparison only."""

    path = study_dir / "processed" / "pubget" / "analyses.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def diff_report(study: str, fresh: list[dict], pond: list[dict]) -> str:
    """Names and point counts, ours versus the corpus's."""

    def key(analysis):
        return (analysis.get("name") or "").strip()

    fresh_names = [key(a) for a in fresh]
    pond_names = [key(a) for a in pond]
    lines = [
        f"# Stage 1 re-parse vs pond — {study}",
        "",
        f"- fresh: **{len(fresh)}** analyses, {sum(len(a.get('points') or []) for a in fresh)} points",
        f"- pond:  **{len(pond)}** analyses, {sum(len(a.get('coordinates') or []) for a in pond)} points",
        "",
        "| # | fresh | pond |",
        "|---|---|---|",
    ]
    for index in range(max(len(fresh_names), len(pond_names))):
        left = fresh_names[index] if index < len(fresh_names) else "—"
        right = pond_names[index] if index < len(pond_names) else "—"
        flag = "" if left == right else "  ⚠"
        lines.append(f"| {index + 1} | {left}{flag} | {right} |")
    only_fresh = sorted(set(fresh_names) - set(pond_names))
    only_pond = sorted(set(pond_names) - set(fresh_names))
    if only_fresh or only_pond:
        lines += ["", "## Set difference", ""]
        for name in only_fresh:
            lines.append(f"- fresh only: `{name}`")
        for name in only_pond:
            lines.append(f"- pond only: `{name}`")
    return "\n".join(lines) + "\n"


def resplit(pmids: Path, texts: Path) -> int:
    """Re-partition stage-1 output already on disk, without re-parsing the tables.

    The split reads only the parsed statistics, so a corpus parsed before this rule existed
    does not have to be re-parsed to get it -- which matters because re-parsing is a model
    call per table and would resample every other decision the parse makes at the same time.
    Already-split entries are left alone: `split_rule` marks them, and the second pass over
    a part that holds one direction finds one sign and does nothing.
    """

    changed = 0
    for pmid, study, _axis in read_pmids(pmids):
        path = paths.stage1(study, texts)
        if not path.is_file():
            print(f"{study}: no stage1/analyses.json", file=sys.stderr)
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        before = doc.get("analyses") or []
        split = split_opposite_signs(before)
        after, notes = list(split.analyses), list(split.notes)
        if not notes:
            print(f"{study}: unchanged ({len(before)} analyses)")
            continue
        for note in notes:
            print(f"{study}: {note}")
        if len(after) != len(before):
            doc["analyses"] = after
            #: Recorded on the document rather than inferred from the parts, so a reader can
            #: tell a file the rule has been applied to from one parsed before it existed.
            doc["sign_split_applied"] = True
            path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"{study}: {len(before)} -> {len(after)} analyses, rewrote {path}")
            changed += 1
    print(f"\n{changed} study file(s) rewritten")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmids", type=Path, default=REPO / "bench-baseline.pmids")
    parser.add_argument("--texts", type=Path, default=paths.CORPUS)
    parser.add_argument("--autonima", type=Path, default=REPO / ".tmp_repos" / "autonima")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        help="reasoning effort; empty string to send none at all",
    )
    parser.add_argument("--key-file", type=Path, default=REPO / ".env")
    parser.add_argument("--dry-run", action="store_true", help="list tables, make no calls")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="tables parsed at once. One table is one HTTP round trip, so this is the "
             "gateway's concurrency, not the machine's",
    )
    parser.add_argument(
        "--redo", action="store_true",
        help="re-parse studies that already have stage1/analyses.json; without it a "
             "restart resumes rather than paying for the tables already done",
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="apply the sign split to the stage1/analyses.json already on "
        "disk and rewrite it; the partition is arithmetic, so this "
        "needs no model call and no autonima checkout",
    )
    args = parser.parse_args()

    if args.resplit:
        return resplit(args.pmids, args.texts)

    # Not import wiring: autonima is a separate checkout the caller points at, with no
    # distribution to depend on. Checked first so a wrong --autonima says so here rather
    # than as an ImportError three lines down.
    checkout = args.autonima.resolve()
    if not (checkout / "autonima").is_dir():
        raise SystemExit(f"no autonima checkout at {checkout}; point --autonima at one")
    sys.path.insert(0, str(checkout))
    from autonima.coordinates.parser import parse_single_table
    from autonima.coordinates.prompts import COORDINATE_PARSING_PROMPT_VERSION

    client = None
    if not args.dry_run:
        if args.key_file and args.key_file.is_file():
            load_env(args.key_file)
        if not os.environ.get("OPENAI_API_KEY"):
            print("no OPENAI_API_KEY; pass --key-file", file=sys.stderr)
            return 2
        client = build_client(args.effort)

    print(
        f"autonima prompt version {COORDINATE_PARSING_PROMPT_VERSION}, "
        f"model {args.model}, effort {args.effort or 'unset'}\n"
    )

    failures = 0
    work: list[tuple[str, str, Path, dict]] = []
    for pmid, study, _axis in read_pmids(args.pmids):
        study_dir = args.texts / study
        if not args.redo and (study_dir / "stage1" / "analyses.json").is_file():
            print(f"{study} (pmid {pmid}): already parsed, skipping")
            continue
        tables = coordinate_tables(study_dir)
        print(f"{study} (pmid {pmid}): {len(tables)} coordinate tables")
        for table in tables:
            if args.dry_run:
                print(f"  {table['table_id']}: {len(table['csv_text']):,} ch (dry run)")
            else:
                work.append((pmid, study, study_dir, table))

    if args.dry_run:
        return 0

    def parse_one(item):
        """One table -> its analyses. A table is a whole HTTP round trip, so the pool is
        threads and the width is the gateway's concurrency rather than the CPU's."""
        _pmid, study, _study_dir, table = item
        try:
            result = parse_single_table(
                table["table_id"], table["caption"], table["footer"],
                table["csv_text"], client, args.model,
            )
            parsed = result["parsed_json"].get("analyses") or []
        except Exception as exc:  # one table must not sink the study
            return study, table, None, f"{type(exc).__name__}: {exc}"[:200]
        for analysis in parsed:
            # Table identity is what disambiguates repeated analysis names, so it
            # is attached here rather than left to the caller to reconstruct.
            analysis["table_id"] = table["table_id"]
            analysis["table_number"] = table["table_number"]
            analysis["table_label"] = table["table_label"]
            analysis["table_caption"] = table["caption"]
            analysis["table_footer"] = table["footer"]
        return study, table, parsed, None

    # Flattened to (study, table) rather than pooled over studies: table counts run 1 to 5
    # here, so a study-wide pool leaves most workers idle behind the widest paper.
    done: dict[str, list[tuple[dict, list[dict]]]] = {}
    if work:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for study, table, parsed, error in pool.map(parse_one, work):
                if error is not None:
                    print(f"  {study} {table['table_id']}: FAILED {error}", file=sys.stderr)
                    failures += 1
                    continue
                done.setdefault(study, []).append((table, parsed))

    for pmid, study, _axis in read_pmids(args.pmids):
        if study not in done:
            continue
        study_dir = args.texts / study
        analyses: list[dict] = []
        # Back into manifest order, so the output does not depend on which worker
        # finished first and two runs of the same corpus stay comparable.
        order = {t["table_id"]: i for i, t in enumerate(coordinate_tables(study_dir))}
        print(f"{study} (pmid {pmid}):")
        for table, parsed in sorted(done[study], key=lambda r: order.get(r[0]["table_id"], 99)):
            split = split_opposite_signs(parsed)
            parsed, notes = list(split.analyses), list(split.notes)
            for note in notes:
                print(f"    {note}")
            analyses.extend(parsed)
            print(
                f"  {table['table_id']}: {len(parsed)} analyses, "
                f"{sum(len(a.get('points') or []) for a in parsed)} points"
            )

        out_dir = study_dir / "stage1"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "analyses.json").write_text(
            json.dumps(
                {
                    "study": study,
                    "pmid": pmid,
                    "model": args.model,
                    "effort": args.effort,
                    "prompt_version": COORDINATE_PARSING_PROMPT_VERSION,
                    "analyses": analyses,
                },
                indent=1,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        pond = pond_analyses(study_dir)
        (out_dir / "diff-vs-pond.md").write_text(
            diff_report(study, analyses, pond), encoding="utf-8"
        )
        mark = "same count" if len(analyses) == len(pond) else f"DIFFERS (pond {len(pond)})"
        print(f"  -> {len(analyses)} analyses, {mark}\n")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
