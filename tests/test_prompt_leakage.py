"""No verified gold record's content may reach the extraction prompt.

An extractor handed a gold value scores for reproducing it, and the score then measures
the prompt rather than the reading. The failure is silent and it flatters: it looks exactly
like the pipeline having improved.

This is not hypothetical. A worked example written into `recheck_cells.py`'s instructions
was lifted verbatim from `xevP8UDRAVh9`'s gold record -- "placebo-associated perfusion" --
and the configuration using it was, briefly, the best-scoring one in the sweep.

The check runs against every record in `benchmarks/gold/`, so a paper added to the gold set is
automatically defended, and it covers the static prompt material only. Text that legitimately
varies with the paper -- the paper itself, the stage-1 parse, a previous pass's payload --
is input, not leakage.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "benchmarks" / "gold"
from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

import extract_record as er  # noqa: E402
import preprocess  # noqa: E402
import schema_utils  # noqa: E402

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

#: Values short or generic enough that a match says nothing. A schema vocabulary term the
#: paper happens to use is the prompt doing its job -- `diagnostic interview` is an
#: `assessment_type` permissible value, and it has to be in the prompt for the field to be
#: fillable at all.
MIN_LENGTH = 12
MAX_LENGTH = 90


def gold_records() -> list[Path]:
    return sorted(GOLD.glob("*.extraction.json"))


def distinctive_strings(record: dict) -> set[str]:
    """Extracted values specific enough that finding one in a prompt means it leaked."""

    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "value" and isinstance(value, str):
                    found.add(value)
                elif key == "value" and isinstance(value, list):
                    found.update(v for v in value if isinstance(v, str))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(record)
    return {s for s in found if MIN_LENGTH <= len(s) <= MAX_LENGTH and " " in s}


def static_prompt_sources() -> dict[str, str]:
    """Everything the pipeline puts in a prompt that does not vary with the paper."""

    recheck = (ROOT / "pondie" / "extraction" / "passes" / "recheck_cells.py").read_text(
        encoding="utf-8")
    sources = {
        "extraction-readme.md": er.README.read_text(encoding="utf-8"),
        "representing-models.md section 5": er.worked_models(),
        "SYSTEM_HEAD": er.SYSTEM_HEAD,
        "recheck_cells.py SYSTEM": recheck.split('SYSTEM = """')[1].split('"""')[0],
        "ZERO_FOCI_RULE": er.ZERO_FOCI_RULE,
        # The preprocessing digests' preambles. Their bodies are derived from the paper
        # and so are input, but the headings and cautions around them are as static as
        # SYSTEM_HEAD and a gold phrase written into one would leak the same way.
        "preprocess.py prompt literals": "\n".join(preprocess.PROMPT_LITERALS),
    }
    sources.update({f"MODE_NOTE[{name}]": note for name, note in er.MODE_NOTE.items()})

    classes = schema_utils.load_imported_classes(er.EXTRACTION_SCHEMA)
    enums = schema_utils.load_imported_classes(er.EXTRACTION_SCHEMA, "enums")
    for mode in ("entities", "analyses"):
        names, keep = er.mode_classes(classes, er.MODE_SCHEMA.get(mode, mode))
        # Enum values are the vocabulary the field is filled from, not leaked content, so
        # they are stripped before the comparison.
        rendered = er.render_schema(classes, enums, names, keep)
        for enum in enums.values():
            for value in (enum or {}).get("permissible_values") or {}:
                rendered = rendered.replace(value, " ")
        sources[f"rendered schema ({mode})"] = rendered
    return sources


@pytest.mark.parametrize("gold_path", gold_records(), ids=lambda p: p.name.split(".")[0])
def test_no_gold_value_appears_in_the_static_prompt(gold_path):
    record = json.loads(gold_path.read_text(encoding="utf-8"))
    values = distinctive_strings(record)
    assert values, f"{gold_path.name} yielded no distinctive strings to check"

    leaks: dict[str, list[str]] = {}
    for name, text in static_prompt_sources().items():
        lowered = text.lower()
        hit = sorted({v for v in values if v.lower() in lowered})
        if hit:
            leaks[name] = hit

    assert not leaks, "gold content reached the prompt:\n" + "\n".join(
        f"  {source}: {values!r}" for source, values in leaks.items())


def test_the_check_would_catch_a_real_leak():
    """A guard that cannot fail is not a guard."""

    record = json.loads(gold_records()[0].read_text(encoding="utf-8"))
    values = distinctive_strings(record)
    planted = max(values, key=len)
    assert planted.lower() in f"an instruction mentioning {planted} verbatim".lower()
