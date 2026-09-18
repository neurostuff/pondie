"""A table reporting both signs is two contrasts. Partitioning it is arithmetic.

The rule behind `StageName.sign_split`, and the reason it can only live before the model
passes: the schema requires a separate Analysis per normalized direction, and this is the
only point in the pipeline that can see a row's values. The extraction passes are shown
captions, never cells -- so asking the model to split on "effects of opposite sign" asks it
to use a signal it cannot observe.

It was in `extraction/corpus/tables.py`, a module otherwise about getting one LLM call per
table over the wire, and the `SignSplit` stage reached it through a deferred import. That
made `corpus/__init__.py`'s claim -- "Nothing here runs during an extraction. All of it is
an input." -- false, when it is the load-bearing statement about that package: a run reads
`data/corpus/<id>/` and never writes it, which is what makes two runs comparable.

`direction.mirror_analysis` is the other half, after extraction: the withheld part is
rebuilt by reversing the directions the model assigned to the half the paper described.
"""

from __future__ import annotations

from collections import defaultdict

from pondie.extraction.models import SplitResult

#: Statistic kinds with no direction to give. A p-value is positive whichever way the
#: contrast runs, so reading a sign off one would split every table.
NON_DIRECTIONAL_KINDS = frozenset({"p-value"})


def _point_sign(point: dict) -> int | None:
    """A row's direction from its statistics: +1, -1, or None when it has no sign to give.

    None covers two different silences that must not be split on. A row carrying only a
    p-value or a cluster extent has no direction printed, and a row whose statistics
    disagree in sign -- a positive t beside a negative correlation -- is a parse to look at
    rather than a row to file.
    """

    signs = {
        1 if value > 0 else -1
        for entry in (point.get("values") or [])
        if entry.get("kind") not in NON_DIRECTIONAL_KINDS
        and isinstance(value := entry.get("value"), (int, float))
        and value != 0
    }
    return signs.pop() if len(signs) == 1 else None


def split_opposite_signs(analyses: list[dict]) -> SplitResult:
    """Split any analysis whose rows report both directions into one analysis per direction.

    The schema requires a separate Analysis per normalized direction, and this is the only
    stage that can see the row values: the extraction passes are shown captions, never
    cells. The downstream pass is therefore asked to split on "effects of opposite sign" using a
    signal it cannot observe. Doing it here makes the partition arithmetic instead.

    Splitting only, never merging, and only on a total partition -- if any row has no sign
    the analysis is reported and left whole, because a partial split files some rows and
    silently strands the rest. Group sizes are not weighed: one surviving cluster in the
    minority direction is an ordinary result of thresholding, and 11% of parsed analyses
    have a single point already, so a lone row is no evidence of a bad parse.

    Only the positive-sign part is offered to the extraction pass, and it keeps the
    parsed name unchanged. A paper that reports "FESZ > NC" prints positive statistics
    for the effects it describes and negative ones for the same contrast read the other
    way; the reversed half is almost never written down, so asking a model to name and
    define it invites invention. The negative part is emitted withheld, carrying
    `mirror_of`, and `direction.mirror_analysis` rebuilds it after extraction by
    reversing the directions the model assigned to the half that was described.
    """

    out: list[dict] = []
    notes: list[str] = []

    for analysis in analyses:
        points = analysis.get("points") or []
        signs = [_point_sign(point) for point in points]
        present = {s for s in signs if s is not None}

        if len(present) < 2:
            out.append(analysis)
            continue

        name = analysis.get("name") or "(unnamed)"
        if None in signs:
            unsigned = sum(1 for s in signs if s is None)
            notes.append(
                f"FLAG {name}: both directions present but {unsigned} of "
                f"{len(points)} rows carry no sign -- left whole"
            )
            out.append(analysis)
            continue

        for sign, label in ((1, "positive"), (-1, "negative")):
            part = dict(analysis)
            part["points"] = [p for p, s in zip(points, signs) if s == sign]
            #: The parent's identity, kept so the split is auditable and so a reviewer can
            #: see that two entries came from one parse rather than from two table rows.
            part["split_from"] = name
            part["split_direction"] = label
            part["split_rule"] = "sign-of-directional-statistic"
            if sign > 0:
                # The half the paper describes. Its name is the paper's.
                part["name"] = name
            else:
                part["name"] = f"{name} (reversed)"
                part["mirror_of"] = name
                #: Never shown to the extraction pass. The reversed contrast has no prose
                #: in the paper to quote, so a model asked to define it can only guess.
                part["withhold"] = True
            out.append(part)

        counts = f"{signs.count(1)}+/{signs.count(-1)}-"
        notes.append(
            f"SPLIT {name} -> ({counts}) on statistic sign; "
            f"the negative half is withheld and mirrored after extraction"
        )

    return SplitResult(analyses=tuple(out), notes=tuple(notes))


def adopt_withholding(analyses: list[dict]) -> SplitResult:
    """Convert a pair split by the earlier rule into a described half and a withheld one.

    A corpus partitioned before the mirror existed holds both halves as ordinary entries,
    `<name> (positive)` and `<name> (negative)`, and both were sent to the extraction
    pass. The negative half has no prose in the paper to quote, so what came back for it
    was invention -- and it cost a full analysis's worth of tokens to obtain.

    Re-splitting cannot reach these: each part already holds one sign, so
    `split_opposite_signs` correctly finds nothing to do. The conversion is done from the
    parts themselves, which carry `split_from` and `split_direction` and so record
    everything needed. Nothing is re-parsed and no statistic is re-read.

    Only a clean pair converts. Three parts sharing a parent means the entry was also
    split on something else -- a band, a session -- and which of them the paper describes
    is not answerable from the sign alone.
    """

    families: dict[str, list[dict]] = defaultdict(list)
    for analysis in analyses:
        if analysis.get("split_rule") == "sign-of-directional-statistic":
            parent = analysis.get("split_from")
            if parent:
                families[parent].append(analysis)

    converted: list[str] = []
    for parent, parts in families.items():
        directions = [p.get("split_direction") for p in parts]
        if sorted(directions) != ["negative", "positive"]:
            continue
        # Already in the described/withheld shape. Reporting a conversion here would
        # make the stage that calls this look permanently unfinished, so a resumed run
        # would re-enter it forever.
        if any(part.get("withhold") for part in parts):
            continue
        for part in parts:
            if part["split_direction"] == "positive":
                part["name"] = parent
                part.pop("withhold", None)
                part.pop("mirror_of", None)
            else:
                part["name"] = f"{parent} (reversed)"
                part["mirror_of"] = parent
                part["withhold"] = True
        converted.append(f"WITHHOLD {parent}: the reversed half is no longer extracted")
    return SplitResult(analyses=tuple(analyses), notes=tuple(converted))
