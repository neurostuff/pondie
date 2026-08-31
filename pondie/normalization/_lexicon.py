"""The closed-target shape: free-text input, a small fixed set of answers.

Rules and not an encoder, and the reason is auditability rather than accuracy. Sixteen
surface forms over two coordinate spaces, or 293 over four correction methods, is a case
where a rule can be read and argued with and a cosine cannot -- and where a wrong answer is
acted on, not merely displayed.

Two conventions every field here shares:

  UNKNOWN is not OTHER.  OTHER asserts an answer outside the known set; UNKNOWN asserts we
  cannot tell. They license different downstream actions -- a transform must refuse OTHER and
  may fall back on a default for UNKNOWN -- so collapsing them loses the distinction that
  matters.

  Nothing is bucketed silently.  An input no rule matches is UNKNOWN with `reason="unmatched"`
  and is reported by `residual`, so a new spelling forces a rule rather than vanishing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from pondie.normalization import OTHER, UNKNOWN


@dataclass(frozen=True)
class Rule:
    """One answer and the pattern that reaches it. Order matters: first match wins."""

    value: str
    pattern: re.Pattern
    #: A decisive rule settles the question regardless of what else matches. Negation is the
    #: case it exists for: "not medicated" contains "medicated", so the two compete and the
    #: negation has to win rather than register as an ambiguity.
    decisive: bool = False

    @classmethod
    def of(cls, value: str, pattern: str, decisive: bool = False) -> "Rule":
        return cls(value, re.compile(pattern, re.I), decisive)


@dataclass(frozen=True)
class Decision:
    value: str
    reason: str
    text: str = ""

    def __bool__(self) -> bool:
        return self.value != UNKNOWN


def classify(text: object, rules: tuple[Rule, ...], ambiguous_to: str = UNKNOWN) -> Decision:
    """First matching rule wins; several distinct matches is an ambiguity, not a choice."""
    raw = text if isinstance(text, str) else ""
    if not raw.strip():
        return Decision(UNKNOWN, "empty", raw)
    decisive = next((r for r in rules if r.decisive and r.pattern.search(raw)), None)
    if decisive is not None:
        return Decision(decisive.value, "decisive", raw)
    hit = [r.value for r in rules if r.pattern.search(raw)]
    distinct = list(dict.fromkeys(hit))
    if not distinct:
        return Decision(UNKNOWN, "unmatched", raw)
    if len(distinct) > 1 and OTHER not in distinct:
        return Decision(ambiguous_to, f"matches {' and '.join(distinct)}", raw)
    return Decision(distinct[0], "lexical", raw)


def residual(decisions: list[Decision]) -> Counter:
    """The inputs that need a rule, most common first."""
    return Counter(d.text for d in decisions if d.reason == "unmatched" and d.text)


def summarize(decisions: list[Decision], values: tuple[str, ...]) -> str:
    counts = Counter(d.value for d in decisions)
    total = max(1, sum(counts.values()))
    lines = [
        f"  {v:34s} {counts[v]:6d}  ({counts[v] / total:4.0%})" for v in values if counts[v]
    ]
    folded = Counter((d.value, d.text) for d in decisions if d.reason == "lexical" and d.text)
    for v in values:
        forms = [(t, n) for (val, t), n in folded.items() if val == v]
        if forms:
            top = sorted(forms, key=lambda kv: -kv[1])[:5]
            lines.append(f"     {v} <- " + ", ".join(f"{t!r}x{n}" for t, n in top))
    left = residual(decisions)
    if left:
        lines.append(f"  {sum(left.values())} value(s) no rule matched:")
        lines += [f"     {n:5d}  {t!r}" for t, n in left.most_common(8)]
    return "\n".join(lines)


@dataclass(frozen=True)
class ClosedField:
    """A field whose answers are a small fixed set: where it lives, and how to read it."""

    path: str
    rules: tuple[Rule, ...]
    values: tuple[str, ...]
    ambiguous_to: str = UNKNOWN

    def normalize(self, text: object) -> Decision:
        return classify(text, self.rules, self.ambiguous_to)

    def scan(self, patterns: tuple[str, ...] | None = None) -> list[Decision]:
        from pondie.normalization._records import DEFAULT, iter_records, strings_at

        return [
            self.normalize(s)
            for _study, body in iter_records(patterns or DEFAULT)
            for s in strings_at(body, self.path)
        ]

    def report(self, patterns: tuple[str, ...] | None = None) -> str:
        decisions = self.scan(patterns)
        return f"{self.path}: {len(decisions)} values\n" + summarize(decisions, self.values)
