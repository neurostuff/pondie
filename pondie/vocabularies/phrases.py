"""What a field value is *about*, before any vocabulary is consulted.

A layer above `labels`, and the one that was missing. `folding` claims orthography and
`labels` claims which tokens carry identity; both assume the value names one thing and
names it positively. A `Group.medical_condition` frequently does neither:

    "no neurological or psychiatric disorder"     an ABSENCE. A disease vocabulary
                                                  retrieves a disease from this every time.
    "alcohol abuse or dependence"                 TWO heads, and one row each is what makes
                                                  a comorbid cohort queryable.
    "first-episode schizophrenia"                 one head plus a course qualifier, which
                                                  is not a different disease.

Measured over the 2,115-record corpus on beast, **22% of `medical_condition` values state
an absence** -- 452 as `no`/`none`/`without` and 414 as a bare `healthy` -- and 4% split into
more than one head under the conservative splitter below. Matching those without triage is
where the ONVOC route's errors came from: `absence of major depressive disorder` returned
Depressive Disorder, and `No clinically significant cognitive impairment No dementia`
returned Depressive Disorder from a cohort that had neither.

Triage first, then look up each head. Both the MONDO route in
`normalization.medical_condition` and the ONVOC route in `vocabularies.onvoc` read it from
here, so the two cannot drift -- which they had, the MONDO one having the gate and the
ONVOC one not.

Negation is read from the sentence's syntax (`normalization._negation`); the regexes here
are the fallback for a host with no parser. Why, and what the anchored version got wrong:
docs/condition-normalization.md.

The rules are domain-general over clinical writing rather than over English, which is why
they are here and not in `folding`: `first-episode` and `treatment-resistant` are course
and state in psychiatry, and `or` separates comorbidities where `and` does not -- "attention
deficit and hyperactivity disorder" is one disorder.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pondie.normalization import _negation as negation
from pondie.vocabularies.folding import fold

# `unimpaired`, `intact`, `good` and `normal weight` are here because the corpus put them
# in `medical_condition` and every one of them was being looked up as a disease: see
# docs/normalization-rationale.md, "is_healthy".
_WELL = (
    r"healthy|health|normal[\s-]weight|normal|unaffected|unimpaired|non-?impaired|intact|"
    r"good|excellent|non-?clinical|non-?patient|non-?smok(?:er|ers|ing)|"
    r"typically\s+develop(?:ing|ed)|typical|neuro-?typical|comparison|none|nil"
)
_ROLE = (
    r"controls?|volunteers?|participants?|subjects?|adults?|individuals?|groups?|child|"
    r"children|adolescents?|teens?|teenagers?|infants?|toddlers?|neonates?|boys?|girls?|"
    r"students?|twins?|siblings?|aged|cognition|"
    r"men|women|males?|females?|elderly|older|younger|young|aging|ageing|"
    r"persons?|people|population|cohort|sample|donors?|reported|noted|documented|"
    r"identified|found|present|"
    # `Normal hearing` and `normal-hearing listeners` are control groups in audiology papers.
    r"hearing|vision|sight|eyesight|listeners?|readers?|speakers?"
)
# Adjectives sit here beside their adverbs: the pattern only uses this list as a modifier
# slot, and "good physical health" needs `physical` exactly where `physically` is allowed.
_ADVERB = (
    r"physically|mentally|medically|generally|otherwise|neurologically|psychiatrically|"
    r"cognitively|clinically|overall|physical|mental|medical|general|neurological|"
    r"psychiatric|cognitive|somatic"
)
#: A value made of nothing but wellness words, study-role nouns and punctuation, anchored
#: at both ends. A word in neither list -- `smokers`, `obese` -- ends the match, so a bare
#: assertion of health is an absence and `healthy` in front of a condition is not.
HEALTHY = re.compile(
    rf"^\W*(?:(?:{_ADVERB})\W+)?(?:{_WELL})(?:\W+(?:{_WELL}|{_ROLE}|{_ADVERB}))*\W*$",
    re.I,
)

#: A non-answer rather than an answer, kept apart from `HEALTHY` because the schema keeps
#: them apart: an unread cohort is not a well one. `extraction_status` carries this when
#: the extractor sets it; this catches it arriving as a string in the value.
UNREAD = re.compile(
    r"^\W*(?:not[_ ]?(?:reported|applicable|stated|specified|available|assessed)|"
    r"unknown|unspecified|undetermined|n/?a)\W*$",
    re.I,
)

#: Course and state, not a different disease. Lifted off the head and kept beside it:
#: `first-episode schizophrenia` and `chronic schizophrenia` are one vocabulary term, and a
#: query that cannot tell them apart is a different problem from one that cannot find them.
QUALIFIER = re.compile(
    r"\b(first[- ]episode|(?:childhood|adolescent|adult|juvenile|youth)[- ]onset|"
    r"chronic|acute|early[- ]onset|late[- ]onset|"
    r"remitted|in remission|treatment[- ]resistant|refractory|"
    r"drug[- ]na[iï]ve|medicated|unmedicated|recent[- ]onset|stable|"
    # `mild` names a severity everywhere but in mild (neuro)cognitive impairment, a condition
    r"current|lifetime|past|subclinical|mild(?!\s+(?:neuro-?)?cognitive\b)|moderate|severe|recurrent|"
    r"childhood|adolescent|adult|p(?:a)?ediatric|comorbid|probable|possible|"
    r"suspected|sporadic|familial|presymptomatic|"
    r"intractable|(?:drug|medication|pharmaco)[- ]?resistant|newly[- ]diagnosed|de novo|"
    r"combat[- ]related|history of|prior|previous|concurrent)\b",
    re.I,
)

#: A prevalence or count, which says how many in the group have the condition and is not part
#: of its name: "panic disorder in 2", "cognitive impairment in 18/44 patients", "PTSD: 9
#: (23.7%)", "(n = 12)". Removed before the split, because `18/44` would otherwise split on
#: its slash, and before retrieval, which accepted "panic disorder in 2" as *panic disorder 2*,
#: an OMIM locus. The item keeps the count, so a comorbid role can still read it.
_NUMBER = (
    r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|a few|few|some|"
    r"several|a subset|a minority|a small number)"
)
_UNITS = r"(?:patients?|participants?|subjects?|cases?|individuals?|children|adults|people)"
_OF = rf"(?:\s+(?:of|out of)\s+(?:the\s+)?(?:\d+\s+)?(?:{_UNITS}|sample|cohort|group|total|\d+))?"
PREVALENCE = re.compile(
    r"\s*\(\s*(?:n\s*=\s*\d+|\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d+|\d+\s+" + _UNITS + r")[^)]*\)"
    r"|[,;:]?\s*\bn\s*=\s*\d+\b"
    r"|[,;:]\s*\d+\s*(?:\(\s*\d+(?:\.\d+)?\s*%\s*\))?\s*$"
    rf"|[,;:]?\s+in\s+(?:\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d+|{_NUMBER}){_OF}"
    rf"(?:\s+{_UNITS})?(?:\s+(?:overall|in total))?(?![\w%])"
    # and a clause about those participants: "in 43 participants meeting PCL-R total >= 30"
    r"(?:\s+(?:meeting|who|whom|with|according|based|defined|scoring|during|receiving|"
    r"having|treated|diagnosed)\b.*$)?"
    rf"|[,;:]?\s+\d+(?:\.\d+)?\s*%{_OF}",
    re.I,
)

#: A slash inside one name, where `SPLIT` would otherwise make two heads of it. The commonest
#: in the corpus: "attention-deficit/hyperactivity disorder" (515 items) split into
#: "attention-deficit", which retrieval placed at *ADHD, inattentive type*. Listed rather than
#: inferred: "panic disorder/agoraphobia" and "overweight/obesity" are two conditions written
#: the same way.
IN_NAME = re.compile(
    r"\b(attention[- ]?deficit|non-?fluent|logopenic|avoidant|tremor|depersonali[sz]ation|"
    r"vegetative state|hyperactive|binge[- ]eating)"
    r"\s*/\s*"
    r"(?=hyperactivity|agrammatic|phonological|restrictive|ataxia|dereali[sz]ation|"
    r"unresponsive|impulsive|purging)",
    re.I,
)
#: What stands in for an `IN_NAME` slash until the split is done. `fold` reads either as a space.
_KEPT = "\u2215"

#: "MCI due to Alzheimer's disease" states two conditions the group has: the impairment and the
#: disease causing it (James, 2026-11-27). Written as two parts before the split.
_DUE_TO = re.compile(
    r"\b((?:amnestic |amnesic )?(?:mild cognitive impairment|MCI))\s+due to\s+", re.I
)

#: "alcohol abuse/dependence" and "cannabis abuse or dependence" name the substance once for
#: both conditions. Split as written, the second head is a bare "dependence", which MONDO
#: places at *substance dependence*; the substance is written onto both sides first.
_SUBSTANCE_PAIR = re.compile(
    r"\b([a-z]+)\s+(abuse|dependence|dependency|misuse)(\s*/\s*|\s+or\s+)"
    r"(abuse|dependence|dependency|misuse)\b",
    re.I,
)

#: Split on `or`, `/` and `;` only -- never on `and` or a comma. The slot is multivalued, so
#: several conditions crammed into one string is an extraction defect, and the splitter is
#: deliberately conservative about calling one: splitting on `and` breaks "attention deficit
#: and hyperactivity disorder", and on a comma it breaks "dementia, Alzheimer's type". 4% of
#: the corpus's values split here, where a loose test finds a separator in 13% -- the other 9%
#: are prose that should not have been put in the slot, and leaving them whole keeps them
#: visible as an extraction defect instead of shredding them into fragments.
SPLIT = re.compile(r"\s+and\s*/\s*or\s+|\s+or\s+|\s*/\s*|\s*;\s*")

#: Study-role nouns at the end of a value. `Schizophrenia patients` is a cohort description
#: whose condition is Schizophrenia; the head noun is about the study, not the disease.
TRAILING = re.compile(
    r"\s*\b(in some participants|in a subset|patients?|subjects?|"
    r"participants?|individuals?|group|cohort|status)\b\s*$",
    re.I,
)

#: What triage concluded a value was, when it was not a condition to look up.
NO_CONDITION = "NO_CONDITION"

#: What triage concluded when the value is a non-answer.
NOT_READ = "NOT_READ"


@dataclass(frozen=True)
class Triaged:
    """A value decomposed into what to look up, what was lifted off it, and what it denies."""

    heads: tuple[str, ...]
    qualifiers: tuple[str, ...]
    kind: str
    #: Heads the value states the ABSENCE of. Never looked up, and carried rather than
    #: dropped: "screened and found none" is not "never asked".
    denied: tuple[str, ...] = ()
    #: How the negation was read -- `parse`, `cue`, `assertion`, or "". The method is the
    #: confidence, the same way `Mapping.method` is.
    scope: str = ""

    @property
    def sentinel(self) -> str:
        """The answer when there is nothing to look up, and "" when there is."""
        return self.kind if self.kind in (NO_CONDITION, NOT_READ, "empty") else ""


def _heads_of(text: str) -> tuple[list[str], list[str]]:
    """Split a positive span into heads, lifting the course-and-state qualifiers off."""
    heads: list[str] = []
    quals: list[str] = []
    for part in (p.strip() for p in SPLIT.split(text) if p and p.strip()):
        quals += [q.lower() for q in QUALIFIER.findall(part)]
        head = TRAILING.sub("", QUALIFIER.sub("", part)).strip(" -,")
        head = re.sub(r"\s{2,}", " ", re.sub(r"\(\s*\)", "", head)).strip().replace(_KEPT, "/")
        if head:
            heads.append(head)
    return heads, quals


def triage(raw: object, known: Callable[[str], bool] | None = None) -> Triaged:
    """Head terms, the qualifiers lifted off them, and what the value turned out to be.

    Four layers in order: `UNREAD`, `HEALTHY`, a negation scope over the WHOLE value, then
    a cue window over each part of what survives. Parsing before the split is what lets a
    cue scope over a coordination -- see docs/condition-normalization.md.

    What the value is *about* is all this decides. Whether it belongs in the slot is the
    schema's question: `healthy smokers` in `medical_condition` is a slot defect, and the
    condition in it is recovered rather than swallowed.

    `TRAILING` runs twice on purpose -- once on the whole value and once per head -- because
    a compound puts a role noun on the last head only.

    `known`, given, says whether a string is a name the caller's vocabulary holds. A value
    that is one whole name is then one head, before negation or splitting could break it:
    "migraine without aura" reads "without" as a negation cue, and "chronic fatigue
    syndrome" loses `chronic` as a course qualifier.
    """
    raw_text = str(raw or "").strip()
    if not raw_text:
        return Triaged((), (), "empty")
    if UNREAD.match(raw_text):
        return Triaged((), (), NOT_READ, scope="assertion")
    text = TRAILING.sub("", PREVALENCE.sub("", raw_text).strip(" ,;:")).strip()
    if not text:
        return Triaged((), (), "empty")
    if HEALTHY.match(text):
        return Triaged((), (), NO_CONDITION, scope="assertion")
    if known is not None and known(text):
        return Triaged((text,), (), "single")
    text = _SUBSTANCE_PAIR.sub(r"\1 \2\3\1 \4", _DUE_TO.sub(r"\1; ", text))
    text = IN_NAME.sub(lambda m: m.group(1) + _KEPT, text)
    if HEALTHY.match(text):
        return Triaged((), (), NO_CONDITION, scope="assertion")

    scoped = negation.scope(text)
    if scoped is not None:
        asserted, denied_text, how = scoped[0], scoped[1], "parse"
    else:
        # The cue window runs over the WHOLE value before the split, not only per part.
        # `SPLIT` breaks on ` or `, and without a parse nothing else knows that a cue
        # scopes over the coordination it introduces: "no neurological or psychiatric
        # disorder" would hand `psychiatric disorder` to the vocabulary as a diagnosis.
        asserted = negation.cue_forward_scope(text)
        denied_text = text[len(asserted) :] if asserted != text else ""
        how = "cue"

    heads, quals, denied = [], [], []
    for part in (p.strip() for p in SPLIT.split(asserted) if p and p.strip()):
        kept = negation.cue_forward_scope(part)
        if fold(kept) != fold(part):
            denied.append(part)
            how = how if kept else ("cue" if scoped is None else how)
        part_heads, part_quals = _heads_of(kept)
        heads += part_heads
        quals += part_quals
    if denied_text.strip():
        denied.append(denied_text.strip())

    # The wellness test again, on what the negation left standing: "healthy adults without
    # neurologic disorders" survives as "healthy adults", which is still an absence.
    if heads and HEALTHY.match(" ".join(heads)):
        return Triaged((), tuple(sorted(set(quals))), NO_CONDITION, tuple(denied), how)

    # And once per head, which the joined test cannot do: a compound states one thing per
    # part, so "Healthy; obesity" is a cohort with obesity and `Healthy` is not a second
    # condition to look up. The joined test stays above it because a head can be a bare
    # role noun -- "healthy; adults" joins to a wellness phrase and neither part is one.
    kept_heads = [h for h in heads if not HEALTHY.match(h)]
    if heads and not kept_heads:
        return Triaged((), tuple(sorted(set(quals))), NO_CONDITION, tuple(denied), how)
    heads = kept_heads

    if not heads:
        kind = NO_CONDITION if denied else "empty"
        return Triaged((), tuple(sorted(set(quals))), kind, tuple(denied), how if denied else "")
    return Triaged(
        tuple(heads),
        tuple(sorted(set(quals))),
        "compound" if len(heads) > 1 else "single",
        tuple(denied),
        how if denied else "",
    )


def absent(text: object, is_healthy: object = None) -> bool:
    """Does this cohort record an absence? The string decides; `is_healthy` fills gaps.

    The flag can add an absence and never remove one: it is derived from this field, so a
    flag that could overrule its own source would be a loop. Measurements in
    docs/condition-normalization.md.
    """
    triaged = triage(text)
    if triaged.kind == NO_CONDITION:
        return True
    if triaged.heads:
        return False
    return is_healthy is True
