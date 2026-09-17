"""`Group.population_characteristics` -> the selective traits, with the rest set aside.

The field holds what a study chose its cohort for: habitual exposure, training, occupation,
lifestyle, atypical body habitus. Its value is that a query can filter on it, and that
value survives only if every entry is discriminative.

Asking for that did not work. Three rewrites of the slot description, each tested by
re-extracting the same ten papers, left non-selective entries at 24%, then 14%, then 7% --
differences of two and three entries out of thirty, on a sample too small to tell the
versions apart. `is_healthy` had already shown why: a description cannot outvote the
source's own wording. So the question stays as it is and the answer is partitioned
afterwards, which is a rule that can be read, tested against the whole corpus, and changed
without a re-extraction.

WHAT MOVES. A value goes to `other_characteristics` when every plausible cohort could carry
it -- "normal weight", "right-handed", "normal or corrected-to-normal vision", "no
psychiatric history", "MRI compatible", "native English speakers". Moved, not dropped: the
value is not wrong, and a reader auditing a cohort wants to see it. It simply cannot share a
field with a trait a filter would select on. `other_characteristics` is `deterministic` in
the storage schema, so the generator never puts it to a model and nothing is asked twice.

TWO ASYMMETRIES, both of which a blunter rule gets backwards:

  Handedness. "right-handed" is normative; "left-handed" and "mixed-handed" are selective,
  because a study recruiting left-handers recruited for that.

  Negation. A negated *condition* is normative -- "no neurological or psychiatric disorder"
  is carried by every control cohort in the corpus. A negated *exposure* is selective: "no
  history of smoking" is the control arm of a smoking study, and "cannabis use less than 50
  times" is how a cue-reactivity paper defines its comparison group. `EXPOSURE` is therefore
  decisive against every rule here, which is what makes the negation rule safe to state
  broadly.

Matching is full-string against a reduced core, never a substring search. "Otherwise healthy
adult smokers" reduces to "healthy smokers", which no rule matches in full, so it stays --
where `search(r"healthy")` would have moved it and lost the cohort's defining trait.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from pondie.normalization._records import DEFAULT, iter_records, strings_at

#: The three outcomes. KEPT is the default and the safe one: no rule calling a value
#: non-selective means it stays where the model put it.
NORMATIVE, KEPT, EMPTY = "NORMATIVE", "KEPT", "EMPTY"

#: Generic person nouns and hedges, stripped from both ends before matching. They carry no
#: trait -- "healthy weight children" and "normal weight" are the same claim -- and leaving
#: them in would need every pattern to end in an optional noun phrase.
PERSON = (
    r"participants?|subjects?|volunteers?|individuals?|persons?|people|controls?|"
    r"comparisons?|adults?|children|kids|adolescents?|teenagers?|youths?|infants?|"
    r"elderly|seniors?|men|man|women|woman|males?|females?|boys?|girls?|"
    r"group|groups?|cohort|sample|cases?"
)
HEDGE = (
    r"all|both|each|every|otherwise|generally|mostly|largely|reportedly|self-?reported(?:ly)?|"
    r"entirely|only|were|was|are|is|being|had|have|with|of|the|a|an|and"
)
_EDGE = re.compile(rf"^(?:(?:{HEDGE}|{PERSON})\b\W*)+|(?:\W*\b(?:{HEDGE}|{PERSON}))+\W*$", re.I)

#: Content-free entries. Not normative -- there is nothing to keep track of -- so they are
#: dropped rather than moved.
BLANK = re.compile(
    r"n/?a|not applicable|not reported|none(?: reported| noted| specified)?|"
    r"no (?:other |specific |notable |additional |particular )?characteristics?"
    r"(?: reported| noted| specified)?|nothing (?:notable|remarkable)|unremarkable|"
    r"not (?:described|stated|mentioned)",
    re.I,
)

#: Decisive against every rule below. These are the exposures the corpus's studies recruit
#: for, so a value naming one is selective however it is phrased -- including negated, which
#: is how a control arm is defined. See the module docstring.
EXPOSURE = re.compile(
    r"smok|cigarette|nicotine|tobacco|vap(?:e|ing)|alcohol|drink|drunk|binge|audit|"
    r"drug|substance|cannabis|marijuana|thc|cocaine|opioid|opiate|heroin|methadone|"
    r"amphetamin|methamphetamin|ketamine|mdma|ecstasy|caffeine|coffee|"
    r"gambl|gaming|internet use|abstin|dependen|withdrawal|craving|"
    r"athlet|musician|meditat|yoga|bilingual|multilingual|expert|train(?:ed|ing)|"
    r"sedentary|vegetarian|vegan|diet(?:er|ing)|exercis|veteran|combat|deploy|"
    r"shift work|undergraduate|student|obes|overweight|underweight|anorexi|bulimi",
    re.I,
)

#: The negation rule, in two parts, because "no illness" and "no anxiety disorder" cannot
#: share a pattern. Three kinds of word are deliberately absent from GENERIC, and each
#: absence removes a false positive the corpus sweep produced:
#:
#:   exposures -- "no substance abuse" and "no history of smoking" are how this corpus's
#:   control arms are defined, so they are selective. `EXPOSURE` enforces this separately.
#:
#:   bare qualifiers (significant, major, chronic, clinical, history). With `significant`
#:   here, "no significant re-experiencing, avoidance, or hyperarousal symptoms" moved -- a
#:   PTSD study's comparison group, and the clearest wrong answer in the sweep. A qualifier
#:   can still sit inside the phrase; it just cannot be the thing negated, which is what
#:   QUALIFIER below allows and GENERIC does not.
#:
#:   named diagnoses. Negating one defines a comparison group in the study that named it,
#:   and the diagnosis belongs in `medical_condition` regardless. This is also why
#:   `diagnosis` and `disorder` are not in GENERIC: with them there, "No PTSD diagnosis"
#:   and "no post-traumatic stress disorder" both moved. They reach BARE instead, which
#:   admits them only when nothing but a qualifier stands between the negation and the
#:   noun -- so "no chronic conditions" moves and "no anxiety disorder" does not.
GENERIC = (
    r"neurolog\w*|psychiatr\w*|psycholog\w*|neuropsychiatr\w*|neuropsycholog\w*|"
    r"medical|mental|somatic|systemic|axis\s*[i1]+|dsm[\w-]*|comorbid\w*|"
    r"patholog\w*|abnormalit\w*|"
    r"head (?:injury|trauma)|brain (?:injury|lesion|damage)"
)
#: Words that may stand between the negation and a bare illness noun without making the
#: value selective. Course, recency and hedging: none of them names what was absent.
QUALIFIER = (
    r"significant|serious|major|minor|chronic|acute|current|currently|past|prior|previous|"
    r"lifetime|known|suspected|documented|reported|self-?reported|diagnosed|relevant|"
    r"specific|other|any|all|overt|concurrent|concomitant|history|histories|"
    r"of|or|and|the|a|an"
)
#: Illness nouns with no domain of their own. Alone they are generic; qualified by anything
#: other than QUALIFIER they name a diagnosis, and BARE will not match.
BARE_ILLNESS = (
    r"disease|diseases|disorder|disorders|illness|illnesses|condition|conditions|"
    r"diagnosis|diagnoses|ailments?|morbidit(?:y|ies)|problems?|complaints?"
)
NEGATION = r"(?:no|not|without|free (?:of|from)|negative for|absence of|lacking|devoid of)"
TRAILING = r"(?:\s+(?:reported|noted|present|documented|whatsoever|at all))?"

#: name -> the pattern, matched in FULL against the reduced core. Order fixes only which
#: name is reported when two match; every one of them means NORMATIVE.
RULES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "health",
        re.compile(
            r"(?:healthy|normal|normals|unaffected|non-?clinical|non-?patient|"
            r"neurotypical|well|in good health|good general health|"
            r"(?:generally|otherwise|physically|medically|mentally) healthy)",
            re.I,
        ),
    ),
    (
        "no_condition",
        re.compile(
            rf"{NEGATION}\b(?:\W|\w){{0,40}}?\b(?:{GENERIC})\b.*"
            rf"|{NEGATION}(?:\W+(?:{QUALIFIER}))*\W+(?:{BARE_ILLNESS}){TRAILING}",
            re.I,
        ),
    ),
    (
        "weight",
        re.compile(
            r"(?:normal|healthy|average|ideal|lean)[\s-]?(?:body[\s-]?)?"
            r"(?:weight|weights|bmi|mass index|habitus|build)"
            r"(?:[\s-]?range)?|"
            r"(?:bmi|body mass index)\s*(?:was\s*)?(?:with)?in (?:the )?normal(?: range| limits)?|"
            r"non-?obese|non-?overweight",
            re.I,
        ),
    ),
    (
        "development",
        re.compile(
            r"typical\w*[\s-]?develop\w*|normal\w*[\s-]?develop\w*|"
            r"development\w*(?:ly)? (?:normal|typical|appropriate)|"
            r"no developmental (?:delay|delays|disorder|disorders|concerns?)|"
            r"age[\s-]?appropriate(?: development)?",
            re.I,
        ),
    ),
    (
        "handedness",
        re.compile(
            r"(?:strongly |consistently |predominantly )?"
            r"right[\s-]?hand(?:ed|edness)?(?:[\s-]?dominant)?|"
            r"right[\s-]?hand(?:[\s-]?side)? dominant|dextral|"
            r"(?:edinburgh|annett)[\w\s]*(?:handedness)?[\w\s]*right",
            re.I,
        ),
    ),
    (
        "senses",
        re.compile(
            r"(?:normal|intact|adequate|unimpaired)"
            r"(?:[\s,]+or corrected(?:[\s-]to[\s-]normal)?)?\s*"
            r"(?:vision|visual acuity|visual function|eyesight|sight|hearing|audition|"
            r"auditory acuity|colou?r vision)|"
            r"corrected[\s-]to[\s-]normal(?: vision| visual acuity)?|"
            r"(?:no|without|free of) (?:hearing loss|hearing impairment|visual impairment|"
            r"colou?r blindness|uncorrected \w+)|"
            r"not colou?r[\s-]?blind|no (?:sensory|auditory|visual) (?:deficits?|impairments?)",
            re.I,
        ),
    ),
    (
        "cognition",
        re.compile(
            r"cognitively (?:normal|intact|unimpaired|healthy)|"
            r"(?:normal|average|intact|unimpaired) "
            r"(?:iq|intelligence|cognition|cognitive function|cognitive ability)|"
            r"(?:iq|mmse|intelligence)\D{0,20}(?:with)?in (?:the )?normal"
            r"(?: range| limits)?|"
            r"no cognitive (?:impairment|decline|deficits?)",
            re.I,
        ),
    ),
    (
        "language",
        re.compile(
            r"native (?:\w+[\s-]?)?speakers?(?: of \w+)?|native \w+[\s-]?speaking|"
            r"(?:fluent|proficient) in \w+|\w+ (?:as|was) (?:a |their )?(?:first|native) "
            r"language|monolingual(?: \w+)?|\w+[\s-]?speaking",
            re.I,
        ),
    ),
    (
        "mri_eligibility",
        re.compile(
            r"mri[\s-]?(?:compatible|eligible|safe|suitable)|"
            r"(?:eligible|suitable|cleared) for (?:mri|scanning|an? mri scan)|"
            r"no (?:metal|metallic|ferromagnetic) implants?|"
            r"no contraindications?(?: (?:to|for) (?:mri|scanning|magnetic resonance))?|"
            r"not pregnant|no(?:t)? (?:pregnancy|claustrophobi\w*)|no claustrophobia",
            re.I,
        ),
    ),
    (
        "consent",
        re.compile(
            r"(?:able|capable|competent) (?:to|of) (?:give|giving|provide|providing|sign) "
            r"(?:written )?(?:informed )?consent|"
            r"(?:gave|provided|signed) (?:written )?informed consent|"
            r"capacity to consent|consented",
            re.I,
        ),
    ),
)

_DASH = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def fold(raw: object) -> str:
    """Case, accents, dash codepoints, whitespace and edge punctuation, and nothing else."""
    text = unicodedata.normalize("NFKD", str(raw or ""))
    text = "".join(c for c in text if not unicodedata.combining(c)).translate(_DASH)
    return re.sub(r"\s{2,}", " ", text.strip().lower()).strip(" .;,:-")


def core(raw: object) -> str:
    """`fold`, then generic person nouns and hedges off both ends until neither end has one.

    Iterated rather than applied once: "all healthy adult participants" needs three passes,
    and one pass would leave "healthy adult participants" for the patterns to allow for.
    """
    text = fold(raw)
    while True:
        stripped = _EDGE.sub("", text).strip(" .;,:-")
        if stripped == text or not stripped:
            return stripped or text
        text = stripped


@dataclass(frozen=True)
class Verdict:
    """What one entry is, which rule said so, and the core the rule actually saw."""

    kind: str
    rule: str
    text: str
    reduced: str = ""

    def __bool__(self) -> bool:
        return self.kind == NORMATIVE


def normalize(text: object) -> Verdict:
    """Classify one `population_characteristics` entry."""
    raw = str(text or "").strip()
    folded = fold(raw)
    # BLANK is tested before the edges are stripped. "n/a" tokenises as "n" and "a", and "a"
    # is a hedge, so stripping first leaves "n" -- which is not blank and not a trait either.
    if not folded or BLANK.fullmatch(folded):
        return Verdict(EMPTY, "blank", raw, folded)
    reduced = core(raw)
    if not reduced or BLANK.fullmatch(reduced):
        return Verdict(EMPTY, "blank", raw, reduced)
    if EXPOSURE.search(reduced):
        return Verdict(KEPT, "exposure", raw, reduced)
    for name, pattern in RULES:
        if pattern.fullmatch(reduced):
            return Verdict(NORMATIVE, name, raw, reduced)
    return Verdict(KEPT, "unmatched", raw, reduced)


def partition(entries: list[str]) -> tuple[list[str], list[str], list[Verdict]]:
    """(selective, non-selective, every verdict). Both sides deduplicated case-blind.

    Deduplication is here because it is the same defect: the first test round produced three
    repeats out of 33 entries, and "Heavy drinkers" three times is one trait.
    """
    keep: dict[str, str] = {}
    other: dict[str, str] = {}
    verdicts = []
    for entry in entries:
        verdict = normalize(entry)
        verdicts.append(verdict)
        if verdict.kind == EMPTY:
            continue
        side = other if verdict.kind == NORMATIVE else keep
        side.setdefault(verdict.reduced or entry.strip().lower(), entry.strip())
    return list(keep.values()), list(other.values()), verdicts


def tally_empty(verdicts: list[Verdict]) -> int:
    return sum(1 for v in verdicts if v.kind == EMPTY)


def apply(record: dict) -> dict[str, int]:
    """Partition every group's `population_characteristics` in place. Returns a tally.

    The kept side stays in its original wrapper, so the model's evidence and `value_source`
    travel with the values it is evidence for. The moved side gets a fresh wrapper marked
    `derived`, the same convention `is_healthy` uses, so a reader can tell the two apart.

    A group whose field was never extracted is left alone. Unread is not empty.
    """
    tally = {"groups": 0, "kept": 0, "moved": 0, "dropped": 0, "deduped": 0}
    for group in record.get("groups") or []:
        if not isinstance(group, dict):
            continue
        wrapper = group.get("population_characteristics")
        if not isinstance(wrapper, dict):
            continue
        entries = strings_at(group, "population_characteristics")
        if not entries:
            continue
        keep, other, verdicts = partition(entries)
        tally["groups"] += 1
        tally["kept"] += len(keep)
        tally["moved"] += len(other)
        tally["dropped"] += sum(1 for v in verdicts if v.kind == EMPTY)
        tally["deduped"] += len(entries) - len(keep) - len(other) - tally_empty(verdicts)
        wrapper["value"] = keep
        if other:
            group["other_characteristics"] = {
                "value": other,
                "extraction_status": "extracted",
                "value_source": "derived",
                "evidence": None,
            }
    return tally


#: Where these traits land in records extracted before the field existed. The committed
#: corpus has no `population_characteristics` at all -- it predates the slot -- so `report`
#: falls back to these, which is where the same strings were going instead. Long prose is
#: excluded by `MAX_WORDS`: `clinical_characteristics` is sentences by design and this rule
#: is for phrases.
PROXY_PATHS = (
    "groups.medical_condition",
    "groups.inclusion_criteria",
    "groups.exclusion_criteria",
    "groups.clinical_characteristics",
)
MAX_WORDS = 8


def scan(
    patterns: tuple[str, ...] = DEFAULT, paths: tuple[str, ...] | None = None
) -> list[Verdict]:
    paths = paths or ("groups.population_characteristics",)
    return [
        normalize(s)
        for _study, body in iter_records(patterns)
        for path in paths
        for s in strings_at(body, path)
        if len(s.split()) <= MAX_WORDS
    ]


def report(patterns: tuple[str, ...] = DEFAULT) -> str:
    """The rule over the corpus, with every distinct match listed so it can be argued with."""
    records = sum(1 for _ in iter_records(patterns))
    if not records:
        return f"no records matched {patterns}"
    verdicts = scan(patterns)
    header = f"groups.population_characteristics: {len(verdicts)} values in {records} records"
    if not verdicts:
        verdicts = scan(patterns, PROXY_PATHS)
        header = (
            f"groups.population_characteristics: 0 values in {records} records -- this "
            "corpus predates the slot.\n"
            f"Proxy: {', '.join(p.split('.')[-1] for p in PROXY_PATHS)}, "
            f"{len(verdicts)} values of at most {MAX_WORDS} words, which is where these "
            "traits were landing instead."
        )
    kinds = Counter(v.kind for v in verdicts)
    total = max(1, len(verdicts))
    lines = [header, ""]
    for kind in (NORMATIVE, KEPT, EMPTY):
        lines.append(f"  {kind:10s} {kinds[kind]:6d}  ({kinds[kind] / total:4.0%})")
    by_rule = Counter(v.rule for v in verdicts if v.kind == NORMATIVE)
    lines.append("\n  moved to other_characteristics, by rule:")
    for name, n in by_rule.most_common():
        forms = Counter(v.text for v in verdicts if v.kind == NORMATIVE and v.rule == name)
        lines.append(f"    {name:16s} {n:5d}  {len(forms)} distinct")
        for form, count in forms.most_common(6):
            lines.append(f"        {count:4d}  {form[:76]!r}")
    held = Counter(v.rule for v in verdicts if v.kind == KEPT)
    lines.append(f"\n  kept: {held['unmatched']} unmatched, {held['exposure']} decided by EXPOSURE")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
