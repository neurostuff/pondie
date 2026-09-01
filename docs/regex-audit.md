# The regular expressions, audited against the corpus

> Where this sits in the pipeline: [pipeline-architecture.md](pipeline-architecture.md).

186 `re.*` call sites in 25 modules, 128 distinct literal patterns and about a dozen more
assembled from fragments. This file is what each group of them is for, what the corpus says
it actually matches, and which ones should change — measured over all 39,273 studies on
`beast:/data/alejandro/projects/ns-pond/data`, one preferred text per paper in `Flavour`
order for prose and every table flavour for tables.

All seven findings below are **implemented**; the eighth is recorded as a proposal and
deliberately left unimplemented. Every number is measured, and the acceptance run compares
the edited modules against the originals over the whole corpus.

## Scope, and why it is not all 156 patterns

The population is lopsided. `preprocess.py` holds 87 of the 186 sites, `table_parse.py`,
`retrieval.py` and `scoring.py` another 29 between them; the remaining 21 modules average
under four each and are almost entirely `re.sub(r"\s+", " ")`, `re.sub(r"[^a-z0-9]+", "")`
and `re.escape`-wrapped literal lookups. Auditing those individually produces a hundred
entries reading "well scoped, no change" and buries the seven that matter.

So this audit covers the four subsystems that carry risk, and states the rule that clears
the rest:

| in scope | why |
|---|---|
| preprocessing digests (`prompt/preprocess.py`) | regex over prose, feeding a prompt |
| evidence units (`evidence/retrieval.py`) | regex deciding what a quote may be |
| table structure (`formats/table_parse.py`) | regex deciding where a coordinate is |
| folding and vocabulary (`vocabularies/folding.py`, `onvoc.py`) | regex deciding whether two names are one |

Out of scope, and why it is safe to leave them: a pattern is **well scoped by
construction** when its input is a string this repo produced, its grammar is fixed, and a
failure is loud rather than silent. `\s+` collapse, `[^a-z0-9]+` folding, `\[\d+\]` path
indices in `scoring.py`, `_STEP` in `adjudicate.py`, `_PLACEHOLDER` in `rebuild.py` and
every `re.escape`-wrapped literal are all of that kind. They are documented here as a class
and recommended unchanged.

## What the roles are

The audit judges a pattern against its job, not against a general idea of tidiness.

- **Candidate retrieval and digests** favour recall. `preprocess.py` labels its output
  "Derived from the paper by regular expression, not read. It over-generates", and the
  model is asked to confirm every entry. Leakage here is cheap.
- **Deterministic derivation, normalization and table parsing** favour precision. A value
  that reaches a schema slot without a model reading it is a claim about the paper.
- **Identifiers, paths, Markdown structure and numeric parsing** need exact grammar. There
  is a right answer and anything else is a bug.

Three of the findings below are patterns whose role and whose scoping disagree.

## What the edit did, measured

The edited `preprocess.py`, `retrieval.py` and `table_parse.py` run against their originals
over all 38,126 papers and 76,131 tables:

| | before | after |
|---|---:|---:|
| evidence units cut at an abbreviation | 1,564,410 | **76** |
| evidence units returned | 13,530,941 | 12,650,056 |
| headings that classify to a section | 267,879 | **267,984** |
| `diagnosis` digest matches | 99,121 | **45,500** |
| `scanner` digest matches | 81,446 | **75,848** |
| tables resolving no coordinate axis | 5,929 | **1,979** |
| tables raising out of `read_table` | 74 | **0** |

And the things that had to not happen, all confirmed zero: no table lost an axis it used to
have, no `axis_cols` changed, `axis_cell` and `axis_cols` are never both set, no `diagnosis`
match was removed that was not a `mini` prefix leak, and none was gained.

## Findings, validated and implemented

### 1. The evidence splitter has no abbreviation guard — implement

[`retrieval.py:463`](../pondie/extraction/evidence/retrieval.py#L463) splits units on
`(?<=[.;!?])\s+|\n\n+`, which cuts at `et al.`, `e.g.` and `vs.`.
[`preprocess.py:183`](../pondie/extraction/prompt/preprocess.py#L183) already carries the
`_NON_TERMINAL` token guard that solves this, tested, in the same package.

Over 38,126 papers, comparing the shipped function against the same function with that
guard applied, both including the `MIN_UNIT`/`MAX_UNIT` filter:

| | shipped | guarded |
|---|---:|---:|
| units returned | 13,530,931 | 12,835,695 |
| units cut at an abbreviation | 1,301,745 (9.6%) | 176 |
| fragments discarded as under `MIN_UNIT` | 4,588,143 | 2,993,542 |
| units discarded as over `MAX_UNIT` | 13,403 | 19,312 |

37,085 papers — 97% — are affected. The second row is the defect; the third is the part
worth stating plainly: **1.59 million fragments are currently being discarded entirely**,
because a split at `Li et al.` leaves a stub under fifteen characters and the filter drops
it. The cost is 5,909 units that merge past `MAX_UNIT`, 0.04% of the total.

Implemented by exporting `preprocess.ends_mid_sentence` and calling it from
`sentence_units`. One abbreviation list, not two: `_split_sentences` now goes through the
same predicate, so the two splitters cannot drift. Residual after the fix is 76 units.

### 2. `retrieval._HEADING` caps at four hashes — implement

[`retrieval.py:55`](../pondie/extraction/evidence/retrieval.py#L55) is `^(#{1,4})\s*(.+?)\s*$`
where [`preprocess.py:44`](../pondie/extraction/prompt/preprocess.py#L44) allows `{1,6}`.
On a level-5 heading the fifth `#` is not consumed by the group and becomes the first
character of the heading *text*:

    ##### Results   ->   ('####', '# Results')

`_canon_heading` strips leading digits and `.:` but not `#`, so `classify_heading` is handed
`# results` and every `^`-anchored entry in `_SECTION_PATTERNS` misses:

    classify_heading('Results')   -> 'results'
    classify_heading('# Results') -> None
    classify_heading('# Discussion') -> None

1,747 papers disagree between the two regexes and 1,710 of them (4.5% of the corpus) contain
a level-5 heading. That is the exposure, and it is worth separating from the impact, which is
smaller: judging each version by its own canonicaliser, the fix classifies **105 more headings**
across the corpus, 267,879 to 267,984. Most level-5 headings read `Participants` or
`Statistical analysis`, and those `_SECTION_PATTERNS` entries are not `^`-anchored, so they
were being classified through the stray marker anyway. It is the anchored labels — `^results?$`,
`^discussion`, `^abstract` — that were silently lost, and those are usually not nested five
deep. A one-character fix for a hundred sections, kept because the failure mode is silent and
the next `#######` costs nothing.

This is the finding a three-paper sample cannot produce: on the nine local texts the three
heading regexes agree byte for byte, and the earlier draft of this audit recorded "leave
alone" on that basis.

Implemented: `{1,6}`, and `_canon_heading` strips `#` so the next such gap is inert.

### 3. `MINI` has no trailing boundary — implement

[`preprocess.py:838`](../pondie/extraction/prompt/preprocess.py#L838), the `diagnosis`
alternation, opens `\b(?:` and never closes. `MINI` therefore matches the prefix of any word
beginning with those letters:

| | corpus |
|---|---:|
| `MINI` matches that are the instrument | 8,388 |
| `MINI` matches followed by another letter | 53,614 |

**86.5% of them are false** — `minimize`, `minimal`, `minimum`, `miniblocks`. Implemented as
`MINI\b`, which takes the digest from 99,121 matches to 45,500. The acceptance run confirms
what a first count only suggested: **every removed match is a `mini` prefix leak and none is
gained**. `Mini-Mental` and `MINI-International` survive, because a hyphen is a word boundary
and both are the instruments the slot asks about. 61% of papers change. This is a precision-role pattern feeding `Group.medical_condition` — unlike the cue
sweeps, its leakage is not free.

The other alternations that open `\b(?:` without closing it — `_COORDINATE_CUE`,
`_CONTRAST_CUES`, the statistic cues — are recall-role and are recommended **unchanged**.

### 4. `_axis_cell` demands `x,y,z` in a header that says `MNI` — implement

The highest-value change in the audit. Across all three table flavours, 76,057 tables read:

| flavour | coordinate tables | axes resolved | unresolved | recoverable |
|---|---:|---:|---:|---:|
| pubget | 19,978 | 17,466 | 2,512 | **1,661** |
| elsevier | 19,551 | 17,350 | 2,201 | **1,595** |
| ace | 7,460 | 6,244 | 1,216 | **634** |

5,929 coordinate tables (12.6%) resolve no axes at all.
[`_axis_cell`](../pondie/formats/table_parse.py#L448) is the tier that handles a single
column holding the whole triple, and it requires `AXIS_TRIPLE` — literally `x`, `y`, `z`
separated by `[,;/]` — in the header. Real headers do not oblige:

    'MNI Peak Coordinates'          over  ['(48, 24, 2)', '(-34, -54, -33)']
    'Location in MNI152'            over  ['(1, 55, -3)', '(-39, -77, 33)']
    'Centroid (in MNI coordinates)' over  ['-53.8, -29.9, -19.4', ...]
    'Peak MNI coordinates(x y z)'   over  ['-3 47 35', '42 38 21']

The last one names all three axes and still fails, because `AXIS_TRIPLE` requires a
punctuation separator and this header uses spaces.

Testing `COORDISH` instead of `AXIS_TRIPLE` in `_axis_cell`, with the existing
majority-of-rows `TRIPLE_CELL` confirmation untouched, resolves **3,890 of the 5,929**
unresolved tables — 66% of them, spread evenly across flavours. The numeric confirmation is
what makes this safe: a column headed `MNI` whose cells are not majority-triples still
resolves to nothing, which is the current behaviour.

Implemented in two passes rather than one condition — `AXIS_TRIPLE` across every header row
first, then `COORDISH` — because merged into a single test a `COORDISH` cell earlier in the
row won the scan from the `AXIS_TRIPLE` cell that used to answer, and five tables changed the
column they resolved to for no reason. The acceptance run: **3,950 tables now resolve, no
table lost an axis, and no `axis_cols` changed**.

One further correction the corpus forced. `axis_cell` documents itself as set "only when the
three axes share one column, and then `axis_cols` is None", and once the header test admits
`COORDISH` eight tables satisfy both readings. `read_table` now computes `axis_cell` only when
`axis_cols` is None, so the key means what it says; seven tables cede an `axis_cell` they used
to report to the `axis_cols` they also had, and none loses an answer.

### 5. `DEDUP.match(...).group(...)` is unguarded — implement

[`table_parse.py:431`](../pondie/formats/table_parse.py#L431) dereferences
`DEDUP.match(row[index] or "")` without checking for `None`. `DEDUP` is
`^(?P<base>.*?)(?:\.(?P<n>\d+))?$` without `DOTALL`, so **any header cell containing an
interior newline returns `None`** and the third tier of `_axis_columns` raises
`AttributeError`:

    DEDUP.match('Peak coordinates')   -> 'Peak coordinates'
    DEDUP.match('Peak\ncoordinates')  -> None

`_clean` only strips the ends of a cell, so an embedded newline survives from a quoted CSV
field or an XML text node. 278 such header cells exist in the corpus, and 74 tables — 73
ace, 1 elsevier — crashed `read_table` outright. It is 0.1% of tables and it is a crash rather
than a degradation, which is why it is listed above the larger-volume findings that only lose
precision. Implemented as a `_dedup_base` helper that treats an unreadable cell as its own
base; the acceptance run reads all 74 without raising, and the colspan tier goes on to resolve
them, which is the shape that tier exists for.

### 6. `squash` does not do what its docstring says — implement

[`folding.py:41`](../pondie/vocabularies/folding.py#L41) claims to be "`fold` with the spaces
removed" and skips `fold`'s NFKD step, so an accent is deleted rather than folded:

    fold('naïve')  = 'naive'      squash('naïve')  = 'nave'
    fold('Étude')  = 'etude'      squash('Étude')  = 'tude'

245 of 160,565 MONDO and Cognitive Atlas surface forms diverge (0.15%) — `Möbitz`,
`Müllerian`, `perlèche` — and 21,820 papers (57%) contain at least one accented character
that folds to ASCII. The single caller is exact-key clustering in
[`_clustering.py:39`](../pondie/normalization/_clustering.py#L39), so an accented and an
unaccented spelling of one name fail to cluster, which is the job clustering exists to do.
Low volume, one line, and strictly correct: `squash` now calls `fold` and drops the spaces,
so the two cannot diverge again.

### 7. Two scanner vendor tokens, not the whole list — implement, narrowed

[`preprocess.py:561`](../pondie/extraction/prompt/preprocess.py#L561) matches vendor and
model names case-sensitively. Scored on the sentence the token sits in, 32,057 matches
involve a token that is also an ordinary word or a non-MRI product:

| | matches |
|---|---:|
| sentence also names a scanner, magnet, coil or field strength | 26,098 |
| sentence names a camera, projector, amplifier or electrode and no MRI noun | 184 |
| sentence names neither | 5,775 |

The confirmed-wrong cases are unambiguous — `microphone (Canon DM-100)`, `LCD projector
(WUX5000, Canon Inc.)`, `amplifier (Biotop; GE Marquette Medical Systems)`,
`BBC/Discovery Channel` — but they are rare. The volume is in the no-context column, and it
is concentrated in exactly two tokens:

- **`Discovery`** (1,966): dominated by `AI Discovery Assistant`, journal website chrome
  scraped into the ace render.
- **`GE`** (2,887): dominated by author initials in reference lists — `Holder GE`,
  `Smith GE`, `John GE`.

Every other ambiguous token is under 200. So the change worth making is not "add a context
guard to the vendor list", it is a guard on `GE` and `Discovery`. Implemented by lifting both
out of the general vendor alternation into a pattern that requires a division name
(`GE Healthcare`), a model (`GE Signa`, `Discovery MR750`, `Discovery 750w`) or an MRI noun
within a dozen characters (`3T MRI scanner (GE)`). Of 13,635 corpus mentions, 7,068 are kept
and 6,567 dropped, and the digest goes from 81,446 scanner matches to 75,848.

A negative lookahead earns its place here: `\bGE\b(?![\s-]?EPI)`. The corpus's most common
non-vendor `GE` is `GE-EPI` — gradient echo — and it sits in exactly the sentence, next to
exactly the field strength, that the context test was written to trust. The `GE` case also says
something the audit did not go looking for: reference-list text is reaching a sweep that is
zone-scoped to `{methods, front}`, so `back` zone detection is failing on the ace flavour.
That is a separate question and is not addressed here.

### 8. `singular()` over-strips Latin nouns — implement, as a lemmatizer

`folding.py` was `\b(\w{4,})s\b`, whose four-letter floor protects `bias` and `axis` and
nothing longer. Over the vocabularies it mangled 2,594 distinct words in 32,629 places —
`sclerosis`→`sclerosi`, `corpus`→`corpu`, `sinus`→`sinu`, `nervous`→`nervou`.

This audit originally recorded it as correct in role and left it alone, reasoning that
`variants()` applies it to both the query and the vocabulary key, so the mangling cancels.
**That reasoning was wrong.** `Vocabulary.surface` is built from `fold` alone
([`mondo.py:83`](../pondie/vocabularies/mondo.py#L83)), never from `singular`, so the
singularised form is only ever a query. Nothing cancelled: a mangled variant was a key that
could not exist, and the 1,505 collisions counted here were collisions among singularised
vocabulary keys that no lookup consults.

`singular()` is now WordNet noun lemmatisation via nltk. Measured over Mondo's 90,244 folded
surface forms, the two disagree on 22,337, and the lemma lands on a real vocabulary key
**128 times against the suffix rule's 53**. It also reaches concepts the suffix rule could
not spell: `psychoses`→`psychosis` resolves to *psychotic disorder*, where `psychose`
resolved to nothing. `gyri`→`gyrus`, `foci`→`focus`, `stimuli`→`stimulus` and
`indices`→`index` are new for the same reason — a suffix rule has nothing to say about an
irregular plural.

The lesson worth keeping is not about plurals. A symmetry argument is only as good as the
claim that both sides get the same treatment, and that claim was never checked against the
code that builds the other side.

## Findings that did not survive validation

### 9. Tightening the cohort count phrase — do not implement

[`preprocess.py:804`](../pondie/extraction/prompt/preprocess.py#L804) uses a `{0,120}?` gap
so a count phrase can cross the parenthetical that usually sits between the number and the
noun. It over-runs badly: of 438,911 matches, 115,035 (26%) exceed 60 characters and 39,300
(9%) exceed 100, producing spans like

    16 trials, a seventeenth "catch" trial contained a 1-back vigilance task, in which the participant

A candidate forbidding bare commas while still allowing one bracketed group cuts over-100
spans by 91%. It also **loses 124,726 matches, of which 6,750 are short, comma-free and
entirely legitimate**, and the matches it gains include `7 million cases` and
`12 Because patient`. It trades one kind of junk for another.

The rule this audit works to is that a change ships when its new matches are valid for the
role and its removed matches are confirmed false positives. This one fails the second half.
Recorded as a proposal, runtime unchanged. A cleaner attack is probably not a regex at all:
the digest wants the sentence, and the caller already has a sentence splitter.

## Confirmed well scoped

- **`TRIPLE_CELL`** — 152,554 matches. The 33,596 cells that hold three numbers and are
  rejected are `18 (14-23)`, `11.5 (5-17)`: median-and-range demographics, correctly refused.
- **`_NUMBER`** — 843,967 rejections, all of the shape `68 ± 13`, `6 (50%)`, `.76`. It exists
  to confirm a coordinate column and a coordinate is none of those.
- **The three axis tiers all earn their place**, though unevenly: `AXIS` is necessary for
  13,339 tables, the triple-label tier for 23,302, and `PAREN_AXIS` for 54. Marginal is not
  dead; keep it.
- **Backtracking.** The nested-quantifier suspects were timed on adversarial input.
  `_SHARED_HEAD` is quadratic — 3.9x per input doubling, 21ms at 890 characters — but it runs
  per sentence, so its input is bounded. `_ANATOMY_HEAD` and the cohort `age` pattern are
  linear. Nothing is exponential and nothing needs rewriting.
- **`preprocess` vs `text_index` headings** — 117 papers differ, on a trailing non-breaking
  space that `\s*` strips and `[ \t]*` keeps. `text_index` owns the offsets every
  `EvidenceSpan` addresses; the gain does not justify touching it.
- **The structural long tail** — whitespace collapse, alphanumeric folding, `\[\d+\]` path
  indices, `re.escape`-wrapped lookups. Fixed grammar over strings this repo produced.

## Method

Both `preprocess.py` and `retrieval.py` are standard-library only and `table_parse.py` is
too, so validation shipped the real modules to the corpus host and exec'd them there rather
than transcribing patterns into a test harness. What is measured above is the shipped
expression. The corpus was read and never written; no model or paid API call was made.

Two harness errors are worth recording so the numbers are read correctly. A first pass
measured `TRIPLE_CELL` "near misses" with a three-group regex that backtracks inside `27.95`
and reported `27 . 95 4.52` as a triple; the count is tokenised now. A first pass also called
`read_table` with the empty `data_file` that `read_manifest` yields for elsevier and ace,
concluded both flavours were unreadable, and was wrong —
[`rebuild.py:174`](../pondie/extraction/corpus/rebuild.py#L174) reconstructs the basename from
the table id. `read_manifest`'s docstring does claim all three flavours "write the same
manifest fields", which is false: elsevier uses `metadata.raw_xml_path` and ace has no path
key at all. The compensating fallback lives in the caller. That is a documentation defect
rather than a bug, and it is the reason the first table pass measured nothing.
