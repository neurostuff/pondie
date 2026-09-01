# Direction without asking

Two rules, both cheap, both measured against the 328 reviewed direction cells in
`data/gold/direction`. They answer different questions and only one of them is what the
statistic can support.

## The statistic sign cannot direct a cell

`parse_tables.split_opposite_signs` already uses the sign of a row's statistic, and that
is the right use of it: a table holding effects of both signs is two analyses, and the
partition is arithmetic. The tempting next step is to skip asking for direction at all
and read it off the sign. That does not work, and the gold says why plainly.

Of 79 reviewed analyses, 60 (76%) carry a statistic with an unambiguous sign. But the
sign is the same either way the contrast is written. In `JzsUUQbDr2bm`, `analysis_01` and
`analysis_02` both have `sign=+1`; the first assigns FESZ positive and NC negative, the
second assigns them the other way round. A paper reporting "FESZ > NC" and one reporting
"NC > FESZ" both print positive t-values. **Polarity is in the contrast's name, not in
its statistics.**

## The contrast's name can

`derive_direction.polarity` reads the comparison out of the analysis's own name and
definition -- `FESZ>NC`, `AD < HC reduced GM volume`, `greater activation in patients
than controls` -- and gives each named level a direction.

| | |
|---|---|
| cells it answers | 55 of 328 (**17%**) |
| correct | 54 (**98%**) |
| declines | 273 |

98% is at parity with the extraction pass's 96.6%, at no cost. The comparison where both
answer is more useful than either number:

| | cells |
|---|---|
| agree, and correct | 49 |
| agree, and both wrong | 0 |
| disagree, rule right | 5 |
| disagree, model right | 0 |

In every disagreement the model had said `absent` -- it declined to assign a direction --
and the rule recovered five of those correctly and lost nothing. The rule is not a
replacement for the pass; it is a filler for the cells the pass gives up on.

The one remaining error is genuinely ambiguous: `Baseline > week 6` on an analysis whose
subject is *change over six weeks*, where the naming convention and the recorded
direction disagree about what the contrast is of.

Levels are matched by word-set containment and never by a similarity ratio. A 0.85
threshold matches `men` to `women` and scores `synchronous` against `asynchronous` at
0.96, and a cell given the wrong level is a wrong direction. Stopwords are dropped so
`ASD` reaches `ASD group`, but a side built only from stopwords -- `patients than
controls` -- is compared on every word instead, because treating it as empty loses the
commonest phrasing there is.

## The reversed half is arithmetic, not extraction

When a table reports both signs, the paper's prose is about one of them. "FESZ > NC"
prints positive statistics for the effects it names; the negative rows are the same
contrast read the other way, and that reading is almost never written down. Asking a
model to name and define a contrast with no prose behind it produces invention, not
extraction -- and costs a full analysis's worth of tokens to do it.

The split therefore gives the extraction pass only the described half, under the parsed name
unchanged. The negative half is emitted with `withhold: True` and `mirror_of`, and
`stage1_block` filters it out of the prompt. After extraction,
`derive_direction.mirror_analysis` rebuilds it: the described half's cells with their
directions flipped, carrying the withheld half's own rows.

`undirected`, `held` and `absent` survive the mirror unchanged -- a level held constant
is held from either side of the contrast, and an undirected effect has no sign to flip.
Only a flipped direction is marked `value_source: generated`; an unflipped one keeps the
warrant it was read from.

## What is and is not wired in

| | |
|---|---|
| sign-based table splitting | `parse_tables.split_opposite_signs` |
| withholding the reversed half | `parse_tables.split_opposite_signs` + `extract_record.stage1_block` |
| direction from the contrast name | `build_record.fill_directions` |
| mirroring the withheld half | `build_record.mirror_withheld` |
| the eight field derivers | `derive_fields.py` at the repo root, measured, **still not wired in** |

## Where the two rules sit in the build

Both run inside `build_record.build`, and the order is the point.

`fill_directions` runs **after** `align_cell_levels`, because it matches a level against
the contrast's name and an unaligned level is the wrong string to match. It fills only
`absent` and never overrides a direction the model committed to: a rule that answers a
sixth of the cells has no standing to overturn the pass on the rest, and a silent
overwrite would hide a disagreement worth reading.

`mirror_withheld` runs **last of the repairs**, on the assembled record rather than the
raw payload, so the mirror is taken from the contrast the model settled on -- including
whatever the wrapper repairs, the level alignment and the direction fill changed about
it. It runs before the span walk, so the mirrored quotes resolve to offsets like any
other and face the same integrity gate.

A withheld half whose described partner is not in the record is reported, never invented.

## The statistics flip with the contrast

A `t` of -2.9 for "FESZ > NC" is +2.9 for the reversed reading: same effect, same
magnitude, the other way round. The mirror therefore carries the withheld rows with their signed
statistics negated, and the mirrored analysis reports positive statistics for the effects
it names, exactly as the described half does.

p-values, cluster sizes and voxel counts are not negated. A p-value is positive in either
reading of a contrast, and flipping it would print a number the paper never did.
