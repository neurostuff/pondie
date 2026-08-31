# Candidate pipelines, and what each would buy

> P1-P7 are here; P8-P12, the data sources and the
> stage-by-stage anatomy are in [pipeline-architecture.md](pipeline-architecture.md).

The goal bounds the design: a queryable reconstruction of the analyses **that coordinate
tables report**, with every value warranted by a span. Two consequences that the current
pipeline does not fully exploit:

- **The table is the anchor.** If an analysis exists because a coordinate table reports it,
  then the stage-1 parse *is* the analysis inventory, and the contrast layer is a question
  about row groups rather than about the paper.
- **Direction carries 0.45 of the composite** and is the one fact a synthesis cannot recover
  from anywhere else. Cost saved anywhere else is worth nothing if direction moves.

Baseline to beat: **P0**, `demand-driven --zero-foci-rule --max-attempts 3` on
`gpt-5.6-luna`. Direction 96.6% on the 101-cell reviewer gold; human ceiling 95.8%.

## P1 — Deterministic direction first, model only on abstention

Stage 1 already names contrasts, and 51% of those names are formal comparisons (`FESZ>NC`,
`Baseline > week 6`). The operator plus the side a level sits on gives the sign outright; for
a slope, the parsed statistic's sign does. Derive those, ask the model only for the rest.

*Measured already:* the deriver answers 55 of 101 signed gold cells at **100%** against
corrected gold, and abstains on the other 46 rather than guessing.

## P2 — Table-anchored contrast schema

Replace the nested `Analysis.effect.cells[]` ask with a flat row per parsed contrast:

    {parse_index, axis_term, plus_levels[], minus_levels[], within_levels[], sign_status}

`parse_index` points into the stage-1 listing, so no analysis can be invented; `axis_term`
and the level lists are closed choices over the model's declared terms. Absence is the
default, so over-celling becomes unrepresentable, and `held`/`undirected`/`not_reported`
collapse into one `sign_status` rather than three shapes the model must pick between.

## P3 — Deterministic + NuExtract3 pre-fill, luna for the remainder

Derive the 8 fields at 93.7-100%, let NuExtract3 fill the ~30 it clears 80% on
(~6s/paper), pass both to luna as candidates in the existing "already extracted" block
shape, and have luna emit only what is left.

*Measured:* the offloadable set is **31% of record JSON** — but only the verified-≥80%
fields, and **not** the contrast layer, where NuExtract3 scores 37%.

## P4 — Agreement cascade

Run the deterministic deriver and NuExtract3 independently. Accept where they agree; route
disagreement and mutual abstention to luna. Agreement is the confidence signal *because the
model's own confidence is not* — per-value logprob correlates **r = −0.40** with
correctness, and sequence logprob **r = −0.71**. Two independent methods agreeing is the only
calibrated signal available.

## P5 — Level shortlist as a closed vocabulary

`Cell.level` must name a declared `FactorLevel`, and level fidelity is what grounds
direction — the exporter and scorer bugs this corpus produced were both level-matching
failures. NuExtract3 recovers levels from table headers at **100%** where the headers encode
them (`delay/immediate`, `IGD/HC`, `AD/HC`) and **0%** where the design is only in the
caption. Offer the recovered levels to luna as a closed vocabulary for that table.

## P6 — Cells-only pass

Ask luna for nothing but `effect.cells`, and derive or defer every other field. Tests
whether the entity and demand passes are load-bearing *for direction*, or whether their
value is elsewhere in the record.

## What is measurable without new luna runs

Direction accuracy, coverage and output size for P1, P4 and P5 are computable from artefacts
already on disk. P2 and P6 need a luna run to score, but their **token** side is computable
now from the record JSON. Numbers below distinguish measured from projected.
