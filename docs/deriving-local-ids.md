# Deriving local_ids, so a reference survives a re-extraction

A `local_id` is an address. The review layer uses it as one directly --
`paper|value|Analysis|<local_id>|<path>` -- and `tasks.py` states the consequence: "a
vanished address orphans the answer". An id invented by the model is an address that moves
when nothing about the paper has.

It moves a lot. Over the same sixteen papers extracted twice, **only four produced
identical analysis ids**:

| paper | one run | the other |
|---|---|---|
| 6oTrCJA43Jcd | `a_ic25`, `a_ic30`, `a_ic35` | `a_independent_component_spatial_maps` |
| 84rGLhCbUJTh | `a_fa_group`, `a_rd_group`, … | `a_fa`, `a_rd`, … |
| DTpwdoGbjqsq | `a_tbl2`, `a_tbl3`, … | `a_tbl2_ad_hc_gm`, `a_tbl3_ad_hc_pk`, … |

In the first the *count* differs too, so the ids cannot correspond even in principle.

## What is derived now

`build_record.derive_analysis_ids` renames every analysis to `a_<table id>_<ordinal>`,
taken from `source_table_analysis` -- a key the table parse determines and the model only
copies. Safe because nothing in the schema references an Analysis by id: `Study.analyses`
is the only slot with that range and it inlines them, so the only pointer to follow is
`mirror_of`.

An analysis with no key keeps the model's id. That is 25% of them, and it is the honest
outcome -- the parse does not identify that row group, and a stable-looking id would claim
it does.

## What the prompt asks for, and what it must not

The prompt previously said only that a `local_id` is "a bare string you assign, unique
within its class", and demonstrated `t_stimulus` / `m_first_level` / `r_ffa` by example.
A convention shown in an example and not stated as a rule is a convention the model
follows sometimes, which is what the prefix table above measures.

It now states the prefix per class and the principle -- an id is an address, use the
shortest thing the *paper* fixes, never a phrase you compose -- with `acq_fmri` against
`acquisition_resting_state_bold` as the contrast.

The split between asking and deriving is deliberate, and it follows the one clean datum
this pipeline has on prompt changes. An unconditional instruction to emit *more* cost 7.4
points of direction accuracy and took `sign_loss` from 0 to 8
(docs/interaction-simple-effects.md). So:

- **Ask** only for the *form* of an identifier. It constrains naming, not extraction: no
  value changes, no field is added or dropped, and the worst case is an id the builder
  then normalises anyway.
- **Derive** anything with a real anchor, at build time, where compliance is not a
  variable. `Analysis` and `Table` are derived and the prompt explicitly tells the model
  not to choose them.

Even the form-only change wants checking on the gold bench before it is trusted, for the
same reason: this pipeline's prompt changes have not been reliably harmless.

## The pattern, for the rest

Two questions decide whether an entity's id can be derived, and they are separate:

1. **Is there a source-determined anchor?** Something the paper fixes, not the model.
2. **Is it unique within the paper?** If not, the anchor needs a positional tiebreak.

An id should be `<prefix>_<folded anchor>` and fall back to `<prefix>_<ordinal>` when the
anchor is missing -- never to a model-chosen phrase, because that is the variance being
removed. The prefix should be fixed per class; today it is not, which is measurable:

| list | distinct id prefixes seen | anchor available |
|---|---|---|
| `groups` | 3 — `grp`, `g`, `group` | `name` 56/56 |
| `tasks` | 1 — `task` | `name` 28/28 |
| `acquisitions` | 2 — `acq`, `me` | `modality` 41/41 |
| `preprocessings` | 3 — `prep`, `preproc`, `pp` | `software` 31/31 |
| `model_estimations` | 3 — `m`, `model`, `me` | `model_family` 64/64 |
| `assessments` | 1 — `asmt` | `name` 61/61 |
| `measures` | 3 — `me`, `measure`, `meas` | `type` 50/50 |
| `regions` | 2 — `reg`, `r` | `name` 42/42 |
| `inference_settings` | 3 — `inf`, `inference`, `is` | `multiple_comparison_method` 51/51 |
| `devices` | 2 — `device`, `dev` | `manufacturer` 42/42 |

Every anchor is filled on every entity. The prefix inconsistency is therefore pure variance with
nothing behind it, and fixing the prefix alone removes a third of the resolution problem
for free.

## Per class, and how strong the anchor is

**Already deterministic.** `Table` — the parse manifest supplies the id, and `Tables`
copies it. `Analysis` — done, above. Nothing else has a parse-supplied anchor.

**Strong: an anchor the paper states in a closed vocabulary.** These can be derived with
no text handling at all, because the value is a schema enum.

- `acquisitions` → `acq_<modality>` (`acq_fmri`, `acq_dwi`), ordinal when a paper has two
  of a modality. Already the commonest style.
- `measures` → `mea_<family>_<type>` (`mea_functional_bold_bold_response`), which also
  makes two measures of the same kind visibly the same kind.
- `model_estimations` → `mod_<model_family>` plus an ordinal, since a paper routinely has
  two GLMs. Better still, once a model's stage is known: `mod_l1` / `mod_l2` from
  `inputs_from` depth, which is what a reader actually wants to distinguish.
- `inference_settings` → `inf_<multiple_comparison_method>_<threshold>`.
- `design.arms` → `arm_<arm_kind>` plus the agent when there are two of a kind.

**Medium: an anchor that is free text but short and repeated.** Fold to lowercase
alphanumerics and truncate; the risk is two entities folding together, so a positional
suffix is required on collision rather than optional.

- `groups` → `grp_<folded name>`. Watch the case this corpus is full of:
  `unaffected siblings`, `unaffected siblings with past depression` and
  `unaffected siblings with no past personality or mood disorder` fold to the same stem
  and are three different cohorts. Truncation must not be the disambiguator.
- `assessments` → `asm_<abbreviation>` where the paper defines one. The abbreviation store
  (`data/vocab/abbreviations.json`, 361 entries mined by scispacy) already resolves
  `MADRS`, `PANSS`, `ADOS` from the paper's own `long form (SF)` definitions, and an
  abbreviation is far more stable than the spelled-out name, which appears in four
  variants across a corpus.
- `regions` → `reg_<folded name>`, and this is the one class where a *shared* vocabulary is
  reachable: ONVOC's Cortical and Subcortical Regions branches matched 44% of region names
  by string alone. An ONVOC id would make the region id stable across papers, not just
  within one -- the only class where that is currently in reach.
- `tasks` → `tsk_<folded name>`, with the Cognitive Atlas as the same kind of opportunity:
  857 task labels, and `tasks.name` matched 38%.

**Weak: no anchor the paper fixes.** Derive positionally and say so.

- `model_estimations[].terms` → the term's name is the anchor, and
  `scope_duplicate_terms` already scopes it as `<model>.<term>` when two models share a
  name. That is the pattern the others should follow: qualify by owner rather than
  renaming.
- `design.timepoints` → `tp_<ordinal>` in the order the paper reports them. A timepoint's
  label (`week 8`, `post-treatment`) is more readable but far less stable.

## Two rules worth stating once

**Qualify, do not rename.** `scope_duplicate_terms` makes a colliding term
`<model>.<term>` rather than picking a winner, which keeps both resolvable. That
generalises: when an anchor is ambiguous, prefix the owner.

**Never derive an id from a value the model chose.** That is the whole failure being
fixed. An id built from a model-written `name` is only as stable as the naming, and the
table above shows the naming is not stable. Prefer an enum, then an abbreviation the paper
defines, then a position -- in that order.

## What this does not fix

A derived id is stable across re-extractions of the same paper. It is still paper-scoped:
`grp_healthy_controls` in two papers are two different cohorts. Cross-paper identity needs
a shared vocabulary, which is `docs/normalizing-across-papers.md`, and only `regions` and
`tasks` currently have one within reach.
