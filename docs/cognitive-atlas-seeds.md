# Seeding categories from the Cognitive Atlas: the list, and how well it matches

`pondie/normalization/atlas.py` normalises the Atlas task list into seed categories;
this is what it produces and how far it gets on the corpus.

**856 labels → 639 seeds.** 5 collapsed as method-word duplicates,
58 as variants of a parent, 15 kept separate against the
containment rule, and the rest dropped as instruments, generics or auto-generated entries.

## The rules, and which ones are hand-written

Automatic, and they carry most of the work:

| | |
|---|---|
| drop instruments | ends in `scale`, `inventory`, `questionnaire`, `battery`, `checklist` |
| drop auto-generated | `motor fMRI task paradigm`, `working memory fMRI task paradigm` — a generic head with the apparatus bolted on |
| collapse duplicates | identical once method words go: `Go-NoGo fMRI paradigm` ≡ `go/no-go task` |
| collapse variants | one core inside another: `Go-No-Go Zoo Task`, `letter n-back task`, `counting Stroop task` |
| refuse negations | `non-spatial cuing paradigm` is not a kind of `Spatial cuing paradigm` |
| pick the parent by position | earliest in the child, then longest |

That last one took two tries. Shortest-parent-first sent `Motor Selective Stop Signal Task`
to a motor label; longest-only left `Stop signal task with dot motion discrimination` on
`dot motion task`, both candidate cores being two tokens. The paradigm is named first and the
qualifier follows, so position breaks the tie — the same rule the ONVOC `contains` layer needed.

**Hand-written, because no rule works:**

- **An eponym list** (34 words: `iowa`, `cambridge`, `penn`, `benton`, `beery`, `uznadze`…). `Iowa Gambling Task` is
  not a kind of `gambling task` the way `Go-No-Go Zoo Task` is a kind of `go/no-go task`.
  Two automatic versions were tried and both failed: ranking by how rare a token is across
  the Atlas refuses the Zoo case (its tokens are two letters long) while still permitting Iowa.
- **A keep-separate list** (12 entries). `Space Fortress with Oddball` uses an oddball and is
  the Space Fortress paradigm; `finger tapping task` is not `block tapping test`. Read off all
  84 automatic collapses, which is the only way to find them.

## How well it matches the corpus

Over the 1,672 tasks the clusterer sees:

| layer | mentions | |
|---|---|---|
| exact (after stimulus-stripping) | 235 | 14.1% |
| core containment | 220 | 13.2% |
| squashed, at a word boundary | 13 | 0.8% |
| **no match** | **1,204** | **72.0%** |

**468 of 1,672 mentions (28%) match, into 65 categories**; 272 of 1,159 distinct names.

**Precision is high.** Hand-judged on a random sample of 40 matches: 1 clear error
(`Facial emotion recognition` → `fMRI Facial Emotion Paradigm`, which should be dropped as
auto-generated and is not caught by the current pattern) and 1 arguable (`regulation task` →
`Emotion Regulation Task`). Nine separate go/no-go spellings — `Go/NoGo`, `go-nogo`,
`Go/No-go monetary reward paradigm`, `food-specific go/nogo tasks` — all land on
`go/no-go task`, which is what the normalisation was for.

So this is a **high-precision, low-recall seeder**, which is the right shape for a seed stage:
what it claims is nearly always right, and the 72% it declines goes to clustering.

## The ceiling is the vocabulary, not the matcher

Of the 1,204 unmatched mentions, **53% are families the Atlas has no term for at all**:

| mentions | family |
|---|---|
| 313 | cue reactivity / exposure / induction |
| 189 | resting state |
| 105 | picture, image, film viewing |
| 25 | food-specific |
| 12 | craving and its regulation |

No amount of matcher work reaches those. The largest genuinely-unmatched names outside those
families are `modified Sternberg working memory paradigm`, `facial expression processing`,
`emotional conflict task` and `fear conditioning and extinction paradigm` — a handful of
studies each, and the long tail of one-paper names that clustering exists to handle.

