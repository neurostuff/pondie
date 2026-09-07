# Notes for an agent working on this repository

Material an agent needs to avoid repeating work, kept out of `docs/` because none of it
describes the package to a user: it describes what was measured, what was tried and
discarded, and the traps that cost time.

## `repair/`

Why the repair pass was rewritten, and what it still gets wrong.

| file | what it is |
|---|---|
| `investigation.md` | the diagnosis, the metrics, the results, and the corrections to all three |
| `review-notes.md` | the adversarial review's own record, including where it corrected itself |
| `delta.py` | what a repair pass did to a record: spans, provenance, findings, fills |
| `references.py` | the reference slots, which carry no wrapper and which `delta.py` cannot see |

Both run against a run directory holding `records/` and `unrepaired/`.

Both measure whether the pass DAMAGED the record. Neither can say whether it HELPED: that
needs papers read by hand, and the four this work used are not in the repository. The
content numbers quoted below and in `investigation.md` came from them and cannot be
reproduced here -- treat them as recorded findings, not as a suite you can re-run.

## The one thing worth reading before changing anything here

On the fields it changes, the repair pass is wrong about half the time, and every one of
those errors is a real fact from the paper attached to the wrong entity -- an excluded
patient's drug as the cohort's medication, a subgroup's mean age as a group's, the analysed
count in the enrolled slot. All of it is grounded, so no evidence check can see it.
`extraction_metadata.repaired_by` is what makes those fields findable.

And the rule this work learned four separate times: **a verbatim test is only valid against
a value the paper was supposed to have printed.** A minted id, a derived label, a model's
paraphrase and a synthesised mirror are none of them quotations, and asking "is this string
in the article" of any of them measures the record's vocabulary rather than the paper's
content.
