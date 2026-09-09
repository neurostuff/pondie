{{block}}
→ **`simple_effect`**. One signed cell, on the task the classifier was run within — not one cell per
decoded class. The classes are `decoded_variable`'s business, and cell-ing them would say the
comparison was between them.

What the accuracy was compared *to* is `reference_value`, per metric, because a study reporting
accuracy against chance and AUC against 0.5 has two references and one field on the Effect could
not hold both. It is unset here, correctly: this paper tests against chance by permutation and
never states a chance figure, so `relation` carries the claim alone.

A between-cohort comparison of accuracies is a crossed cohort term (5.1's shape). Accuracy regressed
on a behavioural score is 5.3's shape. The `Effect` does not change form because the method did — the
method lives in `details`.

**In the paper.**

> Specifically, patches of cortex in inferior frontal and superior temporal regions retained information to significantly discriminate the seven vowels of the Italian language in each condition.
>
> A cross-validation leave-one-stimulus-out procedure was adopted to measure classification accuracy.
>
> To assess significance, group accuracies were tested against chance by a permutation test

**Referent** `3jDCyBsgwY5d` (pmid 29208951) — Vowel decoding across listening, imagery and
production. Transcribed, not corrected: the extractor produced three decoding analyses, each a
single signed cell, which is the schema's own `a-classifier-signs-one-cell-not-two` rule obeyed
without prompting.

Five decoding papers were extracted before this one. Two produced no `DecodingDetails` at all,
one produced no analyses, and one cell-ed **both** classes `undirected` — which derives `omnibus`
and asserts the classes were compared with each other. That last failure is the one this example
is written against, and it is why the rule is worth stating.
