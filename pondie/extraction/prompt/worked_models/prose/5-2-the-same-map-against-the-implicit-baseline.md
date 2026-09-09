{{block}}
→ **`simple_effect`**. One signed cell and no second one is not an incomplete record: a lone `+1`
weight tests that coefficient against zero, and zero is what the unmodelled implicit baseline is. The
sign is what distinguishes activation from deactivation, which is why this is not encoded as "no
cells".

If the paper contrasted against a *modelled* rest condition instead, `rest` takes a `negative` cell
and the result is 5.1 — a `contrast`.

**In the paper.**

> Emotion labeling and emotion matching were also compared with implicit baseline (consisting of unmodeled fixation events during the intertrial intervals).

**Referent** `4CA3Ca2bzfPW` (pmid 25821147), 5.1's paper and 5.1's model — "Emotion Labeling >
Baseline", where the baseline is stated to be "unmodeled fixation events during the intertrial
intervals". Both maps come off the one design matrix, which is why they are one
`ModelEstimation` and two `Analysis` records.
