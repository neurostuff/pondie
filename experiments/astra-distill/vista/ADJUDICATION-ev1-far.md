# Adjudication: ev1-1k evidence units far from Astra's (val, 481 papers)

`evfar.py val-pred-full.jsonl 60` (seed 11): 60 slots, from 30 papers, where both sides cite evidence and the model
cites a unit more than 10 units from every unit Astra cites. Those units are 47% of the model's false positives
(31.3k of 66.7k, `evbreak.py`). Each was read against the slot's value: does the model's far unit state it?

| verdict | n | share [95% CI, Wilson] |
|---|---:|---|
| states the value (a second valid place) | 39 | 0.65 [0.52, 0.76] |
| partly supports it | 6 | 0.10 |
| does not state it | 15 | 0.25 [0.16, 0.37] |

- **Valid, typical:** the same fact stated twice. Group sizes in the abstract and the methods (10, 24, 28, 31, 50),
  a table header or footer printed twice (7, 37, 41), a condition label in the design paragraph and in a contrast
  (18, 21, 22, 34, 36, 38, 54, 56), a result in the text and in its table (11, 32, 43).
- **Partial:** the unit backs part of a long generated value (a `model_settings` or `definition` summary: 9, 14, 46,
  57) or implies it (45: "one participant excluded" for n = 31).
- **Wrong:** a nearby-topic sentence that does not state the value (2, 13, 19, 42, 48, 51), a contradicting one (4
  and 44 cite Z-statistic sentences for family `t`), a table row of another coordinate set (15, 40), and
  `prespecification: exploratory` from sentences that never say exploratory (16, 29). In 5 of the 15 (2, 5, 19, 26,
  51), Astra's own unit does not state the value either.

**Reading.** Most far-away false positives are a second place the paper states the value, which a single-reference
metric counts as wrong. If 65–75% of the 31.3k are valid, about 20–23k of the 66.7k false positives are not errors:
precision would be about 0.81–0.83 rather than 0.73 (recall is unchanged, since Astra's own unit is still missed).
The sample is 30 papers, one of which (FMXAJXKN3akG) gives 10 items.

Verdicts by item number (as `evfar.py` prints them):
- states it: 1 3 6 7 8 10 11 12 17 18 20 21 22 23 24 25 27 28 31 32 33 34 35 36 37 38 39 41 43 47 49 50 52 53 54 55
  56 59 60
- partly: 9 14 45 46 57 58
- does not: 2 4 5 13 15 16 19 26 29 30 40 42 44 48 51
