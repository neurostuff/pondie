# vista/: data and evaluation harness for the Vista phase (one call per paper)

CPU only. Local: `experiments/astra-distill/vista/`; beast: `/data/james/agents/astra-distill/code/vista/`.
Beast env: `/data/james/agents/astra-distill/vista-venv` (jsonschema, numpy, scipy, rapidfuzz, transformers, tokenizers,
jinja2; no torch). Data v5 is the default (`migrate/train/records`, `migrate/run-old/parse`, `migrate/splits`); every
script takes `--records --parse --text --schema --splits --stage1`. The test split is refused by `tasks.py` and `score.py`.

```
PY=/data/james/agents/astra-distill/vista-venv/bin/python; cd /data/james/agents/astra-distill/code/vista

$PY compact.py roundtrip ../../migrate/in/extraction-record.schema.json ../../migrate/train/records   # compact <-> record
$PY tasks.py build --split train.txt --split val.txt --out ../../vista-out/v5/all [--max-tokens N]     # PREFIX.jsonl
$PY tasks.py stats --split train.txt                  # sizes only (add --manifest f.jsonl for per-paper counts)
$PY tasks.py schemas --out-dir schemas                # generic schema-record.json / template-record.json
$PY score.py selftest --split val.txt                 # gold vs gold, gold vs perturbed gold
$PY score.py run PREDS.json --split val.txt --out score.json
$PY score.py compare score_A.json score_B.json --out diff.json   # paired paper bootstrap (2,000) of B - A
$PY infer.py run --split val.txt --out preds.json --url http://HOST:PORT/v1/chat/completions --model NAME
```

## The task

One call per paper. **Input** (a single user turn, NuExtract3's structured format with the record template): an
instruction, the paper text, then the parse listing: every parse entry in document order with key, origin, table id, name,
description, space, table caption (300 chars) and footer (700) from `stage1/analyses.json` (mapped to parse keys with
`migrate/code/oldparse.py`: `stage1.py`), and all its points with statistic and cluster size. **Target** (the assistant
turn): `to_compact` of the pruned record without the root `local_id`: entity collections, design, `coordinate_sets`
(`local_id` = a parse key), analyses (`source_table_analysis` = a parse key), model_estimations, tables. Loss on the
assistant turn only, token-sum (the trainer adds EOS after `target_json`).

Row: `{"sid", "call": 1, "key": "paper", "prompt", "template", "target", "target_json", "parse_keys", "n_tokens",
"n_input_tokens", "n_listing_tokens", "n_target_tokens", "n_text_tokens"}`. `n_tokens` is exact (input as rendered by the chat
template, plus the target and EOS). `--max-tokens N` leaves papers over N out of the JSONL (default: none); stats are
always over all papers.

Guided decoding: one JSON schema for the whole record (`tasks.record_shape(ix, "schema", parse_keys)`): `coordinate_sets[].local_id`
and `analyses[].source_table_analysis` are enums of the paper's listing keys; every other reference is a free string
(ids the output creates), checked after decoding by `reference_resolution`. Only plain required strings are required
(`local_id`, a cell's `term`, ...); open vocabularies are strings (the enum is a hint in the template). `infer.py` is the
one-request-per-paper driver (request body printable with `infer.py prompt SID`; not run against a server).

## Compact form (compact.py)

`to_compact`: ExtractedValue wrappers removed; `not_reported`, null, empty list/object omitted; `evidence`, `value_source`,
`unreported_reason`, `stable_id`, `extraction_metadata`, `Task.paradigm`, `Task.variant_qualifiers`, `Table.title`,
`Table.description` dropped; `local_id`s and references kept. `from_compact` re-wraps (`generated` / `not_found`; absent
wrapped slots `not_reported` / `not_applicable`).

## Scorer (score.py)

Whole record once per paper; prediction file `{"record": {sid: compact record | null}}`. Effects and model slot per parse key
(analyses of a key paired by name, Hungarian; entries with points and no analysis score cells as false positives), entity F1
per class (Hungarian on names, `ent/` similarities, 0.45), compared entities, entity slots, validity (`guided_valid`,
`valid_record`), `reference_resolution`, `listing_keys_share`. Effect columns strict / local / linked (primary) as in
`e2e2/score_e2e2.py` (see the module docstring for the exact rules). 95% CIs: paper-level percentile bootstrap, 2,000.
Not ported: the `inputs_from` / term-link edge scorers, table content, evidence. Not numerically cross-checked against the old
scorer (different data format); validated by the self-test.

Not used any more (earlier two-call design): `assemble.py` (only `reference_stats`).
