# pubget extraction bundle

Everything needed to extract pondie records for the pubget papers in `corpus/`, through
codex on `gpt-6.1-sol` at medium effort. `MANIFEST.json` records the code commit, the
schema commit, the date and the paper count.

## Requirements

- Python 3.12 (`uv` fetches it when installed; otherwise `python3.12 -m venv`)
- the codex CLI, logged in: `npm install -g @openai/codex`, then `codex login`
- disk for a second copy of the corpus: a run copies each paper into `runs/<RUN>/corpus`,
  because the `split` stage rewrites its parse

## Run

```bash
bash setup.sh                          # creates .venv, checks codex
LIMIT=5 bash run.sh                    # try five papers first
bash run.sh                            # then everything, 12 at a time
```

Shards share one run and can go one after another or in parallel:

```bash
OFFSET=0     LIMIT=5000 bash run.sh
OFFSET=5000  LIMIT=5000 bash run.sh
```

Re-running resumes: a paper's finished stages are skipped. `WORKERS`, `RUN`, `MODEL` and
`STAGE_EFFORT` can be set in the environment. A codex rate limit (429) is retried with
backoff and does not fail the paper; if many are logged, lower `WORKERS`.

## What comes out

`runs/<RUN>/` holds `records/<study>.extraction.json` (the records), `run.log`,
`events.jsonl` (per-stage timing), `payloads/` (each model reply) and `unrepaired/` (the
records before `repair`).

`Study.language` and `Study.study_type` are left unset: `build` fills them from PubMed only
when the study id is a PMID, and these are neurostore ids. `pubget.pmids` maps each study
to its PMID, so they can be filled afterwards with `pondie.extraction.pubmed`.

## Contents

| path | |
|---|---|
| `corpus/<study>/` | the paper: pubget text and table manifest, `stage1/analyses.json` (table and prose coordinates) and an untouched copy, `analyses.orig.json` |
| `pubget.pmids` | `pmid<TAB>study<TAB>pubget`, one line per paper |
| `bundle.jsonl` | every scanned folder: taken, or why not |
| `code/` | pondie at the commit in `MANIFEST.json`, with its schema in `code/study_schema` |
| `setup.sh`, `run.sh` | the two commands above |
