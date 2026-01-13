# Pipeline Module

Purpose:
- Orchestrate end-to-end runs across modules.
- Manage run configuration, checkpoints, and error handling.
- Coordinate multi-pass extraction and linking.

Requirements:
- Configurable step ordering and re-run capability.
- Clear logging and run metadata capture.
- Support for batch processing multiple documents.
- CLI entry point for running the pipeline.
- Parallelization across documents with progress bars (tqdm).
- Optional per-document retrieval + rerank stage using sentence and section embeddings with weighted scoring.
- Incremental update workflow: compute needs-update mask, apply lock flag, and re-extract full entities when prompt versions change.
- Merge incremental outputs while recording field-level provenance in a run-level sidecar manifest keyed by entity id + field path.
- Support batching of LLM API submissions when the provider allows it.
- Intelligent caching of intermediate outputs.
- Cache keys include input hash, config, schema version, prompts, and model id (stage-specific).
- Cache state tracked in an SQLite database for the extraction stage (cache.sqlite).

Assumptions:
- Pipeline runs locally for MVP.
- Steps can be re-run without destructive side effects.
- Each transformation writes outputs to a new file/directory per run.
- CLI name is `pondie` and uses workflow stages as subcommands (ingest/text/preprocess/extract/link/normalize/export/review/evaluate/optimize).
- Cache invalidation occurs via manual purge or hash-key changes (no TTL).
- `cache.sqlite` lives under a per-hash cache directory for extraction (e.g., `cache/<hash>/cache.sqlite`).
- Reused evidence spans are treated as trusted unless a field is re-extracted.
