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
