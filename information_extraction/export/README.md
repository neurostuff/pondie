# Export Module

Purpose:
- Serialize `StudyRecord` outputs to JSONL (primary) and JSON (optional).
- Attach run metadata (schema version, model versions, timestamp).
- Provide file naming and output layout conventions.
- Combine per-document JSONL into an aggregated group JSONL at the end of a run.

Requirements:
- Consistent JSONL output shape matching `schema.py`.
- Include provenance and evidence spans.
- Write outputs to local filesystem (MVP).
- Per-document filenames include pmid-doi-pmcid from input directory names.
- Per-document filenames omit missing IDs (e.g., pmid-doi).
- Aggregate JSONL output file includes one line per document.
- Aggregated JSONL is built from final normalized outputs.

Assumptions:
- Primary output is JSONL (one record per line).
- JSON output is optional and derived if needed.
- Downstream consumers read JSONL directly (no database required for MVP).
- Run metadata is stored in a sidecar manifest per run (to avoid schema changes).
