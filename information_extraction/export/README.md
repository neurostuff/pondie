# Export Module

Purpose:
- Serialize `StudyRecord` outputs to JSONL (primary) and JSON (optional).
- Attach run metadata (schema version, model versions, timestamp).
- Provide file naming and output layout conventions.
- Combine per-document JSONL into an aggregated group JSONL at the end of a run.

Requirements:
- Consistent JSONL output shape matching `information_extraction/schema.py`.
- Include provenance and evidence spans.
- Write a run-level provenance sidecar manifest keyed by entity id + field path.
- Write outputs to local filesystem (MVP).
- Per-document filenames include pmid-doi-pmcid from identifiers.json.
- Per-document filenames omit missing IDs (e.g., pmid-doi) and fall back to the hash when needed.
- Aggregate JSONL output file includes one line per document.
- Aggregated JSONL is built from final normalized outputs.

Assumptions:
- Primary output is JSONL (one record per line).
- JSON output is optional and derived if needed.
- Downstream consumers read JSONL directly (no database required for MVP).
- Run metadata and field-level provenance are stored in a sidecar manifest per run (to avoid schema changes).
