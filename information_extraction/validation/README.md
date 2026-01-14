# Validation Module

Purpose:
- Enforce schema constraints after extraction.
- Validate evidence requirements for all non-null values.
- Run consistency checks across related fields.

Requirements:
- Pydantic validation of `StudyRecord` outputs.
- Evidence presence for any non-null `ExtractedValue`.
- Optional consistency checks (e.g., age range vs min/max).
- Emit warnings per article when validation issues are detected; still write JSONL if schema-valid.
- Persist per-article warnings to a JSONL sidecar file.

Assumptions:
- Validation can flag errors without blocking early experiments.
- Some inconsistencies will be resolved in post-processing.
- Warnings are logged per-article for later review.
- No consistency checks are mandatory for MVP.
- Warning sidecar filenames follow `pmid-doi-pmcid.warnings.jsonl` using identifiers.json (omitting missing IDs and falling back to the hash when needed).
