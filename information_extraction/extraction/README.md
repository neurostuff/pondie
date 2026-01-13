# Extraction Module

Purpose:
- Run LangExtract with schema-aligned prompts and examples.
- Populate `ExtractedValue` objects with evidence spans in one step.
- Respect per-field `inference_policy` (explicit, synthesize, post_normalize).

Requirements:
- Prompt library aligned to `information_extraction/schema.py` fields and entity types.
- Few-shot examples that enforce verbatim extraction for evidence alignment.
- Support for chunking, multiple passes, and model selection.
- Accept top-k contexts from the retrieval+rerank stage when enabled.
- Support field-level incremental extraction with a needs-update mask and a lock flag.
- If an entity prompt version changes, re-extract the full entity (skip incremental).
- Support batched LLM API submissions when the provider allows it.
- Model/provider must be specified per run (no default).
- Prompt/example files are per entity type (e.g., `entity.json`) and cover grouped fields.
- For summary fields, store synthesized summaries in `attributes` while keeping `extraction_text` verbatim.

Assumptions:
- Summaries may be synthesized but must cite supporting evidence spans.
- Entity IDs are injected after extraction.
- LangExtract is the primary extraction engine.
- Prompts are organized per entity type (not per field group).
- Existing values/evidence may be provided during incremental runs and are trusted unless re-extracted.
