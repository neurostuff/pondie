# Extraction Module

Purpose:
- Run LangExtract with schema-aligned prompts and examples.
- Populate `ExtractedValue` objects with evidence spans in one step.
- Respect per-field `inference_policy` (explicit, synthesize, post_normalize).

Requirements:
- Prompt library aligned to `schema.py` fields and entity types.
- Few-shot examples that enforce verbatim extraction for evidence alignment.
- Support for chunking, multiple passes, and model selection.
- Model/provider must be specified per run (no default).
- Prompt/example files are per entity type (e.g., `entity.json`) and cover grouped fields.
- For summary fields, store synthesized summaries in `attributes` while keeping `extraction_text` verbatim.

Assumptions:
- Summaries may be synthesized but must cite supporting evidence spans.
- Entity IDs are injected after extraction.
- LangExtract is the primary extraction engine.
- Prompts are organized per entity type (not per field group).
