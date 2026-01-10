# Prompt Optimization Module

Purpose:
- Use DSPy to optimize prompt instructions for individual fields or field groups.
- Evaluate prompt variants against a JSONL gold set.
- Produce updated prompt files in a new output directory.

Requirements:
- DSPy integration with LangExtract-compatible prompts.
- Gold set loader (JSONL) and evaluation hooks.
- Output prompt files without overwriting originals.
- Update group-level prompt files and field-level prompt overrides.

Assumptions:
- Prompt files are stored per entity type and referenced from YAML config.
- Optimization is optional and runs via the `pondie optimize` subcommand.
- Summaries remain synthesized in attributes with verbatim extraction_text.
- Optimization can run per-field within each entity prompt file.
