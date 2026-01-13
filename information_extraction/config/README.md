# Config Module

Purpose:
- Centralize configuration for models, prompts, file paths, and run settings.
- Provide environment overrides for secrets (API keys).
- Define defaults for extraction passes and chunking.

Requirements:
- Typed config schema (Pydantic or dataclasses).
- Support for config files in YAML format.
- Secure handling of API keys via environment variables.
- Prompts/examples are stored in separate files referenced by YAML config.
- Prompt/example files are per entity type (e.g., `entity.json`) with grouped fields.
- Extraction parameters (extraction_passes, max_char_buffer, max_workers) live in YAML.
- Retrieval parameters (top_k, rerank_k, section_weight, embedding model) live in YAML.
- Incremental update flags (lock_existing_fields, reextract_on_prompt_change) live in YAML.
- LLM batching parameters (batch_size, max_batch_tokens) live in YAML when supported.
- Supported providers for v0.1: Gemini and OpenAI (explicitly selected per run).
- Prompt file format is defined in `prompts/README.md`.
- DSPy optimization settings (target fields, eval splits) live in YAML when used.
- Optional field-level prompt override files can be referenced in YAML.

Assumptions:
- Config values will be shared across modules.
- Sensitive values are never committed to version control.
