# Config Files

YAML configuration files for pipeline runs live in this directory.
They should reference:
- input/output directories
- model/provider selection (Gemini/OpenAI)
- LangExtract parameters (extraction_passes, max_char_buffer, max_workers)
- retrieval parameters (top_k, rerank_k, section_weight, embedding model)
- incremental update flags (lock_existing_fields, reextract_on_prompt_change)
- LLM batching parameters (batch_size, max_batch_tokens) when supported
- prompt file paths (per entity)
- caching options
- run metadata (schema version, prompt version, etc.)
