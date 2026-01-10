# Config Files

YAML configuration files for pipeline runs live in this directory.
They should reference:
- input/output directories
- model/provider selection (Gemini/OpenAI)
- LangExtract parameters (extraction_passes, max_char_buffer, max_workers)
- prompt file paths (per entity)
- caching options
- run metadata (schema version, prompt version, etc.)
