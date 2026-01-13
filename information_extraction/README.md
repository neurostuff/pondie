# Package Overview

This package contains the pipeline modules for neuroimaging article information extraction.

Modules:
- `schemas`: Pydantic schema definitions and schema versioning.
- `ingest`: input loading and document metadata capture.
- `text`: text extraction and section recovery (Docling/OCR).
- `preprocess`: offset-preserving normalization, abbreviation expansion, and retrieval unit prep.
- `extraction`: LangExtract prompts/examples and extraction runs (retrieval+rerank, batching, incremental).
- `linking`: StudyLinks edge creation.
- `validation`: evidence and consistency validation.
- `normalization`: ONVOC mapping and unit normalization.
- `review`: human verification artifacts, editable LangExtract review, and feedback capture.
- `export`: JSONL serialization and run metadata.
- `pipeline`: orchestration, incremental updates, and run control.
- `evaluation`: gold set evaluation and metrics.
- `config`: shared configuration and secrets management.
- `optimization`: DSPy-based prompt optimization.
