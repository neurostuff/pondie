# Preprocess Module

Purpose:
- Normalize whitespace/line breaks without breaking offsets.
- Expand abbreviations when needed while retaining an offset map.
- Clean text to improve extraction accuracy.
- Emit sentence units and section metadata needed for per-document retrieval.

Requirements:
- Offset-preserving transformations with a reversible map.
- Optional abbreviation expansion controlled by configuration.
- Preserve section boundaries and locators.
- Emit sentence/caption/table-row units with section path metadata and char offsets.
- Preserve sentence boundaries after normalization for retrieval + evidence alignment.
- Use scispaCy AbbreviationDetector for per-article abbreviation expansion.

Assumptions:
- Abbreviation expansion is beneficial but not mandatory for all runs.
- Preprocessing writes a new normalized text file; source text remains unchanged.
- Evidence spans align to the normalized text produced for extraction.
- Abbreviation lists are built per article, not global.
- Retrieval embeddings are computed from normalized text with section headings stored separately.
