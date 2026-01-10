# Preprocess Module

Purpose:
- Normalize whitespace/line breaks without breaking offsets.
- Expand abbreviations when needed while retaining an offset map.
- Clean text to improve extraction accuracy.

Requirements:
- Offset-preserving transformations with a reversible map.
- Optional abbreviation expansion controlled by configuration.
- Preserve section boundaries and locators.
- Use scispaCy AbbreviationDetector for per-article abbreviation expansion.

Assumptions:
- Abbreviation expansion is beneficial but not mandatory for all runs.
- Preprocessing writes a new normalized text file; source text remains unchanged.
- Evidence spans align to the normalized text produced for extraction.
- Abbreviation lists are built per article, not global.
