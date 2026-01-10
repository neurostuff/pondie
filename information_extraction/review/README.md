# Review Module

Purpose:
- Provide human verification of extracted values against evidence.
- Generate review artifacts (e.g., LangExtract HTML visualizations).
- Capture reviewer feedback and corrections.

Requirements:
- Export review-ready JSONL for visualization.
- Record reviewer decisions and corrections per field.
- Enable re-integration of corrections into final outputs.

Assumptions:
- Human review is out-of-band and may be periodic.
- Evidence spans are sufficient to validate extracted values.
- Review starts with LangExtract HTML visualization.
- Reviewer edits are saved as updated JSONL in a separate output file.
