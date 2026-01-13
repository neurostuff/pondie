# Review Module

Purpose:
- Provide human verification of extracted values against evidence.
- Generate review artifacts (e.g., LangExtract HTML visualizations).
- Capture reviewer feedback and corrections.
- Export corrected extractions for training data.

Requirements:
- Export review-ready JSONL for visualization.
- Record reviewer decisions and corrections per field.
- Enable re-integration of corrections into final outputs.
- Provide an editable LangExtract HTML view with JSONL export.

Assumptions:
- Human review is out-of-band and may be periodic.
- Evidence spans are sufficient to validate extracted values.
- Review starts with LangExtract HTML visualization.
- Reviewer edits are saved as updated JSONL in a separate output file.

Implementation notes:
- `information_extraction/review/editable_visualization.py` exposes
  `visualize_editable(...)` to render an editable LangExtract view with a
  download button that exports the corrected JSONL for training.

Example:
```
from information_extraction.review import visualize_editable

visualize_editable("outputs/review/run_001.jsonl")
```
