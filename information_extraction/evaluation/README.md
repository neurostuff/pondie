# Evaluation Module

Purpose:
- Measure extraction quality against a gold set.
- Compute coverage and accuracy metrics for fields and links.
- Compare results across model/prompt/schema versions.
- Support DSPy prompt optimization loops with evaluation metrics.

Requirements:
- Gold set format definition.
- Metrics for evidence coverage, field accuracy, and link correctness.
- Reporting outputs (CSV/JSON summaries).

Assumptions:
- A curated gold set will be available for evaluation.
- Evaluation is separate from the main extraction pipeline.
- Gold set is JSONL created by manual edits to workflow outputs.

Acceptance Metric:
- F1 score is the primary acceptance metric for v0.1.
