# Linking Module

Purpose:
- Build explicit StudyLinks edges across entities:
  - group_task, task_modality, group_modality
  - analysis_task, analysis_group, analysis_condition
- Ensure links are grounded in text evidence.

Requirements:
- Use LLM inference to propose edges, but require explicit textual support.
- Attach evidence spans to each edge list.
- Capture per-edge evidence when available; otherwise fall back to list-level evidence.
- Support linking across multiple tasks/groups/modalities.

Assumptions:
- Entity IDs are stable and injected by the pipeline.
- Links are created only when the text supports them.
