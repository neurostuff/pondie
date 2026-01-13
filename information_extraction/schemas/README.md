# Schemas Module

Purpose:
- Define the Pydantic schema used for extraction (e.g., `StudyRecord`, `ExtractedValue`).
- Maintain schema versioning and compatibility with extraction metadata.
- Provide JSON schema exports when needed.

Requirements:
- Preserve `ExtractedValue` wrappers and evidence requirements.
- Support schema evolution without breaking downstream consumers.
- Expose stable IDs and entity types for linking.
- Record schema version in run metadata and output filenames (not in the `StudyRecord` payload).
- Record field-level provenance in a run-level sidecar manifest keyed by entity id + field path.

Assumptions:
- Entity IDs are injected by pipeline logic (not LLM generated).
- Evidence spans are required for any non-null values.
- Schema changes will be versioned and tracked in a run-level manifest.
