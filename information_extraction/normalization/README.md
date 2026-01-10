# Normalization Module

Purpose:
- Map medical conditions and concepts to ONVOC.
- Normalize units and formatting where applicable.
- De-duplicate semantically identical labels.

Requirements:
- Access to ONVOC mappings and term lookup.
- Unit normalization rules (e.g., ms -> s, years vs months).
- Preserve evidence spans while normalizing values.
- ONVOC mapping for concepts, domain tags, and medical conditions.
- Normalized outputs add normalized values alongside originals (do not replace).

Assumptions:
- Normalization happens after extraction and before export.
- Evidence spans remain valid even if normalized values differ from verbatim text.
- Normalization outputs to a new file/directory; raw extraction output is preserved.
