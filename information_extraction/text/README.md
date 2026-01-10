# Text Module

Purpose:
- Extract raw text and section structure from documents.
- Use Docling (or similar) with OCR fallback for PDFs.
- Produce stable char offsets for evidence alignment.

Requirements:
- Preserve character offsets across text extraction.
- Capture section titles when available.
- Include table/figure text in the main text stream when possible, with locators.

Assumptions:
- Docling provides stable offsets for extracted text.
- Section headers can be recovered or inferred reliably enough for prompting.
- Tables/figures can be merged into the main text stream without breaking offsets.
- Offsets are document-level (not page-level).
