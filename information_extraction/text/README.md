# Text Module

Purpose:
- Extract raw text and section structure from documents.
- Use Docling (or similar) with OCR fallback for PDFs.
- Produce stable char offsets for evidence alignment.

Requirements:
- Preserve character offsets across text extraction.
- Capture section titles when available.
- Capture section hierarchy/path for use in retrieval and prompting.
- Include table/figure text in the main text stream when possible, with locators.
- Emit figure/table captions and table rows with locators for retrieval units.
- Produce Markdown output with pipe tables and a header row.
- Support a docling backend for PDF/HTML; XML continues to use native parsing.
- Insert external tables at their reference points when anchors are available.
- Abbreviation sidecar output uses scispaCy (install with `uv pip install -e ".[abbrev]"`).

Outputs:
- Markdown outputs live under `outputs/text/<run_id>/<hash>/<provider>/<document_id>.md`.
- Per-document metadata is stored alongside the markdown as `<document_id>.metadata.json`.
- Abbreviation sidecar JSON is stored alongside the markdown as `<document_id>.abbreviations.json`.

CLI:
- `pondie text --input-dir data --output-dir outputs/text`
- `pondie text --input-dir data --output-dir outputs/text --backend docling`

Assumptions:
- Docling provides stable offsets for extracted text.
- Section headers can be recovered or inferred reliably enough for prompting.
- Tables/figures can be merged into the main text stream without breaking offsets.
- Offsets are document-level (not page-level).
- Inputs are organized as <hash>/source/<provider> (ace/pubget/elsevier), with identifiers.json stored at the hash root.
