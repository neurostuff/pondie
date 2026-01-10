# Ingest Module

Purpose:
- Load source documents from local paths (PDF, HTML, XML).
- Capture basic metadata (doc_id, filename, source type, size).
- Hand off raw bytes or file paths to the text extraction stage.

Requirements:
- File type detection for PDF/HTML/XML.
- Consistent document identifiers.
- Preserve source metadata needed for audit.
- Document IDs derived from directory names (pmid-doi-pmcid).

Assumptions:
- Inputs are local files provided in a user-specified directory.
- Remote URL ingestion is out of scope for now.
- Source files are not modified; the pipeline references them in place.
- OCR fallback is triggered downstream during text extraction.
- Directory names omit missing IDs (e.g., pmid-doi or pmid-pmcid).
