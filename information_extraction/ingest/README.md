# Ingest Module

Purpose:
- Load source documents from local paths (PDF, HTML, XML).
- Capture basic metadata (doc_id, filename, source type, size).
- Hand off raw bytes or file paths to the text extraction stage.

Requirements:
- File type detection for PDF/HTML/XML.
- Consistent document identifiers.
- Preserve source metadata needed for audit.
- Document IDs derived from identifiers.json in each hash directory (pmid/doi/pmcid), with missing IDs omitted and fallback to the hash when needed.

Assumptions:
- Inputs are local files provided in a user-specified directory.
- Remote URL ingestion is out of scope for now.
- Source files are not modified; the pipeline references them in place.
- OCR fallback is triggered downstream during text extraction.
- Inputs are organized as <hash>/source/<provider> (ace/pubget/elsevier), with identifiers.json stored at the hash root.
