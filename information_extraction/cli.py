from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json

from information_extraction.text.extractor import (
    build_document_id,
    discover_documents,
    extract_abbreviations,
    extract_markdown_from_path,
)
from information_extraction.extraction import ExtractionConfig, run_full_extraction


def main() -> None:
    parser = argparse.ArgumentParser(prog="pondie")
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text", help="Extract text into markdown")
    text_parser.add_argument("--input-dir", default="data", help="Input directory with hash folders")
    text_parser.add_argument(
        "--output-dir",
        default="outputs/text",
        help="Root output directory for text extraction",
    )
    text_parser.add_argument("--run-id", default=None, help="Run identifier (default: timestamp)")
    text_parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Limit to providers (ace, pubget, elsevier)",
    )
    text_parser.add_argument(
        "--backend",
        choices=["native", "docling"],
        default="native",
        help="Extraction backend for HTML/PDF (XML always uses native).",
    )
    text_parser.add_argument("--limit", type=int, default=None, help="Max documents to process")
    text_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have markdown output",
    )

    extract_parser = subparsers.add_parser("extract", help="Run LangExtract pipeline")
    extract_parser.add_argument(
        "--input-path",
        help="Path to a single extracted text JSON/MD file.",
    )
    extract_parser.add_argument(
        "--input-dir",
        help="Directory containing extracted text JSON/MD files.",
    )
    extract_parser.add_argument(
        "--prompt-dir",
        default="prompts",
        help="Directory containing per-entity prompt files.",
    )
    extract_parser.add_argument(
        "--output-dir",
        default="outputs/extraction",
        help="Root output directory for extraction runs.",
    )
    extract_parser.add_argument("--run-id", default=None, help="Run identifier.")
    extract_parser.add_argument(
        "--model-id",
        default="gemini-2.5-flash",
        help="Model ID to use for LangExtract.",
    )
    extract_parser.add_argument("--batch-length", type=int, default=10)
    extract_parser.add_argument("--max-workers", type=int, default=10)
    extract_parser.add_argument("--extraction-passes", type=int, default=1)
    extract_parser.add_argument("--temperature", type=float, default=None)
    extract_parser.add_argument(
        "--no-reference-strip",
        action="store_true",
        help="Do not attempt to strip references/bibliography.",
    )

    args = parser.parse_args()
    if args.command == "text":
        run_text_extraction(args)
    if args.command == "extract":
        run_extraction_pipeline(args)


def run_text_extraction(args: argparse.Namespace) -> None:
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir) / run_id

    documents = discover_documents(input_dir, providers=args.providers)
    if args.limit:
        documents = documents[: args.limit]

    for doc in documents:
        doc_id = build_document_id(doc.identifiers, doc.hash_id)

        output_dir = output_root / doc.hash_id / doc.provider
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / f"{doc_id}.md"
        abbreviation_path = output_dir / f"{doc_id}.abbreviations.json"
        if args.skip_existing and markdown_path.exists():
            continue
        markdown = extract_markdown_from_path(
            doc.path, doc.table_paths, backend=args.backend
        )
        markdown_path.write_text(markdown, encoding="utf-8")

        metadata = {
            "document_id": doc_id,
            "hash_id": doc.hash_id,
            "provider": doc.provider,
            "source_path": str(doc.path),
            "identifiers": doc.identifiers.raw,
        }
        metadata_path = output_dir / f"{doc_id}.metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

        abbreviations = extract_abbreviations(markdown)
        abbreviation_payload = {
            "document_id": doc_id,
            "abbreviations": [
                {
                    "short": entry.short,
                    "long": entry.long,
                    "definition_count": entry.definition_count,
                    "count": entry.count,
                }
                for entry in abbreviations
            ],
        }
        abbreviation_path.write_text(
            json.dumps(abbreviation_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        print(f"wrote {markdown_path}")


def _collect_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_path:
        return [Path(args.input_path)]
    if args.input_dir:
        root = Path(args.input_dir)
        if not root.exists():
            raise FileNotFoundError(f"Input directory not found: {root}")
        paths = []
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            if path.name.endswith(".metadata.json"):
                continue
            paths.append(path)
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"No input files found in: {root}")
        return paths
    raise ValueError("Provide --input-path or --input-dir for extraction.")


def run_extraction_pipeline(args: argparse.Namespace) -> None:
    input_paths = _collect_input_paths(args)
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    config = ExtractionConfig(
        model_id=args.model_id,
        prompt_dir=Path(args.prompt_dir),
        output_dir=Path(args.output_dir),
        run_id=run_id,
        remove_references=not args.no_reference_strip,
        batch_length=args.batch_length,
        max_workers=args.max_workers,
        extraction_passes=args.extraction_passes,
        temperature=args.temperature,
    )
    run_full_extraction(input_paths, config)


if __name__ == "__main__":
    main()
