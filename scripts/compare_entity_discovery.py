#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from information_extraction.extraction.pipeline import load_text_documents


GLINER_ENTITY_DESCRIPTIONS = {
    "Group": "Participant cohort label (e.g., healthy controls, patients with schizophrenia).",
    "Task": "Task/paradigm name used in the study (e.g., Stroop task).",
    "Modality": "Imaging modality family used (e.g., fMRI, PET, EEG, MEG).",
    "Analysis": "Analysis or contrast label (e.g., whole-brain analysis, Incongruent > Congruent).",
}


_SECTION_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")


def _split_sections(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines(keepends=True)
    sections: list[tuple[int, str]] = []
    buffer: list[str] = []
    pos = 0
    current_start = 0
    for line in lines:
        is_heading = bool(_SECTION_HEADING_RE.match(line))
        if is_heading and buffer:
            sections.append((current_start, "".join(buffer)))
            buffer = [line]
            current_start = pos
        else:
            if not buffer:
                current_start = pos
            buffer.append(line)
        pos += len(line)
    if buffer:
        sections.append((current_start, "".join(buffer)))
    if not sections:
        sections = [(0, text)]
    return sections


def _split_paragraphs(section_text: str, section_start: int) -> list[tuple[int, str]]:
    paragraphs: list[tuple[int, str]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK_RE.finditer(section_text):
        end = match.end()
        paragraphs.append((section_start + cursor, section_text[cursor:end]))
        cursor = end
    if cursor < len(section_text):
        paragraphs.append((section_start + cursor, section_text[cursor:]))
    if not paragraphs:
        paragraphs = [(section_start, section_text)]
    return paragraphs


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _split_text_by_tokens(
    text: str,
    base_start: int,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, str, int]]:
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoded.get("offset_mapping") or []
    if not offsets:
        return [(base_start, text, 0)]
    step = max_tokens - overlap_tokens if overlap_tokens < max_tokens else max_tokens
    chunks: list[tuple[int, str, int]] = []
    index = 0
    while index < len(offsets):
        end_index = min(index + max_tokens, len(offsets))
        start_char = offsets[index][0]
        end_char = offsets[end_index - 1][1]
        chunk_text = text[start_char:end_char]
        chunks.append((base_start + start_char, chunk_text, end_index - index))
        if end_index >= len(offsets):
            break
        index += step if step > 0 else max_tokens
    return chunks


def _chunk_paragraphs(
    paragraphs: list[tuple[int, str]],
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    chunk_parts: list[str] = []
    chunk_start: int | None = None
    chunk_tokens = 0
    for para_start, paragraph in paragraphs:
        para_tokens = _token_count(tokenizer, paragraph)
        if para_tokens > max_tokens:
            if chunk_parts:
                chunks.append((chunk_start or para_start, "".join(chunk_parts)))
                chunk_parts = []
                chunk_start = None
                chunk_tokens = 0
            split_paras = _split_text_by_tokens(
                paragraph, para_start, tokenizer, max_tokens, overlap_tokens
            )
            for split_start, split_text, _ in split_paras:
                chunks.append((split_start, split_text))
            continue
        if chunk_tokens + para_tokens > max_tokens and chunk_parts:
            chunks.append((chunk_start or para_start, "".join(chunk_parts)))
            chunk_parts = [paragraph]
            chunk_start = para_start
            chunk_tokens = para_tokens
        else:
            if chunk_start is None:
                chunk_start = para_start
            chunk_parts.append(paragraph)
            chunk_tokens += para_tokens
    if chunk_parts:
        chunks.append((chunk_start or 0, "".join(chunk_parts)))
    return chunks


def chunk_document(
    text: str, tokenizer, max_tokens: int, overlap_tokens: int
) -> list[tuple[int, str]]:
    if max_tokens <= 0:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    for section_start, section_text in _split_sections(text):
        if _token_count(tokenizer, section_text) <= max_tokens:
            chunks.append((section_start, section_text))
        else:
            paragraphs = _split_paragraphs(section_text, section_start)
            chunks.extend(
                _chunk_paragraphs(paragraphs, tokenizer, max_tokens, overlap_tokens)
            )
    if not chunks:
        chunks = [(0, text)]
    return chunks


def load_gliner2_model(model_id: str):
    try:
        from gliner2 import GLiNER2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "gliner2 import failed. Install it with `uv pip install gliner2` "
            "and ensure torch is available."
        ) from exc

    model = GLiNER2.from_pretrained(model_id)
    model.eval()
    return model


def run_gliner2_discovery(model, text: str, threshold: float) -> dict[str, Any]:
    results = model.extract_entities(
        text,
        GLINER_ENTITY_DESCRIPTIONS,
        threshold=threshold,
        include_confidence=True,
        include_spans=True,
    )
    return results.get("entities", {})


def _normalize_entity_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def dedupe_entities(
    entities: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    deduped: dict[str, list[dict[str, Any]]] = {}
    for entity_type, mentions in entities.items():
        buckets: dict[str, dict[str, Any]] = {}
        for mention in mentions:
            text = str(mention.get("text", "")).strip()
            if not text:
                continue
            key = _normalize_entity_text(text)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {"text": text, "confidence": mention.get("confidence"), "mentions": []}
                buckets[key] = bucket
            bucket["mentions"].append(mention)
            confidence = mention.get("confidence")
            if confidence is not None and (
                bucket["confidence"] is None or confidence > bucket["confidence"]
            ):
                bucket["confidence"] = confidence
                bucket["text"] = text
        deduped[entity_type] = list(buckets.values())
    return deduped


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GLiNER2 entity discovery on one document."
    )
    parser.add_argument("--input-path", required=True, help="Path to a text JSON/MD file.")
    parser.add_argument(
        "--output-dir",
        default="outputs/entity_compare",
        help="Directory for JSONL outputs.",
    )
    parser.add_argument(
        "--gliner2-model-id",
        default="fastino/gliner2-base-v1",
        help="GLiNER2 Hugging Face model id.",
    )
    parser.add_argument(
        "--gliner2-threshold",
        type=float,
        default=0.5,
        help="GLiNER2 confidence threshold.",
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=2048,
        help="Max tokens per chunk (0 to disable chunking).",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=200,
        help="Token overlap when paragraphs exceed max tokens.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    documents = load_text_documents([input_path])
    if not documents:
        raise ValueError(f"No documents found for input: {input_path}")
    document = documents[0]

    model = load_gliner2_model(args.gliner2_model_id)
    tokenizer = model.processor.tokenizer
    chunks = chunk_document(
        document.text,
        tokenizer,
        args.chunk_max_tokens,
        args.chunk_overlap_tokens,
    )
    merged_entities: dict[str, list[dict[str, Any]]] = {}
    chunk_metadata: list[dict[str, Any]] = []
    for index, (chunk_start, chunk_text) in enumerate(chunks):
        if not chunk_text.strip():
            continue
        chunk_metadata.append(
            {
                "index": index,
                "start": chunk_start,
                "end": chunk_start + len(chunk_text),
                "chars": len(chunk_text),
            }
        )
        chunk_entities = run_gliner2_discovery(
            model, chunk_text, args.gliner2_threshold
        )
        for entity_type, mentions in chunk_entities.items():
            for mention in mentions:
                record = dict(mention)
                if "start" in record and "end" in record:
                    record["start"] = record["start"] + chunk_start
                    record["end"] = record["end"] + chunk_start
                record["chunk_index"] = index
                record["chunk_start"] = chunk_start
                merged_entities.setdefault(entity_type, []).append(record)
    gliner_record = {
        "document_id": document.document_id,
        "tool": "gliner2",
        "model_id": args.gliner2_model_id,
        "entity_descriptions": GLINER_ENTITY_DESCRIPTIONS,
        "chunk_max_tokens": args.chunk_max_tokens,
        "chunk_overlap_tokens": args.chunk_overlap_tokens,
        "chunks": chunk_metadata,
        "entities": merged_entities,
        "entities_deduped": dedupe_entities(merged_entities),
    }

    output_dir = Path(args.output_dir)
    _write_jsonl(output_dir / "gliner2_entity_discovery.jsonl", gliner_record)

    print(f"Wrote {output_dir / 'gliner2_entity_discovery.jsonl'}")


if __name__ == "__main__":
    main()
