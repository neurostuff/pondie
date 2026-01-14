#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sentence_transformers import SentenceTransformer
from transformers import pipeline


EXPECTED_NLI_LABELS = {"entailment", "neutral", "contradiction"}


@dataclass(frozen=True)
class Section:
    index: int
    title: str
    sentences: list[str]
    tables: list[str]


@dataclass(frozen=True)
class Record:
    text: str
    section_index: int
    section_title: str
    source_type: str  # sentence|table


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _is_references_title(title: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", title.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return False
    if normalized.startswith("references"):
        return True
    if normalized.startswith("bibliography"):
        return True
    if normalized.startswith("works cited"):
        return True
    if normalized.startswith("literature cited"):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if part.strip()]


def _is_table_header(line: str, next_line: str) -> bool:
    if "|" not in line:
        return False
    if re.search(r"\|?\s*:?-{3,}", next_line):
        return True
    return False


def _parse_markdown_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    sections: list[Section] = []
    current_title = "Document"
    current_sentences: list[str] = []
    current_tables: list[str] = []
    paragraph_lines: list[str] = []
    section_index = 0

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = _normalize_text(" ".join(paragraph_lines))
        current_sentences.extend(_split_sentences(paragraph))
        paragraph_lines.clear()

    def flush_section() -> None:
        nonlocal section_index
        flush_paragraph()
        sections.append(
            Section(
                index=section_index,
                title=current_title,
                sentences=current_sentences.copy(),
                tables=current_tables.copy(),
            )
        )
        section_index += 1
        current_sentences.clear()
        current_tables.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#"):
            flush_section()
            current_title = stripped.lstrip("#").strip() or "Untitled section"
            i += 1
            continue
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if _is_table_header(line, next_line):
            flush_paragraph()
            table_lines = [line, next_line]
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            current_tables.append("\n".join(table_lines))
            continue
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        paragraph_lines.append(stripped)
        i += 1

    flush_section()
    return sections


def _load_entities(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "Group": data.get("groups", []) or [],
        "Task": data.get("tasks", []) or [],
        "Modality": data.get("modalities", []) or [],
        "Analysis": data.get("analyses", []) or [],
    }


def _build_records(sections: list[Section]) -> list[Record]:
    records: list[Record] = []
    for section in sections:
        if _is_references_title(section.title):
            continue
        for sentence in section.sentences:
            records.append(
                Record(
                    text=sentence,
                    section_index=section.index,
                    section_title=section.title,
                    source_type="sentence",
                )
            )
        for table in section.tables:
            records.append(
                Record(
                    text=table,
                    section_index=section.index,
                    section_title=section.title,
                    source_type="table",
                )
            )
    return records


def _embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> Any:
    if not texts:
        return []
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def _canonical_label(label: str, nli) -> str:
    normalized = label.strip().lower()
    if normalized in EXPECTED_NLI_LABELS:
        return normalized
    if normalized.startswith("label_"):
        suffix = normalized.replace("label_", "")
        if suffix.isdigit() and getattr(nli.model, "config", None) is not None:
            mapped = nli.model.config.id2label.get(int(suffix))
            if mapped:
                return str(mapped).strip().lower()
    return normalized


def _scores_from_item(item: Any, nli) -> dict[str, float]:
    out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    scores: list[dict[str, Any]] = []
    if isinstance(item, list):
        scores = item
    elif isinstance(item, dict) and "labels" in item and "scores" in item:
        for label, score in zip(item["labels"], item["scores"]):
            scores.append({"label": label, "score": score})
    elif isinstance(item, dict) and "label" in item and "score" in item:
        scores = [item]

    for score_item in scores:
        label = _canonical_label(str(score_item.get("label", "")), nli)
        if label in out:
            out[label] = float(score_item.get("score", 0.0))
    return out


def _batch_nli_scores(
    nli, premises: list[str], hypothesis: str, batch_size: int
) -> list[dict[str, float]]:
    if not premises:
        return []
    inputs = [{"text": premise, "text_pair": hypothesis} for premise in premises]
    result = nli(
        inputs,
        truncation=True,
        max_length=512,
        top_k=None,
        batch_size=batch_size,
    )
    items: list[Any] = []
    if isinstance(result, list):
        if len(inputs) == 1 and result and isinstance(result[0], dict) and "label" in result[0]:
            items = [result]
        else:
            items = result
    else:
        items = [result]

    scores_list: list[dict[str, float]] = []
    for item in items[: len(inputs)]:
        scores_list.append(_scores_from_item(item, nli))
    while len(scores_list) < len(inputs):
        scores_list.append({"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0})
    return scores_list


def _topk_indices(similarities, top_k: int) -> list[int]:
    if len(similarities) == 0:
        return []
    top_k = min(top_k, len(similarities))
    return list(similarities.argsort()[-top_k:][::-1])


def _derive_md_path(entities_path: Path) -> Path:
    name = entities_path.name
    if name.endswith(".entities.gpt5-nano.json"):
        return entities_path.with_name(name.replace(".entities.gpt5-nano.json", ".md"))
    return entities_path.with_suffix(".md")


def _derive_abbrev_path(entities_path: Path) -> Path:
    name = entities_path.name
    if name.endswith(".entities.gpt5-nano.json"):
        return entities_path.with_name(name.replace(".entities.gpt5-nano.json", ".abbreviations.json"))
    return entities_path.with_suffix(".abbreviations.json")


def _build_hypothesis(entity_type: str, entity: str) -> str:
    entity_type = entity_type.lower()
    if entity_type == "group":
        return f"The document states that '{entity}' is a participant cohort studied."
    if entity_type == "task":
        return f"The document states that participants performed the task '{entity}'."
    if entity_type == "modality":
        return f"The document states that '{entity}' was used to acquire participant data."
    if entity_type == "analysis":
        return f"The document states that the analysis '{entity}' was performed."
    return f"The document states that '{entity}' was used in the study."


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    headers = [
        "entity_type",
        "entity",
        "confirmed",
        "entailment_score",
        "neutral_score",
        "contradiction_score",
        "similarity",
        "source_type",
        "section_title",
        "evidence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Confirm entity mentions with embeddings and NLI."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--entities-path", type=Path)
    group.add_argument("--entities-dir", type=Path)
    parser.add_argument("--md-path", type=Path, default=None)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--entailment-threshold", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    embedding_model = SentenceTransformer(args.embedding_model)
    nli = pipeline(
        "text-classification",
        model=args.nli_model,
        tokenizer=args.nli_model,
        device=-1,
    )

    entity_paths: list[Path] = []
    if args.entities_path:
        entity_paths = [args.entities_path]
    else:
        entities_dir = args.entities_dir
        if not entities_dir.exists():
            raise FileNotFoundError(f"Entities directory not found: {entities_dir}")
        entity_paths = sorted(entities_dir.rglob("*.entities.gpt5-nano.json"))
        if not entity_paths:
            raise FileNotFoundError(f"No entities files found in: {entities_dir}")
    if len(entity_paths) > 1 and args.output_csv is not None:
        raise ValueError("--output-csv can only be used with a single --entities-path.")

    for entities_path in entity_paths:
        md_path = args.md_path or _derive_md_path(entities_path)
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        entities = _load_entities(entities_path)
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
        sections = _parse_markdown_sections(md_text)
        records = _build_records(sections)

        section_titles = [section.title for section in sections]
        section_embeddings = _embed_texts(embedding_model, section_titles, args.batch_size)

        record_texts = [record.text for record in records]
        record_embeddings = _embed_texts(embedding_model, record_texts, args.batch_size)

        rows: list[dict[str, Any]] = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                query_text = entity
                query_embedding = _embed_texts(embedding_model, [query_text], args.batch_size)
                if record_embeddings is None or len(record_embeddings) == 0:
                    rows.append(
                        {
                            "entity_type": entity_type,
                            "entity": entity,
                            "confirmed": False,
                            "entailment_score": "",
                            "neutral_score": "",
                            "contradiction_score": "",
                            "similarity": "",
                            "source_type": "",
                            "section_title": "",
                            "evidence": "",
                        }
                    )
                    continue
                similarities = record_embeddings @ query_embedding[0]
                section_similarities = None
                if section_embeddings is not None and len(section_embeddings) > 0:
                    section_similarities = section_embeddings @ query_embedding[0]
                top_indices = _topk_indices(similarities, args.top_k)

                best_entailment = 0.0
                best_neutral = 0.0
                best_contradiction = 0.0
                best_record: Record | None = None
                best_similarity = 0.0
                best_section_similarity = 0.0

                hypothesis = _build_hypothesis(entity_type, entity)
                top_records = [records[idx] for idx in top_indices]
                premises = [record.text for record in top_records]
                score_list = _batch_nli_scores(
                    nli, premises, hypothesis, args.nli_batch_size
                )
                for idx, record, scores in zip(top_indices, top_records, score_list):
                    entailment = scores["entailment"]
                    section_similarity = 0.0
                    if section_similarities is not None:
                        section_similarity = float(section_similarities[record.section_index])
                    if (
                        entailment > best_entailment
                        or (
                            entailment == best_entailment
                            and section_similarity > best_section_similarity
                        )
                    ):
                        best_entailment = entailment
                        best_neutral = scores["neutral"]
                        best_contradiction = scores["contradiction"]
                        best_record = record
                        best_similarity = float(similarities[idx])
                        best_section_similarity = section_similarity

                confirmed = best_entailment >= args.entailment_threshold
                rows.append(
                    {
                        "entity_type": entity_type,
                        "entity": entity,
                        "confirmed": confirmed,
                        "entailment_score": round(best_entailment, 4),
                        "neutral_score": round(best_neutral, 4),
                        "contradiction_score": round(best_contradiction, 4),
                        "similarity": round(best_similarity, 4),
                        "source_type": best_record.source_type if best_record else "",
                        "section_title": best_record.section_title if best_record else "",
                        "evidence": best_record.text if best_record else "",
                    }
                )

        output_path = args.output_csv
        if output_path is None:
            output_path = entities_path.with_name(
                entities_path.name.replace(".entities.gpt5-nano.json", ".entities.confirmation.csv")
            )
        _write_csv(rows, output_path)
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
