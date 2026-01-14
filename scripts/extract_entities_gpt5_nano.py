#!/usr/bin/env python3
"""
Workflow:
1) Load the source markdown (optionally expand abbreviations and save .expanded.md).
2) Extract entities (study/groups/tasks/modalities/analyses) with GPT-5-nano unless
   a valid .entities.gpt5-nano.json exists and --force-reextract is not set.
3) Normalize IDs, task conditions, and run linking unless --skip-links is set.
4) Run a verification pass to correct missing/extra entities and fields.
5) Expand abbreviations in the record (save .expanded.json) before evidence.
6) Attach evidence spans via embedding retrieval + NLI (with LLM fallback), then
   write the .entities.gpt5-nano.grounded.json output.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import math
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import torch
from tqdm import tqdm
from num2words import num2words
import nltk
from nltk.tokenize import sent_tokenize

from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

from information_extraction import schema as schema_mod
from information_extraction.openai_utils import _request_with_retries
from information_extraction.prompting import (
    EntitySpec,
    FieldMeta,
    _build_entity_prompt,
    _build_field_descriptions,
    _build_messages,
    _build_record_schema,
    _build_verification_prompt,
    _field_meta_map,
    _link_schema,
    _load_prompt_spec_cached,
    _prune_model_schema,
    _verification_schema,
    _wrap_items_schema,
)
from information_extraction.schema_utils import (
    _find_literal_values,
    _is_extracted_value_type,
    _is_model_type,
    _iter_model_fields,
    _unwrap_optional,
)


DEFAULT_EXAMPLE_PATH = Path(
    "outputs/text/20260112_221156/xTs9gzGeybhU/elsevier/15488423-10.1016_j.neuroimage.2004.06.041.md"
)

PROMPT_ROOT = Path("prompts")
SCHEMA_VERIFIED_KEY = "_schema_verified"
LINKS_CACHED_KEY = "_links_cached"
VALUE_ONLY_FIELDS = {
    "study_objective",
    "inclusion_criteria",
    "exclusion_criteria",
    "task_description",
    "design_details",
    "analysis_label",
    "contrast_formula",
}
NUMERIC_GROUP_FIELDS = {
    "age_maximum",
    "age_mean",
    "age_median",
    "age_minimum",
    "age_sd",
    "count",
    "female_count",
    "male_count",
}
COMBINED_PREMISE_TOP_N = 3
SEMANTIC_TOP_K = 5


@dataclass(frozen=True)
class EvidencePolicy:
    value_only: bool | None = None
    hypothesis_template: str | None = None
    enum_values: list[str] | None = None
    is_enum: bool = False


@dataclass(frozen=True)
class EvidenceConfig:
    embedding_model: str
    nli_model: str
    top_k: int
    entailment_threshold: float
    embedding_batch_size: int
    nli_batch_size: int
    cache_dir: Path
    device: int
    show_progress: bool = False


@dataclass(frozen=True)
class SentenceRecord:
    text: str
    start: int
    end: int
    section_title: str
    source_type: str
    alignment_status: str


class AbbreviationExpander:
    _ID_KEYS = {
        "id",
        "group_id",
        "task_id",
        "modality_id",
        "analysis_id",
        "condition_id",
    }

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        ordered = [(short, long) for short, long in pairs if short and long]
        ordered.sort(key=lambda item: len(item[0]), reverse=True)
        self.patterns = [
            (re.compile(rf"\b{re.escape(short)}\b"), long) for short, long in ordered
        ]
        self.regex = None
        if ordered:
            escaped = [re.escape(short) for short, _ in ordered]
            self.regex = re.compile(r"\b(" + "|".join(escaped) + r")\b")
        self.short_to_long = {short: long for short, long in ordered}

    def expand_text(self, text: str) -> str:
        if not self.patterns:
            return text
        expanded = text
        for pattern, long in self.patterns:
            expanded = pattern.sub(long, expanded)
        return expanded

    def expand_with_map(self, text: str) -> tuple[str, list[int]]:
        if not self.regex:
            return text, list(range(len(text)))
        output: list[str] = []
        mapping: list[int] = []
        cursor = 0
        for match in self.regex.finditer(text):
            start, end = match.span()
            if start > cursor:
                chunk = text[cursor:start]
                output.append(chunk)
                mapping.extend(range(cursor, start))
            short = match.group(0)
            long = self.short_to_long.get(short, short)
            output.append(long)
            orig_len = max(end - start, 1)
            for i in range(len(long)):
                mapped = start + min(orig_len - 1, int(i * orig_len / max(len(long), 1)))
                mapping.append(mapped)
            cursor = end
        if cursor < len(text):
            output.append(text[cursor:])
            mapping.extend(range(cursor, len(text)))
        return "".join(output), mapping

    @classmethod
    def from_path(cls, path: Path) -> AbbreviationExpander | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        pairs: list[tuple[str, str]] = []
        for item in payload.get("abbreviations", []) or []:
            short = str(item.get("short", "")).strip()
            long = str(item.get("long", "")).strip()
            if not short or not long:
                continue
            if short.lower() == long.lower():
                continue
            if "|" in long or "\n" in long:
                continue
            if short.islower() and len(short) <= 2:
                continue
            if len(short) <= 1:
                continue
            pairs.append((short, long))
        return cls(pairs) if pairs else None

    def expand_record(self, value: Any) -> Any:
        if isinstance(value, dict):
            expanded: dict[str, Any] = {}
            for key, item in value.items():
                if key in self._ID_KEYS:
                    expanded[key] = item
                    continue
                if key == "value" and isinstance(item, str):
                    expanded[key] = self.expand_text(item)
                    continue
                expanded[key] = self.expand_record(item)
            return expanded
        if isinstance(value, list):
            return [self.expand_record(item) for item in value]
        if isinstance(value, str):
            return self.expand_text(value)
        return value


class EvidenceProgress:
    def __init__(self, total: int, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled and tqdm is not None
        self.bar = None
        if self.enabled:
            self.bar = tqdm(total=total, desc="Evidence", unit="field")

    def update(self, n: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(n)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _model_evidence_policies(model: type[Any]) -> dict[str, EvidencePolicy]:
    policies: dict[str, EvidencePolicy] = {}
    for name, annotation, _description, extra in _iter_model_fields(model):
        value_only = extra.get("evidence_value_only")
        template = extra.get("evidence_hypothesis_template")
        enum_values = _find_literal_values(annotation)
        extraction_type = extra.get("extraction_type")
        if isinstance(extraction_type, list):
            is_enum = "enum" in extraction_type
        else:
            is_enum = extraction_type == "enum"
        if value_only is None and template is None and not enum_values and not is_enum:
            continue
        policies[name] = EvidencePolicy(
            value_only=value_only if isinstance(value_only, bool) else None,
            hypothesis_template=template if isinstance(template, str) else None,
            enum_values=enum_values,
            is_enum=is_enum,
        )
    return policies


ENTITY_SPECS = [
    EntitySpec(
        key="study",
        prompt_path=PROMPT_ROOT / "study.json",
        model=schema_mod.StudyMetadataModel,
        max_items=1,
    ),
    EntitySpec(
        key="groups",
        prompt_path=PROMPT_ROOT / "group.json",
        model=schema_mod.GroupBase,
    ),
    EntitySpec(
        key="tasks",
        prompt_path=PROMPT_ROOT / "task.json",
        model=schema_mod.TaskBase,
    ),
    EntitySpec(
        key="modalities",
        prompt_path=PROMPT_ROOT / "modality.json",
        model=schema_mod.ModalityBase,
    ),
    EntitySpec(
        key="analyses",
        prompt_path=PROMPT_ROOT / "analysis.json",
        model=schema_mod.AnalysisBase,
    ),
]


def _read_text(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    last_was_space = False
    for idx, char in enumerate(text):
        if char.isspace():
            if last_was_space:
                continue
            norm_chars.append(" ")
            norm_to_orig.append(idx)
            last_was_space = True
            continue
        norm_chars.append(char)
        norm_to_orig.append(idx)
        last_was_space = False
    return "".join(norm_chars), norm_to_orig


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _token_overlap_ratio(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    if not doc_set:
        return 0.0
    return len(query_set & doc_set) / max(len(query_set), 1)


def _parse_int_literal(value_text: str) -> int | None:
    normalized = value_text.strip()
    if not re.fullmatch(r"[-+]?\d+", normalized):
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def _integer_value_for_query(value: Any, value_sentence: str) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        parsed = _parse_int_literal(value)
        if parsed is not None:
            return parsed
    return _parse_int_literal(value_sentence)


def _is_references_title(title: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", title.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return False
    return normalized.startswith(("references", "bibliography", "works cited", "literature cited"))


def _strip_heading_markers(text: str) -> str:
    cleaned = re.sub(r"^#+\s*", "", text.lstrip())
    cleaned = re.sub(r"\s#+\s+", " ", cleaned)
    return cleaned.strip()


def _section_boundaries(text: str) -> list[tuple[int, str]]:
    boundaries = [(0, "Document")]
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip() or "Untitled section"
            boundaries.append((offset, title))
        offset += len(line)
    return boundaries


def _split_sentences_with_spans(
    normalized_text: str,
    norm_to_orig: list[int],
    *,
    section_bounds: list[tuple[int, str]],
) -> list[SentenceRecord]:
    sentences: list[SentenceRecord] = []
    try:
        sentence_texts = sent_tokenize(normalized_text)
    except LookupError as exc:  # pragma: no cover - optional dependency
        try:
            nltk.download("punkt_tab", quiet=True)
            sentence_texts = sent_tokenize(normalized_text)
        except Exception:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "NLTK punkt tokenizer not found. Run: python -m nltk.downloader punkt punkt_tab"
            ) from exc

    cursor = 0
    for sentence in sentence_texts:
        start_norm = normalized_text.find(sentence, cursor)
        if start_norm == -1:
            continue
        end_norm = start_norm + len(sentence)
        cursor = end_norm
        sentence = sentence.strip()
        if not sentence:
            continue
        cleaned_sentence = _strip_heading_markers(sentence)
        if not cleaned_sentence:
            continue
        if start_norm >= len(norm_to_orig):
            continue
        end_index = max(start_norm, end_norm - 1)
        if end_index >= len(norm_to_orig):
            end_index = len(norm_to_orig) - 1
        start_orig = norm_to_orig[start_norm]
        end_orig = norm_to_orig[end_index] + 1
        section_title = "Document"
        for boundary_start, title in reversed(section_bounds):
            if start_orig >= boundary_start:
                section_title = title
                break
        if _is_references_title(section_title):
            continue
        sentences.append(
            SentenceRecord(
                text=cleaned_sentence,
                start=start_orig,
                end=end_orig,
                section_title=section_title,
                source_type="sentence",
                alignment_status="match_exact",
            )
        )
    return sentences


def _safe_cache_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_cached_embeddings(
    cache_dir: Path, model_name: str, doc_hash: str, sentences_hash: str
) -> np.ndarray | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_cache_name(model_name)
    meta_path = cache_dir / f"{safe_name}-{doc_hash}.json"
    data_path = cache_dir / f"{safe_name}-{doc_hash}.npz"
    if not (meta_path.exists() and data_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        meta.get("model") != model_name
        or meta.get("doc_hash") != doc_hash
        or meta.get("sentences_hash") != sentences_hash
    ):
        return None
    try:
        data = np.load(data_path)
        embeddings = data.get("embeddings")
    except Exception:
        return None
    if embeddings is None:
        return None
    return embeddings


def _write_cached_embeddings(
    cache_dir: Path,
    model_name: str,
    doc_hash: str,
    sentences_hash: str,
    embeddings: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_cache_name(model_name)
    meta_path = cache_dir / f"{safe_name}-{doc_hash}.json"
    data_path = cache_dir / f"{safe_name}-{doc_hash}.npz"
    meta = {
        "model": model_name,
        "doc_hash": doc_hash,
        "sentences_hash": sentences_hash,
        "count": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(data_path, embeddings=embeddings)


def _canonical_label(label: str, nli) -> str:
    normalized = label.strip().lower()
    if normalized in {"entailment", "neutral", "contradiction"}:
        return normalized
    if normalized.startswith("label_"):
        suffix = normalized.replace("label_", "")
        if suffix.isdigit() and getattr(nli.model, "config", None) is not None:
            mapped = nli.model.config.id2label.get(int(suffix))
            if mapped:
                return str(mapped).strip().lower()
    return normalized


def _scores_from_item(item: Any, nli) -> dict[str, float]:
    scores: list[dict[str, Any]] = []
    if isinstance(item, list):
        scores = item
    elif isinstance(item, dict) and "labels" in item and "scores" in item:
        scores = [
            {"label": label, "score": score}
            for label, score in zip(item["labels"], item["scores"])
        ]
    elif isinstance(item, dict) and "label" in item and "score" in item:
        scores = [item]

    out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
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


class EvidenceIndex:
    def __init__(
        self,
        text: str,
        document_id: str | None,
        config: EvidenceConfig,
        progress: EvidenceProgress | None = None,
        llm_client: OpenAI | None = None,
        llm_service_tier: str | None = None,
        expander: AbbreviationExpander | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Evidence extraction requires the 'semantic' extras. "
                "Install with: uv pip install -e \".[semantic]\""
            ) from exc

        self.document_id = document_id
        self.raw_text = text
        self.config = config
        self.progress = progress
        self.llm_client = llm_client
        self.llm_service_tier = llm_service_tier
        self.llm_cache: dict[tuple[str, tuple[str, ...]], list[int]] = {}
        self.expander = expander
        self.normalized_text, self.norm_to_orig = _normalize_with_map(text)
        section_bounds = _section_boundaries(text)
        self.sentences = _split_sentences_with_spans(
            self.normalized_text,
            self.norm_to_orig,
            section_bounds=section_bounds,
        )
        if self.expander:
            self.sentence_texts = [
                self.expander.expand_text(sentence.text) for sentence in self.sentences
            ]
        else:
            self.sentence_texts = [sentence.text for sentence in self.sentences]

        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.nli = pipeline(
            "text-classification",
            model=config.nli_model,
            tokenizer=config.nli_model,
            device=config.device,
        )

        self.sentence_embeddings = self._load_or_compute_embeddings()
        self.query_cache: dict[str, np.ndarray] = {}
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        self.bm25_tokens = [_tokenize(text) for text in self.sentence_texts]
        self.bm25_doc_freq: dict[str, int] = {}
        for tokens in self.bm25_tokens:
            for token in set(tokens):
                self.bm25_doc_freq[token] = self.bm25_doc_freq.get(token, 0) + 1
        self.bm25_avgdl = (
            sum(len(tokens) for tokens in self.bm25_tokens) / max(len(self.bm25_tokens), 1)
        )

    def _bm25_scores(self, query_tokens: list[str]) -> list[float]:
        if not query_tokens or not self.bm25_tokens:
            return []
        scores = [0.0 for _ in self.bm25_tokens]
        n_docs = len(self.bm25_tokens)
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            df = self.bm25_doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for i, tokens in enumerate(self.bm25_tokens):
                tf = tokens.count(token)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * (len(tokens) / max(self.bm25_avgdl, 1)))
                scores[i] += idf * (tf * (k1 + 1)) / denom
        return scores


    def _load_or_compute_embeddings(self) -> np.ndarray:
        doc_hash = _hash_text(self.normalized_text)
        sentences_hash = _hash_text("\n".join(self.sentence_texts))
        cached = _load_cached_embeddings(
            self.config.cache_dir,
            self.config.embedding_model,
            doc_hash,
            sentences_hash,
        )
        if cached is not None:
            return cached
        if not self.sentence_texts:
            return np.zeros((0, 0), dtype=float)
        embeddings = self.embedding_model.encode(
            self.sentence_texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        _write_cached_embeddings(
            self.config.cache_dir,
            self.config.embedding_model,
            doc_hash,
            sentences_hash,
            embeddings,
        )
        return embeddings

    def _query_embedding(self, query: str) -> np.ndarray:
        cached = self.query_cache.get(query)
        if cached is not None:
            return cached
        embedding = self.embedding_model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        embedding = embedding[0]
        self.query_cache[query] = embedding
        return embedding

    def _value_text(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return f"{value:g}"
        return str(value).strip()

    def _match_value_span(
        self,
        normalized_sentence: str,
        normalized_value: str,
    ) -> tuple[int, int, str] | None:
        if not normalized_value:
            return None
        idx = normalized_sentence.find(normalized_value)
        if idx != -1:
            return idx, idx + len(normalized_value), "match_exact"
        if re.fullmatch(r"[-+]?\\d+(\\.\\d+)?", normalized_value):
            base_raw = normalized_value.rstrip("0").rstrip(".")
            base = re.escape(base_raw or normalized_value)
            if "." in normalized_value:
                pattern = rf"\\b{base}0*\\b"
            else:
                pattern = rf"\\b{base}(?:\\.0+)?\\b"
            match = re.search(pattern, normalized_sentence)
            if match:
                return match.start(), match.end(), "match_exact"
        matcher = difflib.SequenceMatcher(None, normalized_sentence, normalized_value)
        match = matcher.find_longest_match(
            0, len(normalized_sentence), 0, len(normalized_value)
        )
        if match.size and match.size / max(len(normalized_value), 1) >= 0.6:
            return match.a, match.a + match.size, "match_fuzzy"
        return None

    def _split_value_sentences(self, value_text: str) -> list[str]:
        normalized = _normalize_text(value_text)
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return [part.strip() for part in parts if part.strip()]

    def _is_numeric_sentence(self, value_sentence: str) -> bool:
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value_sentence.strip()))

    def _normalize_enum_value(self, value: str) -> str:
        normalized = re.sub(r"[_/]+", " ", value)
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized.lower()

    def _enum_aliases(self, normalized_value: str) -> list[str]:
        aliases = [normalized_value]
        alias_map = {
            "f mri": ["fmri", "functional mri"],
            "structural mri": ["structural mri", "s mri"],
            "diffusion mri": ["diffusion mri", "diffusion", "dti"],
            "seed based connectivity": ["seed-based connectivity"],
            "independent components analysis": ["independent component analysis", "ica"],
            "brain behavior correlation": ["brain-behavior correlation", "behavior correlation"],
            "atlas parcellation": ["atlas/parcellation", "parcellation"],
            "roi": ["region of interest"],
            "bold": ["blood oxygen level dependent"],
            "cbf": ["cerebral blood flow"],
            "cbv": ["cerebral blood volume"],
            "fdg": ["fluorodeoxyglucose"],
            "pet": ["positron emission tomography"],
            "eeg": ["electroencephalography"],
            "meg": ["magnetoencephalography"],
            "15 o-water": ["15o water", "oxygen 15 water", "o-15 water"],
        }
        aliases.extend(alias_map.get(normalized_value, []))
        deduped = list(dict.fromkeys(item for item in aliases if item))
        return deduped

    def _format_hypothesis(
        self,
        *,
        value_sentence: str,
        description: str,
        context_label: str | None,
        template: str | None,
    ) -> str:
        if template:
            context_prefix = f"For {context_label}, " if context_label else ""
            rendered = template.format(
                value=value_sentence,
                context_prefix=context_prefix,
            )
            return " ".join(rendered.split())
        context_prefix = f"For {context_label}, " if context_label else ""
        return f"{context_prefix}{value_sentence} is {description}"

    def _llm_supports_indices(self, hypothesis: str, sentences: list[str]) -> list[int]:
        if self.llm_client is None:
            return []
        if not sentences:
            return []
        key = (hypothesis, tuple(sentences))
        cached = self.llm_cache.get(key)
        if cached is not None:
            return cached
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"supported_indices": {"type": "array", "items": {"type": "integer"}}},
            "required": ["supported_indices"],
        }
        numbered = "\n".join(
            f"{idx}. {sentence}" for idx, sentence in enumerate(sentences, start=1)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Determine which sentences support the hypothesis. "
                    "Return JSON only in the form: {\"supported_indices\": [1,2,...]} "
                    "using 1-based indices from the list provided."
                ),
            },
            {
                "role": "user",
                "content": f"Hypothesis: {hypothesis}\nSentences:\n{numbered}",
            },
        ]
        try:
            result = _request_with_retries(
                self.llm_client,
                messages,
                model="gpt-5-nano",
                service_tier=self.llm_service_tier,
                max_output_tokens=50,
                max_retries=3,
                timeout_s=30.0,
                schema=schema,
                schema_name="SupportCheck",
            )
        except Exception:
            self.llm_cache[key] = []
            return []
        supported: list[int] = []
        if isinstance(result, dict):
            raw_indices = result.get("supported_indices", [])
            if isinstance(raw_indices, list):
                for item in raw_indices:
                    if isinstance(item, int):
                        supported.append(item)
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, int):
                    supported.append(item)
        self.llm_cache[key] = supported
        return supported

    def find_evidence(
        self,
        value: Any,
        field_description: str | None,
        *,
        field_name: str | None = None,
        context_label: str | None = None,
        evidence_policy: EvidencePolicy | None = None,
    ) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            description = _evidence_description(field_name, field_description)
            value_text = self._value_text(value)
            value_sentences = self._split_value_sentences(value_text)
            if not value_sentences:
                return None
            if self.sentence_embeddings is None or len(self.sentence_embeddings) == 0:
                return None
            evidence: list[dict[str, Any]] = []
            seen_spans: set[tuple[int, int, str]] = set()
            value_only_override = (
                evidence_policy.value_only
                if evidence_policy and evidence_policy.value_only is not None
                else None
            )
            use_value_only_default = (
                value_only_override
                if value_only_override is not None
                else field_name in VALUE_ONLY_FIELDS
            )
            hypothesis_template = (
                evidence_policy.hypothesis_template
                if evidence_policy and evidence_policy.hypothesis_template
                else None
            )
            for value_sentence in value_sentences:
                if evidence_policy and evidence_policy.is_enum:
                    value_sentence = self._normalize_enum_value(value_sentence)
                use_value_only_for_field = (
                    value_only_override
                    if value_only_override is not None
                    else field_name in VALUE_ONLY_FIELDS
                )
                use_value_only_for_retrieval = use_value_only_for_field or (
                    evidence_policy is not None
                    and evidence_policy.is_enum
                    and evidence_policy.value_only is True
                )
                numeric_retrieval_only = isinstance(value, (int, float)) or self._is_numeric_sentence(
                    value_sentence
                )
                skip_context_pass = (
                    field_name in NUMERIC_GROUP_FIELDS
                    if field_name is not None
                    else False
                )
                value_only_hypothesis_retrieval = (
                    f"For {context_label}, {value_sentence}"
                    if context_label
                    else value_sentence
                )
                if numeric_retrieval_only:
                    value_only_hypothesis_retrieval = value_sentence
                full_hypothesis_retrieval = self._format_hypothesis(
                    value_sentence=value_sentence,
                    description=description,
                    context_label=context_label,
                    template=hypothesis_template,
                )
                value_only_hypothesis_nli = value_sentence
                context_label_nli = (
                    context_label if skip_context_pass else None
                )
                full_hypothesis_nli = self._format_hypothesis(
                    value_sentence=value_sentence,
                    description=description,
                    context_label=context_label_nli,
                    template=hypothesis_template,
                )
                use_value_only_first = use_value_only_for_field
                hypothesis_retrieval = (
                    full_hypothesis_retrieval
                    if numeric_retrieval_only
                    else (
                        value_only_hypothesis_retrieval
                        if use_value_only_for_retrieval
                        else full_hypothesis_retrieval
                    )
                )
                hypothesis_nli = (
                    value_only_hypothesis_nli if use_value_only_first else full_hypothesis_nli
                )
                if evidence_policy and evidence_policy.is_enum:
                    alias_query = " ".join(self._enum_aliases(value_sentence))
                    query_tokens = _tokenize(alias_query)
                else:
                    query_tokens = _tokenize(value_sentence)
                if numeric_retrieval_only:
                    integer_value = _integer_value_for_query(value, value_sentence)
                    if integer_value is not None:
                        word_tokens = _tokenize(num2words(integer_value))
                        if word_tokens:
                            existing = set(query_tokens)
                            for token in word_tokens:
                                if token not in existing:
                                    query_tokens.append(token)
                                    existing.add(token)
                bm25_scores = self._bm25_scores(query_tokens)
                bm25_indices = [
                    idx for idx, score in enumerate(bm25_scores) if score > 0
                ]
                if len(bm25_indices) > 20:
                    ranked_bm25 = sorted(
                        bm25_indices, key=lambda idx: bm25_scores[idx], reverse=True
                    )
                    bm25_indices = ranked_bm25[:20]
                strong_single = False
                if len(bm25_indices) > 1:
                    ranked_bm25 = sorted(
                        bm25_indices, key=lambda idx: bm25_scores[idx], reverse=True
                    )
                    top_score = bm25_scores[ranked_bm25[0]]
                    second_score = bm25_scores[ranked_bm25[1]]
                    top_idx = ranked_bm25[0]
                    overlap_ratio = _token_overlap_ratio(
                        query_tokens, self.bm25_tokens[top_idx]
                    )
                    if overlap_ratio >= 0.6 and (
                        second_score == 0 or top_score >= second_score * 1.5
                    ):
                        strong_single = True
                        bm25_indices = [top_idx]
                if not bm25_indices:
                    query_embedding = self._query_embedding(hypothesis_retrieval)
                    similarities = self.sentence_embeddings @ query_embedding
                    top_k = min(SEMANTIC_TOP_K, len(similarities))
                    if top_k <= 0:
                        continue
                    top_indices = list(similarities.argsort()[-top_k:][::-1])
                elif len(bm25_indices) == 1 or strong_single:
                    top_indices = bm25_indices
                else:
                    query_embedding = self._query_embedding(hypothesis_retrieval)
                    similarities = self.sentence_embeddings @ query_embedding
                    ranked = sorted(
                        bm25_indices, key=lambda idx: similarities[idx], reverse=True
                    )
                    top_indices = ranked[: min(SEMANTIC_TOP_K, len(ranked))]

                if context_label and not skip_context_pass:
                    context_hypothesis = f"This sentence is about {context_label}."
                    context_texts = [self.sentence_texts[idx] for idx in top_indices]
                    context_scores = _batch_nli_scores(
                        self.nli,
                        context_texts,
                        context_hypothesis,
                        self.config.nli_batch_size,
                    )
                    ranked_pairs = sorted(
                        zip(top_indices, context_scores),
                        key=lambda item: item[1]["entailment"],
                        reverse=True,
                    )
                    top_indices = [idx for idx, _scores in ranked_pairs]

                normalized_value = _normalize_text(value_sentence).lower()
                if (len(top_indices) == 1 and len(bm25_indices) == 1) or strong_single:
                    chosen_indices = top_indices
                else:
                    candidate_texts = [self.sentence_texts[idx] for idx in top_indices]
                    scores_list = _batch_nli_scores(
                        self.nli,
                        candidate_texts,
                        hypothesis_nli,
                        self.config.nli_batch_size,
                    )
                    clear_entailment_indices: list[int] = []
                    neutral_indices: list[int] = []
                    neutral_hypothesis = hypothesis_nli
                    chosen_indices: list[int] | None = None
                    for idx, scores in zip(top_indices, scores_list):
                        entailment = scores["entailment"]
                        neutral = scores["neutral"]
                        contradiction = scores["contradiction"]
                        if (
                            entailment >= self.config.entailment_threshold
                            and entailment > neutral
                            and entailment > contradiction
                        ):
                            clear_entailment_indices.append(idx)
                        elif neutral >= entailment and neutral >= contradiction:
                            neutral_indices.append(idx)
                    if clear_entailment_indices:
                        chosen_indices = clear_entailment_indices
                    else:
                        fallback_value_only = (
                            not use_value_only_first
                            and not self._is_numeric_sentence(value_sentence)
                        )
                        if fallback_value_only:
                            fallback_scores = _batch_nli_scores(
                                self.nli,
                                candidate_texts,
                                value_only_hypothesis_nli,
                                self.config.nli_batch_size,
                            )
                            fallback_entailment: list[int] = []
                            fallback_neutral: list[int] = []
                            for idx, scores in zip(top_indices, fallback_scores):
                                entailment = scores["entailment"]
                                neutral = scores["neutral"]
                                contradiction = scores["contradiction"]
                                if (
                                    entailment >= self.config.entailment_threshold
                                    and entailment > neutral
                                    and entailment > contradiction
                                ):
                                    fallback_entailment.append(idx)
                                elif neutral >= entailment and neutral >= contradiction:
                                    fallback_neutral.append(idx)
                            if fallback_entailment:
                                chosen_indices = fallback_entailment
                            else:
                                neutral_indices = fallback_neutral
                                neutral_hypothesis = value_only_hypothesis_nli
                        if chosen_indices is None:
                            combined_indices = top_indices[: min(COMBINED_PREMISE_TOP_N, len(top_indices))]
                            combined_text = " ".join(
                                self.sentence_texts[idx] for idx in combined_indices
                            )
                            combined_hypothesis = (
                                value_only_hypothesis_nli
                                if use_value_only_first
                                else full_hypothesis_nli
                            )
                            combined_scores = _batch_nli_scores(
                                self.nli,
                                [combined_text],
                                combined_hypothesis,
                                self.config.nli_batch_size,
                            )[0]
                            if (
                                combined_scores["entailment"] >= self.config.entailment_threshold
                                and combined_scores["entailment"] > combined_scores["neutral"]
                                and combined_scores["entailment"] > combined_scores["contradiction"]
                            ):
                                chosen_indices = combined_indices
                        if chosen_indices is None and not self._is_numeric_sentence(value_sentence):
                            candidate_texts = [self.sentence_texts[idx] for idx in top_indices]
                            value_tokens = _tokenize(normalized_value)
                            lexical_indices = [
                                idx
                                for idx, text in zip(top_indices, candidate_texts)
                                if _token_overlap_ratio(value_tokens, _tokenize(text)) >= 0.6
                            ]
                            if lexical_indices:
                                chosen_indices = lexical_indices
                        if chosen_indices is None:
                            candidates = neutral_indices if neutral_indices else top_indices
                            candidate_texts = [self.sentence_texts[idx] for idx in candidates]
                            supported = self._llm_supports_indices(
                                neutral_hypothesis, candidate_texts
                            )
                            chosen_indices = [
                                candidates[idx - 1]
                                for idx in supported
                                if 1 <= idx <= len(candidates)
                            ]
                        if chosen_indices is None:
                            chosen_indices = []

                if chosen_indices is None:
                    chosen_indices = []
                for idx in chosen_indices:
                    record = self.sentences[idx]
                    raw_segment = self.raw_text[record.start:record.end]
                    if self.expander:
                        expanded_segment, expanded_to_raw = self.expander.expand_with_map(
                            raw_segment
                        )
                        normalized_segment, norm_to_expanded = _normalize_with_map(
                            expanded_segment
                        )
                        if not norm_to_expanded:
                            continue
                        norm_to_orig = [expanded_to_raw[i] for i in norm_to_expanded]
                    else:
                        normalized_segment, norm_to_orig = _normalize_with_map(raw_segment)
                    if not norm_to_orig:
                        continue
                    match = self._match_value_span(
                        normalized_segment.lower(), normalized_value
                    )
                    if match is None:
                        start_pos = record.start
                        end_pos = record.end
                        alignment_status = "match_fuzzy"
                        extraction_text = _normalize_text(self.raw_text[start_pos:end_pos])
                    else:
                        match_start, match_end, alignment_status = match
                        if match_start >= len(norm_to_orig):
                            start_pos = record.start
                            end_pos = record.end
                            alignment_status = "match_fuzzy"
                            extraction_text = _normalize_text(self.raw_text[start_pos:end_pos])
                        else:
                            end_index = max(match_start, match_end - 1)
                            if end_index >= len(norm_to_orig):
                                end_index = len(norm_to_orig) - 1
                            start_pos = record.start + norm_to_orig[match_start]
                            end_pos = record.start + norm_to_orig[end_index] + 1
                            extraction_text = _normalize_text(self.raw_text[start_pos:end_pos])
                    span_key = (start_pos, end_pos, record.section_title)
                    if span_key in seen_spans:
                        continue
                    seen_spans.add(span_key)
                    evidence.append(
                        {
                            "source": record.source_type,
                            "section": record.section_title,
                            "extraction_text": extraction_text,
                            "char_interval": {
                                "start_pos": start_pos,
                                "end_pos": end_pos,
                            },
                            "alignment_status": alignment_status,
                            "document_id": self.document_id,
                        }
                    )
            return evidence or None
        finally:
            if self.progress is not None:
                self.progress.update(1)


def _ensure_ids(items: list[Any], prefix: str, *, overwrite: bool = False) -> list[Any]:
    if overwrite:
        return [
            dict(item, id=f"{prefix}{idx}") if isinstance(item, dict) else item
            for idx, item in enumerate(items, start=1)
        ]
    used: set[str] = {
        item.get("id", "").strip()
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id").strip()
    }
    counter = 1
    updated: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            updated.append(item)
            continue
        item_copy = dict(item)
        value = item_copy.get("id")
        if not isinstance(value, str) or not value.strip():
            while f"{prefix}{counter}" in used:
                counter += 1
            item_copy["id"] = f"{prefix}{counter}"
            used.add(item_copy["id"])
            counter += 1
        updated.append(item_copy)
    return updated


def _assign_missing_condition_ids(tasks: list[Any]) -> list[Any]:
    used: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        conditions = task.get("conditions")
        if not isinstance(conditions, list):
            continue
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            value = condition.get("id")
            if isinstance(value, str) and value.strip():
                used.add(value.strip())

    counter = 1
    updated_tasks: list[Any] = []
    for task in tasks:
        if not isinstance(task, dict):
            updated_tasks.append(task)
            continue
        task_copy = dict(task)
        conditions = task_copy.get("conditions")
        if not isinstance(conditions, list):
            updated_tasks.append(task_copy)
            continue
        updated_conditions: list[Any] = []
        for condition in conditions:
            if not isinstance(condition, dict):
                updated_conditions.append(condition)
                continue
            condition_copy = dict(condition)
            value = condition_copy.get("id")
            if not isinstance(value, str) or not value.strip():
                while f"C{counter}" in used:
                    counter += 1
                condition_copy["id"] = f"C{counter}"
                used.add(condition_copy["id"])
                counter += 1
            updated_conditions.append(condition_copy)
        task_copy["conditions"] = updated_conditions
        updated_tasks.append(task_copy)
    return updated_tasks


def _normalize_task_conditions(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    condition_counter = 1
    normalized = []
    for task in tasks:
        task = dict(task)
        conditions = task.get("conditions")
        if conditions is None:
            normalized.append(task)
            continue
        if not isinstance(conditions, list):
            task["conditions"] = None
            normalized.append(task)
            continue
        condition_objs = []
        for item in conditions:
            if isinstance(item, dict):
                condition = dict(item)
                label = condition.get("condition_label")
                if label is None or (isinstance(label, str) and not label.strip()):
                    continue
                if not condition.get("id"):
                    condition["id"] = f"C{condition_counter}"
                condition_objs.append(condition)
                condition_counter += 1
                continue
            if isinstance(item, str):
                label = item.strip()
                if not label:
                    continue
                condition_objs.append({"id": f"C{condition_counter}", "condition_label": label})
                condition_counter += 1
        task["conditions"] = condition_objs or None
        normalized.append(task)
    return normalized


def _assign_missing_ids_to_record(record: dict[str, Any]) -> dict[str, Any]:
    demographics = record.get("demographics")
    if isinstance(demographics, dict):
        groups = demographics.get("groups")
        if isinstance(groups, list):
            demographics["groups"] = _ensure_ids(groups, "G")
        record["demographics"] = demographics
    tasks = record.get("tasks")
    if isinstance(tasks, list):
        tasks = _ensure_ids(tasks, "T")
        tasks = _assign_missing_condition_ids(tasks)
        record["tasks"] = tasks
    modalities = record.get("modalities")
    if isinstance(modalities, list):
        record["modalities"] = _ensure_ids(modalities, "M")
    analyses = record.get("analyses")
    if isinstance(analyses, list):
        record["analyses"] = _ensure_ids(analyses, "A")
    return record


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def _evidence_description(field_name: str | None, description: str | None) -> str:
    if field_name == "contrast_formula":
        return "statistical contrast formula"
    return description or field_name or "field"


def _group_context_label(group: dict[str, Any]) -> str | None:
    label = _unwrap_value(group.get("cohort_label"))
    if isinstance(label, str) and label.strip():
        return f"cohort {label.strip()}"
    label = _unwrap_value(group.get("medical_condition"))
    if isinstance(label, str) and label.strip():
        return f"cohort {label.strip()}"
    group_id = group.get("id")
    if isinstance(group_id, str) and group_id.strip():
        return f"group {group_id.strip()}"
    return None


def _task_context_label(task: dict[str, Any]) -> str | None:
    label = _unwrap_value(task.get("task_name"))
    if isinstance(label, str) and label.strip():
        return f"task {label.strip()}"
    label = _unwrap_value(task.get("task_description"))
    if isinstance(label, str) and label.strip():
        return f"task {label.strip()}"
    task_id = task.get("id")
    if isinstance(task_id, str) and task_id.strip():
        return f"task {task_id.strip()}"
    return None


def _analysis_context_label(analysis: dict[str, Any]) -> str | None:
    label = _unwrap_value(analysis.get("analysis_label"))
    if isinstance(label, str) and label.strip():
        return f"analysis {label.strip()}"
    label = _unwrap_value(analysis.get("contrast_formula"))
    if isinstance(label, str) and label.strip():
        return f"analysis {label.strip()}"
    analysis_id = analysis.get("id")
    if isinstance(analysis_id, str) and analysis_id.strip():
        return f"analysis {analysis_id.strip()}"
    return None


def _modality_context_label(modality: dict[str, Any]) -> str | None:
    modality_type = modality.get("modality_type")
    if isinstance(modality_type, dict):
        family = _unwrap_value(modality_type.get("family"))
        subtype = _unwrap_value(modality_type.get("subtype"))
        label_parts = []
        if isinstance(family, str) and family.strip():
            label_parts.append(family.strip())
        if isinstance(subtype, str) and subtype.strip():
            label_parts.append(subtype.strip())
        if label_parts:
            return f"modality {' '.join(label_parts)}"
    for key in ("sequence_name", "voxel_size", "manufacturer"):
        label = _unwrap_value(modality.get(key))
        if isinstance(label, str) and label.strip():
            return f"modality {label.strip()}"
    modality_id = modality.get("id")
    if isinstance(modality_id, str) and modality_id.strip():
        return f"modality {modality_id.strip()}"
    return None


def _wrap_model_with_evidence(
    obj: dict[str, Any],
    model: type[Any],
    evidence_index: EvidenceIndex,
    *,
    context_label: str | None = None,
    context_exclude: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return obj
    updated = dict(obj)
    policies = _model_evidence_policies(model)
    for field_name, annotation, description, _extra in _iter_model_fields(model):
        if field_name not in obj:
            continue
        value = obj.get(field_name)
        effective_context = context_label
        if context_exclude and field_name in context_exclude:
            effective_context = None
        evidence_policy = policies.get(field_name)
        updated[field_name] = _wrap_value_with_evidence(
            value,
            annotation,
            description,
            evidence_index,
            field_name=field_name,
            context_label=effective_context,
            context_exclude=context_exclude,
            evidence_policy=evidence_policy,
        )
    return updated


def _wrap_value_with_evidence(
    value: Any,
    annotation: Any,
    description: str | None,
    evidence_index: EvidenceIndex,
    *,
    field_name: str | None = None,
    context_label: str | None = None,
    context_exclude: set[str] | None = None,
    evidence_policy: EvidencePolicy | None = None,
) -> Any:
    if value is None:
        return None

    def _wrap_extracted(item: Any) -> dict[str, Any]:
        if isinstance(item, dict) and "value" in item:
            updated = dict(item)
            raw = updated.get("value")
            if raw is None:
                updated["evidence"] = None
                return updated
            updated["evidence"] = evidence_index.find_evidence(
                raw,
                description,
                field_name=field_name,
                context_label=context_label,
                evidence_policy=evidence_policy,
            )
            return updated
        return {
            "value": item,
            "evidence": evidence_index.find_evidence(
                item,
                description,
                field_name=field_name,
                context_label=context_label,
                evidence_policy=evidence_policy,
            ),
        }
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        item_type = args[0] if args else None
        if _is_extracted_value_type(item_type):
            if isinstance(value, list):
                return [_wrap_extracted(item) for item in value]
            return value
        if _is_model_type(item_type):
            if isinstance(value, list):
                return [
                    _wrap_model_with_evidence(
                        item,
                        item_type,
                        evidence_index,
                        context_label=context_label,
                        context_exclude=context_exclude,
                    )
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            return value
        return value
    if _is_extracted_value_type(annotation):
        return _wrap_extracted(value)
    if _is_model_type(annotation):
        if isinstance(value, dict):
            return _wrap_model_with_evidence(
                value,
                annotation,
                evidence_index,
                context_label=context_label,
                context_exclude=context_exclude,
            )
        return value
    return value


def _has_evidence_value(value: Any) -> bool:
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _count_evidence_fields(value: Any, annotation: Any) -> int:
    if value is None:
        return 0
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        item_type = args[0] if args else None
        if _is_extracted_value_type(item_type):
            if isinstance(value, list):
                return sum(1 for item in value if _has_evidence_value(item))
            return 0
        if _is_model_type(item_type):
            if isinstance(value, list):
                return sum(
                    _count_model_evidence_fields(item, item_type)
                    for item in value
                    if isinstance(item, dict)
                )
            return 0
        return 0
    if _is_extracted_value_type(annotation):
        return 1 if _has_evidence_value(value) else 0
    if _is_model_type(annotation):
        if isinstance(value, dict):
            return _count_model_evidence_fields(value, annotation)
        return 0
    return 0


def _count_model_evidence_fields(obj: dict[str, Any], model: type[Any]) -> int:
    total = 0
    for field_name, annotation, _description, _extra in _iter_model_fields(model):
        if field_name not in obj:
            continue
        total += _count_evidence_fields(obj.get(field_name), annotation)
    return total


def _attach_evidence_to_record(
    record: dict[str, Any],
    *,
    text: str,
    document_id: str | None,
    config: EvidenceConfig,
    llm_client: OpenAI | None,
    llm_service_tier: str | None,
    expander: AbbreviationExpander | None,
) -> dict[str, Any]:
    total_fields = _count_model_evidence_fields(record, schema_mod.StudyRecord)
    if config.show_progress and total_fields <= 0:
        config = EvidenceConfig(
            embedding_model=config.embedding_model,
            nli_model=config.nli_model,
            top_k=config.top_k,
            entailment_threshold=config.entailment_threshold,
            embedding_batch_size=config.embedding_batch_size,
            nli_batch_size=config.nli_batch_size,
            cache_dir=config.cache_dir,
            device=config.device,
            show_progress=False,
        )
    progress = None
    if config.show_progress:
        if tqdm is None:
            print("tqdm is not installed; progress bar disabled.", file=sys.stderr)
        else:
            progress = EvidenceProgress(total_fields, enabled=True)
    evidence_index = EvidenceIndex(
        text,
        document_id,
        config,
        progress=progress,
        llm_client=llm_client,
        llm_service_tier=llm_service_tier,
        expander=expander,
    )
    updated = dict(record)

    study = record.get("study")
    if isinstance(study, dict):
        updated["study"] = _wrap_model_with_evidence(
            study, schema_mod.StudyMetadataModel, evidence_index
        )

    demographics = record.get("demographics")
    if isinstance(demographics, dict):
        groups = demographics.get("groups")
        if isinstance(groups, list):
            demographics = dict(demographics)
            demographics["groups"] = [
                _wrap_model_with_evidence(
                    group,
                    schema_mod.GroupBase,
                    evidence_index,
                    context_label=_group_context_label(group) if isinstance(group, dict) else None,
                )
                if isinstance(group, dict)
                else group
                for group in groups
            ]
            updated["demographics"] = demographics

    tasks = record.get("tasks")
    if isinstance(tasks, list):
        updated["tasks"] = [
            _wrap_model_with_evidence(
                task,
                schema_mod.TaskBase,
                evidence_index,
                context_label=_task_context_label(task) if isinstance(task, dict) else None,
                context_exclude={"task_name"},
            )
            if isinstance(task, dict)
            else task
            for task in tasks
        ]

    modalities = record.get("modalities")
    if isinstance(modalities, list):
        updated["modalities"] = [
            _wrap_model_with_evidence(
                modality,
                schema_mod.ModalityBase,
                evidence_index,
                context_label=_modality_context_label(modality)
                if isinstance(modality, dict)
                else None,
            )
            if isinstance(modality, dict)
            else modality
            for modality in modalities
        ]

    analyses = record.get("analyses")
    if isinstance(analyses, list):
        updated["analyses"] = [
            _wrap_model_with_evidence(
                analysis,
                schema_mod.AnalysisBase,
                evidence_index,
                context_label=_analysis_context_label(analysis)
                if isinstance(analysis, dict)
                else None,
                context_exclude={"analysis_label"},
            )
            if isinstance(analysis, dict)
            else analysis
            for analysis in analyses
        ]

    links = record.get("links")
    if isinstance(links, dict):
        updated["links"] = _wrap_model_with_evidence(
            links, schema_mod.StudyLinks, evidence_index
        )

    if progress is not None:
        progress.close()

    return updated


def _build_link_prompt(
    text: str,
    groups: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    modalities: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You extract explicit links between entities. "
        "Only create links when explicitly stated. "
        "Return JSON only."
    )
    doc_message = f"Document text:\n{text}"

    def label_for(item: dict[str, Any], keys: list[str]) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return item.get("id", "")

    def modality_label(item: dict[str, Any]) -> str:
        modality_type = item.get("modality_type")
        if isinstance(modality_type, dict):
            for key in ["family", "subtype"]:
                value = modality_type.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return label_for(item, ["sequence_name", "voxel_size", "manufacturer"])

    def summarize(items: list[dict[str, Any]], label_keys: list[str]) -> list[dict[str, Any]]:
        summarized = []
        for item in items:
            entry = {"id": item.get("id", ""), "label": label_for(item, label_keys)}
            if "conditions" in item:
                entry["conditions"] = item.get("conditions")
            summarized.append(entry)
        return summarized

    entity_block = {
        "groups": summarize(groups, ["cohort_label", "medical_condition"]),
        "tasks": summarize(tasks, ["task_name", "task_description"]),
        "modalities": [
            {"id": item.get("id", ""), "label": modality_label(item)}
            for item in modalities
        ],
        "analyses": summarize(analyses, ["analysis_label", "contrast_formula", "analysis_method"]),
    }

    instructions = (
        "Use the entities listed below (IDs are required). "
        "Create edges only when explicitly stated in the document text. "
        "Do not infer. Always include all edge lists; use [] if none. "
        "For analysis_condition, only link when a named task condition is explicitly referenced.\n\n"
        f"Entities:\n{json.dumps(entity_block, indent=2)}\n\n"
        "Output JSON format:\n"
        "{\n"
        '  "group_task": [{"group_id": "G1", "task_id": "T1"}],\n'
        '  "task_modality": [{"task_id": "T1", "modality_id": "M1"}],\n'
        '  "analysis_task": [{"analysis_id": "A1", "task_id": "T1"}],\n'
        '  "analysis_group": [{"analysis_id": "A1", "group_id": "G1"}],\n'
        '  "analysis_condition": [{"analysis_id": "A1", "condition_id": "C1"}],\n'
        '  "group_modality": [{"group_id": "G1", "modality_id": "M1", "n_scanned": 20}]\n'
        "}\n"
        "\nUse the document text above."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": doc_message},
        {"role": "user", "content": instructions},
    ]


def _load_existing_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if hasattr(schema_mod.StudyRecord, "model_validate"):
            schema_mod.StudyRecord.model_validate(existing)
        else:
            schema_mod.StudyRecord.parse_obj(existing)
        return existing
    except Exception:
        return None


def _collect_field_meta(specs: list[EntitySpec]) -> dict[str, dict[str, FieldMeta]]:
    meta_by_key: dict[str, dict[str, FieldMeta]] = {}
    for spec in specs:
        prompt_spec = _load_prompt_spec_cached(spec.prompt_path)
        fields = prompt_spec.get("fields") or []
        meta_by_key[spec.key] = _field_meta_map(spec.model, fields)
    return meta_by_key


def _extract_entities(
    *,
    text: str,
    specs: list[EntitySpec],
    client: OpenAI,
    model: str,
    service_tier: str | None,
    max_output_tokens: int,
    max_retries: int,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, dict[str, FieldMeta]]]:
    results: dict[str, Any] = {
        "study": None,
        "groups": [],
        "tasks": [],
        "modalities": [],
        "analyses": [],
        "links": None,
    }
    meta_by_key: dict[str, dict[str, FieldMeta]] = {}
    for spec in specs:
        prompt_spec = _load_prompt_spec_cached(spec.prompt_path)
        fields = prompt_spec.get("fields") or []
        field_meta = _field_meta_map(spec.model, fields)
        meta_by_key[spec.key] = field_meta
        entity_prompt = _build_entity_prompt(prompt_spec, field_meta)
        item_schema = _prune_model_schema(spec.model, fields, include_id=False)
        schema = _wrap_items_schema(item_schema, spec.max_items)
        messages = _build_messages(text, entity_prompt)
        data = _request_with_retries(
            client=client,
            messages=messages,
            model=model,
            service_tier=service_tier,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            timeout_s=timeout_s,
            schema=schema,
            schema_name=f"{spec.key}_schema",
        )
        items = data.get("items", []) if isinstance(data, dict) else []
        if spec.key == "study":
            results["study"] = items[0] if items else None
        else:
            results[spec.key] = items
    return results, meta_by_key


def _apply_verification_result(
    record: dict[str, Any],
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    corrected = verification.get("corrected") if isinstance(verification, dict) else None
    if isinstance(corrected, dict):
        corrected["verification_changes"] = verification.get("changes", [])
        corrected[SCHEMA_VERIFIED_KEY] = True
        if record.get(LINKS_CACHED_KEY):
            corrected[LINKS_CACHED_KEY] = True
        return _assign_missing_ids_to_record(corrected)
    return record


def _verify_record(
    *,
    text: str,
    record: dict[str, Any],
    meta_by_key: dict[str, dict[str, FieldMeta]],
    client: OpenAI,
    model: str,
    service_tier: str | None,
    max_output_tokens: int,
    max_retries: int,
    timeout_s: float,
) -> dict[str, Any]:
    field_descriptions = _build_field_descriptions(meta_by_key)
    verify_messages = _build_verification_prompt(text, record, field_descriptions)
    entity_schema = _build_record_schema(ENTITY_SPECS, include_ids=True)
    verify_schema = _verification_schema(entity_schema)
    verification = _request_with_retries(
        client=client,
        messages=verify_messages,
        model=model,
        service_tier=service_tier,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
        timeout_s=timeout_s,
        schema=verify_schema,
        schema_name="verification",
    )
    return _apply_verification_result(record, verification)


def _link_record(
    *,
    record: dict[str, Any],
    text: str,
    client: OpenAI,
    model: str,
    service_tier: str | None,
    max_output_tokens: int,
    max_retries: int,
    timeout_s: float,
) -> dict[str, Any]:
    if record.get("links") is not None:
        record[LINKS_CACHED_KEY] = True
        return record
    groups = (record.get("demographics") or {}).get("groups") or []
    tasks = record.get("tasks") or []
    modalities = record.get("modalities") or []
    analyses = record.get("analyses") or []
    if not (groups or tasks or modalities or analyses):
        return record
    link_messages = _build_link_prompt(
        text=text,
        groups=groups,
        tasks=tasks,
        modalities=modalities,
        analyses=analyses,
    )
    link_schema = _link_schema()
    links = _request_with_retries(
        client=client,
        messages=link_messages,
        model=model,
        service_tier=service_tier,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
        timeout_s=timeout_s,
        schema=link_schema,
        schema_name="study_links",
    )
    record["links"] = links
    record[LINKS_CACHED_KEY] = True
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract entities with GPT-5 nano (no LangExtract) and link them."
    )
    parser.add_argument(
        "--input-md",
        type=Path,
        default=DEFAULT_EXAMPLE_PATH,
        help="Path to markdown text to extract.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write grounded JSON (default: alongside input, .grounded suffix).",
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--service-tier", default="flex")
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-chars", type=int, default=400_000)
    parser.add_argument("--skip-links", action="store_true", help="Skip the linking pass.")
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Re-run extraction even if a .entities.gpt5-nano.json file exists.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for evidence retrieval.",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="NLI model used to validate evidence sentences.",
    )
    parser.add_argument("--evidence-top-k", type=int, default=10)
    parser.add_argument("--evidence-entailment-threshold", type=float, default=0.6)
    parser.add_argument("--evidence-embedding-batch-size", type=int, default=32)
    parser.add_argument("--evidence-nli-batch-size", type=int, default=32)
    parser.add_argument(
        "--evidence-cache-dir",
        type=Path,
        default=Path(".cache/evidence_embeddings"),
    )
    parser.add_argument(
        "--evidence-device",
        type=int,
        default=-1,
        help="Device for evidence models (-1 for CPU, 0+ for CUDA).",
    )
    parser.set_defaults(evidence_progress=True)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Set CPU thread counts for torch and BLAS env vars.",
    )
    args = parser.parse_args()

    if args.cpu_threads is not None:
        cpu_threads = max(1, args.cpu_threads)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(cpu_threads)
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(cpu_threads)

    load_dotenv(".env")
    if not args.input_md.exists():
        print(f"Input not found: {args.input_md}", file=sys.stderr)
        return 2

    text = _read_text(args.input_md, args.max_chars)
    expander = AbbreviationExpander.from_path(args.input_md.with_suffix(".abbreviations.json"))
    if expander is not None:
        expanded_text = expander.expand_text(text)
        if expanded_text != text:
            expanded_text_path = args.input_md.with_suffix(".expanded.md")
            expanded_text_path.write_text(expanded_text, encoding="utf-8")
    client = OpenAI()
    service_tier = args.service_tier or None
    if service_tier and service_tier.lower() == "none":
        service_tier = None

    raw_output_path = args.input_md.with_suffix(".entities.gpt5-nano.json")
    grounded_output_path = args.output_json
    if grounded_output_path is None:
        grounded_output_path = args.input_md.with_suffix(".entities.gpt5-nano.grounded.json")

    record = None
    if not args.force_reextract:
        record = _load_existing_record(raw_output_path)

    meta_by_key: dict[str, dict[str, FieldMeta]] = {}
    if record is None:
        results, meta_by_key = _extract_entities(
            text=text,
            specs=ENTITY_SPECS,
            client=client,
            model=args.model,
            service_tier=service_tier,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            timeout_s=args.timeout,
        )

        # Assign IDs and normalize conditions for linking.
        results["groups"] = _ensure_ids(results["groups"], "G", overwrite=True)
        results["tasks"] = _ensure_ids(results["tasks"], "T", overwrite=True)
        results["tasks"] = _normalize_task_conditions(results["tasks"])
        results["modalities"] = _ensure_ids(results["modalities"], "M", overwrite=True)
        results["analyses"] = _ensure_ids(results["analyses"], "A", overwrite=True)

        record = {
            "study": results["study"],
            "demographics": {"groups": results["groups"]},
            "tasks": results["tasks"],
            "modalities": results["modalities"],
            "analyses": results["analyses"],
            "links": results["links"],
        }

        if not args.skip_links:
            record = _link_record(
                record=record,
                text=text,
                client=client,
                model=args.model,
                service_tier=service_tier,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                timeout_s=args.timeout,
            )

        record = _verify_record(
            text=text,
            record=record,
            meta_by_key=meta_by_key,
            client=client,
            model=args.model,
            service_tier=service_tier,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            timeout_s=args.timeout,
        )

        raw_output_path.write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        if not record.get(SCHEMA_VERIFIED_KEY):
            meta_by_key = _collect_field_meta(ENTITY_SPECS)
            record = _verify_record(
                text=text,
                record=record,
                meta_by_key=meta_by_key,
                client=client,
                model=args.model,
                service_tier=service_tier,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                timeout_s=args.timeout,
            )
        if record.get("links") is not None:
            record[LINKS_CACHED_KEY] = True
        if not args.skip_links and not record.get(LINKS_CACHED_KEY):
            record = _link_record(
                record=record,
                text=text,
                client=client,
                model=args.model,
                service_tier=service_tier,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                timeout_s=args.timeout,
            )
        if record.get(SCHEMA_VERIFIED_KEY) or record.get(LINKS_CACHED_KEY):
            raw_output_path.write_text(
                json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
            )

    if expander is not None:
        expanded_record = expander.expand_record(record)
        expanded_record_path = raw_output_path.with_suffix(".expanded.json")
        if expanded_record != record:
            expanded_record_path.write_text(
                json.dumps(expanded_record, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        record = expanded_record

    evidence_config = EvidenceConfig(
        embedding_model=args.embedding_model,
        nli_model=args.nli_model,
        top_k=args.evidence_top_k,
        entailment_threshold=args.evidence_entailment_threshold,
        embedding_batch_size=args.evidence_embedding_batch_size,
        nli_batch_size=args.evidence_nli_batch_size,
        cache_dir=args.evidence_cache_dir,
        device=args.evidence_device,
        show_progress=args.evidence_progress,
    )
    record = _attach_evidence_to_record(
        record,
        text=text,
        document_id=args.input_md.stem,
        config=evidence_config,
        llm_client=client,
        llm_service_tier=service_tier,
        expander=expander,
    )

    grounded_output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {grounded_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# TODO (eventually)
# finally, taking inspiration from langextract, create a visualization form for the created annotations, highlighting their spans over the markdown text, having different color schemes for the different Entities (Study/Task/Modality/Analysis/Group), and critically, allowing users to edit the value that is associated with that key. Have a button that allows you delete all evidentiary spans for that key/value and add the ability to highlight new spans for that piece of evidence.

# TODO (eventually)
# also when I looked at the prompt, enums are not being forwarded to the prompt so that the prompt knows what values it can use for a certain field.
