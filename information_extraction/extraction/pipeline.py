from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, get_args

from dotenv import load_dotenv
from pydantic import ValidationError
try:
    from pydantic import TypeAdapter
except ImportError:  # pragma: no cover - pydantic v1 fallback
    TypeAdapter = None

from information_extraction.schema import (
    AnalysisBase,
    BoolField,
    CharInterval,
    Condition,
    DemographicsSchema,
    EvidenceSpan,
    ExtractedValue,
    FloatField,
    GroupBase,
    GroupModalityEdge,
    GroupTaskEdge,
    IntField,
    ModalityBase,
    ModalityFamilyLiteral,
    ModalityType,
    SharedDemographics,
    StrField,
    StudyLinks,
    StudyMetadataModel,
    StudyRecord,
    TaskBase,
    TaskModalityEdge,
    AnalysisTaskEdge,
    AnalysisGroupEdge,
    AnalysisConditionEdge,
)


logger = logging.getLogger(__name__)


@dataclass
class PromptSpec:
    version: str
    entity: str
    extraction_class: str
    prompt_description: str
    fields: list[str]
    examples: list[dict[str, Any]]
    field_prompt_overrides: dict[str, str] | None = None


@dataclass
class ExtractionConfig:
    model_id: str = "gemini-2.5-flash"
    prompt_dir: Path = Path("prompts")
    output_dir: Path = Path("outputs/extraction")
    run_id: str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    remove_references: bool = True
    max_char_buffer: int | None = None
    batch_length: int = 10
    max_workers: int = 10
    extraction_passes: int = 1
    temperature: float | None = None
    enable_fuzzy_alignment: bool = True
    fuzzy_alignment_threshold: float = 0.75
    accept_match_lesser: bool = True
    suppress_parse_errors: bool = True
    prompt_validation_level: str = "warning"
    prompt_validation_strict: bool = False
    show_progress: bool = True
    retry_attempts: int = 5
    retry_min_seconds: float = 2.0
    retry_max_seconds: float = 30.0


@dataclass
class TextDocument:
    document_id: str
    text: str
    source_path: Path


PROMPT_FILES = {
    "group": "group.json",
    "task": "task.json",
    "modality": "modality.json",
    "analysis": "analysis.json",
    "study": "study.json",
    "demographics_shared": "demographics_shared.json",
    "links": "links.json",
}

DISCOVERY_FIELDS: dict[str, list[str]] = {
    "group": ["cohort_label", "population_role", "medical_condition", "count"],
    "task": ["task_name", "resting_state", "conditions"],
    "modality": ["modality_family", "modality_subtype"],
    "analysis": ["contrast_formula", "reporting_scope"],
}


def load_prompt_spec(path: Path) -> PromptSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PromptSpec(
        version=data["version"],
        entity=data["entity"],
        extraction_class=data["extraction_class"],
        prompt_description=data["prompt_description"],
        fields=list(data["fields"]),
        examples=list(data["examples"]),
        field_prompt_overrides=data.get("field_prompt_overrides"),
    )


def load_prompt_specs(prompt_dir: Path) -> dict[str, PromptSpec]:
    specs: dict[str, PromptSpec] = {}
    for key, filename in PROMPT_FILES.items():
        path = prompt_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing prompt file: {path}")
        specs[key] = load_prompt_spec(path)
    return specs


def load_text_documents(input_paths: Iterable[Path]) -> list[TextDocument]:
    documents: list[TextDocument] = []
    for path in input_paths:
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            documents.append(
                TextDocument(document_id=path.stem, text=text, source_path=path)
            )
            continue
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            document_id = (
                data.get("document_id")
                or data.get("doc_id")
                or data.get("id")
                or (data.get("metadata") or {}).get("document_id")
                or path.stem
            )
            text = _extract_text_from_json(data)
            documents.append(
                TextDocument(document_id=document_id, text=text, source_path=path)
            )
            continue
        raise ValueError(f"Unsupported input file: {path}")
    return documents


def _extract_text_from_json(data: dict[str, Any]) -> str:
    for key in ("text", "markdown", "content", "document_text", "full_text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    sections = data.get("sections")
    if isinstance(sections, list):
        blocks = []
        for section in sections:
            if isinstance(section, dict):
                text = section.get("text") or section.get("content")
                if text:
                    blocks.append(str(text))
            elif isinstance(section, str):
                blocks.append(section)
        if blocks:
            return "\n\n".join(blocks)
    raise ValueError("Input JSON missing usable text fields (text/markdown/sections).")


def strip_references(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    if len(lines) < 30:
        return text, False

    heading_indices = [i for i, line in enumerate(lines) if _is_reference_heading(line)]
    if not heading_indices:
        return text, False

    min_start = int(len(lines) * 0.6)
    for idx in heading_indices:
        if idx < min_start:
            continue
        window = lines[idx + 1 : idx + 51]
        if not window:
            continue
        score = _reference_line_score(window)
        if score < 0.35:
            continue
        next_heading = _find_next_heading(lines, idx + 1)
        if next_heading is not None and _is_appendix_heading(lines[next_heading]):
            new_lines = lines[:idx] + lines[next_heading:]
            return "\n".join(new_lines).strip(), True
        return "\n".join(lines[:idx]).strip(), True
    return text, False


def _is_reference_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower().lstrip("#").strip()
    lowered = re.sub(r"[:\\s]+$", "", lowered)
    return lowered in {
        "references",
        "reference",
        "bibliography",
        "works cited",
        "literature cited",
        "references and notes",
    }


def _is_appendix_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower().lstrip("#").strip()
    return any(
        phrase in lowered
        for phrase in (
            "appendix",
            "supplement",
            "supplementary",
            "supplemental",
            "supporting information",
        )
    )


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.isupper() and len(stripped) <= 60:
        return True
    return False


def _find_next_heading(lines: list[str], start: int) -> int | None:
    for idx in range(start, len(lines)):
        if _is_heading(lines[idx]):
            return idx
    return None


def _reference_line_score(lines: list[str]) -> float:
    ref_like = 0
    total = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        total += 1
        if _is_reference_line(stripped):
            ref_like += 1
    if total == 0:
        return 0.0
    return ref_like / total


def _is_reference_line(line: str) -> bool:
    if re.match(r"^\\[?\\d+\\]?[).]?\\s", line):
        return True
    if re.search(r"\\(19\\d{2}|20\\d{2}\\)", line):
        return True
    if "doi:" in line.lower():
        return True
    if re.search(r"https?://", line):
        return True
    return False


def build_field_prompt_map() -> dict[str, dict[str, str]]:
    from information_extraction import schema

    return {
        "GroupBase": _field_prompt_map(schema.GroupBase),
        "TaskBase": _field_prompt_map(schema.TaskBase),
        "ModalityBase": {
            **_field_prompt_map(schema.ModalityBase),
            "modality_family": _field_prompt(schema.ModalityType, "family"),
            "modality_subtype": _field_prompt(schema.ModalityType, "subtype"),
        },
        "AnalysisBase": _field_prompt_map(schema.AnalysisBase),
        "StudyMetadataModel": _field_prompt_map(schema.StudyMetadataModel),
        "SharedDemographics": _field_prompt_map(schema.SharedDemographics),
    }


def _field_prompt_map(model: Any) -> dict[str, str]:
    prompts: dict[str, str] = {}
    fields = getattr(model, "model_fields", None)
    if fields:
        for name, field in fields.items():
            extra = getattr(field, "json_schema_extra", None) or {}
            prompt = extra.get("extraction_prompt")
            if prompt:
                prompts[name] = prompt
        return prompts
    fields = getattr(model, "__fields__", None)
    if fields:
        for name, field in fields.items():
            extra = getattr(field.field_info, "extra", None) or {}
            prompt = extra.get("extraction_prompt")
            if prompt:
                prompts[name] = prompt
    return prompts


def _field_prompt(model: Any, field_name: str) -> str:
    return _field_prompt_map(model).get(field_name, "")


def build_prompt_description(
    prompt_spec: PromptSpec,
    field_prompts: dict[str, str],
    fields_subset: list[str] | None = None,
) -> str:
    fields = fields_subset or prompt_spec.fields
    overrides = prompt_spec.field_prompt_overrides or {}
    lines = [prompt_spec.prompt_description.strip(), "", "Fields:"]
    for field in fields:
        prompt = overrides.get(field) or field_prompts.get(field)
        if prompt:
            lines.append(f"- {field}: {prompt}")
        else:
            lines.append(f"- {field}")
    return "\n".join(lines).strip()


def build_examples(
    prompt_spec: PromptSpec,
    fields_subset: list[str] | None = None,
) -> list[Any]:
    from langextract.core import data as lx_data

    examples: list[Any] = []
    fields = set(fields_subset or prompt_spec.fields)
    for example in prompt_spec.examples:
        extractions = []
        for extraction in example.get("extractions", []):
            attrs = extraction.get("attributes", {}) or {}
            if fields_subset:
                attrs = {k: v for k, v in attrs.items() if k in fields}
            extraction_class = extraction.get(
                "extraction_class", prompt_spec.extraction_class
            )
            extractions.append(
                lx_data.Extraction(
                    extraction_class=extraction_class,
                    extraction_text=extraction.get("extraction_text", ""),
                    attributes=attrs,
                )
            )
        examples.append(
            lx_data.ExampleData(text=example.get("text", ""), extractions=extractions)
        )
    return examples


def run_langextract(
    text: str,
    *,
    prompt_spec: PromptSpec,
    field_prompts: dict[str, str],
    config: ExtractionConfig,
    fields_subset: list[str] | None = None,
    additional_context: str | None = None,
    document_id: str | None = None,
    warnings: list[StrField] | None = None,
    warning_context: str | None = None,
) -> Any:
    import langextract
    from langextract import prompt_validation as pv
    from langextract.core import exceptions as lx_exceptions
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    prompt_description = build_prompt_description(
        prompt_spec, field_prompts, fields_subset
    )
    examples = build_examples(prompt_spec, fields_subset)
    resolver_params = {
        "enable_fuzzy_alignment": config.enable_fuzzy_alignment,
        "fuzzy_alignment_threshold": config.fuzzy_alignment_threshold,
        "accept_match_lesser": config.accept_match_lesser,
        "suppress_parse_errors": config.suppress_parse_errors,
    }
    max_char_buffer = config.max_char_buffer
    prompt_validation_level = pv.PromptValidationLevel[
        config.prompt_validation_level.upper()
    ]
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
    kwargs = {}
    if max_char_buffer is not None:
        kwargs["max_char_buffer"] = max_char_buffer
    def _should_retry(exc: BaseException) -> bool:
        if isinstance(exc, lx_exceptions.InferenceRuntimeError):
            status_code = _extract_status_code(exc)
            if status_code in {429, 500, 502, 503, 504}:
                return True
            message = str(exc).lower()
            return any(
                token in message
                for token in ("unavailable", "overloaded", "resource_exhausted", "quota")
            )
        return False

    @retry(
        retry=retry_if_exception(_should_retry),
        wait=wait_exponential_jitter(
            initial=config.retry_min_seconds, max=config.retry_max_seconds
        ),
        stop=stop_after_attempt(config.retry_attempts),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _run() -> Any:
        return langextract.extract(
            text,
            prompt_description=prompt_description,
            examples=examples,
            model_id=config.model_id,
            api_key=api_key,
            temperature=config.temperature,
            batch_length=config.batch_length,
            max_workers=config.max_workers,
            additional_context=additional_context,
            resolver_params=resolver_params,
            extraction_passes=config.extraction_passes,
            prompt_validation_level=prompt_validation_level,
            prompt_validation_strict=config.prompt_validation_strict,
            show_progress=config.show_progress,
            **kwargs,
        )

    try:
        return _run()
    except ValueError as exc:
        if "Source tokens and extraction tokens cannot be empty" in str(exc):
            message = "LangExtract alignment failed: empty source/extraction tokens."
            if warning_context:
                message = f"{warning_context}: {message}"
            logger.warning(message)
            if warnings is not None and document_id is not None:
                _record_warning(warnings, message, None, document_id, None)
            from langextract.core import data as lx_data

            return lx_data.AnnotatedDocument(
                document_id=document_id, text=text, extractions=[]
            )
        raise


def run_full_extraction(
    input_paths: Iterable[Path],
    config: ExtractionConfig,
) -> list[StudyRecord]:
    load_dotenv()
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")):
        raise ValueError("GEMINI_API_KEY or LANGEXTRACT_API_KEY must be set.")

    prompt_specs = load_prompt_specs(config.prompt_dir)
    field_prompt_maps = build_field_prompt_map()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = config.output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    documents = load_text_documents(input_paths)
    aggregated_records: list[StudyRecord] = []
    aggregated_annotations: list[Any] = []

    for doc in documents:
        text = doc.text
        warning_notes: list[StrField] = []
        if config.remove_references:
            text, stripped = strip_references(text)
            if stripped:
                logger.info("Stripped references from %s", doc.document_id)
        if not text.strip():
            _record_warning(
                warning_notes,
                "Empty document text after preprocessing; skipping extraction.",
                None,
                doc.document_id,
                None,
            )
            record = StudyRecord(
                study=None,
                demographics=DemographicsSchema(groups=None, shared=None),
                tasks=None,
                modalities=None,
                analyses=None,
                links=None,
                extraction_notes=warning_notes,
            )
            from langextract.core import data as lx_data

            annotated_doc = lx_data.AnnotatedDocument(
                document_id=doc.document_id, text=text, extractions=[]
            )
            aggregated_records.append(record)
            aggregated_annotations.append(annotated_doc)
            _write_outputs(run_dir, doc.document_id, record, annotated_doc)
            continue

        discovery_results: dict[str, Any] = {}
        for entity_key in ("group", "task", "modality", "analysis"):
            prompt_spec = prompt_specs[entity_key]
            discovery_results[entity_key] = run_langextract(
                text,
                prompt_spec=prompt_spec,
                field_prompts=field_prompt_maps[prompt_spec.entity],
                config=config,
                fields_subset=DISCOVERY_FIELDS.get(entity_key),
                document_id=doc.document_id,
                warnings=warning_notes,
                warning_context=f"{entity_key} discovery",
            )

        record = _build_stub_record(
            discovery_results, doc.document_id, text, warning_notes
        )

        links_spec = prompt_specs["links"]
        links_context = _build_links_context(record)
        links_doc = run_langextract(
            text,
            prompt_spec=links_spec,
            field_prompts={},
            config=config,
            additional_context=links_context,
            document_id=doc.document_id,
            warnings=warning_notes,
            warning_context="links discovery",
        )
        record.links = _parse_links(
            getattr(links_doc, "extractions", []), doc.document_id, warning_notes
        )

        filled_docs = _fill_entities(
            record,
            text,
            doc.document_id,
            prompt_specs,
            field_prompt_maps,
            config,
            warning_notes,
        )
        if warning_notes:
            record.extraction_notes = warning_notes

        aggregated_records.append(record)
        combined = _combine_annotated_docs(
            text,
            doc.document_id,
            [
                discovery_results["group"],
                discovery_results["task"],
                discovery_results["modality"],
                discovery_results["analysis"],
                links_doc,
                *filled_docs,
            ],
        )
        aggregated_annotations.append(combined)

        _write_outputs(run_dir, doc.document_id, record, combined)

    _write_aggregate_outputs(run_dir, aggregated_records, aggregated_annotations)
    return aggregated_records


def _build_stub_record(
    discovery_results: dict[str, Any],
    document_id: str,
    text: str,
    warnings: list[StrField],
) -> StudyRecord:
    groups = _build_groups(
        getattr(discovery_results["group"], "extractions", []), document_id, warnings
    )
    tasks = _build_tasks(
        getattr(discovery_results["task"], "extractions", []), document_id, warnings
    )
    modalities = _build_modalities(
        getattr(discovery_results["modality"], "extractions", []),
        document_id,
        warnings,
    )
    analyses = _build_analyses(
        getattr(discovery_results["analysis"], "extractions", []), document_id, warnings
    )
    demographics = DemographicsSchema(groups=groups, shared=None)
    return StudyRecord(
        study=None,
        demographics=demographics,
        tasks=tasks,
        modalities=modalities,
        analyses=analyses,
        links=None,
        extraction_notes=None,
    )


def _fill_entities(
    record: StudyRecord,
    text: str,
    document_id: str,
    prompt_specs: dict[str, PromptSpec],
    field_prompt_maps: dict[str, dict[str, str]],
    config: ExtractionConfig,
    warnings: list[StrField],
) -> list[Any]:
    annotated_docs: list[Any] = []
    study_spec = prompt_specs["study"]
    study_doc = run_langextract(
        text,
        prompt_spec=study_spec,
        field_prompts=field_prompt_maps[study_spec.entity],
        config=config,
        document_id=document_id,
        warnings=warnings,
        warning_context="study fill",
    )
    record.study = _build_study_metadata(
        getattr(study_doc, "extractions", []), document_id, warnings
    )
    annotated_docs.append(study_doc)

    shared_spec = prompt_specs["demographics_shared"]
    shared_doc = run_langextract(
        text,
        prompt_spec=shared_spec,
        field_prompts=field_prompt_maps[shared_spec.entity],
        config=config,
        document_id=document_id,
        warnings=warnings,
        warning_context="shared demographics fill",
    )
    record.demographics.shared = _build_shared_demographics(
        getattr(shared_doc, "extractions", []), document_id, warnings
    )
    annotated_docs.append(shared_doc)

    group_spec = prompt_specs["group"]
    for group in record.demographics.groups or []:
        context = _build_target_context("Group", group.id, _extract_label(group))
        group_doc = run_langextract(
            text,
            prompt_spec=group_spec,
            field_prompts=field_prompt_maps[group_spec.entity],
            config=config,
            additional_context=context,
            document_id=document_id,
            warnings=warnings,
            warning_context=f"group fill {group.id}",
        )
        _apply_group_attributes(group, group_doc, document_id, warnings)
        annotated_docs.append(group_doc)

    task_spec = prompt_specs["task"]
    for task in record.tasks or []:
        context = _build_target_context("Task", task.id, _extract_label(task))
        task_doc = run_langextract(
            text,
            prompt_spec=task_spec,
            field_prompts=field_prompt_maps[task_spec.entity],
            config=config,
            additional_context=context,
            document_id=document_id,
            warnings=warnings,
            warning_context=f"task fill {task.id}",
        )
        _apply_task_attributes(task, task_doc, document_id, warnings)
        annotated_docs.append(task_doc)

    modality_spec = prompt_specs["modality"]
    for modality in record.modalities or []:
        context = _build_target_context("Modality", modality.id, _extract_label(modality))
        modality_doc = run_langextract(
            text,
            prompt_spec=modality_spec,
            field_prompts=field_prompt_maps[modality_spec.entity],
            config=config,
            additional_context=context,
            document_id=document_id,
            warnings=warnings,
            warning_context=f"modality fill {modality.id}",
        )
        _apply_modality_attributes(modality, modality_doc, document_id, warnings)
        annotated_docs.append(modality_doc)

    analysis_spec = prompt_specs["analysis"]
    for analysis in record.analyses or []:
        context = _build_target_context("Analysis", analysis.id, _extract_label(analysis))
        analysis_doc = run_langextract(
            text,
            prompt_spec=analysis_spec,
            field_prompts=field_prompt_maps[analysis_spec.entity],
            config=config,
            additional_context=context,
            document_id=document_id,
            warnings=warnings,
            warning_context=f"analysis fill {analysis.id}",
        )
        _apply_analysis_attributes(analysis, analysis_doc, document_id, warnings)
        annotated_docs.append(analysis_doc)

    return annotated_docs


def _build_groups(
    extractions: list[Any], document_id: str, warnings: list[StrField]
) -> list[GroupBase]:
    groups: list[GroupBase] = []
    grouped: dict[str, list[Any]] = {}
    order: list[str] = []
    for idx, extraction in enumerate(extractions):
        attrs = extraction.attributes or {}
        label = _normalize_text(attrs.get("cohort_label")) or extraction.extraction_text
        key = _dedupe_key(label) or f"__group_{idx}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(extraction)
    for idx, key in enumerate(order, start=1):
        extraction_group = grouped[key]
        group = GroupBase(id=f"G{idx}")
        attrs = extraction_group[0].attributes or {}
        label = _normalize_text(attrs.get("cohort_label")) or extraction_group[0].extraction_text
        evidence = _collect_evidence(extraction_group, document_id)
        if label:
            _safe_setattr(
                group,
                "cohort_label",
                _make_str_field_with_evidence(label, evidence),
                warnings,
                extraction_group[0],
                document_id,
                group.id,
                prefer_existing=False,
            )
        groups.append(group)
    return groups


def _build_tasks(
    extractions: list[Any], document_id: str, warnings: list[StrField]
) -> list[TaskBase]:
    tasks: list[TaskBase] = []
    condition_counter = 1
    grouped: dict[str, list[Any]] = {}
    order: list[str] = []
    for idx, extraction in enumerate(extractions):
        attrs = extraction.attributes or {}
        name = _normalize_text(attrs.get("task_name")) or extraction.extraction_text
        key = _dedupe_key(name) or f"__task_{idx}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(extraction)
    for idx, key in enumerate(order, start=1):
        extraction_group = grouped[key]
        task = TaskBase(id=f"T{idx}")
        attrs = extraction_group[0].attributes or {}
        name = _normalize_text(attrs.get("task_name")) or extraction_group[0].extraction_text
        evidence = _collect_evidence(extraction_group, document_id)
        if name:
            _safe_setattr(
                task,
                "task_name",
                _make_str_field_with_evidence(name, evidence),
                warnings,
                extraction_group[0],
                document_id,
                task.id,
                prefer_existing=False,
            )
        condition_list: list[Condition] = []
        seen_conditions: set[str] = set()
        for extraction in extraction_group:
            attrs = extraction.attributes or {}
            conditions = _extract_list(attrs.get("conditions"))
            if not conditions:
                continue
            for label in conditions:
                key_label = _dedupe_key(label)
                if key_label and key_label in seen_conditions:
                    continue
                if key_label:
                    seen_conditions.add(key_label)
                condition_data = {
                    "id": f"C{condition_counter}",
                    "condition_label": _make_str_field(label, extraction, document_id),
                }
                condition = _safe_model_init(
                    Condition,
                    condition_data,
                    warnings,
                    extraction,
                    document_id,
                    task.id,
                    field_prefix="Condition",
                )
                if condition is not None:
                    condition_list.append(condition)
                condition_counter += 1
        if condition_list:
            _safe_setattr(
                task,
                "conditions",
                condition_list,
                warnings,
                extraction_group[0],
                document_id,
                task.id,
                prefer_existing=False,
            )
        tasks.append(task)
    return tasks


def _build_modalities(
    extractions: list[Any],
    document_id: str,
    warnings: list[StrField],
) -> list[ModalityBase]:
    modalities: list[ModalityBase] = []
    grouped: dict[str, list[Any]] = {}
    order: list[str] = []
    for idx, extraction in enumerate(extractions):
        attrs = extraction.attributes or {}
        raw_family = _normalize_text(attrs.get("modality_family"))
        raw_subtype = _normalize_text(attrs.get("modality_subtype"))
        family_key = _dedupe_key(raw_family)
        subtype_key = _dedupe_key(raw_subtype)
        if family_key or subtype_key:
            key = f"{family_key or ''}|{subtype_key or ''}"
        else:
            key = f"__modality_{idx}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(extraction)
    for idx, key in enumerate(order, start=1):
        extraction_group = grouped[key]
        modality = ModalityBase(id=f"M{idx}")
        attrs = extraction_group[0].attributes or {}
        raw_family = _normalize_text(attrs.get("modality_family"))
        raw_subtype = _normalize_text(attrs.get("modality_subtype"))
        family, warning = _normalize_modality_family(raw_family)
        if warning:
            _record_warning(warnings, warning, extraction_group[0], document_id, modality.id)
        subtype = _normalize_text(raw_subtype)
        evidence = _collect_evidence(extraction_group, document_id)
        if family or subtype:
            modality_type = _safe_model_init(
                ModalityType,
                {
                    "family": _make_extracted_value_with_evidence(family, evidence),
                    "subtype": _make_str_field_with_evidence(subtype, evidence),
                },
                warnings,
                extraction_group[0],
                document_id,
                modality.id,
                field_prefix="ModalityType",
            )
            if modality_type is not None:
                _safe_setattr(
                    modality,
                    "modality_type",
                    modality_type,
                    warnings,
                    extraction_group[0],
                    document_id,
                    modality.id,
                    prefer_existing=False,
                )
        modalities.append(modality)
    return modalities


def _build_analyses(
    extractions: list[Any], document_id: str, warnings: list[StrField]
) -> list[AnalysisBase]:
    analyses: list[AnalysisBase] = []
    grouped: dict[str, list[Any]] = {}
    order: list[str] = []
    for idx, extraction in enumerate(extractions):
        attrs = extraction.attributes or {}
        label = _normalize_text(attrs.get("contrast_formula")) or extraction.extraction_text
        key = _dedupe_key(label) or f"__analysis_{idx}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(extraction)
    for idx, key in enumerate(order, start=1):
        extraction_group = grouped[key]
        analysis = AnalysisBase(id=f"A{idx}")
        attrs = extraction_group[0].attributes or {}
        label = _normalize_text(attrs.get("contrast_formula")) or extraction_group[0].extraction_text
        evidence = _collect_evidence(extraction_group, document_id)
        if label:
            _safe_setattr(
                analysis,
                "contrast_formula",
                _make_str_field_with_evidence(label, evidence),
                warnings,
                extraction_group[0],
                document_id,
                analysis.id,
                prefer_existing=False,
            )
        analyses.append(analysis)
    return analyses


def _build_study_metadata(
    extractions: list[Any], document_id: str, warnings: list[StrField]
) -> StudyMetadataModel | None:
    if not extractions:
        return None
    extraction = extractions[0]
    attrs = extraction.attributes or {}
    study = StudyMetadataModel()
    _safe_setattr(
        study,
        "study_objective",
        _make_str_field(_normalize_text(attrs.get("study_objective")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        None,
    )
    _safe_setattr(
        study,
        "study_type",
        _make_extracted_value(_normalize_text(attrs.get("study_type")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        None,
    )
    _safe_setattr(
        study,
        "inclusion_criteria",
        _make_str_list(attrs.get("inclusion_criteria"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        None,
    )
    _safe_setattr(
        study,
        "exclusion_criteria",
        _make_str_list(attrs.get("exclusion_criteria"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        None,
    )
    if _is_empty_model(study):
        return None
    return study


def _build_shared_demographics(
    extractions: list[Any], document_id: str, warnings: list[StrField]
) -> SharedDemographics | None:
    if not extractions:
        return None
    extraction = extractions[0]
    attrs = extraction.attributes or {}
    shared = SharedDemographics()
    _safe_setattr(
        shared,
        "pooled_count",
        _make_int_field(attrs.get("pooled_count"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        None,
    )
    _safe_setattr(
        shared,
        "pooled_age_mean",
        _make_float_field(attrs.get("pooled_age_mean"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        None,
    )
    _safe_setattr(
        shared,
        "pooled_age_sd",
        _make_float_field(attrs.get("pooled_age_sd"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        None,
    )
    _safe_setattr(
        shared,
        "pooled_all_right_handed",
        _make_bool_field(attrs.get("pooled_all_right_handed"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        None,
    )
    if _is_empty_model(shared):
        return None
    return shared


def _apply_group_attributes(
    group: GroupBase,
    annotated_doc: Any,
    document_id: str,
    warnings: list[StrField],
) -> None:
    extraction = _first_extraction(annotated_doc)
    if not extraction:
        return
    attrs = extraction.attributes or {}
    _safe_setattr(
        group,
        "population_role",
        _make_extracted_value(_normalize_text(attrs.get("population_role")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "cohort_label",
        _make_str_field(_normalize_text(attrs.get("cohort_label")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "medical_condition",
        _make_str_field(_normalize_text(attrs.get("medical_condition")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "count",
        _make_int_field(attrs.get("count"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "male_count",
        _make_int_field(attrs.get("male_count"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "female_count",
        _make_int_field(attrs.get("female_count"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "age_mean",
        _make_float_field(attrs.get("age_mean"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "age_sd",
        _make_float_field(attrs.get("age_sd"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "age_range",
        _make_str_field(_normalize_text(attrs.get("age_range")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "age_minimum",
        _make_int_field(attrs.get("age_minimum"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "age_maximum",
        _make_int_field(attrs.get("age_maximum"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "age_median",
        _make_float_field(attrs.get("age_median"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "all_right_handed",
        _make_bool_field(attrs.get("all_right_handed"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "inclusion_criteria",
        _make_str_list(attrs.get("inclusion_criteria"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )
    _safe_setattr(
        group,
        "exclusion_criteria",
        _make_str_list(attrs.get("exclusion_criteria"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        group.id,
    )


def _apply_task_attributes(
    task: TaskBase,
    annotated_doc: Any,
    document_id: str,
    warnings: list[StrField],
) -> None:
    extraction = _first_extraction(annotated_doc)
    if not extraction:
        return
    attrs = extraction.attributes or {}
    _safe_setattr(
        task,
        "resting_state",
        _make_bool_field(attrs.get("resting_state"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        task.id,
    )
    _safe_setattr(
        task,
        "task_name",
        _make_str_field(_normalize_text(attrs.get("task_name")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        task.id,
    )
    _safe_setattr(
        task,
        "task_description",
        _make_str_field(_normalize_text(attrs.get("task_description")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        task.id,
    )
    _safe_setattr(
        task,
        "design_details",
        _make_str_field(_normalize_text(attrs.get("design_details")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        task.id,
    )
    _safe_setattr(
        task,
        "task_design",
        _make_enum_list(attrs.get("task_design"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        task.id,
    )
    _safe_setattr(
        task,
        "task_duration",
        _make_str_field(_normalize_text(attrs.get("task_duration")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        task.id,
    )
    _safe_setattr(
        task,
        "concepts",
        _make_str_list(attrs.get("concepts"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        task.id,
    )
    _safe_setattr(
        task,
        "domain_tags",
        _make_enum_list(attrs.get("domain_tags"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        task.id,
    )
    if not task.conditions:
        conditions = _extract_list(attrs.get("conditions"))
        if conditions:
            condition_list: list[Condition] = []
            next_id = 1
            for label in conditions:
                condition = _safe_model_init(
                    Condition,
                    {
                        "id": f"C{next_id}",
                        "condition_label": _make_str_field(label, extraction, document_id),
                    },
                    warnings,
                    extraction,
                    document_id,
                    task.id,
                    field_prefix="Condition",
                )
                if condition is not None:
                    condition_list.append(condition)
                next_id += 1
            if condition_list:
                _safe_setattr(
                    task,
                    "conditions",
                    condition_list,
                    warnings,
                    extraction,
                    document_id,
                    task.id,
                )


def _apply_modality_attributes(
    modality: ModalityBase,
    annotated_doc: Any,
    document_id: str,
    warnings: list[StrField],
) -> None:
    extraction = _first_extraction(annotated_doc)
    if not extraction:
        return
    attrs = extraction.attributes or {}
    raw_family = _normalize_text(attrs.get("modality_family"))
    family, warning = _normalize_modality_family(raw_family)
    if warning:
        _record_warning(warnings, warning, extraction, document_id, modality.id)
    subtype = _normalize_text(attrs.get("modality_subtype"))
    if family or subtype:
        modality_type = _safe_model_init(
            ModalityType,
            {
                "family": _make_extracted_value(family, extraction, document_id),
                "subtype": _make_str_field(subtype, extraction, document_id),
            },
            warnings,
            extraction,
            document_id,
            modality.id,
            field_prefix="ModalityType",
        )
        if modality_type is not None:
            _safe_setattr(
                modality,
                "modality_type",
                modality_type,
                warnings,
                extraction,
                document_id,
                modality.id,
            )
    _safe_setattr(
        modality,
        "manufacturer",
        _make_str_field(_normalize_text(attrs.get("manufacturer")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        modality.id,
    )
    _safe_setattr(
        modality,
        "field_strength_tesla",
        _make_float_field(attrs.get("field_strength_tesla"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        modality.id,
    )
    _safe_setattr(
        modality,
        "sequence_name",
        _make_str_field(_normalize_text(attrs.get("sequence_name")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        modality.id,
    )
    _safe_setattr(
        modality,
        "voxel_size",
        _make_str_field(_normalize_text(attrs.get("voxel_size")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        modality.id,
    )
    _safe_setattr(
        modality,
        "tr_seconds",
        _make_float_field(attrs.get("tr_seconds"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        modality.id,
    )
    _safe_setattr(
        modality,
        "te_seconds",
        _make_float_field(attrs.get("te_seconds"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        modality.id,
    )


def _apply_analysis_attributes(
    analysis: AnalysisBase,
    annotated_doc: Any,
    document_id: str,
    warnings: list[StrField],
) -> None:
    extraction = _first_extraction(annotated_doc)
    if not extraction:
        return
    attrs = extraction.attributes or {}
    _safe_setattr(
        analysis,
        "reporting_scope",
        _make_extracted_value(_normalize_text(attrs.get("reporting_scope")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        analysis.id,
    )
    _safe_setattr(
        analysis,
        "study_design_tags",
        _make_enum_list(attrs.get("study_design_tags"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        analysis.id,
    )
    _safe_setattr(
        analysis,
        "statistical_model",
        _make_str_field(_normalize_text(attrs.get("statistical_model")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        analysis.id,
    )
    _safe_setattr(
        analysis,
        "contrast_formula",
        _make_str_field(_normalize_text(attrs.get("contrast_formula")), extraction, document_id),
        warnings,
        extraction,
        document_id,
        analysis.id,
    )
    _safe_setattr(
        analysis,
        "outcome_measures",
        _make_str_list(attrs.get("outcome_measures"), extraction, document_id),
        warnings,
        extraction,
        document_id,
        analysis.id,
    )


def _parse_links(
    extractions: list[Any],
    document_id: str,
    warnings: list[StrField],
) -> StudyLinks | None:
    if not extractions:
        return None
    links = StudyLinks(
        group_task=[],
        task_modality=[],
        analysis_task=[],
        analysis_group=[],
        analysis_condition=[],
        group_modality=[],
    )
    for extraction in extractions:
        attrs = extraction.attributes or {}
        edge_type = _normalize_text(attrs.get("edge_type"))
        source_id = _normalize_text(attrs.get("source_id"))
        target_id = _normalize_text(attrs.get("target_id"))
        if not edge_type or not source_id or not target_id:
            continue
        evidence = _make_evidence(extraction, document_id)
        edge_id = source_id or target_id
        if edge_type == "group_task":
            edge = _safe_model_init(
                GroupTaskEdge,
                {"group_id": source_id, "task_id": target_id, "evidence": evidence},
                warnings,
                extraction,
                document_id,
                edge_id,
                field_prefix="StudyLinks.group_task",
            )
            if edge is not None:
                links.group_task.append(edge)
        elif edge_type == "task_modality":
            edge = _safe_model_init(
                TaskModalityEdge,
                {"task_id": source_id, "modality_id": target_id, "evidence": evidence},
                warnings,
                extraction,
                document_id,
                edge_id,
                field_prefix="StudyLinks.task_modality",
            )
            if edge is not None:
                links.task_modality.append(edge)
        elif edge_type == "analysis_task":
            edge = _safe_model_init(
                AnalysisTaskEdge,
                {"analysis_id": source_id, "task_id": target_id, "evidence": evidence},
                warnings,
                extraction,
                document_id,
                edge_id,
                field_prefix="StudyLinks.analysis_task",
            )
            if edge is not None:
                links.analysis_task.append(edge)
        elif edge_type == "analysis_group":
            edge = _safe_model_init(
                AnalysisGroupEdge,
                {"analysis_id": source_id, "group_id": target_id, "evidence": evidence},
                warnings,
                extraction,
                document_id,
                edge_id,
                field_prefix="StudyLinks.analysis_group",
            )
            if edge is not None:
                links.analysis_group.append(edge)
        elif edge_type == "analysis_condition":
            edge = _safe_model_init(
                AnalysisConditionEdge,
                {"analysis_id": source_id, "condition_id": target_id, "evidence": evidence},
                warnings,
                extraction,
                document_id,
                edge_id,
                field_prefix="StudyLinks.analysis_condition",
            )
            if edge is not None:
                links.analysis_condition.append(edge)
        elif edge_type == "group_modality":
            edge = _safe_model_init(
                GroupModalityEdge,
                {
                    "group_id": source_id,
                    "modality_id": target_id,
                    "n_scanned": _make_int_field(
                        attrs.get("n_scanned"), extraction, document_id
                    ),
                    "evidence": evidence,
                },
                warnings,
                extraction,
                document_id,
                edge_id,
                field_prefix="StudyLinks.group_modality",
            )
            if edge is not None:
                links.group_modality.append(edge)
    if _links_empty(links):
        return None
    return links


def _links_empty(links: StudyLinks) -> bool:
    return not any(
        [
            links.group_task,
            links.task_modality,
            links.analysis_task,
            links.analysis_group,
            links.analysis_condition,
            links.group_modality,
        ]
    )


def _first_extraction(annotated_doc: Any) -> Any | None:
    if not annotated_doc:
        return None
    extractions = getattr(annotated_doc, "extractions", None)
    if not extractions:
        return None
    return extractions[0]


def _make_evidence(extraction: Any, document_id: str | None) -> list[EvidenceSpan] | None:
    if extraction is None:
        return None
    char_interval = None
    if getattr(extraction, "char_interval", None) is not None:
        char_interval = CharInterval(
            start_pos=extraction.char_interval.start_pos,
            end_pos=extraction.char_interval.end_pos,
        )
    alignment_status = getattr(extraction, "alignment_status", None)
    if alignment_status is not None and hasattr(alignment_status, "value"):
        alignment_status = alignment_status.value
    return [
        EvidenceSpan(
            extraction_text=getattr(extraction, "extraction_text", None),
            char_interval=char_interval,
            alignment_status=alignment_status,
            extraction_index=getattr(extraction, "extraction_index", None),
            group_index=getattr(extraction, "group_index", None),
            document_id=document_id,
        )
    ]


def _collect_evidence(extractions: list[Any], document_id: str | None) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for extraction in extractions:
        evidence = _make_evidence(extraction, document_id)
        if evidence:
            spans.extend(evidence)
    return spans


def _make_str_field(value: str | None, extraction: Any, document_id: str) -> StrField | None:
    value = _normalize_text(value)
    if value is None:
        return None
    return StrField(value=value, evidence=_make_evidence(extraction, document_id))


def _make_int_field(value: Any, extraction: Any, document_id: str) -> IntField | None:
    parsed = _parse_int(value)
    if parsed is None:
        return None
    return IntField(value=parsed, evidence=_make_evidence(extraction, document_id))


def _make_float_field(value: Any, extraction: Any, document_id: str) -> FloatField | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return FloatField(value=parsed, evidence=_make_evidence(extraction, document_id))


def _make_bool_field(value: Any, extraction: Any, document_id: str) -> BoolField | None:
    parsed = _parse_bool(value)
    if parsed is None:
        return None
    return BoolField(value=parsed, evidence=_make_evidence(extraction, document_id))


def _make_extracted_value(
    value: Any, extraction: Any, document_id: str
) -> ExtractedValue | None:
    value = _normalize_text(value)
    if value is None:
        return None
    return ExtractedValue(value=value, evidence=_make_evidence(extraction, document_id))


def _make_str_field_with_evidence(
    value: str | None, evidence: list[EvidenceSpan]
) -> StrField | None:
    value = _normalize_text(value)
    if value is None or not evidence:
        return None
    return StrField(value=value, evidence=evidence)


def _make_extracted_value_with_evidence(
    value: Any, evidence: list[EvidenceSpan]
) -> ExtractedValue | None:
    value = _normalize_text(value)
    if value is None or not evidence:
        return None
    return ExtractedValue(value=value, evidence=evidence)


def _make_str_list(
    value: Any, extraction: Any, document_id: str
) -> list[StrField] | None:
    values = _extract_list(value)
    if not values:
        return None
    return [StrField(value=item, evidence=_make_evidence(extraction, document_id)) for item in values]


def _make_enum_list(
    value: Any, extraction: Any, document_id: str
) -> list[ExtractedValue] | None:
    values = _extract_list(value)
    if not values:
        return None
    return [
        ExtractedValue(value=item, evidence=_make_evidence(extraction, document_id))
        for item in values
    ]


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "null", "none", "n/a", "na"}:
            return None
        return cleaned
    return str(value)


def _dedupe_key(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().lower().split())
    return cleaned or None


def _normalize_modality_family(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    allowed = {str(v) for v in get_args(ModalityFamilyLiteral)}
    normalized = value.strip()
    normalized_lower = normalized.lower()
    allowed_lower = {v.lower(): v for v in allowed}
    if normalized_lower in allowed_lower:
        return allowed_lower[normalized_lower], None
    synonym_map = {
        "fmri": "fMRI",
        "f mri": "fMRI",
        "functional mri": "fMRI",
        "structural mri": "StructuralMRI",
        "diffusion mri": "DiffusionMRI",
        "dti": "DiffusionMRI",
        "pet": "PET",
        "eeg": "EEG",
        "meg": "MEG",
    }
    mapped = synonym_map.get(normalized_lower)
    if mapped and mapped in allowed:
        return mapped, None
    return None, f"Invalid modality_type.family '{value}' (allowed: {sorted(allowed)})."


def _extract_status_code(exc: BaseException) -> int | None:
    original = getattr(exc, "original", None)
    if original is None:
        return None
    for attr in ("status_code", "code"):
        value = getattr(original, attr, None)
        if isinstance(value, int):
            return value
    match = re.search(r"\b(429|500|502|503|504)\b", str(original))
    if match:
        return int(match.group(1))
    return None


def _safe_setattr(
    model: Any,
    field_name: str,
    value: Any,
    warnings: list[StrField],
    extraction: Any,
    document_id: str,
    entity_id: str | None,
    *,
    prefer_existing: bool = True,
) -> None:
    if value is None:
        return
    if prefer_existing and getattr(model, field_name, None) is not None:
        return
    if _validate_field_value(
        model,
        field_name,
        value,
        warnings,
        extraction,
        document_id,
        entity_id,
    ):
        return
    setattr(model, field_name, value)


def _validate_field_value(
    model: Any,
    field_name: str,
    value: Any,
    warnings: list[StrField],
    extraction: Any,
    document_id: str,
    entity_id: str | None,
) -> bool:
    annotation = _get_field_annotation(model, field_name)
    if annotation is None or TypeAdapter is None:
        return False
    adapter = TypeAdapter(annotation)
    try:
        adapter.validate_python(value)
        return False
    except ValidationError as exc:
        _record_validation_errors(
            warnings,
            model.__class__.__name__,
            field_name,
            exc,
            extraction,
            document_id,
            entity_id,
        )
        return True


def _get_field_annotation(model: Any, field_name: str) -> Any | None:
    fields = getattr(model, "model_fields", None)
    if fields and field_name in fields:
        return fields[field_name].annotation
    fields = getattr(model, "__fields__", None)
    if fields and field_name in fields:
        return fields[field_name].outer_type_
    return None


def _safe_model_init(
    model_cls: Any,
    data: dict[str, Any],
    warnings: list[StrField],
    extraction: Any,
    document_id: str,
    entity_id: str | None,
    field_prefix: str | None = None,
) -> Any | None:
    try:
        return model_cls(**data)
    except ValidationError as exc:
        prefix = field_prefix or model_cls.__name__
        _record_validation_errors(
            warnings,
            prefix,
            "",
            exc,
            extraction,
            document_id,
            entity_id,
        )
        invalid_fields = set()
        for err in exc.errors():
            loc = err.get("loc") or ()
            if loc:
                invalid_fields.add(loc[0])
        cleaned = {key: val for key, val in data.items() if key not in invalid_fields}
        if not cleaned:
            return None
        try:
            return model_cls(**cleaned)
        except ValidationError as exc2:
            _record_validation_errors(
                warnings,
                prefix,
                "",
                exc2,
                extraction,
                document_id,
                entity_id,
            )
            return None


def _record_validation_errors(
    warnings: list[StrField],
    model_name: str,
    field_name: str,
    exc: ValidationError,
    extraction: Any,
    document_id: str,
    entity_id: str | None,
) -> None:
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc") or ())
        target = field_name
        if loc:
            target = f"{field_name}.{loc}" if field_name else loc
        if target:
            message = f"Invalid {model_name}.{target}: {err.get('msg')}"
        else:
            message = f"Invalid {model_name}: {err.get('msg')}"
        _record_warning(warnings, message, extraction, document_id, entity_id)


def _record_warning(
    warnings: list[StrField],
    message: str,
    extraction: Any,
    document_id: str,
    entity_id: str | None,
) -> None:
    if entity_id:
        message = f"{entity_id}: {message}"
    logger.warning(message)
    warning_field = _make_warning_field(message, extraction, document_id)
    if warning_field is None:
        return
    warnings.append(warning_field)


def _make_warning_field(
    message: str, extraction: Any, document_id: str
) -> StrField | None:
    if extraction is None:
        evidence = [EvidenceSpan(document_id=document_id)]
        return StrField(value=message, evidence=evidence)
    return _make_str_field(message, extraction, document_id)


def _extract_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "null", "none", "n/a", "na"}:
            return None
        if "," in cleaned or ";" in cleaned or "|" in cleaned:
            parts = re.split(r"[;,|]", cleaned)
            items = [part.strip() for part in parts if part.strip()]
            return items or None
        return [cleaned]
    return [str(value)]


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value)
    match = re.search(r"-?\\d+", text.replace(",", ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = str(value).replace(",", "")
    match = re.search(r"-?\\d+(?:\\.\\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def _extract_label(entity: Any) -> str:
    if isinstance(entity, GroupBase):
        if entity.cohort_label and entity.cohort_label.value:
            return entity.cohort_label.value
    if isinstance(entity, TaskBase):
        if entity.task_name and entity.task_name.value:
            return entity.task_name.value
    if isinstance(entity, ModalityBase):
        if entity.modality_type and entity.modality_type.family:
            return str(entity.modality_type.family.value)
    if isinstance(entity, AnalysisBase):
        if entity.contrast_formula and entity.contrast_formula.value:
            return entity.contrast_formula.value
    return entity.id


def _build_target_context(entity_type: str, entity_id: str, label: str) -> str:
    return "\n".join(
        [
            f"Target {entity_type}:",
            f"- id: {entity_id}",
            f"- label: {label}",
            "Extract attributes only for this entity. Return null for unknown fields.",
        ]
    )


def _build_links_context(record: StudyRecord) -> str:
    lines = ["Known entity IDs for linking:"]
    if record.demographics and record.demographics.groups:
        lines.append("Groups:")
        for group in record.demographics.groups:
            lines.append(f"- {group.id}: {_extract_label(group)}")
    if record.tasks:
        lines.append("Tasks:")
        for task in record.tasks:
            lines.append(f"- {task.id}: {_extract_label(task)}")
            if task.conditions:
                for condition in task.conditions:
                    label = condition.condition_label.value if condition.condition_label else condition.id
                    lines.append(f"  - {condition.id}: {label}")
    if record.modalities:
        lines.append("Modalities:")
        for modality in record.modalities:
            lines.append(f"- {modality.id}: {_extract_label(modality)}")
    if record.analyses:
        lines.append("Analyses:")
        for analysis in record.analyses:
            lines.append(f"- {analysis.id}: {_extract_label(analysis)}")
    lines.append(
        "Use these IDs in edge attributes (edge_type, source_id, target_id)."
    )
    return "\n".join(lines)


def _combine_annotated_docs(
    text: str, document_id: str, docs: list[Any]
) -> Any:
    from langextract.core import data as lx_data

    all_extractions = []
    for doc in docs:
        if not doc:
            continue
        all_extractions.extend(getattr(doc, "extractions", []) or [])
    return lx_data.AnnotatedDocument(
        document_id=document_id,
        text=text,
        extractions=all_extractions,
    )


def _write_outputs(
    run_dir: Path, document_id: str, record: StudyRecord, annotated_doc: Any
) -> None:
    doc_dir = run_dir / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    record_path = doc_dir / "study_record.json"
    record_path.write_text(_dump_model(record), encoding="utf-8")

    from langextract import io as lx_io

    jsonl_path = doc_dir / "langextract.jsonl"
    lx_io.save_annotated_documents(
        iter([annotated_doc]),
        output_dir=doc_dir,
        output_name=jsonl_path.name,
        show_progress=False,
    )

    try:
        from information_extraction.review.editable_visualization import (
            visualize_editable,
        )

        html = visualize_editable(
            annotated_doc,
            output_filename=f"{document_id}.review.jsonl",
            show_legend=True,
            gif_optimized=False,
        )
        html_path = doc_dir / "review.html"
        html_path.write_text(str(html), encoding="utf-8")
    except Exception:
        logger.exception("Failed to generate review HTML for %s", document_id)


def _write_aggregate_outputs(
    run_dir: Path,
    records: list[StudyRecord],
    annotated_docs: list[Any],
) -> None:
    aggregate_path = run_dir / "study_records.jsonl"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_dump_model(record, jsonl=True))
            handle.write("\n")

    from langextract import io as lx_io

    lx_io.save_annotated_documents(
        iter(annotated_docs),
        output_dir=run_dir,
        output_name="langextract.jsonl",
        show_progress=False,
    )


def _dump_model(model: Any, jsonl: bool = False) -> str:
    if hasattr(model, "model_dump"):
        payload = model.model_dump(exclude_none=True)
    else:
        payload = model.dict(exclude_none=True)
    return json.dumps(payload, ensure_ascii=False)


def _is_empty_model(model: Any) -> bool:
    if hasattr(model, "model_dump"):
        payload = model.model_dump(exclude_none=True)
    else:
        payload = model.dict(exclude_none=True)
    return len(payload) == 0
