from __future__ import annotations

from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from pydantic import BaseModel, Field

try:
    from pydantic import model_validator
except ImportError:  # pragma: no cover - pydantic v1 fallback
    model_validator = None
    from pydantic import root_validator
from typing_extensions import Literal


# -----------------------------
# Extraction metadata helpers
# -----------------------------
ExtractionTypeLiteral = Literal[
    "generated",
    "text",
    "enum",
    "numeric",
    "boolean",
    "summary",
    "object",
    "object_list",
    "id_reference",
    "id_reference_list",
    "edge_list",
]
ExtractionTypeSpec = Union[ExtractionTypeLiteral, List[ExtractionTypeLiteral]]

ExtractionPhaseLiteral = Literal[
    "entity",
    "study",
    "group",
    "task",
    "condition",
    "modality",
    "analysis",
    "linking",
    "record",
]

InferencePolicyLiteral = Literal[
    "explicit_only",
    "synthesize",
    "post_normalize",
]

AlignmentStatusLiteral = Literal[
    "match_exact",
    "match_greater",
    "match_lesser",
    "match_fuzzy",
]

def extraction_meta(
    extraction_type: ExtractionTypeSpec,
    prompt: str,
    scope_hint: Optional[str] = None,
    extraction_phase: Optional[ExtractionPhaseLiteral] = None,
    allow_note: bool = False,
    inference_policy: InferencePolicyLiteral = "explicit_only",
    evidence_value_only: Optional[bool] = None,
    evidence_hypothesis_template: Optional[str] = None,
) -> dict:
    meta: Dict[str, Any] = {
        "extraction_type": extraction_type,
        "extraction_prompt": prompt,
    }
    if scope_hint:
        meta["scope_hint"] = scope_hint
    if extraction_phase:
        meta["extraction_phase"] = extraction_phase
    meta["allow_note"] = allow_note
    meta["inference_policy"] = inference_policy
    if evidence_value_only is not None:
        meta["evidence_value_only"] = evidence_value_only
    if evidence_hypothesis_template:
        meta["evidence_hypothesis_template"] = evidence_hypothesis_template
    return meta


# -----------------------------
# Provenance / evidence model
# -----------------------------
class CharInterval(BaseModel):
    start_pos: Optional[int] = Field(
        default=None, description="Character start offset in the normalized text."
    )
    end_pos: Optional[int] = Field(
        default=None, description="Character end offset in the normalized text."
    )


class EvidenceSpan(BaseModel):
    source: Literal["text", "table", "figure_caption", "supplement", "other"] = "text"
    section: Optional[str] = Field(default=None, description="Section path if known.")
    extraction_text: Optional[str] = Field(
        default=None,
        description="Verbatim text span supporting the value.",
    )
    char_interval: Optional[CharInterval] = Field(
        default=None, description="Character interval for the evidence span."
    )
    alignment_status: Optional[AlignmentStatusLiteral] = Field(
        default=None, description="Alignment status reported by LangExtract."
    )
    extraction_index: Optional[int] = Field(
        default=None, description="Extraction index reported by LangExtract."
    )
    group_index: Optional[int] = Field(
        default=None, description="Group index reported by LangExtract."
    )
    document_id: Optional[str] = Field(
        default=None, description="Source document ID for this evidence span."
    )
    locator: Optional[str] = Field(
        default=None,
        description="Optional locator such as sentence_id/table_id/figure_id.",
    )


T = TypeVar("T")


class ExtractedValue(BaseModel, Generic[T]):
    """
    Wrapper for an extracted value + provenance.

    NOTE: Pydantic generics are optional; if you prefer simplicity,
    use ExtractedValue with `value: Any` everywhere.
    """

    value: Optional[T] = Field(default=None, description="Extracted value.")
    evidence: Optional[List[EvidenceSpan]] = Field(
        default=None, description="Evidence spans supporting the value."
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit for numeric values if stated.",
    )
    missing_reason: Optional[str] = Field(
        default=None,
        description="Reason value is missing.",
    )
    note: Optional[str] = Field(
        default=None,
        description="Free-text note.",
    )
    scope: Optional[Literal["group", "shared", "task", "modality", "analysis", "study"]] = Field(
        default=None, description="Scope of the value if relevant."
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Optional confidence score (0-1) if computed by your pipeline.",
        ge=0.0,
        le=1.0,
    )

    if model_validator:

        @model_validator(mode="after")
        def _require_evidence(self):
            if self.evidence is not None and len(self.evidence) == 0:
                raise ValueError("evidence must include at least one span when provided.")
            if self.value is not None and not self.evidence:
                raise ValueError("evidence is required when value is provided.")
            return self

    else:

        @root_validator
        def _require_evidence(cls, values):
            value = values.get("value")
            evidence = values.get("evidence")
            if evidence is not None and len(evidence) == 0:
                raise ValueError("evidence must include at least one span when provided.")
            if value is not None and not evidence:
                raise ValueError("evidence is required when value is provided.")
            return values


# Convenience typed wrappers (optional but nice for clarity)
class IntField(ExtractedValue[int]): ...
class FloatField(ExtractedValue[float]): ...
class StrField(ExtractedValue[str]): ...
class BoolField(ExtractedValue[bool]): ...


# -----------------------------
# Entity bases
# -----------------------------
class EntityBase(BaseModel):
    id: str = Field(
        ...,
        description="Stable ID within this paper, assigned by pipeline logic.",
        json_schema_extra=extraction_meta(
            "generated",
            "Assigned by pipeline logic; do not extract from text.",
            extraction_phase="entity",
        ),
    )


# -----------------------------
# Controlled vocabulary helpers
# -----------------------------
class ModalityFamily(str):
    pass


ModalityFamilyLiteral = Literal[
    "fMRI",
    "StructuralMRI",
    "DiffusionMRI",
    "PET",
    "EEG",
    "MEG",
    "BehaviorOnly",
    "Other",
]

# Use subtype for PET tracers, fMRI flavors, etc.
# Keep subtype within limited suggestions but allow unknown strings.
PETSubtypeLiteral = Literal["FDG", "15O-water", "Other"]
FMRISubtypeLiteral = Literal["BOLD", "CBF", "CBV", "Other"]


class ModalityType(BaseModel):
    family: Optional[ExtractedValue[ModalityFamilyLiteral]] = Field(
        default=None,
        description="Modality family.",
        json_schema_extra=extraction_meta(
            "enum",
            "Choose the modality family explicitly used in this study. "
            "Use Other only if a modality is stated but not covered.",
            scope_hint="Methods/Imaging, Abstract, Results, Captions",
            extraction_phase="modality",
            inference_policy="post_normalize",
            evidence_hypothesis_template="{context_prefix}modality family was {value}.",
        ),
    )
    family_other: Optional[StrField] = Field(
        default=None,
        description="Free-text modality family label when family is Other.",
        json_schema_extra=extraction_meta(
            "text",
            "If family is Other, capture the stated modality family label here. Otherwise null.",
            scope_hint="Methods/Imaging, Abstract, Results, Captions",
            extraction_phase="modality",
            evidence_hypothesis_template="{context_prefix}modality family was {value}.",
        ),
    )
    subtype: Optional[StrField] = Field(
        default=None,
        description="Subtype/tracer/variant if explicitly stated.",
        json_schema_extra=extraction_meta(
            ["enum", "text"],
            "If a subtype is explicitly stated (e.g., BOLD/CBF/CBV, FDG, 15O-water), "
            "populate. Otherwise null.",
            scope_hint="Methods/Imaging, Abstract, Captions",
            extraction_phase="modality",
            evidence_hypothesis_template="{context_prefix}modality subtype was {value}.",
        ),
    )


# -----------------------------
# Modality schema
# -----------------------------
class ModalityBase(EntityBase):
    # Controlled vocabulary
    modality_type: Optional[ModalityType] = Field(
        default=None,
        description="Modality family/subtype.",
        json_schema_extra=extraction_meta(
            "object",
            "Populate family/subtype if explicitly stated.",
            scope_hint="Methods/Imaging, Abstract, Results, Captions",
            extraction_phase="modality",
        ),
    )

    manufacturer: Optional[StrField] = Field(
        default=None,
        description="Scanner manufacturer.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract the manufacturer name as stated. Return null if not stated.",
            scope_hint="Methods/Imaging",
            extraction_phase="modality",
            evidence_hypothesis_template="{context_prefix}scanner manufacturer was {value}.",
        ),
    )

    field_strength_tesla: Optional[FloatField] = Field(
        default=None,
        description="Scanner field strength in Tesla.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract field strength as a number in Tesla. Return null if not stated.",
            scope_hint="Methods/Imaging",
            extraction_phase="modality",
            evidence_hypothesis_template="{context_prefix}field strength was {value} T.",
        ),
    )

    sequence_name: Optional[StrField] = Field(
        default=None,
        description="Sequence name/type.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract sequence/acquisition terminology as stated. Return null if not stated.",
            scope_hint="Methods/Imaging",
            extraction_phase="modality",
            evidence_hypothesis_template="{context_prefix}sequence was {value}.",
        ),
    )

    # Voxel size
    voxel_size: Optional[StrField] = Field(
        default=None,
        description="Voxel size.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract voxel size as stated. Return null if not stated.",
            scope_hint="Methods/Imaging",
            extraction_phase="modality",
            evidence_hypothesis_template="{context_prefix}voxel size was {value}.",
        ),
    )

    # Optional extras often needed
    tr_seconds: Optional[FloatField] = Field(
        default=None,
        description="Repetition time (TR) in seconds, if stated.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract TR. Convert ms to seconds if needed. Return null if not stated.",
            scope_hint="Methods/Imaging",
            extraction_phase="modality",
            evidence_hypothesis_template="{context_prefix}TR was {value} s.",
        ),
    )
    te_seconds: Optional[FloatField] = Field(
        default=None,
        description="Echo time (TE) in seconds, if stated.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract TE. Convert ms to seconds if needed. Return null if not stated.",
            scope_hint="Methods/Imaging",
            extraction_phase="modality",
            evidence_hypothesis_template="{context_prefix}TE was {value} s.",
        ),
    )


# -----------------------------
# Task schema
# -----------------------------
TaskDomainLiteral = Literal[
    "Perception",
    "Attention",
    "Reasoning and decision making",
    "Executive cognitive control",
    "Learning and memory",
    "Language",
    "Action",
    "Emotion",
    "Social function",
    "Motivation",
    "Other",
]

TaskDesignLiteral = Literal["Blocked", "EventRelated", "Mixed", "Other"]
TaskCategoryLiteral = Literal[
    "CognitiveTask",
    "Intervention",
    "Exposure",
    "RestingState",
    "Other",
]


class Condition(EntityBase):
    """
    Condition is an entity so that contrasts can reference it via condition_id.
    Use id like C1, C2 scoped under a task if you want, but keep globally unique within paper.
    """

    condition_label: Optional[StrField] = Field(
        default=None,
        description="Label/name of a condition.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract the condition label as stated. Return null if not stated.",
            scope_hint="Methods/Task, Results, Captions",
            extraction_phase="condition",
            evidence_hypothesis_template="The condition was {value}.",
        ),
    )


class TaskBase(EntityBase):
    task_name: Optional[StrField] = Field(
        default=None,
        description="Task name used in the study.",
        json_schema_extra=extraction_meta(
            "text",
            "Use the exact task or intervention name if stated (e.g., 'Stroop task', "
            "'TaVNS stimulation'). If no name is stated but a task/intervention is "
            "described, create a short label grounded in the text. "
            "Do not include narrative descriptions or timing details. "
            "Return null if no task/intervention is described.",
            scope_hint="Methods/Task, Abstract, Results, Captions",
            extraction_phase="task",
            inference_policy="synthesize",
            evidence_hypothesis_template="The task was {value}.",
        ),
    )

    task_description: Optional[StrField] = Field(
        default=None,
        description="Narrative task summary (what participants did).",
        json_schema_extra=extraction_meta(
            "summary",
            "Summarize what participants did, stimuli used, and goal. "
            "Avoid timing/run details (use design_details). "
            "Do not include total duration (use task_duration). "
            "Return null if not described.",
            scope_hint="Methods/Task",
            extraction_phase="task",
            inference_policy="synthesize",
            evidence_value_only=True,
        ),
    )

    design_details: Optional[StrField] = Field(
        default=None,
        description="Timing/run/block/trial scheduling details summary.",
        json_schema_extra=extraction_meta(
            "summary",
            "Extract timing/run/block/trial details (counts, durations, intervals). "
            "Avoid narrative task description (use task_description). "
            "Do not include total duration (use task_duration). "
            "Return null if not described.",
            scope_hint="Methods/Task",
            extraction_phase="task",
            inference_policy="synthesize",
            evidence_value_only=True,
        ),
    )

    task_design: Optional[List[ExtractedValue[TaskDesignLiteral]]] = Field(
        default=None,
        description="Task design type (only if explicitly stated).",
        json_schema_extra=extraction_meta(
            "enum",
            "Select design type only when it can be explicitly mapped to the existing options. Return null if not stated.",
            scope_hint="Methods/Task",
            extraction_phase="task",
            inference_policy="post_normalize",
            evidence_hypothesis_template="{context_prefix}task design was {value}.",
        ),
    )
    task_design_other: Optional[StrField] = Field(
        default=None,
        description="Free-text task design label when task_design is Other.",
        json_schema_extra=extraction_meta(
            "text",
            "If task_design includes Other, capture the stated design label here. Otherwise null.",
            scope_hint="Methods/Task",
            extraction_phase="task",
            evidence_hypothesis_template="{context_prefix}task design was {value}.",
        ),
    )

    task_category: Optional[List[ExtractedValue[TaskCategoryLiteral]]] = Field(
        default=None,
        description="Task category type (explicit only).",
        json_schema_extra=extraction_meta(
            "enum",
            "Select task category type(s) only when it can be explicitly mapped to the existing options. "
            "Use RestingState only if explicitly described as resting-state/baseline with no task. "
            "Return null if not stated.",
            scope_hint="Methods/Task, Abstract, Results, Captions",
            extraction_phase="task",
            inference_policy="post_normalize",
            evidence_hypothesis_template="{context_prefix}task category was {value}.",
        ),
    )
    task_category_other: Optional[List[StrField]] = Field(
        default=None,
        description="Free-text task category labels when task_category includes Other.",
        json_schema_extra=extraction_meta(
            "text",
            "If task_category includes Other, list the stated category labels here. Otherwise null.",
            scope_hint="Methods/Task, Abstract, Results, Captions",
            extraction_phase="task",
            evidence_hypothesis_template="{context_prefix}task category was {value}.",
        ),
    )

    task_duration: Optional[StrField] = Field(
        default=None,
        description="Total task duration.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract total task duration as stated. Do not include per-run/block timing "
            "(use design_details). Return null if not stated.",
            scope_hint="Methods/Task",
            extraction_phase="task",
            evidence_hypothesis_template="{context_prefix}total task duration was {value}.",
        ),
    )

    conditions: Optional[List[Condition]] = Field(
        default=None,
        description="Conditions for this task.",
        json_schema_extra=extraction_meta(
            "object_list",
            "Extract each task condition/variant as an object with its label "
            "(e.g., left/right/both ear stimulation, on/off, sham/active). "
            "Use conditions for variants of the same task; do not create separate tasks "
            "for variants. "
            "Return null if conditions are not stated.",
            scope_hint="Methods/Task, Results",
            extraction_phase="task",
        ),
    )

    concepts: Optional[List[StrField]] = Field(
        default=None,
        description="Cognitive psychological concepts explicitly associated with the task.",
        json_schema_extra=extraction_meta(
            "text",
            "List cognitive psychological concepts exactly as stated (e.g., attention, "
            "working memory, inhibition, conflict monitoring). "
            "Do not include general neuroscience terms, modalities, diseases, or "
            "study topics. Do not map to domain tags. "
            "Return null if not stated.",
            scope_hint="Methods/Task, Abstract, Introduction (if explicitly describing this study's task)",
            extraction_phase="task",
            evidence_hypothesis_template="{context_prefix}the task involved {value}.",
        ),
    )

    # Separate domain tags from concept text
    domain_tags: Optional[List[ExtractedValue[TaskDomainLiteral]]] = Field(
        default=None,
        description="Cognitive domain tags.",
        json_schema_extra=extraction_meta(
            "enum",
            "Select domain tags only when it can be explicitly mapped to the existing options. "
            "Do not copy raw concept phrases. Return null if not stated.",
            scope_hint="Methods/Task, Abstract",
            extraction_phase="task",
            inference_policy="post_normalize",
            evidence_hypothesis_template="{context_prefix}domain tags included {value}.",
        ),
    )
    domain_tags_other: Optional[List[StrField]] = Field(
        default=None,
        description="Free-text domain tags when domain_tags includes Other.",
        json_schema_extra=extraction_meta(
            "text",
            "If domain_tags includes Other, list the stated domain tags here. Otherwise null.",
            scope_hint="Methods/Task, Abstract",
            extraction_phase="task",
            evidence_hypothesis_template="{context_prefix}domain tags included {value}.",
        ),
    )


# -----------------------------
# Groups + demographics schema
# -----------------------------
PopulationRoleLiteral = Literal["patient", "control", "case", "comparison", "other"]


class GroupBase(EntityBase):
    population_role: Optional[ExtractedValue[PopulationRoleLiteral]] = Field(
        default=None,
        description="Group role (patient/control/etc.) if explicitly stated.",
        json_schema_extra=extraction_meta(
            "enum",
            "Select role only if explicitly stated or unambiguous from explicit wording "
            "(e.g., 'healthy controls'). Do not copy cohort labels or diagnoses. Otherwise null.",
            scope_hint="Methods/Participants, Abstract",
            extraction_phase="group",
            inference_policy="post_normalize",
            evidence_hypothesis_template="{context_prefix}participants were {value}.",
        ),
    )
    population_role_other: Optional[StrField] = Field(
        default=None,
        description="Free-text group role label when population_role is Other.",
        json_schema_extra=extraction_meta(
            "text",
            "If population_role is Other, capture the stated role label here. Otherwise null.",
            scope_hint="Methods/Participants, Abstract",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}participants were {value}.",
        ),
    )

    cohort_label: Optional[StrField] = Field(
        default=None,
        description="Cohort/group label used in the article.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract the group label as stated (can include role/descriptors). "
            "Do not restrict to diagnosis terms. Return null if not stated.",
            scope_hint="Methods/Participants, Abstract, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}the cohort was labeled {value}.",
        ),
    )

    medical_condition: Optional[StrField] = Field(
        default=None,
        description="Diagnosis/condition terminology for this group if explicitly stated.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract diagnosis/condition terminology as stated. Do not infer from label. "
            "Do not use cohort labels or roles. "
            "Return null if not stated.",
            scope_hint="Methods/Participants, Abstract",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}participants had {value}.",
        ),
    )

    count: Optional[IntField] = Field(
        default=None,
        description="Number of participants in this group (included).",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract participant count as a number. Exclude dropouts unless explicitly included. "
            "Do not use pooled totals. Return null if not stated.",
            scope_hint="Methods/Participants, Abstract, Table 1, Results",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}there were {value} participants.",
        ),
    )

    male_count: Optional[IntField] = Field(
        default=None,
        description="Number of male participants in this group.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract male count as a number. Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}there were {value} male participants.",
        ),
    )

    female_count: Optional[IntField] = Field(
        default=None,
        description="Number of female participants in this group.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract female count as a number. Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}there were {value} female participants.",
        ),
    )

    age_mean: Optional[FloatField] = Field(
        default=None,
        description="Mean age for this group.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract mean age as a number. Return null if not stated for this group.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}mean age was {value}.",
        ),
    )

    age_sd: Optional[FloatField] = Field(
        default=None,
        description="Age standard deviation for this group (if stated).",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract age SD as a number if stated. Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}age SD was {value}.",
        ),
    )

    age_range: Optional[StrField] = Field(
        default=None,
        description="Age range.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract reported sample age range as stated. Do not infer from min/max eligibility. "
            "Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}age range was {value}.",
        ),
    )

    age_minimum: Optional[IntField] = Field(
        default=None,
        description="Minimum reported age for this group.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract minimum age as a number only if explicitly stated (e.g., eligibility). "
            "Do not infer from age range. Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}minimum age was {value}.",
        ),
    )
    age_maximum: Optional[IntField] = Field(
        default=None,
        description="Maximum reported age for this group.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract maximum age as a number only if explicitly stated (e.g., eligibility). "
            "Do not infer from age range. Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}maximum age was {value}.",
        ),
    )

    age_median: Optional[FloatField] = Field(
        default=None,
        description="Median age for this group.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract median age as a number. Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}median age was {value}.",
        ),
    )

    all_right_handed: Optional[BoolField] = Field(
        default=None,
        description="Whether all participants in this group are right-handed (group-scoped).",
        json_schema_extra=extraction_meta(
            "boolean",
            "Set true only if explicitly stated all participants in this group are right-handed. "
            "Set false only if explicitly stated not all are right-handed. Otherwise null.",
            scope_hint="Methods/Participants",
            extraction_phase="group",
            evidence_hypothesis_template="{context_prefix}all participants were right-handed: {value}.",
        ),
    )

    # Group-level inclusion/exclusion (optional extension)
    inclusion_criteria: Optional[List[StrField]] = Field(
        default=None,
        description="Group-specific inclusion criteria items.",
        json_schema_extra=extraction_meta(
            "text",
            "List criteria only when explicitly tied to this group. "
            "Do not copy study-wide criteria. Return null if not group-specific.",
            scope_hint="Methods/Participants",
            extraction_phase="group",
            evidence_value_only=True,
        ),
    )
    exclusion_criteria: Optional[List[StrField]] = Field(
        default=None,
        description="Group-specific exclusion criteria items.",
        json_schema_extra=extraction_meta(
            "text",
            "List criteria only when explicitly tied to this group. "
            "Do not copy study-wide criteria. Return null if not group-specific.",
            scope_hint="Methods/Participants",
            extraction_phase="group",
            evidence_value_only=True,
        ),
    )


class DemographicsSchema(BaseModel):
    groups: Optional[List[GroupBase]] = Field(
        default=None,
        description="Participant groups described in the study (group-level demographics only).",
        json_schema_extra=extraction_meta(
            "object_list",
            "Extract all participant groups described in the study. "
            "Return null if none are stated.",
            scope_hint="Methods/Participants, Abstract, Table 1",
            extraction_phase="group",
        ),
    )


StudyDesignLiteral = Literal["CrossSectional", "Longitudinal", "CaseControl", "Intervention", "Other"]
AnalysisReportingLiteral = Literal["WholeBrain", "ROI", "Atlas/Parcellation", "Other"]
AnalysisMethodLiteral = Literal[
    "Activation/Contrastive",
    "SeedBasedConnectivity",
    "IndependentComponentsAnalysis",
    "BrainBehaviorCorrelation",
    "Network",
    "Morphometry",
    "Microstructure",
    "Other",
]


class AnalysisBase(EntityBase):
    """
    One analysis/contrast (A1, A2...) reported in the study.
    """

    analysis_label: Optional[StrField] = Field(
        default=None,
        description="Short label for the analysis/contrast.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract a short analysis/contrast label as stated. "
            "Use the named analysis or contrast label if provided. "
            "Return null if not stated.",
            scope_hint="Methods/Analysis, Results, Captions",
            extraction_phase="analysis",
            evidence_value_only=True,
        ),
    )

    reporting_scope: Optional[ExtractedValue[AnalysisReportingLiteral]] = Field(
        default=None,
        description="Analysis reporting scope.",
        json_schema_extra=extraction_meta(
            "enum",
            "Select WholeBrain/ROI/Atlas/Other only if explicitly stated. Return null if not stated.",
            scope_hint="Methods/Analysis, Results",
            extraction_phase="analysis",
            evidence_hypothesis_template="{context_prefix}reporting scope was {value}.",
            evidence_value_only=True,
        ),
    )
    reporting_scope_other: Optional[StrField] = Field(
        default=None,
        description="Free-text reporting scope label when reporting_scope is Other.",
        json_schema_extra=extraction_meta(
            "text",
            "If reporting_scope is Other, capture the stated scope label here. Otherwise null.",
            scope_hint="Methods/Analysis, Results",
            extraction_phase="analysis",
            evidence_hypothesis_template="{context_prefix}reporting scope was {value}.",
        ),
    )

    analysis_method: Optional[ExtractedValue[AnalysisMethodLiteral]] = Field(
        default=None,
        description="Analysis method category.",
        json_schema_extra=extraction_meta(
            "enum",
            "Select the analysis method category only when it can be explicitly mapped to the existing options."
            "Return null if not stated.",
            scope_hint="Methods/Analysis, Results",
            extraction_phase="analysis",
            inference_policy="post_normalize",
            evidence_hypothesis_template="{context_prefix}the analysis used {value}.",
        ),
    )
    analysis_method_other: Optional[StrField] = Field(
        default=None,
        description="Free-text analysis method label when analysis_method is Other.",
        json_schema_extra=extraction_meta(
            "text",
            "If analysis_method is Other, capture the stated method label here. Otherwise null.",
            scope_hint="Methods/Analysis, Results",
            extraction_phase="analysis",
            evidence_hypothesis_template="{context_prefix}the analysis used {value}.",
        ),
    )

    contrast_formula: Optional[StrField] = Field(
        default=None,
        description="Contrast formula.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract the contrast formula as stated if present. Return null if not stated.",
            scope_hint="Methods/Analysis, Results",
            extraction_phase="analysis",
            evidence_value_only=True,
        ),
    )

# -----------------------------
# Study-level metadata
# -----------------------------
class StudyMetadataModel(BaseModel):
    study_objective: Optional[StrField] = Field(
        default=None,
        description="Primary research objective (1-2 sentences).",
        json_schema_extra=extraction_meta(
            "summary",
            "Summarize the primary objective in 1-2 sentences using only stated information. "
            "Return null if not stated.",
            scope_hint="Abstract, Introduction",
            extraction_phase="study",
            inference_policy="synthesize",
            evidence_value_only=True,
        ),
    )

    inclusion_criteria: Optional[List[StrField]] = Field(
        default=None,
        description="Study-wide inclusion criteria items.",
        json_schema_extra=extraction_meta(
            "text",
            "List each inclusion criterion as a discrete item using the wording in the article. "
            "Do not include criteria stated only for specific groups. Return null if not stated.",
            scope_hint="Methods/Participants",
            extraction_phase="study",
            evidence_value_only=True,
        ),
    )
    exclusion_criteria: Optional[List[StrField]] = Field(
        default=None,
        description="Study-wide exclusion criteria items.",
        json_schema_extra=extraction_meta(
            "text",
            "List each exclusion criterion as a discrete item using the wording in the article. "
            "Do not include criteria stated only for specific groups. Return null if not stated.",
            scope_hint="Methods/Participants",
            extraction_phase="study",
            evidence_value_only=True,
        ),
    )

    study_type: Optional[
        ExtractedValue[Literal["OriginalResearch", "MetaAnalysis", "Review", "Other"]]
    ] = Field(
        default=None,
        description="Study type classification.",
        json_schema_extra=extraction_meta(
            "enum",
            "If explicitly labeled as meta-analysis or review, choose accordingly. "
            "Otherwise choose OriginalResearch only when primary data collection/analysis "
            "of participant data is described. Return null if unclear.",
            scope_hint="Title, Abstract",
            extraction_phase="study",
            evidence_hypothesis_template="The study type was {value}.",
        ),
    )
    study_type_other: Optional[StrField] = Field(
        default=None,
        description="Free-text study type label when study_type is Other.",
        json_schema_extra=extraction_meta(
            "text",
            "If study_type is Other, capture the stated study type label here. Otherwise null.",
            scope_hint="Title, Abstract",
            extraction_phase="study",
            evidence_hypothesis_template="The study type was {value}.",
        ),
    )


# -----------------------------
# Links (explicit edges)
# -----------------------------
class GroupTaskEdge(BaseModel):
    group_id: str
    task_id: str
    evidence: Optional[List[EvidenceSpan]] = Field(
        default=None, description="Evidence spans supporting this link."
    )
    note: Optional[str] = None


class TaskModalityEdge(BaseModel):
    task_id: str
    modality_id: str
    evidence: Optional[List[EvidenceSpan]] = Field(
        default=None, description="Evidence spans supporting this link."
    )
    note: Optional[str] = None


class AnalysisTaskEdge(BaseModel):
    analysis_id: str
    task_id: str
    evidence: Optional[List[EvidenceSpan]] = Field(
        default=None, description="Evidence spans supporting this link."
    )
    note: Optional[str] = None


class AnalysisGroupEdge(BaseModel):
    analysis_id: str
    group_id: str
    evidence: Optional[List[EvidenceSpan]] = Field(
        default=None, description="Evidence spans supporting this link."
    )
    note: Optional[str] = None


class AnalysisConditionEdge(BaseModel):
    analysis_id: str
    condition_id: str
    evidence: Optional[List[EvidenceSpan]] = Field(
        default=None, description="Evidence spans supporting this link."
    )
    note: Optional[str] = None


class GroupModalityEdge(BaseModel):
    group_id: str
    modality_id: str
    n_scanned: Optional[IntField] = Field(
        default=None,
        description="If a subset of this group received this modality, record N scanned if stated.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract N scanned for this group+modality edge only if explicitly stated. "
            "Return null if not stated.",
            scope_hint="Methods/Imaging, Results",
            extraction_phase="linking",
        ),
    )
    evidence: Optional[List[EvidenceSpan]] = Field(
        default=None, description="Evidence spans supporting this link."
    )
    note: Optional[str] = None


class StudyLinks(BaseModel):
    group_task: Optional[List[GroupTaskEdge]] = Field(
        default=None,
        description="Edges linking groups to tasks they performed.",
        json_schema_extra=extraction_meta(
            "edge_list",
            "Create edges only when explicitly stated (e.g., 'all participants completed task X'). "
            "Return null if not stated. Add per-edge evidence when possible.",
            scope_hint="Methods/Task, Methods/Participants, Results",
            extraction_phase="linking",
        ),
    )
    task_modality: Optional[List[TaskModalityEdge]] = Field(
        default=None,
        description="Edges linking tasks to modalities used.",
        json_schema_extra=extraction_meta(
            "edge_list",
            "Create edges only when explicitly stated. Return null if not stated. "
            "Add per-edge evidence when possible.",
            scope_hint="Methods/Imaging, Methods/Task, Results",
            extraction_phase="linking",
        ),
    )
    analysis_task: Optional[List[AnalysisTaskEdge]] = Field(
        default=None,
        description="Edges linking analyses to the tasks they test.",
        json_schema_extra=extraction_meta(
            "edge_list",
            "Create edges only when explicitly stated (e.g., contrast tied to a named task). "
            "Return null if not stated. Add per-edge evidence when possible.",
            scope_hint="Methods/Analysis, Methods/Task, Results",
            extraction_phase="linking",
        ),
    )
    analysis_group: Optional[List[AnalysisGroupEdge]] = Field(
        default=None,
        description="Edges linking analyses to groups included in the analysis.",
        json_schema_extra=extraction_meta(
            "edge_list",
            "Create edges only when explicitly stated (e.g., analysis limited to a specific group). "
            "Return null if not stated. Add per-edge evidence when possible.",
            scope_hint="Methods/Analysis, Methods/Participants, Results",
            extraction_phase="linking",
        ),
    )
    analysis_condition: Optional[List[AnalysisConditionEdge]] = Field(
        default=None,
        description="Edges linking analyses to specific task conditions.",
        json_schema_extra=extraction_meta(
            "edge_list",
            "Create edges only when explicitly stated (e.g., contrast references named conditions). "
            "Return null if not stated. Add per-edge evidence when possible.",
            scope_hint="Methods/Analysis, Methods/Task, Results",
            extraction_phase="linking",
        ),
    )
    group_modality: Optional[List[GroupModalityEdge]] = Field(
        default=None,
        description="Edges linking groups to modalities they underwent (supports subsets).",
        json_schema_extra=extraction_meta(
            "edge_list",
            "Create edges only when explicitly stated. If only pooled imaging is stated, "
            "consider leaving per-group edges null and note pooled in study-level notes. "
            "Add per-edge evidence when possible.",
            scope_hint="Methods/Imaging, Methods/Participants, Results",
            extraction_phase="linking",
        ),
    )


# -----------------------------
# Top-level record
# -----------------------------
class StudyRecord(BaseModel):
    """
    A paper-level record designed for multi-pass extraction:

    Pass 1: entity discovery -> groups/tasks/modalities (labels)
    Pass 2: per-entity attributes -> demographics, task details, modality params, analyses
    Pass 3: linking -> connect entities via edges
    Pass 4: verification/consistency -> populate notes/confidence (optional)
    """

    study: Optional[StudyMetadataModel] = Field(
        default=None,
        description="Study-wide metadata.",
        json_schema_extra=extraction_meta(
            "object",
            "Extract study-wide metadata. Return null if not stated.",
            scope_hint="Title, Abstract, Methods",
            extraction_phase="record",
        ),
    )

    demographics: Optional[DemographicsSchema] = Field(
        default=None,
        description="Groups and demographics (group-level only).",
        json_schema_extra=extraction_meta(
            "object",
            "Extract participant groups and demographics.",
            scope_hint="Methods/Participants, Abstract, Table 1",
            extraction_phase="record",
        ),
    )

    tasks: Optional[List[TaskBase]] = Field(
        default=None,
        description="Tasks/paradigms used in the study (repeatable entities).",
        json_schema_extra=extraction_meta(
            "object_list",
            "Extract all distinct tasks used in this study. "
            "Exclude background/cited tasks not used here.",
            scope_hint="Abstract, Methods/Task, Results, Captions",
            extraction_phase="record",
        ),
    )

    modalities: Optional[List[ModalityBase]] = Field(
        default=None,
        description="Neuroimaging modalities/acquisitions used in the study (repeatable entities).",
        json_schema_extra=extraction_meta(
            "object_list",
            "Extract all distinct imaging modalities used in this study. "
            "Exclude background mentions.",
            scope_hint="Abstract, Methods/Imaging, Results, Captions",
            extraction_phase="record",
        ),
    )

    analyses: Optional[List[AnalysisBase]] = Field(
        default=None,
        description="Analyses/contrasts reported (repeatable entities).",
        json_schema_extra=extraction_meta(
            "object_list",
            "Extract reported analyses/contrasts. Prefer labels/formulas as stated.",
            scope_hint="Methods/Analysis, Results, Captions",
            extraction_phase="record",
        ),
    )

    links: Optional[StudyLinks] = Field(
        default=None,
        description="Explicit edges linking groups, tasks, and modalities.",
        json_schema_extra=extraction_meta(
            "object",
            "Populate links only when explicitly stated. Return null if links cannot be determined from text.",
            scope_hint="Methods/Participants, Methods/Task, Methods/Imaging, Results",
            extraction_phase="record",
            inference_policy="synthesize",
        ),
    )

    # Optional global notes / quality flags
    extraction_notes: Optional[List[StrField]] = Field(
        default=None,
        description="Pipeline notes.",
        json_schema_extra=extraction_meta(
            "text",
            "Add short notes about ambiguities or pooled reporting. Return null if none.",
            extraction_phase="record",
        ),
    )
