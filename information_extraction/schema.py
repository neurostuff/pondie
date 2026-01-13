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
    "demographics_shared",
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
        description="Unit for numeric values if stated (e.g., 'ms', 'years', 'Tesla').",
    )
    missing_reason: Optional[str] = Field(
        default=None,
        description="Reason value is missing (e.g., 'not_reported', 'not_applicable', 'ambiguous').",
    )
    note: Optional[str] = Field(
        default=None,
        description="Free-text note (e.g., 'reported pooled across groups').",
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
        ),
    )

    sequence_name: Optional[StrField] = Field(
        default=None,
        description="Sequence name/type (e.g., 'EPI', 'MPRAGE', 'spin-echo').",
        json_schema_extra=extraction_meta(
            "text",
            "Extract sequence/acquisition terminology as stated. Return null if not stated.",
            scope_hint="Methods/Imaging",
            extraction_phase="modality",
        ),
    )

    # Voxel size
    voxel_size: Optional[StrField] = Field(
        default=None,
        description="Voxel size (e.g., '2×2×2 mm').",
        json_schema_extra=extraction_meta(
            "text",
            "Extract voxel size as stated. Return null if not stated.",
            scope_hint="Methods/Imaging",
            extraction_phase="modality",
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
]

TaskDesignLiteral = Literal["Blocked", "EventRelated", "Mixed", "Other"]


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
        ),
    )


class TaskBase(EntityBase):
    resting_state: Optional[BoolField] = Field(
        default=None,
        description="Whether task is resting state.",
        json_schema_extra=extraction_meta(
            "boolean",
            "Set true only if explicitly described as resting-state/baseline with no task. "
            "Set false only if explicitly described as task-based. Otherwise null.",
            scope_hint="Methods/Task, Abstract, Results, Captions",
            extraction_phase="task",
        ),
    )

    task_name: Optional[StrField] = Field(
        default=None,
        description="Task name used in the study (e.g., 'Stroop task').",
        json_schema_extra=extraction_meta(
            "text",
            "Use the exact task name if stated. If no name is stated but a task is "
            "described, create a short label grounded in the text. "
            "Do not include narrative descriptions or timing details. "
            "Return null if no task is described.",
            scope_hint="Methods/Task, Abstract, Results, Captions",
            extraction_phase="task",
            inference_policy="synthesize",
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
        ),
    )

    task_design: Optional[List[ExtractedValue[TaskDesignLiteral]]] = Field(
        default=None,
        description="Task design tags (only if explicitly stated).",
        json_schema_extra=extraction_meta(
            "enum",
            "Select design tags only when explicitly described. Return null if not stated.",
            scope_hint="Methods/Task",
            extraction_phase="task",
            inference_policy="post_normalize",
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
        ),
    )

    conditions: Optional[List[Condition]] = Field(
        default=None,
        description="Conditions for this task.",
        json_schema_extra=extraction_meta(
            "object_list",
            "Extract each task condition as an object with its label. "
            "Return null if conditions are not stated.",
            scope_hint="Methods/Task, Results",
            extraction_phase="task",
        ),
    )

    concepts: Optional[List[StrField]] = Field(
        default=None,
        description="Cognitive concepts explicitly associated with the task.",
        json_schema_extra=extraction_meta(
            "text",
            "List cognitive concepts exactly as stated. Do not map to domain tags. "
            "Return null if not stated.",
            scope_hint="Methods/Task, Abstract, Introduction (if explicitly describing this study's task)",
            extraction_phase="task",
        ),
    )

    # Separate domain tags from concept text
    domain_tags: Optional[List[ExtractedValue[TaskDomainLiteral]]] = Field(
        default=None,
        description="Cognitive domain tags.",
        json_schema_extra=extraction_meta(
            "enum",
            "Select domain tags only when explicitly stated in the article. "
            "Do not copy raw concept phrases. Return null if not stated.",
            scope_hint="Methods/Task, Abstract",
            extraction_phase="task",
            inference_policy="post_normalize",
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
        ),
    )

    age_range: Optional[StrField] = Field(
        default=None,
        description="Age range (e.g., '18-35').",
        json_schema_extra=extraction_meta(
            "text",
            "Extract reported sample age range as stated. Do not infer from min/max eligibility. "
            "Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="group",
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
        ),
    )


class SharedDemographics(BaseModel):
    """
    Demographics reported pooled across participants (shared across groups).
    Store only when explicitly reported pooled, to avoid copying into per-group fields.
    """

    pooled_count: Optional[IntField] = Field(
        default=None,
        description="Total participant N pooled across groups, if stated.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract pooled total N if explicitly stated for the full sample. "
            "Do not sum group counts. Return null if not stated.",
            scope_hint="Methods/Participants, Abstract, Table 1",
            extraction_phase="demographics_shared",
        ),
    )

    pooled_age_mean: Optional[FloatField] = Field(
        default=None,
        description="Pooled mean age across groups, if explicitly stated.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract pooled mean age if explicitly stated for the full sample. "
            "Do not copy per-group means. Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="demographics_shared",
        ),
    )

    pooled_age_sd: Optional[FloatField] = Field(
        default=None,
        description="Pooled age SD across groups, if explicitly stated.",
        json_schema_extra=extraction_meta(
            "numeric",
            "Extract pooled age SD if explicitly stated for the full sample. "
            "Do not copy per-group SDs. Return null if not stated.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="demographics_shared",
        ),
    )

    pooled_all_right_handed: Optional[BoolField] = Field(
        default=None,
        description="Whether all participants (pooled) are right-handed, if explicitly stated.",
        json_schema_extra=extraction_meta(
            "boolean",
            "Set true only if explicitly stated all participants are right-handed (pooled). "
            "Set false only if explicitly stated not all participants are right-handed. "
            "Do not infer from group-level values. Otherwise null.",
            scope_hint="Methods/Participants",
            extraction_phase="demographics_shared",
        ),
    )


class DemographicsSchema(BaseModel):
    shared: Optional[SharedDemographics] = Field(
        default=None,
        description="Pooled/shared demographics reported across groups.",
        json_schema_extra=extraction_meta(
            "object",
            "Populate only when demographics are explicitly reported pooled/shared across groups.",
            scope_hint="Methods/Participants, Table 1",
            extraction_phase="demographics_shared",
        ),
    )
    groups: Optional[List[GroupBase]] = Field(
        default=None,
        description="Participant groups described in the study.",
        json_schema_extra=extraction_meta(
            "object_list",
            "Extract all participant groups described in the study. "
            "Return null if none are stated.",
            scope_hint="Methods/Participants, Abstract, Table 1",
            extraction_phase="group",
        ),
    )


StudyDesignLiteral = Literal["CrossSectional", "Longitudinal", "CaseControl", "Intervention", "Other"]
AnalysisReportingLiteral = Literal["WholeBrain", "ROI", "Atlas", "Other"]


class AnalysisBase(EntityBase):
    """
    One analysis/contrast (A1, A2...) reported in the study.
    """

    reporting_scope: Optional[ExtractedValue[AnalysisReportingLiteral]] = Field(
        default=None,
        description="Analysis reporting scope.",
        json_schema_extra=extraction_meta(
            "enum",
            "Select WholeBrain/ROI/Atlas/Other only if explicitly stated. Return null if not stated.",
            scope_hint="Methods/Analysis, Results",
            extraction_phase="analysis",
        ),
    )

    study_design_tags: Optional[List[ExtractedValue[StudyDesignLiteral]]] = Field(
        default=None,
        description="Study design tags associated with this analysis (explicit only).",
        json_schema_extra=extraction_meta(
            "enum",
            "Select design tags explicitly linked to this analysis. Do not infer. Return null if not stated.",
            scope_hint="Methods, Abstract",
            extraction_phase="analysis",
            inference_policy="post_normalize",
        ),
    )

    statistical_model: Optional[StrField] = Field(
        default=None,
        description="Statistical model terminology.",
        json_schema_extra=extraction_meta(
            "text",
            "Extract statistical model terminology as stated. Return null if not stated.",
            scope_hint="Methods/Analysis",
            extraction_phase="analysis",
        ),
    )

    contrast_formula: Optional[StrField] = Field(
        default=None,
        description="Contrast formula (e.g., 'Incongruent > Congruent').",
        json_schema_extra=extraction_meta(
            "text",
            "Extract the contrast formula as stated if present. Return null if not stated.",
            scope_hint="Methods/Analysis, Results",
            extraction_phase="analysis",
        ),
    )

    outcome_measures: Optional[List[StrField]] = Field(
        default=None,
        description="Outcome measures reported for the analysis.",
        json_schema_extra=extraction_meta(
            "text",
            "List outcome measures exactly as stated. Return null if none are stated.",
            scope_hint="Results",
            extraction_phase="analysis",
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
        ),
    )

    study_type: Optional[ExtractedValue[Literal["OriginalResearch", "MetaAnalysis", "Review"]]] = Field(
        default=None,
        description="Study type classification.",
        json_schema_extra=extraction_meta(
            "enum",
            "If explicitly labeled as meta-analysis or review, choose accordingly. "
            "Otherwise choose OriginalResearch only when primary data collection/analysis "
            "of participant data is described. Return null if unclear.",
            scope_hint="Title, Abstract",
            extraction_phase="study",
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
        description="Edges linking tasks to modalities used (e.g., task fMRI).",
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
        description="Groups and demographics (group + pooled).",
        json_schema_extra=extraction_meta(
            "object",
            "Extract participant groups and demographics. "
            "Store pooled demographics in demographics.shared when explicitly pooled.",
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
        description="Pipeline notes (e.g., 'demographics only in Table 1', 'subset scanned').",
        json_schema_extra=extraction_meta(
            "text",
            "Add short notes about ambiguities or pooled reporting. Return null if none.",
            extraction_phase="record",
        ),
    )
