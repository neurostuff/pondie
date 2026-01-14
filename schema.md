# Schema Overview

This document summarizes the extraction schema defined in `information_extraction/schema.py` and provides an idealized example `StudyRecord` JSON.

## Core Concepts

- `ExtractedValue[T]` is the wrapper for extracted leaf fields. It includes:
  - `value`: the extracted value.
  - `evidence`: list of 1+ evidence spans when `value` is present.
  - `unit`: unit for numeric values (e.g., "ms", "s", "years", "Tesla").
  - `missing_reason`: optional reason for missing values (e.g., "not_reported").
  - `note`, `scope`, `confidence`: optional metadata.
- Container fields (e.g., `StudyRecord`, `StudyLinks`, entity lists) are plain objects/lists, not `ExtractedValue`.
- List fields that require per-item evidence use `List[ExtractedValue[T]]`.
- Evidence is optional in the schema; pipelines may enforce non-null evidence when values exist.
- Extraction metadata is stored in `json_schema_extra` via `extraction_meta`:
  - `extraction_type`, `extraction_prompt`, `scope_hint`,
    `extraction_phase`, `allow_note`, `inference_policy`.
- `EvidenceSpan` includes `source`, `section`, `extraction_text`, `char_interval`,
  `alignment_status`, `extraction_index`, `group_index`, `document_id`, and optional `locator`.
- Entities are listed separately and linked with explicit edges to form a graph.

## Main Entities

- `StudyRecord`: top-level container.
- `StudyMetadataModel`: study-wide objective, criteria, study type.
- `GroupBase`: participant groups with demographics and group-specific criteria.
- `TaskBase`: tasks, conditions, and cognitive descriptors.
- `Condition`: named conditions for task contrasts.
- `ModalityBase`: imaging modalities and acquisition parameters.
- `AnalysisBase`: reported analyses/contrasts.
- `StudyLinks`: explicit edges between entities.

## Graph Links

- `group_task`: which groups performed which tasks.
- `task_modality`: which modalities were used for tasks.
- `group_modality`: which groups underwent which modalities (supports subsets).
- `analysis_task`: which tasks an analysis tests.
- `analysis_group`: which groups are included in an analysis.
- `analysis_condition`: which conditions are referenced in an analysis.
- Edges may include optional per-edge evidence spans.

## Extraction Guidance Highlights

- `task_name` is a short label; no narrative details.
- `task_description` is narrative; avoid timing or total duration.
- `design_details` is timing/run/block details; avoid narrative.
- `task_duration` is only total duration.
- Tasks can be interventions (e.g., stimulation/drug); variants like left/right/both or on/off belong in `conditions`.
- `task_category` tags the task type (e.g., CognitiveTask, Intervention, Exposure, RestingState) only when explicitly stated.
- `concepts` are raw phrases; `domain_tags` are normalized tags.
- `cohort_label` can be descriptive; `medical_condition` is diagnosis only.
- `population_role` is role only (patient/control/etc.).
- `age_range` is reported sample range; `age_minimum`/`age_maximum` are explicit minimum/maximum if stated.
- Demographics are captured at the group level only.

## Example JSON (one JSONL line, abbreviated)

```json
{
  "study": {
    "study_objective": {
      "value": "Test whether conflict monitoring deficits in schizophrenia are associated with altered task-evoked fMRI responses.",
      "evidence": [
        {
          "section": "Abstract",
          "extraction_text": "We investigated conflict monitoring in schizophrenia using a Stroop task during fMRI.",
          "char_interval": {"start_pos": 120, "end_pos": 210},
          "alignment_status": "match_exact",
          "extraction_index": 1,
          "group_index": 0,
          "document_id": "doc_abc123"
        }
      ]
    },
    "study_type": {
      "value": "OriginalResearch",
      "evidence": [
        {
          "section": "Title",
          "extraction_text": "An fMRI study of Stroop performance in schizophrenia",
          "char_interval": {"start_pos": 0, "end_pos": 58},
          "alignment_status": "match_exact",
          "extraction_index": 2,
          "group_index": 0,
          "document_id": "doc_abc123"
        }
      ]
    },
    "inclusion_criteria": [
      {
        "value": "Diagnosis of schizophrenia",
        "evidence": [
          {
            "section": "Methods/Participants",
            "extraction_text": "Patients met DSM-IV criteria for schizophrenia.",
            "char_interval": {"start_pos": 980, "end_pos": 1035},
            "alignment_status": "match_exact",
            "extraction_index": 3,
            "group_index": 0,
            "document_id": "doc_abc123"
          }
        ]
      }
    ]
  },
  "demographics": {
    "groups": [
      {
        "id": "G1",
        "cohort_label": {
          "value": "Schizophrenia patients",
          "evidence": [
            {
              "section": "Methods/Participants",
              "extraction_text": "Patients with schizophrenia (n=22)",
              "char_interval": {"start_pos": 920, "end_pos": 955},
              "alignment_status": "match_exact",
              "extraction_index": 4,
              "group_index": 0,
              "document_id": "doc_abc123"
            }
          ]
        },
        "count": {
          "value": 22,
          "evidence": [
            {
              "section": "Methods/Participants",
              "extraction_text": "Patients with schizophrenia (n=22)",
              "char_interval": {"start_pos": 920, "end_pos": 955},
              "alignment_status": "match_exact",
              "extraction_index": 5,
              "group_index": 0,
              "document_id": "doc_abc123"
            }
          ]
        }
      }
    ]
  },
  "tasks": [
    {
      "id": "T1",
      "task_name": {
        "value": "Stroop task",
        "evidence": [
          {
            "section": "Methods/Task",
            "extraction_text": "Participants performed a color-word Stroop task.",
            "char_interval": {"start_pos": 1500, "end_pos": 1555},
            "alignment_status": "match_exact",
            "extraction_index": 6,
            "group_index": 0,
            "document_id": "doc_abc123"
          }
        ]
      },
      "task_design": [
        {
          "value": "EventRelated",
          "evidence": [
            {
              "section": "Methods/Task",
              "extraction_text": "Event-related design with 120 trials.",
              "char_interval": {"start_pos": 1622, "end_pos": 1665},
              "alignment_status": "match_exact",
              "extraction_index": 7,
              "group_index": 0,
              "document_id": "doc_abc123"
            }
          ]
        }
      ],
      "conditions": [
        {
          "id": "C1",
          "condition_label": {
            "value": "Incongruent",
            "evidence": [
              {
                "section": "Methods/Task",
                "extraction_text": "Incongruent trials were defined as...",
                "char_interval": {"start_pos": 1755, "end_pos": 1795},
                "alignment_status": "match_exact",
                "extraction_index": 8,
                "group_index": 0,
                "document_id": "doc_abc123"
              }
            ]
          }
        }
      ]
    }
  ],
  "modalities": [
    {
      "id": "M1",
      "modality_type": {
        "family": {
          "value": "fMRI",
          "evidence": [
            {
              "section": "Methods/Imaging",
              "extraction_text": "Functional MRI data were acquired...",
              "char_interval": {"start_pos": 2000, "end_pos": 2045},
              "alignment_status": "match_exact",
              "extraction_index": 9,
              "group_index": 0,
              "document_id": "doc_abc123"
            }
          ]
        }
      },
      "manufacturer": {
        "value": "Siemens",
        "evidence": [
          {
            "section": "Methods/Imaging",
            "extraction_text": "Images were collected on a Siemens 3T scanner.",
            "char_interval": {"start_pos": 2090, "end_pos": 2140},
            "alignment_status": "match_exact",
            "extraction_index": 10,
            "group_index": 0,
            "document_id": "doc_abc123"
          }
        ]
      }
    }
  ],
  "analyses": [
    {
      "id": "A1",
      "contrast_formula": {
        "value": "Incongruent > Congruent",
        "evidence": [
          {
            "section": "Methods/Analysis",
            "extraction_text": "The primary contrast was Incongruent > Congruent.",
            "char_interval": {"start_pos": 2725, "end_pos": 2775},
            "alignment_status": "match_exact",
            "extraction_index": 11,
            "group_index": 0,
            "document_id": "doc_abc123"
          }
        ]
      }
    }
  ],
  "links": {
    "group_task": [
      {
        "group_id": "G1",
        "task_id": "T1",
        "evidence": [
          {
            "section": "Methods/Task",
            "extraction_text": "All participants completed the Stroop task.",
            "char_interval": {"start_pos": 1480, "end_pos": 1515},
            "alignment_status": "match_exact",
            "extraction_index": 12,
            "group_index": 0,
            "document_id": "doc_abc123"
          }
        ]
      }
    ]
  },
  "extraction_notes": [
    {
      "value": "Demographics reported in Table 1.",
      "evidence": [
        {
          "section": "Table 1",
          "extraction_text": "Participant demographics are summarized in Table 1.",
          "char_interval": {"start_pos": 4100, "end_pos": 4150},
          "alignment_status": "match_exact",
          "extraction_index": 13,
          "group_index": 0,
          "document_id": "doc_abc123"
        }
      ]
    }
  ]
}
```
