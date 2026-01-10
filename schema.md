# Schema Overview

This document summarizes the extraction schema defined in `schema.py` and provides an idealized example `StudyRecord` JSON.

## Core Concepts

- `ExtractedValue[T]` is the wrapper for every extracted field. It includes:
  - `value`: the extracted value.
  - `evidence`: list of 1+ evidence spans when `value` is present.
  - `unit`: unit for numeric values (e.g., "ms", "s", "years", "Tesla").
  - `missing_reason`: optional reason for missing values (e.g., "not_reported").
  - `note`, `scope`, `confidence`: optional metadata.
- Evidence is required whenever a `value` exists; evidence lists cannot be empty.
- Extraction metadata is stored in `json_schema_extra` via `extraction_meta`:
  - `extraction_type`, `extraction_prompt`, `scope_hint`, `cardinality`,
    `extraction_phase`, `allow_note`, `inference_policy`.
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
- Edges may include optional per-edge evidence spans (list-level evidence still required).

## Extraction Guidance Highlights

- `task_name` is a short label; no narrative details.
- `task_description` is narrative; avoid timing or total duration.
- `design_details` is timing/run/block details; avoid narrative.
- `task_duration` is only total duration.
- `concepts` are raw phrases; `domain_tags` are normalized tags.
- `cohort_label` can be descriptive; `medical_condition` is diagnosis only.
- `population_role` is role only (patient/control/etc.).
- `age_range` is reported sample range; `age_minimum`/`age_maximum` are explicit minimum/maximum if stated.
- Pooled demographics go only in `demographics.shared` when explicitly pooled.

## Example JSON (one JSONL line)

```json
{
  "study": {
    "value": {
      "study_objective": {
        "value": "Test whether conflict monitoring deficits in schizophrenia are associated with altered task-evoked fMRI responses.",
        "evidence": [
          {
            "section": "Abstract",
            "quote": "We investigated conflict monitoring in schizophrenia using a Stroop task during fMRI.",
            "char_start": 120,
            "char_end": 210
          }
        ]
      },
      "study_type": {
        "value": "OriginalResearch",
        "evidence": [
          {
            "section": "Title",
            "quote": "An fMRI study of Stroop performance in schizophrenia",
            "char_start": 0,
            "char_end": 58
          }
        ]
      },
      "inclusion_criteria": {
        "value": [
          "Diagnosis of schizophrenia",
          "Age 18-45 years"
        ],
        "evidence": [
          {
            "section": "Methods/Participants",
            "quote": "Patients met DSM-IV criteria for schizophrenia and were 18-45 years old.",
            "char_start": 980,
            "char_end": 1070
          }
        ]
      },
      "exclusion_criteria": {
        "value": [
          "History of neurological disorder",
          "Substance abuse in past 6 months"
        ],
        "evidence": [
          {
            "section": "Methods/Participants",
            "quote": "Exclusion criteria included neurological illness and substance abuse within 6 months.",
            "char_start": 1072,
            "char_end": 1175
          }
        ]
      }
    },
    "evidence": [
      {
        "section": "Abstract",
        "quote": "We investigated conflict monitoring in schizophrenia using a Stroop task during fMRI.",
        "char_start": 120,
        "char_end": 210
      }
    ]
  },
  "demographics": {
    "value": {
      "shared": null,
      "groups": {
        "value": [
          {
            "id": "G1",
            "population_role": {
              "value": "patient",
              "evidence": [
                {
                  "section": "Methods/Participants",
                  "quote": "Patients with schizophrenia (n=22)...",
                  "char_start": 920,
                  "char_end": 955
                }
              ]
            },
            "cohort_label": {
              "value": "Schizophrenia patients",
              "evidence": [
                {
                  "section": "Methods/Participants",
                  "quote": "Patients with schizophrenia (n=22)...",
                  "char_start": 920,
                  "char_end": 955
                }
              ]
            },
            "medical_condition": {
              "value": "Schizophrenia",
              "evidence": [
                {
                  "section": "Methods/Participants",
                  "quote": "Patients met DSM-IV criteria for schizophrenia.",
                  "char_start": 980,
                  "char_end": 1035
                }
              ]
            },
            "count": {
              "value": 22,
              "evidence": [
                {
                  "section": "Table 1",
                  "quote": "Schizophrenia patients (n=22)",
                  "char_start": 4200,
                  "char_end": 4230
                }
              ]
            },
            "age_mean": {
              "value": 29.4,
              "unit": "years",
              "evidence": [
                {
                  "section": "Table 1",
                  "quote": "Mean age 29.4 (SD 6.1)",
                  "char_start": 4232,
                  "char_end": 4260
                }
              ]
            },
            "age_sd": {
              "value": 6.1,
              "unit": "years",
              "evidence": [
                {
                  "section": "Table 1",
                  "quote": "Mean age 29.4 (SD 6.1)",
                  "char_start": 4232,
                  "char_end": 4260
                }
              ]
            },
            "all_right_handed": {
              "value": true,
              "evidence": [
                {
                  "section": "Methods/Participants",
                  "quote": "All participants were right-handed.",
                  "char_start": 1178,
                  "char_end": 1215
                }
              ]
            }
          },
          {
            "id": "G2",
            "population_role": {
              "value": "control",
              "evidence": [
                {
                  "section": "Methods/Participants",
                  "quote": "Healthy controls (n=24)...",
                  "char_start": 1220,
                  "char_end": 1250
                }
              ]
            },
            "cohort_label": {
              "value": "Healthy controls",
              "evidence": [
                {
                  "section": "Methods/Participants",
                  "quote": "Healthy controls (n=24)...",
                  "char_start": 1220,
                  "char_end": 1250
                }
              ]
            },
            "count": {
              "value": 24,
              "evidence": [
                {
                  "section": "Table 1",
                  "quote": "Healthy controls (n=24)",
                  "char_start": 4265,
                  "char_end": 4290
                }
              ]
            },
            "age_mean": {
              "value": 28.7,
              "unit": "years",
              "evidence": [
                {
                  "section": "Table 1",
                  "quote": "Mean age 28.7 (SD 5.8)",
                  "char_start": 4292,
                  "char_end": 4320
                }
              ]
            },
            "age_sd": {
              "value": 5.8,
              "unit": "years",
              "evidence": [
                {
                  "section": "Table 1",
                  "quote": "Mean age 28.7 (SD 5.8)",
                  "char_start": 4292,
                  "char_end": 4320
                }
              ]
            },
            "all_right_handed": {
              "value": true,
              "evidence": [
                {
                  "section": "Methods/Participants",
                  "quote": "All participants were right-handed.",
                  "char_start": 1178,
                  "char_end": 1215
                }
              ]
            }
          }
        ],
        "evidence": [
          {
            "section": "Methods/Participants",
            "quote": "Patients with schizophrenia (n=22) and healthy controls (n=24) participated.",
            "char_start": 900,
            "char_end": 970
          }
        ]
      }
    },
    "evidence": [
      {
        "section": "Methods/Participants",
        "quote": "Patients with schizophrenia (n=22) and healthy controls (n=24) participated.",
        "char_start": 900,
        "char_end": 970
      }
    ]
  },
  "tasks": {
    "value": [
      {
        "id": "T1",
        "resting_state": {
          "value": false,
          "evidence": [
            {
              "section": "Methods/Task",
              "quote": "Participants performed a color-word Stroop task.",
              "char_start": 1500,
              "char_end": 1555
            }
          ]
        },
        "task_name": {
          "value": "Stroop task",
          "evidence": [
            {
              "section": "Methods/Task",
              "quote": "Participants performed a color-word Stroop task.",
              "char_start": 1500,
              "char_end": 1555
            }
          ]
        },
        "task_description": {
          "value": "Participants named the ink color of congruent and incongruent color words as quickly and accurately as possible.",
          "evidence": [
            {
              "section": "Methods/Task",
              "quote": "Subjects named the ink color of congruent and incongruent words.",
              "char_start": 1558,
              "char_end": 1620
            }
          ]
        },
        "design_details": {
          "value": "Event-related design with 120 trials, 2 s per trial, inter-trial interval jittered 2-6 s.",
          "evidence": [
            {
              "section": "Methods/Task",
              "quote": "Event-related design with 120 trials (2 s each), ITI 2-6 s.",
              "char_start": 1622,
              "char_end": 1705
            }
          ]
        },
        "task_design": {
          "value": [
            "EventRelated"
          ],
          "evidence": [
            {
              "section": "Methods/Task",
              "quote": "Event-related design with 120 trials...",
              "char_start": 1622,
              "char_end": 1670
            }
          ]
        },
        "task_duration": {
          "value": "12 min",
          "evidence": [
            {
              "section": "Methods/Task",
              "quote": "The task lasted approximately 12 minutes.",
              "char_start": 1708,
              "char_end": 1750
            }
          ]
        },
        "conditions": {
          "value": [
            {
              "id": "C1",
              "condition_label": {
                "value": "Incongruent",
                "evidence": [
                  {
                    "section": "Methods/Task",
                    "quote": "Incongruent trials were defined as...",
                    "char_start": 1755,
                    "char_end": 1795
                  }
                ]
              }
            },
            {
              "id": "C2",
              "condition_label": {
                "value": "Congruent",
                "evidence": [
                  {
                    "section": "Methods/Task",
                    "quote": "Congruent trials were defined as...",
                    "char_start": 1798,
                    "char_end": 1835
                  }
                ]
              }
            }
          ],
          "evidence": [
            {
              "section": "Methods/Task",
              "quote": "Incongruent and congruent trials were included.",
              "char_start": 1750,
              "char_end": 1820
            }
          ]
        },
        "concepts": {
          "value": [
            "conflict monitoring",
            "cognitive control"
          ],
          "evidence": [
            {
              "section": "Introduction",
              "quote": "The Stroop task probes conflict monitoring and cognitive control.",
              "char_start": 520,
              "char_end": 600
            }
          ]
        },
        "domain_tags": {
          "value": [
            "Executive cognitive control"
          ],
          "evidence": [
            {
              "section": "Introduction",
              "quote": "The Stroop task probes conflict monitoring and cognitive control.",
              "char_start": 520,
              "char_end": 600
            }
          ]
        }
      }
    ],
    "evidence": [
      {
        "section": "Methods/Task",
        "quote": "Participants performed a color-word Stroop task.",
        "char_start": 1500,
        "char_end": 1555
      }
    ]
  },
  "modalities": {
    "value": [
      {
        "id": "M1",
        "modality_type": {
          "value": {
            "family": {
              "value": "fMRI",
              "evidence": [
                {
                  "section": "Methods/Imaging",
                  "quote": "Functional MRI data were acquired...",
                  "char_start": 2000,
                  "char_end": 2045
                }
              ]
            },
            "subtype": {
              "value": "BOLD",
              "evidence": [
                {
                  "section": "Methods/Imaging",
                  "quote": "BOLD EPI sequence was used.",
                  "char_start": 2048,
                  "char_end": 2080
                }
              ]
            }
          },
          "evidence": [
            {
              "section": "Methods/Imaging",
              "quote": "Functional MRI data were acquired using a BOLD EPI sequence.",
              "char_start": 2000,
              "char_end": 2085
            }
          ]
        },
        "manufacturer": {
          "value": "Siemens",
          "evidence": [
            {
              "section": "Methods/Imaging",
              "quote": "Images were collected on a Siemens 3T scanner.",
              "char_start": 2090,
              "char_end": 2140
            }
          ]
        },
        "field_strength_tesla": {
          "value": 3.0,
          "unit": "T",
          "evidence": [
            {
              "section": "Methods/Imaging",
              "quote": "Images were collected on a Siemens 3T scanner.",
              "char_start": 2090,
              "char_end": 2140
            }
          ]
        },
        "sequence_name": {
          "value": "EPI",
          "evidence": [
            {
              "section": "Methods/Imaging",
              "quote": "BOLD EPI sequence was used.",
              "char_start": 2048,
              "char_end": 2080
            }
          ]
        },
        "voxel_size": {
          "value": "2x2x2 mm",
          "evidence": [
            {
              "section": "Methods/Imaging",
              "quote": "Voxel size was 2x2x2 mm.",
              "char_start": 2150,
              "char_end": 2180
            }
          ]
        },
        "tr_seconds": {
          "value": 2.0,
          "unit": "s",
          "evidence": [
            {
              "section": "Methods/Imaging",
              "quote": "TR = 2000 ms.",
              "char_start": 2185,
              "char_end": 2200
            }
          ]
        },
        "te_seconds": {
          "value": 0.03,
          "unit": "s",
          "evidence": [
            {
              "section": "Methods/Imaging",
              "quote": "TE = 30 ms.",
              "char_start": 2202,
              "char_end": 2215
            }
          ]
        }
      }
    ],
    "evidence": [
      {
        "section": "Methods/Imaging",
        "quote": "Functional MRI data were acquired using a BOLD EPI sequence.",
        "char_start": 2000,
        "char_end": 2085
      }
    ]
  },
  "analyses": {
    "value": [
      {
        "id": "A1",
        "reporting_scope": {
          "value": "WholeBrain",
          "evidence": [
            {
              "section": "Methods/Analysis",
              "quote": "Whole-brain voxelwise analyses were performed.",
              "char_start": 2600,
              "char_end": 2655
            }
          ]
        },
        "study_design_tags": {
          "value": [
            "CrossSectional"
          ],
          "evidence": [
            {
              "section": "Abstract",
              "quote": "We compared patients and controls at a single time point.",
              "char_start": 300,
              "char_end": 360
            }
          ]
        },
        "statistical_model": {
          "value": "GLM",
          "evidence": [
            {
              "section": "Methods/Analysis",
              "quote": "Data were analyzed using a general linear model (GLM).",
              "char_start": 2660,
              "char_end": 2720
            }
          ]
        },
        "contrast_formula": {
          "value": "Incongruent > Congruent",
          "evidence": [
            {
              "section": "Methods/Analysis",
              "quote": "The primary contrast was Incongruent > Congruent.",
              "char_start": 2725,
              "char_end": 2775
            }
          ]
        },
        "outcome_measures": {
          "value": [
            "BOLD activation",
            "beta weights"
          ],
          "evidence": [
            {
              "section": "Results",
              "quote": "We report BOLD activation and beta weights for the contrast.",
              "char_start": 3100,
              "char_end": 3170
            }
          ]
        }
      }
    ],
    "evidence": [
      {
        "section": "Methods/Analysis",
        "quote": "Whole-brain voxelwise analyses were performed.",
        "char_start": 2600,
        "char_end": 2655
      }
    ]
  },
  "links": {
    "value": {
      "group_task": {
        "value": [
          {
            "group_id": "G1",
            "task_id": "T1"
          },
          {
            "group_id": "G2",
            "task_id": "T1"
          }
        ],
        "evidence": [
          {
            "section": "Methods/Task",
            "quote": "All participants completed the Stroop task.",
            "char_start": 1480,
            "char_end": 1515
          }
        ]
      },
      "task_modality": {
        "value": [
          {
            "task_id": "T1",
            "modality_id": "M1"
          }
        ],
        "evidence": [
          {
            "section": "Methods/Imaging",
            "quote": "Functional MRI data were acquired during the Stroop task.",
            "char_start": 1980,
            "char_end": 2045
          }
        ]
      },
      "analysis_task": {
        "value": [
          {
            "analysis_id": "A1",
            "task_id": "T1"
          }
        ],
        "evidence": [
          {
            "section": "Methods/Analysis",
            "quote": "Analyses focused on the Stroop task contrast.",
            "char_start": 2725,
            "char_end": 2775
          }
        ]
      },
      "analysis_group": {
        "value": [
          {
            "analysis_id": "A1",
            "group_id": "G1"
          },
          {
            "analysis_id": "A1",
            "group_id": "G2"
          }
        ],
        "evidence": [
          {
            "section": "Methods/Analysis",
            "quote": "Patient and control groups were compared in the analysis.",
            "char_start": 2780,
            "char_end": 2840
          }
        ]
      },
      "analysis_condition": {
        "value": [
          {
            "analysis_id": "A1",
            "condition_id": "C1"
          },
          {
            "analysis_id": "A1",
            "condition_id": "C2"
          }
        ],
        "evidence": [
          {
            "section": "Methods/Analysis",
            "quote": "The primary contrast was Incongruent > Congruent.",
            "char_start": 2725,
            "char_end": 2775
          }
        ]
      },
      "group_modality": {
        "value": [
          {
            "group_id": "G1",
            "modality_id": "M1"
          },
          {
            "group_id": "G2",
            "modality_id": "M1"
          }
        ],
        "evidence": [
          {
            "section": "Methods/Imaging",
            "quote": "Both groups underwent fMRI scanning.",
            "char_start": 2050,
            "char_end": 2090
          }
        ]
      }
    },
    "evidence": [
      {
        "section": "Methods",
        "quote": "Participants completed the Stroop task during fMRI scanning.",
        "char_start": 1460,
        "char_end": 1525
      }
    ]
  },
  "extraction_notes": {
    "value": [
      "Demographics reported in Table 1."
    ],
    "evidence": [
      {
        "section": "Table 1",
        "quote": "Participant demographics are summarized in Table 1.",
        "char_start": 4100,
        "char_end": 4150
      }
    ]
  }
}
```
