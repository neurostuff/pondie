# Prompt File Format

Prompt files are JSON files stored per entity type (e.g., `group.json`, `task.json`).
They provide a group-level prompt and few-shot examples for LangExtract. Field-level prompts
remain in `information_extraction/schema.py`; prompt files focus on grouped field extraction.

## Required Fields

- `version`: prompt file version string (e.g., "v1").
- `entity`: schema entity name (e.g., `GroupBase`, `TaskBase`).
- `extraction_class`: LangExtract extraction class label (e.g., "group", "task").
- `prompt_description`: group-level instruction covering the listed fields.
- `fields`: list of schema field names covered by this prompt file.
- `examples`: list of example objects for few-shot guidance.

## Optional Fields

- `field_prompt_overrides`: map of field name -> prompt string to override `information_extraction/schema.py`.
  Use this when DSPy optimization produces field-specific prompt improvements.

## Example Structure

```json
{
  "version": "v1",
  "entity": "GroupBase",
  "extraction_class": "group",
  "prompt_description": "Extract group labels and demographics for each participant group.",
  "fields": [
    "population_role",
    "cohort_label",
    "medical_condition",
    "count",
    "age_mean",
    "age_sd"
  ],
  "field_prompt_overrides": {
    "cohort_label": "Extract the group label as stated; do not infer diagnoses."
  },
  "examples": [
    {
      "text": "Healthy controls (n=24) and patients with schizophrenia (n=22) participated.",
      "extractions": [
        {
          "extraction_text": "Healthy controls",
          "attributes": {
            "population_role": "control",
            "cohort_label": "Healthy controls",
            "count": "24"
          }
        },
        {
          "extraction_text": "patients with schizophrenia",
          "attributes": {
            "population_role": "patient",
            "cohort_label": "patients with schizophrenia",
            "medical_condition": "schizophrenia",
            "count": "22"
          }
        }
      ]
    }
  ]
}
```

## Notes

- `extraction_text` must be verbatim from the example `text` to enable evidence alignment.
- For summary fields, store synthesized text in `attributes` while keeping `extraction_text` verbatim.
- `fields` must match schema field names exactly.
- Prompt files should not duplicate field-level prompts from `information_extraction/schema.py`.
- DSPy can update `prompt_description` and `field_prompt_overrides` in new prompt file versions.
- Bump `version` whenever a prompt change should trigger a full re-extract for that entity.
