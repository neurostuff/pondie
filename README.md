# Neuroimaging Article Information Extraction workflow

The goal is to create a workflow for extracting structured information from neuroimaging articles
with grounding evidence and verification.
Each article will have lists of entities (groups, tasks, modalities, contrasts/analyses).
Each of these entities will have information about them
with evidence from the text.
As well, these entities may be linked to each other
(e.g., a group performing a task with a certain modality and contrast).

## Workflow Steps

1. **Article Ingestion**: Load neuroimaging articles in various formats (PDF, HTML, etc.).

2. **Text Extraction**: Extract raw text from the articles using OCR or text parsing libraries. (likely using docling)

3. **Preprocessing**: Offset-preserving normalization and optional abbreviation expansion (spaCy) with an offset map.

4. **Context Retrieval (optional)**: Build a per-document index of sentences, captions, and table rows. Store section headings as separate embeddings and use a weighted similarity score to retrieve top-k passages, then rerank to 5-10 before extraction.

5. **Extraction + Evidence Alignment**: Use LangExtract to identify and extract entities and their fields with aligned evidence spans.

6. **Incremental Updates (optional)**: For schema changes or partial gaps, load existing entity JSONL + evidence spans, compute a field-level "needs update" mask, and prompt only for missing/updated fields. If the overall prompt for an entity changes, re-extract the entire entity. Merge new outputs while preserving provenance in a run-level manifest keyed by entity id and field path.

7. **Entity Linking**: Link related entities together based on their relationships described in the text.

8. **Verification**: Implement a verification step to ensure the accuracy of extracted entities and their relationships, using evidence spans and LangExtract HTML review.

9. **Normalization**: Standardize entity names for consistency across articles (ONVOC).

10. **Output Generation**: Write per-document JSONL and aggregate into a run-level JSONL for analysis or storage.

## Schema Evolution + Incremental Updates

When the schema evolves or only a few fields are missing, avoid a full re-run by treating extraction as a
field-level migration. Existing entity JSONL (including evidence spans and section context) becomes input to
the next run. The pipeline can compute a per-field update mask (missing, stale, or explicitly requested) and
pass existing values + evidence into prompts so the model only fills in what is needed. If the overall prompt
for an entity changes, the entity is re-extracted in full. A flag controls whether existing fields are locked
(never overwritten) or can be updated. Outputs are merged while recording explicit provenance metadata
(schema version, prompt version, extraction run id, source=existing|updated) in a run-level sidecar manifest
keyed by entity id and field path. Evidence spans are treated as trusted unless a field is re-extracted.

This makes it possible to:
- Reuse prior evidence spans to narrow context and reduce cost.
- Refresh only specific fields when prompts or schema definitions change.
- Keep stable entity IDs while evolving attributes over time.

## Workflow Diagram

![Workflow diagram](docs/workflow.png)

```mermaid
flowchart TD
  A["Ingest articles<br/>(XML/HTML/PDF)"] --> B["Text extraction + OCR fallback<br/>(stable offsets, sections)"]
  B --> C["Preprocess<br/>(offset-preserving normalization,<br/>abbrev expansion + offset map)"]
  C --> Q["Per-doc retrieval + rerank<br/>(sentence + section embeddings)"]
  Q --> D["Extract entities + evidence spans (full)<br/>(LangExtract, per-entity prompts)"]
  D --> E["Evidence-required validation<br/>(ExtractedValue)"]
  E --> F["Link entities<br/>(StudyLinks edges)"]
  F --> G["Normalize concepts/conditions<br/>(ONVOC)"]
  G --> H["Export per-doc JSONL"]
  H --> I["Aggregate run JSONL + run manifest<br/>(provenance sidecar)"]

  D --> R["Human review<br/>(LangExtract HTML)"]
  F --> R

  subgraph Inputs
    S["schema.py<br/>(entities + fields)"]
    P["prompt files<br/>(per-entity JSON)"]
    Cfg["run config<br/>(models, params)"]
    V["ONVOC<br/>ontology"]
    X["Existing outputs<br/>(entity JSONL + evidence)"]
  end

  S --> D
  P --> D
  Cfg --> D
  V --> G

  subgraph Incremental_Updates["Incremental updates (optional)"]
    Y["Compute field mask + lock policy"]
    Z{"Entity prompt changed?"}
    W["Delta extraction<br/>(missing/updated fields)"]
    M["Merge outputs"]
  end

  C --> Y
  X --> Y
  P --> Z
  Y --> Z
  Z -- "yes" --> D
  Z -- "no" --> W
  Q --> W
  W --> M
  M --> E

  subgraph Caching
    Cache["Extraction cache<br/>(cache.sqlite per hash)"]
  end
  D <--> Cache
  W <--> Cache

  subgraph Optional
    O["DSPy prompt optimization<br/>(on gold set)"]
  end
  O --> P
```


## Critical Considerations

What is our schema going to look like?



## Random thoughts on implementation

For the schema, we will be tracking several entities:
- StudyRecord (paper-level container)
- StudyMetadataModel (study-wide metadata)
- DemographicsSchema (groups + pooled demographics)
- GroupBase (participant groups)
- TaskBase (tasks/paradigms)
- Condition (task conditions)
- ModalityBase (imaging acquisitions)
- AnalysisBase (analyses/contrasts)
- StudyLinks (edge container)
- EvidenceSpan (provenance for extracted values)


### Groups
Groups may be one implicit convienience sample; it could be just one person forming a group.

A paper may even include multiple datasets, each with
their own groups.

Or they may combine data from multiple sites.

There is also the possibility of groups being stratified
upon a continuous variable, like age, and there are no
strict categories.

### Tasks
Tasks may or may not be explicitly named in the text.
Tasks can be meta, like a switch task contains multiple
tasks within it. So a task could be hierarchical and
contain other tasks.

### Modalities
Modalities are likely to be a closed set of options,
like fMRI, DTI, EEG, MEG, PET, etc. that I am specifically interested in.


### Contrasts/Analyses
Contrasts and analyses may be more free-form text,
specifically relating to tasks/modalities/groups.
I would like to know outcome measures for the contrasts/analyses as well.

### Evidence
The document text that supports an extraction may have additional noisy formatting. LangExtract returns extractions with aligned evidence spans; summaries can be synthesized but must reference supporting spans.


### Individual Extraction Steps
The extracted information will need to be extracted in small groups of related entities,
and in an hierarchical manner at times. For example, I will extract all the group names,
then within each group, extract their sample sizes, ages, medical conditions, etc.
Linkages may be another step separately after all entities are filled out.
Prompt files will be stored per entity type (JSON) and can be optimized with DSPy.


### Passing in section context
When I'm looking for relevent text for an entity, I want the model to have
context on what section of the paper it is in, sometimes section titles explicitly label
the group being studied.

### Summary fields
Summaries will be stored in attributes while keeping the extraction_text verbatim, with evidence spans supporting the summary.

### Document IDs
Document IDs are derived from directory names (pmid-doi-pmcid), with missing IDs omitted.

### How to find relevant text for information extraction
If I'm grouping the the extractions for specific entities, I do not want to pass the entire document for each extraction, so I want to find relevant text spans to pass into the model, and re-relate the evidentiary text spans back to the full document.
If this step is messy (adding in extra text), then the extraction step will need to be
a better model to decide which sentences are actually relevant to answering the question.
If the sentences can be narrowly identified, then a smaller model can be used for the extraction step, which would be more cost effective. A large concern is that sentences may appear to be relevant, but refer to another entity. If I have two groups, controls, and patients, the demographic text for the patients may be selected when I'm looking for the controls demographics.
For retrieval, store sentence units and section headings separately. Compute sentence embeddings for each unit
and separate embeddings for section headings or paths. At query time, combine them with a weighted score
like: score = alpha * cos(query, sentence) + (1 - alpha) * cos(query, section). This keeps sentence meaning
primary while adding section semantics. Captions and table rows should be stored as their own units with the
relevant section and caption context so they can be retrieved and reranked alongside sentences.

### Robust to changes in schema and document formats
We will be making changes to the schemas over time.
This will be reflected in the final schema, but in a more nuanced way,
the groupings and relationship/linking between entities may change, 
and the order of extraction may change, so I want to make sure that the workflow
is abstracted enough to handle these changes in a modular way.
When schema or prompt changes affect only a subset of fields, the pipeline should
support partial, field-level re-extraction using existing values and evidence spans.


### Be modular to multiple post-processing steps
After the initial extraction, there may be multiple post-processing steps
to normalize units, formatting, naming conventions, mapping to ontologies, etc.

### How to represent schemas
from the perspective of extraction, do the groupings make sense? Some of the fields are inference (not literally pulling from the text, like summarization or choosing from an ENUM), some are literally pulling from text, and some are doing a simple inference (whether something exists or not as a boolean).

There are three perspectives I can see:
1) the view from the data consumer to see the clean data output.

2) the view of grouping logically similar fields together

3) the view of submitting/extracting fields with the same model that are within the same "topic", like the group, some fields may be better extracted with different models.


### Preprocessing of the text

I would like to expand abbreviations before extraction.
This will make it easier for the model to find relevant text.
Abbreviation expansion will use sci-spaCy and must preserve offsets.


### Post-processing graph representation

The final output may be best represented as a knowledge graph
about the paper, since groups, tasks, modalities can all be connected.
Per-document outputs will be named using pmid-doi-pmcid derived from input directory names.


### Reference/Citations graphs

Likely out of scope for this project, but it would be a ripe field
to explore how tests/experiments relate to other papers via references/citations.

### Reducing context for information extraction

How am I going to reduce the context (find the relevant text) for information extraction
for each entity/schema field?
- Using an LLM to select context: expensive.
- Using purely semantic search: does not reason about something being mentioned, versus actually using it in the study.
- using query based search, where the query is a natural language description of the entity/information being sought.
  - "what are the demographics of the participants in the study?"
    - then entailment for measuring relevance.

Conversion of pdf to text:
- Docling
- marker: https://github.com/datalab-to/marker?tab=readme-ov-file


Ensure JSON schema output:
- outlines: https://dottxt-ai.github.io/outlines/latest/


some promising papers: https://chatgpt.com/share/69626a3c-5f3c-800f-8806-c17ccb126575
