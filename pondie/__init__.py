"""pondie: extract a queryable record of what a neuroimaging paper reports, then query it.

Three packages, and the boundary between them is a contract rather than a convention.

    pondie.extraction     papers -> validated records, one stage at a time
    pondie.normalization  a record's own wording -> shared values, one module per field
    pondie.query          records -> the subset a meta-analysis should pool

Each pipeline declares its own inputs and outputs beside itself:
`pondie.extraction.models` for the extraction stages, `pondie.query.engine` for a selection,
`pondie.benchmark.run` for a score. The record's own shape is not among them: that is the
LinkML schema in `study_schema`, which generates the extraction schema, validates records, and
answers whether a slot is multivalued. Restating it here would be a second source of truth.
"""

__all__: list[str] = []
__version__ = "0.1.0"
