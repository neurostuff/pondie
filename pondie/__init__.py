"""pondie: extract a queryable record of what a neuroimaging paper reports, then query it.

Seven packages, listed in dependency order -- which is also reading order, because the
import graph is acyclic and each one only knows about the ones above it.

    pondie.paths          where the data lives. One definition, so a module that moves
                          keeps reading the same place
    pondie.formats        what a record and its source text are MADE OF: the value
                          wrapper, the text normalization offsets address, the table
                          render, the parse's address space. A second implementation of
                          any of it is a bug, and there have been nine
    pondie.schema         the LinkML schema, read through LinkML
    pondie.extraction     papers -> validated records, one stage at a time
    pondie.normalization  a record's own wording -> shared values, one module per field
    pondie.query          records -> the subset a meta-analysis should pool
    pondie.benchmark      how much of a paper an extraction got right

with `pondie.cli` over the top: one entry point, five verbs.

Each pipeline declares its own inputs and outputs beside itself:
`pondie.extraction.models` for the extraction stages, `pondie.query.engine` for a selection,
`pondie.benchmark.run` for a score. The record's own shape is not among them: that is the
LinkML schema authored in the `study_schema` submodule, and restating it here would be a
second source of truth.

The submodule holds YAML and prose and no code. Everything that reads it is here, so
"what a record is" has one implementation and it is in this package rather than split
across a repository boundary an install step had to hold together.
"""

__all__: list[str] = []
__version__ = "0.1.0"
