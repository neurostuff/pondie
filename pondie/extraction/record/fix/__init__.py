"""The deterministic fixes: what each one does, and the order they run in.

A fix takes a record and returns one line per change, mutating in place. Three kinds recur and
the difference is what each may assume about its input:

  shape    make the record match the schema. Runs first, because nothing schema-guided can
           read a record whose slots hold the wrong sort of thing
  derive   fill a slot from another slot, from the stage-1 parse, or from a join the record
           already contains. Reads no paper, which is what makes it deterministic
  link     make the references resolve, and refuse a guess. A fix decides nothing: where two
           answers are possible the record keeps its defect and `rules` tells a human

`sequence` holds the order as data, and names each fix by its kind -- `shape.repair_wrappers`,
`derive.derive_denominators` -- so the one place a reader wants to know which kind a fix is
says so. The flat re-export below is for everyone else, who should not have to know.

Named `fix` and not `repairs`, because `repair` is the model-driven stage that runs after
`build` and decides what the record cannot settle from its own contents. These decide
nothing: where two answers are possible the record keeps its defect and `rules` tells a
human. One word for two opposite things had every caller of the stage aliasing its import.
"""

from pondie.extraction.record.fix.derive import (
    derive_acquisition_types,
    derive_analysis_ids,
    derive_coordinate_spaces,
    derive_denominators,
    derive_table_effects,
    fill_directions,
    mirror_withheld,
    relabel_conclusions,
    resolve_source_table_analysis,
)
from pondie.extraction.record.fix.link import (
    align_cell_levels,
    check_local_ids,
    drop_redundant_cell_levels,
    link_entities_by_name,
    names_agree,
    repair_references,
    repoint_out_of_scope_terms,
    scope_duplicate_terms,
)
from pondie.extraction.record.fix.shape import (
    coerce_numeric_values,
    listify_nested,
    listify_scalars,
    rehome_stray_tables,
    repair_wrappers,
    unwrap_plain_slots,
    unwrap_singleton_lists,
)

from pondie.extraction.record.fix.sequence import (
    AFTER_DEMANDS,
    AFTER_FILL,
    AFTER_SATISFY,
    AT_MERGE,
    Context,
    Repair,
    RepairLog,
    apply_all,
    build_sequence,
    check_order,
)

__all__ = [
    "AFTER_DEMANDS",
    "AFTER_FILL",
    "AFTER_SATISFY",
    "AT_MERGE",
    "Context",
    "Repair",
    "RepairLog",
    "apply_all",
    "build_sequence",
    "check_order",
    "align_cell_levels",
    "check_local_ids",
    "coerce_numeric_values",
    "derive_acquisition_types",
    "derive_analysis_ids",
    "derive_coordinate_spaces",
    "derive_denominators",
    "derive_table_effects",
    "drop_redundant_cell_levels",
    "fill_directions",
    "link_entities_by_name",
    "listify_nested",
    "listify_scalars",
    "mirror_withheld",
    "names_agree",
    "rehome_stray_tables",
    "relabel_conclusions",
    "repair_references",
    "repair_wrappers",
    "repoint_out_of_scope_terms",
    "resolve_source_table_analysis",
    "scope_duplicate_terms",
    "unwrap_plain_slots",
    "unwrap_singleton_lists",
]
