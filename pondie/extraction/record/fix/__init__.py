"""The deterministic fixes, in three kinds. `repairs` holds the order; this holds the work.

A fix takes a record and returns one line per change, mutating in place. Three kinds recur and
the difference is what each may assume about its input:

  shape    make the record match the schema. Runs first, because nothing schema-guided can
           read a record whose slots hold the wrong sort of thing
  derive   fill a slot from another slot, from the stage-1 parse, or from a join the record
           already contains. Reads no paper, which is what makes it deterministic
  link     make the references resolve, and refuse a guess. A fix decides nothing: where two
           answers are possible the record keeps its defect and `rules` tells a human

Re-exported flat because `repairs.build_sequence` names them and a caller should not have to
know which kind a given fix is to call it. The kinds are for the reader.
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

__all__ = [
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
