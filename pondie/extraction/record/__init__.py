"""Turning payloads into a record: assemble, fix, warrant, check.

    builder    merge the payloads and produce the record
    fix        the deterministic fixes -- what each does, and the order they run in
    spans      resolve a verbatim quote into character offsets
    direction  read a contrast's polarity off its own name; mirror a withheld half
    effect     derive an effect's kind from its cells
    validate   does the result conform to the extraction schema
    rules      the things that are legal and scientifically wrong

Fixing here is deterministic and decides nothing: where two answers are possible the record
keeps its defect and `rules` tells a human. The `repair` stage, which runs after this and
asks a model to settle what the record cannot, is `extraction/repair.py`.

`builder.build` is the whole of it: everything else is called from there, in the order
`fix.build_sequence()` declares. A record that skipped any of it is not a partial record,
it is one nothing downstream can check.
"""
