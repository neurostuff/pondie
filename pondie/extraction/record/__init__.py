"""Turning payloads into a record: assemble, repair, warrant, check.

    builder    merge the payloads and produce the record
    repairs    the 16 deterministic fixes, in order, with their reasons
    spans      resolve a verbatim quote into character offsets
    direction  read a contrast's polarity off its own name; mirror a withheld half
    effect     derive an effect's kind from its cells
    validate   does the result conform to the extraction schema
    rules      the thirteen things that are legal and scientifically wrong

`builder.build` is the whole of it: everything else is called from there, in the order
`repairs.build_sequence()` declares. A record that skipped any of it is not a partial record,
it is one nothing downstream can check.
"""
