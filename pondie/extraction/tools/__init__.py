"""Things done to records after a run, none of them part of one.

    adjudicate     read a record the way a curator has to, and apply corrections
    derive_fields  fill from code the fields code can be trusted with, and abstain otherwise

Kept apart from the pipeline because none of them runs inside it: each takes records that
already exist and is safe to re-run, which the stages are not.
"""
