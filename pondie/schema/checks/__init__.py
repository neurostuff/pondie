"""What has to hold for the schema to still mean what it says.

Each module is a check with its own `main`, runnable as `python -m pondie.schema.checks.<name>`,
and each fails in its own way rather than reporting a generic mismatch. They live here rather
than beside the YAML because they are code, and every line of code that reads the schema is
pondie's.
"""
