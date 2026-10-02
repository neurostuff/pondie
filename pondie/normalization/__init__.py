"""Post-hoc normalization of extracted record fields, one module per field.

A field's shape decides its method: closed target, link, seed-and-cluster, or partition.
Every field module exposes `normalize(...)`, returning a value and the reason it was
chosen, and `report(...)` for the residual. Modules with a leading underscore are shared
machinery and are not part of that interface.

Which field takes which shape, what is deliberately not a field module, and why any of it
is arranged this way: docs/normalization-rationale.md.
"""

from __future__ import annotations

UNKNOWN = "UNKNOWN"
OTHER = "OTHER"

__all__ = ["UNKNOWN", "OTHER", "apply_derived", "derived_fields", "fields"]

#: The modules that FILL a slot rather than normalizing one, in the order `apply_derived`
#: runs them. Each exposes `apply(record) -> tally`.
DERIVED = (
    "is_healthy",
    "population_characteristics",
    "age_unit",
    "sex_distribution",
    "handedness_distribution",
)


def derived_fields() -> list[str]:
    """Every module exposing `apply`, sorted. Derived, not kept, as `fields()` is."""
    import importlib
    import pkgutil

    found = []
    for module in pkgutil.iter_modules(__path__):
        loaded = importlib.import_module(f"{__name__}.{module.name}")
        if callable(getattr(loaded, "apply", None)):
            found.append(module.name)
    return sorted(found)


def apply_derived(record: dict) -> dict[str, dict]:
    """Run every derived-slot fill over one record, in place. Tally per module.

    The one seam for "fill what code decides rather than what a model was asked". Before
    it there was none: three modules exposed `apply` and nothing called any of them, so
    slots the storage schema marks `deterministic` were answered by the model and the
    derivation meant to overwrite the answer never ran.

    A field belongs here only if its derivation is question-independent. `Group.role` was
    built, measured and removed for failing that test -- see
    docs/normalization-rationale.md, "group_role - built, measured, removed".
    """
    import importlib

    tallies: dict[str, dict] = {}
    for name in DERIVED:
        module = importlib.import_module(f"{__name__}.{name}")
        tallies[name] = module.apply(record)
    return tallies


def fields() -> list[str]:
    """Every module that exposes `normalize`, sorted. Derived, not kept.

    docs/normalization-rationale.md, "What is and is not a field module".
    """
    import importlib
    import pkgutil

    found = []
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        loaded = importlib.import_module(f"{__name__}.{module.name}")
        if callable(getattr(loaded, "normalize", None)):
            found.append(module.name)
    return sorted(found)
