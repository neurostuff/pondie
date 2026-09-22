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

__all__ = ["UNKNOWN", "OTHER"]


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
