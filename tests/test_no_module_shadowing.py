"""No function may bind a name that shadows a module its own file imports.

This has bitten five times across five review rounds -- a local named `values`, `paths`,
`tables`, `parse_keys`, `gaps` or `schema` over the module of the same name. It is a
particularly bad failure because it is invisible to pyflakes (the name IS defined), invisible
to the tests (the shadowed branch usually is not exercised), and produces an `AttributeError`
that names a type rather than the mistake -- `'list' object has no attribute 'CORPUS'`.

Mechanical renames are what generate it, and this package has just had several.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pondie import paths

SOURCES = sorted(paths.REPO.joinpath("pondie").rglob("*.py"))


def _module_imports(tree: ast.Module) -> set[str]:
    """Names bound at module level by an import, which a function could shadow."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {(a.asname or a.name.split(".")[0]) for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {(a.asname or a.name) for a in node.names}
    return names


def _bindings(fn: ast.FunctionDef) -> set[str]:
    """Every name the function binds: parameters, assignments, loop targets, `with ... as`."""
    args = fn.args
    bound = {a.arg for a in args.args + args.posonlyargs + args.kwonlyargs}
    bound |= {a.arg for a in (args.vararg, args.kwarg) if a}
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node is not fn:
            continue
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign,)):
            targets = list(node.targets)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, ast.withitem) and node.optional_vars:
            targets = [node.optional_vars]
        for target in targets:
            bound |= _names_bound_by(target)
    return bound


def _names_bound_by(target: ast.expr) -> set[str]:
    """The names a target actually rebinds.

    Only bare names and the elements of a tuple/list unpack. `sys.modules[k] = v` and
    `obj.attr = v` mutate something the name already refers to -- walking for every `Name`
    inside the target reports `sys` as shadowed, which is how this check first failed on
    code that was correct.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_names_bound_by(e) for e in target.elts)) if target.elts else set()
    if isinstance(target, ast.Starred):
        return _names_bound_by(target.value)
    return set()


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: str(p.relative_to(paths.REPO)))
def test_no_function_shadows_a_module_its_file_imports(source: Path) -> None:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = _module_imports(tree)
    # Only the ones that read as modules. A `from x import some_function` shadowed by a
    # local of the same name is a different, far less confusing mistake.
    modules = {
        name
        for name in imported
        if name.islower() and not name.startswith("_") and "_" not in name.strip("_")
        or name in {"parse_keys", "table_parse", "text_index", "span_tools"}
    }
    offences = [
        f"{fn.name}() at line {fn.lineno} binds {sorted(_bindings(fn) & modules)}"
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and (_bindings(fn) & modules)
    ]
    assert not offences, (
        f"{source.name}: a local shadows a module imported by this file -- "
        + "; ".join(offences)
    )
