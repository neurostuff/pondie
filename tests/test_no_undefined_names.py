"""Every name a module uses is bound somewhere it can see.

An unbound name is invisible to the test suite whenever the line that uses it is only
reached at runtime. `repair/propose_with_extractor` called `render.paper_block` with no
`render` import for a whole commit: 1,146 tests passed, and the fault surfaced as
`NameError: name 'render' is not defined` on the 127-paper run, after every paper's
`repair` stage had already failed.

pyflakes answers this statically in under a second, which is cheaper than discovering it
from a log. Skipped rather than failed where it is not installed, because it is a
development tool and a missing one is not a defect in the package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_no_module_uses_a_name_it_never_bound() -> None:
    try:
        import pyflakes  # noqa: F401
    except ImportError:  # pragma: no cover -- a dev tool, not a dependency
        pytest.skip("pyflakes is not installed")

    done = subprocess.run(
        [sys.executable, "-m", "pyflakes", "pondie", "scripts", "tests"],
        cwd=ROOT, capture_output=True, text=True,
    )
    # Only the undefined-name family. Unused imports and shadowing are style, and
    # `test_no_module_shadowing` already owns the one of those that bit us.
    offences = [
        line for line in done.stdout.splitlines()
        if "undefined name" in line or "may be undefined" in line
    ]

    assert not offences, "a name is used where nothing binds it:\n" + "\n".join(offences)
