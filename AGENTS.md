# Agent Instructions
1

## Virtual environment
- If `.venv` exists at the repo root, activate it with:
  - `source .venv/bin/activate`
- Sanity check the active interpreter with `which python` and `python -V`.
- If `.venv` is missing and you need one, create it with `python -m venv .venv` (or `uv venv .venv` if preferred), then activate it.

## Installing packages (use uv)
- Install project deps (including dev/test tooling):
  - `uv pip install -e ".[dev]"`
- Add a new dependency:
  - `uv pip install <package>`
- If `uv pip install -e ".[abbrev]"` fails on this environment (nmslib build), install abbrev deps with:
  - `uv pip install spacy`
  - `uv pip install scispacy --no-deps`
- If you update dependencies, keep `pyproject.toml` in sync with what you installed.

## Running tests
- With the venv active and dev deps installed:
  - `pytest`
- Tests are under `tests/` per `pyproject.toml`.

## Keep these instructions current
- If you hit a roadblock and fix it, update this file with the corrected steps.
- Ask clarifying questions until you are ~95% confident in the implementation details.
- When offering choices, present them as a), b), c), d) options for quick selection.
- Use a), b), c), d) as answers/solutions, not as additional questions.
