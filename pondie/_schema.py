"""Where the schema's data files are.

The schema is not only code. `extraction-readme.md` and `representing-models.md` are sent to
the model as part of the prompt, and the extraction and storage YAML are what a record is
validated against -- all of them live in the `study-schema` distribution beside the modules
that read them.

Their location is derived from the installed module rather than from a checkout layout, so it
is correct whether the distribution was installed from this repository's submodule or from
anywhere else, and there is no path for a caller to get wrong.
"""

from __future__ import annotations

from pathlib import Path

import schema_utils

#: The schema checkout, wherever `study-schema` was installed from.
ROOT = Path(schema_utils.__file__).resolve().parent

EXTRACTION = ROOT / "neuroimaging-study-extraction.yaml"
STORAGE = ROOT / "neuroimaging-study-storage.yaml"
