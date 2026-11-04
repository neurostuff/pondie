"""The compact format lives in the pondie-astra package (packages/pondie-astra); this name keeps the scripts here,
which `import compact`, working from a checkout without installing it."""
import sys
from pathlib import Path

try:
    from pondie_astra import compact as _compact
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages/pondie-astra/src"))
    from pondie_astra import compact as _compact

sys.modules[__name__] = _compact
