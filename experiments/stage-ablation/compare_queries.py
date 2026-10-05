"""Score every pool's latest run with two versions of queries.py; list what moves.

    PONDIE_DATA_DIR=... PYTHONPATH=<code> python compare_queries.py BEFORE_QUERIES_PY
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ablate_rules import selected

HERE = Path(__file__).resolve().parent
POOLS = [
    ("36100907", "ptsd55_negatives.pmids", "--overlap", ["ptsd55_s2r3", "ptsd_default"]),
    ("35664889", "neg_dementia.pmids", "", ["dem55_s2r", "dem_p_effort_medium"]),
    ("36115222", "neg_sud.pmids", "", ["sud55_s2r2", "sud_medium-v1"]),
    ("34400176", "cue.neg.pmids", "", ["cue55_s2r2"]),
    ("32078973", "dm.neg.pmids", "", ["dm55_s2r"]),
    ("29944961", "ps.neg.pmids", "", ["ps55_s2r"]),
    ("36436737", "soc.neg.pmids", "", ["soc55_s2r"]),
]


def run(directory: Path) -> tuple[dict, dict]:
    picks, lines = {}, {}
    for meta, neg, extra, runs in POOLS:
        cmd = [sys.executable, "score.py", *runs, "--meta", meta, "--negatives", neg,
               "--labels", "adjudicated", "--gold-coords", "--detail", *extra.split()]
        text = subprocess.run(cmd, cwd=directory, capture_output=True, text=True).stdout
        picks.update(selected(text))
        run_name = None
        for line in text.splitlines():
            if line.startswith("=== "):
                run_name = line.split()[1]
            elif line.strip().startswith("veto ") and run_name:
                lines[run_name] = line.strip()
    return picks, lines


def main() -> int:
    after, after_lines = run(HERE)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "exp"
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(sys.argv[1], copy / "queries.py")
        before, before_lines = run(copy)
    for name in after_lines:
        print(f"{name:22s} before: {before_lines.get(name)}\n{'':22s} after:  {after_lines[name]}")
        lost, gained = before.get(name, set()) - after[name], after[name] - before.get(name, set())
        if lost or gained:
            print(f"{'':22s} lost {sorted(lost)} gained {sorted(gained)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
