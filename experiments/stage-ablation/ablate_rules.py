"""Revert one query rule at a time and rescore stored runs: what each one-paper fix moves.

    PONDIE_DATA_DIR=... PYTHONPATH=<code> python ablate_rules.py

Each variant is a text edit of queries.py in a temporary copy of this directory, scored with
score.py there, so the real query is never touched.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "no 'negative for PTSD'": ('r"negative for (current |lifetime )?ptsd|"\n', ""),
    "grey label is not structural": (
        '("structural_morphometry" in strs(m.get("family"))\n'
        '                          or any(GREY.search(k) for k in strs(m.get("type")) + strs(m.get("source_label"))))',
        '"structural_morphometry" in strs(m.get("family"))'),
    "mixed level is not the case side": (
        "    if True in verdicts:\n        return True, term, level\n    if False in verdicts:\n        return False, term, level",
        "    if verdicts == {True}:\n        return True, term, level\n    if verdicts == {False}:\n        return False, term, level"),
    "dementia modality by type only": (
        "if kinds & GREY_TYPES or (GREY.search(label) and not kinds & (BOLD_TYPES | PET_TYPES)):",
        "if kinds & GREY_TYPES:"),
}
RUNS = [
    ("36100907", "ptsd55_negatives.pmids", "--overlap", ["ptsd_default", "ptsd55_s2r", "ptsd55_s2r2", "ptsd55_s2r3"]),
    ("35664889", "neg_dementia.pmids", "", ["dem_p_effort_medium", "dem55_s2r"]),
    ("36115222", "neg_sud.pmids", "", ["sud_medium-v1", "sud55_s2r", "sud55_s2r2"]),
]


def selected(text: str) -> dict[str, set[str]]:
    """run -> the papers veto selects, from score.py --detail output."""
    out, run = {}, None
    for line in text.splitlines():
        if line.startswith("=== "):
            run = line.split()[1]
            out.setdefault(run, set())
        m = re.match(r"\s+(SEL|veto)\s+(gold|neg)\s+(\d+)", line)
        if m and run:
            out[run].add(f"{m.group(3)}({m.group(2)})")
    return out


def score(directory: Path) -> dict[str, set[str]]:
    result = {}
    for meta, neg, extra, runs in RUNS:
        cmd = [sys.executable, "score.py", *runs, "--meta", meta, "--negatives", neg,
               "--labels", "adjudicated", "--gold-coords", "--detail", *extra.split()]
        text = subprocess.run(cmd, cwd=directory, capture_output=True, text=True).stdout
        result.update(selected(text))
    return result


def main() -> int:
    base = score(HERE)
    for name, (old, new) in VARIANTS.items():
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "exp"
            shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("__pycache__"))
            source = (copy / "queries.py").read_text()
            if old not in source:
                print(f"{name}: pattern not found"); continue
            (copy / "queries.py").write_text(source.replace(old, new, 1))
            varied = score(copy)
        flips = {run: (base[run] - varied.get(run, set()), varied.get(run, set()) - base[run])
                 for run in base}
        moved = {r: f for r, f in flips.items() if f[0] or f[1]}
        print(f"== {name}: " + ("no paper moves" if not moved else ""))
        for run, (lost, gained) in moved.items():
            print(f"   {run}: reverting loses {sorted(lost)} gains {sorted(gained)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
