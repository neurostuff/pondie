"""Shards of ns request bodies (`bodies_ns.py` output) for the Vista extraction, and their run configs.

    python shard_ns.py BODIES.jsonl OUT_DIR --first 85 [--template CONFIG.json] [--configs DIR] [--max-tokens 46000]

Bodies over --max-tokens input tokens are left out (OUT_DIR/over-46k.txt: sid, tokens). The rest go, longest
first, to the shard with the least input so far that has room (at most 500 papers), so shards take about the same
time. Shards are OUT_DIR/sNNN.bodies.jsonl from --first on; with --template (an existing ns config), each gets
DIR/ns-27b-e-sNNN.json with its run name and bodies path on Vista (VISTA_DIR/sNNN.bodies.jsonl). This is how the
2026-10-28 run's 85 shards were built.
"""
import argparse
import json
from pathlib import Path

SIZE = 500


def tokens_and_sid(line):
    i = line.find('"n_input_tokens": ')
    n = int(line[i + 18:line.index(",", i)])
    return n, json.loads(line)["sid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bodies")
    ap.add_argument("out")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=46000)
    ap.add_argument("--template")
    ap.add_argument("--configs")
    ap.add_argument("--vista-dir", help="where the shards will sit on Vista (default: OUT's name under data/ns)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, over = [], []
    for line in open(a.bodies):
        n, sid = tokens_and_sid(line)
        (over if n > a.max_tokens else rows).append((n, sid, line))
    rows.sort(key=lambda r: -r[0])
    count = (len(rows) + SIZE - 1) // SIZE
    shards, load = [[] for _ in range(count)], [0] * count
    for r in rows:
        k = min(range(count), key=lambda j: load[j] if len(shards[j]) < SIZE else float("inf"))
        shards[k].append(r)
        load[k] += r[0]
    names = []
    for k, shard in enumerate(shards):
        name = f"s{a.first + k:03d}"
        names.append(name)
        with open(out / f"{name}.bodies.jsonl", "w") as f:
            f.writelines(r[2] for r in shard)
    (out / "over-46k.txt").write_text("".join(f"{s}\t{n}\n" for n, s, _ in over))
    if a.template:
        template = json.load(open(a.template))
        vista = a.vista_dir or f"/work/07723/jdk3232/vista/astra-distill/data/ns/{out.name}"
        cdir = Path(a.configs)
        cdir.mkdir(parents=True, exist_ok=True)
        for name in names:
            cfg = dict(template, run=f"ns-27b-e-{name}", bodies=f"{vista}/{name}.bodies.jsonl")
            (cdir / f"ns-27b-e-{name}.json").write_text(json.dumps(cfg, indent=1))
    print(f"shards {names[0]}-{names[-1]} ({count}), papers {len(rows)}, over {a.max_tokens}: {len(over)}, "
          f"sizes {min(map(len, shards))}-{max(map(len, shards))}, "
          f"M tokens per shard {min(load) / 1e6:.2f}-{max(load) / 1e6:.2f}")


if __name__ == "__main__":
    main()
