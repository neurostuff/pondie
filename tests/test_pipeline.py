"""The scheduler both workflows run on.

The property worth guarding is the one `extraction` did not have: an output is reused only
while everything it was computed from is unchanged. A cache that cannot say that is a way of
serving last week's answer, and the failure is silent -- which is why most of what follows is
about staleness rather than about hits.
"""

from __future__ import annotations

import json
import threading


from pondie.pipeline import Stamp, Step, digest_of, execute, fresh


class Work:
    """A step over integers, with a dependency the test can change under it."""

    def __init__(self, root, version="1", fail_on=()):
        self.root, self.version, self.fail_on = root, version, set(fail_on)
        self.calls: list[int] = []

    def produces(self, n):
        return self.root / f"{n}.json"

    def depends_on(self, n):
        return {"n": n, "version": self.version}

    def run(self, n):
        self.calls.append(n)
        if n in self.fail_on:
            raise RuntimeError(f"no good: {n}")
        self.produces(n).write_text(json.dumps({"n": n}))
        return f"wrote {n}"

    def step(self, name="work", **kw):
        return Step(
            name=name, produces=self.produces, depends_on=self.depends_on, run=self.run, **kw
        )


def run(items, steps, **kw):
    kw.setdefault("progress", False)
    return execute(items, steps, name_of=lambda n: f"i{n}", **kw)


# --- caching, and the invalidation that makes it safe ------------------------


def test_a_second_run_reuses_what_the_first_produced(tmp_path):
    w = Work(tmp_path)
    assert run([1, 2], [w.step()]).tally() == {"done": 2}
    assert run([1, 2], [w.step()]).tally() == {"cached": 2}
    assert w.calls == [1, 2]


def test_changing_what_a_step_depends_on_invalidates_it(tmp_path):
    """The fault this module exists for. `extraction` was done when the output file existed,
    so a changed prompt, model or schema reused every stale payload without a word."""
    w = Work(tmp_path)
    run([1], [w.step()])
    w.version = "2"
    assert run([1], [w.step()]).tally() == {"done": 1}
    assert w.calls == [1, 1]


def test_redo_ignores_a_stamp_that_still_matches(tmp_path):
    w = Work(tmp_path)
    run([1], [w.step()])
    assert run([1], [w.step()], redo=True).tally() == {"done": 1}


def test_an_output_with_no_stamp_is_stale(tmp_path):
    """What adopting this module looks like: everything already on disk recomputes once,
    rather than being trusted on the strength of existing."""
    w = Work(tmp_path)
    w.produces(1).write_text("{}")
    assert fresh(w.step(), 1) is None


def test_an_unreadable_stamp_is_stale_rather_than_an_error(tmp_path):
    w = Work(tmp_path)
    run([1], [w.step()])
    Stamp.path_for(w.produces(1)).write_text("{ this is not json")
    assert fresh(w.step(), 1) is None


def test_a_failed_step_leaves_no_stamp_behind(tmp_path):
    """Stamping before the work would mark a crashed step complete and the next run would
    trust it. The output here is written by an earlier attempt and the failure comes after."""
    w = Work(tmp_path, fail_on=[1])
    w.produces(1).write_text("{}")
    report = run([1], [w.step()])
    assert report.tally() == {"failed": 1}
    assert not Stamp.path_for(w.produces(1)).exists()


def test_a_step_that_produces_no_file_is_never_cached(tmp_path):
    w = Work(tmp_path)
    step = Step(name="effectful", produces=lambda n: None, depends_on=w.depends_on, run=w.run)
    run([1], [step])
    assert run([1], [step]).tally() == {"done": 1}, "nothing on disk means nothing to trust"


# --- the digest --------------------------------------------------------------


def test_the_digest_ignores_key_order_and_path_objects(tmp_path):
    assert digest_of({"a": 1, "b": tmp_path}) == digest_of({"b": tmp_path, "a": 1})


def test_the_digest_changes_when_any_part_does(tmp_path):
    assert digest_of({"a": 1}) != digest_of({"a": 2})
    assert digest_of({"a": 1}) != digest_of({"a": 1, "b": 1})


# --- provenance ---------------------------------------------------------------


def test_the_stamp_says_what_produced_the_output_and_from_what(tmp_path):
    """Caching and provenance are one file, so the two cannot disagree about what ran."""
    w = Work(tmp_path, version="7")
    run([1], [w.step(name="demands")])
    body = json.loads(Stamp.path_for(w.produces(1)).read_text())
    assert body["step"] == "demands"
    assert body["parts"] == {"n": 1, "version": "7"}
    assert body["digest"] == digest_of({"n": 1, "version": "7"}, "demands")
    assert body["produced_at"] and body["seconds"] >= 0


# --- ordering and failure ------------------------------------------------------


def test_steps_run_in_order_within_one_item(tmp_path):
    order = []
    steps = [
        Step(
            name=f"s{i}",
            produces=lambda n, i=i: tmp_path / f"{n}-{i}.json",
            depends_on=lambda n: {"n": n},
            run=lambda n, i=i: (order.append(i), (tmp_path / f"{n}-{i}.json").write_text("{}")),
        )
        for i in (1, 2, 3)
    ]
    run([9], steps)
    assert order == [1, 2, 3]


def test_a_failed_step_stops_that_item_and_not_the_run(tmp_path):
    w = Work(tmp_path, fail_on=[2])
    after = Step(
        name="after",
        produces=lambda n: tmp_path / f"after-{n}.json",
        depends_on=lambda n: {"n": n},
        run=lambda n: (tmp_path / f"after-{n}.json").write_text("{}"),
    )
    report = run([1, 2, 3], [w.step(), after])
    assert report.tally() == {"done": 4, "failed": 1}
    assert [o.item for o in report.failures] == ["i2"]
    assert not (tmp_path / "after-2.json").exists(), "the step after a failure must not run"
    assert (tmp_path / "after-1.json").exists(), "other items carry on"


def test_the_failure_detail_names_the_exception(tmp_path):
    w = Work(tmp_path, fail_on=[1])
    assert "RuntimeError: no good: 1" in run([1], [w.step()]).failures[0].detail


# --- parallelism and the journal ------------------------------------------------


def test_items_run_concurrently_and_steps_do_not(tmp_path):
    """Two steps of one item cannot overlap: a step whose input is the step before it is
    indistinguishable from one merely listed after it."""
    live, peak, lock = 0, 0, threading.Lock()

    def body(n):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        threading.Event().wait(0.05)
        with lock:
            live -= 1
        (tmp_path / f"{n}.json").write_text("{}")

    step = Step(
        name="slow",
        produces=lambda n: tmp_path / f"{n}.json",
        depends_on=lambda n: {"n": n},
        run=body,
    )
    run(list(range(6)), [step], workers=4)
    assert peak > 1, "the pool must actually overlap items"


def test_max_parallel_bounds_one_step_under_the_run_s_workers(tmp_path):
    live, peak, lock = 0, 0, threading.Lock()

    def body(n):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        threading.Event().wait(0.05)
        with lock:
            live -= 1
        (tmp_path / f"{n}.json").write_text("{}")

    step = Step(
        name="scarce",
        produces=lambda n: tmp_path / f"{n}.json",
        depends_on=lambda n: {"n": n},
        run=body,
        max_parallel=2,
    )
    run(list(range(8)), [step], workers=8)
    assert peak <= 2, f"max_parallel=2 admitted {peak} at once"


def test_the_journal_is_appended_as_the_run_goes(tmp_path):
    """Written as it happens, not at the end: a run killed at minute thirty used to leave no
    record of the thirty minutes."""
    w = Work(tmp_path, fail_on=[2])
    events = tmp_path / "events.jsonl"
    run([1, 2, 3], [w.step()], events=events, workers=3)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert len(rows) == 3
    assert {r["state"] for r in rows} == {"done", "failed"}
    assert all(r["item"] and r["step"] == "work" and r["at"] for r in rows)


def test_two_steps_writing_one_path_do_not_read_each_other_s_stamp(tmp_path):
    """The step's name is in the digest. Without it the second step here finds the first's
    stamp, matches on the shared `depends_on`, and reports a cache hit for work never done."""
    w = Work(tmp_path)
    report = run([1], [w.step("a"), w.step("b")])
    assert report.tally() == {"done": 2}


def test_the_summary_counts_items_rather_than_outcomes(tmp_path):
    a, b = Work(tmp_path / "a"), Work(tmp_path / "b")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    report = run([1, 2], [a.step("a"), b.step("b")])
    assert "2 item(s)" in report.summary()
    assert "4 done" in report.summary()


def test_an_empty_run_is_not_an_error(tmp_path):
    assert execute([], [], progress=False).tally() == {}


def test_a_stamp_is_invisible_to_a_glob_of_the_output_directory(tmp_path):
    """`builder.merge_payloads` globs `<payload_dir>/*.json` and merges what it finds. A
    stamp beside the payload is a file that glob picks up -- the trap the `Tables` and
    `Repair` docstrings each record having been caught by once."""
    w = Work(tmp_path)
    run([1], [w.step()])
    assert Stamp.read(w.produces(1)) is not None, "the stamp must still be findable"
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["1.json"]
