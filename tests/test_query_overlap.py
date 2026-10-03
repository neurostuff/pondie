"""`pondie.query.overlap`: a paper re-reporting an earlier paper's participants.

Each case is a pair from the VBM-of-PTSD pool (experiments/stage-ablation/JOURNAL.md, E5),
reduced to the fields the rule reads.
"""

from pondie.query.overlap import overlapping, re_reports


def _v(value):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported"}


def _group(status, size=None, **sex):
    group = {"local_id": f"grp_{status}_{size}", "_status": status}
    if size is not None:
        group["acquired_count"] = _v(size)
    group["sex_distribution"] = [{"category": _v(k), "count": _v(v)} for k, v in sex.items()]
    return group


def _record(*groups):
    return {"groups": list(groups)}


def status(group):
    return group["_status"]


FIRE_2006 = _record(_group("ptsd", 12, female=8, male=4), _group("tec", 12, female=8, male=4))
FIRE_2009 = _record(_group("ptsd", 12, females=8, males=4), _group("tec", 12, females=8, males=4))
FIRE_NO_SEX = _record(_group("ptsd", 12), _group("tec", 12))
COAL_2011 = _record(_group("ptsd", 10, male=10), _group("tec", 10, male=10))
COAL_2012 = _record(_group("ptsd", 10, male=10), _group("tec", 10, male=10),
                    _group("hc", 20, male=20))
NARDO_2010 = _record(_group("ptsd", 21, women=6, men=15), _group("tec", 22, women=6, men=16))
NARDO_2013 = _record(_group("ptsd", 15, male=12, female=3), _group("tec", 17, male=11, female=6))


def test_the_same_cohorts_reported_again_are_a_re_report():
    assert re_reports(FIRE_2009, FIRE_2006, status)


def test_a_subset_of_each_cohort_is_a_re_report():
    assert re_reports(NARDO_2013, NARDO_2010, status)


def test_a_paper_that_adds_a_cohort_is_not():
    """The 20 unexposed controls fit inside no cohort of the earlier paper."""
    assert not re_reports(COAL_2012, COAL_2011, status)


def test_a_later_paper_may_report_less_than_the_earlier_one():
    assert re_reports(FIRE_NO_SEX, FIRE_2006, status)


def test_but_not_more():
    """10 men from a coal-mine flood do not fit inside 12 fire survivors of unknown sex."""
    assert not re_reports(COAL_2011, FIRE_NO_SEX, status)


def test_a_cohort_with_no_size_cannot_be_shown_to_fit():
    assert not re_reports(_record(_group("ptsd")), FIRE_2006, status)


def test_only_the_later_paper_is_excluded_and_only_with_shared_authors():
    records = {"1": NARDO_2010, "2": NARDO_2013, "3": FIRE_2006, "4": FIRE_2009}
    authorship = {
        "1": {"authors": ["Nardo D", "Pagani M", "Högberg G"], "pubdate": "2010 May"},
        "2": {"authors": ["Nardo D", "Pagani M", "Högberg G"], "pubdate": "2013 Sep"},
        "3": {"authors": ["Chen S"], "pubdate": "2006 Jan 30"},
        "4": {"authors": ["Chen S"], "pubdate": "2009 Jun"},
    }
    assert overlapping(records, set(records), authorship, status) == {"2": "1"}


def test_an_earlier_paper_that_is_not_selected_excludes_nothing():
    records = {"1": NARDO_2010, "2": NARDO_2013}
    authorship = {p: {"authors": ["Nardo D", "Pagani M", "Högberg G"], "pubdate": d}
                  for p, d in (("1", "2010 May"), ("2", "2013 Sep"))}
    assert overlapping(records, {"2"}, authorship, status) == {}


def test_two_shared_authors_is_not_enough():
    """The coal-mine and fire papers share two authors and are different disasters."""
    records = {"1": FIRE_2006, "2": _record(_group("ptsd", 10), _group("tec", 10))}
    authorship = {"1": {"authors": ["Li L", "Zhang J", "Chen S"], "pubdate": "2006 Jan"},
                  "2": {"authors": ["Li L", "Zhang J", "Tan Q"], "pubdate": "2011 May"}}
    assert overlapping(records, set(records), authorship, status) == {}
