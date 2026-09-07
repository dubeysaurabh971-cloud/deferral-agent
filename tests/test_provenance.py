"""Provenance and refusal invariants -- the class of bug where a field reports something false.

Finding 11 was a positional index into an append-only log that silently pointed at the wrong
prompt. The same shape turned up twice more in review: an aggregate reading top_k from ambient
config rather than from the runs (v5_topk5.json claimed top_k=10, and the per-run reports did
not record it, so nothing but a human-typed label distinguished the two frontier configurations),
and a category mean falling back to 0.0 when a run was missing that category.

Both are silent by nature: they produce a plausible number rather than an error. Tests are the
only thing that notices.
"""
import json

import pytest

from src.eval import aggregate as agg


def report(**over) -> dict:
    """A complete, passing report skeleton."""
    base = {
        "n_failed": 0,
        "golden_coverage": "complete",
        "adversarial_coverage": "complete",
        "resolver_model": "test-model",
        "review_clarifications": False,
        "retrieval_top_k": 10,
        "golden_decision_accuracy": 0.9,
        "adversarial_decision_accuracy": 0.7,
        "token_spend": {"input_tokens": 1, "output_tokens": 1, "calls": 1},
        "golden_results": [
            {"item_id": "G001", "decision": "RESOLVE", "decision_correct": True}
        ],
        "adversarial_results": [
            {"item_id": "A001", "category": "out_of_kb", "expected_decision": "ESCALATE",
             "decision": "ESCALATE", "decision_correct": True},
        ],
        "adversarial_by_category": {"out_of_kb": {"n": 1, "decision_accuracy": 0.7}},
    }
    base.update(over)
    return base


@pytest.fixture
def write(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "RESULTS_DIR", tmp_path)

    def _write(name, rep):
        (tmp_path / name).write_text(json.dumps(rep), encoding="utf-8")
        return name

    return _write


# --- top_k provenance --------------------------------------------------------------

def test_top_k_comes_from_the_runs_not_from_ambient_config(write, monkeypatch):
    """The bug: aggregate wrote config.RETRIEVAL_TOP_K, so the recorded value was whatever the
    environment held at aggregation time. Here config says 10 and the runs say 5; 5 must win."""
    monkeypatch.setattr(agg.config, "RETRIEVAL_TOP_K", 10)
    names = [write(f"r{i}.json", report(retrieval_top_k=5)) for i in range(2)]
    out = agg.aggregate(names, "label")
    assert out["retrieval_top_k"] == 5
    assert out["retrieval_top_k_provenance"] == "recorded by the run"


def test_unrecorded_top_k_is_unknown_rather_than_guessed(write, monkeypatch):
    monkeypatch.setattr(agg.config, "RETRIEVAL_TOP_K", 10)
    names = [write(f"r{i}.json", report(retrieval_top_k=None)) for i in range(2)]
    out = agg.aggregate(names, "label")
    assert out["retrieval_top_k"] is None
    assert out["retrieval_top_k_provenance"] == "not recorded"


def test_operator_asserted_top_k_is_labelled_as_such(write):
    names = [write(f"r{i}.json", report(retrieval_top_k=None)) for i in range(2)]
    out = agg.aggregate(names, "label", asserted_top_k=5)
    assert out["retrieval_top_k"] == 5
    assert out["retrieval_top_k_provenance"] == "asserted by the operator"


def test_recorded_top_k_beats_an_operator_assertion(write):
    """If the runs said it, the operator does not get to overrule it."""
    names = [write(f"r{i}.json", report(retrieval_top_k=10)) for i in range(2)]
    out = agg.aggregate(names, "label", asserted_top_k=5)
    assert out["retrieval_top_k"] == 10
    assert out["retrieval_top_k_provenance"] == "recorded by the run"


# --- refusals ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "field,a,b",
    [
        ("retrieval_top_k", 5, 10),
        ("review_clarifications", True, False),
        ("resolver_model", "gpt-5-mini", "gpt-4.1"),
    ],
)
def test_refuses_to_average_runs_from_different_configurations(write, field, a, b):
    """Averaging across configurations produces a number no configuration ever had."""
    names = [write("a.json", report(**{field: a})), write("b.json", report(**{field: b}))]
    with pytest.raises(SystemExit, match=field):
        agg.aggregate(names, "label")


def test_refuses_to_average_a_run_that_lost_items(write):
    names = [write("a.json", report()), write("b.json", report(n_failed=1))]
    with pytest.raises(SystemExit, match="incomplete"):
        agg.aggregate(names, "label")


def test_absent_category_raises_instead_of_averaging_in_a_zero(write):
    """The bug: a missing category fell back to 0.0, which is indistinguishable from a measured
    0% -- and the baseline legitimately scores 0% on three categories, so it is not spottable by
    eye either. It would drag a mean down exactly like a real regression."""
    full = report()
    missing = report(adversarial_by_category={})
    names = [write("a.json", full), write("b.json", missing)]
    with pytest.raises(SystemExit, match="out_of_kb"):
        agg.aggregate(names, "label")


def test_a_genuine_zero_still_aggregates(write):
    """The other side of it: 0% must remain a legal measured value."""
    zero = report(adversarial_by_category={"out_of_kb": {"n": 1, "decision_accuracy": 0.0}})
    names = [write(f"r{i}.json", zero) for i in range(2)]
    out = agg.aggregate(names, "label")
    assert out["by_category"]["out_of_kb"]["mean"] == 0.0


# --- retrieval: defaults must not be captured at import time ------------------------

def test_retrieve_does_not_bind_top_k_at_import_time():
    """The bug: `def retrieve(self, query, top_k=config.RETRIEVAL_TOP_K)` evaluates the default
    once, when the module is imported. Anything setting config.RETRIEVAL_TOP_K afterwards got
    retrieval using the stale value while resolver.py stamped the NEW one into gate_config,
    which is read at call time -- so the trace claimed a top_k retrieval had not used.

    Asserted on the signature because constructing a real HybridRetriever needs the embedded
    index, which is gitignored and absent in CI.
    """
    import inspect

    from src.retrieval import HybridRetriever

    for name, method in (("retrieve", HybridRetriever.retrieve),
                         ("__init__", HybridRetriever.__init__)):
        default = inspect.signature(method).parameters["top_k"].default
        assert default is None, (
            f"{name}'s top_k default is {default!r}, captured at import time. Use None and "
            "resolve it inside, or a trace can report a top_k that retrieval did not use."
        )


def test_retrieve_resolves_top_k_from_the_instance():
    """The value bound at construction is what retrieve() honours, and an explicit argument
    still overrides it. This is the property resolver.py relies on when it records
    retriever.top_k as the run's provenance."""
    from src.retrieval import HybridRetriever

    r = HybridRetriever.__new__(HybridRetriever)
    r.top_k = 3
    r.candidate_pool = 20
    ids = [f"c{i}" for i in range(10)]
    r.chunk_by_id = {i: {"text": "t", "title": "T", "url": "u"} for i in ids}
    r._bm25_ranking = lambda q: ids
    r._dense_ranking = lambda q: ids

    assert len(r.retrieve("q")) == 3, "must use the instance value"
    assert len(r.retrieve("q", top_k=7)) == 7, "an explicit argument must still win"


# --- retrieval: nlargest must rank identically to the full sort it replaced ----------

def test_nlargest_selection_matches_a_full_sort_including_ties():
    """`_bm25_ranking` swapped sorted()[:k] for heapq.nlargest to avoid ranking 10k chunks to
    keep 20. Same result, but only if ties break the same way -- and 63 of the real chunks score
    exactly 0.0, so ties are not hypothetical. Verified on all 160 real queries at the time of
    the change (identical in order and set); this pins the property without the index.
    """
    import heapq

    from src.retrieval import HybridRetriever

    class FakeBM25:
        def __init__(self, scores):
            self._scores = scores

        def get_scores(self, _tokens):
            return self._scores

    for scores in (
        [3.0, 1.0, 2.0, 5.0, 4.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],            # all tied
        [0.0, 0.0, 9.0, 0.0, 9.0, 0.0],       # ties either side of the cut
        [5.0, 5.0, 5.0, 1.0, 0.0],
    ):
        r = HybridRetriever.__new__(HybridRetriever)
        r.candidate_pool = 3
        r._bm25 = FakeBM25(scores)
        r.chunks = [{"chunk_id": f"c{i}"} for i in range(len(scores))]

        got = r._bm25_ranking("q")
        want_idx = heapq.nlargest(3, range(len(scores)), key=scores.__getitem__)
        expected = [f"c{i}" for i in want_idx]
        assert got == expected

        # and that nlargest itself agrees with the construct it replaced
        old_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        assert want_idx == old_idx, f"tie-break diverged on {scores}"
