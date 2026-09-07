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


# --- one definition of the error taxonomy ------------------------------------------

def test_classify_is_consistent_with_taxonomy_over_the_whole_input_space():
    """taxonomy() is built on classify(), so they cannot disagree -- this pins that they are
    still wired together rather than merely equal today.

    There were two definitions. taxonomy() counted false escalations over golden items alone;
    the explorer's exporter called any resolve-expecting item that got deferred a false
    escalation. They differed on exactly one of 160 items, so the published page reported 15
    where its source report said 14, and the page was internally consistent with its own wrong
    classifier.
    """
    import itertools

    from src.eval.compare import (
        FALSE_ESCALATION, FALSE_RESOLUTION, MISROUTED, UNWARRANTED_DEFERRAL, classify, taxonomy,
    )

    decisions = ("RESOLVE", "CLARIFY", "ESCALATE")
    golden, adversarial, expect = [], [], {k: 0 for k in
                                          (FALSE_ESCALATION, FALSE_RESOLUTION, MISROUTED,
                                           UNWARRANTED_DEFERRAL)}
    n = 0
    for expected, decision in itertools.product(decisions, decisions):
        # golden items always expect RESOLVE, so only that row is constructible there
        if expected == "RESOLVE":
            n += 1
            golden.append({"item_id": f"G{n}", "decision": decision,
                           "expected_decision": "RESOLVE"})
            kind = classify("golden", "RESOLVE", decision)
            if kind:
                expect[kind] += 1
        n += 1
        adversarial.append({"item_id": f"A{n}", "decision": decision,
                            "expected_decision": expected, "category": "x"})
        kind = classify("adversarial", expected, decision)
        if kind:
            expect[kind] += 1

    t = taxonomy({"golden_results": golden, "adversarial_results": adversarial})
    assert t["false_escalations"] == expect[FALSE_ESCALATION]
    assert t["false_resolutions"] == expect[FALSE_RESOLUTION]
    assert t["misrouted_deferrals"] == expect[MISROUTED]
    assert t["unwarranted_adversarial_deferrals"] == expect[UNWARRANTED_DEFERRAL]


def test_false_escalation_stays_golden_only():
    """The convention every headline figure and the cost model use. An adversarial ticket that
    expects RESOLVE and was deferred is a real unnecessary handoff, but it is not a false
    escalation, because that is a rate over the 100 answerable golden items."""
    from src.eval.compare import FALSE_ESCALATION, UNWARRANTED_DEFERRAL, classify

    assert classify("golden", "RESOLVE", "CLARIFY") == FALSE_ESCALATION
    assert classify("adversarial", "RESOLVE", "CLARIFY") == UNWARRANTED_DEFERRAL
    assert classify("adversarial", "RESOLVE", "RESOLVE") is None


def test_deferral_errors_counts_every_unnecessary_handoff():
    """All three cost one handoff, so all three are in the cost model's constant term."""
    from src.eval.compare import deferral_errors

    t = {"false_escalations": 12, "misrouted_deferrals": 5,
         "unwarranted_adversarial_deferrals": 1}
    assert deferral_errors(t) == 18


def test_exported_summary_equals_taxonomy_of_its_source_report(monkeypatch, tmp_path):
    """The published artefact must agree with the artefact it was reconstructed from.

    This is the check that would have caught the off-by-one: the page and its source report were
    never compared, so a divergence in the classifier was invisible from either side alone -- the
    page was internally consistent with its own wrong rules.

    Self-contained: the real traces/runs.jsonl is gitignored and absent in CI, so a trace log is
    synthesised from the report's own recorded labels. That also exercises the --from-report join
    itself, since the traces have no gate_config and must be matched on labels alone.
    """
    import json as _json

    from src.eval import export_traces
    from src.eval.compare import taxonomy

    source_name = "v5_shipped_run3.json"
    source = _json.loads(
        (export_traces.config.ROOT_DIR / "eval_results" / source_name).read_text(encoding="utf-8")
    )

    golden = {it["item_id"]: it["question"] for it in export_traces.load_jsonl(
        export_traces.config.DATA_DIR / "golden" / "golden_set.jsonl")}
    adver = {it["item_id"]: it["ticket_text"] for it in export_traces.load_jsonl(
        export_traces.config.DATA_DIR / "adversarial" / "adversarial_set.jsonl")}
    texts = {**golden, **adver}

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    with (traces_dir / "runs.jsonl").open("w", encoding="utf-8") as f:
        for row in source["golden_results"] + source["adversarial_results"]:
            f.write(_json.dumps({
                "resolver": "gated",
                "ticket_text": texts[row["item_id"]],
                "decision": row["decision"],
                "model_decision": row["model_decision"],
                "kb_coverage": row["kb_coverage"],
                "requires_human_authority": row["requires_human_authority"],
                "missing_information": row["missing_information"],
                "clarify_review": row["clarify_review"],
                "policy_override": row["policy_override"],
                "reasoning": "synthesised for this test",
                "answer": "a",
                "retrieved_chunks": [],
            }) + "\n")

    monkeypatch.setattr(export_traces.config, "TRACES_DIR", traces_dir)
    monkeypatch.setattr(export_traces, "OUT", tmp_path / "data.js")
    export_traces.main(from_report=source_name)

    raw = (tmp_path / "data.js").read_text(encoding="utf-8")
    page = _json.loads(raw[raw.index("=") + 1:].rstrip().rstrip(";"))

    t = taxonomy(source)
    expected = {k: v for k, v in {
        "false_resolution": t["false_resolutions"],
        "false_escalation": t["false_escalations"],
        "misrouted": t["misrouted_deferrals"],
        "unwarranted_deferral": t["unwarranted_adversarial_deferrals"],
    }.items() if v}

    assert page["summary"]["n"] == 160, "every item must have been matched"
    assert page["summary"]["errors"] == expected, (
        "the published summary disagrees with taxonomy() of its source report"
    )

    # and the summary must equal the rows it sits above
    from collections import Counter
    from_rows = Counter(r["error_type"] for r in page["rows"] if r["error_type"])
    assert page["summary"]["errors"] == dict(from_rows)


# --- finding 9's figures must match the committed artefact --------------------------

def test_recall_figures_in_the_docs_match_the_committed_sweep():
    """The README and config.py quote finding 9's recall numbers in prose. Committing the sweep
    caught two errors in them immediately -- an 80% that was computed over a different
    denominator, and a "77% -> 80% -> 84%" that was the pool-60 row transposed into the
    top_k=10 column. This pins them to eval_results/v5_recall.json so it cannot recur.

    Reads only committed files, so it runs in CI even though regenerating the sweep needs the
    index.
    """
    import json as _json

    from src import config

    sweep_path = config.ROOT_DIR / "eval_results" / "v5_recall.json"
    data = _json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep, split = data["sweep"], data["split"]

    readme = (config.ROOT_DIR / "README.md").read_text(encoding="utf-8")
    cfg_src = (config.ROOT_DIR / "src" / "config.py").read_text(encoding="utf-8")

    # the top_k curve at the shipped candidate pool, as quoted in config.py's table
    pool = config.RETRIEVAL_CANDIDATE_POOL
    curve = {k: f"{sweep[f'pool{pool}_topk{k}']['recall']:.0%}" for k in (5, 8, 10, 12)}
    assert curve == {5: "76%", 8: "86%", 10: "88%", 12: "90%"}, curve
    for pct in curve.values():
        assert pct in cfg_src, f"config.py's recall table is missing {pct}"

    # the pool comparison at top_k=10 -- a column, not a row
    column = " -> ".join(f"{sweep[f'pool{p}_topk10']['recall']:.0%}" for p in (20, 40, 60))
    assert column == "88% -> 87% -> 84%", column
    assert column in cfg_src
    assert column.replace(" -> ", " → ") in readme

    # the split that turns a global number into a diagnosis
    assert f"{split['recall_on_always_deferred']:.0%}" == "54%"
    assert f"{split['recall_on_the_rest']:.0%}" == "79%"
    assert split["n_always_deferred"] == 13
    assert "**54%** (7 of 13)" in readme
    assert "**79%** (69 of 87)" in readme

    # and the claim the whole finding rests on
    missed = split["n_always_deferred"] - round(
        split["recall_on_always_deferred"] * split["n_always_deferred"]
    )
    assert missed == 6, "6 of the 13 always-deferred items had no reference article retrieved"
    assert sweep[f"pool{pool}_topk5"]["n_hit"] == 76, (
        "24 of 100 answerable tickets were unanswerable from their context at top_k=5 -- the "
        "floor finding 9 is about"
    )
