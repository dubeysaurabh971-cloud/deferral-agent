"""Report-assembly maths. Pure functions over result rows, so no API key or quota needed."""
from src.eval.harness import accuracy, complement, deferral_metrics, mean_or_none


def golden_row(decision="RESOLVE", grounded=None, correctness=None):
    return {
        "item_id": "G001",
        "decision": decision,
        "decision_correct": decision == "RESOLVE",
        "grounded": grounded,
        "correctness_score": correctness,
    }


def adversarial_row(decision, expected, grounded=None):
    return {
        "item_id": "A001",
        "category": "out_of_kb",
        "expected_decision": expected,
        "decision": decision,
        "decision_correct": decision == expected,
        "grounded": grounded,
    }


def test_unmeasured_rows_are_excluded_not_counted_false():
    """A --no-judge run leaves grounded=None. Those rows must not be folded in as failures,
    which would report a hallucination rate that was never observed."""
    rows = [golden_row(grounded=None), golden_row(grounded=None)]
    assert accuracy(rows, "grounded") is None

    mixed = [golden_row(grounded=True), golden_row(grounded=None)]
    assert accuracy(mixed, "grounded") == 1.0


def test_mean_or_none_on_unscored_run():
    """Regression: statistics.mean() raised TypeError on a --no-judge run because every
    correctness_score was None."""
    assert mean_or_none([]) is None
    assert mean_or_none([4, 5, 5]) == 14 / 3


def test_complement_passes_none_through():
    assert complement(None) is None
    assert complement(1.0) == 0.0


def test_false_escalation_counts_only_golden():
    """Golden all expect RESOLVE, so any non-RESOLVE there is a false escalation."""
    golden = [golden_row("RESOLVE"), golden_row("RESOLVE"), golden_row("ESCALATE"), golden_row("CLARIFY")]
    fer, _ = deferral_metrics(golden, [])
    assert fer == 0.5


def test_deferral_precision_spans_both_sets():
    golden = [golden_row("RESOLVE"), golden_row("ESCALATE")]          # 1 unwarranted deferral
    adversarial = [
        adversarial_row("ESCALATE", "ESCALATE"),                       # warranted
        adversarial_row("CLARIFY", "CLARIFY"),                         # warranted
        adversarial_row("RESOLVE", "ESCALATE"),                        # not a deferral at all
    ]
    _, precision = deferral_metrics(golden, adversarial)
    assert precision == 2 / 3


def test_indiscriminate_deferral_is_penalised():
    """Escalating everything scores well on adversarial accuracy alone; deferral_precision and
    false_escalation_rate are what expose it."""
    golden = [golden_row("ESCALATE") for _ in range(5)]
    adversarial = [adversarial_row("ESCALATE", "ESCALATE") for _ in range(5)]
    fer, precision = deferral_metrics(golden, adversarial)
    assert fer == 1.0
    assert precision == 0.5


def test_baseline_shape_never_defers():
    """The Week 1 baseline: no deferrals anywhere, so precision is undefined (0/0) rather
    than 0.0, and the false escalation rate is trivially zero."""
    golden = [golden_row("RESOLVE") for _ in range(3)]
    adversarial = [adversarial_row("RESOLVE", "ESCALATE") for _ in range(3)]
    fer, precision = deferral_metrics(golden, adversarial)
    assert fer == 0.0
    assert precision is None
