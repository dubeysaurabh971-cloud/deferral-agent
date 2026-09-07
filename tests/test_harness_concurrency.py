"""The parallel scoring path in run_full_report, with a stub resolver instead of a model.

Worth its own file because the concurrency was added to make full-coverage runs affordable,
and two of its properties are ones a silent bug would corrupt rather than crash:

  - row ORDER must follow the dataset, not completion order, or two reports of the same set
    diff on scheduling noise and no comparison between prompt versions means anything;
  - a failed item must be RETRIED, and if it still fails, disclosed. Every rate in the report
    is a fraction over the items that survived, so quietly losing items reweights the
    adversarial category mix -- which is exactly what a mid-run connection drop did once,
    turning a 60-item run into 50 and an invalid comparison into a plausible-looking one.

No API key: the resolver is a stub, and judge=False keeps the LLM scorers out.
"""
import threading

import pytest

from src.eval import harness


class StubResolver:
    """Returns a canned decision per ticket text. Optionally fails the first N attempts on
    chosen items, to exercise the retry pass."""

    def __init__(self, decision="RESOLVE", fail_texts=(), fail_times=1, delay=0.0):
        self.decision = decision
        self._fail_budget = {t: fail_times for t in fail_texts}
        self.delay = delay
        self.calls = []
        self._lock = threading.Lock()

    def resolve(self, ticket_text: str) -> dict:
        if self.delay:
            import time

            time.sleep(self.delay)
        with self._lock:
            self.calls.append(ticket_text)
            budget = self._fail_budget.get(ticket_text, 0)
            if budget > 0:
                self._fail_budget[ticket_text] = budget - 1
                raise ConnectionError("simulated connection drop")
        return {
            "decision": self.decision,
            "answer": "a",
            "retrieved_chunks": [],
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "kb_coverage": "full",
            "model_decision": self.decision,
            "requires_human_authority": False,
            "missing_information": None,
            "policy_override": None,
            "clarify_review": None,
        }


# The full sets, not a subsample: --golden-n shuffles and --adversarial-n stratifies, so a
# capped run has no predictable item order or membership to assert against. With a stub
# resolver the full 160 costs milliseconds.
N_GOLDEN = len(harness.load_golden())
N_ADVERSARIAL = len(harness.load_adversarial())


def run(resolver, workers, **kw):
    return harness.run_full_report(
        resolver_name="gated", judge=False, workers=workers, resolver=resolver, **kw
    )


@pytest.mark.parametrize("workers", [1, 4])
def test_rows_come_out_in_dataset_order(workers):
    """Concurrent completion must not reorder the report."""
    report = run(StubResolver(delay=0.005), workers)
    assert [r["item_id"] for r in report["golden_results"]] == [
        it["item_id"] for it in harness.load_golden()
    ]
    assert [r["item_id"] for r in report["adversarial_results"]] == [
        it["item_id"] for it in harness.load_adversarial()
    ]


def test_parallel_and_serial_agree_on_every_metric():
    """workers>1 is a scheduling change, not a scoring change."""
    serial = run(StubResolver(), 1)
    parallel = run(StubResolver(), 6)
    for key in (
        "golden_decision_accuracy",
        "adversarial_decision_accuracy",
        "false_escalation_rate",
        "deferral_precision",
        "n_golden",
        "n_adversarial",
    ):
        assert serial[key] == parallel[key], key
    assert serial["token_spend"] == parallel["token_spend"]


def test_token_accounting_is_not_lost_under_concurrency():
    """The spend counter is incremented from every worker thread; a non-atomic
    read-modify-write would undercount and make a metered run unauditable."""
    report = run(StubResolver(), 8)
    n = N_GOLDEN + N_ADVERSARIAL
    assert report["token_spend"]["calls"] == n
    assert report["token_spend"]["input_tokens"] == n * 3
    assert report["token_spend"]["output_tokens"] == n * 2


def test_transient_failure_is_retried_and_recovered():
    doomed = harness.load_golden()[2]["question"]
    resolver = StubResolver(fail_texts=[doomed], fail_times=1)
    report = run(resolver, 4, adversarial_n=0)

    assert report["n_failed"] == 0, "a once-failing item should be recovered by the retry pass"
    assert report["n_golden"] == N_GOLDEN
    assert resolver.calls.count(doomed) == 2, "exactly one retry, not a retry storm"


def test_persistent_failure_is_disclosed_not_dropped_silently():
    doomed = harness.load_golden()[1]["question"]
    resolver = StubResolver(fail_texts=[doomed], fail_times=99)
    report = run(resolver, 4, adversarial_n=0)

    assert report["n_failed"] == 1
    assert report["n_golden"] == N_GOLDEN - 1
    assert report["n_golden_attempted"] == N_GOLDEN, "must show it tried all and scored one fewer"
    assert report["failures"][0]["error"].endswith("(after retry)")


def test_report_records_the_configuration_it_measured():
    """A report that does not say whether the reviewer was on cannot be compared to one that
    does -- the two configurations differ by ~1 golden point and a call per CLARIFY."""
    report = harness.run_full_report(
        adversarial_n=0, resolver_name="gated", judge=False,
        workers=2, resolver=StubResolver(), review_clarifications=False,
    )
    assert report["review_clarifications"] is False
    assert report["workers"] == 2
