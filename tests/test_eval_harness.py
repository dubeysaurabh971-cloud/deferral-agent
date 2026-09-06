"""Smoke tests for the eval datasets and offline scoring logic.

No API key and no network. The golden set is read from the committed
data/golden/golden_set.jsonl rather than rebuilt via build_golden_set(), which calls
load_dataset() against the HuggingFace Hub. Rebuilding it here coupled CI to an external
dataset host: an outage or a rename would have broken the badge for reasons unrelated to this
code, and the suite only passed because GitHub runners happen to have internet.

The builder still gets covered, by test_golden_builder_matches_committed_file, which is marked
`network` and skipped unless RUN_NETWORK_TESTS=1.
"""
import json
import os

import pytest

from src import config
from src.adversarial import build_adversarial_set
from src.eval.harness import baseline_decision_policy, evaluate_decisions, load_golden
from src.eval.scorers import decision_accuracy
from src.golden import GOLDEN_SIZE, build_golden_set

requires_network = pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="hits the HuggingFace Hub; set RUN_NETWORK_TESTS=1 to run",
)


def test_golden_set_schema():
    golden = load_golden()
    assert len(golden) == GOLDEN_SIZE
    for item in golden:
        assert item["expected_decision"] == "RESOLVE"
        assert item["question"]
        assert item["reference_answer"]
        assert item["article_ids"]


def test_adversarial_set_schema_and_category_counts():
    adversarial = build_adversarial_set()
    assert len(adversarial) == 60

    counts = {}
    for item in adversarial:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
        assert item["ticket_text"]
        assert item["expected_decision"] in {"RESOLVE", "CLARIFY", "ESCALATE"}

    assert counts == {"out_of_kb": 20, "ambiguous": 20, "injection": 10, "near_miss": 10}

    ids = [item["item_id"] for item in adversarial]
    assert len(ids) == len(set(ids)), "item_ids must be unique"


def test_adversarial_out_of_kb_and_near_miss_expect_escalate():
    adversarial = build_adversarial_set()
    for item in adversarial:
        if item["category"] in {"out_of_kb", "near_miss"}:
            assert item["expected_decision"] == "ESCALATE"
        if item["category"] == "ambiguous":
            assert item["expected_decision"] == "CLARIFY"


def test_decision_accuracy_pure_function():
    assert decision_accuracy("RESOLVE", "RESOLVE") is True
    assert decision_accuracy("RESOLVE", "ESCALATE") is False


def test_baseline_offline_decision_metrics_match_hand_calculation():
    golden = load_golden()
    golden_metrics = evaluate_decisions(golden, baseline_decision_policy)
    # Baseline always predicts RESOLVE; every golden item expects RESOLVE.
    assert golden_metrics["accuracy"] == 1.0

    adversarial = build_adversarial_set()
    adversarial_metrics = evaluate_decisions(adversarial, baseline_decision_policy, category_key="category")
    # Baseline never clarifies or escalates, so it's wrong on every out_of_kb/ambiguous/near_miss item.
    assert adversarial_metrics["by_category"]["out_of_kb"]["accuracy"] == 0.0
    assert adversarial_metrics["by_category"]["ambiguous"]["accuracy"] == 0.0
    assert adversarial_metrics["by_category"]["near_miss"]["accuracy"] == 0.0
    # Injection items where the legitimate underlying request is itself answerable (RESOLVE)
    # are the only adversarial cases the naive baseline gets right by accident.
    injection_expected_resolve = sum(
        1 for item in adversarial if item["category"] == "injection" and item["expected_decision"] == "RESOLVE"
    )
    assert adversarial_metrics["by_category"]["injection"]["accuracy"] == (
        injection_expected_resolve / 10
    )


@requires_network
def test_golden_builder_matches_committed_file():
    """The builder is still the source of truth; this pins the committed file to it.

    Skipped by default so the rest of the suite stays genuinely offline. Run it when the
    upstream dataset or the build logic changes:  RUN_NETWORK_TESTS=1 pytest -k builder
    """
    built = build_golden_set()
    committed = load_golden()
    assert len(built) == len(committed) == GOLDEN_SIZE
    assert [i["item_id"] for i in built] == [i["item_id"] for i in committed]
    assert [i["question"] for i in built] == [i["question"] for i in committed]
