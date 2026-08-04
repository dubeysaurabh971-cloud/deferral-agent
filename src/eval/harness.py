"""Eval harness. Two modes:

  offline — decision-accuracy metrics only, no API calls. Works because the Week 1
            baseline's decision policy is a pure function (always "RESOLVE"), so we can
            score it against the golden + adversarial expected labels without spending
            a single token.
  full    — runs the live resolver + LLM-judge scorers (groundedness, correctness) on
            every item. Requires an API key for the configured LLM_PROVIDER.

Run: `python -m src.eval.harness` (auto-picks full if a key is present, else offline).
"""
import json
import statistics

from src import config
from src.eval.scorers import score_correctness, score_groundedness

RESULTS_DIR = config.ROOT_DIR / "eval_results"


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_golden() -> list[dict]:
    return load_jsonl(config.DATA_DIR / "golden" / "golden_set.jsonl")


def load_adversarial() -> list[dict]:
    return load_jsonl(config.DATA_DIR / "adversarial" / "adversarial_set.jsonl")


def baseline_decision_policy(_item: dict) -> str:
    """The Week 1 naive resolver never gates: it always resolves."""
    return "RESOLVE"


def evaluate_decisions(items: list[dict], decision_fn, category_key: str | None = None) -> dict:
    total = len(items)
    correct = sum(1 for it in items if decision_fn(it) == it["expected_decision"])
    result = {"n": total, "accuracy": correct / total if total else None}
    if category_key:
        by_cat: dict[str, dict] = {}
        for it in items:
            cat = it.get(category_key, "unknown")
            bucket = by_cat.setdefault(cat, {"n": 0, "correct": 0})
            bucket["n"] += 1
            if decision_fn(it) == it["expected_decision"]:
                bucket["correct"] += 1
        result["by_category"] = {
            c: {"n": v["n"], "accuracy": v["correct"] / v["n"]} for c, v in by_cat.items()
        }
    return result


def run_offline_report() -> dict:
    golden = load_golden()
    adversarial = load_adversarial()

    report = {
        "mode": "offline (decision-policy only, no LLM calls)",
        "golden_decision_accuracy": evaluate_decisions(golden, baseline_decision_policy),
        "adversarial_decision_accuracy": evaluate_decisions(
            adversarial, baseline_decision_policy, category_key="category"
        ),
        "false_escalation_rate": 0.0,
        "deferral_precision": None,
        "note": (
            "Baseline decision policy is hardcoded to RESOLVE (Week 1 has no gate), so these "
            "numbers are computable without calling the model. Deferral precision is undefined "
            "(0/0) because the baseline never escalates. Groundedness, correctness, and "
            "resolution-rate-by-correctness require the live resolver + LLM judge — pending "
            "ANTHROPIC_API_KEY. False escalation rate is 0% trivially, not as a virtue: a "
            "resolver that never escalates cannot falsely escalate."
        ),
    }
    return report


def run_full_report() -> dict:
    from src.resolver import NaiveResolver

    resolver = NaiveResolver()

    golden = load_golden()
    golden_results = []
    for item in golden:
        run = resolver.resolve(item["question"])
        groundedness = score_groundedness(run["answer"], run["retrieved_chunks"])
        correctness = score_correctness(item["question"], run["answer"], item["reference_answer"])
        golden_results.append({
            "item_id": item["item_id"],
            "decision_correct": run["decision"] == item["expected_decision"],
            "grounded": groundedness.grounded,
            "correctness_score": correctness.score,
        })

    adversarial = load_adversarial()
    adversarial_results = []
    for item in adversarial:
        run = resolver.resolve(item["ticket_text"])
        groundedness = score_groundedness(run["answer"], run["retrieved_chunks"])
        adversarial_results.append({
            "item_id": item["item_id"],
            "category": item["category"],
            "decision_correct": run["decision"] == item["expected_decision"],
            "grounded": groundedness.grounded,
        })

    def accuracy(results, key="decision_correct"):
        return sum(1 for r in results if r[key]) / len(results) if results else None

    report = {
        "mode": "full (live resolver + LLM judge)",
        "llm_provider": config.LLM_PROVIDER,
        "resolver_model": config.active_model(),
        "n_golden": len(golden_results),
        "n_adversarial": len(adversarial_results),
        "golden_decision_accuracy": accuracy(golden_results),
        "golden_groundedness_rate": accuracy(golden_results, "grounded"),
        "golden_hallucination_rate": 1 - accuracy(golden_results, "grounded"),
        "golden_mean_correctness": statistics.mean(r["correctness_score"] for r in golden_results),
        "adversarial_decision_accuracy": accuracy(adversarial_results),
        "adversarial_hallucination_rate": 1 - accuracy(adversarial_results, "grounded"),
        "false_escalation_rate": 0.0,
        "deferral_precision": None,
        "golden_results": golden_results,
        "adversarial_results": adversarial_results,
    }
    return report


def print_summary(report: dict) -> None:
    print(f"\n=== Eval report — {report['mode']} ===\n")
    for key, value in report.items():
        if key in ("golden_results", "adversarial_results"):
            continue
        print(f"{key}: {value}")


def save_report(report: dict, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    if config.api_key_present():
        report = run_full_report()
        save_report(report, "baseline_full.json")
    else:
        report = run_offline_report()
        save_report(report, "baseline_offline.json")
    print_summary(report)
