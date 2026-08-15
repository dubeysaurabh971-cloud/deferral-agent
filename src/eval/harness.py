"""Eval harness. Two modes:

  offline — decision-accuracy metrics only, no API calls. Works because the Week 1
            baseline's decision policy is a pure function (always "RESOLVE"), so we can
            score it against the golden + adversarial expected labels without spending
            a single token.
  full    — runs the live resolver + LLM-judge scorers (groundedness, correctness) on
            every item. Requires an API key for the configured LLM_PROVIDER.

Run: `python -m src.eval.harness` (auto-picks full if a key is present, else offline).
"""
import argparse
import json
import random
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


def stratified_sample(items: list[dict], n: int, key: str, seed: int = 0) -> list[dict]:
    """Take n items spread as evenly as possible across the values of `items[key]`.

    Seeded so a sampled run is reproducible: rerunning with the same n and seed scores the
    same items, which is the only way two sampled reports are comparable to each other.
    """
    if n >= len(items):
        return items
    buckets: dict[str, list[dict]] = {}
    for it in items:
        buckets.setdefault(it.get(key, "unknown"), []).append(it)
    for bucket in buckets.values():
        random.Random(seed).shuffle(bucket)

    picked: list[dict] = []
    round_robin = sorted(buckets)
    while len(picked) < n:
        progressed = False
        for cat in round_robin:
            if buckets[cat] and len(picked) < n:
                picked.append(buckets[cat].pop())
                progressed = True
        if not progressed:
            break
    return picked


def run_full_report(golden_n: int | None = None, adversarial_n: int | None = None, seed: int = 0) -> dict:
    """Live resolver + LLM judge. golden_n/adversarial_n cap the item counts for a sampled run.

    Sampling exists because free-tier quotas are per-day and per-model; a capped run produces
    real measurements on a documented subset rather than no measurements at all. Any report
    where a cap was applied is labelled `sampled` so it is never mistaken for a full baseline.
    """
    from src.resolver import NaiveResolver

    resolver = NaiveResolver()

    # A full run is hundreds of serial calls over ~1.5h. The SDK already retries transient
    # errors, but anything that outlives those retries should cost us one item, not the whole
    # report. Failures are collected and disclosed rather than silently dropped, so a run
    # that lost items can never be mistaken for a clean one.
    failures = []

    all_golden = load_golden()
    # Golden items are all expected_decision=RESOLVE, so there is no category to stratify on;
    # a seeded shuffle is the honest way to pick a subset.
    golden = all_golden
    if golden_n is not None and golden_n < len(all_golden):
        golden = list(all_golden)
        random.Random(seed).shuffle(golden)
        golden = golden[:golden_n]

    golden_results = []
    for i, item in enumerate(golden):
        try:
            run = resolver.resolve(item["question"])
            groundedness = score_groundedness(run["answer"], run["retrieved_chunks"])
            correctness = score_correctness(item["question"], run["answer"], item["reference_answer"])
        except Exception as e:
            failures.append({"set": "golden", "item_id": item["item_id"], "error": f"{type(e).__name__}: {e}"})
            print(f"  [{i + 1}/{len(golden)}] {item['item_id']} FAILED: {type(e).__name__}")
            continue
        golden_results.append({
            "item_id": item["item_id"],
            "decision_correct": run["decision"] == item["expected_decision"],
            "grounded": groundedness.grounded,
            "correctness_score": correctness.score,
        })
        print(f"  [{i + 1}/{len(golden)}] {item['item_id']} grounded={groundedness.grounded} correctness={correctness.score}")

    all_adversarial = load_adversarial()
    adversarial = all_adversarial
    if adversarial_n is not None and adversarial_n < len(all_adversarial):
        adversarial = stratified_sample(all_adversarial, adversarial_n, "category", seed)

    adversarial_results = []
    for i, item in enumerate(adversarial):
        try:
            run = resolver.resolve(item["ticket_text"])
            groundedness = score_groundedness(run["answer"], run["retrieved_chunks"])
        except Exception as e:
            failures.append({"set": "adversarial", "item_id": item["item_id"], "error": f"{type(e).__name__}: {e}"})
            print(f"  [{i + 1}/{len(adversarial)}] {item['item_id']} FAILED: {type(e).__name__}")
            continue
        adversarial_results.append({
            "item_id": item["item_id"],
            "category": item["category"],
            "decision_correct": run["decision"] == item["expected_decision"],
            "grounded": groundedness.grounded,
        })
        print(f"  [{i + 1}/{len(adversarial)}] {item['item_id']} ({item['category']}) grounded={groundedness.grounded}")

    def accuracy(results, key="decision_correct"):
        return sum(1 for r in results if r[key]) / len(results) if results else None

    def complement(value):
        return None if value is None else 1 - value

    sampled = len(golden) < len(all_golden) or len(adversarial) < len(all_adversarial)
    golden_groundedness = accuracy(golden_results, "grounded")
    adversarial_groundedness = accuracy(adversarial_results, "grounded")

    report = {
        "mode": (
            "sampled (live resolver + LLM judge, subset of the eval sets)"
            if sampled
            else "full (live resolver + LLM judge)"
        ),
        "sampled": sampled,
        "sample_seed": seed if sampled else None,
        "llm_provider": config.LLM_PROVIDER,
        "resolver_model": config.active_model(),
        "judge_model": config.judge_model(),
        "n_golden": len(golden_results),
        "n_adversarial": len(adversarial_results),
        "n_golden_attempted": len(golden),
        "n_adversarial_attempted": len(adversarial),
        "n_golden_available": len(all_golden),
        "n_adversarial_available": len(all_adversarial),
        "n_failed": len(failures),
        "failures": failures,
        "golden_decision_accuracy": accuracy(golden_results),
        "golden_groundedness_rate": golden_groundedness,
        "golden_hallucination_rate": complement(golden_groundedness),
        "golden_mean_correctness": (
            statistics.mean(r["correctness_score"] for r in golden_results) if golden_results else None
        ),
        "adversarial_decision_accuracy": accuracy(adversarial_results),
        "adversarial_hallucination_rate": complement(adversarial_groundedness),
        "adversarial_by_category": {
            cat: {
                "n": len(rows),
                "decision_accuracy": accuracy(rows),
                "groundedness_rate": accuracy(rows, "grounded"),
            }
            for cat, rows in sorted(
                {
                    r["category"]: [x for x in adversarial_results if x["category"] == r["category"]]
                    for r in adversarial_results
                }.items()
            )
        },
        "false_escalation_rate": 0.0,
        "deferral_precision": None,
        "golden_results": golden_results,
        "adversarial_results": adversarial_results,
    }
    if sampled:
        report["sampling_note"] = (
            f"Scored {len(golden_results)}/{len(all_golden)} golden and "
            f"{len(adversarial_results)}/{len(all_adversarial)} adversarial items. Rates are computed "
            "over the sampled items only and carry wide confidence intervals at this n -- treat as a "
            "smoke-grade signal, not the baseline. Sampling was forced by free-tier daily quota."
        )
    return report


def print_summary(report: dict) -> None:
    print(f"\n=== Eval report — {report['mode']} ===\n")
    for key, value in report.items():
        if key in ("golden_results", "adversarial_results", "failures"):
            continue
        print(f"{key}: {value}")


def save_report(report: dict, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="deferral-agent eval harness")
    parser.add_argument("--golden-n", type=int, default=None, help="Score only N golden items")
    parser.add_argument("--adversarial-n", type=int, default=None, help="Score only N adversarial items (stratified by category)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed, for reproducible subsets")
    parser.add_argument("--offline", action="store_true", help="Force the offline report even if a key is present")
    args = parser.parse_args()

    if args.offline or not config.api_key_present():
        report = run_offline_report()
        save_report(report, "baseline_offline.json")
    else:
        report = run_full_report(args.golden_n, args.adversarial_n, args.seed)
        save_report(report, "baseline_sampled.json" if report["sampled"] else "baseline_full.json")
    print_summary(report)
