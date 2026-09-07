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
import threading
from concurrent.futures import ThreadPoolExecutor

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


def accuracy(results: list[dict], key: str = "decision_correct") -> float | None:
    """Share of results where `key` is true, over items where it was actually measured.

    Rows with key=None are excluded rather than counted as false: a --no-judge run leaves
    groundedness unmeasured, and folding those in would report a hallucination rate that was
    never observed.
    """
    scored = [r for r in results if r.get(key) is not None]
    return sum(1 for r in scored if r[key]) / len(scored) if scored else None


def mean_or_none(values: list) -> float | None:
    """None rather than 0.0 when nothing was scored, so 'not measured' never reads as 'zero'."""
    return statistics.mean(values) if values else None


def complement(value: float | None) -> float | None:
    return None if value is None else 1 - value


def deferral_metrics(golden_results: list[dict], adversarial_results: list[dict]) -> tuple:
    """(false_escalation_rate, deferral_precision).

    Both were hardcoded stubs while the baseline could only ever RESOLVE; with a gate in place
    they are the metrics that actually matter.

    false_escalation_rate — golden tickets (all expect RESOLVE) the gate refused to answer.
      The cost side of deferring: work handed to humans needlessly.
    deferral_precision — of everything the gate deferred, the share that genuinely warranted
      it. Guards against buying adversarial accuracy by deferring indiscriminately, which
      would otherwise look like a win on decision accuracy alone.
    """
    deferred_golden = [r for r in golden_results if r["decision"] != "RESOLVE"]
    false_escalation_rate = len(deferred_golden) / len(golden_results) if golden_results else None

    all_deferrals = deferred_golden + [r for r in adversarial_results if r["decision"] != "RESOLVE"]
    warranted = [r for r in all_deferrals if r.get("expected_decision", "RESOLVE") != "RESOLVE"]
    deferral_precision = len(warranted) / len(all_deferrals) if all_deferrals else None
    return false_escalation_rate, deferral_precision


def describe_coverage(scored: int, available: int) -> str:
    """complete | subsampled | not scored -- per eval set.

    Distinguishing these matters: a run with --adversarial-n 0 scores golden completely and skips
    the other set, which is not the same claim as having sampled both under a quota.
    """
    if scored == 0:
        return "not scored"
    return "complete" if scored >= available else "subsampled"


def describe_mode(golden_state: str, adversarial_state: str, judged: bool) -> str:
    scorers = "live resolver + LLM judge" if judged else "live resolver, decision metrics only"
    if golden_state == adversarial_state == "complete":
        return f"full ({scorers})"
    parts = [f"golden {golden_state}", f"adversarial {adversarial_state}"]
    return f"{', '.join(parts)} ({scorers})"


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


def run_full_report(
    golden_n: int | None = None,
    adversarial_n: int | None = None,
    seed: int = 0,
    resolver_name: str = "naive",
    judge: bool = True,
    workers: int = 1,
    resolver=None,
    review_clarifications: bool = True,
) -> dict:
    """Live resolver + LLM judge. golden_n/adversarial_n cap the item counts for a sampled run.

    Sampling exists because free-tier quotas are per-day and per-model; a capped run produces
    real measurements on a documented subset rather than no measurements at all. Any report
    where a cap was applied is labelled `sampled` so it is never mistaken for a full baseline.

    judge=False skips the LLM-judge scorers. Decision accuracy is a pure comparison against
    the expected label, so the headline gate metric costs one call per item instead of three
    -- which is the difference between measurable and not on a quota-capped tier.

    workers>1 scores items through a thread pool. Every item is an independent
    retrieve-then-call, so nothing is shared but the retriever (read-only after construction)
    and the trace log (locked in src/trace.py). This is what makes the 160-item run 6 minutes
    instead of 45, and iterating on the gate prompt at full coverage rather than on a subsample
    is the whole reason finding 4's lesson -- always measure both sets -- is affordable to obey.

    resolver may be passed in to reuse an already-built index across successive runs; the
    ~40s load and ~600MB are identical every time.
    """
    if resolver is None:
        from src.resolver import GatedResolver, NaiveResolver

        resolver = (
            GatedResolver(review_clarifications=review_clarifications)
            if resolver_name == "gated"
            else NaiveResolver()
        )

    # A full run is hundreds of model calls. The SDK already retries transient errors, but
    # anything that outlives those retries should cost us one item, not the whole report.
    # Failures are collected and disclosed rather than silently dropped, so a run that lost
    # items can never be mistaken for a clean one.
    failures = []

    # Token accounting for the whole run. On a metered key this is the difference between
    # "the eval cost something" and a number you can multiply by a rate, so it is totalled
    # across both the resolver and the judge rather than reported per-item.
    spend = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    spend_lock = threading.Lock()

    def account(usage) -> None:
        if usage is None:
            return
        if isinstance(usage, dict):
            got_in = usage.get("input_tokens") or 0
            got_out = usage.get("output_tokens") or 0
        else:
            got_in = usage.input_tokens or 0
            got_out = usage.output_tokens or 0
        with spend_lock:
            spend["input_tokens"] += got_in
            spend["output_tokens"] += got_out
            spend["calls"] += 1

    def score_golden(item: dict) -> dict:
        run = resolver.resolve(item["question"])
        account(run.get("usage"))
        groundedness = correctness = None
        if judge:
            groundedness, g_usage = score_groundedness(run["answer"], run["retrieved_chunks"])
            account(g_usage)
            correctness, c_usage = score_correctness(
                item["question"], run["answer"], item["reference_answer"]
            )
            account(c_usage)
        return {
            "item_id": item["item_id"],
            "decision": run["decision"],
            "decision_correct": run["decision"] == item["expected_decision"],
            "grounded": groundedness.grounded if judge else None,
            "correctness_score": correctness.score if judge else None,
            "policy_override": run.get("policy_override"),
            # Kept so a failure can be diagnosed from the report alone, without re-deriving it
            # from traces/runs.jsonl by matching on ticket text.
            "kb_coverage": run.get("kb_coverage"),
            "model_decision": run.get("model_decision"),
            "requires_human_authority": run.get("requires_human_authority"),
            "missing_information": run.get("missing_information"),
            "clarify_review": run.get("clarify_review"),
        }

    def score_adversarial(item: dict) -> dict:
        run = resolver.resolve(item["ticket_text"])
        account(run.get("usage"))
        groundedness = None
        if judge:
            groundedness, g_usage = score_groundedness(run["answer"], run["retrieved_chunks"])
            account(g_usage)
        return {
            "item_id": item["item_id"],
            "category": item["category"],
            "expected_decision": item["expected_decision"],
            "decision": run["decision"],
            "decision_correct": run["decision"] == item["expected_decision"],
            "grounded": groundedness.grounded if judge else None,
            "policy_override": run.get("policy_override"),
            "kb_coverage": run.get("kb_coverage"),
            "model_decision": run.get("model_decision"),
            "requires_human_authority": run.get("requires_human_authority"),
            "missing_information": run.get("missing_information"),
            "clarify_review": run.get("clarify_review"),
        }

    progress_lock = threading.Lock()

    def run_set(label: str, items: list[dict], score_fn, describe) -> list[dict]:
        """Score `items`, in input order, with `workers` in flight.

        Results are collected into a pre-sized slot list rather than appended, so the report
        rows come out in dataset order regardless of completion order -- otherwise two runs of
        the same set would produce diffs that are pure scheduling noise.

        Items that raise are retried once, serially, after the concurrent pass. A dropped item
        is not a neutral loss: every rate in the report is a fraction over the items that
        survived, so losing 10 of 60 adversarial items to a network blip silently reweights the
        category mix and makes two runs incomparable. The SDK's own retries cover a rate-limit
        response; they do not cover a connection that drops mid-run, which is what actually
        happened here. Anything still failing after the retry is disclosed in `failures`.
        """
        slots: list[dict | None] = [None] * len(items)
        done = {"n": 0}
        errors: dict[int, str] = {}

        def work(idx_item):
            idx, item = idx_item
            try:
                slots[idx] = score_fn(item)
            except Exception as e:
                errors[idx] = f"{type(e).__name__}: {e}"
                with progress_lock:
                    done["n"] += 1
                    print(f"  [{done['n']}/{len(items)}] {item['item_id']} FAILED: {type(e).__name__}: {e}")
                return
            errors.pop(idx, None)
            with progress_lock:
                done["n"] += 1
                print(f"  [{done['n']}/{len(items)}] {describe(item, slots[idx])}")

        if workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(work, enumerate(items)))
        else:
            for pair in enumerate(items):
                work(pair)

        if errors:
            print(f"  retrying {len(errors)} failed {label} item(s) serially")
            for idx in sorted(errors):
                item = items[idx]
                try:
                    slots[idx] = score_fn(item)
                except Exception as e:
                    errors[idx] = f"{type(e).__name__}: {e} (after retry)"
                    print(f"    {item['item_id']} FAILED AGAIN: {type(e).__name__}")
                    continue
                errors.pop(idx, None)
                print(f"    recovered {describe(item, slots[idx])}")

        for idx, err in sorted(errors.items()):
            failures.append({"set": label, "item_id": items[idx]["item_id"], "error": err})
        return [s for s in slots if s is not None]

    all_golden = load_golden()
    # Golden items are all expected_decision=RESOLVE, so there is no category to stratify on;
    # a seeded shuffle is the honest way to pick a subset.
    golden = all_golden
    if golden_n is not None and golden_n < len(all_golden):
        golden = list(all_golden)
        random.Random(seed).shuffle(golden)
        golden = golden[:golden_n]

    def describe_golden(item, row):
        detail = (
            f"grounded={row['grounded']} correctness={row['correctness_score']}" if judge else ""
        )
        return f"{item['item_id']} {row['decision']:8s} cov={row['kb_coverage']} {detail}"

    golden_results = run_set("golden", golden, score_golden, describe_golden)

    all_adversarial = load_adversarial()
    adversarial = all_adversarial
    if adversarial_n is not None and adversarial_n < len(all_adversarial):
        adversarial = stratified_sample(all_adversarial, adversarial_n, "category", seed)

    def describe_adversarial(item, row):
        mark = "OK " if row["decision_correct"] else "MISS"
        detail = f" grounded={row['grounded']}" if judge else ""
        return (
            f"{item['item_id']} ({item['category']}) {mark} "
            f"got={row['decision']} want={item['expected_decision']}{detail}"
        )

    adversarial_results = run_set(
        "adversarial", adversarial, score_adversarial, describe_adversarial
    )

    false_escalation_rate, deferral_precision = deferral_metrics(golden_results, adversarial_results)

    # Completeness is per set, and skipping a set is not the same as subsampling it. Running with
    # --adversarial-n 0 scores golden completely; calling that report "sampled ... forced by
    # free-tier quota" told a reader checking the headline numbers that a 100/100 run was partial.
    golden_state = describe_coverage(len(golden_results), len(all_golden))
    adversarial_state = describe_coverage(len(adversarial_results), len(all_adversarial))
    subsampled = "subsampled" in (golden_state, adversarial_state)
    sampled = subsampled  # kept for report back-compat; true only when a set was actually cut down
    golden_groundedness = accuracy(golden_results, "grounded")
    adversarial_groundedness = accuracy(adversarial_results, "grounded")

    report = {
        "mode": describe_mode(golden_state, adversarial_state, judge),
        "sampled": sampled,
        "sample_seed": seed if sampled else None,
        "resolver": resolver_name,
        "judged": judge,
        "llm_provider": config.LLM_PROVIDER,
        "resolver_model": config.active_model(),
        "judge_model": config.judge_model() if judge else None,
        "n_golden": len(golden_results),
        "n_adversarial": len(adversarial_results),
        "n_golden_attempted": len(golden),
        "n_adversarial_attempted": len(adversarial),
        "n_golden_available": len(all_golden),
        "n_adversarial_available": len(all_adversarial),
        "n_failed": len(failures),
        "workers": workers,
        # Read off the retriever that actually served this run, not off config at report-assembly
        # time. Two frontier configurations here differ only by top_k, and for a while nothing in
        # a per-run report distinguished them -- so the only thing separating them was a label a
        # human typed. Provenance has to come from the object that did the work.
        "retrieval_top_k": getattr(getattr(resolver, "retriever", None), "top_k", None),
        "review_clarifications": review_clarifications if resolver_name == "gated" else None,
        "failures": failures,
        "token_spend": {
            **spend,
            "total_tokens": spend["input_tokens"] + spend["output_tokens"],
        },
        "golden_decision_accuracy": accuracy(golden_results),
        "golden_groundedness_rate": golden_groundedness,
        "golden_hallucination_rate": complement(golden_groundedness),
        "golden_mean_correctness": mean_or_none(
            [r["correctness_score"] for r in golden_results if r["correctness_score"] is not None]
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
        "false_escalation_rate": false_escalation_rate,
        "deferral_precision": deferral_precision,
        "n_policy_overrides": sum(
            1 for r in golden_results + adversarial_results if r.get("policy_override")
        ),
        "golden_results": golden_results,
        "adversarial_results": adversarial_results,
    }
    report["golden_coverage"] = golden_state
    report["adversarial_coverage"] = adversarial_state
    if subsampled:
        report["sampling_note"] = (
            f"Scored {len(golden_results)}/{len(all_golden)} golden and "
            f"{len(adversarial_results)}/{len(all_adversarial)} adversarial items. Rates over a "
            "subsampled set carry wide confidence intervals at this n -- treat as a smoke-grade "
            "signal, not a baseline."
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
    parser.add_argument("--resolver", choices=["naive", "gated"], default="naive", help="Which resolver to evaluate")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-judge scorers; decision metrics only (1 call/item)")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Score this many items concurrently (default 1). 8 turns the full 160-item run "
             "from ~45min into ~6min; the per-item work is independent.",
    )
    parser.add_argument("--out", default=None, help="Report filename under eval_results/ (default: auto)")
    parser.add_argument(
        "--no-review", action="store_true",
        help="Gated resolver only, without the clarification reviewer -- isolates the gate's own "
             "contribution from the second-stage call.",
    )
    args = parser.parse_args()

    if args.offline or not config.api_key_present():
        report = run_offline_report()
        save_report(report, args.out or "baseline_offline.json")
    else:
        report = run_full_report(
            args.golden_n, args.adversarial_n, args.seed, args.resolver, not args.no_judge,
            workers=args.workers, review_clarifications=not args.no_review,
        )
        stem = "gated" if args.resolver == "gated" else "baseline"
        suffix = "sampled" if report["sampled"] else "full"
        save_report(report, args.out or f"{stem}_{suffix}.json")
    print_summary(report)
