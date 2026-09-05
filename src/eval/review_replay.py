"""Measure the clarification reviewer against cached v3 decisions.

The reviewer only ever fires on CLARIFY and can only turn CLARIFY into RESOLVE, so its effect
on a completed run is computable without re-running the gate: replay it over the CLARIFY
decisions already in traces/runs.jsonl and recompute the metrics. That is 72 calls instead of
232, and it isolates the reviewer's contribution from any prompt drift in the base gate.

Run: `python -m src.eval.review_replay`
"""
import collections
import json

from src import config, gate, review
from src.eval.harness import load_adversarial, load_golden
from src.retrieval import HybridRetriever

OUT = config.ROOT_DIR / "eval_results" / "clarify_review_replay.json"


def v3_traces() -> dict[str, dict]:
    """Second-to-last gated trace per ticket == the v3 run.

    Traces are append-only and in run order. Golden was scored by v3 then v4; adversarial by
    v1, (v2 for a 20-item subset), v3, then v4. In every case the v3 attempt is the one before
    last. Verified against both v3 reports: 160/160 decisions match.
    """
    recs = [json.loads(l) for l in open(config.TRACES_DIR / "runs.jsonl", encoding="utf-8") if l.strip()]
    by_text: dict[str, list] = collections.OrderedDict()
    for r in recs:
        if r.get("resolver") == "gated":
            by_text.setdefault(r["ticket_text"], []).append(r)
    return {t: occ[-2] for t, occ in by_text.items() if len(occ) >= 2}


def main() -> None:
    traces = v3_traces()
    retriever = HybridRetriever()

    items = [("golden", it["item_id"], it["question"], "RESOLVE", None) for it in load_golden()]
    items += [
        ("adversarial", it["item_id"], it["ticket_text"], it["expected_decision"], it["category"])
        for it in load_adversarial()
    ]

    rows, spend = [], {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    to_review = [x for x in items if traces.get(x[2], {}).get("decision") == "CLARIFY"]
    print(f"Reviewing {len(to_review)} cached CLARIFY decisions ({len(items) - len(to_review)} untouched).\n")

    for i, (dataset, item_id, text, expected, category) in enumerate(to_review):
        chunks = retriever.retrieve(text)
        question = traces[text].get("answer") or ""
        try:
            verdict, usage = review.review_clarification(text, chunks, question)
        except Exception as e:
            print(f"  [{i+1}/{len(to_review)}] {item_id} FAILED {type(e).__name__}")
            continue
        spend["input_tokens"] += usage.input_tokens
        spend["output_tokens"] += usage.output_tokens
        spend["calls"] += 1
        new_decision, _ = review.apply_review(verdict)
        flipped = new_decision != "CLARIFY"
        rows.append({
            "dataset": dataset, "item_id": item_id, "category": category,
            "expected": expected, "flipped_to_resolve": flipped,
            "necessary": verdict.question_is_necessary,
            "competing": verdict.competing_procedures,
            "correct_after": new_decision == expected,
            "correct_before": "CLARIFY" == expected,
        })
        good = "GOOD" if (new_decision == expected) else ("was-right" if expected == "CLARIFY" and not flipped else "")
        print(f"  [{i+1}/{len(to_review)}] {item_id:<5} {dataset:<11} want={expected:<8} "
              f"{'FLIP->RESOLVE' if flipped else 'keep CLARIFY '} {good}")

    # Recompute both sets with the reviewer applied
    flips = {r["item_id"] for r in rows if r["flipped_to_resolve"]}
    final = {}
    for dataset, item_id, text, expected, category in items:
        base = traces.get(text, {}).get("decision")
        final[item_id] = ("RESOLVE" if item_id in flips else base, expected, dataset, category)

    def acc(ds):
        sub = [v for v in final.values() if v[2] == ds and v[0]]
        return sum(1 for d, e, _, _ in sub if d == e) / len(sub), len(sub)

    g_acc, g_n = acc("golden")
    a_acc, a_n = acc("adversarial")
    fe = sum(1 for d, e, ds, _ in final.values() if ds == "golden" and d and d != "RESOLVE")
    fr = sum(1 for d, e, ds, _ in final.values() if ds == "adversarial" and d == "RESOLVE" and e != "RESOLVE")

    by_cat = {}
    for d, e, ds, cat in final.values():
        if ds != "adversarial" or not d:
            continue
        b = by_cat.setdefault(cat, [0, 0]); b[1] += 1; b[0] += (d == e)

    report = {
        "mechanism": "clarification reviewer replayed over cached v3 CLARIFY decisions",
        "n_reviewed": len(rows),
        "n_flipped_to_resolve": len(flips),
        "golden_decision_accuracy": g_acc,
        "adversarial_decision_accuracy": a_acc,
        "false_escalation_rate": fe / g_n,
        "false_resolutions_adversarial": fr,
        "adversarial_by_category": {c: {"n": v[1], "decision_accuracy": v[0] / v[1]} for c, v in sorted(by_cat.items())},
        "token_spend": {**spend, "total_tokens": spend["input_tokens"] + spend["output_tokens"]},
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== v3 + clarification reviewer ===")
    for k, v in report.items():
        if k != "rows":
            print(f"{k}: {v}")
    print("\nv3 alone was: golden 49.0%, adversarial 71.7%, false_escalation 51%, false_resolutions 4")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
