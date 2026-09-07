"""Measure the clarification reviewer against cached v3 decisions.

The reviewer only ever fires on CLARIFY and can only turn CLARIFY into RESOLVE, so its effect on
a completed run is computable without re-running the gate: replay it over the CLARIFY decisions
already in traces/runs.jsonl and recompute the metrics. That is 72 calls instead of 232, and it
isolates the reviewer's contribution from any prompt drift in the base gate.

Two configurations are scored from the same verdicts, because the difference between them is the
project's main result about this mechanism:

  ungoverned  every flip the reviewer proposes is taken.
  governed    a flip is taken only where the gate's own policy would have permitted a RESOLVE
              (apply_policy re-check), and never on a ticket with a detected injection. This is
              what GatedResolver ships.

Ungoverned scores *better* on raw golden accuracy and is the wrong configuration above a cost
ratio of 2, because it buys that accuracy with false resolutions. Both are reported so the
comparison is checkable rather than asserted.

    python -m src.eval.review_replay               # 72 live reviewer calls, writes both reports
    python -m src.eval.review_replay --from-cache  # rescore the saved verdicts, 0 API calls
"""
import argparse
import collections
import json

from src import config, gate, review
from src.eval.harness import load_adversarial, load_golden
from src.retrieval import HybridRetriever

OUT = config.ROOT_DIR / "eval_results" / "clarify_review_replay.json"
OUT_GOVERNED = config.ROOT_DIR / "eval_results" / "clarify_review_replay_governed.json"


def v3_traces() -> dict[str, dict]:
    """Second-to-last gated trace per ticket == the v3 run.

    Traces are append-only and in run order. Golden was scored by v3 then v4; adversarial by v1,
    (v2 for a 20-item subset), v3, then v4. In every case the v3 attempt is the one before last.
    Verified against both v3 reports: 160/160 decisions match.
    """
    recs = [json.loads(l) for l in open(config.TRACES_DIR / "runs.jsonl", encoding="utf-8") if l.strip()]
    by_text: dict[str, list] = collections.OrderedDict()
    for r in recs:
        if r.get("resolver") == "gated" and r.get("kb_coverage"):
            by_text.setdefault(r["ticket_text"], []).append(r)
    return {t: occ[-2] for t, occ in by_text.items() if len(occ) >= 2}


def all_items():
    items = [("golden", it["item_id"], it["question"], "RESOLVE", None) for it in load_golden()]
    items += [
        ("adversarial", it["item_id"], it["ticket_text"], it["expected_decision"], it["category"])
        for it in load_adversarial()
    ]
    return items


def flip_permitted(trace: dict) -> bool:
    """The governance GatedResolver applies: policy re-check, plus never on an injection.

    Mirrors GatedResolver._review_clarify rather than reimplementing the rule, so the two cannot
    drift apart: the proposed RESOLVE is put back through apply_policy exactly as it is at runtime.
    """
    if trace.get("injection_attempt_detected"):
        return False
    probe = gate.ResolutionAttempt(
        reasoning=trace.get("reasoning") or "",
        # See sweep.attempt_from_trace: pre-v5 traces predate supporting_excerpts, and an empty
        # list would refuse every flip for a reason the cached run never had a chance to fail.
        supporting_excerpts=trace.get("supporting_excerpts") or [0],
        kb_coverage=trace["kb_coverage"],
        missing_information=trace.get("missing_information"),
        requires_human_authority=bool(trace.get("requires_human_authority")),
        injection_attempt_detected=bool(trace.get("injection_attempt_detected")),
        decision="RESOLVE",
        answer=trace.get("answer") or "",
    )
    permitted, _ = gate.apply_policy(probe, escalate_on_partial=True)
    return permitted == "RESOLVE"


def score(flips: set[str], traces: dict, label: str) -> dict:
    """Recompute both sets with `flips` applied, and the error taxonomy the README is scored on."""
    items = all_items()
    gc = ac = fe = fr = mis = 0
    by_cat: dict[str, list[int]] = {}

    for dataset, item_id, text, expected, category in items:
        base = traces.get(text, {}).get("decision")
        if base is None:
            continue
        decision = "RESOLVE" if item_id in flips else base
        correct = decision == expected
        if dataset == "golden":
            gc += correct
            if decision != "RESOLVE":
                fe += 1
        else:
            ac += correct
            b = by_cat.setdefault(category, [0, 0])
            b[1] += 1
            b[0] += correct
            if decision == "RESOLVE" and expected != "RESOLVE":
                fr += 1
            elif decision != "RESOLVE" and expected == "RESOLVE":
                fe += 1
            elif not correct:
                mis += 1

    return {
        "configuration": label,
        "n_flips_applied": len(flips),
        "golden_decision_accuracy": gc / 100,
        "adversarial_decision_accuracy": ac / 60,
        "false_escalations": fe,
        "false_resolutions": fr,
        "misrouted_deferrals": mis,
        "error_cost_formula": f"{fr}C + {fe + mis}",
        "false_escalation_rate_golden": (100 - gc) / 100,
        "adversarial_by_category": {
            c: {"n": v[1], "decision_accuracy": v[0] / v[1]} for c, v in sorted(by_cat.items())
        },
    }


def collect_verdicts(traces: dict) -> tuple[list[dict], dict]:
    """Live reviewer calls over every cached CLARIFY. This is the only part that costs money."""
    retriever = HybridRetriever()
    spend = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    rows = []
    to_review = [x for x in all_items() if traces.get(x[2], {}).get("decision") == "CLARIFY"]
    print(f"Reviewing {len(to_review)} cached CLARIFY decisions.\n")

    for i, (dataset, item_id, text, expected, category) in enumerate(to_review):
        chunks = retriever.retrieve(text)
        try:
            verdict, usage = review.review_clarification(text, chunks, traces[text].get("answer") or "")
        except Exception as e:
            print(f"  [{i+1}/{len(to_review)}] {item_id} FAILED {type(e).__name__}")
            continue
        spend["input_tokens"] += usage.input_tokens
        spend["output_tokens"] += usage.output_tokens
        spend["calls"] += 1
        proposed, _ = review.apply_review(verdict)
        rows.append({
            "dataset": dataset, "item_id": item_id, "category": category, "expected": expected,
            "flipped_to_resolve": proposed == "RESOLVE",
            "necessary": verdict.question_is_necessary,
            "competing": verdict.competing_procedures,
        })
        print(f"  [{i+1}/{len(to_review)}] {item_id:<5} {'FLIP->RESOLVE' if proposed == 'RESOLVE' else 'keep CLARIFY '}")
    return rows, spend


def main() -> None:
    ap = argparse.ArgumentParser(description="clarification reviewer replay")
    ap.add_argument("--from-cache", action="store_true",
                    help="Rescore the saved verdicts instead of calling the model (0 API calls)")
    args = ap.parse_args()

    traces = v3_traces()

    if args.from_cache:
        saved = json.load(open(OUT, encoding="utf-8"))
        rows, spend = saved["rows"], saved.get("token_spend", {})
        print(f"Rescoring {len(rows)} saved verdicts — 0 API calls.\n")
    else:
        rows, spend = collect_verdicts(traces)

    by_id = {i[1]: i[2] for i in all_items()}
    proposed = {r["item_id"] for r in rows if r["flipped_to_resolve"]}
    governed = {i for i in proposed if flip_permitted(traces[by_id[i]])}

    ungoverned_report = score(proposed, traces, "ungoverned (every proposed flip taken)")
    governed_report = score(governed, traces, "governed (policy re-check + no injection) — SHIPPED")

    for path, report, extra in (
        (OUT, ungoverned_report, {"n_flips_proposed": len(proposed)}),
        (OUT_GOVERNED, governed_report, {"n_flips_proposed": len(proposed), "n_flips_blocked": len(proposed) - len(governed)}),
    ):
        payload = {
            "mechanism": "clarification reviewer replayed over cached v3 CLARIFY decisions",
            "n_reviewed": len(rows),
            **report,
            **extra,
            "token_spend": spend,
            "rows": rows,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n=== {report['configuration']} ===")
        for k, v in report.items():
            if k != "configuration":
                print(f"  {k}: {v}")
        print(f"  -> {path.name}")

    print("\nv3 with no reviewer: golden 49%, adversarial 71.7%, 4C + 64")


if __name__ == "__main__":
    main()
