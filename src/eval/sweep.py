"""Offline sweep of the gate's escalate_on_partial knob. Costs zero tokens.

gate.apply_policy is a pure function of three fields the gated resolver already writes to
traces/runs.jsonl -- model_decision, kb_coverage, requires_human_authority. So the knob does
not need a second paid eval run to evaluate: replay the policy over the cached attempts and
both settings fall out of one run's traces.

That is the whole point. escalate_on_partial trades adversarial recall (near_miss tickets are
exactly the "topic covered, specific fact absent" case) against false escalations on golden
tickets the model happens to label 'partial'. Deciding it needs both numbers side by side, and
paying twice for a comparison that is arithmetic would be waste.

Run: `python -m src.eval.sweep`  (after at least one `--resolver gated` eval run)
"""
import json

from src import config, gate
from src.eval.harness import accuracy, deferral_metrics, load_adversarial, load_golden

TRACES_PATH = config.TRACES_DIR / "runs.jsonl"


def load_gated_traces() -> dict[str, dict]:
    """Cached gated attempts, keyed by ticket text. Later runs overwrite earlier ones, so a
    re-run of the same item supersedes rather than double-counts."""
    if not TRACES_PATH.exists():
        return {}
    by_text: dict[str, dict] = {}
    with TRACES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("resolver") != "gated" or rec.get("kb_coverage") is None:
                continue
            by_text[rec["ticket_text"]] = rec
    return by_text


def attempt_from_trace(rec: dict) -> gate.ResolutionAttempt:
    """Rebuild the model's attempt from a cached trace.

    supporting_excerpts arrived in v5 and traces written before it do not carry it. An empty
    list means "claimed coverage it could not point at", which apply_policy refuses -- so
    defaulting to empty here would turn every replayed RESOLVE into an ESCALATE and quietly
    invalidate the one thing this sweep measures. A pre-v5 trace simply has nothing to say
    about that rule, so it replays as if the citation check passed and only the knob varies.
    """
    cited = rec.get("supporting_excerpts")
    if cited is None:
        cited = [0]  # pre-v5 trace: rule not replayable, do not let it decide the outcome
    return gate.ResolutionAttempt(
        reasoning=rec.get("reasoning", ""),
        supporting_excerpts=cited,
        kb_coverage=rec["kb_coverage"],
        missing_information=rec.get("missing_information"),
        requires_human_authority=bool(rec.get("requires_human_authority")),
        injection_attempt_detected=bool(rec.get("injection_attempt_detected")),
        decision=rec["model_decision"],
        answer=rec.get("answer", ""),
    )


def replay(escalate_on_partial: bool, traces: dict[str, dict]) -> dict:
    golden_results, adversarial_results = [], []

    for item in load_golden():
        rec = traces.get(item["question"])
        if rec is None:
            continue
        decision, _ = gate.apply_policy(attempt_from_trace(rec), escalate_on_partial)
        golden_results.append({
            "item_id": item["item_id"],
            "decision": decision,
            "decision_correct": decision == item["expected_decision"],
        })

    for item in load_adversarial():
        rec = traces.get(item["ticket_text"])
        if rec is None:
            continue
        decision, _ = gate.apply_policy(attempt_from_trace(rec), escalate_on_partial)
        adversarial_results.append({
            "item_id": item["item_id"],
            "category": item["category"],
            "expected_decision": item["expected_decision"],
            "decision": decision,
            "decision_correct": decision == item["expected_decision"],
        })

    false_escalation_rate, deferral_precision = deferral_metrics(golden_results, adversarial_results)
    by_cat = {}
    for row in adversarial_results:
        by_cat.setdefault(row["category"], []).append(row)

    return {
        "escalate_on_partial": escalate_on_partial,
        "n_golden": len(golden_results),
        "n_adversarial": len(adversarial_results),
        "golden_decision_accuracy": accuracy(golden_results),
        "adversarial_decision_accuracy": accuracy(adversarial_results),
        "adversarial_by_category": {
            cat: {"n": len(rows), "decision_accuracy": accuracy(rows)}
            for cat, rows in sorted(by_cat.items())
        },
        "false_escalation_rate": false_escalation_rate,
        "deferral_precision": deferral_precision,
    }


def main() -> None:
    traces = load_gated_traces()
    if not traces:
        print(
            f"No gated traces in {TRACES_PATH}. Run an eval with --resolver gated first;\n"
            "this sweep only replays cached attempts, it never calls the model."
        )
        return

    print(f"Replaying {len(traces)} cached gated attempts -- 0 API calls.\n")
    reports = [replay(setting, traces) for setting in (True, False)]

    for report in reports:
        print(f"--- escalate_on_partial={report['escalate_on_partial']} ---")
        for key, value in report.items():
            if key == "escalate_on_partial":
                continue
            print(f"  {key}: {value}")
        print()

    out = config.ROOT_DIR / "eval_results" / "gated_policy_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"matched_traces": len(traces), "settings": reports}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
