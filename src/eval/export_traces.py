"""Export every scored decision as one self-contained JSON for the static results explorer.

The eval reports in eval_results/ carry only item_id + decision + correctness flags. Everything
that makes a decision worth reading -- the ticket, the model's reasoning, its coverage call, what
it asked for, whether the reviewer overturned it -- lives in traces/runs.jsonl, which is
gitignored because it is append-only and large. This joins the two and writes a single file the
explorer can load with no backend and no API key.

Run: `python -m src.eval.export_traces`  (writes docs/data.js)
"""
import collections
import json

from src import config

OUT = config.ROOT_DIR / "docs" / "data.js"

# Which gated attempt in the append-only log belongs to which prompt revision. Golden was scored
# by v3 then v4; adversarial by v1, (v2 on a 20-item subset), v3, then v4. So the v3 attempt is
# always the one before last -- verified against both v3 reports, 160/160 decisions match.
V3_INDEX = -2


def gated_by_ticket() -> dict[str, list[dict]]:
    path = config.TRACES_DIR / "runs.jsonl"
    by_text: dict[str, list[dict]] = collections.OrderedDict()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("resolver") == "gated" and rec.get("kb_coverage"):
                by_text.setdefault(rec["ticket_text"], []).append(rec)
    return by_text


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    traces = gated_by_ticket()
    replay = json.load(open(config.ROOT_DIR / "eval_results" / "clarify_review_replay.json", encoding="utf-8"))
    flipped = {r["item_id"] for r in replay["rows"] if r["flipped_to_resolve"]}

    golden = load_jsonl(config.DATA_DIR / "golden" / "golden_set.jsonl")
    adversarial = load_jsonl(config.DATA_DIR / "adversarial" / "adversarial_set.jsonl")

    items = []
    for it in golden:
        items.append(("golden", it["item_id"], it["question"], "RESOLVE", None, it.get("reference_answer")))
    for it in adversarial:
        items.append(
            ("adversarial", it["item_id"], it["ticket_text"], it["expected_decision"], it["category"], None)
        )

    rows, skipped = [], []
    for dataset, item_id, text, expected, category, reference in items:
        occ = traces.get(text, [])
        if len(occ) < abs(V3_INDEX):
            skipped.append(item_id)
            continue
        v3 = occ[V3_INDEX]
        v4 = occ[-1] if len(occ) >= 1 else None

        # The reviewer is governed: it may only flip on full coverage, and never on an injection.
        governed = (
            item_id in flipped
            and v3["kb_coverage"] == "full"
            and not v3.get("injection_attempt_detected")
        )
        final = "RESOLVE" if governed else v3["decision"]

        rows.append({
            "id": item_id,
            "dataset": dataset,
            "category": category,
            "ticket": text,
            "expected": expected,
            "gate_decision": v3["decision"],
            "final_decision": final,
            "correct": final == expected,
            "correct_before_review": v3["decision"] == expected,
            "reviewer_flipped": governed,
            "reviewer_proposed_flip": item_id in flipped,
            "kb_coverage": v3["kb_coverage"],
            "missing_information": v3.get("missing_information"),
            "requires_human_authority": v3.get("requires_human_authority"),
            "injection_detected": v3.get("injection_attempt_detected"),
            "policy_override": v3.get("policy_override"),
            "reasoning": v3.get("reasoning"),
            "answer": v3.get("answer"),
            "v4_decision": v4["decision"] if v4 else None,
            "reference_answer": reference,
            "retrieved": [c["title"] for c in v3.get("retrieved_chunks", [])],
        })

    def err(r):
        """The error taxonomy the README is scored on."""
        if r["correct"]:
            return None
        if r["final_decision"] == "RESOLVE":
            return "false_resolution"
        if r["expected"] == "RESOLVE":
            return "false_escalation"
        return "misrouted"

    for r in rows:
        r["error_type"] = err(r)

    by_category: dict[str, list[dict]] = {}
    for r in rows:
        if r["category"]:
            by_category.setdefault(r["category"], []).append(r)

    summary = {
        "n": len(rows),
        "golden_accuracy": sum(r["correct"] for r in rows if r["dataset"] == "golden") / 100,
        "adversarial_accuracy": sum(r["correct"] for r in rows if r["dataset"] == "adversarial") / 60,
        "errors": dict(collections.Counter(r["error_type"] for r in rows if r["error_type"])),
        "by_category": {
            cat: {"n": len(g), "accuracy": sum(x["correct"] for x in g) / len(g)}
            for cat, g in sorted(by_category.items())
        },
        "reviewer_flips": sum(r["reviewer_flipped"] for r in rows),
        "skipped": skipped,
    }

    # Emitted as a JS global rather than JSON so the page also works opened straight off disk --
    # fetch() is blocked on file:// URLs, a <script src> is not. GitHub Pages serves it either way.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("window.DECISIONS = ")
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
        f.write(";" + chr(10))
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {len(rows)} decisions to {OUT} ({size_kb:.0f} KB)")
    if skipped:
        print(f"SKIPPED {len(skipped)} items with too few traces: {skipped}")
    print(json.dumps(summary, indent=2)[:600])


if __name__ == "__main__":
    main()
