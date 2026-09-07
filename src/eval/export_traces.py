"""Export every scored decision as one self-contained JSON for the static results explorer.

The eval reports in eval_results/ carry only item_id + decision + correctness flags. Everything
that makes a decision worth reading -- the ticket, the model's reasoning, its coverage call, what
it asked for, whether the reviewer overturned it -- lives in traces/runs.jsonl, which is
gitignored because it is append-only and large. This joins the two and writes a single file the
explorer can load with no backend and no API key.

Run: `python -m src.eval.export_traces`  (writes docs/data.js)
"""
import argparse
import collections
import json

from src import config, gate
from src.eval.compare import classify, taxonomy

OUT = config.ROOT_DIR / "docs" / "data.js"

# How the shipped attempt is picked out of the append-only log.
#
# This used to be positional: V3_INDEX = -2, because at the time golden had been scored twice and
# adversarial four times, so "the attempt before last" happened to be v3. That is only true until
# someone runs the eval again, and v5 development ran it eight more times -- after which -2 pointed
# at an arbitrary intermediate prompt and the explorer showed decisions no configuration ever
# shipped, with no error. Traces now carry gate_version, so selection asks for what it wants.
#
# Pre-v5 traces have no gate_version and no reviewer decision baked in, so they still need the
# offline replay overlaid to reconstruct the shipped v3+reviewer configuration. v5 records the
# final decision directly, review included, so nothing has to be reconstructed.
TARGET_GATE_VERSION = gate.GATE_VERSION
LEGACY_INDEX = -2


def current_config() -> dict:
    """The configuration a trace must match to be shown as the shipped one."""
    return {
        "gate_version": gate.GATE_VERSION,
        "review_clarifications": False,
        "escalate_on_partial": True,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
    }


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


# Fields a trace and a report row must agree on for the trace to be that row's. Decision alone
# is too loose -- three runs of the same config agree on most items, and picking the wrong run's
# trace would attach the wrong reasoning text to the right decision.
MATCH_FIELDS = ("decision", "kb_coverage", "model_decision", "requires_human_authority")


def select_by_report(occurrences: list[dict], row: dict) -> dict | None:
    """The most recent trace whose labels all equal this report row's.

    Why this exists. Traces only started carrying gate_config partway through v5 development,
    which means the runs the shipped figures come from predate the stamp the exporter selects on
    -- so the decisions were on disk and unusable, and the apparent fix was to pay to score all
    160 tickets again. A committed report already records what each item decided and under which
    configuration; the trace log holds the reasoning. Joining the two recovers the run exactly,
    for nothing, and is checkable: every item must match or the caller refuses.

    This is now the more robust selector of the two, because it depends on committed data rather
    than on a field having existed at the time.
    """
    matches = [
        t for t in occurrences
        if all(t.get(f) == row.get(f) for f in MATCH_FIELDS)
    ]
    return matches[-1] if matches else None


def select_attempt(occurrences: list[dict], want: dict) -> tuple[dict | None, bool]:
    """(attempt, is_current) -- the newest attempt matching `want`, else the legacy pick.

    Matching on the whole configuration rather than the prompt version is what stops the
    explorer mixing runs. Development leaves many attempts per ticket in the log, at different
    top_k values and with the reviewer both on and off; picking "the newest v5 one" would show
    whichever configuration happened to touch that ticket last, per ticket.

    is_current says whether the returned attempt already carries its final, post-review
    decision. When it does not, the caller must overlay the offline reviewer replay to
    reconstruct what shipped.
    """
    matching = [r for r in occurrences if r.get("gate_config") == want]
    if matching:
        return matching[-1], True
    if len(occurrences) >= abs(LEGACY_INDEX):
        return occurrences[LEGACY_INDEX], False
    return None, False


def main(allow_mixed: bool = False, from_report: str | None = None) -> None:
    traces = gated_by_ticket()
    report = None
    if from_report:
        report = json.load(open(config.ROOT_DIR / "eval_results" / from_report, encoding="utf-8"))
        if report.get("n_failed"):
            raise SystemExit(
                f"{from_report} lost {report['n_failed']} item(s); it does not describe a complete "
                "run and must not be published as one."
            )
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

    want = current_config()
    report_rows = {}
    if report is not None:
        report_rows = {
            r["item_id"]: r
            for r in (report.get("golden_results") or []) + (report.get("adversarial_results") or [])
        }
        want = {
            "gate_version": gate.GATE_VERSION,
            "review_clarifications": report.get("review_clarifications"),
            "escalate_on_partial": True,
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
            "reconstructed_from": from_report,
        }

    rows, skipped = [], []
    for dataset, item_id, text, expected, category, reference in items:
        occ = traces.get(text, [])
        if report is not None:
            row = report_rows.get(item_id)
            attempt = select_by_report(occ, row) if row else None
            is_current = attempt is not None
        else:
            attempt, is_current = select_attempt(occ, want)
        if attempt is None:
            skipped.append(item_id)
            continue
        v3 = attempt
        v4 = occ[-1] if occ else None

        if is_current:
            # The recorded decision is already final: apply_policy and any clarification review
            # ran before it was written. Reconstructing it here would only risk disagreeing.
            final = v3["decision"]
            governed = bool(v3.get("clarify_review") and "unnecessary" in v3["clarify_review"])
        else:
            # The reviewer is governed: it may only flip on full coverage, never on an injection.
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
            "reviewer_proposed_flip": (item_id in flipped) if not is_current else governed,
            "from_target_config": is_current,
            "gate_version": v3.get("gate_version", "pre-v5"),
            # The model's own pre-policy decision, so a policy override is visible as the
            # difference between this and final_decision rather than only as a warning string.
            "model_decision": v3.get("model_decision"),
            "kb_coverage": v3["kb_coverage"],
            "supporting_excerpts": v3.get("supporting_excerpts"),
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

    # Imported, not reimplemented. This module used to carry its own copy of the rules, and it
    # disagreed with compare.taxonomy on adversarial items that expect RESOLVE: it called them
    # false escalations, which over-reported the published page by one against the very report it
    # was reconstructed from. One definition now, in compare.classify.
    for r in rows:
        r["error_type"] = classify(r["dataset"], r["expected"], r["final_decision"])

    by_category: dict[str, list[dict]] = {}
    for r in rows:
        if r["category"]:
            by_category.setdefault(r["category"], []).append(r)

    # The summary is computed by taxonomy() over a report-shaped view of these very rows, so the
    # header cannot disagree with the rows beneath it -- which is how the off-by-one stayed
    # invisible: the page was internally consistent with its own wrong classifier.
    as_report = {
        "golden_results": [
            {"item_id": r["id"], "decision": r["final_decision"], "expected_decision": r["expected"]}
            for r in rows if r["dataset"] == "golden"
        ],
        "adversarial_results": [
            {"item_id": r["id"], "decision": r["final_decision"], "expected_decision": r["expected"],
             "category": r["category"]}
            for r in rows if r["dataset"] == "adversarial"
        ],
    }
    tax = taxonomy(as_report)

    summary = {
        "n": len(rows),
        "golden_accuracy": sum(r["correct"] for r in rows if r["dataset"] == "golden") / 100,
        "adversarial_accuracy": sum(r["correct"] for r in rows if r["dataset"] == "adversarial") / 60,
        "errors": {
            k: v for k, v in {
                "false_resolution": tax["false_resolutions"],
                "false_escalation": tax["false_escalations"],
                "misrouted": tax["misrouted_deferrals"],
                "unwarranted_deferral": tax["unwarranted_adversarial_deferrals"],
            }.items() if v
        },
        "by_category": {
            cat: {"n": len(g), "accuracy": sum(x["correct"] for x in g) / len(g)}
            for cat, g in sorted(by_category.items())
        },
        "reviewer_flips": sum(r["reviewer_flipped"] for r in rows),
        "skipped": skipped,
        "gate_config": want,
    }

    # A page that mixes configurations is worse than no page: every row looks authoritative and
    # the summary at the top is a number no configuration ever produced. This happened -- the
    # config stamp was added mid-development, so 66 tickets had a stamped trace from an
    # interrupted reviewer-on run while the other 94 fell back to a pre-v5 positional pick, and
    # the export reported "0 skipped" as if all were well. Refuse instead, and say what to run.
    # Uniformity is not enough: if every ticket falls back to the legacy positional pick the
    # versions all agree and the page is still built from whichever prompt happened to be
    # second-to-last per ticket. What must hold is that every row came from the config asked for.
    versions = collections.Counter(
        r["gate_version"] if r["from_target_config"] else f"{r['gate_version']} (legacy pick)"
        for r in rows
    )
    off_target = [r["id"] for r in rows if not r["from_target_config"]]
    if off_target and not allow_mixed:
        raise SystemExit(
            "\n".join([
                f"refusing to write an explorer that does not match one configuration.",
                f"{len(off_target)} of {len(rows)} tickets have no trace from the shipped config.",
                f"Selected instead: {dict(versions)}",
                f"Wanted every ticket scored by {want}.",
                "Run one complete pass first:",
                "  python -m src.eval.harness --resolver gated --no-judge --workers 8",
                "then re-run this. Pass --allow-mixed to override, knowing the summary will",
                "be an average over configurations rather than a measurement of one.",
            ])
        )

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
    parser = argparse.ArgumentParser(description="export scored decisions to docs/data.js")
    parser.add_argument(
        "--allow-mixed", action="store_true",
        help="write the page even when the tickets were scored by different configurations",
    )
    parser.add_argument(
        "--from-report", default=None, metavar="NAME",
        help="reconstruct the run from a committed report under eval_results/, by matching each "
             "item's recorded labels against the trace log. Use this for runs made before traces "
             "carried gate_config -- it needs no API calls and is verified item by item.",
    )
    args = parser.parse_args()
    main(args.allow_mixed, args.from_report)
