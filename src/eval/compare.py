"""Side-by-side diff of two eval reports, in the error taxonomy the project is judged on.

Headline accuracy hides the trade that matters. A change that moves golden accuracy up by
deferring less is only good if it did not buy that by resolving tickets it should have
deferred, and the two live in different sets -- so they have to be read together or not at
all. This prints both, plus the per-item decision changes, so a regression is attributable
to specific tickets rather than to a number that moved.

    python -m src.eval.compare eval_results/a.json eval_results/b.json
"""
import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def taxonomy(report: dict) -> dict:
    """The four error kinds, counted the way README's cost model counts them.

    false_resolution  -- an item needing a deferral that got RESOLVE. The expensive error.
    false_escalation  -- an answerable (golden) item that got deferred. The cost side.
    misrouted         -- deferred, correctly, but CLARIFY vs ESCALATE the wrong way round.
    """
    g = report.get("golden_results") or []
    a = report.get("adversarial_results") or []
    false_escalation = [r for r in g if r["decision"] != "RESOLVE"]
    false_resolution = [
        r for r in a if r["expected_decision"] != "RESOLVE" and r["decision"] == "RESOLVE"
    ]
    misrouted = [
        r for r in a
        if r["expected_decision"] != "RESOLVE"
        and r["decision"] != "RESOLVE"
        and r["decision"] != r["expected_decision"]
    ]
    return {
        "n_golden": len(g),
        "n_adversarial": len(a),
        "golden_accuracy": report.get("golden_decision_accuracy"),
        "adversarial_accuracy": report.get("adversarial_decision_accuracy"),
        "false_escalations": len(false_escalation),
        "false_escalation_rate": (len(false_escalation) / len(g)) if g else None,
        "false_resolutions": len(false_resolution),
        "misrouted_deferrals": len(misrouted),
        "by_category": report.get("adversarial_by_category") or {},
        "golden_clarify": sum(1 for r in g if r["decision"] == "CLARIFY"),
        "golden_escalate": sum(1 for r in g if r["decision"] == "ESCALATE"),
        "_false_escalation_ids": [r["item_id"] for r in false_escalation],
        "_false_resolution_ids": [r["item_id"] for r in false_resolution],
    }


def fmt(v) -> str:
    if v is None:
        return "  --"
    return f"{v:.0%}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="diff two eval reports")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--items", action="store_true", help="list per-item decision changes")
    args = ap.parse_args()

    b, a = load(args.before), load(args.after)
    tb, ta = taxonomy(b), taxonomy(a)

    print(f"\n{'':34s}{'before':>10s}{'after':>10s}{'delta':>10s}")
    print("-" * 64)

    def row(label, key, invert=False, pct=True):
        vb, va = tb.get(key), ta.get(key)
        delta = ""
        if isinstance(vb, (int, float)) and isinstance(va, (int, float)):
            d = va - vb
            sign = "+" if d > 0 else ""
            delta = f"{sign}{d:.0%}" if pct and isinstance(vb, float) else f"{sign}{d}"
            if d != 0:
                good = (d < 0) if invert else (d > 0)
                delta += "  ok" if good else "  X"
        print(f"{label:34s}{fmt(vb):>10s}{fmt(va):>10s}{delta:>12s}")

    row("golden decision accuracy", "golden_accuracy")
    row("FALSE ESCALATION RATE", "false_escalation_rate", invert=True)
    row("  golden CLARIFY", "golden_clarify", invert=True, pct=False)
    row("  golden ESCALATE", "golden_escalate", invert=True, pct=False)
    print()
    row("adversarial decision accuracy", "adversarial_accuracy")
    row("FALSE RESOLUTIONS", "false_resolutions", invert=True, pct=False)
    row("misrouted deferrals", "misrouted_deferrals", invert=True, pct=False)

    print()
    cats = sorted(set(tb["by_category"]) | set(ta["by_category"]))
    for c in cats:
        vb = (tb["by_category"].get(c) or {}).get("decision_accuracy")
        va = (ta["by_category"].get(c) or {}).get("decision_accuracy")
        n = (ta["by_category"].get(c) or tb["by_category"].get(c) or {}).get("n")
        delta = ""
        if vb is not None and va is not None:
            d = va - vb
            delta = f"{'+' if d > 0 else ''}{d:.0%}"
        print(f"  {c + f' (n={n})':32s}{fmt(vb):>10s}{fmt(va):>10s}{delta:>12s}")

    # Error cost is linear in C = cost(false resolution) / cost(unnecessary handoff), and the
    # two configurations swap places at the C where the lines cross. Printing the formulae
    # keeps the ship/no-ship argument in the same units the README uses.
    print(f"\nerror cost:  before = {tb['false_resolutions']}C + "
          f"{tb['false_escalations'] + tb['misrouted_deferrals']}"
          f"   after = {ta['false_resolutions']}C + "
          f"{ta['false_escalations'] + ta['misrouted_deferrals']}")
    dfr = ta["false_resolutions"] - tb["false_resolutions"]
    dfe = (tb["false_escalations"] + tb["misrouted_deferrals"]) - (
        ta["false_escalations"] + ta["misrouted_deferrals"]
    )
    if dfr > 0 and dfe > 0:
        print(f"             after wins while C < {dfe / dfr:.2f}")
    elif dfr <= 0 and dfe >= 0:
        print("             after dominates (fewer of both)")

    if args.items:
        bd = {r["item_id"]: r["decision"] for r in (b.get("golden_results") or []) + (b.get("adversarial_results") or [])}
        ad = {r["item_id"]: r["decision"] for r in (a.get("golden_results") or []) + (a.get("adversarial_results") or [])}
        changed = [(k, bd[k], ad[k]) for k in sorted(bd) if k in ad and bd[k] != ad[k]]
        print(f"\n{len(changed)} decisions changed:")
        for k, x, y in changed:
            print(f"  {k}: {x} -> {y}")


if __name__ == "__main__":
    main()
