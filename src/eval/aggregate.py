"""Aggregate the repeated eval runs of one configuration into a single report.

Every headline number in this project comes from a 160-item run, and a single run moves by
several points for no reason at all: re-measuring the *unchanged* shipped v4 configuration
gave a false escalation rate of 37% where the README had recorded 44%. A project whose whole
argument is about a 25-point improvement cannot report point estimates from single runs and
call the comparison honest.

So the shipped figures are means over repeats, with the per-run spread printed next to them.
Zero tokens -- it only reads reports already written.

    python -m src.eval.aggregate --label "v5 gate (top_k=10, citation check)" \
        v5_cite_run1.json v5_cite_run2.json v5_cite_run3.json --out v5_shipped.json
"""
import argparse
import json
import statistics

from src import config
from src.eval.compare import taxonomy

RESULTS_DIR = config.ROOT_DIR / "eval_results"

# The four numbers the error-cost model is built from, plus the two accuracies. Kept explicit
# rather than "every numeric key" so a new report field cannot silently join the headline.
METRICS = (
    "golden_accuracy",
    "false_escalation_rate",
    "false_escalations",
    "false_resolutions",
    "misrouted_deferrals",
    "adversarial_accuracy",
    "golden_clarify",
    "golden_escalate",
)


def spread(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "runs": values,
        # Population stdev over 2-3 runs is a description of these runs, not an estimate of the
        # sampling distribution. Reported so the spread is visible, not so it can be tested on.
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
    }


def aggregate(filenames: list[str], label: str) -> dict:
    reports = [json.loads((RESULTS_DIR / f).read_text(encoding="utf-8")) for f in filenames]
    incomplete = [
        f for f, r in zip(filenames, reports)
        if r.get("n_failed") or r.get("golden_coverage") != "complete"
        or r.get("adversarial_coverage") != "complete"
    ]
    if incomplete:
        raise SystemExit(
            f"refusing to aggregate incomplete runs: {incomplete}. Every rate is a fraction over "
            "the items that survived, so averaging a run that lost items silently reweights it."
        )

    taxes = [taxonomy(r) for r in reports]
    out = {
        "label": label,
        "n_runs": len(reports),
        "source_reports": filenames,
        "resolver_model": reports[0].get("resolver_model"),
        "review_clarifications": reports[0].get("review_clarifications"),
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
        "n_golden": taxes[0]["n_golden"],
        "n_adversarial": taxes[0]["n_adversarial"],
        "metrics": {m: spread([t[m] for t in taxes]) for m in METRICS},
    }

    cats = sorted({c for t in taxes for c in t["by_category"]})
    out["by_category"] = {
        c: spread([(t["by_category"].get(c) or {}).get("decision_accuracy") or 0.0 for t in taxes])
        for c in cats
    }

    fr = out["metrics"]["false_resolutions"]["mean"]
    deferral_errors = (
        out["metrics"]["false_escalations"]["mean"] + out["metrics"]["misrouted_deferrals"]["mean"]
    )
    out["error_cost"] = {
        "formula": f"{fr:.1f}C + {deferral_errors:.1f}",
        "false_resolutions": fr,
        "deferral_errors": deferral_errors,
        "note": "C = cost(false resolution) / cost(unnecessary handoff).",
    }
    out["token_spend_per_run"] = {
        "input_tokens": statistics.mean(r["token_spend"]["input_tokens"] for r in reports),
        "output_tokens": statistics.mean(r["token_spend"]["output_tokens"] for r in reports),
        "calls": statistics.mean(r["token_spend"]["calls"] for r in reports),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="mean + spread over repeated eval runs")
    ap.add_argument("reports", nargs="+", help="filenames under eval_results/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="output filename under eval_results/")
    args = ap.parse_args()

    out = aggregate(args.reports, args.label)
    (RESULTS_DIR / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{out['label']}  ({out['n_runs']} runs, n={out['n_golden']}+{out['n_adversarial']})\n")
    for name, s in out["metrics"].items():
        rng = f"[{s['min']:.3g} .. {s['max']:.3g}]" if s["stdev"] is not None else ""
        print(f"  {name:24s} {s['mean']:>8.3g}   {rng}")
    print()
    for c, s in out["by_category"].items():
        print(f"  {c:24s} {s['mean']:>8.0%}   [{s['min']:.0%} .. {s['max']:.0%}]")
    print(f"\n  error cost = {out['error_cost']['formula']}")
    print(f"\nSaved to {RESULTS_DIR / args.out}")


if __name__ == "__main__":
    main()
