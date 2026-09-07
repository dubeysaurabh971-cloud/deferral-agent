"""Retrieval recall on the golden set. Zero tokens, no model calls.

This is the measurement behind finding 9, and committing it closes the last gap where a
load-bearing number in the README existed only inside a comment. It is also the cheapest thing
in the repo to run, which made the absence harder to justify than any of the paid runs.

WHY IT MATTERS. Every golden item records the `article_ids` its reference answer was written
from. If none of them reach the retrieved top-k, the answer is not in the context, and a
correctly-calibrated gate *has* to defer -- so false escalation has a floor that no amount of
prompt work can lift. Measured at the shipped `top_k=5` of the time, recall was 76%: for 24 of
100 answerable tickets the article containing the answer was never retrieved. Three iterations
of prompt engineering had been spent against a constraint that was never in the prompt.

TWO THINGS IT COMPUTES.

  sweep   recall@k for several (candidate_pool, top_k) pairs. RRF ranks the fused candidate
          list before truncation, so ranking does not depend on top_k -- one retrieval per pool
          at the largest k is truncated for the smaller ones, which is exact and four times
          cheaper than re-retrieving.

  split   recall on the items a set of runs *always* deferred, against recall on the rest. This
          is the half that turns a global number into a diagnosis: a uniform 76% would say
          retrieval is mediocre everywhere, whereas 54% on the always-deferred items against
          80% on the others says the gap explains the failures.

    python -m src.eval.recall                                    # sweep + split at the shipped top_k
    python -m src.eval.recall --split-reports v5_topk5_run1.json v5_topk5_run2.json \
        v5_topk5_run3.json --split-top-k 5                       # reproduces finding 9 exactly

Needs the embedded index, so it cannot run in CI: `python run.py --ingest` first.
"""
import argparse
import json
import statistics

from src import config

RESULTS_DIR = config.ROOT_DIR / "eval_results"

# Wide enough to show where the curve flattens, and to show that a deeper candidate pool makes
# recall *worse* -- RRF rewards agreement between the two rankings, and a deeper pool adds
# rank-tail chunks that dilute it. That result is why only top_k moves in config.py.
POOLS = (config.RETRIEVAL_CANDIDATE_POOL, 40, 60)
TOP_KS = (5, 8, 10, 12)


def load_golden() -> list[dict]:
    path = config.DATA_DIR / "golden" / "golden_set.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def article_ids_of(chunks: list[dict]) -> set[str]:
    """Chunk ids are '<article_id>-<index>', and retrieve() does not return article_id."""
    return {c["chunk_id"].rsplit("-", 1)[0] for c in chunks}


def measure(retriever, golden: list[dict], top_ks) -> dict[int, dict]:
    """recall@k for each k, plus the per-item hit map, from one retrieval per item.

    Retrieves at max(top_ks) and truncates. Exact, because RRF orders the fused list before the
    cut: the top 5 of a top-10 retrieval are the same five chunks in the same order.
    """
    largest = max(top_ks)
    per_item: dict[str, dict[int, bool]] = {}
    for item in golden:
        chunks = retriever.retrieve(item["question"], top_k=largest)
        want = set(item["article_ids"])
        per_item[item["item_id"]] = {
            k: bool(article_ids_of(chunks[:k]) & want) for k in top_ks
        }
    return {
        k: {
            "recall": sum(1 for v in per_item.values() if v[k]) / len(golden),
            "n_hit": sum(1 for v in per_item.values() if v[k]),
            "n": len(golden),
            "hits": {i: v[k] for i, v in per_item.items()},
        }
        for k in top_ks
    }


def always_deferred(report_names: list[str]) -> set[str]:
    """Golden item ids that every one of these runs refused to answer.

    Intersection rather than union: an item that deferred in one run of three is inside the
    run-to-run spread and says nothing. The systematic ones are the diagnosis.
    """
    sets = []
    for name in report_names:
        rep = json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
        sets.append({r["item_id"] for r in rep["golden_results"] if r["decision"] != "RESOLVE"})
    return set.intersection(*sets) if sets else set()


def main() -> None:
    ap = argparse.ArgumentParser(description="golden-set retrieval recall (no model calls)")
    ap.add_argument(
        "--split-reports", nargs="*", default=[
            "v5_topk5_run1.json", "v5_topk5_run2.json", "v5_topk5_run3.json",
        ],
        help="reports whose always-deferred golden items form the split. Defaults to the "
             "top_k=5 runs, which is what finding 9 was measured on.",
    )
    ap.add_argument(
        "--split-top-k", type=int, default=5,
        help="the top_k at which to report the split -- normally whatever those runs used (5).",
    )
    ap.add_argument("--out", default="v5_recall.json")
    args = ap.parse_args()

    from src.retrieval import HybridRetriever

    golden = load_golden()
    out = {
        "n_golden": len(golden),
        "note": (
            "Recall is measured against each golden item's own article_ids -- the articles its "
            "reference answer was written from. An answer available in some OTHER article counts "
            "as a miss, so these figures are lower bounds on the retriever, and the 'could not "
            "have been answered' claim they support is correspondingly an upper bound."
        ),
        "sweep": {},
    }

    print(f"{'pool':>6}{'top_k':>7}{'recall':>9}{'hit':>7}")
    for pool in POOLS:
        retriever = HybridRetriever(candidate_pool=pool)
        ks = [k for k in TOP_KS if k <= pool]
        measured = measure(retriever, golden, ks)
        for k in ks:
            m = measured[k]
            print(f"{pool:>6}{k:>7}{m['recall']:>8.0%}{m['n_hit']:>7}")
            out["sweep"][f"pool{pool}_topk{k}"] = {
                "candidate_pool": pool, "top_k": k,
                "recall": m["recall"], "n_hit": m["n_hit"], "n": m["n"],
            }
        if pool == config.RETRIEVAL_CANDIDATE_POOL:
            out["_hits_at_default_pool"] = {k: measured[k]["hits"] for k in ks}

    # --- the split ---------------------------------------------------------------------
    deferred = always_deferred(args.split_reports)
    hits = (out.get("_hits_at_default_pool") or {}).get(args.split_top_k)
    if hits is None:
        raise SystemExit(
            f"no hit map at top_k={args.split_top_k} for the default candidate pool; "
            f"--split-top-k must be one of {TOP_KS}"
        )
    resolved = [i for i in hits if i not in deferred]
    dr = statistics.mean(hits[i] for i in deferred) if deferred else None
    rr = statistics.mean(hits[i] for i in resolved) if resolved else None

    out["split"] = {
        "reports": args.split_reports,
        "top_k": args.split_top_k,
        "always_deferred_items": sorted(deferred),
        "n_always_deferred": len(deferred),
        "recall_on_always_deferred": dr,
        "recall_on_the_rest": rr,
        "interpretation": (
            "Recall on the items every run refused, against recall on the rest. A uniform figure "
            "would mean retrieval is mediocre everywhere; a gap means it explains the failures."
        ),
    }
    out.pop("_hits_at_default_pool", None)

    print(f"\nsplit at top_k={args.split_top_k}, over {len(args.split_reports)} run(s):")
    print(f"  always deferred ({len(deferred):>3} items): recall {dr:.0%}" if dr is not None
          else "  always deferred: none")
    print(f"  the rest        ({len(resolved):>3} items): recall {rr:.0%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULTS_DIR / args.out}")


if __name__ == "__main__":
    main()
