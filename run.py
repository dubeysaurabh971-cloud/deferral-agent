"""Entry point for deferral-agent.

    python run.py --ingest              # download WixQA, chunk, embed into Chroma (~5 min first run)
    python run.py                       # interactive: paste a ticket, get a decision
    python run.py --resolver naive      # the ungated Week 1 baseline, for comparison
    python run.py --compare             # both, side by side, on the same ticket

The default is the gated resolver. It used to be the baseline, which meant the quickstart shipped
the exact system the README spends its length criticising -- anyone following the instructions got
a resolver that answers everything and reports RESOLVE, then concluded the gate did nothing.
"""
import argparse
import sys

from src import config
from src.tickets import load_tickets, write_tickets_jsonl


def _print_decision(label: str, result: dict, show_trace: bool = True) -> None:
    print(f"\n=== {label} ===")
    print(f"[{result['decision']}] {result['answer']}")
    if show_trace and result.get("resolver") == "gated":
        bits = [
            f"kb_coverage={result['kb_coverage']}",
            f"model_said={result['model_decision']}",
            f"injection={result['injection_attempt_detected']}",
        ]
        if result.get("policy_override"):
            bits.append(f"policy_override={result['policy_override']!r}")
        if result.get("clarify_review"):
            bits.append(f"review={result['clarify_review']!r}")
        if result.get("missing_information"):
            bits.append(f"missing={result['missing_information']!r}")
        print("  " + "  ".join(bits))
    usage = result.get("usage") or {}
    print(f"  ({usage.get('output_tokens')} output tokens, trace in traces/runs.jsonl)")


def run_interactive(resolver_name: str = "gated", compare: bool = False) -> None:
    if not config.api_key_present():
        key_name = config.api_key_env_name()
        print(
            f"{key_name} is not set (LLM_PROVIDER={config.LLM_PROVIDER}). Copy .env.example "
            "to .env and add your key, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.resolver import GatedResolver, NaiveResolver

    print("Loading retriever + resolver...")
    gated = GatedResolver() if (compare or resolver_name == "gated") else None
    naive = None
    if compare or resolver_name == "naive":
        naive = NaiveResolver.__new__(NaiveResolver)
        if gated is not None:
            # Share the index rather than building a second one (~600MB, ~20s, identical).
            naive.retriever = gated.retriever
        else:
            naive = NaiveResolver()

    tickets = load_tickets()
    mode = "gate vs baseline" if compare else resolver_name
    print(f"\nMode: {mode}. {len(tickets)} sample tickets in data/tickets/synthetic_tickets.jsonl.")
    print("Paste a ticket, or type a ticket number (e.g. T003), or 'quit'.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            break

        ticket_text = raw
        for t in tickets:
            if raw.upper() == t["ticket_id"]:
                ticket_text = f"{t['subject']}\n\n{t['body']}"
                break

        try:
            if naive is not None:
                _print_decision("ungated baseline (always answers)", naive.resolve(ticket_text))
            if gated is not None:
                _print_decision("deferral gate", gated.resolve(ticket_text))
        except Exception as e:
            print(f"\n  {type(e).__name__}: {e}\n", file=sys.stderr)
            continue
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="deferral-agent")
    parser.add_argument("--ingest", action="store_true", help="Download KB, chunk, embed into Chroma")
    parser.add_argument(
        "--resolver", choices=["gated", "naive"], default="gated",
        help="Which resolver to run interactively (default: gated)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both resolvers on each ticket, baseline first, to see what the gate changes",
    )
    args = parser.parse_args()

    if args.ingest:
        from src.ingest import run_ingest

        run_ingest()
        write_tickets_jsonl()
        return

    run_interactive(args.resolver, args.compare)


if __name__ == "__main__":
    main()
