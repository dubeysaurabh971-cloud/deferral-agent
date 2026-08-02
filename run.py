"""Entry point for deferral-agent.

    python run.py --ingest   # download WixQA, chunk, embed into Chroma (~5 min first run)
    python run.py            # interactive mode: paste a ticket, get an answer
"""
import argparse
import sys

from src import config
from src.tickets import load_tickets, write_tickets_jsonl


def run_interactive() -> None:
    if not config.ANTHROPIC_API_KEY:
        print(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.resolver import NaiveResolver

    print("Loading retriever + resolver...")
    resolver = NaiveResolver()

    tickets = load_tickets()
    print(f"\n{len(tickets)} sample tickets available (data/tickets/synthetic_tickets.jsonl).")
    print("Paste a ticket, or type a ticket number (e.g. T003), or 'quit'.\n")

    while True:
        raw = input("> ").strip()
        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            break

        ticket_text = raw
        for t in tickets:
            if raw.upper() == t["ticket_id"]:
                ticket_text = f"{t['subject']}\n\n{t['body']}"
                break

        result = resolver.resolve(ticket_text)
        print(f"\n[{result['decision']}] {result['answer']}\n")
        print(f"(trace logged to traces/runs.jsonl, {result['usage']['output_tokens']} output tokens)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="deferral-agent")
    parser.add_argument("--ingest", action="store_true", help="Download KB, chunk, embed into Chroma")
    args = parser.parse_args()

    if args.ingest:
        from src.ingest import run_ingest

        run_ingest()
        write_tickets_jsonl()
        return

    run_interactive()


if __name__ == "__main__":
    main()
