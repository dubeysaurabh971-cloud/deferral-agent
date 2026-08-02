"""Append-only JSONL trace log. One line per run: input, retrieved chunks, decision, etc."""
import json
import uuid

from src import config


def log_trace(record: dict) -> str:
    config.TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_id = record.setdefault("trace_id", str(uuid.uuid4()))
    path = config.TRACES_DIR / "runs.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return trace_id
