"""Append-only JSONL trace log. One line per run: input, retrieved chunks, decision, etc."""
import json
import threading
import uuid

from src import config

# The eval harness can score items concurrently (--workers), and every scored item writes a
# trace. Two threads appending to the same handle can interleave mid-line and produce JSONL
# that no longer parses -- which would silently corrupt the cache that sweep.py and
# review_replay.py replay from. One process-wide lock is enough: the writes are short and the
# contention is nothing next to a 14s model call.
_WRITE_LOCK = threading.Lock()


def log_trace(record: dict) -> str:
    config.TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_id = record.setdefault("trace_id", str(uuid.uuid4()))
    path = config.TRACES_DIR / "runs.jsonl"
    line = json.dumps(record) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    return trace_id
