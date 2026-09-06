"""Assemble a Hugging Face Space from this repo. Run locally; commit the output to the Space.

    python -m deploy.build_space          # writes ./space_build/

The Space must never run ingestion: free Spaces sleep on idle and lose non-persistent disk on
restart, so a boot-time `--ingest` would rerun on every wake and take ~5 minutes each time. This
ships a prebuilt index instead.

The index is slimmed on the way out. Chroma builds an FTS5 full-text index over all 10,068
documents, which this pipeline never queries -- BM25 runs separately in retrieval.py over
chunks.jsonl via rank_bm25, and the Chroma call is `query_texts`, i.e. dense only. Dropping those
tables takes the store from 138MB to 80MB. Verified byte-identical: same chunk ids and same RRF
scores to 8 decimal places across a sample of queries (see verify_slim_index()).
"""
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "space_build"

FTS_PREFIX = "embedding_fulltext_search"

SPACE_README = """---
title: Deferral Gate Demo
emoji: "⚖️"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
short_description: A support agent that decides whether to answer, ask, or hand off
---

# Deferral gate — live demo

A support-ticket RAG agent that decides whether to **answer**, **ask a clarifying question**, or
**hand off to a human**, and shows its full decision trace.

**This over-defers on purpose-of-record.** It refuses about 44% of answerable tickets, well above
the 15% bar the project sets for shipping. That is measured behaviour, not a bug — it cuts
confidently wrong answers from 58 to 5 and pays for it in unnecessary handoffs.

Full results, methodology and the failed fixes: https://github.com/dubeysaurabh971-cloud/deferral-agent
"""

DOCKERFILE = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image layer. Free Spaces have no persistent disk, so a
# runtime download would repeat on every cold start (~90MB, and it is on the critical path
# before the first query can be served).
RUN python -c "from sentence_transformers import SentenceTransformer; \\
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# resolver.py appends a trace per call; the directory has to exist and be writable.
RUN mkdir -p /app/traces && chmod 777 /app/traces

ENV HOME=/app \\
    STREAMLIT_SERVER_HEADLESS=true \\
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

# Pinned loosely to what the project already uses. chromadb and sentence-transformers dominate
# the image; datasets/pytest/streamlit-only-for-dev are dropped -- the Space never ingests.
REQUIREMENTS = """openai
chromadb
sentence-transformers
rank_bm25
pydantic
python-dotenv
streamlit
"""

GITATTRIBUTES = """*.sqlite3 filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.jsonl filter=lfs diff=lfs merge=lfs -text
data/chroma/** filter=lfs diff=lfs merge=lfs -text
"""


def slim_chroma(dest: Path) -> tuple[float, float]:
    """Drop Chroma's unused FTS tables and compact. Returns (before_mb, after_mb)."""
    db = dest / "chroma.sqlite3"
    before = db.stat().st_size / 1048576
    con = sqlite3.connect(db)
    tables = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (FTS_PREFIX + "%",),
        ).fetchall()
    ]
    for t in tables:
        con.execute(f'DROP TABLE IF EXISTS "{t}"')
    con.commit()
    con.execute("VACUUM")
    con.close()
    return before, db.stat().st_size / 1048576


def verify_slim_index(space_dir: Path) -> bool:
    """Retrieval against the slimmed store must match the original exactly, or we do not ship it."""
    import importlib

    from src import config

    queries = [
        "Can I start accepting payments on my site while my Wix Payments account is still under verification?",
        "My site isn't loading for me.",
        "What's the exact daily payout withdrawal limit for Wix Payments accounts based in Canada?",
        "Im looking to create an automation for a standard Wix form.",
        "How do I cancel my subscription?",
    ]

    def fingerprint(chroma_dir: Path):
        config.CHROMA_DIR = chroma_dir
        import src.retrieval

        importlib.reload(src.retrieval)
        r = src.retrieval.HybridRetriever()
        return {q: [(c["chunk_id"], round(c["score"], 8)) for c in r.retrieve(q)] for q in queries}

    original = fingerprint(ROOT / "data" / "chroma")
    slim = fingerprint(space_dir / "data" / "chroma")
    mismatches = [q for q in queries if original[q] != slim[q]]
    for q in queries:
        print(f"  {'IDENTICAL' if q not in mismatches else 'DIFFERS  '}  {q[:60]}")
    return not mismatches


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print("Copying pipeline...")
    shutil.copytree(ROOT / "src", OUT / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "deploy" / "space_app.py", OUT / "app.py")

    print("Copying prebuilt index (no ingestion on the Space)...")
    (OUT / "data" / "kb").mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "kb" / "chunks.jsonl", OUT / "data" / "kb" / "chunks.jsonl")
    shutil.copytree(ROOT / "data" / "chroma", OUT / "data" / "chroma")

    before, after = slim_chroma(OUT / "data" / "chroma")
    print(f"  chroma.sqlite3 {before:.1f} MB -> {after:.1f} MB")

    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    (OUT / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (OUT / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1048576
    print(f"\nSpace assembled at {OUT}  ({total:.1f} MB)")
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.stat().st_size > 1048576:
            print(f"  {f.stat().st_size / 1048576:7.1f} MB  {f.relative_to(OUT)}")

    print("\nVerifying the slimmed index returns identical retrieval...")
    if verify_slim_index(OUT):
        print("\nOK — identical chunk ids and RRF scores. Safe to deploy.")
    else:
        print("\nFAILED — retrieval differs. Do NOT deploy this build.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
