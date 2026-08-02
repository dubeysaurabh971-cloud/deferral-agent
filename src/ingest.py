"""Load the WixQA knowledge-base corpus, chunk it, and embed it into Chroma.

Run via `python run.py --ingest`.
"""
import json

import chromadb
import tiktoken
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datasets import load_dataset

from src import config

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window token chunking with overlap."""
    tokens = _ENCODING.encode(text)
    if not tokens:
        return []
    stride = chunk_size - overlap
    chunks = []
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(_ENCODING.decode(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def build_chunks() -> list[dict]:
    """Download the WixQA KB corpus and split every article into overlapping chunks."""
    articles = load_dataset(config.KB_DATASET, config.KB_CONFIG, split="train")

    records = []
    for article in articles:
        pieces = chunk_text(
            article["contents"],
            chunk_size=config.CHUNK_SIZE_TOKENS,
            overlap=config.CHUNK_OVERLAP_TOKENS,
        )
        for i, piece in enumerate(pieces):
            records.append(
                {
                    "chunk_id": f"{article['id']}-{i}",
                    "article_id": article["id"],
                    "chunk_index": i,
                    "title": article["title"],
                    "url": article["url"],
                    "article_type": article["article_type"],
                    "text": piece,
                }
            )
    return records


def persist_chunks_jsonl(records: list[dict]) -> None:
    """Write chunks to disk so BM25 can be rebuilt without re-querying Chroma."""
    config.DATA_DIR.joinpath("kb").mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / "kb" / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def embed_into_chroma(records: list[dict]) -> None:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    try:
        client.delete_collection(config.KB_COLLECTION_NAME)
    except Exception:
        pass

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
    collection = client.create_collection(
        name=config.KB_COLLECTION_NAME, embedding_function=embedding_fn
    )

    batch_size = 256
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        collection.add(
            ids=[r["chunk_id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[
                {
                    "article_id": r["article_id"],
                    "chunk_index": r["chunk_index"],
                    "title": r["title"],
                    "url": r["url"],
                    "article_type": r["article_type"],
                }
                for r in batch
            ],
        )


def run_ingest() -> None:
    print(f"Downloading {config.KB_DATASET} ({config.KB_CONFIG}) ...")
    records = build_chunks()
    print(f"Chunked into {len(records)} chunks ({config.CHUNK_SIZE_TOKENS}tok / "
          f"{config.CHUNK_OVERLAP_TOKENS}tok overlap).")

    persist_chunks_jsonl(records)
    print(f"Wrote chunks to {config.DATA_DIR / 'kb' / 'chunks.jsonl'}")

    print(f"Embedding into Chroma at {config.CHROMA_DIR} using {config.EMBEDDING_MODEL} ...")
    embed_into_chroma(records)
    print("Ingestion complete.")


if __name__ == "__main__":
    run_ingest()
