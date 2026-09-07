"""Hybrid retrieval over the ingested KB: BM25 (keyword) + dense (embedding), fused with
Reciprocal Rank Fusion (RRF). Hybrid beats dense-only on support docs full of product names
and error codes that embeddings tend to blur together.
"""
import heapq
import json
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from src import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    def __init__(self, candidate_pool: int = 20, top_k: int | None = None):
        """top_k defaults to config.RETRIEVAL_TOP_K, resolved HERE rather than in the signature
        of retrieve().

        It used to be `def retrieve(self, query, top_k=config.RETRIEVAL_TOP_K)`, which binds the
        default once at import time. Same value in a normal run, so it was never a live bug --
        but anything that set config.RETRIEVAL_TOP_K after import (a test, a sweep script) got
        retrieval quietly using the old value while resolver.py stamped the NEW one into
        gate_config, because that field is read at call time. gate_config exists precisely so a
        trace cannot lie about its own provenance, and this was the one path where it still
        could. Binding it to the instance closes that and gives callers something truthful to
        record: retriever.top_k is what retrieval actually used.
        """
        self.candidate_pool = candidate_pool
        self.top_k = config.RETRIEVAL_TOP_K if top_k is None else top_k

        chunks_path = config.DATA_DIR / "kb" / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"{chunks_path} not found. Run `python run.py --ingest` first."
            )
        self.chunks = [json.loads(line) for line in chunks_path.open(encoding="utf-8")]
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}

        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
        self._collection = client.get_collection(
            name=config.KB_COLLECTION_NAME, embedding_function=embedding_fn
        )

    def _bm25_ranking(self, query: str) -> list[str]:
        # nlargest rather than a full sort: this ranks 10,068 chunks to keep 20, and it runs
        # once per ticket alongside the dense query. Same result, O(n log k) instead of
        # O(n log n).
        scores = self._bm25.get_scores(_tokenize(query))
        top = heapq.nlargest(self.candidate_pool, range(len(scores)), key=scores.__getitem__)
        return [self.chunks[i]["chunk_id"] for i in top]

    def _dense_ranking(self, query: str) -> list[str]:
        result = self._collection.query(query_texts=[query], n_results=self.candidate_pool)
        return result["ids"][0]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """top_k=None uses the value bound at construction. See __init__ for why the default is
        not in this signature."""
        top_k = self.top_k if top_k is None else top_k
        bm25_ids = self._bm25_ranking(query)
        dense_ids = self._dense_ranking(query)

        rrf_scores: dict[str, float] = {}
        for rank, chunk_id in enumerate(bm25_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + config.BM25_WEIGHT / (
                config.RRF_K + rank + 1
            )
        for rank, chunk_id in enumerate(dense_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + config.DENSE_WEIGHT / (
                config.RRF_K + rank + 1
            )

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results = []
        for chunk_id, score in ranked:
            chunk = self.chunk_by_id[chunk_id]
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "score": score,
                    "in_bm25_top": chunk_id in bm25_ids,
                    "in_dense_top": chunk_id in dense_ids,
                }
            )
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    hits = retriever.retrieve("How do I verify my Wix Payments account?")
    for hit in hits:
        print(f"{hit['score']:.4f}  {hit['title']}  ({hit['chunk_id']})")
