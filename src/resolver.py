"""Naive resolver: retrieve top-k chunks, stuff them into a prompt, ask Claude to answer.

No grounding gate yet (that's Week 3) — it always answers, which is the point of the
baseline: it will resolve some tickets and hallucinate on others.
"""
from src import config, llm_client, trace
from src.retrieval import HybridRetriever

SYSTEM_PROMPT = """You are a support agent for Wix, a website-building platform.
Answer the customer's question using ONLY the retrieved knowledge-base excerpts below.
Cite the excerpt(s) you used by their [n] marker. If the excerpts don't fully cover the
question, answer as best you can from what's there anyway."""


def build_prompt(ticket_text: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{i + 1}] ({chunk['title']})\n{chunk['text']}" for i, chunk in enumerate(chunks)
    )
    return (
        f"Retrieved excerpts:\n\n{excerpts}\n\n"
        f"Customer question:\n{ticket_text}\n\n"
        "Answer the customer's question, citing excerpts by [n]."
    )


class NaiveResolver:
    def __init__(self):
        self.retriever = HybridRetriever()

    def resolve(self, ticket_text: str) -> dict:
        chunks = self.retriever.retrieve(ticket_text)
        prompt = build_prompt(ticket_text, chunks)

        answer, usage = llm_client.chat(SYSTEM_PROMPT, prompt, max_tokens=1024)

        record = {
            "ticket_text": ticket_text,
            "retrieved_chunks": [
                {"chunk_id": c["chunk_id"], "title": c["title"], "score": c["score"]}
                for c in chunks
            ],
            "decision": "RESOLVE",
            "answer": answer,
            "model": config.active_model(),
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        }
        trace.log_trace(record)

        # The trace record keeps only chunk metadata -- full chunk text would add ~10k chars
        # per line to traces/runs.jsonl. But the groundedness judge needs the actual text to
        # check claims against, so hand the caller the unabridged chunks in-memory.
        return {**record, "retrieved_chunks": chunks}


if __name__ == "__main__":
    resolver = NaiveResolver()
    result = resolver.resolve("How do I verify my Wix Payments account?")
    print(result["answer"])
