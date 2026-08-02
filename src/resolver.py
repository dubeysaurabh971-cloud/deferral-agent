"""Naive resolver: retrieve top-k chunks, stuff them into a prompt, ask Claude to answer.

No grounding gate yet (that's Week 3) — it always answers, which is the point of the
baseline: it will resolve some tickets and hallucinate on others.
"""
import anthropic

from src import config, trace
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
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def resolve(self, ticket_text: str) -> dict:
        chunks = self.retriever.retrieve(ticket_text)
        prompt = build_prompt(ticket_text, chunks)

        response = self.client.messages.create(
            model=config.RESOLVER_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = next((b.text for b in response.content if b.type == "text"), "")

        record = {
            "ticket_text": ticket_text,
            "retrieved_chunks": [
                {"chunk_id": c["chunk_id"], "title": c["title"], "score": c["score"]}
                for c in chunks
            ],
            "decision": "RESOLVE",
            "answer": answer,
            "model": config.RESOLVER_MODEL,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
        trace.log_trace(record)
        return record


if __name__ == "__main__":
    resolver = NaiveResolver()
    result = resolver.resolve("How do I verify my Wix Payments account?")
    print(result["answer"])
