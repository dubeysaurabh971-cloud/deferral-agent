"""Two resolvers, kept side by side so the gate can be measured against the thing it replaces.

NaiveResolver  — Week 1 baseline. Always answers, always reports RESOLVE. Measured at 0/4
                 adversarial decision accuracy: it states "the excerpts don't cover this" in
                 prose and then marks the ticket resolved anyway.
GatedResolver  — Week 3. One structured call yields both the answer and an assessment of
                 coverage, missing detail, and whether a human must act; src/gate.py maps
                 that to the decision. Same number of LLM calls as the baseline.
"""
from src import config, gate, llm_client, trace
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


class GatedResolver:
    """Retrieve, then make one structured call that answers *and* self-assesses.

    The assessment is not a second opinion bolted on after the fact -- it is produced in the
    same pass as the answer, so the model cannot write "the excerpts don't cover this" while
    a separate code path reports RESOLVE. That split is exactly what the baseline got wrong.
    """

    def __init__(self, escalate_on_partial: bool = True):
        self.retriever = HybridRetriever()
        self.escalate_on_partial = escalate_on_partial

    def resolve(self, ticket_text: str) -> dict:
        chunks = self.retriever.retrieve(ticket_text)
        prompt = (
            f"{gate.SYSTEM_PROMPT}\n\n"
            f"Retrieved excerpts:\n\n"
            + "\n\n".join(f"[{i + 1}] ({c['title']})\n{c['text']}" for i, c in enumerate(chunks))
            + f"\n\nCustomer ticket:\n{ticket_text}"
        )
        attempt = llm_client.chat_structured(prompt, gate.ResolutionAttempt, max_tokens=1536)
        decision, override_reason = gate.apply_policy(attempt, self.escalate_on_partial)

        record = {
            "ticket_text": ticket_text,
            "retrieved_chunks": [
                {"chunk_id": c["chunk_id"], "title": c["title"], "score": c["score"]}
                for c in chunks
            ],
            "decision": decision,
            "model_decision": attempt.decision,
            "policy_override": override_reason,
            "kb_coverage": attempt.kb_coverage,
            "missing_information": attempt.missing_information,
            "requires_human_authority": attempt.requires_human_authority,
            "injection_attempt_detected": attempt.injection_attempt_detected,
            "reasoning": attempt.reasoning,
            "answer": attempt.answer,
            "model": config.active_model(),
            "resolver": "gated",
            # This call returns a parsed object rather than a usage object; token accounting
            # for the structured path is not wired up yet.
            "usage": {"input_tokens": None, "output_tokens": None},
        }
        trace.log_trace(record)
        return {**record, "retrieved_chunks": chunks}


if __name__ == "__main__":
    resolver = GatedResolver()
    result = resolver.resolve("How do I verify my Wix Payments account?")
    print(f"[{result['decision']}] {result['answer']}")
