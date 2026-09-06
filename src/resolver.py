"""Two resolvers, kept side by side so the gate can be measured against the thing it replaces.

NaiveResolver  — Week 1 baseline. Always answers, always reports RESOLVE. Measured at 0/4
                 adversarial decision accuracy: it states "the excerpts don't cover this" in
                 prose and then marks the ticket resolved anyway.
GatedResolver  — Week 3. One structured call yields both the answer and an assessment of
                 coverage, missing detail, and whether a human must act; src/gate.py maps
                 that to the decision. Same number of LLM calls as the baseline.
"""
from src import config, gate, llm_client, review, trace
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

    def __init__(
        self,
        escalate_on_partial: bool = True,
        review_clarifications: bool | None = None,
        cost_ratio: float | None = None,
    ):
        """cost_ratio is C = cost(false resolution) / cost(unnecessary handoff).

        Whether to run the clarification reviewer is an operator's economics question, not a
        property of the model: it trades one extra false resolution for seven fewer false
        escalations, which is a win below C = 7 and a loss above it. Supply cost_ratio and the
        right configuration is selected; supply review_clarifications to force it either way.

        The default with neither supplied is reviewer-on, which assumes C < 7. An organisation
        that treats a confidently wrong answer as ten times worse than a needless handoff should
        pass cost_ratio=10 (or review_clarifications=False) and run the gate alone.
        """
        self.retriever = HybridRetriever()
        self.escalate_on_partial = escalate_on_partial
        if review_clarifications is None:
            review_clarifications = (
                True
            )
        self.review_clarifications = review_clarifications
        self.cost_ratio = cost_ratio

    def _review_clarify(self, ticket_text, chunks, attempt, decision):
        """Second-stage review of a CLARIFY. Returns (decision, answer, review_note, usage).

        The reviewer may only turn CLARIFY into RESOLVE, and only where the gate's own policy
        would have permitted a RESOLVE in the first place -- so the flip is put back through
        apply_policy rather than trusted on its own. Without that, the reviewer becomes a way
        to launder a resolution past invariants the system already declared, which measured as
        5 extra false resolutions against 1 with the check in place.

        Injection is the one extra guard: a ticket carrying an active manipulation attempt is
        the last place to relax caution, whatever the reviewer concludes.
        """
        if attempt.injection_attempt_detected:
            return decision, attempt.answer, "review skipped: injection attempt detected", None

        verdict, usage = review.review_clarification(ticket_text, chunks, attempt.answer)
        proposed, answer_override = review.apply_review(verdict)
        if proposed != "RESOLVE":
            return decision, attempt.answer, "review: question stands", usage

        # Would the policy layer have allowed this resolution? If not, the deferral stands.
        probe = attempt.model_copy(update={"decision": "RESOLVE"})
        permitted, blocked_reason = gate.apply_policy(probe, self.escalate_on_partial)
        if permitted != "RESOLVE":
            return decision, attempt.answer, f"review overruled by policy: {blocked_reason}", usage

        return "RESOLVE", answer_override, "review: clarification was unnecessary", usage

    def resolve(self, ticket_text: str) -> dict:
        chunks = self.retriever.retrieve(ticket_text)
        prompt = (
            f"{gate.SYSTEM_PROMPT}\n\n"
            f"Retrieved excerpts:\n\n"
            + "\n\n".join(f"[{i + 1}] ({c['title']})\n{c['text']}" for i, c in enumerate(chunks))
            + f"\n\nCustomer ticket:\n{ticket_text}"
        )
        attempt, usage = llm_client.chat_structured(prompt, gate.ResolutionAttempt, max_tokens=4096)
        decision, override_reason = gate.apply_policy(attempt, self.escalate_on_partial)

        answer, review_note = attempt.answer, None
        if decision == "CLARIFY" and self.review_clarifications:
            decision, answer, review_note, review_usage = self._review_clarify(
                ticket_text, chunks, attempt, decision
            )
            if review_usage is not None:
                usage = llm_client.LLMUsage(
                    usage.input_tokens + review_usage.input_tokens,
                    usage.output_tokens + review_usage.output_tokens,
                )

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
            "answer": answer,
            "model_answer": attempt.answer if answer is not attempt.answer else None,
            "clarify_review": review_note,
            "model": config.active_model(),
            "resolver": "gated",
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        }
        trace.log_trace(record)
        return {**record, "retrieved_chunks": chunks}


if __name__ == "__main__":
    resolver = GatedResolver()
    result = resolver.resolve("How do I verify my Wix Payments account?")
    print(f"[{result['decision']}] {result['answer']}")
