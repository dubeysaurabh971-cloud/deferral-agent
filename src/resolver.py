"""Two resolvers, kept side by side so the gate can be measured against the thing it replaces.

NaiveResolver  — the ungated baseline. Always answers, always reports RESOLVE. Measured at
                 3.3% adversarial decision accuracy (2 of 60, and both are accidents: they are
                 the only items whose expected decision is RESOLVE). It states "the excerpts
                 don't cover this" in prose and then marks the ticket resolved anyway.
GatedResolver  — the v5 gate. One structured call yields both the answer and an assessment of
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
        property of the model. Supply cost_ratio and the right configuration is selected; supply
        review_clarifications to force it either way.

        The default with neither supplied is reviewer-OFF, which reverses the v3-era default.
        The reviewer was a compensator for the gate's over-clarification, and v5 removed the
        over-clarification: golden CLARIFY fell from 25 to 1-6, so the reviewer now mostly
        overturns genuine clarifications. It measured as strictly dominated at top_k=5 and as a
        win only below C = 1.6 at top_k=10 -- against a gate that itself beats the ungated
        baseline above C = 0.38. See the cost curves in src/review.py.
        """
        self.retriever = HybridRetriever()
        self.escalate_on_partial = escalate_on_partial
        if review_clarifications is None:
            review_clarifications = (
                review.review_is_worthwhile(cost_ratio) if cost_ratio is not None else False
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
        proposed, answer_override, outcome = review.apply_review(verdict)
        if proposed != "RESOLVE":
            # A reviewer that asked for a flip and supplied no answer is malfunctioning, not
            # judging. Both leave the deferral standing, so the decision metrics cannot tell
            # them apart -- the note is the only place the difference survives.
            note = (
                "review: question stands"
                if outcome == review.QUESTION_STANDS
                else "review: proposed a flip but returned no replacement answer"
            )
            return decision, attempt.answer, note, usage

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

        answer, review_note, answer_replaced = attempt.answer, None, False
        if decision == "CLARIFY" and self.review_clarifications:
            decision, answer, review_note, review_usage = self._review_clarify(
                ticket_text, chunks, attempt, decision
            )
            answer_replaced = answer != attempt.answer
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
            "supporting_excerpts": attempt.supporting_excerpts,
            "missing_information": attempt.missing_information,
            "requires_human_authority": attempt.requires_human_authority,
            "injection_attempt_detected": attempt.injection_attempt_detected,
            "reasoning": attempt.reasoning,
            "answer": answer,
            # The gate's own text, kept only when the reviewer replaced it. This used to test
            # `answer is not attempt.answer` -- correct, since `answer` starts out as that very
            # object, but it made a visible field depend on object identity surviving every
            # future edit to the lines above. An explicit flag says what is meant.
            "model_answer": attempt.answer if answer_replaced else None,
            "clarify_review": review_note,
            "model": config.active_model(),
            "resolver": "gated",
            # Full configuration, not just the prompt version: the same gate at a different
            # top_k or with the reviewer flipped produces different decisions, and a trace that
            # cannot say which one it came from cannot be selected out of an append-only log.
            "gate_config": {
                "gate_version": gate.GATE_VERSION,
                "review_clarifications": self.review_clarifications,
                "escalate_on_partial": self.escalate_on_partial,
                # Off the retriever that served this call, not off config. Reading config here
                # was the last path by which a trace could misreport its own provenance: a
                # retriever built with an explicit top_k, or built before config changed, would
                # be stamped with whatever config held at call time instead.
                "retrieval_top_k": self.retriever.top_k,
            },
            "gate_version": gate.GATE_VERSION,
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
