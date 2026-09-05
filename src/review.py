"""Second-stage review of CLARIFY decisions -- the mechanism fix for over-deferral.

Finding 3 established that over-clarification does not respond to being told about itself: a
stricter TEST A inside the gate prompt produced *more* clarification, not less (golden 49% ->
42%). The diagnosis was that TEST A competes with six other paragraphs in one prompt, and a
model asked to weigh coverage, authority, specificity and injection at once has no reason to
weigh any one of them the way you intended.

So this is a separate call with exactly one job: given the excerpts, the ticket, and the
question the gate wants to ask, decide whether that question is *necessary*. Nothing else is
on the table -- no coverage label, no escalation, no injection handling.

Two properties make it safe to bolt on, mirroring apply_policy:

  - It is one-directional. It can only turn CLARIFY into RESOLVE. It never creates a deferral,
    so it cannot make false escalation worse, and the only risk it carries is false
    resolutions -- which the eval measures directly.
  - It only fires on CLARIFY, so it costs one extra call on the subset of tickets where the
    error lives, not on every ticket.
"""
from pydantic import BaseModel, Field

from src import config, llm_client

REVIEW_PROMPT = """You are reviewing a support agent's decision to ask the customer a
clarifying question instead of answering their ticket.

You will be given the knowledge-base excerpts the agent retrieved, the customer's ticket, and
the question the agent wants to ask.

Your only job is to decide whether that question is NECESSARY. Do not judge anything else --
not whether a human should handle it, not whether the excerpts are good, not tone.

The question is NECESSARY only if the excerpts contain two or more materially different
procedures or answers, and which one applies genuinely depends on the customer's reply. If so,
name the competing procedures explicitly in your reasoning. Cancelling a site plan, an app and
a domain renewal really are three different procedures -- that is a necessary question.

The question is NOT NECESSARY when the excerpts contain a single procedure that answers the
ticket as asked. This is the common case and the one to watch for:

  - A generally-phrased question usually has one documented answer. "How do I add a custom
    domain" does not need to know which site.
  - Do not require a site name, account id, order number, plan tier, browser or operating
    system that the documented procedure never actually references.
  - If you would answer the ticket the same way regardless of what the customer replied, the
    question is not necessary.
  - Being able to imagine a follow-up question is not the same as needing to ask one. Asking
    when you did not have to costs the customer a round trip for nothing.

If the question is NOT necessary, write the answer to the ticket yourself, using ONLY the
excerpts and citing them by their [n] marker."""


class ClarificationReview(BaseModel):
    """Reasoning first, then the verdict, then the fallback answer -- same ordering the gate
    and the scorers use, for the same reason."""

    reasoning: str = Field(
        description="Name the competing procedures in the excerpts if there are any. If there is "
        "only one procedure that answers the ticket as asked, say so."
    )
    competing_procedures: list[str] = Field(
        default_factory=list,
        description="The materially different procedures the excerpts offer, if more than one. "
        "Empty when a single procedure answers the ticket.",
    )
    question_is_necessary: bool = Field(
        description="True only if the customer's reply would actually change which answer they get."
    )
    answer: str | None = Field(
        default=None,
        description="When question_is_necessary is false, the answer to the ticket drawn from the "
        "excerpts with [n] citations. Null when the question stands.",
    )


def review_clarification(
    ticket_text: str, chunks: list[dict], question_asked: str
) -> tuple[ClarificationReview, llm_client.LLMUsage]:
    excerpts = "\n\n".join(f"[{i + 1}] ({c['title']})\n{c['text']}" for i, c in enumerate(chunks))
    prompt = (
        f"{REVIEW_PROMPT}\n\n"
        f"Retrieved excerpts:\n\n{excerpts}\n\n"
        f"Customer ticket:\n{ticket_text}\n\n"
        f"The question the agent wants to ask:\n{question_asked}"
    )
    return llm_client.chat_structured(prompt, ClarificationReview, max_tokens=4096)


def apply_review(review: ClarificationReview) -> tuple[str, str | None]:
    """Map a review to (decision, answer_override).

    Refuses to flip without a replacement answer: a RESOLVE whose answer is the clarifying
    question would be worse than the deferral it replaced.
    """
    if review.question_is_necessary:
        return "CLARIFY", None
    if not review.answer or not review.answer.strip():
        return "CLARIFY", None
    return "RESOLVE", review.answer
