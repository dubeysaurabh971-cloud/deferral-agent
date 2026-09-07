"""Second-stage review of CLARIFY decisions. NOT ENABLED BY DEFAULT since v5 -- see below.

Why it was built. Finding 3 established that over-clarification did not respond to being told
about itself: a stricter TEST A inside the v3 gate prompt produced *more* clarification, not
less (golden 49% -> 42%). The diagnosis was that TEST A competed with six other paragraphs in
one prompt, and a model asked to weigh coverage, authority, specificity and injection at once
has no reason to weigh any one of them the way you intended. So this became a separate call
with exactly one job: given the excerpts, the ticket, and the question the gate wants to ask,
decide whether that question is *necessary*. Nothing else on the table -- no coverage label,
no escalation, no injection handling. It took false escalations 53 -> 46 for one extra false
resolution, and was what the project shipped.

Why it is now off. v5 fixed the over-clarification in the gate itself, by changing what the
three fields *mean* rather than how firmly the model is urged to weigh them (src/gate.py).
Golden CLARIFY fell from 25 to 1-6 -- and a compensator whose input has almost vanished does
not go neutral, it inverts: nearly every clarification the reviewer now sees is a genuine one,
so its remaining effect is to overturn some of those. Measured, it raises false resolutions
9.3 -> 13.0 at top_k=5 and 12.5 -> 15.0 at top_k=10. The cost curves are below.

The mechanism itself is not disowned. Finding 4's argument -- give a contested judgement its
own call, with nothing to trade against -- is sound, and this file is the evidence for it. What
changed is that the judgement it was isolating no longer needs isolating. Kept, tested, and one
constructor argument away, because a different gate could need it again.

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

A question is NECESSARY only when it asks for a fact ONLY THE CUSTOMER HOLDS, and without
which no useful answer exists. That means: which site, page, product, order, domain, plan,
subscription or team member of theirs is involved; an error message or status they can read
off their screen; or which outcome they want, when the ticket names none at all. A ticket that
is one bare sentence of symptom or goal, naming no instance of anything, needs the question.

A question is NOT NECESSARY -- and this is the common case, the one you are here to catch --
when it asks the customer to do the agent's own work:

  - CHOOSING BETWEEN OUR PROCEDURES. If the excerpts document two or more ways to do what the
    customer plainly wants, that is not a reason to ask which one they want; it is material
    for an answer that gives both, briefly, labelled. The customer wrote in because they do
    not know our procedures. Do not count competing procedures as evidence the question is
    needed -- they are usually evidence it is not.
  - WHICH INTERFACE THEY ARE IN. Which editor (Editor, Editor X, Studio, ADI), desktop or
    mobile app, dashboard or editor. Answer for the main path and note the variant in a line.
  - RUNNING OUR DIAGNOSTICS FOR US. When the customer reports a symptom and the excerpts
    document what to check, sending that checklist IS the answer. Asking them to walk it and
    report back costs a round trip and arrives at the same place.
  - DETAIL THE PROCEDURE NEVER USES. A site name, account id, order number, plan tier,
    browser or OS the documented steps never reference. If you would answer the ticket the
    same way whatever they replied, the question is not necessary.

Being able to imagine a follow-up question is not the same as needing to ask one.

If the question is NOT necessary, write the answer to the ticket yourself, using ONLY the
excerpts and citing them by their [n] marker. Where a detail would genuinely change the
answer but you can cover both branches in a few lines, cover both -- that is an answer, not a
question."""


class ClarificationReview(BaseModel):
    """Reasoning first, then the verdict, then the fallback answer -- same ordering the gate
    and the scorers use, for the same reason."""

    reasoning: str = Field(
        description="State what the question is actually asking the customer for, and whether that "
        "is a fact only they hold or a choice between procedures the excerpts already contain."
    )
    competing_procedures: list[str] = Field(
        default_factory=list,
        description="The materially different procedures the excerpts offer, if more than one. "
        "Recorded for audit: these are usually material for an answer that presents both, NOT "
        "grounds for asking the customer to choose between them.",
    )
    question_is_necessary: bool = Field(
        description="True only if the question asks for a fact that only the customer holds and "
        "without which no useful answer exists."
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


# Whether the reviewer is worth running is not a fact about the reviewer -- it depends on what a
# false resolution costs relative to an unnecessary handoff. Writing C for that ratio, total error
# cost on this 160-item eval set is linear in C.
#
# Under the v3 gate, where these numbers were first taken, the reviewer won for C < 7:
#
#     v3 gate alone           4C + 64
#     v3 + reviewer           5C + 57      <- what this project used to ship
#
# Under v5 it loses, and by a wide margin. Measured at both retrieval settings (means over the
# runs in eval_results/, gate-only vs the same gate with the reviewer on):
#
#     v5 gate, top_k=5         9.3C + 26.0     reviewer on:  13.0C + 26.0
#     v5 gate, top_k=10       12.5C + 18.0     reviewer on:  15.0C + 14.0
#
# Both top_k=10 rows are the build WITHOUT the citation check, which is the only pairing where
# the reviewer was measured on both sides; the credit budget ran out before it could be run
# against the shipped configuration. Deferral errors here include the adversarial items that
# expect RESOLVE and were deferred, which the earlier version of this comment omitted.
#
# At top_k=5 the reviewer is strictly dominated -- 3.7 more false resolutions and no fewer
# deferral errors. At top_k=10 it buys 4 fewer deferral errors for 2.5 more false resolutions,
# which is a win only below C = 1.6; the gate itself only beats the ungated baseline above
# C = 0.38, so the window where the reviewer helps is a slice barely wider than the noise.
#
# The cause is not that the reviewer got worse. It is that the reviewer was a *compensator* for
# the gate's over-clarification: it existed to catch clarifications that should have been
# answers, and v5 stopped producing them. Golden CLARIFY fell from 25 to 1-6, so almost every
# clarification the reviewer now sees is a genuine one, and its only remaining effect is to
# overturn some of those. Fixing a defect at its source does not leave the compensator neutral;
# it inverts it.
#
# Both figures are measured on this eval set, not general constants, and the v5 pair inherits the
# +-3 point run-to-run spread documented in the README.
REVIEW_BREAK_EVEN_C = 1.6

# The v3-era constant, kept so the README's finding 4 stays checkable against the code.
REVIEW_BREAK_EVEN_C_V3 = 7.0


def review_is_worthwhile(cost_ratio: float) -> bool:
    """Does the clarification reviewer lower expected error cost at this cost ratio?

    cost_ratio is C = cost(false resolution) / cost(unnecessary handoff). Under the v5 gate the
    answer is almost always no: see the cost curves above. It stays configurable because the
    mechanism is sound and a different gate could need it again, but it is no longer the default.
    """
    return cost_ratio < REVIEW_BREAK_EVEN_C


#: apply_review outcomes. The two CLARIFY cases are different events and only one of them is
#: the mechanism working as designed:
#:
#:   question_stands  -- the reviewer judged the clarification necessary. Expected, and the
#:                       common case under v5 now that the gate rarely over-clarifies.
#:   no_replacement   -- the reviewer judged it UNNECESSARY but returned no answer to put in
#:                       its place. That is a malfunction, not a verdict: it asked for a flip
#:                       and supplied nothing to flip to. The deferral stands either way, so it
#:                       is invisible in the decision metrics, which is exactly why it needs a
#:                       name -- a reviewer failing this way on every item would look like a
#:                       reviewer that simply never fires.
QUESTION_STANDS = "question_stands"
NO_REPLACEMENT = "no_replacement"
FLIPPED = "flipped"


def apply_review(review: ClarificationReview) -> tuple[str, str | None, str]:
    """Map a review to (decision, answer_override, outcome).

    Refuses to flip without a replacement answer: a RESOLVE whose answer is the clarifying
    question would be worse than the deferral it replaced. `outcome` distinguishes that refusal
    from an ordinary upheld question -- see the constants above.
    """
    if review.question_is_necessary:
        return "CLARIFY", None, QUESTION_STANDS
    if not review.answer or not review.answer.strip():
        return "CLARIFY", None, NO_REPLACEMENT
    return "RESOLVE", review.answer, FLIPPED
