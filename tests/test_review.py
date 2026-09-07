"""Tests for the clarification reviewer and its guards.

The reviewer is the one component that can *undo* a deferral, so the invariants worth pinning
down are the ones that stop it doing so for the wrong reason. Each guard here corresponds to a
measured failure: without the policy re-check the replay produced 5 extra false resolutions
instead of 1, and injection tickets were among the flips that went wrong.
"""
import pytest

from src import gate, review
from src.resolver import GatedResolver


def attempt(**kw):
    base = dict(
        reasoning="r",
        supporting_excerpts=[1],
        kb_coverage="full",
        missing_information="which site",
        requires_human_authority=False,
        injection_attempt_detected=False,
        decision="CLARIFY",
        answer="Which site do you mean?",
    )
    base.update(kw)
    return gate.ResolutionAttempt(**base)


def verdict(necessary, answer="Go to Settings > Domains. [1]"):
    return review.ClarificationReview(
        reasoning="r", competing_procedures=[], question_is_necessary=necessary, answer=answer
    )


class StubResolver(GatedResolver):
    """GatedResolver without the retriever/network, so the guards can be tested in isolation."""

    def __init__(self, verdict_to_return, **kw):
        self.escalate_on_partial = kw.get("escalate_on_partial", True)
        self.review_clarifications = True
        self._verdict = verdict_to_return
        self.review_called = False

    def _call_review(self, *a):
        self.review_called = True
        return self._verdict, None


@pytest.fixture
def patched(monkeypatch):
    def _make(v, **kw):
        r = StubResolver(v, **kw)
        monkeypatch.setattr(
            review, "review_clarification", lambda *a, **k: (r._verdict, _Usage())
        )
        return r

    return _make


class _Usage:
    input_tokens = 10
    output_tokens = 5


# --- apply_review -----------------------------------------------------------------

def test_necessary_question_stands():
    decision, _, outcome = review.apply_review(verdict(True))
    assert decision == "CLARIFY"
    assert outcome == review.QUESTION_STANDS


def test_flip_requires_a_replacement_answer():
    """A RESOLVE whose text is still the clarifying question is worse than the deferral."""
    for empty in (None, "   "):
        decision, _, outcome = review.apply_review(verdict(False, answer=empty))
        assert decision == "CLARIFY"
        assert outcome == review.NO_REPLACEMENT, (
            "a refused flip must be distinguishable from an upheld question: both leave the "
            "deferral standing, so the outcome tag is the only signal a reviewer is malfunctioning"
        )


def test_flip_allowed_with_an_answer():
    decision, answer, outcome = review.apply_review(verdict(False))
    assert decision == "RESOLVE"
    assert "Settings" in answer
    assert outcome == review.FLIPPED


# --- the guards in GatedResolver --------------------------------------------------

def test_injection_ticket_is_never_reviewed(patched):
    r = patched(verdict(False))
    decision, answer, note, usage = r._review_clarify(
        "t", [], attempt(injection_attempt_detected=True), "CLARIFY"
    )
    assert decision == "CLARIFY"
    assert "injection" in note
    assert usage is None, "no review call should be made, so no tokens should be spent"


def test_partial_coverage_flip_is_overruled_by_policy(patched):
    """apply_policy forbids resolving on partial coverage; the reviewer may not bypass it."""
    r = patched(verdict(False))
    decision, answer, note, _ = r._review_clarify("t", [], attempt(kb_coverage="partial"), "CLARIFY")
    assert decision == "CLARIFY"
    assert "overruled by policy" in note


def test_human_authority_flip_is_overruled_by_policy(patched):
    r = patched(verdict(False))
    decision, _, note, _ = r._review_clarify(
        "t", [], attempt(requires_human_authority=True), "CLARIFY"
    )
    assert decision == "CLARIFY"
    assert "overruled by policy" in note


def test_full_coverage_clean_ticket_flips(patched):
    r = patched(verdict(False))
    decision, answer, note, _ = r._review_clarify("t", [], attempt(), "CLARIFY")
    assert decision == "RESOLVE"
    assert "Settings" in answer
    assert "unnecessary" in note


def test_reviewer_is_one_directional(patched):
    """Whatever the reviewer says, it can never produce a deferral that wasn't already there."""
    for v in (verdict(True), verdict(False), verdict(False, answer=None)):
        r = patched(v)
        decision, _, _, _ = r._review_clarify("t", [], attempt(), "CLARIFY")
        assert decision in ("CLARIFY", "RESOLVE"), "must never invent an ESCALATE"


def test_review_can_be_disabled():
    r = GatedResolver.__new__(GatedResolver)
    r.review_clarifications = False
    assert r.review_clarifications is False


# --- configuration is an economics question, not a model property ------------------

def test_break_even_matches_the_measured_cost_curves():
    """v5 gate at top_k=10 (no citation check) = 12.5C+18, same gate with the reviewer =
    15C+14; equal at C=1.6.

    Means over the runs in eval_results/, with deferral errors counting the adversarial
    resolve-expecting items too. Both sides are the pre-citation-check build because that is
    the only top_k=10 pairing where the reviewer was measured on both sides. The v3-era pair is
    kept as well, because the README's finding 4 is stated in those numbers and should stay
    checkable against the code."""
    C = review.REVIEW_BREAK_EVEN_C
    assert 12.5 * C + 18.0 == pytest.approx(15.0 * C + 14.0)

    C_v3 = review.REVIEW_BREAK_EVEN_C_V3
    assert 4 * C_v3 + 64 == pytest.approx(5 * C_v3 + 57)


def test_the_reviewer_lost_its_case_between_v3_and_v5():
    """Not a tautology: it pins the direction of the finding. The reviewer went from winning
    below C=7 to winning only below C=1.6, because v5 removed the over-clarification it existed
    to compensate for. If someone re-tunes it upward, this should be a deliberate act."""
    assert review.REVIEW_BREAK_EVEN_C < review.REVIEW_BREAK_EVEN_C_V3


@pytest.mark.parametrize(
    "c,expected", [(1.0, True), (1.5, True), (1.7, False), (3.0, False), (10.0, False)]
)
def test_review_is_worthwhile(c, expected):
    assert review.review_is_worthwhile(c) is expected


@pytest.mark.parametrize(
    "kwargs,want",
    [
        ({"cost_ratio": 1.2}, True),                              # very cheap bad answers -> review
        ({"cost_ratio": 3.0}, False),                             # anything dearer -> don't
        ({"cost_ratio": 10.0}, False),
        ({}, False),                          # v5 default: reviewer off, see src/review.py
        ({"cost_ratio": 10.0, "review_clarifications": True}, True),   # explicit flag wins
        ({"cost_ratio": 1.0, "review_clarifications": False}, False),  # explicit flag wins
    ],
)
def test_cost_ratio_selects_the_configuration(monkeypatch, kwargs, want):
    """Exercises the real constructor, with the retriever stubbed out."""
    import src.resolver as resolver_mod

    monkeypatch.setattr(resolver_mod, "HybridRetriever", lambda: object())
    assert GatedResolver(**kwargs).review_clarifications is want
