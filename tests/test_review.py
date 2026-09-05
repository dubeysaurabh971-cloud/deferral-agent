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
    assert review.apply_review(verdict(True))[0] == "CLARIFY"


def test_flip_requires_a_replacement_answer():
    """A RESOLVE whose text is still the clarifying question is worse than the deferral."""
    assert review.apply_review(verdict(False, answer=None))[0] == "CLARIFY"
    assert review.apply_review(verdict(False, answer="   "))[0] == "CLARIFY"


def test_flip_allowed_with_an_answer():
    decision, answer = review.apply_review(verdict(False))
    assert decision == "RESOLVE"
    assert "Settings" in answer


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
    """gate alone = 4C+64, governed reviewer = 5C+57; equal at C=7."""
    C = review.REVIEW_BREAK_EVEN_C
    assert 4 * C + 64 == pytest.approx(5 * C + 57)


@pytest.mark.parametrize("c,expected", [(1.0, True), (3.0, True), (6.9, True), (7.1, False), (10.0, False)])
def test_review_is_worthwhile(c, expected):
    assert review.review_is_worthwhile(c) is expected


@pytest.mark.parametrize(
    "kwargs,want",
    [
        ({"cost_ratio": 3.0}, True),                              # cheap bad answers -> review
        ({"cost_ratio": 10.0}, False),                            # expensive bad answers -> don't
        ({}, True),                                               # default assumes C < 7
        ({"cost_ratio": 10.0, "review_clarifications": True}, True),   # explicit flag wins
        ({"cost_ratio": 1.0, "review_clarifications": False}, False),  # explicit flag wins
    ],
)
def test_cost_ratio_selects_the_configuration(monkeypatch, kwargs, want):
    """Exercises the real constructor, with the retriever stubbed out."""
    import src.resolver as resolver_mod

    monkeypatch.setattr(resolver_mod, "HybridRetriever", lambda: object())
    assert GatedResolver(**kwargs).review_clarifications is want
