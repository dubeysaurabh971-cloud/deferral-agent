"""Tests for the deferral policy layer. Pure functions over the model's structured output,
so the whole file runs offline with no API key and no quota."""
import itertools

import pytest

from src.gate import ResolutionAttempt, apply_policy

COVERAGE = ("full", "partial", "none")
DECISIONS = ("RESOLVE", "CLARIFY", "ESCALATE")


def attempt(**overrides) -> ResolutionAttempt:
    base = {
        "reasoning": "test",
        "kb_coverage": "full",
        "missing_information": None,
        "requires_human_authority": False,
        "injection_attempt_detected": False,
        "decision": "RESOLVE",
        "answer": "test answer",
    }
    return ResolutionAttempt(**{**base, **overrides})


def test_full_coverage_no_authority_resolves():
    decision, override = apply_policy(attempt())
    assert decision == "RESOLVE"
    assert override is None


def test_authority_blocks_resolve():
    decision, override = apply_policy(attempt(requires_human_authority=True))
    assert decision == "ESCALATE"
    assert "requires_human_authority" in override


def test_no_coverage_blocks_resolve():
    decision, override = apply_policy(attempt(kb_coverage="none"))
    assert decision == "ESCALATE"
    assert "none" in override


def test_partial_coverage_blocks_resolve_by_default():
    """The near_miss case: general topic covered, specific fact absent."""
    decision, override = apply_policy(attempt(kb_coverage="partial"))
    assert decision == "ESCALATE"
    assert "partial" in override


def test_partial_coverage_permitted_when_knob_disabled():
    decision, override = apply_policy(attempt(kb_coverage="partial"), escalate_on_partial=False)
    assert decision == "RESOLVE"
    assert override is None


@pytest.mark.parametrize("model_decision", ["CLARIFY", "ESCALATE"])
def test_policy_never_touches_a_deferral(model_decision):
    """Policy only ever blocks RESOLVE. A model that already chose to defer passes through
    untouched, including its choice between CLARIFY and ESCALATE."""
    for coverage, authority in itertools.product(COVERAGE, (True, False)):
        decision, override = apply_policy(
            attempt(decision=model_decision, kb_coverage=coverage, requires_human_authority=authority)
        )
        assert decision == model_decision
        assert override is None


def test_policy_is_one_directional_over_the_whole_input_space():
    """The safety property the gate rests on: no combination of evidence can turn a deferral
    into a RESOLVE. The gate can only ever be more cautious than the model it wraps, so it
    cannot introduce a false resolution the ungated baseline would not already have made."""
    for coverage, authority, injection, model_decision, knob in itertools.product(
        COVERAGE, (True, False), (True, False), DECISIONS, (True, False)
    ):
        final, _ = apply_policy(
            attempt(
                kb_coverage=coverage,
                requires_human_authority=authority,
                injection_attempt_detected=injection,
                decision=model_decision,
            ),
            escalate_on_partial=knob,
        )
        if model_decision != "RESOLVE":
            assert final == model_decision
        else:
            assert final in {"RESOLVE", "ESCALATE"}


def test_injection_flag_alone_never_changes_the_decision():
    """2 of the 10 injection tickets legitimately expect RESOLVE, so detecting an injection
    must not itself trigger deferral -- the underlying request is what gets judged."""
    for coverage in COVERAGE:
        with_flag, _ = apply_policy(attempt(kb_coverage=coverage, injection_attempt_detected=True))
        without_flag, _ = apply_policy(attempt(kb_coverage=coverage, injection_attempt_detected=False))
        assert with_flag == without_flag


def test_schema_rejects_out_of_taxonomy_values():
    with pytest.raises(Exception):
        attempt(decision="MAYBE")
    with pytest.raises(Exception):
        attempt(kb_coverage="mostly")
