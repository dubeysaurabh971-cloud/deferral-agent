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
        "supporting_excerpts": [1],
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


# --- the citation check: a claim of coverage has to point at something -------------

def test_full_coverage_must_cite_a_supporting_excerpt():
    """An unearned 'full' is the one error nothing downstream can catch, because every other
    rule trusts the coverage label. Claiming full coverage while pointing at no excerpt is a
    claim with no evidence, and it is refused."""
    decision, override = apply_policy(attempt(supporting_excerpts=[]))
    assert decision == "ESCALATE"
    assert "no supporting excerpt" in override


def test_one_cited_excerpt_is_enough():
    assert apply_policy(attempt(supporting_excerpts=[3]))[0] == "RESOLVE"
    assert apply_policy(attempt(supporting_excerpts=[1, 2, 5]))[0] == "RESOLVE"


def test_citation_check_does_not_apply_to_deferrals():
    """It can only ever block a RESOLVE -- a model that already deferred is left alone, so the
    rule cannot turn a CLARIFY into an ESCALATE."""
    for model_decision in ("CLARIFY", "ESCALATE"):
        decision, override = apply_policy(
            attempt(decision=model_decision, supporting_excerpts=[])
        )
        assert decision == model_decision
        assert override is None


def test_citation_check_is_ordered_after_the_coverage_rules():
    """Coverage and authority are the more informative reasons; the override string should name
    them rather than the citation when both apply, so a trace says why it really deferred."""
    _, override = apply_policy(attempt(kb_coverage="none", supporting_excerpts=[]))
    assert "none" in override
    _, override = apply_policy(attempt(requires_human_authority=True, supporting_excerpts=[]))
    assert "requires_human_authority" in override


def test_one_directional_property_survives_the_new_rule():
    """The whole input space, including the citation field: nothing can turn a deferral into a
    RESOLVE, which is what makes the policy layer safe to bolt on."""
    for coverage, authority, cited, model_decision, knob in itertools.product(
        COVERAGE, (True, False), ([], [1]), DECISIONS, (True, False)
    ):
        final, _ = apply_policy(
            attempt(
                kb_coverage=coverage,
                requires_human_authority=authority,
                supporting_excerpts=cited,
                decision=model_decision,
            ),
            escalate_on_partial=knob,
        )
        if model_decision != "RESOLVE":
            assert final == model_decision
        else:
            assert final in {"RESOLVE", "ESCALATE"}
