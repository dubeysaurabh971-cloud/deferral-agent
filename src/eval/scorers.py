"""Eval scorers: groundedness, correctness, decision accuracy.

Groundedness and correctness are LLM-as-judge; both are prompted to produce a
rationale *before* the score/verdict field, since ordering the judge's own reasoning
ahead of the number materially improves judge consistency.
"""
import anthropic
from pydantic import BaseModel, Field

from src import config

JUDGE_MODEL = config.RESOLVER_MODEL


class GroundednessVerdict(BaseModel):
    rationale: str = Field(description="Walk through each claim in the answer and whether a retrieved chunk supports it.")
    grounded: bool = Field(description="True only if every claim in the answer is traceable to a retrieved chunk.")


class CorrectnessVerdict(BaseModel):
    rationale: str = Field(description="Compare the answer to the reference answer before scoring.")
    score: int = Field(description="1-5 rubric: 1=wrong/irrelevant, 3=partially correct, 5=fully correct and complete.")


def score_groundedness(client: anthropic.Anthropic, answer: str, chunks: list[dict]) -> GroundednessVerdict:
    excerpts = "\n\n".join(f"[{i + 1}] {c['title']}\n{c['text']}" for i, c in enumerate(chunks))
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Retrieved excerpts:\n\n{excerpts}\n\n"
                f"Agent's answer:\n{answer}\n\n"
                "Is every factual claim in the agent's answer traceable to one of the retrieved "
                "excerpts above? Judge strictly: an unsupported claim, even a plausible-sounding "
                "one, makes the answer ungrounded."
            ),
        }],
        output_format=GroundednessVerdict,
    )
    return response.parsed_output


def score_correctness(
    client: anthropic.Anthropic, question: str, answer: str, reference_answer: str
) -> CorrectnessVerdict:
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Reference answer:\n{reference_answer}\n\n"
                f"Agent's answer:\n{answer}\n\n"
                "Score the agent's answer against the reference answer on a 1-5 scale: "
                "1=wrong or irrelevant, 3=partially correct or incomplete, 5=fully correct and complete. "
                "The agent's answer doesn't need to match the reference's wording, only its substance."
            ),
        }],
        output_format=CorrectnessVerdict,
    )
    return response.parsed_output


def decision_accuracy(predicted_decision: str, expected_decision: str) -> bool:
    return predicted_decision == expected_decision
