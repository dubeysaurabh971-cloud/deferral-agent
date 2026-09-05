"""Thin, provider-agnostic LLM client.

Supports Anthropic (Claude, the plan's default) and the OpenAI-wire providers -- xAI
(Grok), Google (Gemini) and OpenAI itself -- switch via LLM_PROVIDER in .env.
resolver.py and eval/scorers.py go through this module instead of a provider SDK
directly, so swapping providers is a config change, not a rewrite.

Both entry points return (result, LLMUsage). Usage is returned rather than logged here
because the caller owns the trace record -- and on a metered key the token counts are
the only record of what a run cost.
"""
import json
from dataclasses import dataclass

from pydantic import BaseModel

from src import config


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int


def _compat_client():
    """Client for any OpenAI-wire-format provider (xAI, Gemini) -- base_url is the only diff.

    max_retries is well above the SDK default of 2: free-tier Gemini returns transient 503s
    under load, and an eval run makes hundreds of serial calls, so a single unlucky 503
    would otherwise abort the whole report. The SDK backs off exponentially with jitter and
    honours retry-after, so this costs nothing when the provider is healthy.
    """
    from openai import OpenAI

    api_key, base_url, _, _ = config.OPENAI_COMPATIBLE[config.LLM_PROVIDER]
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=8, timeout=120.0)


def _anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _compat_usage(response) -> LLMUsage:
    """Token counts off an OpenAI-wire response.

    Defensive because `usage` is the one field these compat endpoints treat as optional --
    Gemini's in particular has been known to omit it. Zeros keep an eval run alive; the
    alternative is an AttributeError that loses every item scored so far.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage(0, 0)
    return LLMUsage(getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)


class TruncatedCompletion(RuntimeError):
    """Raised when the model hit the completion cap before finishing.

    Worth its own exception because on a reasoning model the symptom is baffling: the call
    succeeds, bills for output tokens, and returns an empty string or "{}" -- the budget went
    on hidden reasoning. Failing loudly here beats a Pydantic 'Field required' traceback that
    points at the schema instead of the cap.
    """


def _check_finish(response) -> None:
    choice = response.choices[0] if getattr(response, "choices", None) else None
    if choice is not None and getattr(choice, "finish_reason", None) == "length":
        raise TruncatedCompletion(
            "Model hit the completion-token cap before producing output. Raise max_tokens, or "
            "lower OPENAI_REASONING_EFFORT if this is a gpt-5*/o* model spending the budget on "
            "reasoning tokens."
        )


def chat(system: str, user: str, max_tokens: int = 1024, model: str | None = None) -> tuple[str, LLMUsage]:
    """Plain-text completion. `model` defaults to the configured resolver model."""
    model = model or config.active_model()
    if config.LLM_PROVIDER in config.OPENAI_COMPATIBLE:
        client = _compat_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **{config.max_tokens_field(): max_tokens},
            **config.extra_params(),
        )
        _check_finish(response)
        text = response.choices[0].message.content or ""
        usage = _compat_usage(response)
        return text, usage

    client = _anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = LLMUsage(response.usage.input_tokens, response.usage.output_tokens)
    return text, usage


def chat_structured(
    user: str, schema_model: type[BaseModel], max_tokens: int = 1024, model: str | None = None
) -> tuple[BaseModel, LLMUsage]:
    """Structured-output completion, validated against a Pydantic schema.

    Returns (parsed, usage) to match chat(). The usage half is what lets the gated resolver
    report what it spent -- without it a metered run is unauditable after the fact.
    """
    model = model or config.active_model()
    if config.LLM_PROVIDER in config.OPENAI_COMPATIBLE:
        client = _compat_client()
        schema_json = schema_model.model_json_schema()
        system = (
            "Respond with ONLY a single valid JSON object matching this JSON schema exactly "
            "-- no markdown fences, no commentary before or after it:\n" + json.dumps(schema_json)
        )
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **{config.max_tokens_field(): max_tokens},
            **config.extra_params(),
        )
        _check_finish(response)
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return schema_model.model_validate(data), _compat_usage(response)

    client = _anthropic_client()
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user}],
        output_format=schema_model,
    )
    return response.parsed_output, LLMUsage(response.usage.input_tokens, response.usage.output_tokens)
