"""Thin, provider-agnostic LLM client.

Supports Anthropic (Claude, the plan's default) and xAI (Grok) -- switch via
LLM_PROVIDER in .env. resolver.py and eval/scorers.py go through this module
instead of a provider SDK directly, so swapping providers is a config change,
not a rewrite.
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


def chat(system: str, user: str, max_tokens: int = 1024, model: str | None = None) -> tuple[str, LLMUsage]:
    """Plain-text completion. `model` defaults to the configured resolver model."""
    model = model or config.active_model()
    if config.LLM_PROVIDER in config.OPENAI_COMPATIBLE:
        client = _compat_client()
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = LLMUsage(response.usage.prompt_tokens, response.usage.completion_tokens)
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
) -> BaseModel:
    """Structured-output completion, validated against a Pydantic schema."""
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
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return schema_model.model_validate(data)

    client = _anthropic_client()
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user}],
        output_format=schema_model,
    )
    return response.parsed_output
