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


def _xai_client():
    from openai import OpenAI

    return OpenAI(api_key=config.XAI_API_KEY, base_url=config.XAI_BASE_URL)


def _anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def chat(system: str, user: str, max_tokens: int = 1024) -> tuple[str, LLMUsage]:
    """Plain-text completion."""
    if config.LLM_PROVIDER == "xai":
        client = _xai_client()
        response = client.chat.completions.create(
            model=config.active_model(),
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
        model=config.active_model(),
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = LLMUsage(response.usage.input_tokens, response.usage.output_tokens)
    return text, usage


def chat_structured(user: str, schema_model: type[BaseModel], max_tokens: int = 1024) -> BaseModel:
    """Structured-output completion, validated against a Pydantic schema."""
    if config.LLM_PROVIDER == "xai":
        client = _xai_client()
        schema_json = schema_model.model_json_schema()
        system = (
            "Respond with ONLY a single valid JSON object matching this JSON schema exactly "
            "-- no markdown fences, no commentary before or after it:\n" + json.dumps(schema_json)
        )
        response = client.chat.completions.create(
            model=config.active_model(),
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
        model=config.active_model(),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user}],
        output_format=schema_model,
    )
    return response.parsed_output
