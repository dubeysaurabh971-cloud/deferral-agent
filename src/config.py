import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRACES_DIR = ROOT_DIR / "traces"
CHROMA_DIR = ROOT_DIR / "data" / "chroma"

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RESOLVER_MODEL = os.environ.get("RESOLVER_MODEL", "claude-sonnet-5")

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4-fast")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# Providers that speak the OpenAI chat-completions wire format. Each differs only by
# base_url, so they all share one client + one code path in llm_client.
# Maps provider -> (api_key, base_url, model, env var name for the key).
OPENAI_COMPATIBLE = {
    "xai": (XAI_API_KEY, XAI_BASE_URL, XAI_MODEL, "XAI_API_KEY"),
    "gemini": (GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL, "GEMINI_API_KEY"),
    "openai": (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, "OPENAI_API_KEY"),
}

# OpenAI's newer model families (gpt-5*, o*) reject the legacy `max_tokens` field on
# chat.completions and require `max_completion_tokens` instead. The compat endpoints of
# the *other* providers only understand `max_tokens`, so the field name is per-provider
# rather than global. llm_client reads this to pick the right one.
MAX_TOKENS_FIELD = {
    "openai": "max_completion_tokens",
}


def max_tokens_field() -> str:
    return MAX_TOKENS_FIELD.get(LLM_PROVIDER, "max_tokens")


# gpt-5*/o* models spend completion budget on hidden reasoning tokens *before* emitting any
# visible output, and those tokens bill at the output rate. Left at the default, a structured
# call can burn its whole cap reasoning and return "{}" -- observed here at max_tokens=1536.
# "low" keeps enough headroom for the JSON and cuts the expensive half of the bill; this gate
# is a classification task, not one that needs long deliberation.
# Set to empty to omit the parameter entirely, which is required for non-reasoning models
# (gpt-4.1*, gpt-4o*) that reject it.
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")


def extra_params() -> dict:
    """Provider-specific request params that have no equivalent elsewhere."""
    if LLM_PROVIDER == "openai" and OPENAI_REASONING_EFFORT:
        return {"reasoning_effort": OPENAI_REASONING_EFFORT}
    return {}


def active_model() -> str:
    if LLM_PROVIDER in OPENAI_COMPATIBLE:
        return OPENAI_COMPATIBLE[LLM_PROVIDER][2]
    return RESOLVER_MODEL


def api_key_present() -> bool:
    if LLM_PROVIDER in OPENAI_COMPATIBLE:
        return bool(OPENAI_COMPATIBLE[LLM_PROVIDER][0])
    return bool(ANTHROPIC_API_KEY)


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")


def judge_model() -> str:
    """Model the LLM-judge scorers run on.

    Falls back to the resolver model, but set JUDGE_MODEL to a *different* model where you
    can: a judge grading output from its own model exhibits self-preference bias, which
    inflates groundedness and correctness. As a side benefit the two roles then draw on
    separate per-model rate-limit buckets, which matters on quota-capped free tiers.
    """
    return JUDGE_MODEL or active_model()


def api_key_env_name() -> str:
    if LLM_PROVIDER in OPENAI_COMPATIBLE:
        return OPENAI_COMPATIBLE[LLM_PROVIDER][3]
    return "ANTHROPIC_API_KEY"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KB_DATASET = "Wix/WixQA"
KB_CONFIG = "wix_kb_corpus"
KB_COLLECTION_NAME = "wix_kb"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

RETRIEVAL_TOP_K = 5
BM25_WEIGHT = 0.5
DENSE_WEIGHT = 0.5
RRF_K = 60
