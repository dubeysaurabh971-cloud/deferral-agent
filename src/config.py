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


def active_model() -> str:
    return XAI_MODEL if LLM_PROVIDER == "xai" else RESOLVER_MODEL


def api_key_present() -> bool:
    return bool(XAI_API_KEY) if LLM_PROVIDER == "xai" else bool(ANTHROPIC_API_KEY)

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
