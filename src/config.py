import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRACES_DIR = ROOT_DIR / "traces"
CHROMA_DIR = ROOT_DIR / "data" / "chroma"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RESOLVER_MODEL = os.environ.get("RESOLVER_MODEL", "claude-sonnet-5")

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
