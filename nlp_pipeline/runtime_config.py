"""
Shared runtime configuration helpers for the NLP pipeline.

This keeps path/model defaults consistent across scripts and avoids repeating
hardcoded local-machine assumptions.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAINING_MASTER_PATH = os.path.join(DATA_DIR, "training_master.jsonl")

DEFAULT_CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db_fixed")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", DEFAULT_CHROMA_PATH)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "legal-search-agent")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
KG_GROQ_MODEL = os.getenv("KG_GROQ_MODEL", "llama-3.1-8b-instant")


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
