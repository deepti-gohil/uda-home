"""
Central config for UDA-Hub.

Every path here is computed as an ABSOLUTE path anchored to this file's
location (not the process's current working directory), so it doesn't matter
whether you run a notebook from solution/, run `python 03_agentic_app.py`
from repo root, or run pytest from tests/ — the DBs and the FAISS index
always resolve to the same files under solution/data/.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
CORE_DIR = DATA_DIR / "core"
MODELS_DIR = DATA_DIR / "models"

EXTERNAL_DB_PATH = EXTERNAL_DIR / "cultpass.db"
CORE_DB_PATH = CORE_DIR / "udahub.db"
CHECKPOINT_DB_PATH = CORE_DIR / "checkpoints.sqlite"
KNOWLEDGE_INDEX_DIR = MODELS_DIR / "knowledge_index"

for _d in (EXTERNAL_DIR, CORE_DIR, MODELS_DIR, KNOWLEDGE_INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("UDA_HUB_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("UDA_HUB_EMBEDDING_MODEL", "text-embedding-3-small")

CONFIDENCE_THRESHOLD = 0.72
REFUND_WINDOW_DAYS = 30
MAX_MEMORIES_RECALLED = 3
