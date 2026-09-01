"""
Shared test fixtures.

Tests run against the SAME dev SQLite files as the rest of the project
(data/external/cultpass.db, data/core/udahub.db) for simplicity — this is a
local class project, not a production system. To avoid mutating the seed
data that 03_agentic_app.py's scripted demo depends on (e.g. refunding
Jordan Blake's demo booking), any test that needs to *write* creates its own
throwaway account/user/booking rather than touching the seeded rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from data.core.db import init_core_db  # noqa: E402
from data.core.seed_knowledge import seed as seed_knowledge  # noqa: E402
from data.external.db import init_external_db  # noqa: E402
from data.external.seed_accounts import seed as seed_accounts  # noqa: E402

requires_openai_key = pytest.mark.skipif(
    not config.OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set — skipping tests that call OpenAI (embeddings/chat).",
)


@pytest.fixture(scope="session", autouse=True)
def _seeded_databases():
    init_external_db()
    init_core_db()
    seed_accounts()
    seed_knowledge()
    yield
