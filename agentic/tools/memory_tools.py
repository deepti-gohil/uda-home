"""
Long-term memory tools (core DB, `long_term_memory` table).

Long-term memory persists ACROSS sessions/threads — unlike the LangGraph
checkpointer, which only remembers state within one thread_id. Each row
stores an OpenAI embedding (JSON-encoded) of its own content; search does a
plain cosine-similarity scan in NumPy over one customer's rows, which is
plenty fast at this scale (a handful to a few hundred memories per
customer) without needing a dedicated vector DB. See architecture.md
section 4.
"""
from __future__ import annotations

import json

import numpy as np
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

import config
from data.core.db import get_core_session
from data.core.models import LongTermMemory

_embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)


def _embed(text: str) -> list[float]:
    return _embeddings.embed_query(text)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


@tool
def write_long_term_memory(
    content: str,
    memory_type: str,
    external_user_id: int | None = None,
    external_account_id: int | None = None,
) -> dict:
    """Persist a durable memory about a customer that should be recalled in
    FUTURE, unrelated sessions — e.g. a stated preference ("prefers email over
    phone") or a one-line resolution summary ("2026-08-02: refunded a
    duplicate booking charge"). Do not use this for anything scoped to just
    the current conversation.

    Args:
        content: the memory text to store.
        memory_type: "preference" or "resolution_summary".
        external_user_id: the CultPass user id this memory is about, if known.
        external_account_id: the CultPass account id this memory is about, if known.
    """
    if memory_type not in ("preference", "resolution_summary"):
        return {"error": "invalid_memory_type", "reason": "Must be 'preference' or 'resolution_summary'."}
    if not content or not content.strip():
        return {"error": "missing_input", "reason": "content is required."}

    embedding = _embed(content)
    with get_core_session() as session:
        row = LongTermMemory(
            external_user_id=external_user_id,
            external_account_id=external_account_id,
            memory_type=memory_type,
            content=content,
            embedding=json.dumps(embedding),
        )
        session.add(row)
        session.flush()
        return {"status": "stored", "memory_id": row.id}


@tool
def search_long_term_memory(
    query: str,
    external_user_id: int | None = None,
    external_account_id: int | None = None,
    k: int = 3,
) -> dict:
    """Semantically search a customer's long-term memory (past preferences and
    resolution summaries) for entries relevant to the current ticket. Scope by
    external_user_id and/or external_account_id — always pass at least one, or
    you'll search across every customer.

    Args:
        query: what to search for (e.g. the current ticket's summary).
        external_user_id: restrict to this CultPass user id, if known.
        external_account_id: restrict to this CultPass account id, if known.
        k: how many memories to return (default 3).
    """
    if external_user_id is None and external_account_id is None:
        return {"error": "missing_scope", "reason": "Provide external_user_id and/or external_account_id."}

    with get_core_session() as session:
        q = session.query(LongTermMemory)
        if external_user_id is not None:
            q = q.filter(LongTermMemory.external_user_id == external_user_id)
        if external_account_id is not None:
            q = q.filter(LongTermMemory.external_account_id == external_account_id)
        rows = q.all()

    if not rows:
        return {"results": []}

    query_vec = np.array(_embed(query))
    scored = [
        (row, _cosine(query_vec, np.array(json.loads(row.embedding))))
        for row in rows
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return {
        "results": [
            {
                "memory_id": row.id,
                "memory_type": row.memory_type,
                "content": row.content,
                "score": round(score, 4),
                "created_at": row.created_at.isoformat(),
            }
            for row, score in scored[:k]
        ]
    }
