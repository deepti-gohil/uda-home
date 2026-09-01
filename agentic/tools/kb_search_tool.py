"""
RAG retrieval over the `knowledge` table (core DB).

Builds a FAISS index over every knowledge article the first time it's
needed, persists it under data/models/knowledge_index/, and rebuilds it
automatically if the number of rows in `knowledge` has changed since the
index was built. See agentic/design/architecture.md section 5 for the full
retrieval + confidence-gating design.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import config
from data.core.db import get_core_session
from data.core.models import Knowledge

_METADATA_FILE = config.KNOWLEDGE_INDEX_DIR / "_meta.json"
_vectorstore: FAISS | None = None


def _load_articles() -> list[Knowledge]:
    with get_core_session() as session:
        return session.query(Knowledge).order_by(Knowledge.id).all()


def _build_or_load_index() -> FAISS:
    global _vectorstore

    articles = _load_articles()
    current_meta = {"count": len(articles), "max_id": max((a.id for a in articles), default=0)}

    stale = True
    if _METADATA_FILE.exists():
        saved_meta = json.loads(_METADATA_FILE.read_text())
        stale = saved_meta != current_meta

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)

    if not stale and any(config.KNOWLEDGE_INDEX_DIR.glob("index.faiss")):
        _vectorstore = FAISS.load_local(
            str(config.KNOWLEDGE_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        return _vectorstore

    texts = [f"{a.title}\n\n{a.content}" for a in articles]
    metadatas = [
        {"article_id": a.id, "title": a.title, "category": a.category, "tags": a.tags} for a in articles
    ]
    _vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    _vectorstore.save_local(str(config.KNOWLEDGE_INDEX_DIR))
    _METADATA_FILE.write_text(json.dumps(current_meta))
    return _vectorstore


def _get_index() -> FAISS:
    if _vectorstore is None:
        return _build_or_load_index()
    return _vectorstore


@tool
def kb_search_tool(query: str, k: int = 3) -> dict:
    """Search the CultPass knowledge base for articles relevant to a customer's
    question. Returns the top-k matches with a 0..1 cosine similarity score so
    the caller can gate on confidence before answering. Use this before
    drafting any customer-facing resolution.

    Args:
        query: the customer's question or a short description of their issue.
        k: how many articles to return (default 3).
    """
    if not query or not query.strip():
        return {"error": "empty_query", "results": []}

    index = _get_index()
    hits = index.similarity_search_with_relevance_scores(query, k=k)

    results = [
        {
            "article_id": doc.metadata["article_id"],
            "title": doc.metadata["title"],
            "category": doc.metadata["category"],
            "content": doc.page_content.split("\n\n", 1)[-1],
            "score": round(max(0.0, min(1.0, score)), 4),
        }
        for doc, score in hits
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    top_score = results[0]["score"] if results else 0.0
    return {"query": query, "results": results, "top_score": top_score}
