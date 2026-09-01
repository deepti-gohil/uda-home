"""
Structured, searchable logging for agent decisions, routing choices, and
tool usage. Every node in the graph calls `log_event()` at least once; rows
land in the core DB's `agent_run_log` table (queryable with plain SQL) and
are also printed so they show up in notebook/console output while the demo
runs.
"""
from __future__ import annotations

from data.core.db import get_core_session
from data.core.models import AgentRunLog


def log_event(
    thread_id: str,
    node_name: str,
    event_type: str,
    detail: dict,
    ticket_id: int | None = None,
) -> None:
    """event_type is one of: 'decision', 'tool_call', 'routing', 'outcome'."""
    print(f"[{node_name}] {event_type}: {detail}")
    with get_core_session() as session:
        session.add(
            AgentRunLog(
                thread_id=thread_id,
                ticket_id=ticket_id,
                node_name=node_name,
                event_type=event_type,
                detail=detail,
            )
        )
