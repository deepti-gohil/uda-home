"""
Supervisor Agent — deterministic, rule-based routing (deliberately NOT an LLM
call: routing must be auditable and reproducible). See architecture.md
section 7 for the routing table.

Also recalls long-term memory for the customer (if we know who they are) so
downstream nodes can personalize their response.
"""
from __future__ import annotations

from agentic.logging_utils import log_event
from agentic.state import TicketState
from agentic.tools.memory_tools import search_long_term_memory
from config import MAX_MEMORIES_RECALLED

ESCALATE_URGENCY = {"critical"}
ESCALATE_SENTIMENT = {"very_negative"}


def supervisor_agent(state: TicketState) -> dict:
    classification = state.get("classification") or {}

    long_term_context: list[dict] = []
    user_id = state.get("external_user_id")
    account_id = state.get("external_account_id")
    if user_id is not None or account_id is not None:
        recall = search_long_term_memory.invoke(
            {
                "query": classification.get("summary", state.get("description", "")),
                "external_user_id": user_id,
                "external_account_id": account_id,
                "k": MAX_MEMORIES_RECALLED,
            }
        )
        long_term_context = recall.get("results", [])

    should_escalate = (
        classification.get("urgency") in ESCALATE_URGENCY
        or classification.get("sentiment") in ESCALATE_SENTIMENT
        or classification.get("category") == "unknown"
    )
    route = "escalation" if should_escalate else "resolver"

    log_event(
        thread_id=state["thread_id"],
        node_name="supervisor",
        event_type="routing",
        detail={
            "route": route,
            "reason": "urgency/sentiment/category rule" if should_escalate else "default to resolver",
            "classification": classification,
            "memories_recalled": len(long_term_context),
        },
        ticket_id=state.get("ticket_id"),
    )

    return {"route": route, "long_term_context": long_term_context}


def route_after_supervisor(state: TicketState) -> str:
    return state.get("route", "resolver")
