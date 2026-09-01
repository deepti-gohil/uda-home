"""
UDA-Hub graph orchestration — built from scratch with LangGraph's
`StateGraph`. No prebuilt agent/supervisor constructor is used; every node
and edge below is explicit. See agentic/design/architecture.md for the full
design rationale and the routing diagram.

Graph shape:

    START -> classifier -> supervisor -(route)-> resolver -(route)-> memory_writer -> END
                                       |-> escalation ----------------^
                            resolver -(low confidence / ineligible action)-> escalation
"""
from __future__ import annotations

import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

import config
from agentic.agents.classifier_agent import classifier_agent
from agentic.agents.escalation_agent import escalation_agent
from agentic.agents.memory_agent import memory_agent
from agentic.agents.resolver_agent import resolver_agent
from agentic.agents.supervisor_agent import route_after_supervisor, supervisor_agent
from agentic.agents.resolver_agent import route_after_resolver
from agentic.state import TicketState
from agentic.tools.account_tools import account_lookup_tool
from data.core.db import get_core_session, init_core_db
from data.core.models import Ticket, TicketMessage
from data.external.db import init_external_db

_compiled_graph = None
_checkpointer_conn: sqlite3.Connection | None = None


def build_graph():
    """Construct and compile the StateGraph. Cached at module level so the
    SQLite checkpointer connection (and the FAISS/embedding clients loaded by
    the agents) are only set up once per process."""
    global _compiled_graph, _checkpointer_conn
    if _compiled_graph is not None:
        return _compiled_graph

    graph = StateGraph(TicketState)
    graph.add_node("classifier", classifier_agent)
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("resolver", resolver_agent)
    graph.add_node("escalation", escalation_agent)
    graph.add_node("memory_writer", memory_agent)

    graph.add_edge(START, "classifier")
    graph.add_edge("classifier", "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_after_supervisor, {"resolver": "resolver", "escalation": "escalation"}
    )
    graph.add_conditional_edges(
        "resolver", route_after_resolver, {"escalation": "escalation", "memory_writer": "memory_writer"}
    )
    graph.add_edge("escalation", "memory_writer")
    graph.add_edge("memory_writer", END)

    _checkpointer_conn = sqlite3.connect(str(config.CHECKPOINT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(_checkpointer_conn)

    _compiled_graph = graph.compile(checkpointer=checkpointer)
    return _compiled_graph


def _resolve_identity(email: str | None) -> tuple[int | None, int | None]:
    """Best-effort lookup of (external_user_id, external_account_id) from an
    email, so the graph can personalize via long-term memory and so the
    resolver's account tools have an id to act on without re-searching."""
    if not email:
        return None, None
    result = account_lookup_tool.invoke({"email": email})
    if "error" in result:
        return None, None
    user = next((u for u in result["users"] if u["email"].lower() == email.lower()), None)
    return (user["user_id"] if user else None), result["account_id"]


def run_ticket(
    subject: str,
    description: str,
    channel: str = "chat",
    metadata: dict | None = None,
    thread_id: str | None = None,
) -> TicketState:
    """Entry point for a new OR continuing support ticket.

    - thread_id=None starts a brand-new ticket: a new Ticket row is created
      in the core DB and a fresh thread_id is generated, which becomes the
      LangGraph checkpoint key for this session's short-term memory.
    - Passing an existing thread_id continues that same session: the graph
      resumes from its last checkpoint (so `messages`, `ticket_id`, and the
      resolved external identity all carry over), and this call's
      subject/description are treated as a new customer message on the same
      ticket.
    """
    init_core_db()
    init_external_db()
    graph = build_graph()
    metadata = metadata or {}

    is_new_session = thread_id is None
    thread_id = thread_id or str(uuid.uuid4())

    state_update: dict = {
        "thread_id": thread_id,
        "channel": channel,
        "subject": subject,
        "description": description,
        "metadata": metadata,
        "needs_escalation": False,
    }

    if is_new_session:
        external_user_id, external_account_id = _resolve_identity(metadata.get("email"))
        with get_core_session() as session:
            ticket = Ticket(
                thread_id=thread_id,
                external_account_id=external_account_id,
                external_user_id=external_user_id,
                channel=channel,
                subject=subject,
                description=description,
            )
            session.add(ticket)
            session.flush()
            ticket_id = ticket.id
            session.add(TicketMessage(ticket_id=ticket_id, sender="customer", content=description))

        state_update.update(
            {
                "ticket_id": ticket_id,
                "external_user_id": external_user_id,
                "external_account_id": external_account_id,
            }
        )
    else:
        # follow-up turn: append the new customer message; ticket_id / identity
        # are restored from the checkpoint automatically by LangGraph.
        current = graph.get_state({"configurable": {"thread_id": thread_id}})
        ticket_id = (current.values or {}).get("ticket_id")
        if ticket_id is not None:
            with get_core_session() as session:
                session.add(TicketMessage(ticket_id=ticket_id, sender="customer", content=description))

    final_state = graph.invoke(state_update, config={"configurable": {"thread_id": thread_id}})
    final_state["thread_id"] = thread_id
    return final_state


def get_session_state(thread_id: str) -> dict:
    """Inspect a session's current checkpointed state (messages, tool_calls_log,
    classification, etc.) — useful for debugging/demoing short-term memory."""
    graph = build_graph()
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values
