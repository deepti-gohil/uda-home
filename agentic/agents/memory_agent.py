"""
Memory Agent — terminal node. Persists the outcome to the core DB (ticket
status + a new ticket_metadata row + the agent's message), and writes a
resolution summary to long-term memory so future sessions with this customer
have context. This is the node that makes memory durable ACROSS sessions
(as opposed to `messages`, which is short-term/in-session via the
checkpointer).
"""
from __future__ import annotations

from agentic.logging_utils import log_event
from agentic.state import TicketState
from agentic.tools.memory_tools import write_long_term_memory
from data.core.db import get_core_session
from data.core.models import Ticket, TicketMessage, TicketMetadata


def memory_agent(state: TicketState) -> dict:
    status = state.get("status") or ("resolved" if state.get("resolution") else "escalated")
    classification = state.get("classification") or {}
    ticket_id = state.get("ticket_id")

    agent_message = state.get("resolution") or state.get("escalation_summary") or ""

    with get_core_session() as session:
        if ticket_id is not None:
            ticket = session.get(Ticket, ticket_id)
            if ticket is not None:
                ticket.status = status
                if not ticket.ticket_metadata:
                    session.add(
                        TicketMetadata(
                            ticket_id=ticket_id,
                            platform=state.get("channel", "unknown"),
                            category=classification.get("category"),
                            urgency=classification.get("urgency"),
                            sentiment=classification.get("sentiment"),
                            confidence=state.get("confidence"),
                            tags=[classification.get("category")] if classification.get("category") else [],
                        )
                    )
                if agent_message:
                    session.add(TicketMessage(ticket_id=ticket_id, sender="agent", content=agent_message))

    memory_note = None
    user_id = state.get("external_user_id")
    account_id = state.get("external_account_id")
    if (user_id is not None or account_id is not None) and (state.get("resolution") or state.get("escalation_summary")):
        memory_type = "resolution_summary"
        memory_note = (
            f"Ticket #{ticket_id} ({classification.get('category', 'general')}, {status}): "
            f"{classification.get('summary', state.get('subject', ''))}"
        )
        write_long_term_memory.invoke(
            {
                "content": memory_note,
                "memory_type": memory_type,
                "external_user_id": user_id,
                "external_account_id": account_id,
            }
        )

    log_event(
        thread_id=state["thread_id"],
        node_name="memory_writer",
        event_type="outcome",
        detail={"status": status, "ticket_id": ticket_id, "memory_written": memory_note is not None},
        ticket_id=ticket_id,
    )

    # NOTE: we deliberately don't append to `messages` here — the resolver and
    # escalation nodes already added the customer-facing AIMessage for this
    # turn; re-adding it here would duplicate it in the session history.
    return {"status": status}
