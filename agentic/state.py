"""Shared graph state. See architecture.md section 4 for the full design."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TicketState(TypedDict, total=False):
    # identity / session
    thread_id: str
    ticket_id: int | None
    external_account_id: int | None
    external_user_id: int | None

    # input
    channel: str
    subject: str
    description: str
    metadata: dict

    # short-term (in-session) conversational memory
    messages: Annotated[list[BaseMessage], add_messages]

    # classifier output
    classification: dict | None

    # supervisor routing
    route: str | None

    # resolver output
    retrieved_docs: list[dict]
    confidence: float | None
    tool_calls_log: list[dict]
    resolution: str | None
    needs_escalation: bool

    # escalation output
    escalation_summary: str | None

    # long-term memory recalled for this customer
    long_term_context: list[dict]

    # final
    status: str
