"""Classifier Agent — the only node that reads the customer's raw text."""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from agentic.logging_utils import log_event
from agentic.state import TicketState

class Classification(BaseModel):
    category: Literal[
        "billing", "account", "subscription", "bookings", "technical",
        "notifications", "privacy", "referrals", "accessibility", "support", "unknown",
    ] = Field(description="Best-matching topic for this ticket. Use 'unknown' only if truly unclear.")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="critical = customer is blocked right now (e.g. can't get into a venue today, "
        "locked out of account with an urgent need); high = time-sensitive but not blocking; "
        "medium/low = can wait for normal queue handling."
    )
    sentiment: Literal["positive", "neutral", "negative", "very_negative"] = Field(
        description="Customer's emotional tone. very_negative = angry/threatening to leave/repeated complaint."
    )
    summary: str = Field(description="One sentence summarizing the customer's issue, for hand-off/logging.")


_llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)
_structured_llm = _llm.with_structured_output(Classification)


def classifier_agent(state: TicketState) -> dict:
    prompt = (
        "Classify this CultPass customer support ticket.\n\n"
        f"Channel: {state.get('channel', 'unknown')}\n"
        f"Subject: {state.get('subject', '')}\n"
        f"Description: {state.get('description', '')}\n"
        f"Caller-supplied metadata: {state.get('metadata', {})}\n"
    )
    result: Classification = _structured_llm.invoke([HumanMessage(content=prompt)])
    classification = result.model_dump()

    log_event(
        thread_id=state["thread_id"],
        node_name="classifier",
        event_type="decision",
        detail=classification,
        ticket_id=state.get("ticket_id"),
    )

    return {
        "classification": classification,
        "messages": [HumanMessage(content=state.get("description", ""))],
    }
