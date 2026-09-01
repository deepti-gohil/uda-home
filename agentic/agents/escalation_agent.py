"""Escalation Agent — builds a structured hand-off summary for a human agent."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

import config
from agentic.logging_utils import log_event
from agentic.state import TicketState

_llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0.2, api_key=config.OPENAI_API_KEY)


def escalation_agent(state: TicketState) -> dict:
    classification = state.get("classification") or {}

    reason_bits = []
    if classification.get("urgency") == "critical":
        reason_bits.append("critical urgency")
    if classification.get("sentiment") == "very_negative":
        reason_bits.append("very negative sentiment")
    if classification.get("category") == "unknown":
        reason_bits.append("unrecognized category")
    if state.get("confidence") is not None and state["confidence"] < config.CONFIDENCE_THRESHOLD:
        reason_bits.append(f"low retrieval confidence ({state['confidence']:.2f})")
    if any(
        c["tool"] == "refund_tool" and "error" in c["result"]
        for c in state.get("tool_calls_log", [])
        if isinstance(c.get("result"), dict)
    ):
        reason_bits.append("requested action not eligible for self-service")
    reason = "; ".join(reason_bits) or "routed to escalation by supervisor"

    prompt = (
        "Write a concise hand-off note (4-6 sentences) for a human support agent picking up this "
        "escalated CultPass ticket. Include: what the customer wants, what's already been tried "
        "(if anything), and why it was escalated. Do not address the customer — this is an internal "
        "note.\n\n"
        f"Subject: {state.get('subject', '')}\n"
        f"Description: {state.get('description', '')}\n"
        f"Classification: {classification}\n"
        f"Escalation reason(s): {reason}\n"
        f"Retrieved KB articles considered: {[d['title'] for d in state.get('retrieved_docs', [])]}\n"
        f"Tool calls made: {state.get('tool_calls_log', [])}\n"
    )
    summary = _llm.invoke([HumanMessage(content=prompt)]).content

    log_event(
        thread_id=state["thread_id"],
        node_name="escalation",
        event_type="outcome",
        detail={"reason": reason, "summary": summary},
        ticket_id=state.get("ticket_id"),
    )

    return {
        "escalation_summary": summary,
        "status": "escalated",
        "messages": [AIMessage(content=summary)],
    }
