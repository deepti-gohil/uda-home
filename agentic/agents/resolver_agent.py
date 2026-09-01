"""
Resolver Agent — retrieves knowledge (RAG), optionally invokes account
tools, and drafts a resolution. This is the only node that calls tools with
side effects (refund_tool writes to the external DB).

Flow:
  1. Always run kb_search_tool first (deterministic) to get retrieved_docs
     and a confidence score. Confidence gating happens BEFORE any LLM call —
     if retrieval confidence is too low, we escalate rather than let the LLM
     guess (see architecture.md section 5).
  2. If the category plausibly needs an account action (billing/bookings/
     account/subscription), give the LLM account_lookup_tool + refund_tool
     and let it decide whether to call them, in a small hand-rolled
     tool-calling loop (max 3 rounds — no prebuilt agent executor).
  3. Draft the final resolution message grounded in the retrieved article(s)
     and any tool results. If a tool call reports the action isn't possible
     (e.g. refund outside window), escalate instead of half-answering.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

import config
from agentic.logging_utils import log_event
from agentic.state import TicketState
from agentic.tools.account_tools import account_lookup_tool
from agentic.tools.kb_search_tool import kb_search_tool
from agentic.tools.refund_tool import refund_tool

_ACCOUNT_ACTION_CATEGORIES = {"billing", "bookings", "account", "subscription"}
_ACTION_TOOLS = {"account_lookup_tool": account_lookup_tool, "refund_tool": refund_tool}
_MAX_TOOL_ROUNDS = 3

_llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)
_llm_with_tools = _llm.bind_tools(list(_ACTION_TOOLS.values()))


def resolver_agent(state: TicketState) -> dict:
    classification = state.get("classification") or {}
    query = classification.get("summary") or state.get("description", "")

    kb_result = kb_search_tool.invoke({"query": query, "k": 3})
    retrieved_docs = kb_result.get("results", [])
    confidence = kb_result.get("top_score", 0.0)

    tool_calls_log: list[dict] = [
        {"tool": "kb_search_tool", "args": {"query": query}, "result": kb_result}
    ]

    log_event(
        thread_id=state["thread_id"],
        node_name="resolver",
        event_type="tool_call",
        detail={"tool": "kb_search_tool", "top_score": confidence, "n_results": len(retrieved_docs)},
        ticket_id=state.get("ticket_id"),
    )

    if confidence < config.CONFIDENCE_THRESHOLD or not retrieved_docs:
        log_event(
            thread_id=state["thread_id"],
            node_name="resolver",
            event_type="decision",
            detail={"decision": "escalate", "reason": "low retrieval confidence", "confidence": confidence},
            ticket_id=state.get("ticket_id"),
        )
        return {
            "retrieved_docs": retrieved_docs,
            "confidence": confidence,
            "tool_calls_log": tool_calls_log,
            "needs_escalation": True,
        }

    articles_text = "\n\n".join(
        f"[Article {d['article_id']}] {d['title']}\n{d['content']}" for d in retrieved_docs
    )
    memory_text = "\n".join(f"- ({m['memory_type']}) {m['content']}" for m in state.get("long_term_context", []))

    allow_action_tools = classification.get("category") in _ACCOUNT_ACTION_CATEGORIES
    system_prompt = (
        "You are the CultPass support Resolver agent. Answer the customer's ticket using ONLY "
        "the knowledge base articles provided below — do not invent policy. Be concise and warm.\n\n"
        f"Relevant knowledge base articles:\n{articles_text}\n\n"
        + (f"What we remember about this customer:\n{memory_text}\n\n" if memory_text else "")
        + (
            "If the customer is asking for an account action you can perform (like a refund), "
            "look up their account with account_lookup_tool (by email) to find the right booking_id, "
            "then call refund_tool if it's warranted and eligible per the knowledge base policy. "
            "CRITICAL: you must actually CALL refund_tool to execute the refund — never write a reply "
            "that says a refund is being processed, has been issued, or is done, unless refund_tool has "
            "already been called in this conversation and returned status: refunded. Describing the "
            "action instead of calling the tool is a critical failure — the refund will not actually "
            "happen. If a tool tells you the action isn't eligible, say so plainly and do not claim it's "
            "done.\n\n"
            if allow_action_tools
            else ""
        )
        + "When you have enough information, reply with the final customer-facing message as plain text "
        "(no tool call) — that reply is sent directly to the customer."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Subject: {state.get('subject', '')}\n"
                f"Message: {state.get('description', '')}\n"
                f"Customer email (if you need it for account_lookup_tool): "
                f"{(state.get('metadata') or {}).get('email', 'unknown')}"
            )
        ),
    ]

    llm = _llm_with_tools if allow_action_tools else _llm
    resolution_text = None
    action_ineligible = False  # a refund (or similar) tool said the action can't be done — per KB
    # policy this always needs a human, regardless of how the LLM words its reply.

    for _ in range(_MAX_TOOL_ROUNDS):
        ai_msg: AIMessage = llm.invoke(messages)
        messages.append(ai_msg)

        if not getattr(ai_msg, "tool_calls", None):
            resolution_text = ai_msg.content
            break

        for tool_call in ai_msg.tool_calls:
            tool_fn = _ACTION_TOOLS[tool_call["name"]]
            result = tool_fn.invoke(tool_call["args"])
            tool_calls_log.append({"tool": tool_call["name"], "args": tool_call["args"], "result": result})
            log_event(
                thread_id=state["thread_id"],
                node_name="resolver",
                event_type="tool_call",
                detail={"tool": tool_call["name"], "args": tool_call["args"], "result": result},
                ticket_id=state.get("ticket_id"),
            )
            if tool_call["name"] == "refund_tool" and isinstance(result, dict) and "error" in result:
                action_ineligible = True
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))
    else:
        resolution_text = messages[-1].content if isinstance(messages[-1], AIMessage) else None

    # Safety net (not just a prompt instruction): if the reply claims a refund
    # happened but refund_tool was never actually called and confirmed
    # "refunded", that's a hallucinated action on a money-handling flow — don't
    # ship it to the customer, escalate instead. Caught this exact failure mode
    # in testing (the LLM looked up the account, then just wrote "I will
    # process your refund now" without calling refund_tool).
    refund_confirmed = any(
        c["tool"] == "refund_tool" and isinstance(c["result"], dict) and c["result"].get("status") == "refunded"
        for c in tool_calls_log
    )
    unverified_refund_claim = (
        allow_action_tools
        and bool(resolution_text)
        and any(kw in resolution_text.lower() for kw in ("refund", "reimburse", "money back"))
        and any(kw in resolution_text.lower() for kw in ("will process", "have processed", "has been", "is being", "i've refunded", "i have refunded", "processed your"))
        and not refund_confirmed
    )

    needs_escalation = action_ineligible or unverified_refund_claim

    if unverified_refund_claim:
        # Don't send an unconfirmed "your refund is processed" claim to the
        # customer or persist it as the ticket's agent message — the
        # escalation node's own summary becomes the record of this turn
        # instead (see memory_agent.py, which falls back to
        # escalation_summary when resolution is empty).
        resolution_text = None

    log_event(
        thread_id=state["thread_id"],
        node_name="resolver",
        event_type="decision",
        detail={
            "decision": "escalate" if needs_escalation else "resolved",
            "confidence": confidence,
            "action_ineligible": action_ineligible,
            "unverified_refund_claim": unverified_refund_claim,
        },
        ticket_id=state.get("ticket_id"),
    )

    return {
        "retrieved_docs": retrieved_docs,
        "confidence": confidence,
        "tool_calls_log": tool_calls_log,
        "resolution": resolution_text,
        "needs_escalation": needs_escalation,
        "messages": [AIMessage(content=resolution_text)] if resolution_text else [],
    }


def route_after_resolver(state: TicketState) -> str:
    return "escalation" if state.get("needs_escalation") else "memory_writer"
