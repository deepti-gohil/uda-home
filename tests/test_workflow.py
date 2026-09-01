"""End-to-end tests for the LangGraph workflow: classification, routing,
retrieval/tool usage, resolution vs. escalation, and memory. These call the
real OpenAI API, so they're skipped automatically if OPENAI_API_KEY isn't set.
"""
from __future__ import annotations

from tests.conftest import requires_openai_key

pytestmark = requires_openai_key


def test_resolved_ticket_uses_knowledge_base():
    from agentic.workflow import run_ticket

    state = run_ticket(
        subject="How do I turn off push notifications?",
        description="I'm getting too many push notifications, how do I turn off just those but "
        "keep booking reminder emails?",
        channel="email",
    )
    assert state["status"] == "resolved"
    assert state["resolution"]
    assert state["confidence"] >= 0.72
    assert any(d["category"] == "notifications" for d in state["retrieved_docs"])


def test_critical_urgency_escalates_via_supervisor_without_low_confidence():
    from agentic.workflow import run_ticket

    state = run_ticket(
        subject="Locked out right now",
        description="URGENT - I'm at the venue right now and my digital pass won't scan, the show "
        "starts in 5 minutes, I need help immediately!",
        channel="chat",
    )
    assert state["status"] == "escalated"
    assert state["classification"]["urgency"] == "critical"
    assert state["escalation_summary"]
    # supervisor should have routed straight to escalation, so the resolver
    # never ran a kb_search — tool_calls_log stays empty/unset
    assert not state.get("tool_calls_log")


def test_off_topic_ticket_escalates_on_low_confidence():
    from agentic.workflow import run_ticket

    state = run_ticket(
        subject="Birthday party planning",
        description="Can your team plan and host a surprise 40th birthday party with catering at "
        "one of your venues?",
        channel="chat",
    )
    assert state["status"] == "escalated"
    assert state["confidence"] < 0.72


def test_session_continuity_across_two_turns_same_thread():
    from agentic.workflow import get_session_state, run_ticket

    first = run_ticket(
        subject="Question about my plan",
        description="What's included in the Premium plan?",
        channel="chat",
    )
    thread_id = first["thread_id"]
    ticket_id = first["ticket_id"]

    second = run_ticket(
        subject="follow-up",
        description="And how is that different from Elite?",
        channel="chat",
        thread_id=thread_id,
    )

    assert second["thread_id"] == thread_id
    assert second["ticket_id"] == ticket_id  # continuity: same ticket, not a new one

    snapshot = get_session_state(thread_id)
    assert len(snapshot["messages"]) >= 4  # 2 customer + 2 agent turns


def test_long_term_memory_recalled_in_new_session_for_known_customer():
    from agentic.workflow import run_ticket

    # First session: creates a resolution_summary memory for this customer.
    run_ticket(
        subject="Notification question",
        description="How do I manage my email notification preferences?",
        channel="email",
        metadata={"email": "wei.chen@example.com"},
    )

    # Second, brand-new session (no thread_id passed) for the SAME customer.
    second = run_ticket(
        subject="Another question",
        description="What plans do you offer and what's the difference between them?",
        channel="chat",
        metadata={"email": "wei.chen@example.com"},
    )
    assert second["external_user_id"] is not None
    assert len(second.get("long_term_context", [])) >= 1
