"""Unit tests for the support-operation tools (database abstraction layer)."""
from __future__ import annotations

import datetime as dt

from tests.conftest import requires_openai_key

from agentic.tools.account_tools import account_lookup_tool
from agentic.tools.refund_tool import refund_tool
from data.external.db import get_external_session
from data.external.models import Account, Booking, User


def _make_booking(days_ago: int) -> tuple[int, int]:
    """Create a throwaway account/user/booking so refund tests never touch
    the shared seed data the demo script relies on. Returns (user_id, booking_id)."""
    with get_external_session() as session:
        account = Account(
            account_name="Test Fixture Household",
            plan_tier="basic",
            status="active",
            billing_cycle="monthly",
            renewal_date=dt.date.today(),
        )
        session.add(account)
        session.flush()

        user = User(account_id=account.id, full_name="Test User", email=f"test.{days_ago}@example.com")
        session.add(user)
        session.flush()

        booking = Booking(
            user_id=user.id,
            experience_name="Test Experience",
            amount_usd=10.0,
            booking_date=dt.date.today() - dt.timedelta(days=days_ago),
            status="confirmed",
        )
        session.add(booking)
        session.flush()
        return user.id, booking.id


def test_account_lookup_tool_found():
    result = account_lookup_tool.invoke({"email": "lan.nguyen@example.com"})
    assert "error" not in result
    assert result["plan_tier"] == "premium"
    assert result["status"] == "active"
    assert any(u["email"] == "lan.nguyen@example.com" for u in result["users"])


def test_account_lookup_tool_not_found():
    result = account_lookup_tool.invoke({"email": "nobody-here@example.com"})
    assert result["error"] == "not_found"


def test_account_lookup_tool_requires_input():
    result = account_lookup_tool.invoke({})
    assert result["error"] == "missing_input"


def test_refund_tool_eligible_within_window():
    _, booking_id = _make_booking(days_ago=5)
    result = refund_tool.invoke({"booking_id": booking_id, "reason": "customer request"})
    assert result["status"] == "refunded"
    assert result["booking_id"] == booking_id


def test_refund_tool_ineligible_outside_window():
    _, booking_id = _make_booking(days_ago=45)
    result = refund_tool.invoke({"booking_id": booking_id, "reason": "customer request"})
    assert result["error"] == "not_eligible"


def test_refund_tool_already_refunded():
    _, booking_id = _make_booking(days_ago=2)
    first = refund_tool.invoke({"booking_id": booking_id, "reason": "customer request"})
    assert first["status"] == "refunded"
    second = refund_tool.invoke({"booking_id": booking_id, "reason": "customer request again"})
    assert second["error"] == "already_refunded"


def test_refund_tool_unknown_booking():
    result = refund_tool.invoke({"booking_id": 999999, "reason": "customer request"})
    assert result["error"] == "not_found"


@requires_openai_key
def test_kb_search_tool_finds_relevant_article():
    from agentic.tools.kb_search_tool import kb_search_tool

    result = kb_search_tool.invoke({"query": "Can I get a refund for a booking I made last week?", "k": 3})
    titles = [r["title"] for r in result["results"]]
    assert any("Refund" in t for t in titles)
    assert result["top_score"] > 0.5


@requires_openai_key
def test_kb_search_tool_low_confidence_for_off_topic_query():
    from agentic.tools.kb_search_tool import kb_search_tool

    on_topic = kb_search_tool.invoke({"query": "What is the refund policy?", "k": 3})
    off_topic = kb_search_tool.invoke(
        {"query": "Can you help me plan a surprise birthday party with catering?", "k": 3}
    )
    assert off_topic["top_score"] < on_topic["top_score"]


@requires_openai_key
def test_memory_tools_write_and_search_roundtrip():
    from agentic.tools.memory_tools import search_long_term_memory, write_long_term_memory

    write_result = write_long_term_memory.invoke(
        {
            "content": "Customer strongly prefers being contacted by email, not phone.",
            "memory_type": "preference",
            "external_user_id": 987654,
        }
    )
    assert write_result["status"] == "stored"

    search_result = search_long_term_memory.invoke(
        {"query": "How does this customer like to be contacted?", "external_user_id": 987654, "k": 3}
    )
    assert len(search_result["results"]) >= 1
    assert "email" in search_result["results"][0]["content"].lower()
