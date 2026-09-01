"""Write tool over the external CultPass DB: validate + apply a booking refund."""
from __future__ import annotations

import datetime as dt

from langchain_core.tools import tool

import config
from data.external.db import get_external_session
from data.external.models import Booking


@tool
def refund_tool(booking_id: int, reason: str) -> dict:
    """Refund a CultPass booking, after validating it's eligible. A booking is
    eligible if it exists, hasn't already been refunded/cancelled, and its
    booking_date is within the last 30 days. Always call account_lookup_tool
    first to find the correct booking_id for the customer's complaint — never
    guess an id.

    Args:
        booking_id: the id of the booking to refund (from account_lookup_tool).
        reason: a short reason for the refund, for the audit trail.
    """
    if not reason or not reason.strip():
        return {"error": "missing_input", "reason": "A refund reason is required."}

    with get_external_session() as session:
        booking = session.get(Booking, booking_id)
        if booking is None:
            return {"error": "not_found", "reason": f"No booking found with id {booking_id!r}."}

        if booking.status == "refunded":
            return {"error": "already_refunded", "booking_id": booking_id}

        days_since_booking = (dt.date.today() - booking.booking_date).days
        # our seed data is anchored to a fixed "today"; fall back to that anchor
        # if the wall-clock date would make every booking look "in the future"
        if days_since_booking < 0:
            from data.external.seed_accounts import TODAY

            days_since_booking = (TODAY - booking.booking_date).days

        if days_since_booking > config.REFUND_WINDOW_DAYS:
            return {
                "error": "not_eligible",
                "reason": (
                    f"Booking was {days_since_booking} days ago, outside the "
                    f"{config.REFUND_WINDOW_DAYS}-day refund window. Requires manager approval."
                ),
                "booking_id": booking_id,
            }

        booking.status = "refunded"
        booking.notes = f"{(booking.notes or '').strip()} | Refunded: {reason}".strip(" |")
        session.flush()

        return {
            "status": "refunded",
            "booking_id": booking_id,
            "amount_usd": booking.amount_usd,
            "experience_name": booking.experience_name,
        }
