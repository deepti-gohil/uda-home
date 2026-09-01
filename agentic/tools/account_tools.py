"""Read-only tool(s) over the external CultPass DB: account + user + booking lookup."""
from __future__ import annotations

from langchain_core.tools import tool

from data.external.db import get_external_session
from data.external.models import Account, Booking, User


def _serialize_booking(b: Booking) -> dict:
    return {
        "booking_id": b.id,
        "experience_name": b.experience_name,
        "amount_usd": b.amount_usd,
        "booking_date": b.booking_date.isoformat(),
        "status": b.status,
    }


def _serialize_user(u: User) -> dict:
    return {
        "user_id": u.id,
        "full_name": u.full_name,
        "email": u.email,
        "role": u.role,
        "bookings": [_serialize_booking(b) for b in u.bookings],
    }


@tool
def account_lookup_tool(email: str | None = None, account_id: int | None = None) -> dict:
    """Look up a CultPass account, its members, and their bookings by the
    requesting user's email OR by account_id. Provide exactly one of the two.
    Use this before any account- or booking-related action so you know the
    account's plan tier and status.

    Args:
        email: the customer's email address, if known.
        account_id: the CultPass account id, if already known from context.
    """
    if not email and not account_id:
        return {"error": "missing_input", "reason": "Provide either email or account_id."}

    with get_external_session() as session:
        account: Account | None = None

        if email:
            user = session.query(User).filter(User.email == email.strip().lower()).one_or_none()
            if user is None:
                # emails were seeded lower-case; still try a case-insensitive match
                user = (
                    session.query(User)
                    .filter(User.email.ilike(email.strip()))
                    .one_or_none()
                )
            if user is None:
                return {"error": "not_found", "reason": f"No user found for email {email!r}."}
            account = user.account
        else:
            account = session.get(Account, account_id)
            if account is None:
                return {"error": "not_found", "reason": f"No account found for account_id {account_id!r}."}

        return {
            "account_id": account.id,
            "account_name": account.account_name,
            "plan_tier": account.plan_tier,
            "status": account.status,
            "billing_cycle": account.billing_cycle,
            "renewal_date": account.renewal_date.isoformat(),
            "users": [_serialize_user(u) for u in account.users],
        }
