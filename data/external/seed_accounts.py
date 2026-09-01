"""
Seed data for CultPass's external DB. Run via 01_external_db_setup.ipynb.

CultPass is UDA-Hub's first customer: a subscription club that gives members
access to cultural experiences (museum exhibits, concerts, workshops). This
seed gives us a handful of accounts in different states (active / paused /
cancelled), users with different roles, and bookings spread across dates —
some inside and some outside the 30-day refund window — so the agentic
workflow's tools (account_lookup_tool, refund_tool) have realistic scenarios
to exercise, including at least one "not eligible" refund case.
"""
from __future__ import annotations

import datetime as dt

from data.external.db import get_external_session, init_external_db
from data.external.models import Account, Booking, User

TODAY = dt.date(2026, 9, 1)  # anchor date for the "current" state of seed data


def _d(days_ago: int) -> dt.date:
    return TODAY - dt.timedelta(days=days_ago)


ACCOUNTS = [
    dict(
        account_name="Nguyen Household",
        plan_tier="premium",
        status="active",
        billing_cycle="monthly",
        renewal_date=_d(-12),
        users=[
            dict(full_name="Lan Nguyen", email="lan.nguyen@example.com", role="owner"),
            dict(full_name="Minh Nguyen", email="minh.nguyen@example.com", role="member"),
        ],
    ),
    dict(
        account_name="Downtown Book Club",
        plan_tier="elite",
        status="active",
        billing_cycle="annual",
        renewal_date=_d(-200),
        users=[
            dict(full_name="Priya Shah", email="priya.shah@example.com", role="owner"),
            dict(full_name="Omar Haddad", email="omar.haddad@example.com", role="member"),
            dict(full_name="Grace Kim", email="grace.kim@example.com", role="member"),
        ],
    ),
    dict(
        account_name="Maria Gomez Solo",
        plan_tier="basic",
        status="paused",
        billing_cycle="monthly",
        renewal_date=_d(-40),
        users=[
            dict(full_name="Maria Gomez", email="maria.gomez@example.com", role="owner"),
        ],
    ),
    dict(
        account_name="Chen Household",
        plan_tier="premium",
        status="cancelled",
        billing_cycle="monthly",
        renewal_date=_d(-90),
        users=[
            dict(full_name="Wei Chen", email="wei.chen@example.com", role="owner"),
            dict(full_name="Amy Chen", email="amy.chen@example.com", role="member"),
        ],
    ),
    dict(
        account_name="Acme Household",
        plan_tier="basic",
        status="active",
        billing_cycle="monthly",
        renewal_date=_d(-5),
        users=[
            dict(full_name="Jordan Blake", email="jordan.blake@example.com", role="owner"),
        ],
    ),
]

# bookings keyed by user email -> list of bookings
BOOKINGS_BY_USER_EMAIL = {
    "lan.nguyen@example.com": [
        dict(experience_name="Modern Art After Hours", amount_usd=45.0, booking_date=_d(5), status="confirmed"),
        dict(experience_name="Jazz in the Courtyard", amount_usd=30.0, booking_date=_d(50), status="confirmed"),
    ],
    "minh.nguyen@example.com": [
        dict(experience_name="Ceramics Workshop", amount_usd=60.0, booking_date=_d(2), status="confirmed"),
    ],
    "priya.shah@example.com": [
        dict(experience_name="Opera Gala Night", amount_usd=120.0, booking_date=_d(10), status="confirmed"),
        dict(
            experience_name="Sculpture Garden Tour",
            amount_usd=25.0,
            booking_date=_d(75),
            status="confirmed",
            notes="Customer reports being charged twice for this booking.",
        ),
    ],
    "omar.haddad@example.com": [
        dict(experience_name="Film Noir Screening", amount_usd=18.0, booking_date=_d(3), status="cancelled"),
    ],
    "maria.gomez@example.com": [
        dict(experience_name="Photography Walk", amount_usd=35.0, booking_date=_d(60), status="confirmed"),
    ],
    "wei.chen@example.com": [
        dict(experience_name="Wine & Watercolor", amount_usd=55.0, booking_date=_d(15), status="refunded"),
    ],
    "jordan.blake@example.com": [
        dict(experience_name="Planetarium Night Show", amount_usd=22.0, booking_date=_d(1), status="confirmed"),
    ],
}


def seed() -> None:
    init_external_db()
    with get_external_session() as session:
        if session.query(Account).count() > 0:
            print("External DB already seeded — skipping.")
            return

        email_to_user: dict[str, User] = {}

        for acc in ACCOUNTS:
            account = Account(
                account_name=acc["account_name"],
                plan_tier=acc["plan_tier"],
                status=acc["status"],
                billing_cycle=acc["billing_cycle"],
                renewal_date=acc["renewal_date"],
            )
            session.add(account)
            session.flush()  # get account.id

            for u in acc["users"]:
                user = User(
                    account_id=account.id,
                    full_name=u["full_name"],
                    email=u["email"],
                    role=u["role"],
                )
                session.add(user)
                session.flush()
                email_to_user[u["email"]] = user

        for email, bookings in BOOKINGS_BY_USER_EMAIL.items():
            user = email_to_user[email]
            for b in bookings:
                session.add(Booking(user_id=user.id, **b))

        session.flush()
        print(
            f"Seeded {len(ACCOUNTS)} accounts, {len(email_to_user)} users, "
            f"{sum(len(v) for v in BOOKINGS_BY_USER_EMAIL.values())} bookings."
        )


if __name__ == "__main__":
    seed()
