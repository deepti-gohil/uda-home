"""
CultPass's own database — this is DATA OWNED BY UDA-HUB'S CUSTOMER, not by
UDA-Hub itself. UDA-Hub only ever reads/writes it through tools
(agentic/tools/account_tools.py, refund_tool.py), never directly from an agent.

CultPass sells subscription access ("passes") to cultural experiences
(museum exhibits, concerts, workshops). An Account is the billing entity
(a household or a company); a User is a named person under that account;
a Booking is a reservation a User made against an experience, which is what
gets refunded/cancelled when a customer complains.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_name: Mapped[str] = mapped_column(String(200), nullable=False)
    plan_tier: Mapped[str] = mapped_column(String(50), nullable=False)  # basic|premium|elite
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")  # active|paused|cancelled
    billing_cycle: Mapped[str] = mapped_column(String(20), nullable=False, default="monthly")
    renewal_date: Mapped[dt.date] = mapped_column(nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)

    users: Mapped[list["User"]] = relationship(back_populates="account", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="member")  # owner|member
    joined_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)
    last_login_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)

    account: Mapped["Account"] = relationship(back_populates="users")
    bookings: Mapped[list["Booking"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    experience_name: Mapped[str] = mapped_column(String(200), nullable=False)
    amount_usd: Mapped[float] = mapped_column(nullable=False)
    booking_date: Mapped[dt.date] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="confirmed")  # confirmed|cancelled|refunded
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="bookings")
