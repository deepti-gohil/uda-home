"""
UDA-Hub's own database: the ticketing system, the knowledge base, long-term
memory, and the agent decision log. This is UDA-Hub's data — separate from
whatever CultPass (or any other plugged-in customer) owns in its own system
(see data/external/models.py).
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import ForeignKey, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    external_account_id: Mapped[int | None] = mapped_column(nullable=True)
    external_user_id: Mapped[int | None] = mapped_column(nullable=True)
    channel: Mapped[str] = mapped_column(String(30), nullable=False)  # zendesk|intercom|freshdesk|chat|email
    subject: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="open")
    # open -> resolved | escalated -> closed
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    ticket_metadata: Mapped["TicketMetadata"] = relationship(
        back_populates="ticket", uselist=False, cascade="all, delete-orphan"
    )
    messages: Mapped[list["TicketMessage"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan", order_by="TicketMessage.created_at"
    )


class TicketMetadata(Base):
    __tablename__ = "ticket_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False, unique=True)
    platform: Mapped[str] = mapped_column(String(30), nullable=False)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    urgency: Mapped[str | None] = mapped_column(String(20), nullable=True)  # low|medium|high|critical
    sentiment: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)

    ticket: Mapped["Ticket"] = relationship(back_populates="ticket_metadata")


class TicketMessage(Base):
    __tablename__ = "ticket_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False)
    sender: Mapped[str] = mapped_column(String(20), nullable=False)  # customer|agent|system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)

    ticket: Mapped["Ticket"] = relationship(back_populates="messages")


class Knowledge(Base):
    __tablename__ = "knowledge"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_user_id: Mapped[int | None] = mapped_column(nullable=True, index=True)
    external_account_id: Mapped[int | None] = mapped_column(nullable=True, index=True)
    memory_type: Mapped[str] = mapped_column(String(30), nullable=False)  # preference|resolution_summary
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-encoded float list
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class AgentRunLog(Base):
    """Structured, searchable log of every agent decision / tool call in a run."""

    __tablename__ = "agent_run_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ticket_id: Mapped[int | None] = mapped_column(nullable=True, index=True)
    node_name: Mapped[str] = mapped_column(String(50), nullable=False)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)  # decision|tool_call|routing|outcome
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)
