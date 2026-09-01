"""Connection helper for CultPass's external DB. Path is absolute (see config.py)."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import config
from data.external.models import Base

_engine = create_engine(f"sqlite:///{config.EXTERNAL_DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_external_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def get_external_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
