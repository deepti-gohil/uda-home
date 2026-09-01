"""Connection helper for UDA-Hub's own core DB. Path is absolute (see config.py)."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import config
from data.core.models import Base

_engine = create_engine(f"sqlite:///{config.CORE_DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_core_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def get_core_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
