from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path(os.getenv("IDEMPOTENCY_DB", "/tmp/seen_events.db"))


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_events (
            event_id TEXT PRIMARY KEY,
            seen_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        _init_db(conn)
        yield conn
    finally:
        conn.close()


def seen(event_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("SELECT 1 FROM seen_events WHERE event_id = ?", (event_id,))
        return cur.fetchone() is not None


def mark_seen(event_id: str) -> None:
    with _conn() as conn:
        conn.execute("INSERT OR IGNORE INTO seen_events(event_id) VALUES (?)", (event_id,))
        conn.commit()
