"""Per-user library of past analyses, stored next to the accounts in SQLite."""
import re
import sqlite3

from core.auth import DB_PATH

FIELDS = ["title", "source", "engine", "language", "duration", "elapsed",
          "transcript", "summary", "action_items", "key_decisions", "open_questions"]


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            title          TEXT NOT NULL,
            source         TEXT,
            engine         TEXT,
            language       TEXT,
            duration       REAL,
            elapsed        REAL,
            transcript     TEXT NOT NULL,
            summary        TEXT,
            action_items   TEXT,
            key_decisions  TEXT,
            open_questions TEXT,
            created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id, created_at DESC)")
    return conn


def youtube_id(source: str | None) -> str | None:
    if not source:
        return None
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", source)
    return m.group(1) if m else None


def save_analysis(user_id: int, result: dict) -> int:
    with _connect() as conn:
        cur = conn.execute(
            f"INSERT INTO analyses (user_id, {', '.join(FIELDS)}) VALUES (?, {', '.join('?' * len(FIELDS))})",
            [user_id] + [result.get(f) for f in FIELDS],
        )
        return cur.lastrowid


def list_analyses(user_id: int, limit: int = 24) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, title, source, engine, language, duration, created_at,
                      substr(summary, 1, 400) AS summary_preview
               FROM analyses WHERE user_id = ? ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_analysis(user_id: int, analysis_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ? AND user_id = ?",
                           (analysis_id, user_id)).fetchone()
    return dict(row) if row else None


def delete_analysis(user_id: int, analysis_id: int):
    with _connect() as conn:
        conn.execute("DELETE FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id))


def count_analyses(user_id: int) -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM analyses WHERE user_id = ?", (user_id,)).fetchone()[0]
