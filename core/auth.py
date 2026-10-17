"""User accounts: SQLite storage + bcrypt password hashing."""
import os
import re
import time
import hashlib
import secrets
import sqlite3
import bcrypt

DB_PATH = os.getenv("USERS_DB", "users.db")
SESSION_DAYS = 30
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised with a user-friendly message when sign-in / sign-up fails."""


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT NOT NULL UNIQUE,
            password_hash BLOB NOT NULL,
            created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at INTEGER NOT NULL
        )"""
    )
    return conn


def _public(row) -> dict:
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def sign_up(name: str, email: str, password: str, confirm: str) -> dict:
    name, email = name.strip(), email.strip().lower()

    if not name:
        raise AuthError("Please enter your name.")
    if not EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address.")
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    if len(password.encode("utf-8")) > 72:   # bcrypt's hard limit
        raise AuthError("Password must be at most 72 characters.")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise AuthError("Password must contain at least one letter and one number.")
    if password != confirm:
        raise AuthError("Passwords don't match.")

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    with _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, password_hash),
            )
        except sqlite3.IntegrityError:
            raise AuthError("An account with this email already exists. Try signing in.")
        return {"id": cur.lastrowid, "name": name, "email": email}


def sign_in(email: str, password: str) -> dict:
    email = email.strip().lower()
    if not email or not password:
        raise AuthError("Please enter your email and password.")

    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    # Same message for unknown email and wrong password — don't reveal which accounts exist.
    if row is None or not bcrypt.checkpw(password.encode("utf-8")[:72], row["password_hash"]):
        raise AuthError("Incorrect email or password.")
    return _public(row)


# ── "Remember me" sessions ────────────────────────────────────────────────────
# The browser keeps a random token in a cookie; the DB stores only its SHA-256 hash,
# so a leaked database can't be used to hijack sessions.

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (int(time.time()),))
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (_hash_token(token), user_id, int(time.time()) + SESSION_DAYS * 86400),
        )
    return token


def user_from_session(token: str | None) -> dict | None:
    if not isinstance(token, str) or not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            """SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token_hash = ? AND s.expires_at > ?""",
            (_hash_token(token), int(time.time())),
        ).fetchone()
    return _public(row) if row else None


def delete_session(token: str | None):
    if isinstance(token, str) and token:
        with _connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))
