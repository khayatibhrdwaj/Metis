"""
Metis Backend - Accounts & Profile History

Simple email/password accounts backed by SQLite (Python's stdlib sqlite3 -
no extra dependency, deliberately, since this project already hit
Windows compiled-dependency pain once with pandas). Passwords are salted
and hashed with PBKDF2-HMAC-SHA256 (stdlib hashlib, no bcrypt/passlib
dependency for the same reason) - never stored in plain text.

Each time a logged-in user submits the intake form, their metrics are
appended to profile_history (not overwritten), so logging in again with
the same email can restore the most recent profile and the history list
is queryable.

This is a small self-hosted prototype's auth system, not a production-
grade one: no email verification, no password reset flow, no rate
limiting on login attempts. Good enough for a course project / demo;
would need hardening before any real deployment.
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "metis_users.db"

PBKDF2_ITERATIONS = 200_000


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS profile_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            submitted_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    conn.close()


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS).hex()


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def register_user(email: str, password: str) -> dict:
    email = _normalize_email(email)
    if not email or "@" not in email:
        return {"success": False, "error": "Enter a valid email address."}
    if not password or len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    conn = _connect()
    try:
        existing = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return {"success": False, "error": "An account with this email already exists — try logging in instead."}

        salt = os.urandom(16)
        password_hash = _hash_password(password, salt)
        conn.execute(
            "INSERT INTO users (email, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (email, salt.hex(), password_hash, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return {"success": True, "error": None}
    finally:
        conn.close()


def verify_login(email: str, password: str) -> dict:
    email = _normalize_email(email)
    conn = _connect()
    try:
        row = conn.execute("SELECT salt, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        if not row:
            return {"success": False, "error": "No account with that email. Register first.", "profile": None}
        salt = bytes.fromhex(row["salt"])
        candidate_hash = _hash_password(password, salt)
        if candidate_hash != row["password_hash"]:
            return {"success": False, "error": "Incorrect password.", "profile": None}

        latest = conn.execute(
            "SELECT profile_json FROM profile_history WHERE email = ? ORDER BY id DESC LIMIT 1",
            (email,),
        ).fetchone()
        profile = json.loads(latest["profile_json"]) if latest else None
        return {"success": True, "error": None, "profile": profile}
    finally:
        conn.close()


def save_profile(email: str, profile: dict) -> dict:
    email = _normalize_email(email)
    conn = _connect()
    try:
        user = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return {"success": False, "error": "Unknown account - log in again."}
        conn.execute(
            "INSERT INTO profile_history (email, profile_json, submitted_at) VALUES (?, ?, ?)",
            (email, json.dumps(profile), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return {"success": True, "error": None}
    finally:
        conn.close()


def get_profile_history(email: str) -> list:
    email = _normalize_email(email)
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, profile_json, submitted_at FROM profile_history WHERE email = ? ORDER BY id DESC",
            (email,),
        ).fetchall()
        return [
            {"id": r["id"], "profile": json.loads(r["profile_json"]), "submitted_at": r["submitted_at"]}
            for r in rows
        ]
    finally:
        conn.close()


init_db()
