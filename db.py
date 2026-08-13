"""
Database layer for the palate-learning cooking agent.
Uses SQLite — a single local file, no external server needed.
Every recipe/rating is tagged with a username so multiple people
can use the same app without their data or predictions mixing.
"""

import sqlite3
import json
import hashlib
import secrets
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = "palate.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                recipe_text TEXT NOT NULL,
                cuisine TEXT,
                time_minutes INTEGER,
                ingredients_on_hand TEXT,
                features TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                rating INTEGER NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (recipe_id) REFERENCES recipes(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                username TEXT NOT NULL,
                day TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (username, day)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


# --- User management ---

def _hash_password(password, salt):
    return hashlib.sha256((salt + password).encode()).hexdigest()


def create_user(username, password):
    """Returns (success: bool, message: str)."""
    with get_conn() as conn:
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return False, "That username is already taken."
        salt = secrets.token_hex(8)
        password_hash = _hash_password(password, salt)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, datetime.now(timezone.utc).isoformat()),
        )
        return True, "User created."


def user_exists(username):
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        return row is not None


def list_usernames():
    with get_conn() as conn:
        rows = conn.execute("SELECT username FROM users ORDER BY username ASC").fetchall()
        return [row["username"] for row in rows]


def verify_password(username, password):
    with get_conn() as conn:
        row = conn.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return False
        return _hash_password(password, row["salt"]) == row["password_hash"]


def delete_user(username, password):
    """Deletes a user and all their recipes/ratings/usage, only if password is correct.
    Returns (success: bool, message: str)."""
    if not verify_password(username, password):
        return False, "Incorrect password."
    with get_conn() as conn:
        conn.execute("DELETE FROM ratings WHERE username = ?", (username,))
        conn.execute("DELETE FROM recipes WHERE username = ?", (username,))
        conn.execute("DELETE FROM usage WHERE username = ?", (username,))
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        return True, "User deleted."


def save_recipe(username, title, recipe_text, cuisine, time_minutes, ingredients_on_hand, features=None):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO recipes (username, title, recipe_text, cuisine, time_minutes, ingredients_on_hand, features, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (username, title, recipe_text, cuisine, time_minutes, ingredients_on_hand,
             json.dumps(features) if features else None,
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def save_rating(recipe_id, username, rating, note=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO ratings (recipe_id, username, rating, note, created_at) VALUES (?, ?, ?, ?, ?)",
            (recipe_id, username, rating, note, datetime.now(timezone.utc).isoformat()),
        )


def delete_rating(rating_id, username):
    """Deletes a rating, but only if it belongs to this username."""
    with get_conn() as conn:
        conn.execute("DELETE FROM ratings WHERE id = ? AND username = ?", (rating_id, username))


def check_and_increment_usage(username, daily_limit):
    """Returns (allowed: bool, remaining: int). Increments the counter if allowed."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT count FROM usage WHERE username = ? AND day = ?",
            (username, today),
        ).fetchone()
        current = row["count"] if row else 0

        if current >= daily_limit:
            return False, 0

        if row:
            conn.execute(
                "UPDATE usage SET count = count + 1 WHERE username = ? AND day = ?",
                (username, today),
            )
        else:
            conn.execute(
                "INSERT INTO usage (username, day, count) VALUES (?, ?, 1)",
                (username, today),
            )
        return True, daily_limit - current - 1


def get_all_rated_recipes(username):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT rt.id as rating_id, r.id as recipe_id, r.title, r.cuisine, r.time_minutes, r.features,
                   rt.rating, rt.note, rt.created_at as rated_at
            FROM recipes r
            JOIN ratings rt ON rt.recipe_id = r.id
            WHERE rt.username = ?
            ORDER BY rt.created_at DESC
        """, (username,)).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["features"] = json.loads(d["features"]) if d["features"] else {}
            results.append(d)
        return results
