"""
Database layer for the palate-learning cooking agent.
Uses SQLite — a single local file, no external server needed.
"""

import sqlite3
import json
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
                rating INTEGER NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (recipe_id) REFERENCES recipes(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                session_id TEXT NOT NULL,
                day TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (session_id, day)
            )
        """)


def save_recipe(title, recipe_text, cuisine, time_minutes, ingredients_on_hand, features=None):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO recipes (title, recipe_text, cuisine, time_minutes, ingredients_on_hand, features, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, recipe_text, cuisine, time_minutes, ingredients_on_hand,
             json.dumps(features) if features else None,
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def save_rating(recipe_id, rating, note=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO ratings (recipe_id, rating, note, created_at) VALUES (?, ?, ?, ?)",
            (recipe_id, rating, note, datetime.now(timezone.utc).isoformat()),
        )


def check_and_increment_usage(session_id, daily_limit):
    """Returns (allowed: bool, remaining: int). Increments the counter if allowed."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT count FROM usage WHERE session_id = ? AND day = ?",
            (session_id, today),
        ).fetchone()
        current = row["count"] if row else 0

        if current >= daily_limit:
            return False, 0

        if row:
            conn.execute(
                "UPDATE usage SET count = count + 1 WHERE session_id = ? AND day = ?",
                (session_id, today),
            )
        else:
            conn.execute(
                "INSERT INTO usage (session_id, day, count) VALUES (?, ?, 1)",
                (session_id, today),
            )
        return True, daily_limit - current - 1


def delete_rating(rating_id):
    """Deletes a specific rating. The underlying recipe row stays (harmless), just the rating is removed."""
    with get_conn() as conn:
        conn.execute("DELETE FROM ratings WHERE id = ?", (rating_id,))


def get_all_rated_recipes():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT rt.id as rating_id, r.id as recipe_id, r.title, r.cuisine, r.time_minutes, r.features,
                   rt.rating, rt.note, rt.created_at as rated_at
            FROM recipes r
            JOIN ratings rt ON rt.recipe_id = r.id
            ORDER BY rt.created_at DESC
        """).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["features"] = json.loads(d["features"]) if d["features"] else {}
            results.append(d)
        return results
