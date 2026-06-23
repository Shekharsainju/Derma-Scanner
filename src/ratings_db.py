"""Community product ratings, stored in a local SQLite database.

Persists across restarts. Includes server-side validation: stars must be 1-5.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "ratings.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                stars INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def add_rating(product_name: str, stars: int) -> Dict:
    # Server-side validation - never trust the client
    name = (product_name or "").strip()
    if not name or len(name) > 200:
        return {"ok": False, "error": "Invalid product name."}
    if not isinstance(stars, int) or stars < 1 or stars > 5:
        return {"ok": False, "error": "Stars must be an integer from 1 to 5."}

    with _connect() as conn:
        conn.execute(
            "INSERT INTO ratings (product_name, stars, created_at) VALUES (?, ?, ?)",
            (name, stars, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    return {"ok": True, **get_rating(name)}


def get_rating(product_name: str) -> Dict:
    name = (product_name or "").strip()
    with _connect() as conn:
        row = conn.execute(
            "SELECT AVG(stars) AS avg_stars, COUNT(*) AS count FROM ratings WHERE product_name = ?",
            (name,),
        ).fetchone()
    avg = round(row["avg_stars"], 1) if row["avg_stars"] is not None else 0.0
    count = row["count"] or 0
    return {"product_name": name, "average": avg, "count": count}