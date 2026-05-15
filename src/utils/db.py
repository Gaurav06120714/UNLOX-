"""
src/utils/db.py
───────────────
SQLite helper – creates tables and provides a session factory.
Uses SQLAlchemy Core (no ORM) to stay beginner-friendly.
"""

import sqlite3
from pathlib import Path
from config.settings import DB_PATH
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Schema definitions ────────────────────────────────────────────────────────

CREATE_VIDEOS_TABLE = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    title           TEXT,
    description     TEXT,
    published_at    TEXT,
    channel_id      TEXT,
    channel_title   TEXT,
    view_count      INTEGER DEFAULT 0,
    like_count      INTEGER DEFAULT 0,
    comment_count   INTEGER DEFAULT 0,
    duration        TEXT,
    tags            TEXT,
    thumbnail_url   TEXT,
    viral_score     REAL DEFAULT 0.0,
    fetched_at      TEXT
);
"""

CREATE_COMMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS comments (
    comment_id      TEXT PRIMARY KEY,
    video_id        TEXT,
    author          TEXT,
    text            TEXT,
    like_count      INTEGER DEFAULT 0,
    published_at    TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);
"""

CREATE_AB_TESTS_TABLE = """
CREATE TABLE IF NOT EXISTS ab_tests (
    test_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name       TEXT,
    variant         TEXT,
    metric_name     TEXT,
    metric_value    REAL,
    created_at      TEXT
);
"""

CREATE_FORECASTS_TABLE = """
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT,
    forecast_date   TEXT,
    predicted_views REAL,
    lower_bound     REAL,
    upper_bound     REAL,
    created_at      TEXT
);
"""


# ── Public helpers ────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't already exist."""
    log.info(f"Initialising database at {DB_PATH}")
    with get_connection() as conn:
        for ddl in [
            CREATE_VIDEOS_TABLE,
            CREATE_COMMENTS_TABLE,
            CREATE_AB_TESTS_TABLE,
            CREATE_FORECASTS_TABLE,
        ]:
            conn.execute(ddl)
        conn.commit()
    log.info("Database ready.")


def execute_query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return results as a list of dicts."""
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


def execute_write(sql: str, params: tuple = ()) -> None:
    """Run INSERT / UPDATE / DELETE."""
    with get_connection() as conn:
        conn.execute(sql, params)
        conn.commit()


def bulk_insert(table: str, rows: list[dict]) -> int:
    """
    Insert a list of dicts into `table`.
    Uses INSERT OR REPLACE so re-runs are idempotent.
    Returns number of rows inserted.
    """
    if not rows:
        return 0
    cols = ", ".join(rows[0].keys())
    placeholders = ", ".join(["?"] * len(rows[0]))
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
    with get_connection() as conn:
        conn.executemany(sql, [tuple(r.values()) for r in rows])
        conn.commit()
    log.debug(f"Inserted/replaced {len(rows)} rows into '{table}'.")
    return len(rows)


if __name__ == "__main__":
    init_db()
    print("Database initialised successfully.")
