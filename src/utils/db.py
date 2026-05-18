import sqlite3
from config.settings import DB_PATH
from src.utils.logger import get_logger

log = get_logger(__name__)

CREATE_VIDEOS_TABLE = """
CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT PRIMARY KEY,
    title         TEXT,
    description   TEXT,
    published_at  TEXT,
    channel_id    TEXT,
    channel_title TEXT,
    view_count    INTEGER DEFAULT 0,
    like_count    INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    duration      TEXT,
    tags          TEXT,
    thumbnail_url TEXT,
    viral_score   REAL DEFAULT 0.0,
    fetched_at    TEXT
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


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    log.info(f"Setting up database at {DB_PATH}")
    with get_connection() as conn:
        conn.execute(CREATE_VIDEOS_TABLE)
        conn.execute(CREATE_COMMENTS_TABLE)
        conn.commit()
    log.info("Database ready")


def execute_query(sql, params=()):
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


def execute_write(sql, params=()):
    with get_connection() as conn:
        conn.execute(sql, params)
        conn.commit()


def bulk_insert(table, rows):
    if not rows:
        return 0
    cols         = ", ".join(rows[0].keys())
    placeholders = ", ".join(["?"] * len(rows[0]))
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
    with get_connection() as conn:
        conn.executemany(sql, [tuple(r.values()) for r in rows])
        conn.commit()
    log.debug(f"Saved {len(rows)} rows to {table}")
    return len(rows)


if __name__ == "__main__":
    init_db()
    print("Database created successfully")
