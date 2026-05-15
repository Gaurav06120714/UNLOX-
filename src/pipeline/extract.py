"""
src/pipeline/extract.py
────────────────────────
PHASE 1 – Data Extraction Pipeline

Flow:
  1. Try YouTube Data API (real key in .env)
  2. If API key missing → fall back to dummy data (for local dev)
  3. Save results to:
       • data/raw/videos.csv
       • data/raw/comments.csv
       • SQLite tables: videos, comments

Run:
    python -m src.pipeline.extract
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    YOUTUBE_API_KEY, YOUTUBE_CHANNEL_ID,
    DATA_RAW_DIR,
)
from src.utils.db import init_db, bulk_insert
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Step 1: Choose data source ────────────────────────────────────────────────

_PLACEHOLDERS = {"", "YOUR_YOUTUBE_API_KEY_HERE", "YOUR_CHANNEL_ID_HERE"}


def _api_configured() -> bool:
    return (YOUTUBE_API_KEY  not in _PLACEHOLDERS and
            YOUTUBE_CHANNEL_ID not in _PLACEHOLDERS)


_MIN_VIDEOS   = 30   # pad with dummy data if real channel has fewer
_MIN_COMMENTS = 200  # pad comments if channel has fewer


def _get_videos() -> tuple[list[dict], list[str]]:
    """Returns (all_videos, real_video_ids)."""
    from src.api.dummy_data import generate_dummy_videos

    if _api_configured():
        log.info("Real API key found — fetching from YouTube Data API.")
        from src.api.youtube_client import fetch_video_ids, fetch_video_details
        ids  = fetch_video_ids(max_videos=100)
        real = fetch_video_details(ids)
        real_ids = [v["video_id"] for v in real]
        if len(real) < _MIN_VIDEOS:
            pad = generate_dummy_videos(n=_MIN_VIDEOS - len(real))
            log.info(f"Channel has {len(real)} real video(s). "
                     f"Padding with {len(pad)} dummy videos for a richer dashboard.")
            real.extend(pad)
        return real, real_ids
    else:
        log.warning("No API key detected — using dummy data.")
        videos = generate_dummy_videos(n=60)
        return videos, []


def _get_comments(video_ids: list[str], real_video_ids: list[str]) -> list[dict]:
    from src.api.dummy_data import generate_dummy_comments

    all_comments: list[dict] = []

    if _api_configured():
        from src.api.youtube_client import fetch_comments
        for vid in real_video_ids[:10]:  # only real video IDs (quota)
            all_comments.extend(fetch_comments(vid))

    # Always pad with dummy comments so dashboard has enough data
    dummy_ids = [v for v in video_ids if v not in set(real_video_ids)]
    needed    = max(0, _MIN_COMMENTS - len(all_comments))
    if needed > 0 and dummy_ids:
        per_vid = max(10, needed // min(len(dummy_ids), 20))
        pad     = generate_dummy_comments(dummy_ids[:20], comments_per_video=per_vid)
        log.info(f"{len(all_comments)} real comments + {len(pad)} dummy comments added.")
        all_comments.extend(pad)

    return all_comments


# ── Step 2: Compute basic engagement metrics ──────────────────────────────────

def compute_engagement(videos: list[dict]) -> list[dict]:
    """
    Add derived columns:
      • like_ratio       = likes / views
      • comment_ratio    = comments / views
      • engagement_rate  = (likes + comments) / views
    """
    for v in videos:
        views    = max(v["view_count"], 1)   # avoid /0
        likes    = v["like_count"]
        comments = v["comment_count"]
        v["like_ratio"]      = round(likes    / views, 5)
        v["comment_ratio"]   = round(comments / views, 5)
        v["engagement_rate"] = round((likes + comments) / views, 5)
    return videos


# ── Step 3: Save to CSV ───────────────────────────────────────────────────────

def save_to_csv(data: list[dict], filename: str) -> Path:
    path = DATA_RAW_DIR / filename
    df = pd.DataFrame(data) if data else pd.DataFrame()
    df.to_csv(path, index=False)
    log.info(f"Saved {len(data)} rows → {path}")
    return path


# ── Step 4: Save to SQLite ────────────────────────────────────────────────────

def save_to_db(videos: list[dict], comments: list[dict]) -> None:
    init_db()

    # Strip extra cols not in DB schema before inserting
    video_cols   = {"video_id","title","description","published_at","channel_id",
                    "channel_title","view_count","like_count","comment_count",
                    "duration","tags","thumbnail_url","viral_score","fetched_at"}
    comment_cols = {"comment_id","video_id","author","text","like_count",
                    "published_at","sentiment_label","sentiment_score"}

    # Normalise every row to exactly the DB schema columns (fill missing with None)
    def _normalise(row: dict, cols: set) -> dict:
        return {c: row.get(c) for c in sorted(cols)}

    clean_videos   = [_normalise(row, video_cols)   for row in videos]
    clean_comments = [_normalise(row, comment_cols) for row in comments]

    bulk_insert("videos",   clean_videos)
    bulk_insert("comments", clean_comments)
    log.info("Data saved to SQLite.")


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_extraction() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("═" * 60)
    log.info("  PHASE 1 – Data Extraction Started")
    log.info("═" * 60)

    # 1. Fetch
    videos, real_video_ids = _get_videos()
    all_video_ids          = [v["video_id"] for v in videos]
    comments               = _get_comments(all_video_ids, real_video_ids)

    # 2. Enrich
    videos = compute_engagement(videos)

    # 3. Persist
    save_to_csv(videos,   "videos.csv")
    save_to_csv(comments, "comments.csv")
    save_to_db(videos, comments)

    df_videos   = pd.DataFrame(videos)
    df_comments = pd.DataFrame(comments)

    log.info(f"Extraction complete: {len(df_videos)} videos, {len(df_comments)} comments.")
    log.info("═" * 60)
    return df_videos, df_comments


if __name__ == "__main__":
    df_v, df_c = run_extraction()
    print("\n── Video sample ─────────────────────────────")
    print(df_v[["title", "view_count", "like_count", "engagement_rate"]].head(5).to_string())
    print("\n── Comment sample ───────────────────────────")
    print(df_c[["author", "text", "like_count"]].head(5).to_string())
