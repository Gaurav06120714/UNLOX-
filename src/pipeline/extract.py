import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import YOUTUBE_API_KEY, YOUTUBE_CHANNEL_ID, DATA_RAW_DIR
from src.utils.db import init_db, bulk_insert
from src.utils.logger import get_logger

log = get_logger(__name__)

PLACEHOLDER_KEYS = {"", "YOUR_YOUTUBE_API_KEY_HERE", "YOUR_CHANNEL_ID_HERE"}

MIN_VIDEOS   = 30
MIN_COMMENTS = 200


def api_is_configured():
    return (YOUTUBE_API_KEY not in PLACEHOLDER_KEYS and
            YOUTUBE_CHANNEL_ID not in PLACEHOLDER_KEYS)


def get_videos():
    from src.api.dummy_data import generate_dummy_videos

    if api_is_configured():
        log.info("API key found - fetching from YouTube")
        from src.api.youtube_client import fetch_video_ids, fetch_video_details
        ids      = fetch_video_ids(max_videos=100)
        videos   = fetch_video_details(ids)
        real_ids = [v["video_id"] for v in videos]

        if len(videos) < MIN_VIDEOS:
            pad = generate_dummy_videos(n=MIN_VIDEOS - len(videos))
            log.info(f"Only {len(videos)} real videos found, padding with {len(pad)} dummy ones")
            videos.extend(pad)
        return videos, real_ids
    else:
        log.warning("No API key - using dummy data")
        videos = generate_dummy_videos(n=60)
        return videos, []


def get_comments(all_video_ids, real_video_ids):
    from src.api.dummy_data import generate_dummy_comments

    comments = []

    if api_is_configured():
        from src.api.youtube_client import fetch_comments
        for vid in real_video_ids[:10]:
            comments.extend(fetch_comments(vid))

    dummy_ids = [v for v in all_video_ids if v not in set(real_video_ids)]
    needed    = max(0, MIN_COMMENTS - len(comments))
    if needed > 0 and dummy_ids:
        per_vid = max(10, needed // min(len(dummy_ids), 20))
        pad     = generate_dummy_comments(dummy_ids[:20], comments_per_video=per_vid)
        log.info(f"{len(comments)} real comments + {len(pad)} dummy comments")
        comments.extend(pad)

    return comments


def compute_engagement(videos):
    for v in videos:
        views    = max(v["view_count"], 1)
        likes    = v["like_count"]
        comments = v["comment_count"]
        v["like_ratio"]      = round(likes / views, 5)
        v["comment_ratio"]   = round(comments / views, 5)
        v["engagement_rate"] = round((likes + comments) / views, 5)
    return videos


def save_csv(data, filename):
    path = DATA_RAW_DIR / filename
    df   = pd.DataFrame(data) if data else pd.DataFrame()
    df.to_csv(path, index=False)
    log.info(f"Saved {len(data)} rows to {path}")
    return path


def save_to_db(videos, comments):
    init_db()

    video_cols = {"video_id","title","description","published_at","channel_id",
                  "channel_title","view_count","like_count","comment_count",
                  "duration","tags","thumbnail_url","viral_score","fetched_at"}
    comment_cols = {"comment_id","video_id","author","text","like_count",
                    "published_at","sentiment_label","sentiment_score"}

    clean_videos   = [{c: row.get(c) for c in sorted(video_cols)}   for row in videos]
    clean_comments = [{c: row.get(c) for c in sorted(comment_cols)} for row in comments]

    bulk_insert("videos",   clean_videos)
    bulk_insert("comments", clean_comments)
    log.info("Saved to database")


def run_extraction():
    log.info("Starting data extraction")

    videos, real_ids = get_videos()
    all_ids          = [v["video_id"] for v in videos]
    comments         = get_comments(all_ids, real_ids)

    videos = compute_engagement(videos)

    save_csv(videos,   "videos.csv")
    save_csv(comments, "comments.csv")
    save_to_db(videos, comments)

    df_videos   = pd.DataFrame(videos)
    df_comments = pd.DataFrame(comments)

    log.info(f"Done: {len(df_videos)} videos, {len(df_comments)} comments")
    return df_videos, df_comments


if __name__ == "__main__":
    df_v, df_c = run_extraction()
    print(df_v[["title", "view_count", "like_count", "engagement_rate"]].head(5).to_string())
    print(df_c[["author", "text", "like_count"]].head(5).to_string())
