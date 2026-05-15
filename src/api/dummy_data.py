"""
src/api/dummy_data.py
──────────────────────
Generates realistic dummy YouTube data so you can run the entire
pipeline without a real API key. Seed is fixed for reproducibility.

Usage:
    from src.api.dummy_data import generate_dummy_videos, generate_dummy_comments
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta, timezone

import numpy as np

SEED = 42
rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)

# ── Sample content pools ──────────────────────────────────────────────────────

TOPICS = [
    "Python Tutorial", "Machine Learning Crash Course", "Data Science Full Guide",
    "AI Tools You Need in 2024", "ChatGPT Masterclass", "Streamlit Dashboard Build",
    "YouTube Analytics Tips", "How to Go Viral on YouTube", "NLP with Python",
    "Deep Learning Explained Simply", "SQL for Beginners", "Pandas vs Polars",
    "Data Analyst Portfolio Project", "Power BI vs Tableau", "Freelancing as a Data Scientist",
]

TAGS_POOL = [
    "python", "machinelearning", "datascience", "ai", "tutorial",
    "streamlit", "nlp", "analytics", "youtube", "viral", "beginner",
    "sql", "pandas", "deeplearning", "portfolio",
]

COMMENT_POSITIVES = [
    "This video is absolutely amazing! I learned so much.",
    "Best tutorial on this topic I have ever seen. Thank you!",
    "You explain things so clearly. Subscribed immediately!",
    "This is exactly what I needed. Saved me hours of searching.",
    "Brilliant content as always. Keep it up!",
    "I watched this three times. So valuable.",
    "Great job! Very well structured and easy to follow.",
]

COMMENT_NEGATIVES = [
    "Honestly disappointed. Expected more depth.",
    "The audio quality is really bad in this one.",
    "Too basic for me. I already knew all of this.",
    "Please add timestamps. Hard to navigate.",
    "Some of the information here is outdated.",
]

COMMENT_NEUTRALS = [
    "At what timestamp do you cover the installation?",
    "Can you do a follow-up video on this topic?",
    "What laptop are you using in this video?",
    "This is for Python 3.10 or 3.11?",
    "Does this work on Windows as well?",
    "Is the source code available somewhere?",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_id(length: int = 11) -> str:
    return "".join(rng.choices(string.ascii_letters + string.digits, k=length))


def _random_date(days_back: int = 365) -> str:
    offset = timedelta(days=rng.randint(0, days_back))
    dt = datetime.now(timezone.utc) - offset
    return dt.isoformat()


def _duration_iso() -> str:
    mins = rng.randint(5, 45)
    secs = rng.randint(0, 59)
    return f"PT{mins}M{secs}S"


# ── Public generators ─────────────────────────────────────────────────────────

def generate_dummy_videos(n: int = 50) -> list[dict]:
    """
    Generate n realistic dummy video records.
    View counts follow a log-normal distribution to mimic real channels.
    """
    videos = []
    base_views = np_rng.lognormal(mean=10.5, sigma=1.5, size=n).astype(int)

    for i in range(n):
        views    = int(base_views[i])
        likes    = int(views * rng.uniform(0.02, 0.12))
        comments = int(views * rng.uniform(0.001, 0.03))
        topic    = rng.choice(TOPICS)
        tags     = rng.sample(TAGS_POOL, k=rng.randint(3, 8))

        videos.append({
            "video_id":      _random_id(),
            "title":         f"{topic} – Part {i+1}",
            "description":   f"In this video we cover {topic.lower()} in depth. "
                             "Like, share, and subscribe for more content!",
            "published_at":  _random_date(730),
            "channel_id":    "UC_DUMMY_CHANNEL_001",
            "channel_title": "DataScienceWithGaurav",
            "view_count":    views,
            "like_count":    likes,
            "comment_count": comments,
            "duration":      _duration_iso(),
            "tags":          "|".join(tags),
            "thumbnail_url": f"https://i.ytimg.com/vi/{_random_id()}/hqdefault.jpg",
            "viral_score":   0.0,   # filled by pipeline
            "fetched_at":    datetime.now(timezone.utc).isoformat(),
        })

    return videos


def generate_dummy_comments(video_ids: list[str], comments_per_video: int = 30) -> list[dict]:
    """
    Generate dummy comments for a list of video IDs.
    Sentiment distribution: ~60% positive, 20% negative, 20% neutral.
    """
    all_comments = []
    pools = (
        COMMENT_POSITIVES * 6 +   # weight positive heavier
        COMMENT_NEGATIVES * 2 +
        COMMENT_NEUTRALS  * 2
    )

    for vid in video_ids:
        for _ in range(comments_per_video):
            all_comments.append({
                "comment_id":      _random_id(26),
                "video_id":        vid,
                "author":          f"User_{_random_id(6)}",
                "text":            rng.choice(pools),
                "like_count":      rng.randint(0, 500),
                "published_at":    _random_date(365),
                "sentiment_label": None,   # filled by NLP module
                "sentiment_score": None,
            })

    return all_comments
