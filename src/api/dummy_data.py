import random
import string
from datetime import datetime, timedelta, timezone

import numpy as np

SEED   = 42
rng    = random.Random(SEED)
np_rng = np.random.default_rng(SEED)

VIDEO_TOPICS = [
    "Python Tutorial", "Machine Learning Crash Course", "Data Science Full Guide",
    "AI Tools You Need in 2024", "ChatGPT Masterclass", "Streamlit Dashboard Build",
    "YouTube Analytics Tips", "How to Go Viral on YouTube", "NLP with Python",
    "Deep Learning Explained Simply", "SQL for Beginners", "Pandas vs Polars",
    "Data Analyst Portfolio Project", "Power BI vs Tableau", "Freelancing as a Data Scientist",
]

ALL_TAGS = [
    "python", "machinelearning", "datascience", "ai", "tutorial",
    "streamlit", "nlp", "analytics", "youtube", "viral", "beginner",
    "sql", "pandas", "deeplearning", "portfolio",
]

POSITIVE_COMMENTS = [
    "This video is absolutely amazing! I learned so much.",
    "Best tutorial on this topic I have ever seen. Thank you!",
    "You explain things so clearly. Subscribed immediately!",
    "This is exactly what I needed. Saved me hours of searching.",
    "Brilliant content as always. Keep it up!",
    "I watched this three times. So valuable.",
    "Great job! Very well structured and easy to follow.",
]

NEGATIVE_COMMENTS = [
    "Honestly disappointed. Expected more depth.",
    "The audio quality is really bad in this one.",
    "Too basic for me. I already knew all of this.",
    "Please add timestamps. Hard to navigate.",
    "Some of the information here is outdated.",
]

NEUTRAL_COMMENTS = [
    "At what timestamp do you cover the installation?",
    "Can you do a follow-up video on this topic?",
    "What laptop are you using in this video?",
    "This is for Python 3.10 or 3.11?",
    "Does this work on Windows as well?",
    "Is the source code available somewhere?",
]


def random_id(length=11):
    return "".join(rng.choices(string.ascii_letters + string.digits, k=length))


def random_date(days_back=365):
    offset = timedelta(days=rng.randint(0, days_back))
    dt = datetime.now(timezone.utc) - offset
    return dt.isoformat()


def random_duration():
    mins = rng.randint(5, 45)
    secs = rng.randint(0, 59)
    return f"PT{mins}M{secs}S"


def generate_dummy_videos(n=50):
    videos      = []
    view_counts = np_rng.lognormal(mean=10.5, sigma=1.5, size=n).astype(int)

    for i in range(n):
        views    = int(view_counts[i])
        likes    = int(views * rng.uniform(0.02, 0.12))
        comments = int(views * rng.uniform(0.001, 0.03))
        topic    = rng.choice(VIDEO_TOPICS)
        tags     = rng.sample(ALL_TAGS, k=rng.randint(3, 8))

        videos.append({
            "video_id":      random_id(),
            "title":         f"{topic} – Part {i+1}",
            "description":   f"In this video we cover {topic.lower()} in detail. Like and subscribe for more content!",
            "published_at":  random_date(730),
            "channel_id":    "UC_DUMMY_CHANNEL_001",
            "channel_title": "DataScienceWithGaurav",
            "view_count":    views,
            "like_count":    likes,
            "comment_count": comments,
            "duration":      random_duration(),
            "tags":          "|".join(tags),
            "thumbnail_url": f"https://i.ytimg.com/vi/{random_id()}/hqdefault.jpg",
            "viral_score":   0.0,
            "fetched_at":    datetime.now(timezone.utc).isoformat(),
        })

    return videos


def generate_dummy_comments(video_ids, comments_per_video=30):
    all_comments = []
    comment_pool = (
        POSITIVE_COMMENTS * 6 +
        NEGATIVE_COMMENTS * 2 +
        NEUTRAL_COMMENTS  * 2
    )

    for vid in video_ids:
        for _ in range(comments_per_video):
            all_comments.append({
                "comment_id":      random_id(26),
                "video_id":        vid,
                "author":          f"User_{random_id(6)}",
                "text":            rng.choice(comment_pool),
                "like_count":      rng.randint(0, 500),
                "published_at":    random_date(365),
                "sentiment_label": None,
                "sentiment_score": None,
            })

    return all_comments
