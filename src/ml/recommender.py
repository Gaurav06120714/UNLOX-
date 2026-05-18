import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)


class Recommendation:
    def __init__(self, category, action, detail, confidence, impact, evidence):
        self.category   = category
        self.action     = action
        self.detail     = detail
        self.confidence = confidence
        self.impact     = impact
        self.evidence   = evidence


class RecommendationReport:
    def __init__(self, generated_at, total_videos, top_videos_n, recommendations=None, weekly_brief=""):
        self.generated_at    = generated_at
        self.total_videos    = total_videos
        self.top_videos_n    = top_videos_n
        self.recommendations = recommendations or []
        self.weekly_brief    = weekly_brief

    def top(self, n=5):
        return sorted(self.recommendations, key=lambda r: r.confidence, reverse=True)[:n]


def parse_duration_secs(dur):
    if not isinstance(dur, str):
        return 0
    m = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur)
    if not m:
        return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def get_impact_label(confidence):
    if confidence >= 0.70:
        return "High"
    if confidence >= 0.45:
        return "Medium"
    return "Low"


def calc_confidence(lift_pct, n_top):
    base       = min(abs(lift_pct) / 200, 0.70)
    size_bonus = min(n_top / 20, 0.25)
    return round(min(base + size_bonus, 0.95), 3)


def posting_time_recs(top, all_videos):
    recs = []

    if "hour" in top.columns and len(top) >= 3:
        top_hours = top["hour"].dropna()
        all_hours = all_videos["hour"].dropna()
        best_hour = int(top_hours.mode().iloc[0]) if len(top_hours) else 10
        avg_hour  = all_hours.mean()
        lift      = abs(best_hour - avg_hour) / max(avg_hour, 1) * 100
        conf      = calc_confidence(lift, len(top))
        recs.append(Recommendation(
            category   = "Posting Time",
            action     = f"Publish videos at {best_hour:02d}:00 UTC",
            detail     = (f"Your top {len(top)} videos were mostly published around "
                          f"{best_hour:02d}:00 UTC, compared to channel avg of {avg_hour:.0f}:00."),
            confidence = conf,
            impact     = get_impact_label(conf),
            evidence   = f"Top modal hour: {best_hour}h | Channel avg: {avg_hour:.1f}h",
        ))

    if "day_of_week" in top.columns and len(top) >= 3:
        day_counts = top["day_of_week"].value_counts()
        best_day   = day_counts.index[0] if len(day_counts) else "Wednesday"
        count      = day_counts.iloc[0]
        conf       = calc_confidence(count / max(len(top), 1) * 100, len(top))
        recs.append(Recommendation(
            category   = "Posting Time",
            action     = f"Post on {best_day}s for maximum reach",
            detail     = f"{best_day} appears {count} times in your top {len(top)} videos.",
            confidence = conf,
            impact     = "High" if count >= 3 else "Medium",
            evidence   = str(day_counts.head(3).to_dict()),
        ))

    return recs


def video_length_recs(top, bottom):
    recs = []
    if "duration_secs" not in top.columns:
        return recs

    top_avg = top["duration_secs"].mean() / 60
    bot_avg = bottom["duration_secs"].mean() / 60
    lift    = (top_avg - bot_avg) / max(bot_avg, 1) * 100

    if abs(lift) > 10:
        direction = "longer" if top_avg > bot_avg else "shorter"
        conf      = calc_confidence(abs(lift), len(top))
        recs.append(Recommendation(
            category   = "Video Length",
            action     = f"Aim for ~{int(top_avg)} minute videos ({direction} than average)",
            detail     = f"Top videos avg {top_avg:.1f} min vs {bot_avg:.1f} min for lower performers.",
            confidence = conf,
            impact     = get_impact_label(conf),
            evidence   = f"Top avg: {top_avg:.1f} min | Bottom avg: {bot_avg:.1f} min | Lift={lift:+.1f}%",
        ))

    return recs


def tag_recs(top, all_videos):
    recs = []
    if "tags" not in top.columns:
        return recs

    top_tag_count = top["tags"].fillna("").apply(lambda t: len([x for x in t.split("|") if x]))
    all_tag_count = all_videos["tags"].fillna("").apply(lambda t: len([x for x in t.split("|") if x]))
    top_mean      = top_tag_count.mean()
    all_mean      = all_tag_count.mean()
    lift          = (top_mean - all_mean) / max(all_mean, 1) * 100
    conf          = calc_confidence(abs(lift), len(top))

    recs.append(Recommendation(
        category   = "Tag Strategy",
        action     = f"Use around {int(round(top_mean))} tags per video",
        detail     = f"Top videos use {top_mean:.1f} tags on average vs {all_mean:.1f} for all videos.",
        confidence = conf,
        impact     = get_impact_label(conf),
        evidence   = f"Top avg tags: {top_mean:.1f} | Channel avg: {all_mean:.1f} | Lift={lift:+.1f}%",
    ))

    all_tags = []
    for tag_str in top["tags"].fillna(""):
        all_tags.extend([t.strip().lower() for t in tag_str.split("|") if t.strip()])
    best_tags = [tag for tag, _ in Counter(all_tags).most_common(8)]

    if best_tags:
        recs.append(Recommendation(
            category   = "Tag Strategy",
            action     = f"Use these tags in every video: {', '.join(best_tags[:5])}",
            detail     = "These tags appear most in your highest-performing videos.",
            confidence = min(0.65 + len(top) * 0.01, 0.90),
            impact     = "High",
            evidence   = f"Most frequent tags in top videos: {best_tags}",
        ))

    return recs


def topic_recs(top):
    recs = []
    if "title" not in top.columns or len(top) < 3:
        return recs

    titles = top["title"].fillna("").tolist()

    try:
        tfidf    = TfidfVectorizer(stop_words="english", max_features=20, ngram_range=(1, 2))
        tfidf.fit(titles)
        keywords = tfidf.get_feature_names_out().tolist()
    except Exception:
        words     = " ".join(titles).lower().split()
        stopwords = {"the","a","an","in","on","of","is","to","for","and","with","my","i"}
        keywords  = [w for w, _ in Counter(w for w in words if w not in stopwords).most_common(10)]

    if keywords:
        recs.append(Recommendation(
            category   = "Topic & Keywords",
            action     = f"Focus content on: {', '.join(keywords[:6])}",
            detail     = "These keywords were extracted from your best-performing video titles using TF-IDF.",
            confidence = min(0.60 + len(top) * 0.015, 0.90),
            impact     = "High",
            evidence   = f"TF-IDF keywords: {keywords[:10]}",
        ))

    return recs


def engagement_recs(top, all_videos):
    recs = []

    if "engagement_rate" in top.columns:
        top_er = top["engagement_rate"].mean()
        all_er = all_videos["engagement_rate"].mean()
        lift   = (top_er - all_er) / max(all_er, 1) * 100
        recs.append(Recommendation(
            category   = "Engagement Strategy",
            action     = "Add a clear call-to-action in the first 30 seconds",
            detail     = (f"Top videos get {top_er*100:.2f}% engagement vs {all_er*100:.2f}% channel avg. "
                          "Asking viewers to like/comment early improves this significantly."),
            confidence = 0.78,
            impact     = "High",
            evidence   = f"Top ER: {top_er*100:.2f}% | Channel ER: {all_er*100:.2f}% | Lift={lift:+.1f}%",
        ))

    if "like_ratio" in top.columns:
        top_lr = top["like_ratio"].mean()
        recs.append(Recommendation(
            category   = "Engagement Strategy",
            action     = f"Target a like ratio of {top_lr*100:.1f}%",
            detail     = "Explicitly asking for likes mid-video increases the like ratio by ~30%.",
            confidence = 0.72,
            impact     = "Medium",
            evidence   = f"Top avg like ratio: {top_lr*100:.2f}%",
        ))

    return recs


def build_weekly_brief(report):
    top5 = report.top(5)
    now  = datetime.now(timezone.utc).strftime("%B %d, %Y")
    lines = [
        f"WEEKLY CONTENT STRATEGY BRIEF - {now}",
        f"{'='*55}",
        f"Channel analysed : {report.total_videos} videos",
        f"Top performers   : {report.top_videos_n} videos (top 25%)",
        f"{'='*55}",
        f"TOP {len(top5)} RECOMMENDATIONS:",
        f"{'='*55}",
    ]
    for i, r in enumerate(top5, 1):
        lines.append(f"{i}. [{r.impact}] {r.action}")
    lines += [
        f"{'='*55}",
        "ACTION PLAN FOR THIS WEEK:",
        "  - Update tags on existing videos",
        "  - Schedule next upload on recommended day/time",
        "  - Use top keywords in your next video title",
        "  - Add a like CTA at the 30-second mark",
        f"{'='*55}",
    ]
    return "\n".join(lines)


def generate_recommendations(df):
    log.info("Generating content recommendations")

    df = df.copy()
    df["published_at"]  = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["hour"]          = df["published_at"].dt.hour.fillna(12).astype(int)
    df["day_of_week"]   = df["published_at"].dt.day_name().fillna("Unknown")
    df["duration_secs"] = df["duration"].apply(parse_duration_secs)

    if "engagement_rate" not in df.columns:
        df["engagement_rate"] = (df["like_count"] + df["comment_count"]) / df["view_count"].clip(lower=1)
    if "like_ratio" not in df.columns:
        df["like_ratio"] = df["like_count"] / df["view_count"].clip(lower=1)

    if "viral_score" not in df.columns or df["viral_score"].isna().all():
        df["viral_score"] = df["engagement_rate"] * 100

    threshold = df["viral_score"].quantile(0.75)
    top_df    = df[df["viral_score"] >= threshold].copy()
    bottom_df = df[df["viral_score"] <  threshold].copy()

    log.info(f"Top quartile: {len(top_df)} videos (score >= {threshold:.1f})")

    all_recs = []
    all_recs.extend(posting_time_recs(top_df, df))
    all_recs.extend(video_length_recs(top_df, bottom_df))
    all_recs.extend(tag_recs(top_df, df))
    all_recs.extend(topic_recs(top_df))
    all_recs.extend(engagement_recs(top_df, df))
    all_recs.sort(key=lambda r: r.confidence, reverse=True)

    report = RecommendationReport(
        generated_at    = datetime.now(timezone.utc).isoformat(),
        total_videos    = len(df),
        top_videos_n    = len(top_df),
        recommendations = all_recs,
    )
    report.weekly_brief = build_weekly_brief(report)

    log.info(f"Generated {len(all_recs)} recommendations")
    return report


if __name__ == "__main__":
    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"

    df     = pd.read_csv(path)
    report = generate_recommendations(df)

    print(report.weekly_brief)
    print()
    for i, r in enumerate(report.recommendations, 1):
        print(f"{i}. [{r.impact} | conf={r.confidence:.2f}] {r.category}")
        print(f"   {r.action}")
        print(f"   Evidence: {r.evidence}")
        print()
