import sys
from pathlib import Path

import pandas as pd
from textblob import TextBlob

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.utils.db import execute_write
from src.utils.logger import get_logger

log = get_logger(__name__)

POSITIVE_THRESHOLD =  0.10
NEGATIVE_THRESHOLD = -0.10


def get_label(score):
    if score > POSITIVE_THRESHOLD:
        return "Positive"
    if score < NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyse_sentiment(df):
    log.info(f"Running sentiment analysis on {len(df)} comments")
    df    = df.copy()
    blobs = df["text"].fillna("").apply(TextBlob)

    df["sentiment_score"]    = blobs.apply(lambda b: round(b.sentiment.polarity,     4))
    df["sentiment_label"]    = df["sentiment_score"].apply(get_label)
    df["subjectivity_score"] = blobs.apply(lambda b: round(b.sentiment.subjectivity, 4))

    dist = df["sentiment_label"].value_counts().to_dict()
    log.info(f"Sentiment distribution: {dist}")
    return df


def run_sentiment_pipeline():
    path = DATA_RAW_DIR / "comments.csv"
    if not path.exists():
        log.error("comments.csv not found. Run extract.py first")
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty or "text" not in df.columns:
        log.warning("No comments to analyse")
        return df

    df = analyse_sentiment(df)

    out = DATA_PROCESSED_DIR / "comments_sentiment.csv"
    df.to_csv(out, index=False)
    log.info(f"Sentiment results saved to {out}")

    for _, row in df.iterrows():
        execute_write(
            "UPDATE comments SET sentiment_label=?, sentiment_score=? WHERE comment_id=?",
            (row["sentiment_label"], row["sentiment_score"], row["comment_id"]),
        )

    return df


if __name__ == "__main__":
    df = run_sentiment_pipeline()
    print(df[["author", "text", "sentiment_score", "sentiment_label"]].head(10).to_string())
    print("\nDistribution:")
    print(df["sentiment_label"].value_counts())
