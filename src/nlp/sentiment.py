"""
src/nlp/sentiment.py
─────────────────────
PHASE 3 – Audience Sentiment Analyser

Uses TextBlob for fast polarity scoring.
Labels: Positive | Neutral | Negative

For each comment:
  • polarity ∈ [-1.0, +1.0]
  • subjectivity ∈ [0.0, 1.0]  (bonus feature)
  • label: "Positive" if > 0.1, "Negative" if < -0.1, else "Neutral"

Run:
    python -m src.nlp.sentiment
"""

from __future__ import annotations

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

POS_THRESH =  0.10
NEG_THRESH = -0.10


def _label(score: float) -> str:
    if score > POS_THRESH:
        return "Positive"
    if score < NEG_THRESH:
        return "Negative"
    return "Neutral"


def analyse_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment_score and sentiment_label columns to a comments DataFrame.
    Input df must have a 'text' column.
    """
    log.info(f"Running sentiment analysis on {len(df)} comments …")
    df = df.copy()

    blobs = df["text"].fillna("").apply(TextBlob)
    df["sentiment_score"]     = blobs.apply(lambda b: round(b.sentiment.polarity,     4))
    df["sentiment_label"]     = df["sentiment_score"].apply(_label)
    df["subjectivity_score"]  = blobs.apply(lambda b: round(b.sentiment.subjectivity, 4))

    dist = df["sentiment_label"].value_counts().to_dict()
    log.info(f"Sentiment distribution: {dist}")
    return df


def run_sentiment_pipeline() -> pd.DataFrame:
    path = DATA_RAW_DIR / "comments.csv"
    if not path.exists():
        log.error("comments.csv not found. Run extract.py first.")
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty or "text" not in df.columns:
        log.warning("No comments to analyse — skipping sentiment stage.")
        return df
    df = analyse_sentiment(df)

    out = DATA_PROCESSED_DIR / "comments_sentiment.csv"
    df.to_csv(out, index=False)
    log.info(f"Sentiment data saved → {out}")

    # Push labels back to SQLite
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
