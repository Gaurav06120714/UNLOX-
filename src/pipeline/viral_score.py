import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import VIRAL_WEIGHTS, DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.utils.db import execute_write
from src.utils.logger import get_logger

log = get_logger(__name__)


def normalize(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_viral_scores(df):
    log.info("Calculating viral scores")
    w  = VIRAL_WEIGHTS
    df = df.copy()

    df["n_views"]      = normalize(df["view_count"].astype(float))
    df["n_likes"]      = normalize(df["like_count"].astype(float))
    df["n_comments"]   = normalize(df["comment_count"].astype(float))
    df["n_like_ratio"] = normalize(df["like_ratio"].astype(float))
    df["n_engagement"] = normalize(df["engagement_rate"].astype(float))

    df["viral_score"] = (
        w["views"]      * df["n_views"]      +
        w["likes"]      * df["n_likes"]      +
        w["comments"]   * df["n_comments"]   +
        w["like_ratio"] * df["n_like_ratio"] +
        w["engagement"] * df["n_engagement"]
    ) * 100

    df["viral_score"] = df["viral_score"].clip(0, 100).round(2)
    df.drop(columns=["n_views","n_likes","n_comments","n_like_ratio","n_engagement"], inplace=True)

    log.info(f"Viral score range: {df['viral_score'].min():.1f} - {df['viral_score'].max():.1f}")
    return df


def run_viral_scoring():
    raw_path = DATA_RAW_DIR / "videos.csv"
    if not raw_path.exists():
        log.error("videos.csv not found. Run extract.py first")
        raise FileNotFoundError(raw_path)

    df = pd.read_csv(raw_path)

    if "like_ratio" not in df.columns:
        df["like_ratio"]      = df["like_count"]  / df["view_count"].clip(1)
        df["engagement_rate"] = (df["like_count"] + df["comment_count"]) / df["view_count"].clip(1)

    df = compute_viral_scores(df)

    out = DATA_PROCESSED_DIR / "videos_scored.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved scored data to {out}")

    for _, row in df.iterrows():
        execute_write(
            "UPDATE videos SET viral_score = ? WHERE video_id = ?",
            (row["viral_score"], row["video_id"]),
        )

    return df


if __name__ == "__main__":
    df = run_viral_scoring()
    print(df[["title", "view_count", "like_count", "engagement_rate", "viral_score"]]
          .sort_values("viral_score", ascending=False)
          .head(10)
          .to_string(index=False))
