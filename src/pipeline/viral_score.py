"""
src/pipeline/viral_score.py
────────────────────────────
Computes a 0-100 viral score for every video using a
weighted, min-max normalised formula.

Formula (configurable in config/settings.py → VIRAL_WEIGHTS):
    raw_score = w_views * norm(views)
              + w_likes * norm(likes)
              + w_comments * norm(comments)
              + w_like_ratio * norm(like_ratio)
              + w_engagement * norm(engagement_rate)

    viral_score = raw_score * 100  (clamped 0-100)

Run:
    python -m src.pipeline.viral_score
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import VIRAL_WEIGHTS, DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.utils.db import bulk_insert, execute_query
from src.utils.logger import get_logger

log = get_logger(__name__)


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1]. Handles zero-range edge case."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_viral_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `viral_score` column (0-100) to a videos DataFrame.
    Expects columns: view_count, like_count, comment_count,
                     like_ratio, engagement_rate
    """
    log.info("Computing viral scores …")
    w = VIRAL_WEIGHTS

    df = df.copy()

    # Normalise each metric
    df["n_views"]      = _minmax(df["view_count"].astype(float))
    df["n_likes"]      = _minmax(df["like_count"].astype(float))
    df["n_comments"]   = _minmax(df["comment_count"].astype(float))
    df["n_like_ratio"] = _minmax(df["like_ratio"].astype(float))
    df["n_engagement"] = _minmax(df["engagement_rate"].astype(float))

    # Weighted sum → scale to 100
    df["viral_score"] = (
        w["views"]      * df["n_views"]      +
        w["likes"]      * df["n_likes"]      +
        w["comments"]   * df["n_comments"]   +
        w["like_ratio"] * df["n_like_ratio"] +
        w["engagement"] * df["n_engagement"]
    ) * 100

    df["viral_score"] = df["viral_score"].clip(0, 100).round(2)

    # Drop normalised helper columns
    df.drop(columns=["n_views","n_likes","n_comments","n_like_ratio","n_engagement"],
            inplace=True)

    log.info(f"Viral scores: min={df['viral_score'].min():.1f}, "
             f"max={df['viral_score'].max():.1f}, "
             f"mean={df['viral_score'].mean():.1f}")
    return df


def run_viral_scoring() -> pd.DataFrame:
    """Load raw videos, score them, save processed CSV + update DB."""
    raw_path = DATA_RAW_DIR / "videos.csv"
    if not raw_path.exists():
        log.error(f"{raw_path} not found. Run extract.py first.")
        raise FileNotFoundError(raw_path)

    df = pd.read_csv(raw_path)

    # Ensure derived cols exist
    for col in ["like_ratio", "engagement_rate"]:
        if col not in df.columns:
            df["like_ratio"]      = df["like_count"]  / df["view_count"].clip(1)
            df["engagement_rate"] = (df["like_count"] + df["comment_count"]) / df["view_count"].clip(1)
            break

    df = compute_viral_scores(df)

    # Save processed CSV
    out = DATA_PROCESSED_DIR / "videos_scored.csv"
    df.to_csv(out, index=False)
    log.info(f"Scored data saved → {out}")

    # Update DB viral_score column
    for _, row in df.iterrows():
        from src.utils.db import execute_write
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
