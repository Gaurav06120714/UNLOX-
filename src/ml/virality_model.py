"""
src/ml/virality_model.py
─────────────────────────
PHASE 4 – Virality Prediction ML Model

Architecture:
  • Feature engineering from raw video metadata
  • XGBoost Regressor trained to predict viral_score (0-100)
  • Evaluation: RMSE, MAE, R²
  • Feature importance chart
  • SHAP-style manual importance analysis
  • Model persisted to models/virality_model.joblib

Why XGBoost?
  • Handles non-linear relationships between views, likes, tags etc.
  • Robust to outliers (viral videos are extreme outliers by nature)
  • Fast training even on small datasets
  • Built-in feature importance — explainable for stakeholders

Run:
    python -m src.ml.virality_model
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_PROCESSED_DIR, DATA_RAW_DIR,
    MODELS_DIR, RANDOM_STATE, TEST_SIZE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = MODELS_DIR / "virality_model.joblib"
FEATURES_PATH = MODELS_DIR / "virality_features.joblib"


# ── Feature engineering ───────────────────────────────────────────────────────

def _parse_duration(dur: str) -> int:
    """Convert ISO 8601 duration (PT5M30S) to total seconds."""
    if not isinstance(dur, str):
        return 0
    m = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur)
    if not m:
        return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML feature matrix from raw video data.

    Features created:
      Numerical  : view_count, like_count, comment_count, duration_secs,
                   like_ratio, comment_ratio, engagement_rate,
                   tag_count, desc_len, title_len, title_word_count
      Categorical : hour_of_day (0-23), day_of_week (0-6)
      Interaction : views_per_minute (views / duration_mins)
    """
    df = df.copy()

    # Parse dates
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["hour"]         = df["published_at"].dt.hour.fillna(12).astype(int)
    df["day_of_week"]  = df["published_at"].dt.dayofweek.fillna(0).astype(int)

    # Duration
    df["duration_secs"] = df["duration"].apply(_parse_duration)
    df["duration_mins"] = (df["duration_secs"] / 60).clip(lower=0.5)

    # Engagement ratios (guard against 0 views)
    views = df["view_count"].clip(lower=1)
    df["like_ratio"]       = df["like_count"]    / views
    df["comment_ratio"]    = df["comment_count"]  / views
    df["engagement_rate"]  = (df["like_count"] + df["comment_count"]) / views

    # Views per minute of content (efficiency metric)
    df["views_per_minute"] = df["view_count"] / df["duration_mins"]

    # Text features
    df["tag_count"]        = df["tags"].fillna("").apply(
        lambda t: len([x for x in t.split("|") if x]) if t else 0
    )
    df["desc_len"]         = df["description"].fillna("").str.len()
    df["title_len"]        = df["title"].fillna("").str.len()
    df["title_word_count"] = df["title"].fillna("").str.split().str.len().fillna(0)

    # Log-transform skewed counts (viral videos have huge view counts)
    for col in ["view_count", "like_count", "comment_count", "views_per_minute"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


FEATURE_COLS = [
    "log_view_count", "log_like_count", "log_comment_count",
    "like_ratio", "comment_ratio", "engagement_rate",
    "duration_secs", "duration_mins", "views_per_minute", "log_views_per_minute",
    "tag_count", "desc_len", "title_len", "title_word_count",
    "hour", "day_of_week",
]

TARGET_COL = "viral_score"


# ── Model training ────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> dict:
    """
    Train XGBoost regressor on viral_score.

    Returns a dict with:
        model, feature_names, metrics (dict), feature_importance (DataFrame)
    """
    log.info("Starting virality model training …")

    df = engineer_features(df)

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    # Use only available feature cols
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET_COL]

    log.info(f"Training data: {len(X)} rows × {len(available)} features")

    if len(X) < 10:
        log.warning("Very few training samples — model may not generalise well. "
                    "Upload more videos to improve accuracy.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Build pipeline: scale → XGBoost
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,       # L1 regularisation (prevents overfitting)
            reg_lambda=1.0,      # L2 regularisation
            random_state=RANDOM_STATE,
            verbosity=0,
        )),
    ])

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 100)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "r2":   float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }

    # Cross-validation (3-fold for small datasets, 5-fold for larger)
    cv_folds = 3 if len(X) < 50 else 5
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")
    metrics["cv_r2_mean"] = float(cv_scores.mean())
    metrics["cv_r2_std"]  = float(cv_scores.std())

    log.info(f"Model metrics: RMSE={metrics['rmse']:.2f}, "
             f"MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
    log.info(f"Cross-val R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")

    # Feature importance from XGBoost
    xgb_model = model.named_steps["xgb"]
    importance = pd.DataFrame({
        "feature":    available,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save model and feature list
    joblib.dump(model, MODEL_PATH)
    joblib.dump(available, FEATURES_PATH)
    log.info(f"Model saved → {MODEL_PATH}")

    return {
        "model":              model,
        "feature_names":      available,
        "metrics":            metrics,
        "feature_importance": importance,
        "X_test":             X_test,
        "y_test":             y_test,
        "y_pred":             y_pred,
    }


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_viral_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load saved model and predict viral_score for new/unseen videos.
    Adds 'predicted_viral_score' column to df.
    """
    if not MODEL_PATH.exists():
        log.error("Model not found. Run train_model() first.")
        raise FileNotFoundError(MODEL_PATH)

    model: Pipeline = joblib.load(MODEL_PATH)
    features: list  = joblib.load(FEATURES_PATH)

    df = engineer_features(df)
    X  = df[features].fillna(0)
    df["predicted_viral_score"] = np.clip(model.predict(X), 0, 100).round(2)
    return df


def predict_single(
    view_count: int = 0,
    like_count: int = 0,
    comment_count: int = 0,
    duration_str: str = "PT10M0S",
    tag_count: int = 5,
    title: str = "My Video",
    description: str = "",
    hour: int = 12,
    day_of_week: int = 2,
) -> float:
    """Predict viral score for a single hypothetical video."""
    row = pd.DataFrame([{
        "view_count":    view_count,
        "like_count":    like_count,
        "comment_count": comment_count,
        "duration":      duration_str,
        "tags":          "|".join(["tag"] * tag_count),
        "title":         title,
        "description":   description,
        "published_at":  f"2024-01-01T{hour:02d}:00:00+00:00",
    }])
    result = predict_viral_score(row)
    return float(result["predicted_viral_score"].iloc[0])


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"

    df = pd.read_csv(path)
    result = train_model(df)

    print("\n── Model Performance ─────────────────────────")
    m = result["metrics"]
    print(f"  RMSE         : {m['rmse']:.2f} (lower is better)")
    print(f"  MAE          : {m['mae']:.2f}")
    print(f"  R² Score     : {m['r2']:.3f}  (1.0 = perfect)")
    print(f"  Cross-Val R² : {m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f}")
    print(f"  Train / Test : {m['n_train']} / {m['n_test']} samples")

    print("\n── Top 10 Features ───────────────────────────")
    print(result["feature_importance"].head(10).to_string(index=False))

    print("\n── Sample Prediction ─────────────────────────")
    score = predict_single(
        view_count=50000, like_count=3000, comment_count=150,
        duration_str="PT12M30S", tag_count=8,
        title="How I Learned Python in 30 Days",
    )
    print(f"  Hypothetical video viral score: {score:.1f} / 100")
