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

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH    = MODELS_DIR / "virality_model.joblib"
FEATURES_PATH = MODELS_DIR / "virality_features.joblib"


def parse_duration(dur):
    if not isinstance(dur, str):
        return 0
    m = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur)
    if not m:
        return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def build_features(df):
    df = df.copy()

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["hour"]         = df["published_at"].dt.hour.fillna(12).astype(int)
    df["day_of_week"]  = df["published_at"].dt.dayofweek.fillna(0).astype(int)

    df["duration_secs"] = df["duration"].apply(parse_duration)
    df["duration_mins"] = (df["duration_secs"] / 60).clip(lower=0.5)

    views = df["view_count"].clip(lower=1)
    df["like_ratio"]      = df["like_count"]    / views
    df["comment_ratio"]   = df["comment_count"] / views
    df["engagement_rate"] = (df["like_count"] + df["comment_count"]) / views

    df["views_per_minute"] = df["view_count"] / df["duration_mins"]

    df["tag_count"]        = df["tags"].fillna("").apply(
        lambda t: len([x for x in t.split("|") if x]) if t else 0
    )
    df["desc_len"]         = df["description"].fillna("").str.len()
    df["title_len"]        = df["title"].fillna("").str.len()
    df["title_word_count"] = df["title"].fillna("").str.split().str.len().fillna(0)

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


def train_model(df):
    log.info("Training virality prediction model")

    df        = build_features(df)
    df        = df.dropna(subset=["viral_score"])
    available = [c for c in FEATURE_COLS if c in df.columns]
    X         = df[available].fillna(0)
    y         = df["viral_score"]

    log.info(f"Training on {len(X)} videos with {len(available)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators     = 200,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            random_state     = RANDOM_STATE,
            verbosity        = 0,
        )),
    ])

    model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), 0, 100)

    metrics = {
        "rmse":    float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":     float(mean_absolute_error(y_test, y_pred)),
        "r2":      float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }

    cv_folds          = 3 if len(X) < 50 else 5
    cv_scores         = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")
    metrics["cv_r2_mean"] = float(cv_scores.mean())
    metrics["cv_r2_std"]  = float(cv_scores.std())

    log.info(f"RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.3f}")

    xgb_step     = model.named_steps["xgb"]
    importance_df = pd.DataFrame({
        "feature":    available,
        "importance": xgb_step.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    joblib.dump(model,     MODEL_PATH)
    joblib.dump(available, FEATURES_PATH)
    log.info(f"Model saved to {MODEL_PATH}")

    return {
        "model":              model,
        "feature_names":      available,
        "metrics":            metrics,
        "feature_importance": importance_df,
        "X_test":             X_test,
        "y_test":             y_test,
        "y_pred":             y_pred,
    }


def predict_viral_score(df):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train it first.")

    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    df = build_features(df)
    X  = df[features].fillna(0)
    df["predicted_viral_score"] = np.clip(model.predict(X), 0, 100).round(2)
    return df


def predict_single(view_count=0, like_count=0, comment_count=0,
                   duration_str="PT10M0S", tag_count=5, title="My Video",
                   description="", hour=12, day_of_week=2):
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


if __name__ == "__main__":
    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"

    df     = pd.read_csv(path)
    result = train_model(df)
    m      = result["metrics"]

    print(f"\nModel Performance:")
    print(f"  RMSE: {m['rmse']:.2f}")
    print(f"  MAE:  {m['mae']:.2f}")
    print(f"  R2:   {m['r2']:.3f}")
    print(f"  Cross-Val R2: {m['cv_r2_mean']:.3f} +/- {m['cv_r2_std']:.3f}")
    print("\nTop Features:")
    print(result["feature_importance"].head(10).to_string(index=False))

    score = predict_single(view_count=50000, like_count=3000, comment_count=150,
                           duration_str="PT12M30S", tag_count=8,
                           title="How I Learned Python in 30 Days")
    print(f"\nSample viral score: {score:.1f} / 100")
