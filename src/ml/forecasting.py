"""
src/ml/forecasting.py
──────────────────────
PHASE 5 – Trend Forecasting Module

What it does:
  • Aggregates video performance into a daily/weekly time series.
  • Fits Facebook Prophet to forecast views, likes, engagement for the next 90 days.
  • Falls back to a statsmodels Exponential Smoothing (Holt-Winters) if Prophet
    is not installed — so the project always works.
  • Performs keyword trend analysis using TF-IDF over rolling time windows.
  • Outputs forecast DataFrames ready for Plotly visualisation.

Why Prophet?
  • Handles YouTube-style data perfectly: seasonal spikes, trend changes,
    missing days, rapid growth curves.
  • Zero feature engineering required for time-series forecasting.
  • Confidence intervals give stakeholders a realistic range, not a single line.

Run:
    python -m src.ml.forecasting
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)


# ── Time-series builder ───────────────────────────────────────────────────────

def build_timeseries(df: pd.DataFrame, metric: str = "view_count") -> pd.DataFrame:
    """
    Aggregate video data into a daily time series for Prophet.

    Prophet expects two columns: ds (date) and y (value).
    We sum the metric for all videos published on each date.
    """
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["ds"] = df["published_at"].dt.date

    ts = (
        df.groupby("ds")[metric]
        .sum()
        .reset_index()
        .rename(columns={metric: "y"})
    )
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.sort_values("ds").reset_index(drop=True)

    # Remove obvious zeros (upload gaps are fine; zero-view days are noise)
    ts = ts[ts["y"] > 0]
    return ts


# ── Prophet forecaster ────────────────────────────────────────────────────────

def _forecast_prophet(ts: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Fit Prophet and return forecast DataFrame."""
    from prophet import Prophet  # deferred import — optional dependency

    model = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = True,
        daily_seasonality   = False,
        changepoint_prior_scale = 0.3,   # flexibility of trend changes
        seasonality_prior_scale = 10,
        interval_width      = 0.80,      # 80% confidence interval
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(ts)

    future   = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Clip negatives (views can't be negative)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ── Exponential Smoothing fallback ────────────────────────────────────────────

def _forecast_ets(ts: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Holt-Winters Exponential Smoothing fallback when Prophet is unavailable
    or dataset is too small.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ts = ts.set_index("ds")["y"]

    # Need at least 2× seasonal period
    seasonal = "add" if len(ts) >= 14 else None
    sp       = 7 if seasonal else 1

    model = ExponentialSmoothing(
        ts,
        trend    = "add",
        seasonal = seasonal,
        seasonal_periods = sp,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(optimized=True)

    pred       = fitted.forecast(periods)
    last_date  = ts.index[-1]
    future_idx = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq="D")

    # Build a combined historical + forecast frame
    hist_df = pd.DataFrame({"ds": ts.index, "yhat": ts.values,
                             "yhat_lower": ts.values * 0.85,
                             "yhat_upper": ts.values * 1.15})
    pred_df = pd.DataFrame({"ds": future_idx, "yhat": pred.values,
                             "yhat_lower": pred.values * 0.80,
                             "yhat_upper": pred.values * 1.20})
    return pd.concat([hist_df, pred_df], ignore_index=True)


# ── Main forecast function ────────────────────────────────────────────────────

def run_forecast(
    df: pd.DataFrame,
    metric: str   = "view_count",
    periods: int  = 90,
    freq: str     = "D",
) -> dict:
    """
    Forecast `metric` for the next `periods` days.

    Returns dict with keys:
        ts          – historical time series (ds, y)
        forecast    – full forecast (ds, yhat, yhat_lower, yhat_upper)
        future_only – only the forecast portion
        engine      – "prophet" | "ets"
        metric      – name of the metric
    """
    log.info(f"Forecasting '{metric}' for next {periods} days …")

    ts = build_timeseries(df, metric)

    if len(ts) < 3:
        log.warning(f"Only {len(ts)} data points for '{metric}'. "
                    "Need more videos for reliable forecasting. Generating synthetic series.")
        # Synthesise a plausible series from the single available point
        base_val  = float(ts["y"].iloc[0]) if len(ts) > 0 else 1000
        dates     = pd.date_range(end=datetime.now(timezone.utc).date(), periods=30, freq="D")
        ts        = pd.DataFrame({
            "ds": dates,
            "y":  (base_val * np.random.lognormal(0, 0.3, 30)).clip(1),
        })

    engine = "ets"
    try:
        import prophet  # noqa: F401 — check availability
        if len(ts) >= 10:
            forecast = _forecast_prophet(ts, periods, freq)
            engine   = "prophet"
        else:
            log.warning("Too few points for Prophet (<10). Using ETS fallback.")
            forecast = _forecast_ets(ts, periods)
    except ImportError:
        log.warning("Prophet not installed — using Exponential Smoothing fallback.")
        forecast = _forecast_ets(ts, periods)
    except Exception as e:
        log.warning(f"Prophet failed ({e}) — falling back to ETS.")
        forecast = _forecast_ets(ts, periods)

    # Identify which rows are future
    last_real = ts["ds"].max()
    future_df = forecast[forecast["ds"] > last_real].copy()

    log.info(f"Forecast complete using {engine}. "
             f"Next {periods} days: "
             f"avg={future_df['yhat'].mean():.0f}, "
             f"peak={future_df['yhat'].max():.0f}")

    return {
        "ts":          ts,
        "forecast":    forecast,
        "future_only": future_df,
        "engine":      engine,
        "metric":      metric,
    }


# ── Keyword trend analysis ────────────────────────────────────────────────────

def analyse_keyword_trends(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Split videos into early vs recent halves.
    Extract TF-IDF keywords from each half.
    Identify keywords that are rising (newer > older) or falling.

    Returns DataFrame with columns:
        keyword, old_score, new_score, trend, change_pct
    """
    log.info("Analysing keyword trends …")

    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at").dropna(subset=["published_at"])

    if len(df) < 4:
        log.warning("Too few videos for keyword trend analysis.")
        return pd.DataFrame()

    mid    = len(df) // 2
    old_df = df.iloc[:mid]
    new_df = df.iloc[mid:]

    def _extract_keywords(frame: pd.DataFrame) -> dict[str, float]:
        texts = frame["title"].fillna("").tolist()
        if not any(texts):
            return {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec   = TfidfVectorizer(stop_words="english", max_features=50, ngram_range=(1,2))
            mat   = vec.fit_transform(texts)
            names = vec.get_feature_names_out()
            scores = mat.toarray().mean(axis=0)
            return dict(zip(names, scores))
        except Exception:
            words = " ".join(texts).lower().split()
            freq  = Counter(words)
            total = max(sum(freq.values()), 1)
            return {w: c/total for w, c in freq.most_common(50)}

    old_kw = _extract_keywords(old_df)
    new_kw = _extract_keywords(new_df)

    all_keys = set(old_kw) | set(new_kw)
    rows = []
    for kw in all_keys:
        old_s = old_kw.get(kw, 0.0)
        new_s = new_kw.get(kw, 0.0)
        change = ((new_s - old_s) / max(old_s, 1e-6)) * 100
        trend  = "🔥 Rising" if change > 20 else ("📉 Falling" if change < -20 else "➡ Stable")
        rows.append({
            "keyword":    kw,
            "old_score":  round(old_s, 5),
            "new_score":  round(new_s, 5),
            "trend":      trend,
            "change_pct": round(change, 1),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("change_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    log.info(f"Keyword trends computed for {len(result)} terms.")
    return result


# ── Engagement velocity ───────────────────────────────────────────────────────

def compute_engagement_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute month-over-month growth rates for views, likes, comments.
    Useful for dashboard sparklines and 'channel momentum' KPI.
    """
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["month"] = df["published_at"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("month")
        .agg(
            views    =("view_count",     "sum"),
            likes    =("like_count",     "sum"),
            comments =("comment_count",  "sum"),
            videos   =("video_id",       "count"),
        )
        .reset_index()
        .sort_values("month")
    )

    for col in ["views", "likes", "comments"]:
        monthly[f"{col}_growth_%"] = monthly[col].pct_change() * 100

    return monthly


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"

    df = pd.read_csv(path)

    # Run forecasts
    for metric in ["view_count", "like_count"]:
        result = run_forecast(df, metric=metric, periods=90)
        fut    = result["future_only"]
        print(f"\n── {metric} Forecast (next 90 days) [{result['engine']}] ──")
        print(f"  Avg : {fut['yhat'].mean():,.0f}")
        print(f"  Peak: {fut['yhat'].max():,.0f}")
        print(f"  Low : {fut['yhat_lower'].min():,.0f}")

    # Keyword trends
    trends = analyse_keyword_trends(df)
    if not trends.empty:
        print("\n── Rising Keywords ──")
        print(trends[trends["trend"].str.contains("Rising")].head(5).to_string(index=False))

    # Velocity
    vel = compute_engagement_velocity(df)
    print("\n── Monthly Engagement Velocity ──")
    print(vel.tail(6).to_string(index=False))
