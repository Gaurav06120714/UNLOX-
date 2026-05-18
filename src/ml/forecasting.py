import sys
import warnings
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)


def build_timeseries(df, metric="view_count"):
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
    ts = ts[ts["y"] > 0]
    return ts


def forecast_with_prophet(ts, periods, freq):
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality      = True,
        weekly_seasonality      = True,
        daily_seasonality       = False,
        changepoint_prior_scale = 0.3,
        seasonality_prior_scale = 10,
        interval_width          = 0.80,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(ts)

    future   = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def forecast_with_ets(ts, periods):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ts_series = ts.set_index("ds")["y"]

    seasonal = "add" if len(ts_series) >= 14 else None
    sp       = 7 if seasonal else 1

    model = ExponentialSmoothing(
        ts_series,
        trend            = "add",
        seasonal         = seasonal,
        seasonal_periods = sp,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(optimized=True)

    pred       = fitted.forecast(periods)
    last_date  = ts_series.index[-1]
    future_idx = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq="D")

    hist_df = pd.DataFrame({"ds": ts_series.index, "yhat": ts_series.values,
                             "yhat_lower": ts_series.values * 0.85,
                             "yhat_upper": ts_series.values * 1.15})
    pred_df = pd.DataFrame({"ds": future_idx, "yhat": pred.values,
                             "yhat_lower": pred.values * 0.80,
                             "yhat_upper": pred.values * 1.20})
    return pd.concat([hist_df, pred_df], ignore_index=True)


def run_forecast(df, metric="view_count", periods=90, freq="D"):
    log.info(f"Forecasting '{metric}' for next {periods} days")

    ts = build_timeseries(df, metric)

    if len(ts) < 3:
        log.warning(f"Only {len(ts)} data points - generating synthetic series")
        base  = float(ts["y"].iloc[0]) if len(ts) > 0 else 1000
        dates = pd.date_range(end=datetime.now(timezone.utc).date(), periods=30, freq="D")
        ts    = pd.DataFrame({
            "ds": dates,
            "y":  (base * np.random.lognormal(0, 0.3, 30)).clip(1),
        })

    engine = "ets"
    try:
        import prophet
        if len(ts) >= 10:
            forecast = forecast_with_prophet(ts, periods, freq)
            engine   = "prophet"
        else:
            log.warning("Too few data points for prophet, using ETS")
            forecast = forecast_with_ets(ts, periods)
    except ImportError:
        log.warning("Prophet not installed - using ETS instead")
        forecast = forecast_with_ets(ts, periods)
    except Exception as e:
        log.warning(f"Prophet failed ({e}) - using ETS instead")
        forecast = forecast_with_ets(ts, periods)

    last_real = ts["ds"].max()
    future_df = forecast[forecast["ds"] > last_real].copy()

    log.info(f"Forecast done using {engine}. Avg next {periods}d: {future_df['yhat'].mean():.0f}")

    return {
        "ts":          ts,
        "forecast":    forecast,
        "future_only": future_df,
        "engine":      engine,
        "metric":      metric,
    }


def analyse_keyword_trends(df, top_n=20):
    log.info("Analysing keyword trends in video titles")

    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at").dropna(subset=["published_at"])

    if len(df) < 4:
        return pd.DataFrame()

    mid    = len(df) // 2
    old_df = df.iloc[:mid]
    new_df = df.iloc[mid:]

    def get_keywords(frame):
        texts = frame["title"].fillna("").tolist()
        if not any(texts):
            return {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec    = TfidfVectorizer(stop_words="english", max_features=50, ngram_range=(1, 2))
            mat    = vec.fit_transform(texts)
            names  = vec.get_feature_names_out()
            scores = mat.toarray().mean(axis=0)
            return dict(zip(names, scores))
        except Exception:
            words = " ".join(texts).lower().split()
            freq  = Counter(words)
            total = max(sum(freq.values()), 1)
            return {w: c / total for w, c in freq.most_common(50)}

    old_kw = get_keywords(old_df)
    new_kw = get_keywords(new_df)

    all_keys = set(old_kw) | set(new_kw)
    rows = []
    for kw in all_keys:
        old_s  = old_kw.get(kw, 0.0)
        new_s  = new_kw.get(kw, 0.0)
        change = ((new_s - old_s) / max(old_s, 1e-6)) * 100
        trend  = "Rising" if change > 20 else ("Falling" if change < -20 else "Stable")
        rows.append({
            "keyword":    kw,
            "old_score":  round(old_s, 5),
            "new_score":  round(new_s, 5),
            "trend":      trend,
            "change_pct": round(change, 1),
        })

    result = (pd.DataFrame(rows)
               .sort_values("change_pct", ascending=False)
               .head(top_n)
               .reset_index(drop=True))
    return result


def compute_engagement_velocity(df):
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["month"]        = df["published_at"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("month")
        .agg(
            views    = ("view_count",    "sum"),
            likes    = ("like_count",    "sum"),
            comments = ("comment_count", "sum"),
            videos   = ("video_id",      "count"),
        )
        .reset_index()
        .sort_values("month")
    )

    for col in ["views", "likes", "comments"]:
        monthly[f"{col}_growth_%"] = monthly[col].pct_change() * 100

    return monthly


if __name__ == "__main__":
    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"

    df = pd.read_csv(path)

    for metric in ["view_count", "like_count"]:
        result = run_forecast(df, metric=metric, periods=90)
        fut    = result["future_only"]
        print(f"\n{metric} Forecast (next 90 days) [{result['engine']}]:")
        print(f"  Avg : {fut['yhat'].mean():,.0f}")
        print(f"  Peak: {fut['yhat'].max():,.0f}")

    trends = analyse_keyword_trends(df)
    if not trends.empty:
        print("\nRising Keywords:")
        print(trends[trends["trend"] == "Rising"].head(5).to_string(index=False))
