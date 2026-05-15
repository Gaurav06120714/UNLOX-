"""
run_pipeline.py
───────────────
Master orchestrator — runs every pipeline stage in order.

Usage:
    python run_pipeline.py                   # full pipeline
    python run_pipeline.py --stage extract
    python run_pipeline.py --stage score
    python run_pipeline.py --stage sentiment
    python run_pipeline.py --stage ab_test
    python run_pipeline.py --stage virality
    python run_pipeline.py --stage recommend
    python run_pipeline.py --stage forecast
    python run_pipeline.py --stage all
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

log = get_logger("pipeline")


def run_extract():
    log.info("▶  Stage 1: Data Extraction")
    from src.pipeline.extract import run_extraction
    run_extraction()


def run_score():
    log.info("▶  Stage 2: Viral Scoring")
    from src.pipeline.viral_score import run_viral_scoring
    run_viral_scoring()


def run_sentiment():
    log.info("▶  Stage 3: Sentiment Analysis")
    from src.nlp.sentiment import run_sentiment_pipeline
    run_sentiment_pipeline()


def run_ab_test():
    log.info("▶  Stage 4: A/B Testing")
    import pandas as pd
    from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
    from src.ml.ab_testing import run_all_ab_tests

    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"
    df = pd.read_csv(path)

    results = run_all_ab_tests(df)
    for r in results:
        log.info(f"  {r.test_name}: winner={r.winner}, p={r.p_value:.4f}")


def run_virality():
    log.info("▶  Stage 5: Virality ML Model Training")
    import pandas as pd
    from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
    from src.ml.virality_model import train_model

    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"
    df = pd.read_csv(path)

    result = train_model(df)
    m = result["metrics"]
    log.info(f"  RMSE={m['rmse']:.2f}, R²={m['r2']:.3f}, "
             f"CV-R²={m['cv_r2_mean']:.3f}±{m['cv_r2_std']:.3f}")


def run_recommend():
    log.info("▶  Stage 6: Recommendation Engine")
    import pandas as pd
    from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
    from src.ml.recommender import generate_recommendations

    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"
    df = pd.read_csv(path)

    report = generate_recommendations(df)
    log.info(f"  {len(report.recommendations)} recommendations generated.")
    for r in report.top(3):
        log.info(f"  [{r.impact}] {r.action[:65]}")


def run_forecast():
    log.info("▶  Stage 7: Trend Forecasting")
    import pandas as pd
    from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
    from src.ml.forecasting import run_forecast as _forecast, analyse_keyword_trends

    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"
    df = pd.read_csv(path)

    for metric in ["view_count", "like_count"]:
        res = _forecast(df, metric=metric, periods=90)
        fut = res["future_only"]
        log.info(f"  {metric}: avg={fut['yhat'].mean():,.0f}, "
                 f"peak={fut['yhat'].max():,.0f} [{res['engine']}]")

    trends = analyse_keyword_trends(df)
    if not trends.empty:
        rising = trends[trends["trend"].str.contains("Rising")]["keyword"].tolist()
        log.info(f"  Rising keywords: {rising[:5]}")


STAGES = {
    "extract":   run_extract,
    "score":     run_score,
    "sentiment": run_sentiment,
    "ab_test":   run_ab_test,
    "virality":  run_virality,
    "recommend": run_recommend,
    "forecast":  run_forecast,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data-Driven Social Engagement Pipeline")
    parser.add_argument(
        "--stage", default="all",
        choices=list(STAGES.keys()) + ["all"],
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    log.info("═" * 60)
    log.info("  Data-Driven Social Engagement Initiative")
    log.info(f"  Stage: {args.stage.upper()}")
    log.info("═" * 60)

    if args.stage == "all":
        for fn in STAGES.values():
            fn()
    else:
        STAGES[args.stage]()

    log.info("Pipeline finished. Launch dashboard with:")
    log.info("  streamlit run src/dashboard/app.py")
