import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)


class ABTestResult:
    def __init__(self, test_name, metric, group_a_mean, group_b_mean,
                 t_stat, p_value, significant, winner, effect_size):
        self.test_name    = test_name
        self.metric       = metric
        self.group_a_mean = group_a_mean
        self.group_b_mean = group_b_mean
        self.t_stat       = t_stat
        self.p_value      = p_value
        self.significant  = significant
        self.winner       = winner
        self.effect_size  = effect_size


def parse_duration_secs(dur):
    if not isinstance(dur, str):
        return 0
    m = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur)
    if not m:
        return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def run_ab_test(df, group_col, metric_col, test_name, threshold=0.05):
    df = df.copy().dropna(subset=[group_col, metric_col])

    median_val = df[group_col].median()
    group_a    = df[df[group_col] >= median_val][metric_col].values
    group_b    = df[df[group_col] <  median_val][metric_col].values

    if len(group_a) < 2 or len(group_b) < 2:
        log.warning(f"Not enough data for test: {test_name}")
        return None

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    mean_a      = float(np.mean(group_a))
    mean_b      = float(np.mean(group_b))
    significant = bool(p_value < threshold)

    pooled_std  = np.sqrt((np.std(group_a)**2 + np.std(group_b)**2) / 2)
    effect_size = float(abs(mean_a - mean_b) / pooled_std) if pooled_std > 0 else 0.0

    if significant:
        winner = f"High {group_col}" if mean_a > mean_b else f"Low {group_col}"
    else:
        winner = "No significant difference"

    return ABTestResult(
        test_name    = test_name,
        metric       = metric_col,
        group_a_mean = round(mean_a, 4),
        group_b_mean = round(mean_b, 4),
        t_stat       = round(float(t_stat), 4),
        p_value      = round(float(p_value), 6),
        significant  = significant,
        winner       = winner,
        effect_size  = round(effect_size, 4),
    )


def run_all_ab_tests(df):
    log.info("Running A/B tests")

    df = df.copy()
    df["published_at"]  = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["hour"]          = df["published_at"].dt.hour.fillna(12).astype(int)
    df["duration_secs"] = df["duration"].apply(parse_duration_secs)

    if "engagement_rate" not in df.columns:
        df["engagement_rate"] = (df["like_count"] + df["comment_count"]) / df["view_count"].clip(1)
    if "tag_count" not in df.columns:
        df["tag_count"] = df["tags"].fillna("").apply(
            lambda t: len([x for x in t.split("|") if x])
        )
    if "desc_len" not in df.columns:
        df["desc_len"] = df["description"].fillna("").str.len()

    tests = [
        ("hour",          "viral_score",    "Posting Hour vs Viral Score"),
        ("duration_secs", "engagement_rate","Video Length vs Engagement Rate"),
        ("tag_count",     "viral_score",    "Tag Count vs Viral Score"),
        ("desc_len",      "viral_score",    "Description Length vs Viral Score"),
    ]

    results = []
    for group_col, metric_col, name in tests:
        result = run_ab_test(df, group_col, metric_col, name)
        if result:
            results.append(result)
            sig = "YES" if result.significant else "no"
            log.info(f"  {name}: p={result.p_value:.4f} | significant={sig} | winner={result.winner}")

    return results


if __name__ == "__main__":
    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"

    df = pd.read_csv(path)
    results = run_all_ab_tests(df)

    for r in results:
        print(f"\n{r.test_name}")
        print(f"  Group A mean : {r.group_a_mean:.4f}")
        print(f"  Group B mean : {r.group_b_mean:.4f}")
        print(f"  p-value      : {r.p_value:.6f}")
        print(f"  Significant  : {r.significant}")
        print(f"  Winner       : {r.winner}")
        print(f"  Effect size  : {r.effect_size:.4f}")
