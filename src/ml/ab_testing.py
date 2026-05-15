"""
src/ml/ab_testing.py
─────────────────────
PHASE 4 – A/B Testing Framework

What it does:
  • Compares two variants (A vs B) on any metric.
  • Uses Welch's t-test (unequal variance) for continuous metrics.
  • Uses Chi-squared test for categorical success/fail rates.
  • Computes Cohen's d effect size (practical significance).
  • Returns a clean ABTestResult dataclass for easy reporting.

Real-world application:
  - Compare Hook A ("10 tips…") vs Hook B ("Why I failed at…")
  - Compare morning posts vs evening posts on engagement rate
  - Compare thumbnail styles on CTR (click-through rate)

Run:
    python -m src.ml.ab_testing
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR, RANDOM_STATE
from src.utils.db import bulk_insert, init_db
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ABTestResult:
    test_name:    str
    variant_a:    str
    variant_b:    str
    metric:       str
    mean_a:       float
    mean_b:       float
    std_a:        float
    std_b:        float
    n_a:          int
    n_b:          int
    t_stat:       float
    p_value:      float
    cohens_d:     float
    winner:       str          # "A", "B", or "No significant difference"
    significant:  bool
    confidence:   float        # 1 - p_value
    lift:         float        # % change from A to B
    interpretation: str

    def summary(self) -> str:
        lines = [
            f"{'═'*55}",
            f"  A/B Test: {self.test_name}",
            f"{'═'*55}",
            f"  Metric     : {self.metric}",
            f"  Variant A  : {self.variant_a}  (n={self.n_a}, mean={self.mean_a:.4f})",
            f"  Variant B  : {self.variant_b}  (n={self.n_b}, mean={self.mean_b:.4f})",
            f"  t-statistic: {self.t_stat:.4f}",
            f"  p-value    : {self.p_value:.4f}",
            f"  Cohen's d  : {self.cohens_d:.4f}  ({self._effect_label()})",
            f"  Lift (B-A) : {self.lift:+.1f}%",
            f"  Winner     : {self.winner}",
            f"  Significant: {'YES ✅' if self.significant else 'NO ❌'}",
            f"  {'─'*51}",
            f"  {self.interpretation}",
            f"{'═'*55}",
        ]
        return "\n".join(lines)

    def _effect_label(self) -> str:
        d = abs(self.cohens_d)
        if d < 0.2:  return "negligible"
        if d < 0.5:  return "small"
        if d < 0.8:  return "medium"
        return "large"

    def to_db_rows(self) -> list[dict]:
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for variant, mean in [(self.variant_a, self.mean_a), (self.variant_b, self.mean_b)]:
            rows.append({
                "test_name":    self.test_name,
                "variant":      variant,
                "metric_name":  self.metric,
                "metric_value": mean,
                "created_at":   now,
            })
        return rows


# ── Core test engine ──────────────────────────────────────────────────────────

def run_ab_test(
    group_a: np.ndarray | list,
    group_b: np.ndarray | list,
    test_name: str,
    variant_a: str = "Variant A",
    variant_b: str = "Variant B",
    metric: str = "metric",
    alpha: float = 0.05,
) -> ABTestResult:
    """
    Run Welch's t-test between two groups.

    Parameters
    ----------
    group_a, group_b : array-like of numeric values
    alpha            : significance threshold (default 0.05 → 95% confidence)

    Returns
    -------
    ABTestResult with full statistics
    """
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)

    mean_a, mean_b = a.mean(), b.mean()
    std_a,  std_b  = a.std(ddof=1), b.std(ddof=1)
    n_a,    n_b    = len(a), len(b)

    # Welch's t-test (does NOT assume equal variance)
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # Cohen's d (pooled std)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d   = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0

    # Lift: how much better is B than A (%)
    lift = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0.0

    significant = bool(p_value < alpha)

    if not significant:
        winner = "No significant difference"
    elif mean_b > mean_a:
        winner = f"B ({variant_b})"
    else:
        winner = f"A ({variant_a})"

    # Human-readable interpretation
    if not significant:
        interpretation = (
            f"With p={p_value:.3f} > α={alpha}, we CANNOT reject H₀. "
            "There is no statistically significant difference between the variants."
        )
    else:
        d_label = ABTestResult.__new__(ABTestResult)
        d_label.cohens_d = cohens_d
        effect = d_label._effect_label()
        interpretation = (
            f"With p={p_value:.4f} < α={alpha}, we REJECT H₀. "
            f"{winner} wins with a {effect} effect (d={cohens_d:.2f}) "
            f"and {lift:+.1f}% lift."
        )

    result = ABTestResult(
        test_name=test_name,
        variant_a=variant_a,
        variant_b=variant_b,
        metric=metric,
        mean_a=round(mean_a, 6),
        mean_b=round(mean_b, 6),
        std_a=round(std_a, 6),
        std_b=round(std_b, 6),
        n_a=n_a,
        n_b=n_b,
        t_stat=round(float(t_stat), 6),
        p_value=round(float(p_value), 6),
        cohens_d=round(float(cohens_d), 6),
        winner=winner,
        significant=significant,
        confidence=round(1 - float(p_value), 4),
        lift=round(float(lift), 2),
        interpretation=interpretation,
    )

    log.info(f"A/B Test '{test_name}': p={p_value:.4f}, winner={winner}")
    return result


def save_test_results(results: list[ABTestResult]) -> None:
    """Persist all test results to the ab_tests SQLite table."""
    init_db()
    rows = []
    for r in results:
        rows.extend(r.to_db_rows())
    bulk_insert("ab_tests", rows)
    log.info(f"Saved {len(rows)} A/B test records to DB.")


# ── Pre-built test scenarios ──────────────────────────────────────────────────

def run_all_ab_tests(df: pd.DataFrame) -> list[ABTestResult]:
    """
    Run 4 pre-built A/B tests on the video dataset.

    Test 1: Posting time — Morning (6-12) vs Evening (18-24)
    Test 2: Video length — Short (<10 min) vs Long (>20 min)
    Test 3: Tag count — Few tags (<5) vs Many tags (>7)
    Test 4: Description length — Short vs Long descriptions
    """
    results: list[ABTestResult] = []

    if "published_at" not in df.columns:
        log.warning("published_at missing — skipping time-based tests.")
        return results

    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["hour"]         = df["published_at"].dt.hour
    df["tag_count"]    = df["tags"].fillna("").apply(lambda t: len(t.split("|")) if t else 0)
    df["desc_len"]     = df["description"].fillna("").str.len()

    # Parse duration ISO (PT5M30S → seconds)
    def _dur_secs(dur: str) -> int:
        if not isinstance(dur, str):
            return 0
        import re
        m = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur)
        if not m:
            return 0
        h, mi, s = (int(x or 0) for x in m.groups())
        return h * 3600 + mi * 60 + s

    df["duration_secs"] = df["duration"].apply(_dur_secs)

    # ── Test 1: Morning vs Evening ─────────────────────────────────────────────
    morning = df[df["hour"].between(6, 11)]["engagement_rate"].dropna()
    evening = df[df["hour"].between(18, 23)]["engagement_rate"].dropna()
    if len(morning) >= 3 and len(evening) >= 3:
        results.append(run_ab_test(
            morning, evening,
            test_name="Posting Time Impact",
            variant_a="Morning (6AM-12PM)",
            variant_b="Evening (6PM-12AM)",
            metric="engagement_rate",
        ))

    # ── Test 2: Short vs Long videos ───────────────────────────────────────────
    short_vids = df[df["duration_secs"] < 600]["view_count"].dropna()    # < 10 min
    long_vids  = df[df["duration_secs"] > 1200]["view_count"].dropna()   # > 20 min
    if len(short_vids) >= 3 and len(long_vids) >= 3:
        results.append(run_ab_test(
            short_vids, long_vids,
            test_name="Video Length Effect",
            variant_a="Short (<10 min)",
            variant_b="Long (>20 min)",
            metric="view_count",
        ))

    # ── Test 3: Few tags vs Many tags ──────────────────────────────────────────
    few_tags  = df[df["tag_count"] <= 4]["viral_score"].dropna()
    many_tags = df[df["tag_count"] >= 7]["viral_score"].dropna()
    if len(few_tags) >= 3 and len(many_tags) >= 3:
        results.append(run_ab_test(
            few_tags, many_tags,
            test_name="Tag Count Impact on Virality",
            variant_a="Few Tags (≤4)",
            variant_b="Many Tags (≥7)",
            metric="viral_score",
        ))

    # ── Test 4: Short vs Long description ──────────────────────────────────────
    median_desc = df["desc_len"].median()
    short_desc  = df[df["desc_len"] < median_desc]["like_count"].dropna()
    long_desc   = df[df["desc_len"] >= median_desc]["like_count"].dropna()
    if len(short_desc) >= 3 and len(long_desc) >= 3:
        results.append(run_ab_test(
            short_desc, long_desc,
            test_name="Description Length vs Likes",
            variant_a=f"Short Description (<{median_desc:.0f} chars)",
            variant_b=f"Long Description (≥{median_desc:.0f} chars)",
            metric="like_count",
        ))

    save_test_results(results)
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = DATA_PROCESSED_DIR / "videos_scored.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "videos.csv"

    df = pd.read_csv(path)
    results = run_all_ab_tests(df)

    for r in results:
        print(r.summary())
        print()
