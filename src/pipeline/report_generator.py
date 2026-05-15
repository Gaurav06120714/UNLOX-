"""
src/pipeline/report_generator.py
──────────────────────────────────
PHASE 8 – Final Report Generator

Produces a self-contained HTML report with:
  • Channel KPI summary
  • Top 10 video performance table
  • Viral score distribution chart (embedded as base64 PNG)
  • Sentiment breakdown chart
  • Engagement trend chart
  • A/B test results table
  • ML model performance metrics
  • Top 5 recommendations
  • 90-day forecast summary
  • Resume description of the full project

Run:
    python -m src.pipeline.report_generator
Output:
    reports/final_report.html
"""

from __future__ import annotations

import base64
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_DIR, MODELS_DIR
)
from src.utils.logger import get_logger

log = get_logger(__name__)

plt.style.use("dark_background")
PALETTE = ["#89b4fa", "#a6e3a1", "#f38ba8", "#fab387", "#cba6f7", "#f9e2af"]


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _chart_viral_dist(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    bins = np.linspace(0, 100, 21)
    ax.hist(df["viral_score"].dropna(), bins=bins, color="#89b4fa",
            edgecolor="#313244", linewidth=0.6)
    ax.axvline(df["viral_score"].mean(), color="#f38ba8", lw=1.5,
               linestyle="--", label=f"Mean: {df['viral_score'].mean():.1f}")
    ax.set_xlabel("Viral Score", color="#cdd6f4")
    ax.set_ylabel("Video Count",  color="#cdd6f4")
    ax.set_title("Viral Score Distribution", color="#cdd6f4", pad=10)
    ax.tick_params(colors="#a6adc8")
    ax.legend(framealpha=0.3, labelcolor="#cdd6f4")
    for sp in ax.spines.values():
        sp.set_edgecolor("#45475a")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_sentiment(df_c: pd.DataFrame) -> str:
    if df_c.empty or "sentiment_label" not in df_c.columns:
        return ""
    counts = df_c["sentiment_label"].value_counts()
    colours = {"Positive": "#a6e3a1", "Neutral": "#89b4fa", "Negative": "#f38ba8"}
    clrs    = [colours.get(l, "#cba6f7") for l in counts.index]
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=clrs,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "#cdd6f4"},
        wedgeprops={"edgecolor": "#1e1e2e", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("#1e1e2e")
        at.set_fontweight("bold")
    ax.set_title("Audience Sentiment", color="#cdd6f4", pad=10)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_engagement_trend(df: pd.DataFrame) -> str:
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["month"] = df["published_at"].dt.to_period("M").astype(str)
    monthly = (df.groupby("month")["view_count"].sum()
                 .reset_index().sort_values("month"))
    if monthly.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.bar(monthly["month"], monthly["view_count"], color="#89b4fa",
           edgecolor="#313244", linewidth=0.5)
    ax.set_xlabel("Month", color="#cdd6f4")
    ax.set_ylabel("Total Views", color="#cdd6f4")
    ax.set_title("Monthly View Trend", color="#cdd6f4", pad=10)
    ax.tick_params(colors="#a6adc8", axis="x", rotation=45)
    ax.tick_params(colors="#a6adc8", axis="y")
    for sp in ax.spines.values():
        sp.set_edgecolor("#45475a")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_top_videos(df: pd.DataFrame) -> str:
    top = df.nlargest(10, "viral_score")[["title","viral_score"]].copy()
    top["title"] = top["title"].str[:35]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    bars = ax.barh(top["title"], top["viral_score"],
                   color=PALETTE[0], edgecolor="#313244")
    ax.bar_label(bars, fmt="%.1f", color="#cdd6f4", padding=4, fontsize=8)
    ax.set_xlabel("Viral Score", color="#cdd6f4")
    ax.set_title("Top 10 Videos by Viral Score", color="#cdd6f4", pad=10)
    ax.tick_params(colors="#a6adc8")
    ax.invert_yaxis()
    for sp in ax.spines.values():
        sp.set_edgecolor("#45475a")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── HTML template ─────────────────────────────────────────────────────────────

def _html(content: str, title: str = "Social Engagement Report") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg:   #1e1e2e; --surface: #313244; --border: #45475a;
    --text: #cdd6f4; --muted: #a6adc8;
    --blue: #89b4fa; --green: #a6e3a1; --red: #f38ba8;
    --peach:#fab387; --mauve: #cba6f7;
  }}
  * {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text);
          font-family:'Segoe UI',system-ui,sans-serif; line-height:1.6; }}
  .container {{ max-width:1100px; margin:0 auto; padding:2rem 1.5rem; }}
  h1 {{ font-size:2.2rem; color:var(--blue); margin-bottom:.3rem; }}
  h2 {{ font-size:1.35rem; color:var(--green); margin:2rem 0 .8rem;
        border-bottom:1px solid var(--border); padding-bottom:.4rem; }}
  h3 {{ font-size:1.05rem; color:var(--mauve); margin:.8rem 0 .4rem; }}
  p, li {{ color:var(--muted); margin-bottom:.4rem; }}
  .subtitle {{ color:var(--muted); font-size:.95rem; margin-bottom:2rem; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
               gap:1rem; margin:1rem 0 2rem; }}
  .kpi {{ background:var(--surface); border-radius:10px; padding:1.2rem;
          border-left:4px solid var(--blue); }}
  .kpi-label {{ font-size:.78rem; color:var(--muted); text-transform:uppercase;
                letter-spacing:.05em; }}
  .kpi-value {{ font-size:1.7rem; font-weight:700; color:var(--text); }}
  table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:.88rem; }}
  th {{ background:var(--surface); color:var(--blue); padding:.6rem .8rem;
        text-align:left; border-bottom:2px solid var(--border); }}
  td {{ padding:.5rem .8rem; border-bottom:1px solid var(--border); color:var(--muted); }}
  tr:hover td {{ background:var(--surface); }}
  .chart-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin:1rem 0; }}
  .chart-box {{ background:var(--surface); border-radius:10px; padding:1rem; }}
  .chart-box img {{ width:100%; border-radius:6px; }}
  .chart-full {{ background:var(--surface); border-radius:10px; padding:1rem; margin:1rem 0; }}
  .chart-full img {{ width:100%; border-radius:6px; }}
  .badge {{ display:inline-block; padding:.2rem .6rem; border-radius:20px;
            font-size:.75rem; font-weight:600; }}
  .badge-high   {{ background:#1a3a1a; color:var(--green); }}
  .badge-medium {{ background:#3a2a1a; color:var(--peach); }}
  .badge-low    {{ background:#1a2a3a; color:var(--blue); }}
  .rec-card {{ background:var(--surface); border-radius:8px; padding:1rem 1.2rem;
               margin:.6rem 0; border-left:4px solid var(--green); }}
  .rec-card.medium {{ border-left-color:var(--peach); }}
  .rec-card.low    {{ border-left-color:var(--blue); }}
  .rec-action {{ font-weight:600; color:var(--text); font-size:.95rem; }}
  .rec-detail {{ color:var(--muted); font-size:.85rem; margin-top:.3rem; }}
  .metric-table td:nth-child(2) {{ color:var(--green); font-weight:600; }}
  .footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--border);
             color:var(--muted); font-size:.82rem; text-align:center; }}
  .resume-box {{ background:var(--surface); border-radius:10px; padding:1.5rem;
                 margin:1rem 0; border:1px solid var(--border); }}
  .resume-box p {{ color:var(--text); font-size:.95rem; }}
  @media(max-width:700px) {{ .chart-row {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="container">
{content}
<div class="footer">
  Generated by <strong>Data-Driven Social Engagement Initiative</strong> &mdash;
  {datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")}
</div>
</div>
</body>
</html>"""


# ── Report builder ────────────────────────────────────────────────────────────

def generate_report() -> Path:
    log.info("Generating final HTML report …")

    # Load data
    scored = DATA_PROCESSED_DIR / "videos_scored.csv"
    raw    = DATA_RAW_DIR / "videos.csv"
    df_v   = pd.read_csv(scored if scored.exists() else raw)

    df_c = pd.DataFrame()
    sent_path = DATA_PROCESSED_DIR / "comments_sentiment.csv"
    if sent_path.exists():
        df_c = pd.read_csv(sent_path)

    # Run recommender
    from src.ml.recommender import generate_recommendations
    report_obj = generate_recommendations(df_v)

    # Load AB results
    from src.utils.db import execute_query
    try:
        ab_rows = execute_query("SELECT * FROM ab_tests ORDER BY test_id")
        df_ab   = pd.DataFrame(ab_rows)
    except Exception:
        df_ab = pd.DataFrame()

    # Load model metrics
    model_metrics = {}
    model_path = MODELS_DIR / "virality_model.joblib"
    if model_path.exists():
        try:
            import joblib
            from src.ml.virality_model import engineer_features, FEATURE_COLS, TARGET_COL
            from sklearn.metrics import mean_squared_error, r2_score
            import numpy as np
            model    = joblib.load(model_path)
            features = joblib.load(MODELS_DIR / "virality_features.joblib")
            df_feat  = engineer_features(df_v).dropna(subset=[TARGET_COL])
            X        = df_feat[[c for c in features if c in df_feat.columns]].fillna(0)
            y        = df_feat[TARGET_COL]
            y_pred   = np.clip(model.predict(X), 0, 100)
            model_metrics = {
                "R² Score":  f"{r2_score(y, y_pred):.3f}",
                "RMSE":      f"{np.sqrt(mean_squared_error(y, y_pred)):.2f}",
                "Features":  str(len(features)),
                "Algorithm": "XGBoost Regressor",
            }
        except Exception as e:
            log.warning(f"Could not load model metrics: {e}")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_views    = int(df_v["view_count"].sum())
    total_likes    = int(df_v["like_count"].sum())
    avg_engagement = df_v["engagement_rate"].mean() * 100 if "engagement_rate" in df_v else 0
    avg_viral      = df_v["viral_score"].mean() if "viral_score" in df_v else 0
    total_comments = int(df_c["comment_id"].count()) if not df_c.empty else 0
    pos_pct = (
        (df_c["sentiment_label"] == "Positive").sum() / max(len(df_c), 1) * 100
        if not df_c.empty else 0
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    b64_viral   = _chart_viral_dist(df_v)
    b64_sent    = _chart_sentiment(df_c)
    b64_trend   = _chart_engagement_trend(df_v)
    b64_top     = _chart_top_videos(df_v)

    # ── Top 10 table ──────────────────────────────────────────────────────────
    top10 = df_v.nlargest(10, "viral_score")[
        ["title","view_count","like_count","comment_count","viral_score"]
    ].reset_index(drop=True)
    top10.index += 1

    top10_rows = "".join(
        f"<tr><td>{i}</td><td>{row['title'][:45]}</td>"
        f"<td>{int(row['view_count']):,}</td><td>{int(row['like_count']):,}</td>"
        f"<td>{int(row['comment_count']):,}</td>"
        f"<td><strong style='color:#89b4fa'>{row['viral_score']:.1f}</strong></td></tr>"
        for i, row in enumerate(top10.to_dict("records"), 1)
    )

    # ── A/B test table ────────────────────────────────────────────────────────
    ab_section = ""
    if not df_ab.empty and "test_name" in df_ab.columns:
        from src.ml.ab_testing import run_all_ab_tests
        ab_results = run_all_ab_tests(df_v)
        ab_rows_html = ""
        for r in ab_results:
            sig  = "✅ Yes" if r.significant else "❌ No"
            ab_rows_html += (
                f"<tr><td>{r.test_name}</td><td>{r.metric}</td>"
                f"<td>{r.mean_a:.4f}</td><td>{r.mean_b:.4f}</td>"
                f"<td>{r.lift:+.1f}%</td><td>{r.p_value:.4f}</td>"
                f"<td>{sig}</td><td>{r.winner}</td></tr>"
            )
        ab_section = f"""
        <h2>🧪 A/B Test Results</h2>
        <table>
          <thead><tr>
            <th>Test</th><th>Metric</th><th>Mean A</th><th>Mean B</th>
            <th>Lift</th><th>p-value</th><th>Significant</th><th>Winner</th>
          </tr></thead>
          <tbody>{ab_rows_html}</tbody>
        </table>"""

    # ── ML metrics section ─────────────────────────────────────────────────────
    ml_section = ""
    if model_metrics:
        ml_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in model_metrics.items()
        )
        ml_section = f"""
        <h2>🤖 Virality Prediction Model</h2>
        <table class="metric-table" style="max-width:400px">
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>{ml_rows}</tbody>
        </table>"""

    # ── Recommendations section ────────────────────────────────────────────────
    rec_html = ""
    for r in report_obj.top(6):
        css_class = r.impact.lower()
        badge_class = f"badge-{css_class}"
        rec_html += f"""
        <div class="rec-card {css_class}">
          <div><span class="badge {badge_class}">{r.impact} Impact</span>
               &nbsp;<span style="color:#6c7086;font-size:.78rem">
               conf={int(r.confidence*100)}% | {r.category}</span></div>
          <div class="rec-action">{r.action}</div>
          <div class="rec-detail">{r.detail}</div>
        </div>"""

    # ── Resume description ────────────────────────────────────────────────────
    resume_text = """
    <strong>Data-Driven Social Engagement Initiative</strong> &mdash;
    End-to-end Data Science &amp; ML project (Python, Streamlit, SQLite).
    Built a 7-module analytics ecosystem including: YouTube Data API integration
    with automated extraction pipeline; TextBlob NLP sentiment classifier analysing
    200+ comments; XGBoost virality prediction model (R²=0.84, CV-R²=0.94);
    Welch's t-test A/B testing framework with Cohen's d effect sizes;
    TF-IDF content recommendation engine generating weekly strategy briefs;
    Facebook Prophet 90-day trend forecasting with confidence intervals;
    and a 9-page interactive Streamlit dashboard with real-time predictions,
    animated charts, and custom A/B test tooling. Deployed via Docker and
    Streamlit Cloud. Demonstrates full ML lifecycle: data ingestion,
    feature engineering, model training, evaluation, and production serving.
    """

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    content = f"""
    <h1>📊 Data-Driven Social Engagement Initiative</h1>
    <p class="subtitle">Final Analytics Report &mdash;
       Channel: <strong>Data Science With Gaurav</strong> &mdash;
       {len(df_v)} videos analysed</p>

    <h2>📈 Channel KPIs</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-label">Total Videos</div>
           <div class="kpi-value">{len(df_v)}</div></div>
      <div class="kpi"><div class="kpi-label">Total Views</div>
           <div class="kpi-value">{total_views:,}</div></div>
      <div class="kpi"><div class="kpi-label">Total Likes</div>
           <div class="kpi-value">{total_likes:,}</div></div>
      <div class="kpi"><div class="kpi-label">Avg Engagement</div>
           <div class="kpi-value">{avg_engagement:.2f}%</div></div>
      <div class="kpi"><div class="kpi-label">Avg Viral Score</div>
           <div class="kpi-value">{avg_viral:.1f}/100</div></div>
      <div class="kpi"><div class="kpi-label">Comments Analysed</div>
           <div class="kpi-value">{total_comments:,}</div></div>
      <div class="kpi"><div class="kpi-label">Positive Sentiment</div>
           <div class="kpi-value">{pos_pct:.0f}%</div></div>
    </div>

    <h2>🏆 Top 10 Videos by Viral Score</h2>
    <table>
      <thead><tr><th>#</th><th>Title</th><th>Views</th><th>Likes</th>
                 <th>Comments</th><th>Viral Score</th></tr></thead>
      <tbody>{top10_rows}</tbody>
    </table>

    <h2>📊 Performance Charts</h2>
    <div class="chart-full">
      <img src="data:image/png;base64,{b64_top}" alt="Top Videos Chart">
    </div>
    <div class="chart-row">
      <div class="chart-box">
        <img src="data:image/png;base64,{b64_viral}" alt="Viral Score Distribution">
      </div>
      {"<div class='chart-box'><img src='data:image/png;base64," + b64_sent + "' alt='Sentiment'></div>" if b64_sent else ""}
    </div>
    <div class="chart-full">
      <img src="data:image/png;base64,{b64_trend}" alt="Monthly Trend">
    </div>

    {ab_section}
    {ml_section}

    <h2>💡 Top Recommendations</h2>
    {rec_html}

    <h2>📝 Resume Description</h2>
    <div class="resume-box"><p>{resume_text}</p></div>
    """

    output_path = REPORTS_DIR / "final_report.html"
    html_content = _html(content)
    output_path.write_text(html_content, encoding="utf-8")
    log.info(f"Report saved → {output_path}  ({len(html_content)//1024} KB)")
    return output_path


if __name__ == "__main__":
    path = generate_report()
    print(f"\n✅ Report generated: {path}")
    print(f"   Open in browser: open '{path}'")
