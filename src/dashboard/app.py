"""
Data-Driven Social Engagement Initiative
Streamlit Dashboard
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR

st.set_page_config(
    page_title="Social Engagement Dashboard",
    layout="wide",
)

st.markdown("""
<style>
    body { background-color: #f5f7fa; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e0e4ea;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] { color: #555e6b; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: #1a202c; font-size: 1.5rem; font-weight: 700; }
    .block-container { padding-top: 1.8rem; }
    h1 { color: #1a202c; font-size: 1.6rem; margin-bottom: 0.2rem; }
    h2 { color: #2d3748; font-size: 1.2rem; }
    h3 { color: #4a5568; font-size: 1.05rem; }
    section[data-testid="stSidebar"] { background: #1a202c; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# data loaders

@st.cache_data(ttl=300)
def load_videos():
    scored = DATA_PROCESSED_DIR / "videos_scored.csv"
    raw    = DATA_RAW_DIR        / "videos.csv"
    path   = scored if scored.exists() else raw
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["published_at"])
    for col in ["view_count", "like_count", "comment_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "engagement_rate" not in df.columns:
        df["engagement_rate"] = (df["like_count"] + df["comment_count"]) / df["view_count"].clip(1)
    if "viral_score" not in df.columns:
        df["viral_score"] = 0.0
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["day_of_week"]  = df["published_at"].dt.day_name()
    df["hour"]         = df["published_at"].dt.hour
    df["month"]        = df["published_at"].dt.to_period("M").dt.strftime("%Y-%m")
    return df


@st.cache_data(ttl=300)
def load_comments():
    sentiment = DATA_PROCESSED_DIR / "comments_sentiment.csv"
    raw       = DATA_RAW_DIR        / "comments.csv"
    path      = sentiment if sentiment.exists() else raw
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_ab_results():
    try:
        from src.utils.db import execute_query
        rows = execute_query("SELECT * FROM ab_tests ORDER BY test_id")
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_feature_importance():
    import joblib
    from config.settings import MODELS_DIR
    feat_path  = MODELS_DIR / "virality_features.joblib"
    model_path = MODELS_DIR / "virality_model.joblib"
    if not model_path.exists():
        return pd.DataFrame()
    try:
        model    = joblib.load(model_path)
        features = joblib.load(feat_path)
        xgb      = model.named_steps["xgb"]
        return pd.DataFrame({
            "feature":    features,
            "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# sidebar

with st.sidebar:
    st.markdown("## Engagement Hub")
    st.markdown("**Data-Driven Social Engagement Initiative**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Engagement Trends",
            "Viral Score",
            "Sentiment Analysis",
            "Best Posting Time",
            "A/B Testing",
            "Virality Predictor",
            "Recommendations",
            "Trend Forecast",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Major Project — Unlox Data Science Program")


# load data

df_v = load_videos()
df_c = load_comments()

if df_v.empty:
    st.warning("No video data found. Run `python run_pipeline.py` first.")
    st.stop()


# =============================================================================
#  PAGE 1 — OVERVIEW
# =============================================================================

if page == "Overview":
    st.title("Channel Overview")
    st.caption(f"Analysing **{len(df_v)}** videos from the channel")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Videos",       f"{len(df_v):,}")
    c2.metric("Total Views",         f"{df_v['view_count'].sum():,.0f}")
    c3.metric("Total Likes",         f"{df_v['like_count'].sum():,.0f}")
    c4.metric("Avg Engagement Rate", f"{df_v['engagement_rate'].mean()*100:.2f}%")
    c5.metric("Avg Viral Score",     f"{df_v['viral_score'].mean():.1f} / 100")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Top 10 Videos by Views")
        top10 = df_v.nlargest(10, "view_count")[
            ["title", "view_count", "like_count", "comment_count", "viral_score"]
        ].reset_index(drop=True)
        top10.index += 1
        st.dataframe(
            top10.style
                 .background_gradient(subset=["view_count"], cmap="Blues")
                 .format({"view_count": "{:,.0f}", "like_count": "{:,.0f}", "viral_score": "{:.1f}"}),
            use_container_width=True,
        )

    with col_right:
        st.subheader("Metric Distribution")
        metric_sel = st.selectbox(
            "Select metric",
            ["view_count", "like_count", "comment_count", "engagement_rate"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        fig = px.histogram(
            df_v, x=metric_sel, nbins=30,
            color_discrete_sequence=["#3b82f6"],
            template="simple_white",
        )
        fig.update_layout(margin=dict(t=10, b=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Views vs Likes  (bubble size = comment count)")
    fig2 = px.scatter(
        df_v, x="view_count", y="like_count",
        size="comment_count", color="viral_score",
        hover_name="title",
        color_continuous_scale="Blues",
        template="simple_white",
        labels={"view_count": "Views", "like_count": "Likes", "viral_score": "Viral Score"},
    )
    fig2.update_layout(height=400, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
#  PAGE 2 — ENGAGEMENT TRENDS
# =============================================================================

elif page == "Engagement Trends":
    st.title("Engagement Trends Over Time")
    st.markdown("---")

    monthly = (
        df_v.groupby("month")
        .agg(
            total_views    =("view_count",     "sum"),
            total_likes    =("like_count",     "sum"),
            total_comments =("comment_count",  "sum"),
            avg_engagement =("engagement_rate","mean"),
            video_count    =("video_id",       "count"),
        )
        .reset_index()
        .sort_values("month")
    )

    tab1, tab2, tab3 = st.tabs(["Views and Likes", "Engagement Rate", "Video Frequency"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["month"], y=monthly["total_views"],
                             name="Views", marker_color="#3b82f6"))
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["total_likes"],
                                 name="Likes", mode="lines+markers",
                                 line=dict(color="#ef4444", width=2)))
        fig.update_layout(template="simple_white", height=380,
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.area(monthly, x="month", y="avg_engagement",
                       color_discrete_sequence=["#10b981"],
                       template="simple_white",
                       labels={"avg_engagement": "Avg Engagement Rate"})
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.bar(monthly, x="month", y="video_count",
                      color_discrete_sequence=["#f59e0b"],
                      template="simple_white",
                      labels={"video_count": "Videos Published"})
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
#  PAGE 3 — VIRAL SCORE
# =============================================================================

elif page == "Viral Score":
    st.title("Virality Prediction Engine")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Top 20 Videos by Viral Score")
        top20 = df_v.nlargest(20, "viral_score")[
            ["title", "viral_score", "view_count", "like_count", "engagement_rate"]
        ].reset_index(drop=True)
        fig = px.bar(top20, x="viral_score", y="title", orientation="h",
                     color="viral_score", color_continuous_scale="RdYlGn",
                     template="simple_white",
                     labels={"viral_score": "Viral Score", "title": ""})
        fig.update_layout(height=520, yaxis=dict(autorange="reversed"), margin=dict(l=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Score Distribution")
        fig2 = px.violin(df_v, y="viral_score", box=True, points="all",
                         color_discrete_sequence=["#6366f1"],
                         template="simple_white")
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Score Buckets**")
        buckets = pd.cut(
            df_v["viral_score"],
            bins=[0, 25, 50, 75, 100],
            labels=["Low (<25)", "Medium (25-50)", "High (50-75)", "Viral (75+)"],
        ).value_counts().reset_index()
        buckets.columns = ["Bucket", "Count"]
        st.dataframe(buckets, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Inspect a Specific Video")
    selected = st.selectbox("Choose a video", df_v["title"].tolist())
    row = df_v[df_v["title"] == selected].iloc[0]

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Viral Score",     f"{row['viral_score']:.1f}")
    g2.metric("Views",           f"{row['view_count']:,.0f}")
    g3.metric("Likes",           f"{row['like_count']:,.0f}")
    g4.metric("Engagement Rate", f"{row['engagement_rate']*100:.2f}%")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=row["viral_score"],
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": "#10b981"},
            "steps": [
                {"range": [0,  25], "color": "#fee2e2"},
                {"range": [25, 50], "color": "#fef3c7"},
                {"range": [50, 75], "color": "#d1fae5"},
                {"range": [75,100], "color": "#a7f3d0"},
            ],
        },
        title={"text": "Viral Score"},
    ))
    gauge.update_layout(template="simple_white", height=260, margin=dict(t=20, b=0))
    st.plotly_chart(gauge, use_container_width=True)


# =============================================================================
#  PAGE 4 — SENTIMENT ANALYSIS
# =============================================================================

elif page == "Sentiment Analysis":
    st.title("Audience Sentiment Analyser")
    st.markdown("---")

    if df_c.empty:
        st.warning("No comment data found. Run the extraction pipeline first.")
        st.stop()

    if "sentiment_label" not in df_c.columns or df_c["sentiment_label"].isna().all():
        st.info("Sentiment not computed yet. Run `python -m src.nlp.sentiment`.")
        df_c["sentiment_label"] = "Unknown"
        df_c["sentiment_score"] = 0.0

    total = len(df_c)
    pos   = (df_c["sentiment_label"] == "Positive").sum()
    neg   = (df_c["sentiment_label"] == "Negative").sum()
    neu   = (df_c["sentiment_label"] == "Neutral" ).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Comments", f"{total:,}")
    k2.metric("Positive",       f"{pos:,} ({pos/total*100:.1f}%)")
    k3.metric("Negative",       f"{neg:,} ({neg/total*100:.1f}%)")
    k4.metric("Neutral",        f"{neu:,} ({neu/total*100:.1f}%)")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Breakdown")
        pie = px.pie(df_c, names="sentiment_label",
                     color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#3b82f6"},
                     template="simple_white", hole=0.4)
        pie.update_layout(height=340)
        st.plotly_chart(pie, use_container_width=True)

    with right:
        st.subheader("Score Distribution")
        hist = px.histogram(df_c, x="sentiment_score", nbins=40, color="sentiment_label",
                            color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#3b82f6"},
                            template="simple_white")
        hist.update_layout(height=340, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(hist, use_container_width=True)

    if "video_id" in df_c.columns and "video_id" in df_v.columns:
        st.subheader("Per-Video Sentiment Heatmap")
        vid_sent = (
            df_c.groupby(["video_id", "sentiment_label"])
            .size().unstack(fill_value=0).reset_index()
        )
        vid_sent = vid_sent.merge(df_v[["video_id", "title"]], on="video_id", how="left")
        vid_sent["title"] = vid_sent["title"].str[:40]
        # deduplicate titles to avoid plotly error
        vid_sent = vid_sent.drop_duplicates(subset=["title"])
        heat_data = vid_sent.set_index("title")[
            [c for c in ["Positive", "Neutral", "Negative"] if c in vid_sent.columns]
        ].head(20)
        if not heat_data.empty:
            fig_heat = px.imshow(heat_data.T,
                                 color_continuous_scale="RdYlGn",
                                 template="simple_white",
                                 aspect="auto")
            fig_heat.update_layout(height=280)
            st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Sample Comments")
    label_filter = st.select_slider(
        "Filter by sentiment", ["Negative", "Neutral", "Positive"], value="Positive"
    )
    sample = df_c[df_c["sentiment_label"] == label_filter][["author", "text", "sentiment_score"]].head(8)
    st.dataframe(sample, use_container_width=True, hide_index=True)


# =============================================================================
#  PAGE 5 — BEST POSTING TIME
# =============================================================================

elif page == "Best Posting Time":
    st.title("Best Posting Time Analytics")
    st.markdown("---")

    days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    day_perf = (
        df_v.groupby("day_of_week")["view_count"]
        .mean()
        .reindex(days_order)
        .reset_index()
    )
    day_perf.columns = ["Day", "Avg Views"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Views by Day of Week")
        fig = px.bar(day_perf, x="Day", y="Avg Views",
                     color="Avg Views", color_continuous_scale="Blues",
                     template="simple_white")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Engagement by Day")
        eng_day = (
            df_v.groupby("day_of_week")["engagement_rate"]
            .mean()
            .reindex(days_order)
            .reset_index()
        )
        fig2 = px.line(eng_day, x="day_of_week", y="engagement_rate",
                       markers=True, color_discrete_sequence=["#10b981"],
                       template="simple_white")
        fig2.update_layout(height=340)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Performance Heatmap  (Day vs Hour of Day)")
    heat = (
        df_v.groupby(["day_of_week", "hour"])["view_count"]
        .mean()
        .unstack(fill_value=0)
    )
    heat = heat.reindex(days_order).fillna(0)
    fig3 = px.imshow(heat,
                     labels=dict(x="Hour of Day", y="Day of Week", color="Avg Views"),
                     color_continuous_scale="YlOrRd",
                     template="simple_white",
                     aspect="auto")
    fig3.update_layout(height=360, margin=dict(t=10))
    st.plotly_chart(fig3, use_container_width=True)

    best_day  = day_perf.loc[day_perf["Avg Views"].idxmax(), "Day"]
    best_hour = int(df_v.groupby("hour")["view_count"].mean().idxmax())
    st.success(f"Best time to post: **{best_day}** at **{best_hour:02d}:00 UTC**")


# =============================================================================
#  PAGE 6 — A/B TESTING
# =============================================================================

elif page == "A/B Testing":
    st.title("A/B Testing Framework")
    st.caption("Welch's t-test (alpha = 0.05) across content variables")
    st.markdown("---")

    col_run, col_info = st.columns([1, 3])
    with col_run:
        if st.button("Run A/B Tests", type="primary"):
            with st.spinner("Running statistical tests..."):
                from src.ml.ab_testing import run_all_ab_tests
                results = run_all_ab_tests(load_videos())
                st.cache_data.clear()
                st.success(f"Ran {len(results)} tests.")
                st.rerun()
    with col_info:
        st.info("Tests compare Posting Time, Video Length, Tag Count, and Description Length.")

    df_ab = load_ab_results()
    if df_ab.empty:
        st.warning("No A/B test results yet. Click Run A/B Tests above.")
        st.stop()

    tests = df_ab["test_name"].unique() if "test_name" in df_ab.columns else []
    for test in tests:
        subset = df_ab[df_ab["test_name"] == test]
        if len(subset) < 2:
            continue
        with st.expander(f"{test}", expanded=True):
            variants   = subset["variant"].tolist()
            means      = subset["metric_value"].tolist()
            metric_lbl = subset["metric_name"].iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Variant A ({variants[0]})", f"{means[0]:.4f}")
            if len(means) > 1:
                c2.metric(f"Variant B ({variants[1]})", f"{means[1]:.4f}",
                          delta=f"{means[1]-means[0]:+.4f}")
            c3.metric("Metric", metric_lbl)

            fig = px.bar(subset, x="variant", y="metric_value",
                         color="variant",
                         color_discrete_sequence=["#3b82f6", "#10b981"],
                         template="simple_white",
                         labels={"metric_value": metric_lbl, "variant": ""},
                         text_auto=".4f")
            fig.update_layout(height=260, showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("All Tests Summary")

    with st.spinner("Computing statistics..."):
        from src.ml.ab_testing import run_all_ab_tests
        full_results = run_all_ab_tests(load_videos())

    summary_rows = [{
        "Test":        r.test_name,
        "Metric":      r.metric,
        "Mean A":      f"{r.group_a_mean:.4f}",
        "Mean B":      f"{r.group_b_mean:.4f}",
        "Lift":        f"{((r.group_a_mean - r.group_b_mean) / max(abs(r.group_b_mean), 1e-9)) * 100:+.1f}%",
        "p-value":     f"{r.p_value:.4f}",
        "Cohen's d":   f"{r.effect_size:.3f}",
        "Significant": "Yes" if r.significant else "No",
        "Winner":      r.winner,
    } for r in full_results]

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Run a Custom A/B Test")
    cc1, cc2 = st.columns(2)
    with cc1:
        raw_a  = st.text_area("Variant A values (comma-separated)", "120,145,130,160,140,155,125,170")
        name_a = st.text_input("Variant A label", "Control")
    with cc2:
        raw_b  = st.text_area("Variant B values (comma-separated)", "180,210,195,220,205,190,215,230")
        name_b = st.text_input("Variant B label", "Treatment")

    metric_name = st.text_input("Metric name", "engagement_rate")
    alpha       = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)

    if st.button("Run Custom Test"):
        try:
            from src.ml.ab_testing import run_ab_test
            a_vals = [float(x.strip()) for x in raw_a.split(",") if x.strip()]
            b_vals = [float(x.strip()) for x in raw_b.split(",") if x.strip()]
            result = run_ab_test(a_vals, b_vals, test_name="Custom Test",
                                 variant_a=name_a, variant_b=name_b,
                                 metric=metric_name, alpha=alpha)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("p-value",   f"{result.p_value:.4f}")
            r2.metric("Cohen's d", f"{result.cohens_d:.3f}")
            r3.metric("Lift",      f"{result.lift:+.1f}%")
            r4.metric("Winner",    result.winner)
            if result.significant:
                st.success(f"Statistically significant! {result.interpretation}")
            else:
                st.warning(f"Not significant. {result.interpretation}")
        except ValueError as e:
            st.error(f"Invalid input: {e}")


# =============================================================================
#  PAGE 7 — VIRALITY PREDICTOR
# =============================================================================

elif page == "Virality Predictor":
    st.title("Virality Prediction Engine")
    st.caption("XGBoost model trained on channel data")
    st.markdown("---")

    from config.settings import MODELS_DIR
    model_exists = (MODELS_DIR / "virality_model.joblib").exists()

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        if st.button("Train Model", type="primary"):
            with st.spinner("Training..."):
                from src.ml.virality_model import train_model
                res = train_model(load_videos())
                st.cache_data.clear()
                m = res["metrics"]
                st.success(f"Done! R2={m['r2']:.3f}, RMSE={m['rmse']:.2f}")
                st.rerun()
    with col_status:
        if model_exists:
            st.success("Model loaded from models/virality_model.joblib")
        else:
            st.warning("No model found. Click Train Model above.")

    if model_exists:
        df_imp = load_feature_importance()
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Feature Importance")
            if not df_imp.empty:
                fig_imp = px.bar(df_imp.head(12), x="importance", y="feature",
                                 orientation="h",
                                 color="importance", color_continuous_scale="Blues",
                                 template="simple_white",
                                 labels={"importance": "Importance Score", "feature": ""})
                fig_imp.update_layout(height=420, yaxis=dict(autorange="reversed"),
                                      margin=dict(l=0, t=10))
                st.plotly_chart(fig_imp, use_container_width=True)

        with col_right:
            st.subheader("Predicted vs Actual")
            try:
                from src.ml.virality_model import predict_viral_score
                df_pred = predict_viral_score(load_videos().copy())
                if "viral_score" in df_pred.columns:
                    fig_sc = px.scatter(df_pred, x="viral_score", y="predicted_viral_score",
                                        hover_name="title",
                                        color_discrete_sequence=["#6366f1"],
                                        template="simple_white",
                                        labels={"viral_score": "Actual Score",
                                                "predicted_viral_score": "Predicted Score"})
                    fig_sc.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                                     line=dict(color="#ef4444", dash="dash"))
                    fig_sc.update_layout(height=380, margin=dict(t=10))
                    st.plotly_chart(fig_sc, use_container_width=True)
            except Exception as e:
                st.info(f"Chart unavailable: {e}")

        st.markdown("---")
        st.subheader("Predict Score for a Hypothetical Video")

        f1, f2, f3 = st.columns(3)
        with f1:
            sim_views    = st.number_input("Expected Views",    0, 10_000_000, 50_000, step=1000)
            sim_likes    = st.number_input("Expected Likes",    0, 500_000,    3_000,  step=100)
            sim_comments = st.number_input("Expected Comments", 0, 50_000,     150,    step=10)
        with f2:
            sim_duration = st.slider("Duration (minutes)", 1, 60, 12)
            sim_tags     = st.slider("Number of Tags",     0, 15, 7)
            sim_hour     = st.slider("Publishing Hour (UTC)", 0, 23, 10)
        with f3:
            sim_title    = st.text_input("Video Title", "How I Learned Python in 30 Days")
            sim_desc_len = st.slider("Description Length (chars)", 0, 1000, 300)
            sim_day      = st.selectbox("Publishing Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

        day_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}

        if st.button("Predict Viral Score", type="primary"):
            try:
                from src.ml.virality_model import predict_single
                score = predict_single(
                    view_count    = int(sim_views),
                    like_count    = int(sim_likes),
                    comment_count = int(sim_comments),
                    duration_str  = f"PT{sim_duration}M0S",
                    tag_count     = sim_tags,
                    title         = sim_title,
                    description   = "x" * sim_desc_len,
                    hour          = sim_hour,
                    day_of_week   = day_map[sim_day],
                )
                colour = "#10b981" if score >= 60 else "#f59e0b" if score >= 30 else "#ef4444"
                st.markdown(
                    f"<h2 style='text-align:center; color:{colour};'>"
                    f"Predicted Viral Score: {score:.1f} / 100</h2>",
                    unsafe_allow_html=True,
                )
                label = "High Viral Potential!" if score >= 60 \
                        else "Moderate — optimise further" if score >= 30 \
                        else "Low — rethink your strategy"
                st.info(label)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": colour},
                        "steps": [
                            {"range": [0,  30], "color": "#fee2e2"},
                            {"range": [30, 60], "color": "#fef3c7"},
                            {"range": [60,100], "color": "#d1fae5"},
                        ],
                        "threshold": {"line": {"color": "#ef4444", "width": 4},
                                      "thickness": 0.75, "value": 75},
                    },
                ))
                gauge.update_layout(template="simple_white", height=260, margin=dict(t=20, b=0))
                st.plotly_chart(gauge, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}. Make sure the model is trained.")


# =============================================================================
#  PAGE 8 — RECOMMENDATIONS
# =============================================================================

elif page == "Recommendations":
    st.title("Engagement Optimization Recommender")
    st.caption("Weekly strategy based on your top-performing videos")
    st.markdown("---")

    from src.ml.recommender import generate_recommendations

    with st.spinner("Analysing content patterns..."):
        report = generate_recommendations(df_v)

    st.subheader("Weekly Strategy Brief")
    st.code(report.weekly_brief, language=None)
    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Recommendations", len(report.recommendations))
    k2.metric("Videos Analysed",       report.total_videos)
    k3.metric("Top Performers",        report.top_videos_n)
    high_conf = sum(1 for r in report.recommendations if r.confidence >= 0.70)
    k4.metric("High-Confidence Tips",  high_conf)

    st.markdown("---")
    impact_filter = st.multiselect("Filter by Impact Level",
                                   ["High", "Medium", "Low"],
                                   default=["High", "Medium"])
    cat_filter = st.multiselect("Filter by Category",
                                sorted({r.category for r in report.recommendations}),
                                default=sorted({r.category for r in report.recommendations}))

    filtered = [r for r in report.recommendations
                if r.impact in impact_filter and r.category in cat_filter]

    IMPACT_COLOURS = {"High": "#10b981", "Medium": "#f59e0b", "Low": "#3b82f6"}
    CAT_ICONS = {
        "Posting Time":        "Time",
        "Video Length":        "Length",
        "Tag Strategy":        "Tags",
        "Topic & Keywords":    "Topics",
        "Engagement Strategy": "Engagement",
    }

    for i, rec in enumerate(filtered, 1):
        colour = IMPACT_COLOURS.get(rec.impact, "#4a5568")
        conf_pct = int(rec.confidence * 100)
        st.markdown(
            f"""
            <div style="border-left:4px solid {colour}; padding:12px 16px;
                        margin-bottom:12px; background:#ffffff;
                        border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.06);">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:{colour}; font-weight:700; font-size:0.82rem;">
                        {rec.category} | {rec.impact} Impact
                    </span>
                    <span style="color:#6b7280; font-size:0.78rem;">Confidence: {conf_pct}%</span>
                </div>
                <p style="color:#1a202c; font-size:1rem; font-weight:600; margin:8px 0 4px;">
                    {i}. {rec.action}
                </p>
                <p style="color:#4a5568; font-size:0.86rem; margin:0;">{rec.detail}</p>
                <p style="color:#9ca3af; font-size:0.76rem; margin-top:6px;">Evidence: {rec.evidence}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not filtered:
        st.info("No recommendations match the selected filters.")

    st.markdown("---")
    st.subheader("Recommendation Confidence Overview")
    conf_df = pd.DataFrame([{
        "Action":     r.action[:50],
        "Category":   r.category,
        "Confidence": r.confidence * 100,
        "Impact":     r.impact,
    } for r in report.recommendations])

    fig = px.bar(conf_df.sort_values("Confidence", ascending=True),
                 x="Confidence", y="Action", orientation="h",
                 color="Impact",
                 color_discrete_map={"High": "#10b981", "Medium": "#f59e0b", "Low": "#3b82f6"},
                 template="simple_white",
                 labels={"Confidence": "Confidence %", "Action": ""})
    fig.update_layout(height=max(300, len(conf_df) * 45), margin=dict(l=0, t=10))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
#  PAGE 9 — TREND FORECAST
# =============================================================================

elif page == "Trend Forecast":
    st.title("Trend Forecasting Module")
    st.caption("90-day performance forecast and keyword trend analysis")
    st.markdown("---")

    from src.ml.forecasting import run_forecast as do_forecast, analyse_keyword_trends, compute_engagement_velocity

    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        metric_sel = st.selectbox("Forecast metric",
                                  ["view_count", "like_count", "comment_count", "engagement_rate"],
                                  format_func=lambda x: x.replace("_", " ").title())
    with ctrl2:
        horizon = st.select_slider("Forecast horizon (days)", [30, 60, 90], value=90)

    with st.spinner(f"Forecasting {metric_sel}..."):
        fc_result = do_forecast(df_v, metric=metric_sel, periods=horizon)

    ts       = fc_result["ts"]
    forecast = fc_result["forecast"]
    future   = fc_result["future_only"]
    engine   = fc_result["engine"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Forecast Engine",      engine.upper())
    k2.metric("Historical Points",    len(ts))
    k3.metric(f"Avg (next {horizon}d)", f"{future['yhat'].mean():,.0f}")
    k4.metric("Peak Predicted",       f"{future['yhat'].max():,.0f}")

    st.subheader(f"{metric_sel.replace('_',' ').title()} — {horizon}-Day Forecast")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["ds"], y=ts["y"], name="Actual",
                             mode="lines+markers",
                             line=dict(color="#3b82f6", width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast",
                             mode="lines", line=dict(color="#10b981", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(16,185,129,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Confidence Interval",
    ))
    # add divider line between history and forecast
    if not ts.empty:
        last_real = ts["ds"].max()
        fig.add_shape(type="line",
                      x0=last_real, x1=last_real, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="#ef4444", dash="dot", width=1.5))
        fig.add_annotation(x=last_real, y=1, yref="paper",
                           text="Forecast starts", showarrow=False,
                           xanchor="left", font=dict(color="#ef4444", size=11))
    fig.update_layout(template="simple_white", height=420,
                      legend=dict(orientation="h", y=1.05),
                      margin=dict(t=10, b=10),
                      xaxis_title="Date",
                      yaxis_title=metric_sel.replace("_", " ").title())
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Engine: {engine.upper()} | Shaded area = 80% confidence interval")

    st.markdown("---")
    st.subheader("Keyword and Topic Trend Analysis")

    with st.spinner("Analysing keywords..."):
        trends = analyse_keyword_trends(df_v, top_n=25)

    if not trends.empty:
        tab_rising, tab_falling, tab_all = st.tabs(["Rising", "Falling", "All Keywords"])

        with tab_rising:
            rising_df = trends[trends["trend"].str.contains("Rising")].copy()
            if not rising_df.empty:
                fig_r = px.bar(rising_df.head(10), x="change_pct", y="keyword",
                               orientation="h", color="change_pct",
                               color_continuous_scale="Greens",
                               template="simple_white",
                               labels={"change_pct": "Growth %", "keyword": ""},
                               text="change_pct")
                fig_r.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
                fig_r.update_layout(height=340, margin=dict(l=0, t=10),
                                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_r, use_container_width=True)
                st.success("Use these keywords in upcoming video titles and descriptions.")
            else:
                st.info("No significantly rising keywords detected yet.")

        with tab_falling:
            falling_df = trends[trends["trend"].str.contains("Falling")].copy()
            if not falling_df.empty:
                fig_f = px.bar(falling_df.head(10), x="change_pct", y="keyword",
                               orientation="h", color="change_pct",
                               color_continuous_scale="Reds_r",
                               template="simple_white",
                               labels={"change_pct": "Change %", "keyword": ""})
                fig_f.update_layout(height=340, margin=dict(l=0, t=10),
                                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_f, use_container_width=True)
                st.warning("These keywords are losing traction.")
            else:
                st.info("No significantly falling keywords.")

        with tab_all:
            st.dataframe(trends.style.background_gradient(subset=["change_pct"], cmap="RdYlGn"),
                         use_container_width=True, hide_index=True)
    else:
        st.info("Not enough video history for keyword trend analysis.")

    st.markdown("---")
    st.subheader("Month-over-Month Growth Velocity")

    with st.spinner("Computing growth velocity..."):
        vel = compute_engagement_velocity(df_v)

    if len(vel) >= 2:
        tab_v, tab_l, tab_c = st.tabs(["Views Growth", "Likes Growth", "Comments Growth"])
        for tab, col, label in [(tab_v,"views","Views"),(tab_l,"likes","Likes"),(tab_c,"comments","Comments")]:
            with tab:
                fig_v = go.Figure()
                fig_v.add_trace(go.Bar(x=vel["month"], y=vel[col], name=label, marker_color="#3b82f6"))
                fig_v.add_trace(go.Scatter(x=vel["month"], y=vel[f"{col}_growth_%"],
                                           name="MoM Growth %", mode="lines+markers",
                                           yaxis="y2", line=dict(color="#10b981", width=2),
                                           marker=dict(size=6)))
                fig_v.update_layout(template="simple_white", height=360,
                                    yaxis=dict(title=label),
                                    yaxis2=dict(title="Growth %", overlaying="y", side="right", showgrid=False),
                                    legend=dict(orientation="h", y=1.1), margin=dict(t=10))
                st.plotly_chart(fig_v, use_container_width=True)
    else:
        st.info("Need at least 2 months of data for velocity charts.")
